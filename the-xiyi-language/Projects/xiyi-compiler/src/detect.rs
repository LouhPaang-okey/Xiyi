// detect.rs
//
// 底层查询/判定库。所有"给定 IR 节点、回答一个问题"的纯函数都放这里
// ——不建表、不跑不动点、不保存状态，输入是 MIR 节点或整个 body，输出
// 是 bool / Option / 小集合。
//
// 分工边界：
//   - detect.rs   ：查询/判定（本文件），不依赖 analyze/simplify/control
//   - analyze.rs  ：分析结果容器（CfgInfo / SccpResult / GvnResult /
//                    LoopInfo / AliasInfo / 函数纯度表），有状态、要建表
//   - simplify.rs / control.rs：改写 pass，消费前两者
//
// 依赖方向只有一个：analyze/simplify/control -> detect。detect 不能反过来
// use analyze，否则立刻成环——所以依赖 LoopInfo 的 find_loop_invariants
// 放在 analyze.rs，不放这里。
//
// 之所以把散落在各 pass 里的判定收拢进来：同一个判定（比如"这个语句是
// 不是 Phi"）在 control.rs 里有一份、simplify.rs 里有一份、将来别的 pass
// 还会再写一份——每份都可能跟其它份悄悄不一致（这类"两处手抄同一份
// 逻辑"的漏改在这个项目里反复出现过）。收进 detect 之后只有一处，改一处
// 所有调用方跟上。
//
// 项目约定：不用 matches! 这类纯逻辑糖宏，match 尽量显式穷尽、不用通配
// 兜底——新增枚举变体时编译器会在这里报错，逼着当场表态。
use crate::intrinsic::{get_intrinsic, IntrinsicFn};
use crate::mir::*;
use std::collections::{HashMap, HashSet};

// ==================== 副作用模型 ====================
//
// 三级格：Pure ⊑ ReadOnly ⊑ Impure。格的方向跟优化自由度相反——越纯
// 的东西优化越自由。
//
// 这三级的含义在整个项目里只有这一份，所有用到纯度的地方（purity_of_fn、
// compute_purity_table、AliasInfo、GVN、死代码消除、LICM）都必须按它理解：
//
// Pure：没有任何可观察副作用，也不读取参数值以外的任何状态，运算本身也
//       不会陷入——结果只由参数的值决定。相同参数的两次调用一定得到同一个
//       值（GVN 可以合并），结果没被使用可以整条删（erasable）。
//
// ReadOnly：*不写*任何外部可见状态（不通过 Deref 写、不写全局、没有
//       io/rng/ai/ffi 效应），但可能：
//         - 读到调用者的内存（通过 `&T` 参数）——相同参数的两次调用不保证
//           同一个值（中间内存可能被改过），不能合并；
//         - 陷入（运算本身隐式 panic：除数不是非零常量的除法/取模、索引
//           越界，以及将来若出现的可能失败的转换，见 binop_may_trap /
//           cast_may_trap / place_may_trap）——panic 终止程序是可观察行为，
//           结果没被使用也不能整条删掉，也不能被投机执行。
//       这一级的共同点：不能删、不能合并（读内存的情形）、不会改动内存。
//
// Impure：其余一切——可能写入外部状态（含拿到 `&mut` 参数：它就是写入
//       调用者内存的通道），或者带有 io/rng/ai/ffi 效应，或者*显式声明*了
//       panic 效应（EffectSet.has_panic、Panic 内建函数）。
//       "显式 panic"与"运算可能陷入"的区别：前者是程序员写的、一定发生的
//       终止；后者是某些输入下才发生的隐式陷入，只要求别被删除/投机。
//
// 关键约束：ReadOnly 里绝不能混进"可能写"的东西（"可能陷入"可以，它不改
// 内存）。AliasInfo 判断"这个调用会不会写入循环里的内存"时只有 Impure 算
// 写——把可能写的函数判成 ReadOnly，就等于告诉别名分析"它不写"，外提
// ReadOnly 调用时会静默越过一个其实在写同一块内存的调用。
//
// 关键说明：这里只定义"纯度"这个概念和"怎么从调用点/签名读出它"，不
// 负责回答"某个用户函数到底纯不纯"——那需要跨函数的不动点，见
// analyze.rs::compute_purity_table。原因是 EffectSet 并不是传递闭包：
// hir_builder.rs 里一个 Call 的 effects 只合并了它*实参*的 effects，被
// 调函数自己的 effects 并没有合并进来，而且只有函数名字面恰好是
// "print" 才会置 has_io（别的 io/rng/ffi 都没有来源）。只看 EffectSet
// 会把"调用了 print 的函数的调用者"误判成 Pure。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purity {
    Pure,
    ReadOnly,
    Impure,
}

impl Purity {
    /// 结果没被使用时能不能整条删掉。只有 Pure 可以：ReadOnly 虽然严格
    /// 不写，但可能读到会变的内存、可能陷入，也没有证明它终止，保守不删。
    pub fn erasable(&self) -> bool {
        match self {
            Purity::Pure => true,
            Purity::ReadOnly | Purity::Impure => false,
        }
    }

    /// 格上的 join：取更不纯的那个。
    pub fn join(self, other: Purity) -> Purity {
        match (self, other) {
            (Purity::Impure, _) | (_, Purity::Impure) => Purity::Impure,
            (Purity::ReadOnly, _) | (_, Purity::ReadOnly) => Purity::ReadOnly,
            (Purity::Pure, Purity::Pure) => Purity::Pure,
        }
    }
}

/// 从函数级的 EffectSet + 参数列表映射到 Purity。只用来给纯度表提供
/// 初值（见 analyze.rs::compute_purity_table），不能单独当作"这个函数纯"
/// 的依据。
///
/// 有 &mut 参数就判 Impure：`&mut` 就是函数写入调用者内存的通道。ReadOnly
/// 的定义是"严格只读"（见上面格的说明），带 `&mut` 的函数哪怕实际没写，
/// 也不能证明它不写——这里不分析函数体，保守判 Impure，只是少优化，不是错。
/// 这个函数只给纯度表提供初值，函数体里的 Deref 写入、对 Impure 函数的调用
/// 之后还会在 compute_purity_table 里 join 上来。
pub fn purity_of_fn(effects: &EffectSet, params: &[(String, Type)]) -> Purity {
    if !effects.is_pure() {
        return Purity::Impure;
    }
    let mut has_mut_ref = false;
    for (_, ty) in params {
        if let Type::Ref { mutable, .. } = ty {
            if *mutable {
                has_mut_ref = true;
            }
        }
    }
    if has_mut_ref {
        Purity::Impure
    } else {
        Purity::Pure
    }
}

/// 调用点的纯度。
///   - intrinsic 直接查 Intrinsic::is_pure（intrinsic.rs 里维护）。表里的
///     is_pure 必须跟这里的 Purity::Pure 同义——"结果没用可以整条删、相同
///     参数的两次调用可以合并成一次"。会 panic 的（Reshape 形状对不上）、
///     带随机性的（Dropout）都不满足，在表里标 false，这里不需要也不应该
///     再按名字特判它们。
///   - 普通调用查 purity 表
///   - 表里查不到：保守 Impure（未定义 / 跨模块 / mangled 名不匹配）
///
/// TensorCond / TensorWhileLoop 在 Intrinsic::is_pure 里标的是 true，但它们
/// 是编译期展开的控制流构造，不是真正的调用——如果 MIR 里出现了这两个
/// 作为普通 Call，保守返回 Impure。
pub fn call_purity(
    func: &str,
    is_intrinsic: bool,
    intrinsic_name: Option<IntrinsicFn>,
    purity: &HashMap<String, Purity>,
) -> Purity {
    if is_intrinsic {
        match intrinsic_name {
            Some(IntrinsicFn::TensorCond) | Some(IntrinsicFn::TensorWhileLoop) => Purity::Impure,
            Some(name) => match get_intrinsic(name) {
                Some(i) => {
                    if i.is_pure {
                        Purity::Pure
                    } else {
                        Purity::Impure
                    }
                }
                None => Purity::Impure,
            },
            None => Purity::Impure,
        }
    } else {
        match purity.get(func) {
            Some(p) => *p,
            None => Purity::Impure,
        }
    }
}

/// 右值有没有可观察副作用。有副作用的语句即使结果没被用也不能删。
/// "不能删"有两个互相独立的来源，缺一不可：
///   - 调用的函数本身不 erasable（ReadOnly / Impure）；
///   - 运算本身可能陷入（rvalue_may_trap：除法、Index 越界等），包括
///     嵌在调用实参里的 place（`f(a[i])` 里 a[i] 越界是这个调用点自己的 panic）。
/// MethodCall 要靠 receiver 类型的 trait 解析才知道调的是哪个方法，查不了
/// 纯度表，一律保守。
pub fn rvalue_has_side_effect(
    rv: &MirRvalue,
    purity: &HashMap<String, Purity>,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> bool {
    match rv {
        MirRvalue::Call { func, is_intrinsic, intrinsic_name, .. } => {
            !call_purity(func, *is_intrinsic, *intrinsic_name, purity).erasable()
                || rvalue_may_trap(rv, body, structs, enums)
        }
        MirRvalue::MethodCall { .. } => true,
        MirRvalue::Use(_)
        | MirRvalue::BinaryOp(_, _, _)
        | MirRvalue::UnaryOp(_, _)
        | MirRvalue::Cast(_, _)
        | MirRvalue::StructInit { .. }
        | MirRvalue::EnumVariantConstruction { .. }
        | MirRvalue::Ref { .. }
        | MirRvalue::ArrayLiteral(_)
        | MirRvalue::Discriminant { .. }
        | MirRvalue::Phi { .. } => rvalue_may_trap(rv, body, structs, enums),
    }
}

// ==================== MIR 结构判定 ====================

/// 语句是不是 Phi。
pub fn stmt_is_phi(stmt: &MirStmt) -> bool {
    match stmt {
        MirStmt::Assign { value, .. } => match value {
            MirRvalue::Phi { .. } => true,
            MirRvalue::Use(_)
            | MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Cast(_, _)
            | MirRvalue::Call { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. } => false,
        },
        MirStmt::ExprStmt(_)
        | MirStmt::Drop { .. }
        | MirStmt::SetMetadata { .. }
        | MirStmt::EffectCheck { .. } => false,
    }
}

/// 块里有没有 Phi。
pub fn block_contains_phi(block: &MirBlock) -> bool {
    block.stmts.iter().any(stmt_is_phi)
}

/// 块里所有 Phi 的"定义"和"入边"。Liveness 需要它做边敏感的活跃分析：
///   - 第一项：这个块里所有 Phi 定义的 SSA 值（dest）；
///   - 第二项：每条 incoming 记成 (前驱块 id, 来自该前驱的 SSA 值)，非 SSA
///     操作数（常量、字段读取等）不产生活跃值，直接略过。
///
/// Phi 的语义是"从哪条边进来就取哪条边对应的值"：incoming 的值只在对应的
/// 那条边上活跃，不是在 Phi 所在块里"被读一次"。把它当成 Phi 所在块的普通
/// use，会让每个 incoming 值在所有前驱路径上都显得活跃，比实际保守。
pub fn block_phi_info(block: &MirBlock) -> (Vec<SsaLocal>, Vec<(usize, SsaLocal)>) {
    let mut defs: Vec<SsaLocal> = Vec::new();
    let mut incoming: Vec<(usize, SsaLocal)> = Vec::new();
    for stmt in &block.stmts {
        if let MirStmt::Assign { dest: MirPlace::Ssa(dest), value: MirRvalue::Phi { values } } = stmt {
            defs.push(*dest);
            for (pred, op) in values {
                match op {
                    MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => {
                        incoming.push((*pred, *s));
                    }
                    MirOperand::Copy(_)
                    | MirOperand::Move(_)
                    | MirOperand::Constant(_)
                    | MirOperand::Sym(_)
                    | MirOperand::Static(_) => {}
                }
            }
        }
    }
    (defs, incoming)
}

/// 块是不是"纯标记块"：语句全是 SetMetadata/EffectCheck（且至少有一条），
/// 终止器是纯 Goto。
///
/// 不含"入口块不算"这条判定——那是调用点的语义（入口块没有前驱，把它的
/// 语句挪去别处没意义），不是"这个块自身是不是标记块"的属性。调用方
/// 自己加 `i != 0`。
pub fn is_marker_block(block: &MirBlock) -> bool {
    if block.stmts.is_empty() {
        return false;
    }
    for stmt in &block.stmts {
        match stmt {
            MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
            MirStmt::Assign { .. } | MirStmt::ExprStmt(_) | MirStmt::Drop { .. } => return false,
        }
    }
    match &block.terminator {
        MirTerminator::Goto(_) => true,
        MirTerminator::If { .. }
        | MirTerminator::Switch { .. }
        | MirTerminator::Return(_)
        | MirTerminator::Unreachable
        | MirTerminator::Placeholder => false,
    }
}

// ==================== 字面量查询 ====================

/// 从操作数里取出整数值（不管来源是哪个位宽/符号的整数变体）。取不出来
/// （不是整数字面量）返回 None。
///
/// 真正该抽出来复用的不是"判断是不是零/一"本身，而是"不管来源是哪个
/// 整数字面量变体，把里面的数值统一取出来"这一步——取出来之后，是零是一
/// 只是跟 0/1 比大小，不需要再重复列一遍十二个变体。
pub fn int_literal_value(op: &MirOperand) -> Option<i128> {
    match op {
        MirOperand::Constant(lit) => match lit {
            Literal::Int8(v) => Some(*v as i128),
            Literal::Int16(v) => Some(*v as i128),
            Literal::Int32(v) => Some(*v as i128),
            Literal::Int64(v) => Some(*v as i128),
            Literal::Int128(v) => Some(*v),
            Literal::UInt8(v) => Some(*v as i128),
            Literal::UInt16(v) => Some(*v as i128),
            Literal::UInt32(v) => Some(*v as i128),
            Literal::UInt64(v) => Some(*v as i128),
            // u128 超出 i128 正数范围的值转成 i128 会变成负数。这里只拿来
            // 跟 0/1 比较，u128 的 0 和 1 转换后不变，超大值不会被误认成
            // 0 或 1，所以这个有损转换对现有调用方是安全的；将来如果有人
            // 拿它做别的数值判断，要留意这一点。
            Literal::UInt128(v) => Some(*v as i128),
            Literal::Isize(v) => Some(*v as i128),
            Literal::Usize(v) => Some(*v as i128),
            Literal::Float16(_)
            | Literal::Float32(_)
            | Literal::Float64(_)
            | Literal::Bool(_)
            | Literal::Char(_)
            | Literal::String(_)
            | Literal::Unit
            | Literal::ByteString(_) => None,
        },
        MirOperand::Copy(_) | MirOperand::Move(_) | MirOperand::Sym(_) | MirOperand::Static(_) => None,
    }
}

/// 同样的道理：is_bool_true/is_bool_false 共用"取出 bool 字面量的值"。
pub fn bool_literal_value(op: &MirOperand) -> Option<bool> {
    match op {
        MirOperand::Constant(Literal::Bool(v)) => Some(*v),
        MirOperand::Constant(_)
        | MirOperand::Copy(_)
        | MirOperand::Move(_)
        | MirOperand::Sym(_)
        | MirOperand::Static(_) => None,
    }
}

pub fn is_zero(op: &MirOperand) -> bool {
    match int_literal_value(op) {
        Some(v) => v == 0,
        None => false,
    }
}

pub fn is_one(op: &MirOperand) -> bool {
    match int_literal_value(op) {
        Some(v) => v == 1,
        None => false,
    }
}

pub fn is_bool_true(op: &MirOperand) -> bool {
    match bool_literal_value(op) {
        Some(v) => v,
        None => false,
    }
}

pub fn is_bool_false(op: &MirOperand) -> bool {
    match bool_literal_value(op) {
        Some(v) => !v,
        None => false,
    }
}

// ==================== 运算会不会陷入 ====================
//
// 关键说明：本节是整个项目里"哪些运算可能 panic"的唯一定义。函数纯度
// （analyze.rs 的 rvalue_purity / body_purity）、死代码删除（rvalue_has_side_effect）、
// 循环不变量外提的投机安全判断（simplify.rs 的 licm_speculation_safe）
// 都只读这里——各处各写一份的话，改语义时总会漏掉一边。
// 调用方应当用 rvalue_may_trap / place_may_trap / operand_may_trap 这三个
// 聚合入口，不要自己在 BinaryOp / Cast 分支里分别调用 binop_may_trap /
// cast_may_trap：那样新增一种会陷入的形态时，每个调用点都得记得补一次。
//
// 语言语义上的假设（改语义只改这里）：
//   - 整数的 Add/Sub/Mul/Neg 溢出按回绕处理，不陷入。如果这门语言里整数
//     溢出会 panic，把 binop_may_trap 里这几个分支改成 true 即可。
//   - 除法/取模：除数是非零、且不是 -1 的整数常量才不陷入（-1 会让
//     MIN / -1 溢出）；除数是变量或者 0/-1 一律当作可能陷入。操作数类型
//     在这里拿不到，浮点除法（IEEE，不陷入）也被保守地算成可能陷入。
//   - `as` 是全函数，永不陷入（见 cast_may_trap）。
//   - 索引访问 `a[i]` 本身会做越界检查，越界即 panic（读和写都一样）。
//     默认出现 Index 投影就算可能陷入；唯一的例外是"下标是整数常量、且
//     base 的静态类型是定长数组、且下标在界内"（见 place_may_trap）。
//     静态类型从 MirBody.locals 和结构体/枚举定义表查，所以这一族函数都要带
//     `&MirBody` 以及 `structs` / `enums`。
//   - Deref 的空/悬垂指针不建模：借用检查保证引用有效，裸指针不在这里处理。

pub fn binop_may_trap(op: BinaryOp, rhs: &MirOperand) -> bool {
    match op {
        BinaryOp::Div | BinaryOp::Mod => match int_literal_value(rhs) {
            Some(v) => v == 0 || v == -1,
            None => true,
        },
        BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Mul
        | BinaryOp::Eq
        | BinaryOp::Neq
        | BinaryOp::Lt
        | BinaryOp::Gt
        | BinaryOp::Le
        | BinaryOp::Ge
        | BinaryOp::And
        | BinaryOp::Or => false,
    }
}

/// 类型转换会不会陷入：永远不会。
///
/// 语言规定 `as` 是全函数：整数↔整数回绕/扩展，浮点→整数饱和（NaN→0），
/// 整数→浮点取最近值，等等，每一种合法的 `as` 对每个输入都有确定结果。
/// 非法的转换（张量之间的转换、改变隐私标签的转换……）在类型检查阶段就
/// 是编译错误，到不了 MIR，所以这里不需要、也不应该再为它们留"可能陷入"。
/// （model 块里对运行期张量值的 `as` 是动态控制流，在 MD003 里单独处理，
/// 同样不会作为 Cast 右值出现在这一层。）
///
/// 由此 `Cast` 是纯运算：结果没被使用可以删，LICM 可以外提。
/// 参数保留着：如果将来语义变了（比如引入会失败的转换），只改这一个函数，
/// 纯度分析、DCE、LICM 经由 rvalue_may_trap 同时跟上。
pub fn cast_may_trap(_operand: &MirOperand, _to: &Type) -> bool {
    false
}

/// 剥掉隐私包装，返回里面真正的类型。
fn strip_privacy(ty: &Type) -> &Type {
    let mut cur = ty;
    while let Type::Privacy(inner, _) = cur {
        cur = &**inner;
    }
    cur
}

/// place 的静态类型。只覆盖能从 locals、结构体/枚举定义表和类型结构直接
/// 推出来的形态，推不出来一律 None（调用方按"不知道"处理，保守）：
///   - Ssa：查 body.locals；
///   - Field：元组（字段名是十进制下标）和*非泛型*结构体（查 structs）；
///     泛型结构体实例要按实参替换字段里的类型参数，这里不做，返回 None；
///   - Index：定长数组/切片的元素类型；
///   - Deref：引用的被指类型；
///   - EnumPayload：*非泛型*枚举的该变体的 payload 类型（查 enums）。
fn place_type(
    place: &MirPlace,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> Option<Type> {
    match place {
        MirPlace::Ssa(s) => {
            for l in &body.locals {
                if l.id == s.base_id {
                    return Some(l.ty.clone());
                }
            }
            None
        }
        MirPlace::Field { base, field } => {
            let base_ty = place_type(base, body, structs, enums)?;
            match strip_privacy(&base_ty) {
                Type::Tuple(elems) => {
                    let idx: usize = field.parse().ok()?;
                    elems.get(idx).cloned()
                }
                Type::Struct(name) => {
                    let def = structs.iter().find(|s| s.name == *name)?;
                    if !def.generic_params.is_empty() {
                        return None;
                    }
                    for (fname, fty) in &def.fields {
                        if fname == field {
                            return Some(fty.clone());
                        }
                    }
                    None
                }
                // 泛型结构体实例（Type::Generic）和其余类型：不知道。Type 变体
                // 很多，"没有可按字段名往里走的结构"是对它们的正确回答。
                _ => None,
            }
        }
        MirPlace::Index { base, .. } => {
            let base_ty = place_type(base, body, structs, enums)?;
            match strip_privacy(&base_ty) {
                Type::Array(elem, _) => Some((**elem).clone()),
                Type::Slice(elem) => Some((**elem).clone()),
                _ => None,
            }
        }
        MirPlace::Deref(base) => {
            let base_ty = place_type(base, body, structs, enums)?;
            match strip_privacy(&base_ty) {
                Type::Ref { inner, .. } => Some((**inner).clone()),
                _ => None,
            }
        }
        MirPlace::EnumPayload { base, enum_name, variant_name } => {
            // base 自己的类型推不出来时整体放弃：base 的类型是否真的是这个
            // 枚举这里无从确认，宁可不答。
            let _ = place_type(base, body, structs, enums)?;
            let def = enums.iter().find(|e| e.name == *enum_name)?;
            if !def.generic_params.is_empty() {
                return None;
            }
            for (vname, vty) in &def.variants {
                if vname == variant_name {
                    return vty.clone();
                }
            }
            None
        }
    }
}

/// place 的求值会不会陷入。Index 投影本身做越界检查，所以*默认*出现即为真，
/// 只有一种情形能静态省掉这次检查：
///   - 下标是整数常量，且 base 的静态类型是定长数组 `Array(_, n)`：能静态
///     判定下标是否在界内——界内不陷入，界外一定陷入（返回 true）。
/// 下标是变量、或 base 是切片/类型未知（长度不固定）时，无法静态判定，
/// 保守返回 true。base 和下标操作数自己也可能陷入（`a[b[i]]`、嵌套的
/// Index），递归检查。
///
/// `body` 只用来查 base 的静态类型（place_type）。精确化之后 `a[0]` 这类
/// 常量下标 + 定长数组的组合不再被当成会陷入的运算：结果没被使用可以删，
/// LICM 也不会因此被挡住。
pub fn place_may_trap(
    place: &MirPlace,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> bool {
    match place {
        MirPlace::Ssa(_) => false,
        MirPlace::Index { base, index } => {
            if place_may_trap(base, body, structs, enums)
                || operand_may_trap(index, body, structs, enums)
            {
                return true;
            }
            if let Some(idx) = int_literal_value(index) {
                if let Some(base_ty) = place_type(base, body, structs, enums) {
                    if let Type::Array(_, len) = strip_privacy(&base_ty) {
                        // 统一提到 u128 比较：负下标一定越界；len 的具体整数
                        // 类型不影响这里。
                        return !(idx >= 0 && (idx as u128) < (*len as u128));
                    }
                }
            }
            true
        }
        MirPlace::Field { base, .. }
        | MirPlace::EnumPayload { base, .. }
        | MirPlace::Deref(base) => place_may_trap(base, body, structs, enums),
    }
}

pub fn operand_may_trap(
    op: &MirOperand,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> bool {
    match op {
        MirOperand::Copy(p) | MirOperand::Move(p) => place_may_trap(p, body, structs, enums),
        MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => false,
    }
}

/// 右值求值（不含被调函数内部）会不会陷入。Call/MethodCall 只看实参：
/// 被调函数自己会不会陷入由纯度表回答。`body` / `structs` / `enums` 用来查
/// place 的静态类型（判常量下标是否在定长数组界内）。
pub fn rvalue_may_trap(
    rv: &MirRvalue,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> bool {
    let op_traps = |o: &MirOperand| operand_may_trap(o, body, structs, enums);
    match rv {
        MirRvalue::BinaryOp(op, l, r) => binop_may_trap(*op, r) || op_traps(l) || op_traps(r),
        MirRvalue::Cast(o, ty) => cast_may_trap(o, ty) || op_traps(o),
        MirRvalue::UnaryOp(_, o) => op_traps(o),
        MirRvalue::Use(op) => op_traps(op),
        MirRvalue::Ref { place, .. } => place_may_trap(place, body, structs, enums),
        MirRvalue::Call { args, .. } => args.iter().any(|a| op_traps(a)),
        MirRvalue::MethodCall { receiver, args, .. } => {
            op_traps(receiver) || args.iter().any(|a| op_traps(a))
        }
        MirRvalue::StructInit { fields, .. } => fields.iter().any(|(_, op)| op_traps(op)),
        MirRvalue::EnumVariantConstruction { args, .. } => args.iter().any(|a| op_traps(a)),
        MirRvalue::ArrayLiteral(elems) => elems.iter().any(|e| op_traps(e)),
        MirRvalue::Discriminant { value, .. } => op_traps(value),
        MirRvalue::Phi { values } => values.iter().any(|(_, op)| op_traps(op)),
    }
}

// ==================== Deref 读写扫描 ====================
//
// 回答"这个节点的求值过程里有没有经过 Deref 读/写"。纯查询，别名分析
// （AliasInfo::build）和函数纯度（body_purity）共用，所以放在这里。
// 约定：只会把 *reads / *writes 置 true，从不清除，调用方可以对同一对
// 标志连续累积多个节点。
//
// 注意这里的 writes 只记录"通过操作数/place 扫描能看到的写"——目前 place
// 作为被读取对象扫描时只产生 reads；赋值目标的写入由调用方用
// place_has_deref(dest) 另行判定，调用的间接写入由 AliasInfo 的
// scan_call_writes 处理。

pub fn scan_place_deref(place: &MirPlace, reads: &mut bool, writes: &mut bool) {
    match place {
        MirPlace::Ssa(_) => {}
        MirPlace::Field { base, .. } | MirPlace::EnumPayload { base, .. } => {
            scan_place_deref(base, reads, writes);
        }
        MirPlace::Index { base, index } => {
            scan_place_deref(base, reads, writes);
            scan_operand_deref(index, reads, writes);
        }
        MirPlace::Deref(base) => {
            *reads = true;
            scan_place_deref(base, reads, writes);
        }
    }
}

pub fn scan_operand_deref(op: &MirOperand, reads: &mut bool, writes: &mut bool) {
    match op {
        MirOperand::Copy(p) | MirOperand::Move(p) => scan_place_deref(p, reads, writes),
        MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => {}
    }
}

pub fn scan_rvalue_deref(rv: &MirRvalue, reads: &mut bool, writes: &mut bool) {
    match rv {
        MirRvalue::Use(op) => scan_operand_deref(op, reads, writes),
        MirRvalue::BinaryOp(_, l, r) => {
            scan_operand_deref(l, reads, writes);
            scan_operand_deref(r, reads, writes);
        }
        MirRvalue::UnaryOp(_, o) => scan_operand_deref(o, reads, writes),
        MirRvalue::Cast(o, _) => scan_operand_deref(o, reads, writes),
        MirRvalue::Call { args, .. } => {
            for a in args {
                scan_operand_deref(a, reads, writes);
            }
        }
        MirRvalue::MethodCall { receiver, args, .. } => {
            scan_operand_deref(receiver, reads, writes);
            for a in args {
                scan_operand_deref(a, reads, writes);
            }
        }
        MirRvalue::StructInit { fields, .. } => {
            for (_, op) in fields {
                scan_operand_deref(op, reads, writes);
            }
        }
        MirRvalue::EnumVariantConstruction { args, .. } => {
            for a in args {
                scan_operand_deref(a, reads, writes);
            }
        }
        MirRvalue::Ref { place, .. } => scan_place_deref(place, reads, writes),
        MirRvalue::ArrayLiteral(elems) => {
            for e in elems {
                scan_operand_deref(e, reads, writes);
            }
        }
        MirRvalue::Discriminant { value, .. } => scan_operand_deref(value, reads, writes),
        MirRvalue::Phi { values } => {
            for (_, op) in values {
                scan_operand_deref(op, reads, writes);
            }
        }
    }
}

/// 终止器里的操作数（If 条件、Switch 判别式、返回值）的三项事实：
/// (经过 Deref 读, 经过 Deref 写, 求值可能陷入)。
/// AliasInfo::build 取前两项，body_purity 取第一、三项——两处共用，
/// 新增带操作数的终止器时只改这里。
pub fn scan_terminator_facts(
    term: &MirTerminator,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> (bool, bool, bool) {
    let mut reads = false;
    let mut writes = false;
    let mut traps = false;
    match term {
        MirTerminator::If { cond, .. } => {
            scan_operand_deref(cond, &mut reads, &mut writes);
            traps = operand_may_trap(cond, body, structs, enums);
        }
        MirTerminator::Switch { discr, .. } => {
            scan_operand_deref(discr, &mut reads, &mut writes);
            traps = operand_may_trap(discr, body, structs, enums);
        }
        MirTerminator::Return(Some(op)) => {
            scan_operand_deref(op, &mut reads, &mut writes);
            traps = operand_may_trap(op, body, structs, enums);
        }
        MirTerminator::Goto(_)
        | MirTerminator::Return(None)
        | MirTerminator::Unreachable
        | MirTerminator::Placeholder => {}
    }
    (reads, writes, traps)
}

// ==================== MIR 属性判定 ====================

/// 哪些局部变量是 Copy 类型。复制传播（copy_map）和 GVN 都用它，两处必须
/// 用同一个判定——各写一份的话，将来 is_copy 的口径调整时很容易只改到一边。
pub fn copy_local_ids(body: &MirBody) -> HashSet<usize> {
    let mut out: HashSet<usize> = HashSet::new();
    for local in &body.locals {
        if local.ty.is_copy() {
            out.insert(local.id);
        }
    }
    out
}

/// 整个函数里被定义（作为 Assign 的 SSA 目标）超过一次的 SSA 值。GVN 靠
/// "每个 SSA 名对应唯一一处定义"这条前提，遇到重复定义时主动跳过——这个
/// 函数就是那个"跳过清单"。
pub fn multiply_defined_ssa(body: &MirBody) -> HashSet<SsaLocal> {
    let mut seen: HashSet<SsaLocal> = HashSet::new();
    let mut multi: HashSet<SsaLocal> = HashSet::new();
    for block in &body.blocks {
        for stmt in &block.stmts {
            if let MirStmt::Assign { dest: MirPlace::Ssa(ssa), .. } = stmt {
                if !seen.insert(*ssa) {
                    multi.insert(*ssa);
                }
            }
        }
    }
    multi
}

/// SSA 值 -> 它被定义的那个块。全函数扫描一次建好，供"这个 holder 的定义
/// 是否支配某块"这类查询用（见 simplify.rs::pick_holder，以及 control.rs
/// 里按定义追溯布尔条件的 resolve_boolean）。跟 multiply_defined_ssa 是同
/// 一类按 SSA 号查信息的辅助表。
///
/// **只在规范 SSA 上有意义**：同名多定义时只记首次定义所在的块，它不一定
/// 支配所有使用点。调用方要么先用 multiply_defined_ssa 把这类名字从表里
/// 剔掉（simplify.rs 的做法），要么只在已确认规范的 SSA 上用。
pub fn ssa_def_blocks(body: &MirBody) -> HashMap<SsaLocal, usize> {
    let mut out: HashMap<SsaLocal, usize> = HashMap::new();
    for (bi, block) in body.blocks.iter().enumerate() {
        for stmt in &block.stmts {
            if let MirStmt::Assign { dest: MirPlace::Ssa(s), .. } = stmt {
                out.entry(*s).or_insert(bi);
            }
        }
    }
    out
}

/// 操作数是不是"常量，或者 Copy 类型的 SSA"。GVN 给一个右值编号之前，它的
/// 每个操作数都得满足这条。
///
/// Field/Index/Deref/EnumPayload 读的是内存里的位置，中间可能被别的写入
/// 改掉——同一个位置读两次不保证同一个值，不能编号。Sym/Static 不指向
/// 局部变量，也不在处理范围。非 Copy 的 SSA 读第二次会消耗所有权，同一个
/// 值不可能被两次合法地读到，同样不编号。
pub fn operand_is_copy_ssa_or_const(op: &MirOperand, copy_base: &HashSet<usize>) -> bool {
    match op {
        MirOperand::Constant(_) => true,
        MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => {
            copy_base.contains(&s.base_id)
        }
        MirOperand::Copy(_)
        | MirOperand::Move(_)
        | MirOperand::Sym(_)
        | MirOperand::Static(_) => false,
    }
}

/// place 里有没有 Deref 投影。借用的权威展开判据、别名分析的入口判据都
/// 用它。
pub fn place_has_deref(place: &MirPlace) -> bool {
    match place {
        MirPlace::Ssa(_) => false,
        MirPlace::Field { base, .. }
        | MirPlace::Index { base, .. }
        | MirPlace::EnumPayload { base, .. } => place_has_deref(base),
        MirPlace::Deref(_) => true,
    }
}

/// place 的根 SSA——忽略所有投影一路找到底（经过 Deref 也照样往下找）。
/// 注意它只回答"根变量是谁"，不看下标操作数里的 SSA；要"place 读了哪些
/// SSA"（含下标）见 analyze.rs 的 ssa_uses_in_place。
pub fn place_root_ssa(place: &MirPlace) -> Option<SsaLocal> {
    match place {
        MirPlace::Ssa(s) => Some(*s),
        MirPlace::Field { base, .. }
        | MirPlace::Index { base, .. }
        | MirPlace::Deref(base)
        | MirPlace::EnumPayload { base, .. } => place_root_ssa(base),
    }
}

/// place 访问路径上有没有枚举 payload 投影。EnumPayload 只有在 Switch 已经
/// 确认了变体之后才有意义，读它在 codegen 里会落成"变体不对就 unreachable!"，
/// 所以它不能被投机执行（挪到判别式检查之外）。
pub fn place_has_enum_payload(place: &MirPlace) -> bool {
    match place {
        MirPlace::Ssa(_) => false,
        MirPlace::EnumPayload { .. } => true,
        MirPlace::Field { base, .. } | MirPlace::Index { base, .. } | MirPlace::Deref(base) => {
            place_has_enum_payload(base)
        }
    }
}

// ==================== 内存位置的别名判定 ====================
//
// 给定两个 MirPlace，回答"它们*一定*不指向同一块内存吗"。
//
// 用途：内存优化要证明"这个读跟那个写不冲突"才能变换（LICM 外提内存读、
// 以后 GVN 合并内存读、死存储消除）。兜底方向跟 borrow.rs 的 places_overlap
// 相反：那里判"*可能*重叠"（宁可误报，不能漏报）；这里判"*一定*不重叠"
// （宁可不动，不能改错）。任何不能从 place 本身确认的东西（Deref、动态下标、
// 不同种类的投影）一律保守返回 false。不依赖 points-to，纯结构判定。

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProjKind {
    Field(String),
    ConstIndex(i128),
    DynIndex,
    Variant(String),
}

/// 把 place 拆成 (根变量的 base_id, 从根往外的投影序列)。带 Deref 的 place
/// 返回 None（不知道指向哪）。比较的是 base_id 而不是带版本的 SsaLocal：
/// 同一个变量的不同版本保守地当作同一块内存。
fn place_root_and_projs(place: &MirPlace) -> Option<(usize, Vec<ProjKind>)> {
    let mut projs: Vec<ProjKind> = Vec::new();
    let mut cur = place;
    loop {
        match cur {
            MirPlace::Ssa(s) => {
                projs.reverse();
                return Some((s.base_id, projs));
            }
            MirPlace::Field { base, field } => {
                projs.push(ProjKind::Field(field.clone()));
                cur = base;
            }
            MirPlace::Index { base, index } => {
                let proj = match int_literal_value(index) {
                    Some(v) => ProjKind::ConstIndex(v),
                    None => ProjKind::DynIndex,
                };
                projs.push(proj);
                cur = base;
            }
            MirPlace::EnumPayload { base, variant_name, .. } => {
                projs.push(ProjKind::Variant(variant_name.clone()));
                cur = base;
            }
            MirPlace::Deref(_) => return None,
        }
    }
}

pub fn must_not_alias(a: &MirPlace, b: &MirPlace) -> bool {
    let (ra, pa) = match place_root_and_projs(a) {
        Some(x) => x,
        None => return false,
    };
    let (rb, pb) = match place_root_and_projs(b) {
        Some(x) => x,
        None => return false,
    };
    // 不同的局部变量是不同的存储（带 Deref 的已经在上面排除了：通过引用
    // 访问的内存可能就是另一个变量）。
    if ra != rb {
        return true;
    }
    // 同一个根：沿着投影序列同步往里走，第一次出现"一定不同的分叉"就不相交；
    // 走完较短的一条还没分叉，说明一条是另一条的前缀（或相等），重叠。
    let n = pa.len().min(pb.len());
    for i in 0..n {
        match (&pa[i], &pb[i]) {
            (ProjKind::Field(x), ProjKind::Field(y)) => {
                if x != y {
                    return true;
                }
            }
            (ProjKind::ConstIndex(x), ProjKind::ConstIndex(y)) => {
                if x != y {
                    return true;
                }
            }
            // 枚举不同变体的 payload 共用同一块存储（联合体布局），名字不同
            // 并不意味着不重叠：只有同名才继续往里比，不同名保守地算重叠。
            (ProjKind::Variant(x), ProjKind::Variant(y)) => {
                if x != y {
                    return false;
                }
            }
            // 动态下标可能落在任何位置；种类不同（字段 vs 下标 vs 变体）
            // 说明两条路径对同一块内存的看法不一致，也保守处理。
            (ProjKind::DynIndex, _)
            | (_, ProjKind::DynIndex)
            | (ProjKind::Field(_), ProjKind::ConstIndex(_))
            | (ProjKind::Field(_), ProjKind::Variant(_))
            | (ProjKind::ConstIndex(_), ProjKind::Field(_))
            | (ProjKind::ConstIndex(_), ProjKind::Variant(_))
            | (ProjKind::Variant(_), ProjKind::Field(_))
            | (ProjKind::Variant(_), ProjKind::ConstIndex(_)) => return false,
        }
    }
    false
}

/// place 访问路径上"直接出现的引用变量"——如果 place 里有一层
/// `Deref(Ssa(s))`，返回 s。borrow.rs 用它判定"这次访问正是通过引用 s
/// 进行的"：s 携带的 loan 是这次访问的授权来源，不是冲突源（`*r = 5` 时
/// r 的 loan 授权了这次写入）。
///
/// 这里只回答"通过哪个引用变量访问"，不回答"这个变量携带哪些 loan"——
/// 后者要沿别名和 Phi 做数据流（见 borrow.rs 的 RefFacts.carried），不是
/// 单点查询。嵌套解引用（`**r`、`(*r).f` 再解引用）返回最里面的那个
/// 引用变量。
pub fn authorized_borrow_var(place: &MirPlace) -> Option<SsaLocal> {
    match place {
        MirPlace::Deref(base) => match base.as_ref() {
            MirPlace::Ssa(s) => Some(*s),
            MirPlace::Field { .. }
            | MirPlace::Index { .. }
            | MirPlace::Deref(_)
            | MirPlace::EnumPayload { .. } => authorized_borrow_var(base),
        },
        MirPlace::Field { base, .. }
        | MirPlace::Index { base, .. }
        | MirPlace::EnumPayload { base, .. } => authorized_borrow_var(base),
        MirPlace::Ssa(_) => None,
    }
}

// ==================== 终结器查询 ====================

/// 顺着 redirect 链爬到最终代表。redirect 只会指向更小的下标（构建时
/// 保证），不会成环，guard 纯粹防御性。
pub fn redirect_canonical(redirect: &[Option<usize>], start: usize) -> usize {
    let mut id = start;
    let mut guard = 0usize;
    while let Some(next) = redirect.get(id).copied().flatten() {
        id = next;
        guard += 1;
        if guard > 256 {
            break;
        }
    }
    id
}

/// 两个终结器是不是"等价"。Goto/If/Switch 的目标先顺着 redirect 链归一化
/// 再比——这样只要目标最终等价，中间差着几层重定向也认得出来。
pub fn terminators_equivalent(a: &MirTerminator, b: &MirTerminator, redirect: &[Option<usize>]) -> bool {
    match (a, b) {
        (MirTerminator::Goto(x), MirTerminator::Goto(y)) => {
            redirect_canonical(redirect, *x) == redirect_canonical(redirect, *y)
        }
        (
            MirTerminator::If { cond: c1, then_block: t1, else_block: e1 },
            MirTerminator::If { cond: c2, then_block: t2, else_block: e2 },
        ) => {
            c1 == c2
                && redirect_canonical(redirect, *t1) == redirect_canonical(redirect, *t2)
                && redirect_canonical(redirect, *e1) == redirect_canonical(redirect, *e2)
        }
        (
            MirTerminator::Switch { discr: d1, discr_ty: dt1, targets: tg1, default: df1 },
            MirTerminator::Switch { discr: d2, discr_ty: dt2, targets: tg2, default: df2 },
        ) => {
            d1 == d2
                && dt1 == dt2
                && redirect_canonical(redirect, *df1) == redirect_canonical(redirect, *df2)
                && tg1.len() == tg2.len()
                && tg1.iter().zip(tg2.iter()).all(|((v1, t1), (v2, t2))| {
                    v1 == v2 && redirect_canonical(redirect, *t1) == redirect_canonical(redirect, *t2)
                })
        }
        (MirTerminator::Return(x), MirTerminator::Return(y)) => x == y,
        (MirTerminator::Unreachable, MirTerminator::Unreachable) => true,
        (MirTerminator::Placeholder, MirTerminator::Placeholder) => true,
        // 两边变体不同：不可能等价。显式列出 a 的每个变体而不是 `_`，新增
        // 终结器变体时这里会编译失败，提醒在上面补上对应的比较分支。
        (MirTerminator::Goto(_), _)
        | (MirTerminator::If { .. }, _)
        | (MirTerminator::Switch { .. }, _)
        | (MirTerminator::Return(_), _)
        | (MirTerminator::Unreachable, _)
        | (MirTerminator::Placeholder, _) => false,
    }
}
