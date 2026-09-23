// mir.rs
//
// 设计基础采用三地址码（3AC）风格：MirRvalue 右边只允许"一层"操作，
// 操作数（MirOperand）永远是 Copy/Move/常量三选一，不能直接嵌套另一个
// MirRvalue。任何复杂子表达式在构建阶段就必须先落进一个临时局部变量，
// 再引用那个临时变量——这是为了让后面 simplify.rs 的常量折叠/死代码
// 消除可以逐条语句独立处理，不用递归下钻表达式树。
// 关键修复：这里必须是 pub use，不能是私有 use——simplify.rs 全篇直接
// 写 `Literal::Int(0)`、`use BinaryOp::*;`，却没有自己
// `use crate::ast::{BinaryOp, Literal};`，指望 `use crate::mir::*;`
// 能把这些类型带出去。私有 use 只在 mir.rs 内部可见，做不到这件事。
pub use crate::ast::{Type, BinaryOp, UnaryOp, Literal};
pub use crate::hir::{EffectSet, HirGenericParam};
use crate::intrinsic::IntrinsicFn;

#[derive(Debug, Clone)]
pub struct MirProgram {
    pub structs: Vec<MirStruct>,
    pub enums: Vec<MirEnum>,
    pub fns: Vec<MirFn>,
    // 保留：codegen 需要知道这次编译实际用到了哪些内建函数，才能决定
    // 要不要在生成的 Rust 里带上对应的运行时支持代码。
    // 关键修复：这条注释原来说"等 intrinsic.rs 的注册表做好之后……
    // 这轮先不动"——intrinsic.rs 的注册表现在已经做好了，
    // mir_builder.rs 也已经改成通过 IntrinsicFn::from_str + get_intrinsic
    // 真正查表（不再是写死的小集合判断，那个过渡用的 lookup_by_str
    // 兼容接口也已经删掉了），字段类型也从 Vec<String> 换成了
    // Vec<IntrinsicFn>。这条注释是上一轮的"还没做"记录，留着没更新
    // 会误导下一个读到这里的人以为还是权宜之计，所以更新掉。
    pub intrinsics_used: Vec<IntrinsicFn>,
}

#[derive(Debug, Clone)]
pub struct MirStruct {
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone)]
pub struct MirEnum {
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub variants: Vec<(String, Option<Type>)>,
}

#[derive(Debug, Clone)]
pub struct MirFn {
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: MirBody,
    // 保留：codegen 生成 Rust 函数时要知道函数体有没有 io/rng/ai/ffi/panic
    // 副作用。直接复用 hir::EffectSet，不重新定义一遍类型（重新定义会
    // 导致 f.effects.clone() 塞不进这个字段——这个坑上一轮已经踩过一次）。
    pub effect_set: EffectSet,
}

// ---- 函数体：局部变量表 + 基本块列表 ----
#[derive(Debug, Clone)]
pub struct MirBody {
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<MirBlock>,
}

#[derive(Debug, Clone)]
pub struct MirLocal {
    pub id: usize,
    pub name: Option<String>, // None 表示编译器自己造的临时变量
    pub ty: Type,
    pub mutable: bool,
    // 保留：persist 域穿透——`persist let`/`persist var` 声明的张量，在
    // model 块的 forward 调用之间要保留状态，不能被当成普通临时变量
    // 随意丢弃/覆盖。
    pub persist: bool,
    // 关键新增：这个 local 是不是函数参数。mir_builder.rs 为了让参数
    // 也能通过统一的 MirPlace::Local(id) 寻址，把每个参数都用
    // new_local 注册进了 body.locals（跟普通临时变量放在同一张表
    // 里）——这本身没问题，但 codegen.rs 之前对 body.locals 里的每一项
    // 都无条件生成一条新的 `let name: Ty = 默认值;` 语句，对参数来说，
    // 这等于在函数体最开头用一个假的占位值把刚传进来的真实参数值
    // 原地覆盖掉：函数看起来编译得过、跑起来所有参数却都读到默认值
    // （0/false/...），是个不报错但结果全错的坑。加这个字段，让
    // codegen.rs 能明确分辨"这个 local 该不该重新 let 声明一遍"——
    // 是参数就直接复用 Rust 函数签名里已经绑定好的同名变量，不是的
    // 才需要自己声明。
    pub is_param: bool,
}

#[derive(Debug, Clone)]
pub struct MirBlock {
    pub id: usize,
    pub stmts: Vec<MirStmt>,
    pub terminator: MirTerminator,
}

// ---- 语句：三地址码风格，右边只允许"一层"操作，不能嵌套表达式 ----
// 关键修复：补上 PartialEq——control.rs 的 tail-merge 要比较两个块的
// 语句列表是不是逐条相同，才能判断"这两个块的尾巴能不能合并成一个"，
// 需要 `MirStmt` 支持 `==`。同一批修复也给了 MirRvalue/MirTerminator，
// 三者互相嵌套，缺哪个都比较不完整；里面用到的 Literal/BinaryOp/
// UnaryOp/Type/IntrinsicFn/MirPlace/MirOperand 都已经有 PartialEq，
// 这里补上不会引出新的连锁缺口。
#[derive(Debug, Clone, PartialEq)]
pub enum MirStmt {
    Assign { dest: MirPlace, value: MirRvalue },
    /// 只为副作用执行、不关心结果的调用，比如 print(...)
    ExprStmt(MirRvalue),
    /// 保留：显式 Drop（所有权边界/作用域结束），自举编译器生成正确的
    /// Rust 所有权代码迟早要用上。
    Drop { place: MirPlace },
    /// 保留：隐私标签 Join 结果等元数据标记，供后端生成校验代码。
    SetMetadata { place: MirPlace, key: String, value: String },
    /// 保留：unsafe 块入口的效果标记。
    EffectCheck { effect: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaLocal {
    pub base_id: usize,
    pub version: u32,
}

// ---- 左值：一个局部变量，或者它的字段/索引/解引用 ----
// 关键修复：补上 PartialEq——simplify.rs 里 `left == right`（比较两个
// MirOperand，间接需要 MirPlace 也实现 PartialEq）编译不过，之前只
// derive 了 Debug/Clone。
//
// 关键重构（Sym/Static 搬家）：原来这里有 Static(String) 和
// Sym(String) 两个变体，是"内建关联常量"（i128::MAX）和"符号形状
// 引用"（Sym<B> 里的 B）拆开之后各自的落点。但拆开之后再回头看，这
// 两个变体本来就不该留在 MirPlace 里——MirPlace 的定义是"能被赋值/
// 取地址的位置"，Ssa/Field/Index/Deref/EnumPayload 都符合这条：可以
// 出现在赋值左边、可以借用、消耗之后需要 Drop。i128::MAX 和 Sym<B>
// 一条都不符合：
//   - 不能被赋值（`i128::MAX = 5;`/`B = 5;` 都没有意义——前者是只读
//     常量，后者是 model 的超参数，编译期/JIT 期绑定，不是运行时能
//     写的东西）
//   - 不能被取地址、不能被 move（它们不指向内存里的字节）
//   - 不需要 Drop（没有资源）
// 它们其实是*值*，跟 Literal::Usize(32) 是同一类东西，只是具体数值在
// 编译期还不知道（Sym）或者是一个有名字的编译期常量（Static）。值该
// 待在 MirOperand 里，不该套一层 Move/Copy 假装自己是在读某个"位置"。
// 现在把这两个变体挪到下面 MirOperand 里，跟 Constant 并列。
#[derive(Debug, Clone, PartialEq)]
pub enum MirPlace {
    Ssa(SsaLocal),
    Field { base: Box<MirPlace>, field: String },
    /// 新增：索引访问（bytes[i]/arr[i] 这种写法标准库里到处都是，不能
    /// 没有）。索引本身必须是 MirOperand，不是 Box<MirRvalue>——如果索引
    /// 是复杂表达式（比如 arr[i + 1]），i + 1 得先算进一个临时变量，这里
    /// 只存"读那个临时变量"这个操作数，跟整个文件的三地址码原则保持
    /// 一致，不能因为"只是个索引"就破例塞一整棵表达式树进来。
    Index { base: Box<MirPlace>, index: Box<MirOperand> },
    Deref(Box<MirPlace>),
    /// 关键新增：枚举变体的 payload——EnumVariantWithBinding 模式
    /// （比如 `Ok(v) => ...` 里的 v）要把 base 这个枚举值里"假设它是
    /// variant_name 这个变体"时带的那份数据取出来，绑定成一个新局部
    /// 变量。之所以是"假设"：这个 MirPlace 只应该出现在 Switch 已经
    /// 把判别式跟 variant_name 对应的下标匹配过、确实进了那个分支块
    /// 之后——在那之前用它是没有意义的（读到的东西不知道是谁）。跟
    /// Index 一样，base 存的是"读哪个局部变量"，不是一整棵表达式树。
    /// 关键修复：补上 enum_name。codegen 生成 Rust 代码时要落地成类似
    /// `match base { EnumName::Variant(x) => x, _ => unreachable!() }`
    /// 这样的表达式，`EnumName::` 这个限定路径是必需的（Rust 里裸写
    /// `Variant(x)` 只有在 `use EnumName::*` 之类的前提下才可能生效，
    /// 不能假设调用方一定这么写了）——只存 variant_name、不存
    /// enum_name，codegen 那边根本无从得知该拼哪个类型名。
    EnumPayload { base: Box<MirPlace>, enum_name: String, variant_name: String },
}

// ---- 操作数：读一个左值的值，要么拷贝要么移动，或者是字面量/编译期
// 已知的符号常量 ----
#[derive(Debug, Clone, PartialEq)]
pub enum MirOperand {
    Copy(MirPlace),
    Move(MirPlace),
    Constant(Literal),
    /// 新增（从 MirPlace::Sym 搬过来，见上面 MirPlace 定义处的说明）：
    /// 符号形状引用（`Sym<B>` 里的 `B`）。它是一个 usize 值，就像
    /// `Literal::Usize(32)` 一样，只是具体数值在编译期还不知道，要留到
    /// JIT 期才能绑定——这正是它不能是 Constant(Literal) 的原因：
    /// Literal 要求一个编译期就确定的具体值，Sym 没有。
    /// Sym 目前在 sema 里还没有真正的符号求解语义（一律按 I32
    /// 处理），这里先给它一个跟内建关联常量干净分开的落点。
    ///
    /// **别处需要跟着改的地方（这份文件之外，本次修改没有触达）**：
    /// codegen.rs 对 MirOperand 的 match 目前大概率是穷尽的（否则新增
    /// 变体不会导致 E0004），必须补一条 `MirOperand::Sym(name) => ...`
    /// 分支——在 calc.rs/simplify.rs 真正实现符号求解、决定 Sym<N>
    /// 该怎么落地（比如降维成一个运行时/JIT 期绑定的标量参数）之前，
    /// 这条分支至少应该显式报错（"Sym<N> 尚未实现代码生成"）。
    Sym(String),
    /// 新增（从 MirPlace::Static 搬过来，见上面 MirPlace 定义处的
    /// 说明）：内建关联常量（如 i128::MAX、f64::NAN）。codegen 落地成
    /// `name.clone()` 是对的——这些本来就是 Rust 里合法的、可以裸写
    /// 的关联常量路径；"字符串是不是合法 Rust 路径"是 codegen 的实现
    /// 细节，不是 MIR 层该关心的语义，MIR 层它就是一个值。
    ///
    /// **别处需要跟着改的地方**：codegen.rs 对 MirOperand 的 match
    /// 需要补一条 `MirOperand::Static(name) => name.clone()`（把原来
    /// `MirPlace::Static` 分支的逻辑原样搬过来）。
    Static(String),
}

// ---- 右值：能一步算出来的操作，是"拍平"这个词真正的落点 ----
// 同上补 PartialEq，供 control.rs 的 tail-merge 比较用。
#[derive(Debug, Clone, PartialEq)]
pub enum MirRvalue {
    Use(MirOperand),
    BinaryOp(BinaryOp, MirOperand, MirOperand),
    /// 新增：一元运算符（Neg/Not）
    UnaryOp(UnaryOp, MirOperand),
    /// 新增：as 类型转换
    Cast(MirOperand, Type),
    Call {
        func: String,
        args: Vec<MirOperand>,
        is_intrinsic: bool,
        /// 新增：显式泛型实参。`Vec::new()`/`Vec::with_capacity()` 这类
        /// "返回值带泛型参数、但参数列表完全看不出来"的调用，sema 那边
        /// 已经推导出具体类型了（比如 T=u8），这份信息必须传到这里、再
        /// 传给 codegen，才能生成 `Vec::<u8>::new()` 这种明确的 Rust 代码
        /// ——不然 rustc 自己也推不出来，这正是之前 `E0282: type
        /// annotations needed` 那批报错的根因。
        intrinsic_name: Option<IntrinsicFn>,
        generic_args: Vec<Type>,
    },
    MethodCall {
        receiver: MirOperand,
        method: String,
        args: Vec<MirOperand>,
        generic_args: Vec<Type>,
    },
    // 关键修复：补上 generic_args。hir.rs 里 HirExprKind::StructInit 本来
    // 就带着 generic_args: Vec<Type>（sema 推导出的具体类型实参，比如
    // Box<T> 的 T=i32），但 lower 到 MIR 这一步把它弄丢了——mir_builder.rs
    // 手上明明有这份信息，MirRvalue::StructInit 这个类型上却没地方放，
    // 传不过来。后果是 monomorphic.rs 的结构体单态化完全没法工作：它要
    // 知道"这次 StructInit 实例化的是哪个具体类型"才能生成
    // `Point_i32` 这样的具体结构体、并把这个调用点改写成引用那个具体
    // 结构体，缺了 generic_args 就是瞎子摸象，只能靠字段值反推类型，
    // 太脆弱。这里补上，跟 Call 早就有的 generic_args 是同一个道理
    // （Call 的注释里已经讲过一次这个坑）。
    StructInit {
        struct_name: String,
        generic_args: Vec<Type>,
        fields: Vec<(String, MirOperand)>,
    },
    /// 新增：枚举变体构造。Ok/Err/Some/None 全靠它，语言里几乎所有错误
    /// 处理路径都会用到，不能没有。
    // 关键修复：跟 StructInit 同一个坑——hir.rs 的
    // HirExprKind::EnumVariantConstruction 本来就带 generic_args
    // （Option<T>/Result<T,E> 这类泛型枚举，T/E 在这里），lower 到 MIR
    // 时也弄丢了。没有这份信息，monomorphic.rs 没法知道某次
    // `Some(x)`/`Ok(x)` 构造的到底是 `Option<i32>` 还是 `Option<String>`，
    // 泛型枚举的单态化根本无从谈起。
    EnumVariantConstruction {
        enum_name: String,
        generic_args: Vec<Type>,
        variant_name: String,
        args: Vec<MirOperand>,
    },
    Ref { mutable: bool, place: MirPlace },
    ArrayLiteral(Vec<MirOperand>),
    /// 关键新增：取一个枚举值的判别式（哪个变体），配合
    /// MirTerminator::Switch 使用。Switch.discr 要求的是一个能直接跟
    /// i64 比较的标量，但 match 的条件表达式算出来的是完整的枚举值
    /// 本身（可能带 payload，比如 Option<i32>）——中间少了"从聚合值里
    /// 取出判别式"这一步，不能直接把整个枚举值原样塞进 discr 假装它
    /// 已经是标量。这个变体把"取判别式"显式记下来，是只读操作，不
    /// 消耗/不改变 operand 指向的值——同一个枚举值后面还可能要在匹配
    /// 到的分支里用 EnumPayload 取 payload，不能在算判别式这一步就把
    /// 它耗尽（所以 mir_builder.rs 这边传进来的操作数用的是 Copy，不
    /// 是 Move）。
    /// 关键修复：补上 enum_name，原因跟 EnumPayload 一样——Rust 的
    /// data-carrying enum 没法直接 `as i64`，真要落地成 Rust 代码，
    /// 只能生成一个 `match value { EnumName::V0(..) => 0, EnumName::V1
    /// (..) => 1, ... }` 这样的表达式，codegen 得知道是哪个枚举、以及
    /// 它按声明顺序都有哪些变体（这份变体列表在 MirProgram.enums 里，
    /// 用 enum_name 去查）——只存"要取判别式的操作数"，不存是哪个枚举，
    /// codegen 没法凭空补全这个信息。
    Discriminant { value: MirOperand, enum_name: String },
    Phi {
        values: Vec<(usize, MirOperand)>,
    }
}

// ---- 终结指令：一个基本块怎么结束、往哪跳 ----
// 同上补 PartialEq，供 control.rs 的 tail-merge 比较用。
#[derive(Debug, Clone, PartialEq)]
pub enum MirTerminator {
    Goto(usize),
    If { cond: MirOperand, then_block: usize, else_block: usize },
    Return(Option<MirOperand>),
    // 关键重构（Placeholder/Unreachable 拆分）：这个变体原来还兼职当
    // "新块默认值/还没决定终止器"用（state.rs::new_block 创建块时给的
    // 初始值），跟它真正的语义——"已知执行不到这里"（发散之后、match
    // 穷尽之后……）——完全是两件事，用同一个值表示导致
    // `current_terminator_is_placeholder()` 这类判断名不副实：它问的
    // 其实是"这个块的终止器还没被真正设置过"，答案却要靠"terminator
    // 是不是恰好等于 Unreachable"这个巧合来兜底，一旦真的有代码路径
    // 需要往一个块里显式写"这里已知不可达"（而不是"还没想好写什么"），
    // 这两种情况就分不清了。现在两件事拆成两个变体：
    //   - Unreachable：真正的、已知的"没有下一步"——发散调用之后
    //     （build_expr_rvalue 里那次 Never 调用）、if/match 分支自身
    //     发散、所有分支都发散……这些地方设置的是这个变体，语义上
    //     "这里确实不可达"。
    //   - Placeholder：纯粹的"还没写"，只有 new_block 会用它初始化，
    //     任何真正跑完的构建路径最终都应该把它替换成上面某个具体的
    //     终止器；如果 MIR 里最终还留着一个 Placeholder 没被替换掉，
    //     那是构建逻辑本身有缺口，不是"这段代码确实到不了这里"。
    Placeholder,
    Unreachable,
    // 关键修复：这个变体原来在这里断掉了——`Switch { ... }` 自己的花括号
    // 倒是配对了（`default: usize,` 后面那个 `}` 关的是 Switch 这个
    // struct-variant 自己），但外面 `enum MirTerminator { ... }` 整个的
    // 收尾花括号漏写了，文件就在这里硬生生截断，rustc 会一路找到文件
    // 末尾都找不到匹配的 `}`，报"unclosed delimiter"，整个文件过不了
    // 语法分析，是比任何语义问题都更基础的一层错误。
    //
    // 设计上没问题：Switch 是给多分支的 match（Pattern::EnumVariant /
    // EnumVariantWithBinding）用的，比逐个 if-else 链更贴近 LLVM 的
    // `switch` 指令，也更方便以后接 LLVM 后端时直接对应过去——
    // discr 是判别式操作数，targets 是"判别式等于某个 i64 就跳到哪个
    // 块"的列表，default 是都不匹配时的兜底块（穷尽匹配时对应
    // Wildcard 分支，没有 Wildcard 时是一个永远不会被跳进来的
    // Unreachable 块）。
    Switch {
        discr: MirOperand,
        discr_ty: Type,
        targets: Vec<(Literal, usize)>,
        default: usize,
    },
}