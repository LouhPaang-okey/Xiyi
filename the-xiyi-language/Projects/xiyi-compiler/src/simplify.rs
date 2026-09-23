// simplify.rs
use crate::mir::*;
use crate::calc::Calc;
use std::collections::{HashMap, HashSet};

const MAX_OPT_ITERATIONS: usize = 20; // 收敛性控制

pub struct Simplify;

impl Simplify {
    pub fn run(mut program: MirProgram) -> MirProgram {
        for f in &mut program.fns {
            Self::optimize_function(f);
        }
        program
    }

    // ===== 优化主循环 =====
    // 关键修复：eliminate_dead_stores 和 fold_terminator 这两个函数在
    // 旧版本里已经写好了，但从来没有被这里调用过——旧版 fold_block 的
    // 签名/实现本身也编译不过（下面会细说），大概率是上一轮改 SSA 改
    // 到一半的中间状态。这一轮全部接回主循环：
    //   1) 块内折叠（常量传播 + 复制传播 + 代数恒等式 + 终止器折叠）
    //   2) 死赋值消除（需要整个函数范围的"谁被读到了"信息，比逐块折叠
    //      更全局，所以放在块内折叠之后单独一趟）
    //   3) 空块消除
    //   4) 死块消除
    fn optimize_function(f: &mut MirFn) {
        // 关键修复（这一轮加上真正的 Drop 语义之后才会暴露出来的
        // 问题，见下面 fold_stmt 里的详细说明）：预先算出"哪些局部
        // 变量是 Copy 类型"，传给 fold_block/fold_stmt——复制传播
        // （copy_map）只有在来源是 Copy 类型时才能安全折叠。
        let copy_types: HashSet<usize> = f.body.locals.iter()
            .filter(|l| l.ty.is_copy())
            .map(|l| l.id)
            .collect();

        for _ in 0..MAX_OPT_ITERATIONS {
            let before_blocks = f.body.blocks.len();
            let before_stmts: usize = f.body.blocks.iter().map(|b| b.stmts.len()).sum();

            for block in &mut f.body.blocks {
                let mut const_map: HashMap<SsaLocal, Literal> = HashMap::new();
                let mut copy_map: HashMap<SsaLocal, MirOperand> = HashMap::new();
                Self::fold_block(block, &mut const_map, &mut copy_map, &copy_types);
            }

            Self::simplify_phi(&mut f.body);
            Self::eliminate_dead_stores(&mut f.body);
            Self::collapse_empty_blocks(&mut f.body);
            Self::remove_dead_blocks(&mut f.body);

            let after_blocks = f.body.blocks.len();
            let after_stmts: usize = f.body.blocks.iter().map(|b| b.stmts.len()).sum();
            if after_blocks == before_blocks && after_stmts == before_stmts {
                break;
            }
        }
    }

    // 关键重构：这里原来有一份本地的 is_copy_type，跟 borrow.rs 里的
    // 另一份各自维护——两边判断的是同一件事，标准必须完全一致，却分散
    // 在两个文件里。这不是假设性的风险：borrow.rs 那份后来补上了
    // Privacy/Tuple/Array 的递归处理（`(i32, i32)` 这种纯标量元组也该
    // 算 Copy），这里那份完全没跟上，导致复制传播在这些类型上过度
    // 保守，且没有任何编译错误提示两处已经不一致了。现在统一收进
    // ast.rs::Type::is_copy()，这里预先用它算出 `copy_types` 这张表，
    // 后面 fold_stmt 只需要查表（`copy_types.contains(...)`），不用在
    // 每个语句上重新跑一遍类型判断。

    // ===== 块内折叠：常量传播 + 复制传播 =====
    // 关键设计：const_map / copy_map 都只按 SsaLocal 记事实，而且每个
    // block 进来时都是一张新的空表，不带着上一个 block 的结论——原因
    // 分两层：
    //
    // 1）为什么只认 SsaLocal，不认 MirPlace::Local：
    //    SsaLocal 才有"整个函数里最多被赋值一次"这条静态保证（builder
    //    那边每次 let/赋值都会分配一个新的 version）。MirPlace::Local
    //    在现在的 MIR 里专门留给"会被多次写"的场景——最典型的就是
    //    If/Match 的汇合点：then_block 和 else_block 各自都会往同一个
    //    dest local 里 Assign 一次，随便挑一次的值记进按"这个 id 现在
    //    等于什么"建模的表里，会把另一条分支的事实错误地带到不该带到
    //    的地方。cond_temp/disc_temp 这类 mir_builder.rs 内部临时变量
    //    虽然实际只写一次，但类型上没有 SsaLocal 这个身份，两张表都是
    //    `HashMap<SsaLocal, _>`，也放不进去——干脆不追踪，读的时候原样
    //    保留，不算错误，只是错过一点点折叠机会。
    //
    // 2）为什么不做跨 block 的传播（每个 block 都重开一张空表）：
    //    SsaLocal 的"最多赋值一次"是静态成立的，但"在某处读到它时，
    //    它的定义一定已经执行过"这件事靠的是支配关系（dominance），
    //    不是 block id 的数值顺序。While/Loop 会产生回边（循环体块
    //    Goto 回条件块，其 id 比循环体块小），如果不先建支配树、只是
    //    按 block 列表顺序线性扫一遍网 const_map/copy_map 里塞事实，
    //    遇到回边会把只在某次循环迭代里成立的"事实"错误地套用到之前
    //    的迭代。真要做对需要先做支配树分析，这是明显更大的一块工作，
    //    这次没有被要求做，所以先按 block-local 来：传播范围小一点，
    //    但一定不会算错。
    fn fold_block(
        block: &mut MirBlock,
        const_map: &mut HashMap<SsaLocal, Literal>,
        copy_map: &mut HashMap<SsaLocal, MirOperand>,
        copy_types: &HashSet<usize>,
    ) {
        let mut new_stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            new_stmts.push(Self::fold_stmt(stmt, const_map, copy_map, copy_types));
        }
        block.stmts = new_stmts;
        Self::fold_terminator(block, const_map, copy_map);
    }

    // ===== 语句折叠：维护 const_map / copy_map =====
    // 关键修复：旧版本这里写的是 `if let MirPlace::Local(id) = dest`，
    // 但 let/赋值语句的 dest 现在几乎全是 MirPlace::Ssa（见
    // mir_builder.rs），这个分支实际上永远不会命中，const_map/copy_map
    // 两张表也就永远不会被写入——同时旧代码里 new_copy 这个变量根本
    // 没有声明过（对着一个不存在的变量 insert/get，类型也从
    // HashMap<usize, usize> 变成 HashMap<SsaLocal, Literal> 的返回值
    // 缝在一起），这个函数原本就编译不过。这次改成直接认 Ssa。
    //
    // 关键修复（复制传播为什么只吃 Copy/Move(Ssa(_))，不吃
    // Copy/Move(Local/Field/Index/...)）：
    //    `y = x` 这种赋值，如果 x 是另一个 SsaLocal，那么 x 一旦定义就
    //    不会再变，把"读 y"替换成"读 x"在任何后续位置都是安全的。但如
    //    果 x 是 `MirPlace::Field { base, .. }` 这类内存位置（结构体
    //    字段/数组元素/解引用），它可能在 y 定义之后、y 被读之前，通过
    //    另一条路径被改写（比如同一个 base 又被赋值了一次）——这时候
    //    把"读 y"替换成"重新读一次那个字段"读到的就不再是 y 定义时刻
    //    的那个值了，是错的。所以 copy_map 只记录"这个 SsaLocal 就是
    //    另一个 SsaLocal 的别名"这一种情况，其余的 Use(Copy/Move(..))
    //    原样保留、不追踪，不去冒这个险。
    //
    // 关键修复（这一轮加上真正的 Drop 语义之后才会暴露的新问题）：
    // 光"来源是不是另一个 SsaLocal"还不够——还必须是 Copy 类型才能
    // 安全折叠。举例：`y = x;`（x 不是 Copy 类型，比如是个 String，
    // mir_builder.rs 对着一个裸标识符统一走 Move），如果这里不管
    // 类型就把 copy_map[y] 记成 "= x"，后面随便一处"读 y"就会被替换
    // 成"读 x"——如果这条 `y = x;` 语句自己因为别的原因（哪怕只是
    // 因为 y 后面跟着一条 `drop(y)`，这一轮才刚加上的东西）没有被
    // eliminate_dead_stores 当成死代码删掉，最终生成的代码里 x 就会
    // 被读两次：一次是 `y = x;` 本身，一次是替换出来的那个新读取点。
    // x 是非 Copy 类型，只能被消费一次，读两次就是对同一个值
    // use-after-move，编译不过。Copy 类型读多少次都没关系（这是唯一
    // 安全的情况），所以只有 `copy_types` 里登记过的 base_id 才允许
    // 记进 copy_map。
    fn fold_stmt(
        stmt: &MirStmt,
        const_map: &mut HashMap<SsaLocal, Literal>,
        copy_map: &mut HashMap<SsaLocal, MirOperand>,
        copy_types: &HashSet<usize>,
    ) -> MirStmt {
        match stmt {
            MirStmt::Assign { dest, value } => {
                let folded_value = Self::fold_rvalue(value, const_map, copy_map);

                if let MirPlace::Ssa(ssa) = dest {
                    match &folded_value {
                        MirRvalue::Use(MirOperand::Constant(lit)) => {
                            const_map.insert(*ssa, lit.clone());
                        }
                        MirRvalue::Use(
                            op @ (MirOperand::Copy(MirPlace::Ssa(_))
                            | MirOperand::Move(MirPlace::Ssa(_))),
                        ) if copy_types.contains(&ssa.base_id) => {
                            // folded_value 是 fold_rvalue 算出来的，
                            // 内部的操作数已经在 fold_operand 里顺着
                            // const_map/copy_map 尽量解析过一轮了——
                            // 走到这里还是 Copy/Move(Ssa(src))，说明
                            // src 目前在两张表里都还没有已知的结论，
                            // 这就是"最终来源"，直接存，不需要在查表
                            // 那一侧再递归链式往下追一次。
                            copy_map.insert(*ssa, op.clone());
                        }
                        _ => {}
                    }
                }

                MirStmt::Assign {
                    dest: dest.clone(),
                    value: folded_value,
                }
            }
            MirStmt::ExprStmt(value) => {
                MirStmt::ExprStmt(Self::fold_rvalue(value, const_map, copy_map))
            }
            // Drop / SetMetadata / EffectCheck 不产出新的值，也不会
            // 让已经记录的事实失效（它们都不是"给一个 SsaLocal 重新
            // 赋值"），原样保留。
            MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {
                stmt.clone()
            }
        }
    }

    // ===== 终止器折叠：支持 If/Switch 的常量退化 =====
    // 关键修复：这个函数原来就写好了，但从没被 fold_block/optimize_function
    // 调过，是死代码。这次接进 fold_block，跟语句折叠共用同一张
    // const_map/copy_map（终止器在语句列表的最后执行，能看到本块内
    // 所有语句折叠后留下的结论）。
    fn fold_terminator(
        block: &mut MirBlock,
        const_map: &HashMap<SsaLocal, Literal>,
        copy_map: &HashMap<SsaLocal, MirOperand>,
    ) {
        match &block.terminator {
            MirTerminator::If { cond, then_block, else_block } => {
                let folded_cond = Self::fold_operand(cond, const_map, copy_map);
                match folded_cond {
                    MirOperand::Constant(Literal::Bool(true)) => {
                        block.terminator = MirTerminator::Goto(*then_block);
                    }
                    MirOperand::Constant(Literal::Bool(false)) => {
                        block.terminator = MirTerminator::Goto(*else_block);
                    }
                    _ => {
                        block.terminator = MirTerminator::If {
                            cond: folded_cond,
                            then_block: *then_block,
                            else_block: *else_block,
                        };
                    }
                }
            }
            // 关键修复：`MirTerminator::Switch` 这一轮多了 `discr_ty`
            // 字段，原来的解构漏了它，编译不过（E0027）；`targets` 也
            // 从 `Vec<(i64, usize)>` 变成了 `Vec<(Literal, usize)>`——
            // `Literal::Int(v)` 这个变体已经不存在了（拆成 Int8..Int128/
            // UInt8..UInt128/Isize/Usize），而且现在判别式不再统一转成
            // i64，折叠出来的常量可能是任何一种整数字面量、也可能是
            // Bool/Char（标量 match 现在直接拿 cond 原本的类型当判别式，
            // 不再 Cast，见 mir_builder.rs 的改动）。改成直接用
            // `Literal` 自己的 `PartialEq` 去跟 targets 里的每一项比较，
            // 不必关心具体是哪个变体。
            MirTerminator::Switch { discr, discr_ty, targets, default } => {
                let folded_discr = Self::fold_operand(discr, const_map, copy_map);
                if let MirOperand::Constant(lit) = &folded_discr {
                    let target = targets
                        .iter()
                        .find(|(val, _)| val == lit)
                        .map(|(_, t)| *t)
                        .unwrap_or(*default);
                    block.terminator = MirTerminator::Goto(target);
                } else {
                    block.terminator = MirTerminator::Switch {
                        discr: folded_discr,
                        discr_ty: discr_ty.clone(),
                        targets: targets.clone(),
                        default: *default,
                    };
                }
            }
            _ => {}
        }
    }

    // ===== 右值折叠 =====
    fn fold_rvalue(
        rv: &MirRvalue,
        const_map: &HashMap<SsaLocal, Literal>,
        copy_map: &HashMap<SsaLocal, MirOperand>,
    ) -> MirRvalue {
        match rv {
            MirRvalue::BinaryOp(op, left, right) => {
                let l = Self::fold_operand(left, const_map, copy_map);
                let r = Self::fold_operand(right, const_map, copy_map);
                if let (MirOperand::Constant(lc), MirOperand::Constant(rc)) = (&l, &r) {
                    if let Some(result) = Calc::eval_binary_op(*op, lc.clone(), rc.clone()) {
                        return MirRvalue::Use(MirOperand::Constant(result));
                    }
                }
                if let Some(simplified) = Self::apply_algebraic_identity(*op, &l, &r) {
                    return simplified;
                }
                MirRvalue::BinaryOp(*op, l, r)
            }
            MirRvalue::UnaryOp(op, operand) => {
                let folded = Self::fold_operand(operand, const_map, copy_map);
                if let MirOperand::Constant(lit) = &folded {
                    if let Some(result) = Calc::eval_unary_op(*op, lit.clone()) {
                        return MirRvalue::Use(MirOperand::Constant(result));
                    }
                }
                MirRvalue::UnaryOp(*op, folded)
            }
            MirRvalue::Use(operand) => {
                MirRvalue::Use(Self::fold_operand(operand, const_map, copy_map))
            }
            MirRvalue::Cast(operand, ty) => {
                MirRvalue::Cast(Self::fold_operand(operand, const_map, copy_map), ty.clone())
            }
            MirRvalue::Call { func, args, is_intrinsic, intrinsic_name, generic_args } => {
                MirRvalue::Call {
                    func: func.clone(),
                    args: args
                        .iter()
                        .map(|a| Self::fold_operand(a, const_map, copy_map))
                        .collect(),
                    is_intrinsic: *is_intrinsic,
                    intrinsic_name: intrinsic_name.clone(),
                    generic_args: generic_args.clone(),
                }
            }
            MirRvalue::MethodCall { receiver, method, args, generic_args } => {
                MirRvalue::MethodCall {
                    receiver: Self::fold_operand(receiver, const_map, copy_map),
                    method: method.clone(),
                    args: args
                        .iter()
                        .map(|a| Self::fold_operand(a, const_map, copy_map))
                        .collect(),
                    generic_args: generic_args.clone(),
                }
            }
            // 关键修复：StructInit/EnumVariantConstruction 后来在 mir.rs
            // 里补上了 generic_args: Vec<Type> 字段（monomorphic.rs 单
            // 态化要用），这里折叠时要原样透传——generic_args 存的是
            // 类型，不是操作数，不需要也不能拿去 fold_operand，只是
            // 单纯 clone 带过去，不然这个字段会在优化过程中丢失。
            MirRvalue::StructInit { struct_name, generic_args, fields } => {
                MirRvalue::StructInit {
                    struct_name: struct_name.clone(),
                    generic_args: generic_args.clone(),
                    fields: fields
                        .iter()
                        .map(|(name, op)| (name.clone(), Self::fold_operand(op, const_map, copy_map)))
                        .collect(),
                }
            }
            MirRvalue::EnumVariantConstruction { enum_name, generic_args, variant_name, args } => {
                MirRvalue::EnumVariantConstruction {
                    enum_name: enum_name.clone(),
                    generic_args: generic_args.clone(),
                    variant_name: variant_name.clone(),
                    args: args
                        .iter()
                        .map(|a| Self::fold_operand(a, const_map, copy_map))
                        .collect(),
                }
            }
            MirRvalue::ArrayLiteral(elements) => {
                MirRvalue::ArrayLiteral(
                    elements
                        .iter()
                        .map(|e| Self::fold_operand(e, const_map, copy_map))
                        .collect(),
                )
            }
            MirRvalue::Discriminant { value, enum_name } => {
                MirRvalue::Discriminant {
                    value: Self::fold_operand(value, const_map, copy_map),
                    enum_name: enum_name.clone(),
                }
            }
            // Ref 借用的是一个地址本身（place），"把它替换成常量"没有
            // 意义；Phi 的每个分支值要在真正做了支配树/跨块传播之后再
            // 折叠才有意义（现在 mir_builder.rs 也还没有真的生成过
            // Phi），这两个变体先原样透传。
            MirRvalue::Ref { .. } | MirRvalue::Phi { .. } => rv.clone(),
        }
    }

    // ===== 操作数折叠：先查 const_map，再查 copy_map =====
    // 关键修复：旧版本这里判断的是 `if let MirPlace::Ssa(ssa) = place`
    // 且只查 const_map——这部分的类型是对的（跟 mir_builder.rs 生成的
    // Ssa 读取对得上），但因为写入侧（旧 fold_stmt_with_map）从来没有
    // 真正往 const_map 里塞过东西（见上面的说明），这一侧其实一直在
    // 查一张空表。这次两头都改对，这个函数本身的判断逻辑不用变，只是
    // 多加一步 copy_map 的查询。
    fn fold_operand(
        op: &MirOperand,
        const_map: &HashMap<SsaLocal, Literal>,
        copy_map: &HashMap<SsaLocal, MirOperand>,
    ) -> MirOperand {
        match op {
            MirOperand::Copy(MirPlace::Ssa(ssa)) | MirOperand::Move(MirPlace::Ssa(ssa)) => {
                if let Some(lit) = const_map.get(ssa) {
                    return MirOperand::Constant(lit.clone());
                }
                if let Some(resolved) = copy_map.get(ssa) {
                    // resolved 在写入 copy_map 的时候已经是解析过的
                    // 结果（见 fold_stmt 里的说明），这里一次查表就够，
                    // 不需要再递归链式解析。
                    return resolved.clone();
                }
                op.clone()
            }
            // 非 Ssa 的 place（Field/Index/Deref/EnumPayload）不在这两张
            // 表的追踪范围内，原样返回——理由见 fold_block 顶部的说明。
            // Sym/Static 同样原样返回：它们是 mir.rs 这一轮从 MirPlace
            // 搬到 MirOperand 的两个变体，不指向任何局部变量，根本不
            // 可能出现在 const_map/copy_map 里，没有什么可折叠的。
            MirOperand::Copy(_) | MirOperand::Move(_) | MirOperand::Constant(_)
            | MirOperand::Sym(_) | MirOperand::Static(_) => op.clone(),
        }
    }

    // ===== 代数恒等式折叠 =====
    // 关键修复：这里原来在结果是常量 0/true/false 的几个分支里，直接
    // 手写死了 `Literal::Int(0)`/`Literal::Bool(...)`——`Literal::Int`
    // 这个变体已经不存在了（这一轮 ast.rs 把整数拆成了 Int8..Int128/
    // UInt8..UInt128/Isize/Usize 一堆按位宽区分的变体），编译不过。
    //
    // 更深一层的问题不只是"改成哪个变体"：这个函数本身拿不到
    // left/right 的类型信息（只有两个 MirOperand，没有 f.body.locals），
    // 没法凭空决定"这次该用哪个宽度的整数类型去表示 0"——瞎猜一个
    // （比如固定用 Int32）如果猜错了，生成的常量类型跟原来 left/right
    // 的真实类型对不上，会在更后面的阶段变成一个不好排查的类型不匹配
    // 错误。解法是压根不去"新造"一个 0：Mul 的两个分支，如果 left 或
    // right 本身就是那个值为 0 的常量，直接原样把它当结果返回——它
    // 自带正确的类型，不需要凭空构造。
    //
    // Sub 那条"left == right 时结果是 0"的恒等式（比如 `x - x`）
    // 没有这个退路——这时 left/right 都不是常量（是同一个变量读了两
    // 次），没有任何一个"自带类型的 0"可以借用，这个函数目前又没有
    // 类型上下文可以查——干脆去掉这一条，等以后真要做，需要先把
    // `&[MirLocal]`（或者一份 base_id -> Type 的表）传进这个函数才行，
    // 不在这次修复范围内。
    fn apply_algebraic_identity(
        op: BinaryOp,
        left: &MirOperand,
        right: &MirOperand,
    ) -> Option<MirRvalue> {
        use BinaryOp::*;
        match op {
            Mul => {
                if Self::is_zero(left) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if Self::is_zero(right) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if Self::is_one(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if Self::is_one(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            Add => {
                if Self::is_zero(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if Self::is_zero(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            Sub => {
                if Self::is_zero(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            And => {
                if Self::is_bool_true(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if Self::is_bool_true(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if Self::is_bool_false(left) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if Self::is_bool_false(right) {
                    return Some(MirRvalue::Use(right.clone()));
                }
            }
            Or => {
                if Self::is_bool_true(left) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if Self::is_bool_true(right) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if Self::is_bool_false(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if Self::is_bool_false(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            Eq | Neq => {
                if left == right {
                    let result = match op {
                        Eq => true,
                        Neq => false,
                        _ => unreachable!(),
                    };
                    return Some(MirRvalue::Use(MirOperand::Constant(Literal::Bool(result))));
                }
            }
            _ => {}
        }
        None
    }

    // ---- 辅助判断函数 ----
    // 关键修复：原来只认 `Literal::Int(0)`/`Literal::Int(1)`，这个变体
    // 已经不存在了；而且就算改名，也只够判断"一种"整数类型的 0/1——
    // 现在整数按位宽/符号拆成了十二个变体，随便一个字面量是 `Int8(0)`
    // 还是 `UInt64(0)`，都得算作"是零"，不能只挑其中一个变体判断，
    // 不然大多数情况下这条优化直接就不生效了。
    //
    // 关键重构：项目里不用宏（包括 matches!，好几处都专门改回过显式
    // match），is_zero/is_one 原来各写了一份 `matches!(op,
    // MirOperand::Constant(Literal::IntN(0) | ... ))`——按位宽/符号列了
    // 十二个变体，是零判断和是一判断这两份几乎一模一样，唯一的区别是
    // 每个变体里包的数字是 0 还是 1。真正该抽出来复用的不是"判断是不是
    // 零/一"本身，而是"不管来源是哪个整数字面量变体，把里面的数值统一
    // 取出来"这一步——取出来之后，是零是一只是跟 0/1 比大小，不需要
    // 再重复列一遍十二个变体。
    fn int_literal_value(op: &MirOperand) -> Option<i128> {
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
                Literal::UInt128(v) => Some(*v as i128),
                Literal::Isize(v) => Some(*v as i128),
                Literal::Usize(v) => Some(*v as i128),
                _ => None,
            },
            _ => None,
        }
    }
    fn is_zero(op: &MirOperand) -> bool {
        match Self::int_literal_value(op) {
            Some(v) => v == 0,
            None => false,
        }
    }
    fn is_one(op: &MirOperand) -> bool {
        match Self::int_literal_value(op) {
            Some(v) => v == 1,
            None => false,
        }
    }
    // 同样的道理：is_bool_true/is_bool_false 共用"取出 bool 字面量的值"
    // 这一步。
    fn bool_literal_value(op: &MirOperand) -> Option<bool> {
        match op {
            MirOperand::Constant(Literal::Bool(v)) => Some(*v),
            _ => None,
        }
    }
    fn is_bool_true(op: &MirOperand) -> bool {
        match Self::bool_literal_value(op) {
            Some(v) => v,
            None => false,
        }
    }
    fn is_bool_false(op: &MirOperand) -> bool {
        match Self::bool_literal_value(op) {
            Some(v) => !v,
            None => false,
        }
    }

    // ===== 空块消除（不涉及 Ssa/Local 的读写事实，跟 SSA 改造本身无关；
    // 下面第二个循环里 If/Switch 分支的写法后来发现有借用检查过不去的
    // 问题，已经修——见那两个分支上面的说明） =====
    fn collapse_empty_blocks(body: &mut MirBody) {
        if body.blocks.len() <= 1 {
            return;
        }

        let block_count = body.blocks.len();
        let mut replaced_targets = HashMap::new();
        for (block_id, block) in body.blocks.iter().enumerate() {
            // 关键重构：原来先用 matches! 问一遍"终止器是不是 Goto"，
            // 再紧接着用 if let 把里面的目标块 id 解出来——同一件事问了
            // 两遍。项目里不用 matches! 宏，这里也没必要真的用它：直接
            // 上 if let，配合 else 分支 continue 跳过不满足的情况，一步
            // 到位。
            if block_id == 0 || !block.stmts.is_empty() {
                continue;
            }
            let target = match &block.terminator {
                MirTerminator::Goto(t) => *t,
                _ => continue,
            };
            if target == block_id {
                continue;
            }
            replaced_targets.insert(block_id, target);
        }

        if replaced_targets.is_empty() {
            return;
        }

        for block in &mut body.blocks {
            match &mut block.terminator {
                MirTerminator::Goto(target) => {
                    while let Some(&new_target) = replaced_targets.get(target) {
                        *target = new_target;
                    }
                }
                MirTerminator::If { then_block, else_block, .. } => {
                    while let Some(&new_target) = replaced_targets.get(then_block) {
                        *then_block = new_target;
                    }
                    while let Some(&new_target) = replaced_targets.get(else_block) {
                        *else_block = new_target;
                    }
                }
                MirTerminator::Switch { targets, default, .. } => {
                    while let Some(&new_target) = replaced_targets.get(default) {
                        *default = new_target;
                    }
                    for (_, target) in targets.iter_mut() {
                        while let Some(&new_target) = replaced_targets.get(target) {
                            *target = new_target;
                        }
                    }
                }
                _ => {}
            }
        }

        let mut new_blocks = Vec::new();
        let mut id_map: Vec<Option<usize>> = vec![None; block_count];
        for (old_id, block) in body.blocks.iter().enumerate() {
            if old_id == 0 || !replaced_targets.contains_key(&old_id) {
                let new_id = new_blocks.len();
                id_map[old_id] = Some(new_id);
                new_blocks.push(block.clone());
            }
        }

        // 关键修复：If/Switch 这两个分支原来是"边读 then_block/
        // else_block/targets/default 这些从 `&mut block.terminator`
        // 借出来的引用，边在某个分支里直接 `block.terminator = ...`
        // 整个改写"——一旦某一次检查失败要退化成 Unreachable，紧接着
        // 后面还要用同一个 match 借出来的另一个字段（比如 If 里先判
        // then_block 不行就重写一次 block.terminator，然后马上还要读
        // else_block；Switch 里先判 default，再在 for 循环里继续用
        // targets 的可变借用），编译器认为这些借用在那次写之后仍然
        // "之后还会被用到"，于是把写和后续读之间判定成冲突
        // （E0506，你贴的三个报错都是这个）。
        //
        // 这跟 remove_dead_blocks 里 Switch 分支之前的老问题是同一个
        // 根源，那边已经改成"先把要用到的值都读成局部变量、拼出一个
        // 完整的新 terminator，最后一次性整体赋值"，这里用同样的手法：
        // 不在中途改字段，只在最后 `block.terminator = new_term;`
        // 赋值一次，这时候前面那些借用都已经用完了，不再冲突。
        //
        // 另外说明一下为什么下面 If/Switch 分支里的"解析不到就退化成
        // Unreachable"这条兜底路径理论上不会真的走到：上面第一趟循环
        // （453 行开始那个）已经把所有 then_block/else_block/targets/
        // default 顺着 replaced_targets 重定向到不会被折叠掉的块了，
        // 走到这里 id_map 查出来应该总是 Some。保留这条兜底只是防御性
        // 的——万一这个前提以后被破坏，宁可安全地退化成 Unreachable，
        // 也不要带着一个不存在的块 id 继续跑下去。
        for block in &mut new_blocks {
            match &mut block.terminator {
                MirTerminator::Goto(target) => {
                    if let Some(new_target) = id_map[*target] {
                        *target = new_target;
                    } else {
                        block.terminator = MirTerminator::Unreachable;
                    }
                }
                MirTerminator::If { cond, then_block, else_block } => {
                    let new_then = id_map[*then_block];
                    let new_else = id_map[*else_block];
                    block.terminator = match (new_then, new_else) {
                        (Some(nt), Some(ne)) => MirTerminator::If {
                            cond: cond.clone(),
                            then_block: nt,
                            else_block: ne,
                        },
                        _ => MirTerminator::Unreachable,
                    };
                }
                MirTerminator::Switch { discr, discr_ty, targets, default } => {
                    let new_default = id_map[*default];
                    let mut new_targets = Vec::with_capacity(targets.len());
                    let mut all_resolved = true;
                    for (val, target) in targets.iter() {
                        match id_map[*target] {
                            // 关键修复：`*val` 要求 Literal: Copy，不再
                            // 成立（见前面的说明），改成 `.clone()`。
                            Some(nt) => new_targets.push((val.clone(), nt)),
                            None => all_resolved = false,
                        }
                    }
                    block.terminator = match (new_default, all_resolved) {
                        (Some(nd), true) => MirTerminator::Switch {
                            discr: discr.clone(),
                            // 关键修复：漏了 discr_ty 字段，编译不过。
                            discr_ty: discr_ty.clone(),
                            targets: new_targets,
                            default: nd,
                        },
                        _ => MirTerminator::Unreachable,
                    };
                }
                _ => {}
            }
        }

        body.blocks = new_blocks;
    }

    // ===== 死赋值消除 =====
    // 关键修复：旧版本的"谁被用到了"只认 `MirPlace::Local(id)`（无论
    // 是收集用到的变量、还是判断 dest 能不能删），但现在的赋值语句
    // dest 几乎全是 `MirPlace::Ssa`——两边类型对不上，导致这个函数
    // 就算被调用也等于是瞎判断（永远收集不到任何"被用到"的 Ssa，
    // 于是所有 Ssa 赋值看起来都"从未被读取"，真调了就是一场大屠杀）。
    // 这也是它之前没有被接入主循环的可能原因。这次把"用到的变量"拆成
    // used_ssa / used_local 两个集合，读写两侧统一按这两个集合走。
    fn eliminate_dead_stores(body: &mut MirBody) {
        let mut used_ssa: HashSet<SsaLocal> = HashSet::new();
        // 关键说明（找回被之前那处删掉的解释）：MirPlace 现在只有
        // Ssa/Field/Index/Deref/EnumPayload 五种，全部递归到底都落在
        // Ssa 上——没有任何 MirPlace 变体会往 used_local 里塞东西（见
        // 下面 collect_used_vars_in_place 的实现），这张表因此永远是
        // 空的。保留它的类型/参数没有删，是为了不去动
        // eliminate_dead_stores 整条调用链的函数签名（影响面更大），
        // 只做编译能过所必需的最小改动；真要彻底清理，需要把
        // used_local 从这一整条调用链里都摘掉，这次先不做。
        let mut used_local: HashSet<usize> = HashSet::new();

        for block in &body.blocks {
            for stmt in &block.stmts {
                Self::collect_used_vars(stmt, &mut used_ssa, &mut used_local);
            }
            Self::collect_used_vars_in_terminator(&block.terminator, &mut used_ssa, &mut used_local);
        }

        for block in &mut body.blocks {
            block.stmts.retain(|stmt| {
                if let MirStmt::Assign { dest, value } = stmt {
                    let is_dead = match dest {
                        MirPlace::Ssa(ssa) => !used_ssa.contains(ssa),
                        // 关键修复：`MirPlace::Local` 这个变体已经不
                        // 存在了（"SSA 完整版，Local 被废弃"），原来这
                        // 一支对着一个已删除的枚举变体匹配，编译不过。
                        // 现在所有赋值目标要么是 Ssa，要么是下面这几种
                        // 复合 place，不会再有裸的 Local 出现，直接删掉
                        // 这一支即可，不需要另找地方安放。
                        // Field/Index/Deref/EnumPayload：写的是
                        // 内存里某个更复杂的位置（结构体字段、数组
                        // 元素……），这个简化版分析不追踪"这个具体位置
                        // 有没有被读到"（可能被另一条路径别名读取），
                        // 一律保留，不删。
                        _ => false,
                    };
                    if is_dead && !Self::has_side_effect(value) {
                        return false;
                    }
                }
                true
            });
        }
    }

    fn collect_used_vars(
        stmt: &MirStmt,
        used_ssa: &mut HashSet<SsaLocal>,
        used_local: &mut HashSet<usize>,
    ) {
        match stmt {
            MirStmt::Assign { dest, value } => {
                Self::collect_used_vars_in_rvalue(value, used_ssa, used_local);
                // dest 本身是"定值"，不算使用；但如果 dest 是复合
                // place（写进某个 base 的字段/元素），先得有这个
                // base，那个 base 是被读到的，要算作使用——不然"这个
                // base 是哪条赋值产生的"会被当成死赋值误删。
                match dest {
                    // 关键修复：Local 已经不存在了，Ssa 不需要往下递归——
                    // Sym/Static 现在也不可能出现在这里了（它们从
                    // MirPlace 挪去了 MirOperand，见 mir.rs 里那条搬家
                    // 说明），MirPlace 只剩 Ssa/Field/Index/Deref/
                    // EnumPayload 五种，这个 match 天然穷尽。
                    MirPlace::Ssa(_) => {}
                    MirPlace::Field { base, .. }
                    | MirPlace::Deref(base)
                    | MirPlace::EnumPayload { base, .. } => {
                        Self::collect_used_vars_in_place(base, used_ssa, used_local);
                    }
                    MirPlace::Index { base, index } => {
                        Self::collect_used_vars_in_place(base, used_ssa, used_local);
                        Self::collect_used_vars_in_operand(index, used_ssa, used_local);
                    }
                }
            }
            MirStmt::ExprStmt(value) => {
                Self::collect_used_vars_in_rvalue(value, used_ssa, used_local);
            }
            MirStmt::Drop { place } => {
                Self::collect_used_vars_in_place(place, used_ssa, used_local);
            }
            MirStmt::SetMetadata { place, .. } => {
                Self::collect_used_vars_in_place(place, used_ssa, used_local);
            }
            MirStmt::EffectCheck { .. } => {}
        }
    }

    fn collect_used_vars_in_rvalue(
        rv: &MirRvalue,
        used_ssa: &mut HashSet<SsaLocal>,
        used_local: &mut HashSet<usize>,
    ) {
        match rv {
            MirRvalue::Use(op) => Self::collect_used_vars_in_operand(op, used_ssa, used_local),
            MirRvalue::BinaryOp(_, left, right) => {
                Self::collect_used_vars_in_operand(left, used_ssa, used_local);
                Self::collect_used_vars_in_operand(right, used_ssa, used_local);
            }
            MirRvalue::UnaryOp(_, operand) => {
                Self::collect_used_vars_in_operand(operand, used_ssa, used_local)
            }
            MirRvalue::Cast(operand, _) => {
                Self::collect_used_vars_in_operand(operand, used_ssa, used_local)
            }
            MirRvalue::Call { args, .. } => {
                for arg in args {
                    Self::collect_used_vars_in_operand(arg, used_ssa, used_local);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::collect_used_vars_in_operand(receiver, used_ssa, used_local);
                for arg in args {
                    Self::collect_used_vars_in_operand(arg, used_ssa, used_local);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::collect_used_vars_in_operand(op, used_ssa, used_local);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for arg in args {
                    Self::collect_used_vars_in_operand(arg, used_ssa, used_local);
                }
            }
            MirRvalue::ArrayLiteral(elements) => {
                for elem in elements {
                    Self::collect_used_vars_in_operand(elem, used_ssa, used_local);
                }
            }
            MirRvalue::Discriminant { value, .. } => {
                Self::collect_used_vars_in_operand(value, used_ssa, used_local)
            }
            MirRvalue::Ref { place, .. } => {
                Self::collect_used_vars_in_place(place, used_ssa, used_local)
            }
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::collect_used_vars_in_operand(op, used_ssa, used_local);
                }
            }
        }
    }

    fn collect_used_vars_in_operand(
        op: &MirOperand,
        used_ssa: &mut HashSet<SsaLocal>,
        used_local: &mut HashSet<usize>,
    ) {
        match op {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                Self::collect_used_vars_in_place(place, used_ssa, used_local);
            }
            MirOperand::Constant(_) => {}
            // 关键修复（Sym/Static 搬家）：这两个变体从 MirPlace 挪来
            // MirOperand 之后，不指向任何局部变量，不产生 used_ssa/
            // used_local 记录，跟 Constant 是同一个道理。
            MirOperand::Sym(_) => {}
            MirOperand::Static(_) => {}
        }
    }

    // 关键修复：旧版本没有这个函数——它只在乎 MirOperand 直接包着的
    // MirPlace 是不是 Local，压根没有处理 MirPlace::Ssa，也没有往
    // Field/Index/Deref/EnumPayload 的 base 里递归。这是导致
    // eliminate_dead_stores 之前完全不可用的核心原因之一：现在几乎
    // 所有读取都是 Copy/Move(Ssa(..))，一个只认 Local 的收集函数
    // 等于什么都收集不到。
    //
    // 关键修复（Sym/Static 搬家）：MirPlace::Static 已经不存在了——
    // mir.rs 把它挪去了 MirOperand（见那边的说明），MirPlace 现在只剩
    // Ssa/Field/Index/Deref/EnumPayload 五种，这个 match 天然穷尽，
    // 不再需要给 Static 单写一条"什么都不做"的分支。
    fn collect_used_vars_in_place(
        place: &MirPlace,
        used_ssa: &mut HashSet<SsaLocal>,
        used_local: &mut HashSet<usize>,
    ) {
        match place {
            MirPlace::Ssa(ssa) => {
                used_ssa.insert(*ssa);
            }
            MirPlace::Field { base, .. } => {
                Self::collect_used_vars_in_place(base, used_ssa, used_local);
            }
            MirPlace::Index { base, index } => {
                Self::collect_used_vars_in_place(base, used_ssa, used_local);
                Self::collect_used_vars_in_operand(index, used_ssa, used_local);
            }
            MirPlace::Deref(base) => {
                Self::collect_used_vars_in_place(base, used_ssa, used_local);
            }
            MirPlace::EnumPayload { base, .. } => {
                Self::collect_used_vars_in_place(base, used_ssa, used_local);
            }
        }
    }

    fn collect_used_vars_in_terminator(
        term: &MirTerminator,
        used_ssa: &mut HashSet<SsaLocal>,
        used_local: &mut HashSet<usize>,
    ) {
        match term {
            MirTerminator::If { cond, .. } => {
                Self::collect_used_vars_in_operand(cond, used_ssa, used_local)
            }
            MirTerminator::Switch { discr, .. } => {
                Self::collect_used_vars_in_operand(discr, used_ssa, used_local)
            }
            MirTerminator::Return(Some(op)) => {
                Self::collect_used_vars_in_operand(op, used_ssa, used_local)
            }
            // 关键修复（Placeholder/Unreachable 拆分）：mir.rs 新增的
            // Placeholder 没有任何操作数（它就表示"这个块的终止器还没
            // 设置"），跟 Return(None)/Goto/Unreachable 落进同一支，
            // 不产生任何"用到了谁"的记录。
            MirTerminator::Return(None) | MirTerminator::Goto(_) | MirTerminator::Unreachable
            | MirTerminator::Placeholder => {}
        }
    }

    // 判断右值是否有副作用。
    // 关键修复：旧版本是 `Call { is_intrinsic, .. } => *is_intrinsic`
    // ——这等于说"只有内建调用才可能有副作用，普通函数调用一律视为
    // 无副作用"，跟它自己的注释"只有 Call 可能是副作用，保守起见
    // 返回 true"正好反着。MIR 里的 Call 节点不带调用目标函数的
    // EffectSet（那份信息现在只挂在 HirFn/MirFn 级别，没有下沉到
    // 每个调用点），没有办法从这里判断一个非内建调用到底纯不纯——
    // 保守起见，只要是 Call 或 MethodCall 就一律当作有副作用，不去
    // 猜（这样死赋值消除最多是少删几条本来能删的语句，不会删掉真正
    // 有副作用、不能删的调用）。
    fn has_side_effect(rv: &MirRvalue) -> bool {
        match rv {
            MirRvalue::Call { .. } | MirRvalue::MethodCall { .. } => true,
            _ => false,
        }
    }

    // ===== 死块消除 =====
    fn remove_dead_blocks(body: &mut MirBody) {
        let block_count = body.blocks.len();
        if block_count == 0 {
            return;
        }

        // 1. 标记可达块
        let reachable = Self::compute_reachable(body);
        if reachable.is_empty() || reachable.iter().all(|&r| r) {
            return;
        }

        // 2. 构建 old_id -> new_id 映射
        let mut id_map: Vec<Option<usize>> = vec![None; block_count];
        let mut new_blocks = Vec::new();
        for (old_id, &reached) in reachable.iter().enumerate() {
            if reached {
                let new_id = new_blocks.len();
                id_map[old_id] = Some(new_id);
                new_blocks.push(body.blocks[old_id].clone());
            }
        }

        // 3. 更新所有保留块的跳转目标
        for block in &mut new_blocks {
            let new_terminator = match &block.terminator {
                MirTerminator::Goto(target) => {
                    if let Some(new_target) = id_map[*target] {
                        MirTerminator::Goto(new_target)
                    } else {
                        MirTerminator::Unreachable
                    }
                }
                MirTerminator::If { cond, then_block, else_block } => {
                    let new_then = id_map[*then_block];
                    let new_else = id_map[*else_block];
                    match (new_then, new_else) {
                        (Some(nt), Some(ne)) => MirTerminator::If {
                            cond: cond.clone(),
                            then_block: nt,
                            else_block: ne,
                        },
                        (Some(nt), None) => MirTerminator::Goto(nt),
                        (None, Some(ne)) => MirTerminator::Goto(ne),
                        (None, None) => MirTerminator::Unreachable,
                    }
                }
                // 关键修复：同前面几处一样，漏了 discr_ty 字段，且
                // `*val`/`new_targets[0]`按值用要求 Literal: Copy，
                // 不成立，改用 `.clone()`。
                MirTerminator::Switch { discr, discr_ty, targets, default } => {
                    let new_default = id_map[*default];
                    let mut new_targets = Vec::new();
                    for (val, target) in targets {
                        if let Some(new_target) = id_map[*target] {
                            new_targets.push((val.clone(), new_target));
                        }
                    }
                    if new_targets.is_empty() {
                        if let Some(nd) = new_default {
                            MirTerminator::Goto(nd)
                        } else {
                            MirTerminator::Unreachable
                        }
                    } else {
                        if let Some(nd) = new_default {
                            MirTerminator::Switch {
                                discr: discr.clone(),
                                discr_ty: discr_ty.clone(),
                                targets: new_targets,
                                default: nd,
                            }
                        } else {
                            let fallback_default = new_targets[0].1;
                            MirTerminator::Switch {
                                discr: discr.clone(),
                                discr_ty: discr_ty.clone(),
                                targets: new_targets.clone(),
                                default: fallback_default,
                            }
                        }
                    }
                }
                _ => block.terminator.clone(),
            };
            block.terminator = new_terminator;
        }

        body.blocks = new_blocks;
    }

    // ===== 计算可达块 =====
    fn compute_reachable(body: &MirBody) -> Vec<bool> {
        let block_count = body.blocks.len();
        if block_count == 0 {
            return Vec::new();
        }
        let mut reachable = vec![false; block_count];
        let mut stack = vec![0];
        reachable[0] = true;

        while let Some(id) = stack.pop() {
            let block = &body.blocks[id];
            match &block.terminator {
                MirTerminator::Goto(target) => {
                    if !reachable[*target] {
                        reachable[*target] = true;
                        stack.push(*target);
                    }
                }
                MirTerminator::If { then_block, else_block, .. } => {
                    if !reachable[*then_block] {
                        reachable[*then_block] = true;
                        stack.push(*then_block);
                    }
                    if !reachable[*else_block] {
                        reachable[*else_block] = true;
                        stack.push(*else_block);
                    }
                }
                MirTerminator::Switch { targets, default, .. } => {
                    if !reachable[*default] {
                        reachable[*default] = true;
                        stack.push(*default);
                    }
                    for (_, target) in targets {
                        if !reachable[*target] {
                            reachable[*target] = true;
                            stack.push(*target);
                        }
                    }
                }
                _ => {}
            }
        }
        reachable
    }

    // ===== Phi 简化 =====
    fn simplify_phi(body: &mut MirBody) {
        let reachable = Self::compute_reachable(body);
        let block_count = body.blocks.len();
        // 借用检查修复：下面要 `for block in &mut body.blocks`，对
        // body.blocks 持有一个贯穿整个循环体的可变借用；而循环体内部
        // 原来又用 `body.blocks.get(*pred)` 去查前驱块的 terminator，
        // 这是对同一个字段的不可变借用，两者同时存在触发 E0502
        // （跟 mir_builder.rs 里那次是同一类问题：迭代器/引用的生命周期
        // 覆盖了整个循环体，循环体里就不能再借用被迭代的容器本身）。
        // 这里要的信息其实很简单——每个块的 terminator 是不是
        // Unreachable——跟 compute_reachable 一样，提前算成一份不依赖
        // body 的独立数组，就可以在可变借用 body.blocks 的循环里安全
        // 使用，不用再回头查 body.blocks。
        let is_unreachable_terminator: Vec<bool> = body
            .blocks
            .iter()
            .map(|b| match b.terminator {
                MirTerminator::Unreachable => true,
                _ => false,
            })
            .collect();

        for block in &mut body.blocks {
            let mut new_stmts = Vec::with_capacity(block.stmts.len());
            for stmt in &block.stmts {
                let simplified = match stmt {
                    MirStmt::Assign { dest, value: MirRvalue::Phi { values } } => {
                        // 过滤掉不可达前驱或已终止于 Unreachable 的前驱
                        let mut live_values: Vec<(usize, &MirOperand)> = Vec::new();
                        for (pred, op) in values.iter() {
                            if *pred >= block_count {
                                continue;
                            }
                            if reachable[*pred] {
                                // 用预先算好的 is_unreachable_terminator 代替
                                // body.blocks.get(*pred)，避免在 &mut body.blocks
                                // 的循环体里再借用 body.blocks 本身。
                                if !is_unreachable_terminator[*pred] {
                                    live_values.push((*pred, op));
                                }
                            }
                        }

                        if live_values.is_empty() {
                            // 所有前驱都不可达或发散，此 Phi 实际不会被执行
                            MirStmt::Assign {
                                dest: dest.clone(),
                                value: MirRvalue::Phi { values: values.clone() },
                            }
                        } else if live_values.len() == 1 {
                            // 只有一个有效值，退化为 Use
                            let (_, op) = live_values[0];
                            MirStmt::Assign {
                                dest: dest.clone(),
                                value: MirRvalue::Use(op.clone()),
                            }
                        } else {
                            // 检查是否所有有效值都相同
                            let first_op = live_values[0].1;
                            let all_same = live_values.iter().all(|(_, op)| *op == first_op);
                            if all_same {
                                MirStmt::Assign {
                                    dest: dest.clone(),
                                    value: MirRvalue::Use(first_op.clone()),
                                }
                            } else {
                                // 保留 Phi，但压缩 values 列表
                                let new_values: Vec<(usize, MirOperand)> = live_values
                                    .into_iter()
                                    .map(|(block, op)| (block, op.clone()))
                                    .collect();
                                MirStmt::Assign {
                                    dest: dest.clone(),
                                    value: MirRvalue::Phi { values: new_values },
                                }
                            }
                        }
                    }
                    _ => stmt.clone(),
                };
                new_stmts.push(simplified);
            }
            block.stmts = new_stmts;
        }
    }
}
