// state.rs
//
// 从 mir_builder.rs 拆出来的一部分：MirBuilder 自己的构建期状态管理——
// 局部变量表怎么增删、SSA 版本号怎么发、作用域怎么进出、基本块/语句/
// 终结指令怎么写。这些函数不关心"正在构建的是什么表达式"，只关心
// "MirBuilder 手上这几张表该怎么维护"，跟 build_expr/build_stmt 那种
// "HIR 节点 -> MIR 节点"的转换逻辑是不同层次的关注点，分开之后
// mir_builder.rs 里剩下的才是真正的"翻译"代码。

use crate::ast::Type;
// 关键修复：`crate::hir::Literal` 是私有的枚举导入（E0603）——hir.rs
// 内部大概率是用不带 pub 的 `use crate::ast::Literal;` 引进来自用的，
// 只在 hir.rs 自己模块内部可见，不构成 hir 对外公开的路径，state.rs
// 作为平级模块按名字导入会被挡。回头看，mir_builder.rs 原来的代码能
// 用裸 `Literal::Int64(...)` 这类写法，靠的其实是 `use crate::mir::*;`
// ——mir.rs 是 `pub use crate::ast::{Type, ..., Literal};`，真正公开
// 转发了这个类型，不是 hir.rs 那边的功劳。这里干脆直接从定义它的
// ast.rs 导入，不依赖某个中间模块顺手公开转发了它这件事。
use crate::ast::Literal;
use crate::mir::*;
use crate::mir_builder::{LoopCtx, LoopPhiInfo, MirBuilder};
use std::collections::{BTreeSet, HashMap};

impl MirBuilder {
    // -------- 局部变量 --------
    pub(crate) fn new_local(&mut self, name: Option<String>, ty: Type, mutable: bool, add_to_scope: bool) -> usize {
        let id = self.locals.len();
        self.locals.push(MirLocal { id, name, ty, mutable, persist: false, is_param: false });
        if add_to_scope {
            self.scope_vars.last_mut().unwrap().push(id);
        }
        id
    }

    // 关键新增：专门给函数参数用——is_param: true，codegen.rs 靠这个
    // 字段知道"这个 local 不用重新 let 声明，Rust 函数签名里已经有
    // 同名的绑定了"。参数永远不是 persist（persist 是给 model 块里
    // `persist let`/`persist var` 用的，跟参数是两回事），也不需要
    // mutable（函数体内要重新赋值的话，语言层面应该是 `let mut x = 参数`
    // 这种显式重绑定，走的是普通 new_local，不是这里）。
    pub(crate) fn new_param_local(&mut self, name: String, ty: Type) -> usize {
        let id = self.locals.len();
        self.locals.push(MirLocal { id, name: Some(name), ty, mutable: false, persist: false, is_param: true });
        id
    }

    pub(crate) fn new_persist_local(&mut self, name: Option<String>, ty: Type, mutable: bool) -> usize {
        let id = self.locals.len();
        self.locals.push(MirLocal { id, name, ty, mutable, persist: true, is_param: false });
        self.scope_vars.last_mut().unwrap().push(id);
        id
    }

    // 关键修复：SSA 版本号的分配只有这一个出口。原来 ssa_versions 同时
    // 当"当前读到哪个版本"和"下一个版本号是多少"两张表用，分支处理里
    // 这两个职责是冲突的：else 分支的读取要回到分支之前的版本（需要把
    // ssa_versions 回滚），但 else 分支里新赋的值绝不能拿到 then 分支
    // 已经用过的版本号（不能回滚计数），否则两条分支各造一个同名的
    // x_6，"每个 SSA 值只定义一次"就破了，后面 SCCP/GVN 拿同名值当同一个
    // 值处理，得出的是错误结论。所以拆成两张表：ssa_versions 管"当前
    // 版本"，可以随分支保存/恢复；ssa_next 管"已分配的最大版本"，只增
    // 不减，任何路径都不回滚它。
    //
    // 调用方不许再手写 `get + 1 + insert` 自己算版本号——那样绕过
    // ssa_next，回滚之后就会重复发号。
    pub(crate) fn new_version(&mut self, base_id: usize) -> u32 {
        let next = self.ssa_next.entry(base_id).or_insert(0);
        *next += 1;
        let ver = *next;
        self.ssa_versions.insert(base_id, ver);
        ver
    }

    pub(crate) fn new_temp(&mut self, ty: Type) -> usize {
        self.new_local(None, ty, false, true)
    }

    pub(crate) fn current_ssa(&self, base_id: usize) -> SsaLocal {
        let version = self.ssa_versions.get(&base_id).copied().unwrap_or(0);
        SsaLocal { base_id, version }
    }

    // ===== 分支汇合：版本快照的恢复与 Phi =====
    //
    // 关键设计：if/match 的每条分支都必须从"分支之前"的变量版本出发，
    // 互不看见对方的赋值；汇合点再把"各条分支结束时读到的版本"用 Phi
    // 合成一个。restore_versions 负责前一半，emit_join_phis 负责后一半。

    // 关键修复：只回滚 id < locals_limit 的那部分（分支之前就存在的
    // 变量），id >= locals_limit 的条目（分支内部才创建的 local、以及
    // 汇合用的 dest_base 之类临时变量）原样保留，不能整张表替换成快照。
    // 原因：这些临时变量可能登记在分支外层的作用域里（比如 `else if`
    // 链里嵌套的那层 if 没有自己的 Block，它的临时变量就挂在外层
    // scope_vars 上），作用域结束时 emit_drops_for_scopes 用
    // current_ssa 去查它们的版本、再去 moved 里比对——版本被整张表
    // 替换抹回 0，查出来的是一个从没被赋值过的 t_0，比对不上 moved，
    // 就会给它生成一条 Drop，对一个从未初始化的变量调用 drop()。
    pub(crate) fn restore_versions(&mut self, snapshot: &HashMap<usize, u32>, locals_limit: usize) {
        self.ssa_versions.retain(|&id, _| id >= locals_limit);
        for (&id, &ver) in snapshot {
            if id < locals_limit {
                self.ssa_versions.insert(id, ver);
            }
        }
    }

    // 在当前块（调用方已经 switch 到汇合块）里，为"各条存活路径上版本
    // 不一致"的既有变量插 Phi，并把每个变量的当前版本设成汇合之后该
    // 读的那个：
    //   - 所有路径版本一致（含只剩一条存活路径）：不需要 Phi，直接沿用
    //     那个版本。只剩一条路径时汇合块只有一个前驱，这条路径上的
    //     定义天然支配汇合块，插单路 Phi 只会增加噪声。
    //   - 版本不一致：new_version 出一个新版本，Phi 的 incoming 按 arms
    //     的顺序逐条列出（前驱块 id, 该路径上的版本）。
    // arms 只包含"真的会走到汇合块"的路径（发散的分支不在里面）；
    // 每一项的版本表必须是那条路径结束时 ssa_versions 的完整拷贝。
    //
    // 先无条件把 id < locals_limit 的版本回滚到 base（见 restore_versions
    // 的说明）：发散分支留在 ssa_versions 里的版本，定义只存在于一条
    // 走不到汇合块的路径上，汇合之后绝不能被读到。
    //
    // 遍历用 BTreeSet 保证按 id 升序——HashSet 的迭代顺序每次运行都
    // 不同，生成的 MIR 就会不确定，diff/调试/快照测试全都没法做。
    pub(crate) fn emit_join_phis(
        &mut self,
        base: &HashMap<usize, u32>,
        locals_limit: usize,
        arms: &[(usize, HashMap<usize, u32>)],
    ) {
        self.restore_versions(base, locals_limit);
        if arms.is_empty() {
            return;
        }

        let mut ids: BTreeSet<usize> = BTreeSet::new();
        for &id in base.keys() {
            if id < locals_limit {
                ids.insert(id);
            }
        }
        for (_, versions) in arms {
            for &id in versions.keys() {
                if id < locals_limit {
                    ids.insert(id);
                }
            }
        }

        for id in ids {
            // arms 里的版本表是完整拷贝，没有条目就是从未被赋值过（版本 0）。
            let per_arm: Vec<u32> = arms
                .iter()
                .map(|(_, versions)| versions.get(&id).copied().unwrap_or(0))
                .collect();
            let first = per_arm[0];
            if per_arm.iter().all(|&v| v == first) {
                self.ssa_versions.insert(id, first);
                continue;
            }

            let phi_ver = self.new_version(id);
            let phi_ssa = SsaLocal { base_id: id, version: phi_ver };
            let mut values = Vec::with_capacity(arms.len());
            let mut any_moved = false;
            for ((pred_block, _), &ver) in arms.iter().zip(per_arm.iter()) {
                let incoming = SsaLocal { base_id: id, version: ver };
                if self.moved.contains(&incoming) {
                    any_moved = true;
                }
                values.push((*pred_block, MirOperand::Move(MirPlace::Ssa(incoming))));
            }
            // moved 是按具体版本登记的，新造的 Phi 版本天生不在里面。只要有
            // 一条路径上这个变量已经被移走，汇合之后它就是"可能已被移走"，
            // 作用域结束时不能再对它显式 Drop——跟 moved 字段一直以来的取舍
            // 一致：宁可少 Drop，也不能对可能已移动的值调用 drop()。
            if any_moved {
                self.moved.insert(phi_ssa);
            }
            self.push_stmt(MirStmt::Assign {
                dest: MirPlace::Ssa(phi_ssa),
                value: MirRvalue::Phi { values },
            });
        }
    }

    // 关键修复：FieldAccess/Index 读一个字段/下标（`let x = p.x;`/
    // `let x = arr[i];`）之前完全没往 moved 里登记任何东西——build_place
    // 递归拆出来的最终 MirPlace 是 Field{base:...}/Index{base:...}，
    // 只存了"读的是哪个字段/下标"，从没往上找过"这次操作到底 touch 的
    // 是哪个变量"，导致 pop_scope 在 p 离开作用域时照常给它插一条
    // Drop——对一个已经被（哪怕只是部分）移动过的值调用 drop()，生成
    // 的 Rust 编译不过。这个 helper 就是补上"往上找到底"这一步：Field/
    // Index/Deref/EnumPayload 都是"在某个 place 上面套一层投影"，顺着
    // base 一路往下找，找到 Ssa 就是这次操作真正touch 到的那个变量。
    // 关键重构（Sym/Static 搬家）：mir.rs 把 Static（内建关联常量，如
    // i128::MAX）和 Sym（符号形状引用，如 Sym<B>）都从 MirPlace 挪去了
    // MirOperand——它们是值，不是"能被赋值/取地址的位置"，见 mir.rs 里
    // MirPlace 定义处的完整说明。挪走之后 MirPlace 只剩 Ssa/Field/
    // Index/Deref/EnumPayload 这五种真正的"位置"，这个 match 天然就是
    // 穷尽的，不再需要给 Static/Sym 各写一条"返回 None"的分支（它们
    // 现在根本不可能作为 MirPlace 出现，无需在这里处理）。
    pub(crate) fn place_base_ssa(place: &MirPlace) -> Option<SsaLocal> {
        match place {
            MirPlace::Ssa(s) => Some(*s),
            MirPlace::Field { base, .. } => Self::place_base_ssa(base),
            MirPlace::Index { base, .. } => Self::place_base_ssa(base),
            MirPlace::Deref(base) => Self::place_base_ssa(base),
            MirPlace::EnumPayload { base, .. } => Self::place_base_ssa(base),
        }
    }

    // 关键新增：跟 place_base_ssa 配套——那个函数负责"从一个 Place 里
    // 一路往下找，找到这次操作真正 touch 到的是哪个局部变量"，这个
    // 函数负责"如果这次求值交出去的是对某个 Ssa 局部变量的 Move，就把
    // 它登记进 moved"。
    //
    // 关键修复：这一步原来只在 build_expr_rvalue 的 HirExprKind::Ident
    // 分支（读一个裸变量）和 FieldAccess/Index 分支（读字段/下标，标记
    // 的是 place_base_ssa 找到的根变量）里各自手写了一遍，遗漏了两类
    // 同样会把"对某个 Ssa 局部变量的 Move"交给调用方的地方：
    //
    // 1) build_expr 自己为了求值任何"不是字面量、不是裸标识符"的表达式
    //    （BinaryOp/Call/StructInit/……）而临时开的那个 __tmp 局部变量，
    //    以及 if/match 表达式为了汇合各分支值而开的 Phi 目标变量——
    //    这两种都会被 new_temp 登记进当前作用域的 scope_vars，值构造
    //    完成后又总是原样以 `MirOperand::Move(Ssa(..))` 的形式交回给
    //    调用方（调用方接下来会把这个 operand 嵌进它自己的表达式树，
    //    这个临时变量的值已经被取走了）。如果这里不登记 moved，
    //    pop_scope 会在这个临时变量的作用域结束时，对一个已经被移动
    //    走的值重复调用 drop()，生成的 Rust 编译不过——跟 moved 字段
    //    最初为 Ident/FieldAccess/Index 修的那类问题一模一样，只是这次
    //    踩中的是"编译器自己造的临时变量"，不是"用户写的变量"，因为
    //    build_expr 落地临时变量、Phi 落地目标变量这两处从一开始就没
    //    补这一步登记。
    //
    // 2) HirStmt::Assign / HirStmt::Expr 直接调用 build_expr_rvalue（而
    //    不是 build_expr）拿到的顶层 MirRvalue::Use(operand)——比如
    //    `x = if c { a } else { b };` 或者把一个 if/match 表达式整个
    //    当成语句丢弃结果——这两处拿到的 operand 完全绕开了 build_expr
    //    尾部那次统一处理，同样可能是对 Phi 目标变量的 Move，同样需要
    //    登记，否则该变量作用域结束时会被重复 Drop。
    //
    // 不把这条登记规则各自散落地在每个调用点重写一遍，收进这一个
    // helper：调用方只要在"即将把一个 operand 真正交出去、这个 operand
    // 会被嵌进别处的表达式树"这个时刻调用一次即可。对已经登记过的
    // ssa 再调用一次是幂等的（HashSet::insert），不会因为 Ident/
    // FieldAccess/Index 那几处已经手动登记过而重复出问题。
    pub(crate) fn mark_moved_operand(&mut self, operand: &MirOperand) {
        if let MirOperand::Move(MirPlace::Ssa(ssa)) = operand {
            self.moved.insert(*ssa);
        }
    }

    // 关键新增：Pattern::IntLiteral 只存了一个裸 i64（这是 ast.rs 里
    // Pattern 自己的限制，还没跟着这一轮 Literal 拆分成按位宽/符号
    // 区分的一堆变体一起升级——也就是说目前没法用字面量模式匹配超出
    // i64 范围的 i128/u128 值，这是个已知的、比这次修复范围更大的
    // 缺口，这里先不动 ast.rs，只保证"i64 范围内的值，按 discr 的具体
    // 类型转换成正确的 Literal 变体"这件事是对的）。
    pub(crate) fn int_literal_for_type(v: i64, ty: &Type) -> Result<Literal, String> {
        Ok(match ty {
            Type::I8 => Literal::Int8(v as i8),
            Type::I16 => Literal::Int16(v as i16),
            Type::I32 => Literal::Int32(v as i32),
            Type::I64 => Literal::Int64(v),
            Type::I128 => Literal::Int128(v as i128),
            Type::U8 => Literal::UInt8(v as u8),
            Type::U16 => Literal::UInt16(v as u16),
            Type::U32 => Literal::UInt32(v as u32),
            Type::U64 => Literal::UInt64(v as u64),
            Type::U128 => Literal::UInt128(v as u128),
            other => return Err(format!(
                "match 条件是整数类型，但字面量模式配的类型是 {:?}——不是任何已知的整数类型，\
                 这本该在 sema 阶段就被拦下",
                other
            )),
        })
    }

    // -------- 作用域 --------
    pub(crate) fn push_scope(&mut self) {
        self.scope.push(HashMap::new());
        self.scope_vars.push(Vec::new());
    }

    pub(crate) fn pop_scope(&mut self) {
        // 关键新增：scope（变量名 -> id 的查找表）和 scope_vars（这层
        // 作用域登记了哪些 id，供 Drop 用）本该永远同步 push/pop——
        // push_scope 一次性把两者都 push 一层，理论上不该有任何路径
        // 只碰一个不碰另一个。但这只是"理论上"：这两个 Vec 各自独立，
        // 编译器不会替我们检查这件事，一旦以后哪条新路径（或者一次
        // 疏忽的重构）在某个分支里漏调了 push_scope、或者提前调了一次
        // pop_scope，这里的 `self.scope.pop()` 会静默地多弹一层或者
        // 少弹一层——多弹会把外层作用域的变量表整个丢掉（后面的
        // lookup 全部失败或者查到错误的外层同名变量），少弹会让一层
        // 该消失的作用域一直挂在 self.scope 里（后面同名变量的
        // lookup 可能被这层"本该已经不存在"的绑定错误地挡住）。两种
        // 后果都不是"编译器直接报错"，而是构建期状态悄悄错位、多半要
        // 等生成的 Rust 跑出诡异结果或者编译失败时才会被发现，届时
        // 已经很难跟"某处忘了配对 push/pop"这个真正原因联系起来。
        // 这里加一行 debug_assert 把这条不变式钉死：只要两个栈的深度
        // 曾经有一刻不一致，立刻在 debug 构建里 panic 报出来，不用等
        // 到下游某个更迷惑的错误现象才回头怀疑到这里。
        debug_assert_eq!(
            self.scope.len(),
            self.scope_vars.len(),
            "MirBuilder.scope 和 scope_vars 深度不同步——某处 push_scope/pop_scope 没有配对调用"
        );
        // 关键重构：pop_scope 不再自己维护一份"收集要 Drop 的变量、
        // 检查 moved、push_stmt"的逻辑——那份逻辑现在只在
        // emit_drops_for_scopes 里存在一份。pop_scope 要 Drop 的，正是
        // "当前最内层、也就是即将被弹出的那一层作用域"，这恰好就是
        // `emit_drops_for_scopes(self.scope_vars.len() - 1)`：
        // `scope_vars[len-1..]` 是一个只含最后一层的切片，跟原来
        // `self.scope_vars.pop()` 拿到的是同一批 id。
        //
        // 调用顺序很重要：必须先让 emit_drops_for_scopes 读到这层
        // 作用域（此时它还在 scope_vars 里），再真正把它弹出去——
        // emit_drops_for_scopes 自己只读不弹，弹栈这一步得由 pop_scope
        // 来做。
        if !self.scope_vars.is_empty() {
            let depth = self.scope_vars.len() - 1;
            self.emit_drops_for_scopes(depth);
            self.scope_vars.pop();
        }
        self.scope.pop();
    }

    // -------- 提前退出前的批量 Drop（Drop 语句生成的唯一出口） --------
    // 关键重构：这个方法原来只给 Return/Break 用，pop_scope 另外单独
    // 写了一份几乎一样的逻辑——同样是"把一段还开着的作用域里、没被
    // 消费过的变量补上 Drop"，一个登记进 moved、一个不登记。两份分开
    // 维护本身就是隐患：眼下 pop_scope 不登记 moved 还没有生成真正的
    // `drop(x); drop(x);`（pop_scope 处理的永远是"最后一个"会碰到某段
    // scope_vars 的地方——它把这层作用域整个弹出 scope_vars 之后，
    // 后面不会再有别的代码路径回头找这批 id 了），但只要以后新增一种
    // 提前退出的语句、或者调整一下调用顺序，两边只要有一次范围重叠，
    // 少了这一步登记就会真的生成重复 Drop。让这个方法成为生成 Drop
    // 语句的唯一出口，pop_scope 退化成对它的一次特例调用（只处理
    // "最内层那一层作用域"，见下面 pop_scope 的实现），不再自己维护
    // 一份独立逻辑。
    //
    // 顺带带来一个小的正确性改进：原来 pop_scope 是按变量的声明顺序
    // 正着 Drop 的，这个方法（当初为 Return 设计）按声明顺序倒着
    // Drop——跟 Rust 真实的析构顺序（后声明的先析构）一致。统一之后
    // pop_scope 也自然跟着改成这个更正确的顺序。
    pub(crate) fn emit_drops_for_scopes(&mut self, from_depth: usize) {
        // 先只读地把要 Drop 的 SsaLocal 收集进独立 Vec（只碰
        // scope_vars/moved，用的都是 &self），这一轮不可变借用结束后，
        // 再单独一轮调用 push_stmt——避免一边遍历 scope_vars（&self）
        // 一边调用需要 &mut self 的 push_stmt 触发 E0502。
        let mut to_drop = Vec::new();
        for scope_ids in self.scope_vars[from_depth..].iter().rev() {
            for &id in scope_ids.iter().rev() {
                let ssa = self.current_ssa(id);
                if !self.moved.contains(&ssa) {
                    to_drop.push(ssa);
                }
            }
        }
        for ssa in to_drop {
            self.push_stmt(MirStmt::Drop { place: MirPlace::Ssa(ssa) });
            self.moved.insert(ssa);
        }
    }

    // -------- 循环栈（给 break / continue 用） --------
    // 关键新增：跟 push_scope/pop_scope 配套，在进入 While/Loop 的循环
    // 体之前调用——此时循环体自己的 push_scope 还没发生，记下的
    // scope_depth 就是"循环体之外"的作用域层数，break 时用它切出
    // "循环体内部、该被跳过并补 Drop"的那一段 scope_vars（见 LoopCtx
    // 定义处的注释）。
    //
    // 关键修复（continue 支持）：签名加了 continue_target 和
    // loop_phis 两个参数——continue 语句既要知道"跳到哪继续下一轮"
    // （continue_target），也要知道"循环头有哪些 Phi 需要给这条新增
    // 的回边补一个 incoming"（loop_phis）。两者都是调用点
    // （mir_builder.rs 的 While/Loop 分支）在插好循环头 Phi、算出
    // continue 该去的块之后就已经手上有的信息，这里只是原样存进
    // LoopCtx，供之后 HirStmt::Continue 分支从栈顶取用。
    //
    // 关键新增（SSA 强化）：LoopCtx 现在还带着三样东西——
    //   header_versions：循环头 Phi 插完之后 ssa_versions 的快照，回边/
    //     continue 处用它检查"有没有变量在循环里被改写却没有 Phi"；
    //   locals_limit：进入循环时 locals 的数量，id 比它小的才是循环外
    //     的变量；
    //   break_arms：每个 break 发生时（所在块，当时的版本表）——它们
    //     和 While 条件为假的那条边一起，构成循环出口的全部前驱，出口
    //     处按这些路径汇合变量版本。
    pub(crate) fn push_loop(
        &mut self,
        continue_target: usize,
        break_target: usize,
        loop_phis: Vec<LoopPhiInfo>,
        header_versions: HashMap<usize, u32>,
        locals_limit: usize,
    ) {
        self.loop_stack.push(LoopCtx {
            continue_target,
            break_target,
            scope_depth: self.scope_vars.len(),
            loop_phis,
            header_versions,
            locals_limit,
            break_arms: Vec::new(),
        });
    }

    // 循环体构建完毕（不管成功还是中途报错）都要弹栈，避免栈里留着
    // 一层已经不存在的循环，把外层同名/同层级的另一个循环的 break
    // 误导到这层失效的目标上。调用点用"先 pop 再 `?` 传播错误"的顺序
    // 保证这一点，见 mir_builder.rs 里 While/Loop 分支的写法。弹出的
    // LoopCtx 交还给调用方——出口汇合要用里面收集到的 break_arms。
    pub(crate) fn pop_loop(&mut self) -> Option<LoopCtx> {
        self.loop_stack.pop()
    }

    // 关键新增：循环携带变量的健全性检查。循环头只给"循环体里被赋值过
    // 的变量"插 Phi，而"哪些变量被赋值过"靠 collect_modified_vars_* 预
    // 扫描——它明说了覆盖不全（比如藏在算术子表达式里的赋值块）。漏扫
    // 的后果是静默的：变量在循环里被改写了，但循环头没有 Phi，第二轮
    // 迭代读到的还是进入循环之前的旧版本，生成的程序悄悄算错。这里在
    // 每条回边（正常掉出尾部、continue）上对比当前版本表和循环头的
    // 快照：循环外的变量、没有 Phi、版本却变了，就直接报错，不让它
    // 静默通过。
    pub(crate) fn check_loop_carried(
        locals: &[MirLocal],
        ctx: &LoopCtx,
        current: &HashMap<usize, u32>,
    ) -> Result<(), String> {
        let mut offending: BTreeSet<usize> = BTreeSet::new();
        for (&id, &ver) in current {
            if id >= ctx.locals_limit {
                continue;
            }
            if ctx.loop_phis.iter().any(|p| p.base_id == id) {
                continue;
            }
            let before = ctx.header_versions.get(&id).copied().unwrap_or(0);
            if ver != before {
                offending.insert(id);
            }
        }
        if let Some(&id) = offending.iter().next() {
            let name = locals
                .get(id)
                .and_then(|l| l.name.clone())
                .unwrap_or_else(|| format!("#{}", id));
            return Err(format!(
                "internal error: 变量 `{}` 在循环体内被改写，但循环头没有为它插入 Phi \
                 （collect_modified_vars_* 漏扫了某种写入位置）——继续构建会让第二轮\
                 迭代读到过期版本",
                name
            ));
        }
        Ok(())
    }

    pub(crate) fn bind(&mut self, name: String, id: usize) {
        self.scope.last_mut().unwrap().insert(name, id);
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<usize> {
        for scope in self.scope.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(*id);
            }
        }
        None
    }

    // -------- 基本块 --------
    // 关键设计：块在用到之前就先创建好（占位终止器），之后随时可以用
    // id 引用它、往里面塞语句，最后再补上真正的终止器。这是为了支持
    // if/while 这类需要"提前知道 then/else 块的 id 才能设置当前块的
    // 跳转目标"的控制流——不这样做的话，构建顺序会陷入"先有鸡还是先
    // 有蛋"的死结。
    //
    // 关键重构（Placeholder/Unreachable 拆分，见 mir.rs::MirTerminator
    // 定义处的说明）：这里原来拿 Unreachable 当占位终止器用，语义上
    // 是两件不同的事——Unreachable 该表示"已知执行不到这里"（发散
    // 之后、match 穷尽之后……），Placeholder 才是"这个块还没被真正
    // 写过终止器"。改成 Placeholder，`current_block_terminator_is_unset`
    // （下面，原名 current_terminator_is_placeholder）也跟着改成查
    // Placeholder，不再用"terminator 恰好等于 Unreachable"这个巧合
    // 兜底判断"是不是还没写"。
    pub(crate) fn new_block(&mut self) -> usize {
        let id = self.blocks.len();
        self.blocks.push(MirBlock {
            id,
            stmts: Vec::new(),
            terminator: MirTerminator::Placeholder,
        });
        id
    }

    pub(crate) fn switch_to_block(&mut self, id: usize) {
        self.current_block = id;
    }

    pub(crate) fn push_stmt(&mut self, stmt: MirStmt) {
        self.blocks[self.current_block].stmts.push(stmt);
    }

    pub(crate) fn set_terminator(&mut self, term: MirTerminator) {
        self.blocks[self.current_block].terminator = term;
    }

    // 关键重构：改名自 current_terminator_is_placeholder——"is_placeholder"
    // 这个名字在 Placeholder 变体存在之后会引起歧义（听起来像是在问
    // "terminator 是不是 Placeholder 这个具体变体"，实际问的正好就是
    // 这件事，但改名成 is_unset 更直白：不用记住"占位"具体对应哪个
    // 变体，望文生义就知道这是在问"这个块的终止器是不是还没被设置过"。
    pub(crate) fn current_block_terminator_is_unset(&self) -> bool {
        match self.blocks[self.current_block].terminator {
            MirTerminator::Placeholder => true,
            _ => false,
        }
    }
}
