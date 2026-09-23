// mir_builder.rs
//
// 拆分说明：这个文件原来快 1520 行，臃肿到难以维护。现在拆成四份：
//   - intrinsic.rs：内建常量/内建函数的识别逻辑（不需要 &mut MirBuilder
//     本身，只需要几个状态位，见 intrinsic.rs 里新加那两个函数的注释）。
//   - state.rs：Local / SSA / Scope / 基本块这些"构建期状态"怎么增删。
//   - guide.rs：Place（左值）怎么从一个 HIR 表达式构造出来。
//   - 这个文件：剩下的——build() 入口、build_fn、以及真正的"HIR 节点
//     -> MIR 节点"翻译主体（build_block/build_stmt/build_expr/
//     build_expr_rvalue/build_call_arg）。
//
// SharedContext/MirBuilder 的字段从私有改成 pub(crate)：拆到 state.rs/
// guide.rs 之后，那两个文件是跟这个文件平级的顶层模块（不是子模块），
// 平级模块之间访问私有字段过不了编译，只能放宽到 crate 内可见——用
// pub(crate) 而不是整个 pub，是不想让这些内部字段被拆分之外的、
// crate 外部的使用者（这个 crate 本来就是被 main.rs 当库用的）看到，
// 它们纯粹是构建过程的内部细节。
use crate::ast::{Pattern, Type};
use crate::hir::*;
use crate::mir::*;
use std::collections::HashMap;
use std::collections::HashSet;
use crate::intrinsic::IntrinsicFn;

// ===== 跨函数共享的只读上下文（build() 里构建一次，每个函数复用） =====
pub(crate) struct SharedContext {
    // struct_name -> (field_name -> field_ty)，FieldAccess 查真实字段
    // 类型用，不再靠猜。
    pub(crate) struct_fields: HashMap<String, HashMap<String, Type>>,
    // variant_name -> enum_name（要求全局唯一）。裸 Ok/Err/Some/None 这类
    // 不带 :: 前缀的写法，语法上跟普通函数调用（HirExprKind::Call）长得
    // 一模一样，sema.rs 那边靠"在所有已注册枚举里找恰好一个同名变体"
    // 识别出来，但那个识别结果只体现在类型检查上，没有改写 HIR 节点
    // 本身——MIR 构建这里得重新做一遍同样的查找，才能正确区分
    // "Err(())" 这种裸枚举变体构造和真正的函数调用。
    pub(crate) variant_to_enum: HashMap<String, String>,
    pub(crate) variant_indices: HashMap<(String, String), usize>,
    // 关键新增：(enum_name, variant_name) -> 这个变体自己声明的 payload
    // 类型。EnumVariantWithBinding 模式（`Ok(v) => ...` 里的 v）要把
    // payload 解出来绑定成一个新的 MirLocal，而 MirLocal.ty 是必填
    // 字段——不能瞎猜一个类型糊弄过去，也没法从 Switch 那边反推出来
    // （Switch 只留了 i64 下标，早就不知道原来的 payload 长什么样了），
    // 只能在这里从 hir.enums 的原始声明里查。
    pub(crate) variant_payload_types: HashMap<(String, String), Type>,
}

// 关键新增：break 要跳到哪个 end_block，取决于"离它最近的那层 while/
// loop"，天然是一个栈——嵌套循环时，内层 break 只影响内层，不能捅穿
// 到外层循环的 end_block，进入内层循环时 push 一层，构建完内层循环体
// 后 pop 掉，栈顶永远是"当前离 break 语句最近的那层循环"该跳去的地方。
//
// scope_depth 记的是"进入这层循环体之前，self.scope_vars 已经有多少层"
// （即循环体自己的 push_scope 还没发生时的深度）。break 语句执行时，
// 当前可能已经在循环体内部又嵌套了若干层 block/match 分支的作用域
// （build_block/HirExprKind::Block 每进一层都会 push_scope），这些
// "循环体内部"的作用域到 break 发生时全部要被跳过、里面活着的变量全部
// 要在跳走之前补 Drop——但循环体之外（scope_depth 那一层以下）的作用域
// 不受 break 影响，仍然活着，不能碰。用这个深度值就能精确切出
// "该 Drop 哪一段 scope_vars"，不多不少。
#[derive(Clone, Copy)]
pub(crate) struct LoopCtx {
    pub(crate) break_target: usize,
    pub(crate) scope_depth: usize,
}

// 关键新增（循环头 Phi）：insert_loop_header_phis 插入的每一条 Phi，
// 需要在循环体构建完之后回去补第二个 incoming（回边）——finalize_
// loop_header_phis 需要知道"回去补哪一条"，这三个字段就是定位信息：
// 插在哪个块（header_block）、块里第几条语句（stmt_index，Vec 下标，
// 插入时就是 push 之前的长度）、对应哪个变量（base_id）。
struct LoopPhiInfo {
    header_block: usize,
    stmt_index: usize,
    base_id: usize,
}

// ===== 重构：把"发散"变成显式返回值，不再靠类型层反推 =====
//
// 背景：Type::Never 是后期追加进这条流水线的概念。追加之前，"这个
// 表达式/语句会不会发散"这件事根本没有结构化的表达方式，只能在几个
// 手选的"关键位置"（build_fn 收尾、HirStmt::Expr、If 的两个分支、
// Match 的每个分支）里各自重新查一遍 `expr.ty.is_never()`，反推"这条
// 路径是不是不会正常产生值"。这是四类互相独立、容易漏、且没有覆盖到
// 更深层子表达式（`let x = 1 + panic();` 这种）的特判点。
//
// 现在反过来：发散只有一个真正的源头——build_expr_rvalue 里构造一次
// 返回类型是 Type::Never 的调用（panic、或任何签名声明为 -> never 的
// 函数；方法调用/裸枚举变体构造/内建常量引用这几条路径在这门语言里
// 从来不会是发散源，不需要为它们复制一份同样的检查）。以及 Return、
// Break 这两种直接设置终止器的语句。从这个源头开始，Diverging::Diverged
// 沿着调用链原样往上传播——BinaryOp 的另一个操作数、函数调用的下一个
// 实参、if/match 的另一个分支、块里的下一条语句……每一层看到 Diverged
// 都只做一件事：不再往下求值，把 Diverged 原样交给自己的调用方。不再
// 需要在这些地方重新查一次 expr.ty，四类特判点全部收敛成"收到 Diverged
// 就传播"这一句话。
//
// 用一个泛型枚举而不是给 build_expr/build_expr_rvalue/build_block/
// build_stmt/guide.rs::build_place 各写一份同构的三行枚举：内容
// 一模一样，只是 Value 里包的类型不同（MirOperand / MirRvalue /
// Option<MirOperand> / MirPlace），没必要写五遍。下面的类型别名把
// 每个场景该用哪个具体实例化点出来，签名上仍然一眼看出"这是操作数级
// 的结果"还是"这是块级的结果"。
pub(crate) enum Diverging<T> {
    // 正常求值完成，携带产出的值。
    Value(T),
    // 已经发散：当前块（self.current_block，在发散发生的那一刻）已经
    // 有了真正的终止器（Return/Goto/Unreachable 之一），调用方不应该、
    // 也不需要再对这次求值做任何事——不再 push_stmt，不再 set_terminator，
    // 直接把 Diverged 原样继续往外传。这是整份重构唯一要维护的不变式：
    // "谁制造了 Diverged，谁负责在那一刻把终止器设成真正正确的样子；
    // 谁只是转手传播 Diverged，谁绝对不碰 self.current_block"。
    Diverged,
}

pub(crate) type ExprResult = Diverging<MirOperand>;
pub(crate) type RvalueResult = Diverging<MirRvalue>;
pub(crate) type BlockResult = Diverging<Option<MirOperand>>;

// 关键重构（去掉 macro_rules!）：项目里明确不用 Rust 的任何宏（包括
// matches!）——check_type.rs / intrinsic.rs / check_model.rs 都专门为
// 这条规矩改过代码。这里原来为了不在几十个调用点重复"见到 Diverged
// 就 return"这句话，加了一个 `propagate!` 宏，属于重新破例。
//
// 换成一个普通泛型函数：把 `Diverging<T>` 转成标准库的 `Result<T, ()>`
// ——正常求值是 `Ok(v)`，发散是 `Err(())`（不需要在 Err 里带上
// `Diverging::Diverged` 本身，调用点只关心"是不是发散了"这一件事，
// `()` 已经完整表达了这件事，不用再包一层同义的类型）。
//
// 但函数没法替宏做到"从调用方所在的函数提前 return"这件事——`return`
// 只能出现在字面写着它的那个函数体里，一个被调用的普通函数即使自己
// 执行了 `return`，退出的也只是它自己，不会影响调用方。所以这个函数
// 只负责"分类"（是 Value 还是 Diverged），调用点仍然要自己写一个
// `match`，在 Err 分支里显式 `return Ok(Diverging::Diverged)`——这跟
// 宏展开出来的代码在运行时效果一致，只是不再靠宏做代码复制，而是每个
// 调用点都老老实实写一遍这几行。
pub(crate) fn propagated<T>(r: Diverging<T>) -> Result<T, ()> {
    match r {
        Diverging::Value(v) => Ok(v),
        Diverging::Diverged => Err(()),
    }
}

pub struct MirBuilder {
    pub(crate) locals: Vec<MirLocal>,
    pub(crate) blocks: Vec<MirBlock>,
    pub(crate) current_block: usize,
    pub(crate) scope: Vec<HashMap<String, usize>>, // 变量名 -> local id
    pub(crate) scope_vars: Vec<Vec<usize>>,
    pub(crate) unsafe_depth: usize,
    pub(crate) in_forward: bool,
    pub(crate) ssa_versions: HashMap<usize, u32>,
    // 关键修复（找回上一轮被回退掉的东西）：这一版是从更早的快照分支
    // 出来重新改的，上一轮为了配合真正的 Drop 语义加的 `moved` 追踪
    // （连带 pop_scope/Return 那两处修复）整个不见了，pop_scope 现在
    // 又是无条件对作用域里的每个变量插 Drop——回到了"对已经被移动走
    // （哪怕只是部分移动）的值重复调用 drop()，生成的 Rust 编译不过"
    // 这个问题。理由和之前完全一样，不重复展开，直接照抄那一轮的实现。
    //
    // 已知限制（不是这次要修的范围，如实记录）：这是个"变量级"的二值
    // 集合，只能表达"整个变量被移走了 / 没被移走"，没有字段级精度。
    // `let x = p.x;` 之后，真实 Rust 语义下 p 只是部分移动——p 离开
    // 作用域时，rustc 会自动 partial-drop 剩下没被移走的字段，不需要
    // 任何显式代码。但这里的处理是"只要 p 的任意一部分被移动过，就
    // 把整个 p 登记进 moved、pop_scope 直接跳过它，不再显式 Drop"——
    // 这保证了不会生成"对部分移动的值调用 drop()"这种编译不过的代码
    // （这是硬错误，必须优先避免），代价是放弃了对 p 剩余字段做主动
    // 显式 Drop 这件事，指望生成的 Rust 代码本身的作用域退出来兜底。
    // 跟 borrow.rs 开头"部分移动暂按不改变初始化状态处理"是同一个
    // 已经写明的简化，不是这次新引入的缺口——一旦真要做字段级精度，
    // 这个字段要么升级成按 (base_id, 字段路径) 记录的结构，要么彻底
    // 换一种不依赖显式 Drop 语句的设计。
    pub(crate) moved: std::collections::HashSet<SsaLocal>,
    // 关键新增：接上 elaborate.rs 那条链路——elaborate 已经把 for 展开
    // 成 `loop { match __iter.next() { Some(v) => body, None => break } }`，
    // MIR 构建这边必须知道 break 该跳去哪，不然这条链路就是断的（HIR
    // 里带着 Break 节点，MIR 侧却处理不了）。见上面 LoopCtx 的注释。
    pub(crate) loop_stack: Vec<LoopCtx>,
}

impl MirBuilder {
    // 关键新增：build_fn 原来是直接手写一个 MirBuilder { ... } 结构体
    // 字面量，把全部字段的初始值都摊开列一遍——字段一多就容易漏（新加
    // 一个字段，忘了在这唯一的构造点补初始值，编译器会因为缺字段报错
    // 提醒；但如果是把某个字段的"该有的初始值"改错了，比如新加一个
    // `loop_stack: Vec<usize>` 忘了写成 `Vec::new()`，这种编译器不会
    // 帮你查）。收口成一个构造器，以后要加新的构建期状态字段（比如
    // 处理 break/continue 需要的 loop_stack、break_target），只用改
    // 这一个地方，不用担心散落在别处的构造点漏改。
    fn new(in_forward: bool) -> Self {
        Self {
            locals: Vec::new(),
            blocks: Vec::new(),
            current_block: 0,
            scope: vec![HashMap::new()],
            scope_vars: vec![Vec::new()],
            unsafe_depth: 0,
            in_forward,
            ssa_versions: HashMap::new(),
            moved: std::collections::HashSet::new(),
            loop_stack: Vec::new(),
        }
    }

    pub fn build(hir: &HirProgram) -> Result<MirProgram, String> {
        let struct_fields: HashMap<String, HashMap<String, Type>> = hir
            .structs
            .iter()
            .map(|s| {
                let fields = s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
                (s.name.clone(), fields)
            })
            .collect();

        let mut variant_to_enum: HashMap<String, String> = HashMap::new();
        let mut ambiguous: std::collections::HashSet<String> = std::collections::HashSet::new();
        for e in &hir.enums {
            for v in &e.variants {
                if variant_to_enum.contains_key(&v.name) {
                    ambiguous.insert(v.name.clone());
                } else {
                    variant_to_enum.insert(v.name.clone(), e.name.clone());
                }
            }
        }
        for name in &ambiguous {
            variant_to_enum.remove(name);
        }

        let mut variant_indices: HashMap<(String, String), usize> = HashMap::new();
        let mut variant_payload_types: HashMap<(String, String), Type> = HashMap::new();
        for e in &hir.enums {
            for (idx, v) in e.variants.iter().enumerate() {
                variant_indices.insert((e.name.clone(), v.name.clone()), idx);
                if let Some(ty) = &v.ty {
                    variant_payload_types.insert((e.name.clone(), v.name.clone()), ty.clone());
                }
            }
        }

        let shared = SharedContext {
            struct_fields,
            variant_to_enum,
            variant_indices,
            variant_payload_types,
        };

        let mut fns = Vec::new();
        // 顶层函数
        for f in &hir.fns {
            fns.push(Self::build_fn(f, &shared)?);
        }
        // model 里的方法（forward 等）
        for m in &hir.models {
            for f in &m.functions {
                fns.push(Self::build_fn(f, &shared)?);
            }
        }
        // 关键修复：之前完全没遍历 hir.impls——Vec/String/Rational 等等
        // 标准库里几乎所有方法都定义在 implement 块里，只在 hir.impls
        // 而不在 hir.fns 里。不补上这段，标准库的全部方法在 MIR 这层会
        // 直接消失。
        for imp in &hir.impls {
            for f in &imp.functions {
                fns.push(Self::build_fn(f, &shared)?);
            }
        }

        let structs = hir
            .structs
            .iter()
            .map(|s| MirStruct {
                name: s.name.clone(),
                generic_params: s.generic_params.clone(),
                fields: s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
            })
            .collect();

        let enums = hir
            .enums
            .iter()
            .map(|e| MirEnum {
                name: e.name.clone(),
                generic_params: e.generic_params.clone(),
                variants: e.variants.iter().map(|v| (v.name.clone(), v.ty.clone())).collect(),
            })
            .collect();

        // 内建函数使用情况：intrinsic.rs 的注册表现在已经做好了，这里
        // 汇总的是每个函数构建时各自收集到的真实 IntrinsicFn（不再是
        // 权宜之计）。
        let mut intrinsics_used: Vec<IntrinsicFn> = fns
            .iter()
            .flat_map(|f| Self::collect_intrinsics_in_body(&f.body))
            .collect();
        intrinsics_used.sort();
        intrinsics_used.dedup();

        // TODO: consts / protos / interfaces 目前没有对应的 Mir* 结构，
        // 先只覆盖 fns/structs/enums(+ 现在补上的 impls/models) 把主干
        // 打通。
        Ok(MirProgram { structs, enums, fns, intrinsics_used })
    }

    fn collect_intrinsics_in_body(body: &MirBody) -> Vec<IntrinsicFn> {
        let mut out = Vec::new();
        for block in &body.blocks {
            for stmt in &block.stmts {
                let rv = match stmt {
                    MirStmt::Assign { value, .. } => Some(value),
                    MirStmt::ExprStmt(value) => Some(value),
                    _ => None,
                };
                if let Some(MirRvalue::Call { intrinsic_name: Some(name), .. }) = rv {
                    out.push(*name);
                }
            }
        }
        out
    }

    fn build_fn(f: &HirFn, shared: &SharedContext) -> Result<MirFn, String> {
        let mut builder = MirBuilder::new(f.is_forward);
        builder.new_block(); // 入口块，id = 0（struct 里 current_block 已经是 0，不用再赋一次）

        for param in &f.params {
            let id = builder.new_param_local(param.name.clone(), param.ty.clone());
            builder.scope.last_mut().unwrap().insert(param.name.clone(), id);
        }

        // 关键重构（去掉 return_type == Never 的特判）：原来这里要先
        // 查一遍函数签名的 return_type 是不是 Type::Never，才知道函数体
        // 正常"掉出"最后一句时该塞一个 Unreachable 还是 Return(ret)——
        // 这是从签名反推控制流。现在不用反推了：build_block 会通过
        // Diverging 把"函数体是不是真的处处发散"结构化地带出来。
        //
        // 关键重构（这里曾经因为 Unreachable/Placeholder 没分开而出过
        // 一次真 bug，现在补一句说明这条判断为什么现在是安全的）：
        // 曾经有一版实现，无论 build_block 返回 Diverging::Value 还是
        // Diverging::Diverged，都无条件调用
        // `current_block_terminator_is_unset()` 来决定要不要
        // `set_terminator(Return(ret))`——那时候"占位符"和"发散源头
        // 设置的真终止器"共用同一个值（Unreachable），比如
        // `fn foo() -> i32 { panic("x") }`：build_block 返回
        // Diverging::Diverged，但当时的判断读到的 Unreachable 跟"这个
        // 块从没被设置过"是同一个值，分不出来，于是把发散源头正确
        // 设置的 Unreachable 硬生生覆盖成 `Return(None)`，生成的 Rust
        // 里出现 `return;`，在声明返回 i32 的函数里编译不过。
        //
        // 现在 mir.rs 把"占位"（Placeholder）和"已知不可达"
        // （Unreachable）拆成了两个不同的终止器变体，
        // `current_block_terminator_is_unset()` 问的是"是不是
        // Placeholder"，不再跟 Unreachable 混在一起——Diverged 分支下，
        // 发散源头设置的终止器（不管是 Unreachable、Return 还是 Goto）
        // 都不是 Placeholder，这个判断在两个分支下都天然给出正确答案，
        // 不需要再靠 match Diverging 的哪个变体来人工避开这个歧义。
        let ret = match builder.build_block(&f.body, shared)? {
            Diverging::Value(v) => v,
            Diverging::Diverged => None,
        };
        if builder.current_block_terminator_is_unset() {
            builder.set_terminator(MirTerminator::Return(ret));
        }

        Ok(MirFn {
            name: f.name.clone(),
            generic_params: f.generic_params.clone(),
            params: f.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect(),
            return_type: f.return_type.clone(),
            body: MirBody { locals: builder.locals, blocks: builder.blocks },
            effect_set: f.effects.clone(),
        })
    }

    // -------- Block / Stmt --------
    fn build_block(&mut self, block: &HirBlock, shared: &SharedContext) -> Result<BlockResult, String> {
        self.push_scope();
        let mut last: Option<MirOperand> = None;
        for stmt in &block.stmts {
            // 关键修复：原来是 `match self.build_stmt(stmt, shared)? {
            // ... }`——`?` 在 Err 分支直接把 Err(String) 从 build_block
            // 整个 return 出去，不会走到任何一个分支里的 `pop_scope()`，
            // 上面 push_scope 开的这层作用域就永远不会被弹掉。跟
            // build_rvalue_expr_and_pop_scope 是同一类问题：先把结果存
            // 起来，不管是 Err 还是 Diverged 都先 pop_scope，再决定
            // 是把错误传播出去还是继续这一轮循环。
            let stmt_result = self.build_stmt(stmt, shared);
            match stmt_result {
                Ok(Diverging::Diverged) => {
                    // 关键重构：这条语句已经发散了（不管是 Return/Break，
                    // 还是内部某个子表达式碰到了一次 Never 调用），这个
                    // 块从这里往后的语句在真实控制流里根本不会被执行到。
                    // 直接停止遍历，不再挨个把它们"建"进一个反正也用不上
                    // 的死块——旧设计靠 Return/Break 自己开一个新块承接
                    // 死代码，指望 control::simplify 在 borrow_check 之前
                    // 把死块删掉；现在干脆不生成这段死代码，连带避免了
                    // "死代码自己又踩中某个 moved/Drop 边界情况"这类需要
                    // 依赖别的 pass 执行顺序才能兜底的隐式依赖。scope 的
                    // push/pop 必须配平，所以 pop_scope 这一步不能省。
                    self.pop_scope();
                    return Ok(Diverging::Diverged);
                }
                Ok(Diverging::Value(v)) => {
                    last = v;
                }
                Err(e) => {
                    self.pop_scope();
                    return Err(e);
                }
            }
        }
        self.pop_scope();
        Ok(Diverging::Value(last))
    }

    fn build_stmt(&mut self, stmt: &HirStmt, shared: &SharedContext) -> Result<BlockResult, String> {
        match stmt {
            HirStmt::Let { name, ty, init, mutable, persist, .. } => {
                // 关键修复（这次重构顺带补上）：`let x = panic();` 这种
                // 写法原来完全没被检查过——is_never() 特判从来没覆盖过
                // Let 语句，build_expr(init) 拿到的是 panic() 的返回值
                // "操作数"，会被当成正常值继续走下面 new_local + Assign，
                // 生成一段类型上说得过去但语义上荒谬的代码（给一个永远
                // 不会被赋值的变量赋一个不存在的值）。现在 init 发散时
                // build_expr 会如实带出 Diverged，这里直接原样传播——
                // 不创建这个 local，也不生成这条 Assign。
                let value = match propagated(self.build_expr(init, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                let local_ty = ty.clone().unwrap_or_else(|| init.ty.clone());
                let id = if *persist {
                    self.new_persist_local(Some(name.clone()), local_ty, *mutable)
                } else {
                    self.new_local(Some(name.clone()), local_ty, *mutable, true)
                };
                self.bind(name.clone(), id);
                // 递增版本号（第一次赋值，版本从 0 → 1）
                let version = self.ssa_versions.get(&id).copied().unwrap_or(0);
                let new_version = version + 1;
                self.ssa_versions.insert(id, new_version);
                let dest = MirPlace::Ssa(SsaLocal { base_id: id, version: new_version });
                self.push_stmt(MirStmt::Assign {
                    dest,
                    value: MirRvalue::Use(value),
                });
                Ok(Diverging::Value(None))
            }
            HirStmt::Expr { expr, .. } => {
                // 关键重构：原来这里靠"先无条件 push_stmt，再单独查一次
                // expr.ty 是不是 Type::Never，是的话再补一个 Unreachable
                // + 开新块"来处理发散——从类型层反推。现在 build_expr_rvalue
                // 自己在真正发散的源头（内部某次 Never 调用）就已经把
                // 该发的语句 push 好、该设的终止器也设好了，见到
                // Diverging::Diverged 时这里什么都不用做，直接原样传播；
                // 不会重复 push 那条语句（发散源头返回的是 Diverged，不
                // 是一个还需要在这里包一层 ExprStmt 的 MirRvalue::Value）。
                // 也不再需要"开新块承接死代码"这一步——build_block 的
                // 循环见到 Diverged 会直接停止遍历，根本不会有后续语句
                // 被建进任何块里。
                match self.build_expr_rvalue(expr, shared)? {
                    Diverging::Diverged => Ok(Diverging::Diverged),
                    Diverging::Value(value) => {
                        // 关键修复：这里直接消费 build_expr_rvalue 的
                        // 结果，绕开了 build_expr 尾部那次统一的
                        // mark_moved_operand 调用（见那边新加的说明）。
                        // 大多数语句表达式（Call/BinaryOp/……）的顶层
                        // rvalue 不是裸 Use，这里不受影响；但如果这条
                        // 语句就是一个整个被丢弃结果的 if/match 表达式
                        // （`if c { a } else { b };`），value 会是
                        // `Use(Move(Ssa(phi_ssa)))`——phi_ssa 这次被
                        // ExprStmt 消费掉了，同样需要登记进 moved，不然
                        // 它的作用域结束时会被重复 Drop。
                        if let MirRvalue::Use(operand) = &value {
                            self.mark_moved_operand(operand);
                        }
                        self.push_stmt(MirStmt::ExprStmt(value));
                        Ok(Diverging::Value(None))
                    }
                }
            }
            HirStmt::Return { expr, .. } => {
                // 先计算返回值——build_expr 如果读到的是某个局部变量，
                // 会顺带把它标进 self.moved（见 emit_drops_for_scopes 为
                // 什么要查这张表），这个顺序不能反。
                //
                // 关键重构：如果这个返回表达式自己就发散了（比如
                // `return panic();`），build_expr 早已在更深处把
                // current_block 的终止器设成了 Unreachable——这里必须
                // 立刻原样传播、直接退出，绝不能再往下走到
                // `self.set_terminator(Return(...))` 那一步：那会把刚刚
                // 正确设好的 Unreachable 硬生生覆盖成一个错误的 Return，
                // 而且下面 emit_drops_for_scopes 那套"函数正常返回前清理
                // 栈上变量"的逻辑，在"根本没有正常返回这回事"的场景下
                // 也没有意义，不该执行。
                let operand = match expr.as_ref() {
                    Some(e) => Some(match propagated(self.build_expr(e, shared)?) {
                        Ok(v) => v,
                        Err(()) => return Ok(Diverging::Diverged),
                    }),
                    None => None,
                };

                // 关键修复（找回上一轮的修复，这次抽成 emit_drops_for_scopes
                // 复用给 Break）：只 Drop"当前仍然开着的那些作用域"里
                // 登记的变量（self.scope_vars，从内层到外层遍历），跳过
                // 已经被消费过的变量。Return 退出的是整个函数，所以从
                // 深度 0（最外层）开始，把所有还开着的作用域全部算进去。
                self.emit_drops_for_scopes(0);

                self.set_terminator(MirTerminator::Return(operand));
                Ok(Diverging::Diverged)
            }
            HirStmt::Assign { target, expr, .. } => {
                // 关键修复（这次重构顺带补上）：赋值目标本身也可能发散
                // （比如 `arr[panic()] = 5;` 的下标表达式），guide.rs 的
                // build_place 现在也走同一套 Diverging 传播，这里跟其他
                // 地方一样，见到 Diverged 就不再往下求值。
                let place = match propagated(self.build_place(target, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                // 如果目标是 Ssa，则递增版本；否则（字段/索引等）保留原样
                let dest = match place {
                    MirPlace::Ssa(ssa) => {
                        let new_ver = self.new_version(ssa.base_id);
                        MirPlace::Ssa(SsaLocal { base_id: ssa.base_id, version: new_ver })
                    }
                    _ => place,
                };
                let value = match propagated(self.build_expr_rvalue(expr, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                // 关键修复：同 HirStmt::Expr 那处——这里直接消费
                // build_expr_rvalue 的结果，绕开了 build_expr 尾部的
                // mark_moved_operand。`x = if c { a } else { b };` 这类
                // 写法，value 会是 `Use(Move(Ssa(phi_ssa)))`，phi_ssa
                // 在这里被真正消费（赋给 dest），必须登记进 moved，
                // 否则它的作用域结束时会被重复 Drop。
                if let MirRvalue::Use(operand) = &value {
                    self.mark_moved_operand(operand);
                }
                self.push_stmt(MirStmt::Assign { dest, value });
                Ok(Diverging::Value(None))
            }
            HirStmt::While { cond, body, .. } => {
                let cond_block = self.new_block();
                let body_block = self.new_block();
                let end_block = self.new_block();

                // 关键新增（循环头 Phi）：保存 preheader 块 id——这是
                // 进入循环之前所在的那个块，也是循环头 Phi 的第一个
                // incoming 来源。必须在 `set_terminator(Goto(cond_block))`
                // 之前读，那之后 self.current_block 还是这个块，只是
                // 即将被賦予一个真终止器，读它的 id 不受影响，但为了
                // 代码顺序清晰，习惯性地在改动它之前先把 id 存下来。
                let preheader_block = self.current_block;

                self.set_terminator(MirTerminator::Goto(cond_block));

                self.switch_to_block(cond_block);
                // 关键新增：在 cond_block 最前面插入循环头 Phi——此时
                // cond_block 里还没有任何语句（下面这行才开始求值
                // cond），插入操作等价于"放在最前面"。
                //
                // 为什么循环需要 Phi：MirBuilder 的 SSA 是直线代码
                // 版——同一个 base_id 每被 Assign 一次就 new_version，
                // current_ssa(base_id) 返回"最后一次赋值对应的版本"。
                // 这对顺序执行的代码没问题，但循环有一条"回边"（body
                // 执行完跳回 cond_block），循环第二轮开始时，cond_block
                // 里读到的 base_id 应该是"上一轮循环体结束时的值"，不是
                // "进入循环之前的值"——直线 SSA 表达不出"这个值取决于
                // 我们是第一次到这里、还是绕了一圈回来的"这种二义性，
                // 必须在循环头显式插入 Phi 节点，两个 incoming 分别对应
                // "从 preheader 进来"和"从循环体尾部绕回来"这两条边。
                // 这也是为支配树、GVN、SCCP（稀疏条件常量传播）、聚合
                // 常量折叠这些后续要做的分析做准备——它们都要求输入是
                // 规范的 SSA 形式，循环头没有 Phi 的话，"同一个变量在
                // 不同到达路径上可能取不同值"这件事根本没有被表达出来。
                let loop_phis = self.insert_loop_header_phis(preheader_block, body);

                // 求值 cond——现在读到 body 里被修改过的变量，读到的是
                // 上面刚插入的 phi 版本（insert_loop_header_phis 内部
                // 调用 new_version 时已经更新了 self.ssa_versions，
                // current_ssa 自然返回 phi 版本），是正确的。
                let cond_operand = match propagated(self.build_expr(cond, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                self.set_terminator(MirTerminator::If {
                    cond: cond_operand,
                    then_block: body_block,
                    else_block: end_block,
                });

                self.switch_to_block(body_block);
                // 关键新增：接上循环栈——body 内部（可能嵌套若干层
                // block/match）出现的 Break 要能找到这里的 end_block。
                // push_loop 记的 scope_depth 是"body 自己的 push_scope
                // 还没发生"时的深度，build_block 马上会 push 一层，跟
                // Break 分支里"该 Drop 到哪一层为止"的计算对得上。
                //
                // 关键修复（? 跳过 pop_loop）：原来这里是
                // `self.build_block(body, shared)?;`——`?` 在 Err 分支
                // 直接把整个函数 return 掉，不会走到下面的
                // `self.pop_loop()`，loop_stack 就会残留一层。当前"整个
                // 编译失败就直接退出"的架构掩盖了这个问题（反正马上就
                // 不再使用这个 MirBuilder 了），但不能依赖这个前提——
                // 先存住结果，无论如何先 pop_loop，再 `?`。
                self.push_loop(end_block);
                let body_result = self.build_block(body, shared);
                self.pop_loop();
                let body_result = body_result?;

                // 关键说明：body 是不是发散不影响 while 语句本身要不要
                // 报告发散——循环外仍然可能通过"条件一开始就为假"这条
                // 路径直接落到 end_block，while 语句永远不发散，这一点
                // 跟原来的设计意图一致。
                //
                // 关键新增（回填循环头 Phi 的第二个 incoming）：只有
                // body 正常掉出尾部、且尾部所在的块此刻还没有真正的
                // 终止器时，才存在一条"循环体跑完、回去重新判断条件"的
                // 回边——这时候才需要把这条回边记进 Phi。如果 body 以
                // break/return/一次无条件发散收尾，尾部所在的块已经有
                // 了明确的终止器（Goto(end_block)/Return(..)/
                // Unreachable 之一），根本不会 Goto 回 cond_block，也就
                // 没有这条回边，Phi 就只保留 insert_loop_header_phis
                // 那时候插入的第一个 incoming，退化成单路 Phi——后续
                // control::simplify 的 simplify_phi 会把这种单路 Phi
                // 折叠回一次普通的 Use，不需要 mir_builder.rs 这里操心。
                match body_result {
                    Diverging::Value(_) => {
                        if self.current_block_terminator_is_unset() {
                            let back_edge_block = self.current_block;
                            self.set_terminator(MirTerminator::Goto(cond_block));
                            self.finalize_loop_header_phis(&loop_phis, back_edge_block);
                        }
                    }
                    Diverging::Diverged => {}
                }

                self.switch_to_block(end_block);
                Ok(Diverging::Value(None))
            }
            HirStmt::Loop { body, .. } => {
                let body_block = self.new_block();
                let end_block = self.new_block();

                let preheader_block = self.current_block;

                self.set_terminator(MirTerminator::Goto(body_block));
                self.switch_to_block(body_block);
                // 同 While 分支：body_block 兼任循环头，Phi 插在这里
                // 最前面。跟 While 不同的是，body_block 里马上会开始
                // 塞循环体自己的语句（没有单独的 cond_block），但
                // insert_loop_header_phis 此刻调用时 body_block 依然
                // 是空的（body 还没开始构建），插入操作同样等价于
                // "放在最前面"。
                let loop_phis = self.insert_loop_header_phis(preheader_block, body);

                self.push_loop(end_block);
                let body_result = self.build_block(body, shared);
                self.pop_loop();
                let body_result = body_result?;

                // 同 While 分支的说明：loop 语句本身是否发散跟 body 的
                // Diverging 结果无关（这里选择跟原来的行为一致，仍然不
                // 尝试证明"这个裸 loop 里到处都没有 break、因此整个 loop
                // 语句真的发散"——那是一个独立的、这次不做的分析，跟
                // Type::Never 特判去留没有关系）。
                match body_result {
                    Diverging::Value(_) => {
                        if self.current_block_terminator_is_unset() {
                            let back_edge_block = self.current_block;
                            self.set_terminator(MirTerminator::Goto(body_block));
                            self.finalize_loop_header_phis(&loop_phis, back_edge_block);
                        }
                    }
                    Diverging::Diverged => {}
                }

                self.switch_to_block(end_block);
                Ok(Diverging::Value(None))
            }
            HirStmt::Break { .. } => {
                // 关键修复：接上 elaborate.rs 那条链路——for 循环被展开成
                // `loop { match __iter.next() { Some(v) => body, None => break } }`
                // （§expand_for/build_for_next_match），HIR 里这条 Break
                // 早就在了，缺的只是 MIR 这边怎么用循环栈找到该跳去哪。
                // loop_stack 为空说明 break 出现在任何 while/loop 之外，
                // sema.rs 应该已经拦住这种情况，这里报错而不是 panic，
                // 方便定位是不是两边检查不一致。
                let loop_ctx = self.loop_stack.last().copied().ok_or_else(|| {
                    "internal error: HirStmt::Break 出现在循环之外 \
                     (sema.rs 的检查应该已经拦住这种情况，走到这里说明两边检查不一致)"
                        .to_string()
                })?;

                // 关键修复（emit_drops_for_scopes 顺带修的那个重复 Drop
                // 问题）：Drop 范围只到 loop_ctx.scope_depth 为止——那是
                // 循环体自己的作用域，再往外是循环外层还活着的作用域，
                // break 不影响它们，不能一起 Drop 掉（这跟 Return 一次性
                // Drop 所有作用域的语义不一样：Return 退出的是整个函数，
                // Break 只退出这一层循环）。
                self.emit_drops_for_scopes(loop_ctx.scope_depth);

                self.set_terminator(MirTerminator::Goto(loop_ctx.break_target));
                Ok(Diverging::Diverged)
            }
            HirStmt::For { .. } => {
                // 这一支现在纯粹是防御性检查：按流水线设计
                // （pipeline.rs：Elaborate::elaborate 在 MirBuilder::build
                // 之前跑），HirStmt::For 应该已经被 elaborate.rs 的
                // expand_for 展开成 `let __iter = ...; loop { match
                // __iter.next() { ... } }`，不会有 For 节点活着走到这一层。
                // 真走到这里，说明要么漏调了 elaborate 那一趟，要么有
                // 别的路径绕过了它——报错而不是 panic，方便定位是流水线
                // 顺序问题还是 elaborate.rs 本身漏了某种 For 场景。
                Err("internal error: HirStmt::For 不该走到 mir_builder 这一层——elaborate.rs 应已在 MIR 构建之前将其展开成 loop + match".to_string())
            }
            HirStmt::UnsafeBlock { body, .. } => {
                self.unsafe_depth += 1;
                self.push_stmt(MirStmt::EffectCheck { effect: "unsafe".to_string() });
                // 关键修复：原来是 `let result = self.build_block(body,
                // shared)?;`——`?` 在 Err 分支直接 return，跳过下面的
                // `self.unsafe_depth -= 1`，这个计数器就再也降不回去了，
                // 会让后续（如果流水线在同一次编译里还会继续构建别的
                // 函数）所有代码都被误判成"处于 unsafe 上下文"。跟前面
                // 几处一样：先存住结果，无论如何先把 depth 降回去，再
                // 决定要不要把错误传播出去。
                let result = self.build_block(body, shared);
                self.unsafe_depth -= 1;
                let result = result?;
                // 注意：UnsafeBlock 本身是一个语句，它不能产生一个
                // MirOperand 结果，所以直接把 build_block 的 BlockResult
                // 原样转发——包括发散信息：unsafe { panic(); } 作为语句
                // 出现时，跟外面不裹 unsafe 是同一个道理，一样要把
                // Diverged 传播上去。
                Ok(result)
            }
        }
    }

    // -------- Expr --------
    // build_expr：求值一个表达式，结果materialize 成一个 MirOperand
    // （字面量直接返回 Constant，否则落进临时变量返回 Copy/Move）。
    // 关键修复：guide.rs 的 build_place 处理 Index 时要调用它算下标
    // 表达式的值（`self.build_expr(index, shared)`）——guide.rs 是跟
    // 这个文件平级的顶层模块，之前这个方法是私有的（E0624），平级模块
    // 访问不到，改成 pub(crate)。
    pub(crate) fn build_expr(&mut self, expr: &HirExpr, shared: &SharedContext) -> Result<ExprResult, String> {
        if let HirExprKind::Literal(lit) = &expr.kind {
            return Ok(Diverging::Value(MirOperand::Constant(lit.clone())));
        }
        // 关键修复：这里原来还有一段跟 build_expr_rvalue 的
        // HirExprKind::Ident 分支一模一样的代码（查作用域、算 SSA、插
        // moved、包成 Move 返回）。两份不会同时执行（这里直接 return），
        // 但逻辑重复——以后 Ident 的语义要是从 Move 改成 Copy，容易
        // 只改一处漏掉另一处。删掉这个特判，让 Ident 落进下面的通用
        // 路径：build_expr_rvalue 对 Ident 本来就返回
        // `MirRvalue::Use(Move(...))`，下面 `if let MirRvalue::Use(operand)
        // = rvalue { return Ok(operand); }` 会原样把这个 operand 展开
        // 返回——跟删之前的特判行为完全一致，只是现在只有一个地方
        // 写着"Ident 该怎么处理"。
        let rvalue = match propagated(self.build_expr_rvalue(expr, shared)?) {
            Ok(v) => v,
            Err(()) => return Ok(Diverging::Diverged),
        };
        // 已经是 Use(operand) 的情况，直接展开，不用画蛇添足再包一层
        // 临时变量。
        let operand = if let MirRvalue::Use(operand) = rvalue {
            operand
        } else {
            let temp = self.new_temp(expr.ty.clone());
            let version = self.ssa_versions.get(&temp).copied().unwrap_or(0);
            let new_version = version + 1;
            self.ssa_versions.insert(temp, new_version);
            let dest = MirPlace::Ssa(SsaLocal { base_id: temp, version: new_version });
            self.push_stmt(MirStmt::Assign { dest, value: rvalue });
            MirOperand::Move(MirPlace::Ssa(SsaLocal { base_id: temp, version: new_version }))
        };
        // 关键修复：这次求值即将把 operand 交给调用方，调用方接下来会
        // 把它嵌进自己的表达式树（当函数实参、当另一个 BinaryOp 的
        // 操作数……），这个 operand 指向的变量的值就已经被这次求值
        // "取走"了。如果这个 operand 恰好是对某个 Ssa 局部变量的
        // Move，就必须在这里登记进 moved——不然这个变量的作用域结束
        // 时，pop_scope 会以为它还没被动过，照常补一条 Drop，对一个
        // 已经被移动走的值重复调用 drop()，生成的 Rust 编译不过。
        //
        // 这条登记原来只在 build_expr_rvalue 的 Ident/FieldAccess/Index
        // 几个分支里各自写了一遍，唯独漏了这里——也就是"上面刚刚新建
        // 的临时变量"（承接 BinaryOp/Call/StructInit/……这些 rvalue 的
        // __tmp）和"rvalue 本来就已经是 Use(Move(Ssa(..))) 的情况"（比如
        // if/match 表达式汇合各分支值之后的 Phi 结果，经上面的
        // `if let MirRvalue::Use(operand) = rvalue` 直接展开出来）这两
        // 类：它们都会被 new_temp 登记进当前作用域的 scope_vars 等着
        // 被 Drop，却从没有在"值被交出去"的这一刻被标记过 moved。对
        // 已经在别处标记过的 ssa（比如这里 operand 其实来自一次 Ident
        // 读取）再插一次是幂等的，不会有副作用。
        self.mark_moved_operand(&operand);
        Ok(Diverging::Value(operand))
    }

    // build_expr_rvalue：跟 build_expr 的区别是不强制把结果落进临时
    // 变量——调用方（比如 Stmt::Let/Stmt::Expr）自己决定怎么处理这个
    // Rvalue（要么自己 Assign 进一个具名变量，要么当 ExprStmt 直接丢弃
    // 结果）。build_expr 内部对非字面量/非标识符的情况也是靠它算出
    // Rvalue，再包一层临时变量。
    fn build_expr_rvalue(&mut self, expr: &HirExpr, shared: &SharedContext) -> Result<RvalueResult, String> {
        match &expr.kind {
            HirExprKind::Literal(lit) => Ok(Diverging::Value(MirRvalue::Use(MirOperand::Constant(lit.clone())))),
            HirExprKind::Ident(name) => {
                let id = self.lookup(name).ok_or_else(|| format!("undefined variable `{}`", name))?;
                let ssa = self.current_ssa(id);
                // 关键修复（找回上一轮的修复）：同 build_expr 里那处。
                self.moved.insert(ssa);
                Ok(Diverging::Value(MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(ssa)))))
            }
            HirExprKind::Sym(name) => {
                // Sym 目前当成一个不可变的具名符号引用处理，精确的
                // 编译期符号求解语义留给 calc.rs/simplify.rs 之后接手。
                //
                // 关键重构（Sym 搬家：MirPlace → MirOperand）：Sym<B>
                // 里的 B 是一个值——一个 usize，只是具体数值要留到 JIT
                // 期才能绑定，跟 Literal::Usize(32) 是同一类东西——不是
                // "能被赋值/取地址的位置"：不能写 `B = 5;`，不能借用，
                // 不需要 Drop。之前它被塞进 MirPlace::Sym、再用
                // MirOperand::Move 包一层去"读"它，是在假装它是内存里的
                // 一个变量；现在 mir.rs 把它挪进了 MirOperand 自己的
                // 变体，不再需要 Move/Copy 这层假装，直接构造。
                Ok(Diverging::Value(MirRvalue::Use(MirOperand::Sym(name.clone()))))
            }
            HirExprKind::BinaryOp { op, left, right } => {
                let l = match propagated(self.build_expr(left, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                let r = match propagated(self.build_expr(right, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                Ok(Diverging::Value(MirRvalue::BinaryOp(op.clone(), l, r)))
            }
            HirExprKind::Unary { op, expr: inner } => {
                let operand = match propagated(self.build_expr(inner, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                Ok(Diverging::Value(MirRvalue::UnaryOp(op.clone(), operand)))
            }
            HirExprKind::Cast { expr: inner, ty } => {
                let operand = match propagated(self.build_expr(inner, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                Ok(Diverging::Value(MirRvalue::Cast(operand, ty.clone())))
            }
            HirExprKind::FieldAccess { .. } | HirExprKind::Index { .. } => {
                // 关键修复（这次重构顺带补上）：guide.rs 的 build_place
                // 现在也走 Diverging 传播（下标表达式本身可能发散，比如
                // `arr[panic()]`），这里跟别处一样，见到 Diverged 就
                // 原样传播。
                let place = match propagated(self.build_place(expr, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                // 关键修复：读一个字段/下标也是对底层变量的一次移动
                // （部分移动），跟裸 Ident 读取一样得登记进 moved——不然
                // pop_scope 会在这个变量离开作用域时照常插一条 Drop，
                // 对一个已经被部分移动过的值调用 drop()，生成的 Rust
                // 编译不过。这是"变量级"的登记（见 MirBuilder.moved
                // 字段上的说明），不是真正的字段级精度，但保证了不生成
                // 编译不过的代码，这是眼下要修的问题。
                if let Some(base) = Self::place_base_ssa(&place) {
                    self.moved.insert(base);
                }
                Ok(Diverging::Value(MirRvalue::Use(MirOperand::Move(place))))
            }
            HirExprKind::Call { qualifier, func, generic_args, args, is_method } => {
                // 1) 裸枚举变体构造（Ok/Err/Some/None 这类不带前缀的
                //    写法）——见 SharedContext.variant_to_enum 的注释。
                if qualifier.is_none() && !*is_method {
                    if let Some(enum_name) = shared.variant_to_enum.get(func).cloned() {
                        let mut mir_args = Vec::new();
                        for a in args {
                            // 关键修复（这次重构顺带补上）：`Some(panic())`
                            // 这种写法——参数本身发散——原来完全没被
                            // 处理过。现在参数发散时直接原样传播，不继续
                            // 构造这个枚举变体，也不再尝试求值后面的
                            // 参数（真实控制流里它们根本不会被求值到）。
                            mir_args.push(match propagated(self.build_call_arg(a, shared)?) {
                                Ok(v) => v,
                                Err(()) => return Ok(Diverging::Diverged),
                            });
                        }
                        // 关键修复：这条路径处理的正是最常见的写法——
                        // `Ok(x)`/`Some(x)` 这种裸调用——而不是走
                        // HirExprKind::EnumVariantConstruction 那条限定
                        // 路径。generic_args 本来就是 HirExprKind::Call
                        // 解构出来的字段，之前这里直接没传，是最容易
                        // 漏掉、也是实际最常触发的一个丢数据点。
                        return Ok(Diverging::Value(MirRvalue::EnumVariantConstruction {
                            enum_name,
                            generic_args: generic_args.clone(),
                            variant_name: func.clone(),
                            args: mir_args,
                        }));
                    }
                }

                // 2) 方法调用：第一个参数是 receiver，其余才是真正的实参。
                if *is_method {
                    if args.is_empty() {
                        return Err("method call requires a receiver".to_string());
                    }
                    let receiver = match propagated(self.build_call_arg(&args[0], shared)?) {
                        Ok(v) => v,
                        Err(()) => return Ok(Diverging::Diverged),
                    };
                    let mut mir_args = Vec::new();
                    for a in &args[1..] {
                        mir_args.push(match propagated(self.build_call_arg(a, shared)?) {
                            Ok(v) => v,
                            Err(()) => return Ok(Diverging::Diverged),
                        });
                    }
                    return Ok(Diverging::Value(MirRvalue::MethodCall {
                        receiver,
                        method: func.clone(),
                        args: mir_args,
                        generic_args: generic_args.clone(),
                    }));
                }

                // 3) 内建关联常量（比如 i128::MAX）：这些在 intrinsic.rs
                //    里注册在 get_constant，不是 get_intrinsic——
                //    get_intrinsic 特意把常量排除在外（见 intrinsic.rs
                //    里 IntrinsicFn/IntrinsicConst 是两个不同枚举的
                //    注释）。不在这里单独拦一道的话，下面第 4 步的
                //    resolve_intrinsic_call 对这类名字必然返回
                //    (false, None)，会一路落进最后"当成普通函数调用"的
                //    通用分支，生成出 `MAX()` 这种把常量当零参函数调用
                //    的假代码——常量不该走 MirRvalue::Call 这条路，
                //    MirOperand::Static 才是它本该落的地方。
                //
                // 关键修复（这次拆分）：常量检测和内建函数检测这两段
                // 逻辑挪到了 intrinsic.rs 的 try_resolve_constant_ref /
                // resolve_intrinsic_call，这里只是委托调用——两个函数
                // 都不需要 &mut MirBuilder，只需要这次调用长什么样和
                // 两个状态位（in_forward/unsafe_depth），所以 intrinsic.rs
                // 不用反过来认识 MirBuilder/mir.rs 的类型，具体原因见
                // 那两个函数上面的注释。
                //
                // 关键重构（Static 搬家：MirPlace → MirOperand）：跟
                // 上面 Sym 那处是同一个道理——i128::MAX 这类内建关联
                // 常量是一个值，不是"位置"，虽然它的字符串碰巧是合法
                // 的 Rust 路径、可以直接裸写，但这是 codegen 的实现
                // 细节，不该让 MIR 层为了这个巧合套一层 Move 假装在读
                // 某个变量。
                if let Some(const_name) = crate::intrinsic::try_resolve_constant_ref(
                    qualifier, func, *is_method, args.len(),
                ) {
                    return Ok(Diverging::Value(MirRvalue::Use(MirOperand::Static(
                        const_name.to_string(),
                    ))));
                }

                // 4) 真正的内建/固有函数：查 intrinsic.rs 的注册表。
                let expr_diverges = expr.ty.is_never();
                let (is_intrinsic, intrinsic_name) = crate::intrinsic::resolve_intrinsic_call(
                    qualifier, func, *is_method, self.in_forward, self.unsafe_depth, expr_diverges,
                )?;

                let full_name = match qualifier {
                    Some(q) => format!("{}::{}", q, func),
                    None => func.clone(),
                };
                let mut mir_args = Vec::new();
                for a in args {
                    mir_args.push(match propagated(self.build_call_arg(a, shared)?) {
                        Ok(v) => v,
                        Err(()) => return Ok(Diverging::Diverged),
                    });
                }
                let call_rvalue = MirRvalue::Call {
                    func: full_name,
                    args: mir_args,
                    is_intrinsic,
                    intrinsic_name,
                    generic_args: generic_args.clone(),
                };

                // ===== 整份重构里唯一真正的发散源头 =====
                // expr_diverges（上面已经算过，顺带喂给了
                // resolve_intrinsic_call 做校验）说明这次调用的返回类型
                // 是 Type::Never——sema 已经确认过，panic、或任何签名
                // 声明为 -> never 的函数。调用本身仍然要真的发生（副作用
                // 不能丢，所以先老老实实 push 一条 ExprStmt），但它绝不
                // 会把控制流交回来：这里直接把 current_block 的终止器标
                // 成 Unreachable（这是一条"没有下一步"的边，跟 Return/
                // Goto 那种"有具体去处"的边不是一回事），然后把 Diverged
                // 报给上一层。从这里往上，每一层调用方（BinaryOp 的另
                // 一个操作数、Call 的下一个实参、if/match 的另一个
                // 分支……）见到 Diverged 直接原样继续往上传，不用各自
                // 重新查一遍 expr.ty.is_never()——发散只在这一个地方被
                // "发现"，其余所有地方只是转手传播。
                if expr_diverges {
                    self.push_stmt(MirStmt::ExprStmt(call_rvalue));
                    self.set_terminator(MirTerminator::Unreachable);
                    return Ok(Diverging::Diverged);
                }
                Ok(Diverging::Value(call_rvalue))
            }
            HirExprKind::EnumVariantConstruction { enum_name, generic_args, variant_name, args } => {
                let mut mir_args = Vec::new();
                for a in args {
                    mir_args.push(match propagated(self.build_call_arg(a, shared)?) {
                        Ok(v) => v,
                        Err(()) => return Ok(Diverging::Diverged),
                    });
                }
                // 关键修复：这里原来用 `..` 把 HIR 节点自带的 generic_args
                // 直接丢掉了——sema 已经推导出 Some(x)/Ok(x) 这类构造具体
                // 实例化成了哪个类型（比如 Option<i32> 的 i32），这份信息
                // 传不到 MIR 层，monomorphic.rs 的泛型枚举单态化就没法做。
                Ok(Diverging::Value(MirRvalue::EnumVariantConstruction {
                    enum_name: enum_name.clone(),
                    generic_args: generic_args.clone(),
                    variant_name: variant_name.clone(),
                    args: mir_args,
                }))
            }
            HirExprKind::EnumVariantAccess { enum_name, variant_name } => {
                // 注：ast::ExprKind::EnumVariantAccess 本身就没有
                // generic_args 字段（裸的 `EnumName::Variant` 访问，不像
                // EnumVariantConstruction 那样在语法层面就带类型实参），
                // 这里传空 Vec 不是漏传，是这条路径目前确实拿不到——如果
                // 以后要支持 `Option::<i32>::None` 这种写法，需要先在
                // ast.rs/hir.rs 里给 EnumVariantAccess 也加上 generic_args。
                Ok(Diverging::Value(MirRvalue::EnumVariantConstruction {
                    enum_name: enum_name.clone(),
                    generic_args: Vec::new(),
                    variant_name: variant_name.clone(),
                    args: Vec::new(),
                }))
            }
            HirExprKind::StructInit { struct_name, generic_args, fields } => {
                let mut mir_fields = Vec::new();
                for (name, e) in fields {
                    mir_fields.push((name.clone(), match propagated(self.build_expr(e, shared)?) {
                        Ok(v) => v,
                        Err(()) => return Ok(Diverging::Diverged),
                    }));
                }
                // 关键修复：同上，原来 `..` 把 generic_args 丢了。
                Ok(Diverging::Value(MirRvalue::StructInit {
                    struct_name: struct_name.clone(),
                    generic_args: generic_args.clone(),
                    fields: mir_fields,
                }))
            }
            HirExprKind::ArrayLiteral(elements) => {
                let mut mir_elements = Vec::new();
                for e in elements {
                    mir_elements.push(match propagated(self.build_expr(e, shared)?) {
                        Ok(v) => v,
                        Err(()) => return Ok(Diverging::Diverged),
                    });
                }
                Ok(Diverging::Value(MirRvalue::ArrayLiteral(mir_elements)))
            }
            HirExprKind::LackSlice(_ty) => Ok(Diverging::Value(MirRvalue::ArrayLiteral(Vec::new()))),
            HirExprKind::Block(b) => {
                match self.build_block(b, shared)? {
                    Diverging::Diverged => Ok(Diverging::Diverged),
                    Diverging::Value(last) => Ok(Diverging::Value(MirRvalue::Use(
                        last.unwrap_or(MirOperand::Constant(crate::ast::Literal::Unit)),
                    ))),
                }
            }
            HirExprKind::UnsafeBlock { body, .. } => {
                self.unsafe_depth += 1;
                self.push_stmt(MirStmt::EffectCheck { effect: "unsafe".to_string() });
                // 关键修复：同 HirStmt::UnsafeBlock 那处——原来的 `?`
                // 会在 Err 时跳过 `self.unsafe_depth -= 1`，让计数器
                // 泄漏。先存住结果，无条件把 depth 降回去，再 `?`。
                let last = self.build_block(body, shared);
                self.unsafe_depth -= 1;
                match last? {
                    Diverging::Diverged => Ok(Diverging::Diverged),
                    Diverging::Value(last) => Ok(Diverging::Value(MirRvalue::Use(
                        last.unwrap_or(MirOperand::Constant(crate::ast::Literal::Unit)),
                    ))),
                }
            }
            HirExprKind::If { kind: _, cond, then_expr, else_expr } => {
                let cond_operand = match propagated(self.build_expr(cond, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                let then_block = self.new_block();
                let else_block = self.new_block();
                let end_block = self.new_block();
                self.set_terminator(MirTerminator::If { cond: cond_operand, then_block, else_block });

                // ---------- Then 分支 ----------
                // 关键重构（Type::Never 去特判化）：原来这里要先查一遍
                // `then_expr.ty.is_never()` 才知道这个分支会不会发散——
                // 从类型层反推控制流。现在 build_expr 自己会在真正发散
                // 的源头（build_expr_rvalue 里那次 Never 调用）把这件事
                // 结构化地带出来，这里直接看 Diverging 的哪个变体就行。
                self.switch_to_block(then_block);
                let then_result = self.build_expr(then_expr, shared)?;
                // 关键修复：then_expr 内部如果自己还有嵌套的 if/match/
                // block（比如 `if a { if b { 1 } else { 2 } } else {
                // ... }`），build_expr 求值完之后 self.current_block 已经
                // 不是这里的 then_block 了，而是那层嵌套控制流自己的
                // end_block——真正该往里面写 Assign、真正该记进 Phi
                // 来源的，是这个"最终块"，不是最初 switch_to_block 进去
                // 的 then_block。原来的实现直接假设两者永远是同一个块，
                // 对不含嵌套控制流的简单分支碰巧是对的，但对这种嵌套
                // 场景会往错误的块（早已经跑完、甚至终止器已经设定好
                // 的 then_block）里补语句、往 Phi 里记一个从来不是真正
                // 前驱的块 id，两者都是错的。
                let then_final_block = self.current_block;

                // ---------- Else 分支 ----------
                self.switch_to_block(else_block);
                let else_result = match else_expr {
                    Some(e) => self.build_expr(e, shared)?,
                    // 没有 else 分支：sema 保证 then 分支是 Unit，这里
                    // 直接给一个 Unit 常量当 else 分支的"值"，不需要真的
                    // build 什么，也不可能发散。
                    None => Diverging::Value(MirOperand::Constant(crate::ast::Literal::Unit)),
                };
                let else_final_block = self.current_block;

                // 关键修复：dest_base 原来在两个分支求值之前就无条件
                // `self.new_temp(...)` 创建好——new_temp 会把它登记进
                // 当前作用域的 scope_vars。如果两个分支都发散（比如
                // `if c { panic() } else { panic() }`），下面两个分支都
                // 走不到"产生值"的路径，dest_base 从头到尾没有任何一条
                // Assign 语句给它赋值，但它已经作为"这个作用域里的一个
                // 变量"被记了下来。等这个作用域将来被 pop_scope /
                // emit_drops_for_scopes 处理时，会发现 dest_base 的初始
                // SSA（version 0）没在 moved 里，照常给它插一条
                // `Drop { place: Ssa(dest_base, 0) }`——生成的 Rust 是对
                // 一个从来没被初始化过的变量调用 drop()，"borrow of
                // possibly-uninitialized variable"，编译不过。
                //
                // 修法：把 dest_base 的创建推迟到"确认至少有一个分支真的
                // 产生了值"之后——两个分支都发散时，压根不调用
                // `new_temp`，从根上不让这个"注册了但从没被赋值"的临时
                // 变量存在。判断"两者是否都发散"需要先把 then_result/
                // else_result 摆在一起 match 一次，所以下面按值（不是按
                // 引用）消费掉它们。
                match (then_result, else_result) {
                    (Diverging::Diverged, Diverging::Diverged) => {
                        // 两个分支都发散：整个 if 表达式也是发散的，原样
                        // 传播给上一层，不创建 dest_base，也不需要切到
                        // end_block——它从此没有任何前驱，是一个真正的
                        // 死块，会在后续的 control::simplify 里被清理掉，
                        // 不需要 mir_builder.rs 这里操心。
                        Ok(Diverging::Diverged)
                    }
                    (then_result, else_result) => {
                        let dest_base = self.new_temp(expr.ty.clone());
                        let mut phi_values = Vec::new();

                        if let Diverging::Value(then_operand) = then_result {
                            self.switch_to_block(then_final_block);
                            let then_ver = self.new_version(dest_base);
                            let then_ssa = SsaLocal { base_id: dest_base, version: then_ver };
                            self.push_stmt(MirStmt::Assign {
                                dest: MirPlace::Ssa(then_ssa),
                                value: MirRvalue::Use(then_operand),
                            });
                            if self.current_block_terminator_is_unset() {
                                self.set_terminator(MirTerminator::Goto(end_block));
                            }
                            phi_values.push((then_final_block, MirOperand::Move(MirPlace::Ssa(then_ssa))));
                        }

                        if let Diverging::Value(else_operand) = else_result {
                            self.switch_to_block(else_final_block);
                            let else_ver = self.new_version(dest_base);
                            let else_ssa = SsaLocal { base_id: dest_base, version: else_ver };
                            self.push_stmt(MirStmt::Assign {
                                dest: MirPlace::Ssa(else_ssa),
                                value: MirRvalue::Use(else_operand),
                            });
                            if self.current_block_terminator_is_unset() {
                                self.set_terminator(MirTerminator::Goto(end_block));
                            }
                            phi_values.push((else_final_block, MirOperand::Move(MirPlace::Ssa(else_ssa))));
                        }

                        // ---------- End 块：插入 Phi ----------
                        self.switch_to_block(end_block);
                        // phi_values 不可能为空：上面已经把"两者都发散"的
                        // 情况单独处理并提前返回，走到这个分支至少有一个
                        // 分支贡献了值。
                        let phi_ver = self.new_version(dest_base);
                        let phi_ssa = SsaLocal { base_id: dest_base, version: phi_ver };
                        self.push_stmt(MirStmt::Assign {
                            dest: MirPlace::Ssa(phi_ssa),
                            value: MirRvalue::Phi { values: phi_values },
                        });
                        Ok(Diverging::Value(MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(phi_ssa)))))
                    }
                }
            }
            // ===== 明确留白，不是漏改 =====
            // Closure 需要处理捕获变量列表和生成匿名结构体，codegen 那边
            // 目前也没有闭包的生成策略；Range 作为独立表达式值（不是被
            // for 直接消费）目前语言里没有真实场景会用到。都是需要独立
            // 设计的功能，硬凑一个实现大概率是错的，不如显式报错等真正
            // 设计。（Match 已经在下面实现了，不再属于这一类留白。）
            HirExprKind::Match { cond, arms } => {
                // cond 只求值一次，落进一个临时变量——不管接下来走哪条
                // 路径，分支/解构里都可能要再读一次这个值（枚举取
                // payload、结构体取字段、数组取下标……），不能在这里就
                // 把它 Move 掉。
                let cond_op = match propagated(self.build_expr(cond, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                let cond_temp = self.new_temp(cond.ty.clone());
                // 关键修复：原来 `self.new_version(cond_temp)` 是直接内嵌
                // 在 push_stmt 参数的结构体字面量里算的——push_stmt 的
                // receiver `self` 先要了一个 &mut self（可变借用 A），
                // 参数里的 `self.new_version(...)` 又要一次 &mut self
                // （可变借用 B），两个可变借用同时活着，是真正的"两次可变
                // 借用重叠"，两阶段借用（two-phase borrow）救不了这种
                // 情况——它只能让"外层 &mut 借用 + 参数里的 &self 借用"
                // 共存（比如下面几处 `self.current_ssa(...)` 那种只读
                // 调用能直接嵌在 push_stmt 参数里，就是靠这个机制），
                // 两个都要 &mut 的调用没法这样叠。
                //
                // 拆成两步：先单独调用 self.new_version 拿到版本号存进
                // 局部变量（这次 &mut 借用调用完就还回去了），再用这个
                // 局部变量构造 SsaLocal 传给 push_stmt，这时候 push_stmt
                // 的 &mut 借用不会跟任何还没算完的借用重叠。
                let cond_ssa = SsaLocal { base_id: cond_temp, version: self.new_version(cond_temp) };
                self.push_stmt(MirStmt::Assign {
                    dest: MirPlace::Ssa(cond_ssa),
                    value: MirRvalue::Use(cond_op),
                });

                // ===== match 该怎么判断走哪条分支，完全取决于 cond 的
                // 类型，三条路径互不相通：
                //   1) 枚举——和类型，天然适合 Switch（原有逻辑，原样
                //      保留在下面）。
                //   2) 整数/布尔/字符——本身已经是标量，字面量模式就是
                //      "跟一个具体常量比较"，同样适合 Switch，只是判
                //      别式不是从聚合值里拆出来的，是把标量本身 Cast
                //      成 i64（复用已有的 MirRvalue::Cast，不新造机制；
                //      bool/char 落地成 Rust 的 `as i64` 天然可行，false/
                //      true 变成 0/1，char 变成码点）。
                //   3) 结构体/元组/数组——都不是和类型，没有"判别式"这
                //      回事，一个 match 表达式对着它们中的一种来匹配，
                //      语义上只可能是"无条件解构绑定"，不存在"没匹配
                //      上、换下一条分支再试"这种可能性——所以完全不碰
                //      Switch，也不需要为了分支而新开 block，当前块直
                //      接往下走，解构出来的绑定就是普通语句。因为没有
                //      "重试下一条"这回事，这三条路径都要求 arms 里有
                //      且只有一条对应的解构 arm（外加可选的 Wildcard 表
                //      示"解构但不关心任何字段/位置"）——多于一条在语义
                //      上是死代码，sema 应该已经保证了这一点，这里只是
                //      不假设、显式检查一遍。
                match &cond.ty {
                    // ---------- 结构体：无条件解构 ----------
                    Type::Struct(struct_name) => {
                        if arms.len() != 1 {
                            return Err(format!(
                                "match 条件是结构体类型 `{}`，结构体不是和类型，不支持多路\
                                 分支——只能有唯一一条解构 arm，实际有 {} 条（这本该在 sema \
                                 的可达性检查里被拦下）",
                                struct_name, arms.len()
                            ));
                        }
                        let arm = &arms[0];
                        self.push_scope();
                        match &arm.pattern {
                            Pattern::Struct { fields, .. } => {
                                for (field_name, binding_name) in fields {
                                    let field_ty = shared
                                        .struct_fields
                                        .get(struct_name)
                                        .and_then(|fs| fs.get(field_name))
                                        .cloned()
                                        .ok_or_else(|| format!(
                                            "struct `{}` has no field `{}`", struct_name, field_name
                                        ))?;
                                    let binding_id = self.new_local(Some(binding_name.clone()), field_ty, false, true);
                                    let binding_ver = self.new_version(binding_id);
                                    self.push_stmt(MirStmt::Assign {
                                        dest: MirPlace::Ssa(SsaLocal { base_id: binding_id, version: binding_ver }),
                                        value: MirRvalue::Use(MirOperand::Move(MirPlace::Field {
                                            base: Box::new(MirPlace::Ssa(self.current_ssa(cond_temp))),
                                            field: field_name.clone(),
                                        })),
                                    });
                                    self.bind(binding_name.clone(), binding_id);
                                    // 关键修复（找回上一轮的修复）：这一
                                    // 步把 cond_temp 的某个字段 Move 走
                                    // 了，是对 cond_temp 的一次部分移动，
                                    // cond_temp 自己也登记在当前作用域
                                    // 里，之后 pop_scope 给它补 Drop 的
                                    // 时候要知道这件事，不然会对一个已经
                                    // 被部分移动过的值调用 drop()。
                                    self.moved.insert(self.current_ssa(cond_temp));
                                }
                            }
                            Pattern::Wildcard => {}
                            other => {
                                return Err(format!(
                                    "match 条件是结构体类型 `{}`，但这个 arm 的模式是 {:?}",
                                    struct_name, other
                                ));
                            }
                        }
                        // 关键重构：结构体/元组/数组这三条"无条件
                        // 解构"路径，处理 arm.expr 之后要做的事逐字节
                        // 相同——见 build_rvalue_expr_and_pop_scope 上面
                        // 的说明，抽成一个方法，这里只是调用点。
                        self.build_rvalue_expr_and_pop_scope(&arm.expr, shared)
                    }

                    // 对应 ast.rs 里新增的 Type::Tuple(Vec<Type>)。
                    Type::Tuple(elem_tys) => {
                        if arms.len() != 1 {
                            return Err(format!(
                                "match 条件是元组类型，元组不是和类型，不支持多路分支——只能\
                                 有唯一一条解构 arm，实际有 {} 条", arms.len()
                            ));
                        }
                        let arm = &arms[0];
                        self.push_scope();
                        match &arm.pattern {
                            Pattern::Tuple(bindings) => {
                                for (i, binding_name) in bindings.iter().enumerate() {
                                    // 这个位置写 None（对应源码里的 `_`）
                                    // 表示不绑定，跳过——跟 Wildcard 整条
                                    // 不绑定是同一个约定，只是落到单个
                                    // 位置上。
                                    let binding_name = match binding_name {
                                        Some(name) => name,
                                        None => continue,
                                    };
                                    let elem_ty = elem_tys.get(i).cloned().ok_or_else(|| {
                                        "tuple pattern has more bindings than the tuple type has elements".to_string()
                                    })?;
                                    let binding_id = self.new_local(Some(binding_name.clone()), elem_ty, false, true);
                                    let binding_ver = self.new_version(binding_id);
                                    self.push_stmt(MirStmt::Assign {
                                        dest: MirPlace::Ssa(SsaLocal { base_id: binding_id, version: binding_ver }),
                                        // 元组的第 i 个位置复用
                                        // MirPlace::Field，用十进制下标
                                        // 字符串当"字段名"——不新增一个
                                        // MirPlace 变体：跟 EnumPayload/
                                        // Index 注释里反复强调的原则一
                                        // 样，能用已有的落点就不另起一
                                        // 个，codegen 落地成 Rust 元组
                                        // 下标 `.0`/`.1` 时，字段名恰好
                                        // 就是要拼的那个数字。
                                        value: MirRvalue::Use(MirOperand::Move(MirPlace::Field {
                                            base: Box::new(MirPlace::Ssa(self.current_ssa(cond_temp))),
                                            field: i.to_string(),
                                        })),
                                    });
                                    self.bind(binding_name.clone(), binding_id);
                                    // 关键修复（找回上一轮的修复）：同结
                                    // 构体解构那处的说明，这也是对
                                    // cond_temp 的一次部分移动。
                                    self.moved.insert(self.current_ssa(cond_temp));
                                }
                            }
                            Pattern::Wildcard => {}
                            other => {
                                return Err(format!(
                                    "match 条件是元组类型，但这个 arm 的模式是 {:?}", other
                                ));
                            }
                        }
                        // 关键重构：结构体/元组/数组这三条"无条件
                        // 解构"路径，处理 arm.expr 之后要做的事逐字节
                        // 相同——见 build_rvalue_expr_and_pop_scope 上面
                        // 的说明，抽成一个方法，这里只是调用点。
                        self.build_rvalue_expr_and_pop_scope(&arm.expr, shared)
                    }
                    Type::Array(elem_ty, _len) => {
                        if arms.len() != 1 {
                            return Err(format!(
                                "match 条件是数组类型，数组不是和类型，不支持多路分支——只能\
                                 有唯一一条解构 arm，实际有 {} 条", arms.len()
                            ));
                        }
                        let arm = &arms[0];
                        self.push_scope();
                        match &arm.pattern {
                            Pattern::Array(bindings) => {
                                for (i, binding_name) in bindings.iter().enumerate() {
                                    let binding_name = match binding_name {
                                        Some(name) => name,
                                        None => continue,
                                    };
                                    let binding_id = self.new_local(Some(binding_name.clone()), (**elem_ty).clone(), false, true);
                                    let binding_ver = self.new_version(binding_id);
                                    self.push_stmt(MirStmt::Assign {
                                        dest: MirPlace::Ssa(SsaLocal { base_id: binding_id, version: binding_ver }),
                                        value: MirRvalue::Use(MirOperand::Move(MirPlace::Index {
                                            base: Box::new(MirPlace::Ssa(self.current_ssa(cond_temp))),
                                            // 关键修复：`Literal::Int` 这
                                            // 个变体在 ast.rs 这一轮改动
                                            // 里已经不存在了（Literal 现在
                                            // 按位宽/符号拆成了
                                            // Int8..Int128/UInt8..UInt128/
                                            // Isize/Usize 一堆变体），原来
                                            // 这行代码引用的是一个已经被
                                            // 删掉的枚举变体，编译不过。
                                            // 数组下标在 Rust 里必须是
                                            // usize（`arr[idx]` 要求
                                            // `idx: usize`），`i` 本来就是
                                            // `enumerate()` 给出的 usize，
                                            // 直接用 Literal::Usize 存，
                                            // 不用再转 i64 又转回来。
                                            index: Box::new(MirOperand::Constant(crate::ast::Literal::Usize(i))),
                                        })),
                                    });
                                    self.bind(binding_name.clone(), binding_id);
                                    // 关键修复（找回上一轮的修复）：同上，
                                    // 数组下标提取也是对 cond_temp 的一次
                                    // 部分移动。
                                    self.moved.insert(self.current_ssa(cond_temp));
                                }
                            }
                            Pattern::Wildcard => {}
                            other => {
                                return Err(format!(
                                    "match 条件是数组类型，但这个 arm 的模式是 {:?}", other
                                ));
                            }
                        }
                        // 关键重构：结构体/元组/数组这三条"无条件
                        // 解构"路径，处理 arm.expr 之后要做的事逐字节
                        // 相同——见 build_rvalue_expr_and_pop_scope 上面
                        // 的说明，抽成一个方法，这里只是调用点。
                        self.build_rvalue_expr_and_pop_scope(&arm.expr, shared)
                    }

                    // ---------- 枚举：原有逻辑，原样保留 ----------
                    Type::Enum(enum_name) => {
                        let cond_enum_name = enum_name.clone();

                        // 1. 计算判别式，存入 disc_temp（用 SSA 版本）
                        let disc_temp = self.new_temp(Type::I64);
                        let disc_ver = self.new_version(disc_temp);
                        let disc_ssa = SsaLocal { base_id: disc_temp, version: disc_ver };
                        self.push_stmt(MirStmt::Assign {
                            dest: MirPlace::Ssa(disc_ssa),
                            value: MirRvalue::Discriminant {
                                value: MirOperand::Copy(MirPlace::Ssa(self.current_ssa(cond_temp))),
                                enum_name: cond_enum_name,
                            },
                        });

                        let end_block = self.new_block();
                        let mut arm_infos: Vec<(usize, &HirExpr, Option<(String, String)>)> = Vec::new();

                        let mut targets = Vec::new();
                        let mut default_block = None;

                        // 2. 收集所有分支
                        for arm in arms {
                            match &arm.pattern {
                                Pattern::EnumVariant { enum_name, variant_name } => {
                                    let idx = *shared
                                        .variant_indices
                                        .get(&(enum_name.clone(), variant_name.clone()))
                                        .ok_or_else(|| format!("unknown enum variant: {}::{}", enum_name, variant_name))? as i64;
                                    let block = self.new_block();
                                    // 关键修复：`Literal::Int` 不存在了
                                    // （见上面数组下标那处的说明），
                                    // disc_temp 声明的是 Type::I64，这里
                                    // 要用跟它匹配的 Int64。
                                    targets.push((Literal::Int64(idx), block));
                                    arm_infos.push((block, &arm.expr, None));
                                }
                                Pattern::EnumVariantWithBinding { enum_name, variant_name, binding } => {
                                    let idx = *shared
                                        .variant_indices
                                        .get(&(enum_name.clone(), variant_name.clone()))
                                        .ok_or_else(|| format!("unknown enum variant: {}::{}", enum_name, variant_name))? as i64;
                                    let block = self.new_block();
                                    targets.push((Literal::Int64(idx), block));
                                    arm_infos.push((block, &arm.expr, Some((binding.clone(), variant_name.clone()))));
                                }
                                Pattern::Wildcard => {
                                    let block = self.new_block();
                                    default_block = Some(block);
                                    arm_infos.push((block, &arm.expr, None));
                                }
                                other => {
                                    return Err(format!(
                                        "match 条件是枚举类型，但这个 arm 的模式是 {:?}——只能用 \
                                         EnumVariant/EnumVariantWithBinding/Wildcard 匹配",
                                        other
                                    ));
                                }
                            }
                        }

                        let default = default_block.unwrap_or_else(|| self.new_block());

                        // 3. 设置 Switch，discr 使用 disc_ssa
                        self.set_terminator(MirTerminator::Switch {
                            discr: MirOperand::Move(MirPlace::Ssa(disc_ssa)),
                            discr_ty: Type::I64,
                            targets,
                            default,
                        });

                        // 4. 处理每个分支，记录每个分支产生的值
                        // 关键修复（dest_base 悬挂）：dest_base 原来在
                        // 第 2 步之前就无条件 `new_temp` 创建好——如果
                        // 所有分支都发散（比如每条 arm 都是
                        // `=> panic()`），它永远不会被赋值，但已经登记
                        // 进了 scope_vars，将来 pop_scope/
                        // emit_drops_for_scopes 会照常给它插一条 Drop，
                        // 对一个从未初始化的变量调用 drop()，编译不过。
                        // 见 If 分支那处一模一样的问题和修法：先只收集
                        // "哪些分支产生了值、产生在哪个块、值是什么"
                        // （不创建 dest_base、不做任何 Assign），等循环
                        // 结束、确认至少有一条分支真的有值之后，才创建
                        // dest_base，回到各自记录的块里补 Assign。
                        let mut value_arms: Vec<(usize, MirOperand)> = Vec::new();

                        for (block, arm_expr, binding_info) in arm_infos {
                            self.switch_to_block(block);
                            self.push_scope();

                            // 处理 binding（如果有）
                            if let Some((binding_name, variant_name)) = &binding_info {
                                let enum_name = shared
                                    .variant_to_enum
                                    .get(variant_name)
                                    .cloned()
                                    .ok_or_else(|| format!("unknown enum variant: {}", variant_name))?;
                                let payload_ty = shared
                                    .variant_payload_types
                                    .get(&(enum_name.clone(), variant_name.clone()))
                                    .cloned()
                                    .ok_or_else(|| format!(
                                        "variant `{}` has no payload but pattern binds `{}`",
                                        variant_name, binding_name
                                    ))?;
                                let binding_id = self.new_local(Some(binding_name.clone()), payload_ty, false, true);
                                let binding_ver = self.new_version(binding_id);
                                self.push_stmt(MirStmt::Assign {
                                    dest: MirPlace::Ssa(SsaLocal { base_id: binding_id, version: binding_ver }),
                                    value: MirRvalue::Use(MirOperand::Move(MirPlace::EnumPayload {
                                        base: Box::new(MirPlace::Ssa(self.current_ssa(cond_temp))),
                                        enum_name,
                                        variant_name: variant_name.clone(),
                                    })),
                                });
                                self.bind(binding_name.clone(), binding_id);
                                // 关键修复（找回上一轮的修复）：取
                                // payload 是对 cond_temp 的一次部分移动，
                                // 跟结构体/元组/数组解构那三处是同一个
                                //道理。只在"确实有 binding"这个分支里
                                // 才标记——纯 EnumVariant（不带 payload
                                // 绑定）和 Wildcard 分支不会碰 cond_temp
                                // 的任何部分，那些路径下 cond_temp 后面
                                // 正常 Drop 就行。
                                self.moved.insert(self.current_ssa(cond_temp));
                            }

                            // 关键重构（Type::Never 去特判化）：原来这里
                            // 要先查一遍 `arm_expr.ty.is_never()` 才知道
                            // 这条分支会不会发散——现在直接看 build_expr
                            // 带出来的 Diverging 变体，发散那支下面什么
                            // 都不用做（终止器已经在更深处设成
                            // Unreachable，没有值可以送进 Phi），也不往
                            // value_arms 里记。
                            //
                            // 关键修复（? 跳过 pop_scope）：原来是
                            // `match self.build_expr(arm_expr, shared)? {
                            // ... }`——`?` 在 Err 分支直接把整个函数
                            // return 掉，不会走到下面任何一支的
                            // `self.pop_scope()`，这一层作用域（binding
                            // 如果有的话）就永远不会被弹掉。跟前面几处
                            // 一样，先存住结果，无论 Err/Diverged/Value
                            // 都先 pop_scope，再决定下一步。
                            let arm_result = self.build_expr(arm_expr, shared);
                            match arm_result {
                                Err(e) => {
                                    self.pop_scope();
                                    return Err(e);
                                }
                                Ok(Diverging::Diverged) => {
                                    self.pop_scope();
                                }
                                Ok(Diverging::Value(operand)) => {
                                    // 关键修复：跟 If 分支那处一样，
                                    // arm_expr 内部如果自己还有嵌套的
                                    // if/match/block，build_expr 求值完
                                    // 之后 self.current_block 已经不是
                                    // 这里的 block 了。真正该记进
                                    // value_arms（进而记进 Phi 来源）的，
                                    // 是这个"最终块"，不是循环变量
                                    // 里最初 switch_to_block 进去的
                                    // block。
                                    let final_block = self.current_block;
                                    self.pop_scope();
                                    if self.current_block_terminator_is_unset() {
                                        self.set_terminator(MirTerminator::Goto(end_block));
                                    }
                                    value_arms.push((final_block, operand));
                                }
                            }
                        }

                        // 5. 切换到 end_block
                        self.switch_to_block(end_block);
                        if value_arms.is_empty() {
                            // 关键修复：原来是 unreachable!()——对
                            // "每个分支都以 panic 结尾"这种合法用户代码
                            // 会让编译器自己崩溃。value_arms 为空恰好就
                            // 说明所有分支都发散了，那么整个 match
                            // 表达式也是发散的，原样传播，不再自己
                            // panic，也不创建 dest_base（见上面的说明）。
                            return Ok(Diverging::Diverged);
                        }

                        // 走到这里，至少有一条分支产生了值，现在才创建
                        // dest_base，回到每条产生值的分支记录的块里补上
                        // Assign——那时候这些块的终止器已经设成了
                        // Goto(end_block)，往它们的语句列表末尾（终止器
                        // 之前）追加一条 Assign 完全合法，语义上就是
                        // "赋值发生在跳转之前"，顺序是对的。
                        let dest_base = self.new_temp(expr.ty.clone());
                        let mut phi_values = Vec::new();
                        for (block, operand) in value_arms {
                            self.switch_to_block(block);
                            let arm_ver = self.new_version(dest_base);
                            let arm_ssa = SsaLocal { base_id: dest_base, version: arm_ver };
                            self.push_stmt(MirStmt::Assign {
                                dest: MirPlace::Ssa(arm_ssa),
                                value: MirRvalue::Use(operand),
                            });
                            phi_values.push((block, MirOperand::Move(MirPlace::Ssa(arm_ssa))));
                        }

                        self.switch_to_block(end_block);
                        let phi_ver = self.new_version(dest_base);
                        let phi_ssa = SsaLocal { base_id: dest_base, version: phi_ver };
                        self.push_stmt(MirStmt::Assign {
                            dest: MirPlace::Ssa(phi_ssa),
                            value: MirRvalue::Phi { values: phi_values },
                        });
                        Ok(Diverging::Value(MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(phi_ssa)))))
                    }

                    // ---------- 整数/布尔/字符：标量字面量，复用 Switch ----------
                    // 新增：这三种类型本身已经是标量，字面量模式就是
                    // "判别式取某个具体的 i64 值就跳到哪个块"，跟枚举
                    // match 走的是同一套 Switch 骨架，唯一的区别是"discr
                    // 怎么算出来"——枚举要先从聚合值里拆判别式
                    // （MirRvalue::Discriminant），标量类型本身已经是
                    // 标量了，不需要拆，只需要统一 Cast 成 i64（复用
                    // 已有的 MirRvalue::Cast，不新造机制）：bool 的 Cast
                    // 落地成 Rust 的 `as i64`（false/true 变成 0/1），
                    // char 落地成 `as i64`（拿到码点），整数自己 Cast
                    // 成 i64 在源类型已经是 i64 时是恒等操作——不管原来
                    // 是 i32/i64/u8/bool/char 里的哪个，都走同一条 Cast
                    // 语句，不用在这里分开重复代码。
                    _ => {
                        // 1. 直接使用 cond_temp 作为判别式，不 Cast
                        let discr_ty = cond.ty.clone();
                        let discr = MirOperand::Copy(MirPlace::Ssa(self.current_ssa(cond_temp)));

                        let end_block = self.new_block();

                        let mut targets = Vec::new();
                        let mut default_block = None;
                        // 关键修复：这个 Vec 原来也叫 branch_results，
                        // 跟第 4 步"记录每个分支产生的 SSA 版本"那个
                        // Vec 撞了同一个名字——第 4 步那句
                        // `let mut branch_results = Vec::new();` 会把
                        // 这里收集到的 (block, &arm.expr) 列表直接遮蔽
                        // 掉，然后下面的 for 循环写的是
                        // `for (block, arm_expr) in arm_results`——
                        // `arm_results` 这个名字在整个函数里根本没有
                        // 声明过，编译不过（E0425）。改名成 arm_infos，
                        // 跟上面枚举 match 分支里同样用途的变量命名
                        // 保持一致，两个 Vec 各自独立，不会互相覆盖。
                        let mut arm_infos: Vec<(usize, &HirExpr)> = Vec::new();

                        // 2. 收集所有分支（targets 存 Literal）
                        for arm in arms {
                            match &arm.pattern {
                                Pattern::IntLiteral(v) => {
                                    let block = self.new_block();
                                    // 关键修复：`Literal::Int` 不存在了，
                                    // 按 discr 的具体类型（i8/u32/...）
                                    // 转换成对应的 Literal 变体——见
                                    // int_literal_for_type 的说明。
                                    let lit = Self::int_literal_for_type(*v, &discr_ty)?;
                                    targets.push((lit, block));
                                    arm_infos.push((block, &arm.expr));
                                }
                                Pattern::BoolLiteral(v) => {
                                    let block = self.new_block();
                                    targets.push((Literal::Bool(*v), block));
                                    arm_infos.push((block, &arm.expr));
                                }
                                Pattern::CharLiteral(c) => {
                                    let block = self.new_block();
                                    targets.push((Literal::Char(*c), block));
                                    arm_infos.push((block, &arm.expr));
                                }
                                Pattern::Wildcard => {
                                    let block = self.new_block();
                                    default_block = Some(block);
                                    arm_infos.push((block, &arm.expr));
                                }
                                other => {
                                    return Err(format!(
                                        "match 条件是标量类型（整数/布尔/字符），但这个 arm 的\
                                         模式是 {:?}——只能用 \
                                         IntLiteral/BoolLiteral/CharLiteral/Wildcard 匹配",
                                        other
                                    ));
                                }
                            }
                        }

                        // 关键修复：跟枚举 match 不同，字面量分支不可能
                        // 自证穷尽（比如 i32 的取值范围远不是几个字面量
                        // 分支能覆盖完的），缺 Wildcard 分支时不能悄悄
                        // 放过——之前这里的错误信息被偷懒地写成裸
                        // `"..."`，改成一句真正能定位问题的话。
                        let default = default_block.ok_or_else(|| {
                            "match 条件是标量类型，但没有 Wildcard 兜底分支——标量字面量\
                             模式不可能自证穷尽，这应该在 sema 阶段就被拦下".to_string()
                        })?;

                        // 3. 设置 Switch
                        self.set_terminator(MirTerminator::Switch {
                            discr,
                            discr_ty,
                            targets,
                            default,
                        });

                        // 4. 处理每个分支
                        // 关键修复（dest_base 悬挂，同枚举 match 那处）：
                        // 先只收集"哪些分支产生了值、产生在哪个块、值是
                        // 什么"，不提前创建 dest_base，等确认至少一条
                        // 分支真的有值之后再创建、再回去补 Assign。
                        let mut value_arms: Vec<(usize, MirOperand)> = Vec::new();

                        for (block, arm_expr) in arm_infos {
                            self.switch_to_block(block);
                            // 关键重构（Type::Never 去特判化）：跟枚举
                            // match 那处同一个道理，直接看 Diverging。
                            // 这里没有 push_scope/pop_scope（标量字面量
                            // 模式不引入绑定），`?` 直接传播错误不会漏掉
                            // 任何清理工作，不用像枚举 match 那样拆开
                            // 处理 Err。
                            match self.build_expr(arm_expr, shared)? {
                                Diverging::Diverged => {}
                                Diverging::Value(operand) => {
                                    // 关键修复：跟枚举 match/If 分支同一个
                                    // 问题——arm_expr 内部如果自己还有
                                    // 嵌套控制流，求值完之后
                                    // self.current_block 已经不是这里的
                                    // block 了，真正该记的是这个"最终块"。
                                    let final_block = self.current_block;
                                    if self.current_block_terminator_is_unset() {
                                        self.set_terminator(MirTerminator::Goto(end_block));
                                    }
                                    value_arms.push((final_block, operand));
                                }
                            }
                        }

                        // 5. End block：Phi
                        self.switch_to_block(end_block);
                        if value_arms.is_empty() {
                            // 关键修复：同枚举 match 那处，原来是
                            // unreachable!()，现在如实传播 Diverged，
                            // 也不创建 dest_base。
                            return Ok(Diverging::Diverged);
                        }

                        let dest_base = self.new_temp(expr.ty.clone());
                        let mut phi_values = Vec::new();
                        for (block, operand) in value_arms {
                            self.switch_to_block(block);
                            let arm_ver = self.new_version(dest_base);
                            let arm_ssa = SsaLocal { base_id: dest_base, version: arm_ver };
                            self.push_stmt(MirStmt::Assign {
                                dest: MirPlace::Ssa(arm_ssa),
                                value: MirRvalue::Use(operand),
                            });
                            phi_values.push((block, MirOperand::Move(MirPlace::Ssa(arm_ssa))));
                        }

                        self.switch_to_block(end_block);
                        let phi_ver = self.new_version(dest_base);
                        let phi_ssa = SsaLocal { base_id: dest_base, version: phi_ver };
                        self.push_stmt(MirStmt::Assign {
                            dest: MirPlace::Ssa(phi_ssa),
                            value: MirRvalue::Phi { values: phi_values },
                        });
                        Ok(Diverging::Value(MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(phi_ssa)))))
                    }
                }
            }

            HirExprKind::Closure { .. } => {
                Err("MIR lowering for Closure: 捕获变量与闭包结构体生成策略还没设计".to_string())
            }
            
            HirExprKind::Range { .. } => {
                Err("MIR lowering for bare Range: Range 目前只应该在 for 循环展开后被消费，独立出现说明 elaborate.rs 该做的展开还没做".to_string())
            }
            // ===== 这条留白已经解决 =====
            // 原来这里记着一句"Never 支持还没延伸到任意子表达式里，比如
            // `let x = 1 + panic();` 这种把 never 表达式嵌在算术/调用
            // 参数等更深层位置的写法"——这正是这一轮 Diverging<T> 重构
            // 要解决的问题：发散不再靠 build_fn/HirStmt::Expr/If/Match
            // 这几个手选位置反查 expr.ty.is_never() 才能发现，而是从
            // build_expr_rvalue 里唯一真正的发散源头（一次 Never 调用）
            // 开始，通过返回值结构化地一路往上传播——BinaryOp 的操作数、
            // 调用的实参、struct 字段、数组元素……每一层看到 Diverged
            // 就直接原样转手，不需要各自再查一次类型。`let x = 1 +
            // panic();` 现在会被正确处理：BinaryOp 的右操作数发散，
            // 整个 BinaryOp 发散，HirStmt::Let 见到 Diverged 直接传播，
            // 不创建 x 这个变量、不生成错误的 Assign。
        }
    }

    fn build_call_arg(&mut self, arg: &HirCallArg, shared: &SharedContext) -> Result<ExprResult, String> {
        match arg {
            HirCallArg::Positional(e) => self.build_expr(e, shared),
            HirCallArg::Named(_, e) => self.build_expr(e, shared),
        }
    }

    // 关键重构：build_expr_rvalue 里处理 Match 时，结构体/元组/数组
    // 这三种"模式无条件解构成功、不需要 Switch/多分支"的路径，各自
    // 求值完 arm.expr 之后要做的事逐字节相同——不管 arm.expr 发不发散
    // 都要 pop_scope（进这个分支之前，各自的调用点都为模式绑定的变量
    // push_scope 过一次，这里得配对弹出去），然后把结果包成
    // `Result<RvalueResult, String>` 交回去。三处重复的代码抽成这一个
    // 方法，调用点只剩一行。
    fn build_rvalue_expr_and_pop_scope(&mut self, expr: &HirExpr, shared: &SharedContext) -> Result<RvalueResult, String> {
        // 关键修复：原来是 `match self.build_expr(expr, shared)? { ... }`
        // ——`?` 在 Err 分支直接把 Err(String) 从这个函数整个 return
        // 出去，根本没走到下面两条分支里的 `self.pop_scope()`。这个
        // helper 存在的唯一理由就是"pop_scope 不管求值成功还是发散都
        // 必须执行"，结果这份职责本身在抽出来的实现里被 `?` 绕过去了——
        // 抽象完了但没把被抽象掉的问题一起修掉，比不抽更糟：调用点看到
        // 一个专门叫这个名字的函数，会理所当然地以为这件事已经处理好了。
        // 现在把 `build_expr` 的结果先存起来（不提前用 `?` 短路），无
        // 论如何都先 pop_scope，再对存好的结果做 `?`/match。
        let result = self.build_expr(expr, shared);
        self.pop_scope();
        match result? {
            Diverging::Diverged => Ok(Diverging::Diverged),
            Diverging::Value(operand) => Ok(Diverging::Value(MirRvalue::Use(operand))),
        }
    }

    // ===== 循环头 Phi =====
    //
    // 背景：MirBuilder 的 SSA 是直线代码版——同一个 base_id 每被
    // Assign 一次就 new_version，current_ssa(base_id) 返回"最后一次
    // 赋值对应的版本"。这在顺序执行的代码里没问题，但循环有一条
    // "回边"（body 执行完跳回循环头），循环第二轮开始时，循环头里读到
    // 的 base_id 应该是"上一轮循环体结束时的值"，不是"进入循环之前的
    // 值"——直线 SSA 表达不出"这个值取决于我们是第一次到这里、还是
    // 绕了一圈回来的"这种二义性，必须在循环头显式插入 Phi 节点。这也
    // 是为支配树、GVN、SCCP（稀疏条件常量传播）、聚合常量折叠这些后续
    // 分析做准备——它们都要求输入是规范的 SSA 形式。

    // 预扫描循环体，找出被 Assign 语句直接修改过的变量名——这些变量
    // 才需要在循环头插 Phi；从未在循环体内被修改的变量，循环体每一轮
    // 读到的都是循环外的同一个值，不存在"这一轮该读哪个版本"的二义性，
    // 不需要 Phi。
    fn collect_modified_vars_block(block: &HirBlock, out: &mut HashSet<String>) {
        for stmt in &block.stmts {
            Self::collect_modified_vars_stmt(stmt, out);
        }
    }

    fn collect_modified_vars_stmt(stmt: &HirStmt, out: &mut HashSet<String>) {
        match stmt {
            HirStmt::Assign { target, .. } => {
                // 只认 Ident 目标——Field/Index 目标现在 mir_builder 不
                // 递增 SSA 版本（见 HirStmt::Assign 分支：只有目标是
                // MirPlace::Ssa 时才 new_version，Field/Index 保留原样），
                // 它们不参与循环 Phi，这跟当前 SSA 的粒度保持一致。
                if let HirExprKind::Ident(name) = &target.kind {
                    out.insert(name.clone());
                }
            }
            HirStmt::Let { .. } => {
                // let 引入新局部变量，不修改外层已有的，不算。
            }
            HirStmt::While { cond, body, .. } => {
                Self::collect_modified_vars_expr(cond, out);
                Self::collect_modified_vars_block(body, out);
            }
            // 关键：递归进所有子块，包括嵌套循环体。内层循环里对 i 的
            // 赋值同样会改变外层循环下一次迭代看到的 i，外层循环头也
            // 需要给 i 建 Phi。
            HirStmt::Loop { body, .. } => {
                Self::collect_modified_vars_block(body, out);
            }
            HirStmt::For { body, .. } => {
                // 正常情况下不会走到这里——elaborate.rs 在 MIR 构建之前
                // 已经把 for 展开成 loop + match（build_stmt 里
                // HirStmt::For 分支同样的说明）。真出现说明流水线顺序
                // 有问题，但递归进去本身没有坏处，不特殊处理、不报错，
                // 跟 build_stmt 那边"防御性报错"的取舍不同——这里只是
                // 收集信息，不是真正构建 MIR，没有必要在这个信息收集
                // 阶段就终止整个编译。
                Self::collect_modified_vars_block(body, out);
            }
            HirStmt::UnsafeBlock { body, .. } => {
                Self::collect_modified_vars_block(body, out);
            }
            HirStmt::Break { .. } => {}
            HirStmt::Return { expr, .. } => {
                if let Some(e) = expr {
                    Self::collect_modified_vars_expr(e, out);
                }
            }
            HirStmt::Expr { expr, .. } => {
                Self::collect_modified_vars_expr(expr, out);
            }
        }
    }

    // 这门语言里 if/match/block 都是表达式，不是独立的语句节点
    // （`if`作为语句出现时，落在 HirStmt::Expr 里包着一个
    // HirExprKind::If），所以"循环体里有没有条件赋值"这件事，得从
    // HirStmt::Expr 往下钻进表达式树，找出其中可能包着 HirBlock（也就
    // 可能包着 Assign 语句）的那几种：Block/UnsafeBlock 直接包一个
    // HirBlock；If 的 then/else 分支、Match 的每个 arm 是 HirExpr，
    // 但通常又是 HirExprKind::Block(...)，一路递归下去自然会展开到
    // 真正的 HirBlock。
    //
    // 覆盖范围说明：这里只递归 If/Match/Block/UnsafeBlock 这几种
    // "长得像控制流"的位置，没有递归 BinaryOp/Call 参数/StructInit
    // 字段这些位置——理论上 `1 + { x = 5; x }` 这种把赋值藏进算术
    // 子表达式的写法也能把一个 HirBlock 嵌进任意表达式位置，但这在
    // 实际代码里极其罕见，穷举所有表达式位置递归下去要把这个函数
    // 变成一整套通用 HIR walker，性价比很低。已知的不完整之处，
    // 先记在这里。
    fn collect_modified_vars_expr(expr: &HirExpr, out: &mut HashSet<String>) {
        match &expr.kind {
            HirExprKind::Block(block) => {
                Self::collect_modified_vars_block(block, out);
            }
            HirExprKind::UnsafeBlock { body, .. } => {
                Self::collect_modified_vars_block(body, out);
            }
            HirExprKind::If { kind: _, cond, then_expr, else_expr } => {
                Self::collect_modified_vars_expr(cond, out);
                Self::collect_modified_vars_expr(then_expr, out);
                if let Some(e) = else_expr {
                    Self::collect_modified_vars_expr(e, out);
                }
            }
            HirExprKind::Match { cond, arms } => {
                Self::collect_modified_vars_expr(cond, out);
                for arm in arms {
                    Self::collect_modified_vars_expr(&arm.expr, out);
                }
            }
            // 其余表达式种类本身不可能直接携带 HirStmt（它们的子节点
            // 要么是别的表达式，要么是纯数据如字符串/类型），因此不
            // 可能包含 Assign 语句，不需要递归。穷尽列出，不用 `_`
            // 兜底——新增一种 HirExprKind 变体时，这里会因为
            // "non-exhaustive match" 编译不过，逼着回来决定它要不要
            // 参与这份扫描，而不是被通配符悄悄吞掉。
            HirExprKind::Literal(_)
            | HirExprKind::Ident(_)
            | HirExprKind::Sym(_)
            | HirExprKind::BinaryOp { .. }
            | HirExprKind::Unary { .. }
            | HirExprKind::Cast { .. }
            | HirExprKind::FieldAccess { .. }
            | HirExprKind::Index { .. }
            | HirExprKind::Call { .. }
            | HirExprKind::EnumVariantConstruction { .. }
            | HirExprKind::EnumVariantAccess { .. }
            | HirExprKind::StructInit { .. }
            | HirExprKind::ArrayLiteral(_)
            | HirExprKind::LackSlice(_)
            | HirExprKind::Closure { .. }
            | HirExprKind::Range { .. } => {}
        }
    }

    /// 在当前块（循环头）最前面插入一组 Phi。
    /// `preheader_block` 是进入循环之前所在的那个块——Phi 的第一个
    /// incoming。调用时机很重要：必须在循环头块（cond_block 或
    /// While 没有独立 cond_block 时的 body_block）刚 switch_to_block
    /// 进去、还没塞任何语句的时候调用——插入的 Phi 语句直接 push 到
    /// 块尾，此时块尾就是块头。
    fn insert_loop_header_phis(
        &mut self,
        preheader_block: usize,
        body: &HirBlock,
    ) -> Vec<LoopPhiInfo> {
        let mut modified_names: HashSet<String> = HashSet::new();
        Self::collect_modified_vars_block(body, &mut modified_names);

        let header_block = self.current_block;
        let mut phis = Vec::new();

        for name in &modified_names {
            let Some(base_id) = self.lookup(name) else { continue };
            let entry_ssa = self.current_ssa(base_id); // 进入循环前的版本
            let phi_version = self.new_version(base_id); // 分配新版本——
            // 之后循环体内部读 base_id，current_ssa 会返回这个新版本，
            // 也就是 phi_ssa，这正是我们想要的：循环体内的读取，不管
            // 是第一轮还是绕回来的后续轮次，统一读这一个 Phi 汇合出来
            // 的值。
            let phi_ssa = SsaLocal { base_id, version: phi_version };

            let stmt_index = self.blocks[header_block].stmts.len();
            // 直接 push 会加到末尾；既然现在 header_block 里除了 Phi
            // 还没别的语句（求值 cond 的语句在下面才 push，Loop 分支
            // 循环体自己的语句也还没开始构建），push 即等价于放到
            // 开头。多个 Phi 之间顺序无关。
            self.push_stmt(MirStmt::Assign {
                dest: MirPlace::Ssa(phi_ssa),
                value: MirRvalue::Phi {
                    values: vec![
                        (preheader_block, MirOperand::Move(MirPlace::Ssa(entry_ssa))),
                        // 第二个 incoming（回边）暂时留空，等循环体
                        // 构建完，由 finalize_loop_header_phis 回填。
                    ],
                },
            });

            phis.push(LoopPhiInfo { header_block, stmt_index, base_id });
        }
        phis
    }

    /// 循环体构建完后，回填每条 Phi 的第二个 incoming（回边）。只有
    /// 循环体真的会绕回循环头时才调用这个函数（调用点：While/Loop
    /// 分支里，只在 body 正常掉出尾部、且尾部所在块此刻还没有真终止器
    /// 时才调用——没有回边就不存在第二个 incoming，Phi 保持单路，交给
    /// control::simplify 的 simplify_phi 折叠回普通 Use）。
    fn finalize_loop_header_phis(&mut self, phis: &[LoopPhiInfo], back_edge_block: usize) {
        for info in phis {
            let back_ssa = self.current_ssa(info.base_id);
            if let MirStmt::Assign { value: MirRvalue::Phi { values }, .. }
                = &mut self.blocks[info.header_block].stmts[info.stmt_index]
            {
                values.push((back_edge_block, MirOperand::Move(MirPlace::Ssa(back_ssa))));
            }
        }
    }
}