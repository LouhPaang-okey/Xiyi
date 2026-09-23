// borrow.rs
//
// 重构说明（破釜沉舟版）：这一版彻底废除了"借用由 MirStmt::Drop 释放"
// 的词法借用模型。旧的实现有一个根本性的架构缺陷：借用状态只会在
// `MirRvalue::Ref` 处新增、在 `MirStmt::Drop` 处删除，这意味着一个
// `&x` 从被创建开始，会一直被算作"借用中"，直到 x 自己走完作用域被
// Drop 才释放——这是词法作用域语义，会对大量合法程序过度报错
// （Rust 当初也因此才去做 NLL 和后来的 Polonius）。
//
// 这一版改成 NLL（Non-Lexical Lifetimes）风格的正确模型：
//
//   1. 先做后向 liveness 分析，算出每个变量的"最后一次使用"落在哪条
//      语句之后（精确到语句粒度，不是块粒度）。
//   2. 一个借用（loan）的活跃期 = 从它被创建的那条 `&`/`&mut`
//      语句开始，到它绑定的那个引用变量（borrow_var）最后一次被使用
//      为止。borrow_var 不再 live，这个 loan 就自然消亡，跟原变量的
//      作用域毫无关系。
//   3. 借用冲突检查在前向扫描里做：任一条语句处，活跃的 loan 就是
//      "已经创建、且 borrow_var 仍然 live"的那些，据此检查读/写/再次
//      借用是否越界。
//   4. 初始化检查（use-of-uninitialized / move / drop）彻底独立出来，
//      作为一条前向 must（全称）数据流，跟借用检查解耦，不再挤在
//      同一个 HashMap 状态里互相干扰。
//
// 和旧版相比，借用冲突还精确到了 place 路径：`&mut p.x` 之后读 `p.y`
// 不再被误判为冲突（两个不同字段），只有 `p.x`（或被借位置的任何
// 前缀/扩展重叠）才会跟这个借用冲突。这是 NLL 之外的另一层精度提升。
//
// 这一版直接解决了旧文件开头那条警告——旧检查器整套能不能用，系在
// mir_builder.rs 会不会真的生成 MirStmt::Drop 这一件事上。现在借用的
// 生死完全由 liveness 决定，跟 Drop 没有任何关系；Drop 只在初始化检查
// 里起"消费后变回未初始化"的辅助作用，就算 mir_builder.rs 从来不生成
// Drop，借用检查这条主线依然完整可用，顶多是少一次"不能 drop 被借用
// 的值"的额外检查，不会像旧版那样整体失效。
//
// 已知边界（如实标注，不做假装）：
//   - Deref 读写通过 place 的 root 天然区分了"引用变量"和被借的
//     "目标"，不会误报，但也因此不做别名依赖分析（`*p` 写 x 的同时又
//     直接写 x 这种双重路径冲突暂不拦截，那是 Polonius 的 subscription
//     级别工作）。
//   - 引用的再借用/转移（`let r2 = r;` 把一个 `&mut T` move 出去之后
//     借用的活性本应跟着 r2 走）暂不追踪：loan 的活性只挂到最初绑定
//     的那个 borrow_var 上，borrow_var 一死 loan 就消亡，r2 这条别名
//     链上的后续冲突会漏报。这是 reborrow 追踪，留到后续。
//   - 部分移动（move 一个字段 p.x）暂按"不改变 p 的初始化状态"处理，
//     整体 move 才是会置未初始化的那一类。后续补 partial move。
//   - 两阶段借用（two-phase borrow，例如 v.push(v.len()) 里 push 的
//     &mut self 先占位后激活）暂不实现，这类写法先报错。
//   - Loan 的 borrow_var 只按 base_id 区分，不看 SSA 版本号：如果同一个
//     引用变量在不同分支里绑定了不同的借用目标（`if c { r = &x } else
//     { r = &y }`），合并点之后两条 loan 会被同时当成"可能活跃"一起
//     参与冲突检查——比实际运行时路径更保守，可能在个别场景下多拦一次
//     本该合法的操作，但不会漏检（安全，只是不够精确）。真要按分支精
//     确区分需要按 (base_id, version) 而不是单纯 base_id 追踪 loan，
//     留到以后。

use crate::mir::*;
use std::collections::{BTreeSet, VecDeque};

pub struct BorrowChecker;

// ---- 借用种类 ----
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoanKind {
    Shared,
    Mut,
}

impl std::fmt::Display for LoanKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoanKind::Shared => write!(f, "shared(immutably)"),
            LoanKind::Mut => write!(f, "mutably"),
        }
    }
}

// ---- 一次借用（loan）----
#[derive(Debug, Clone)]
struct Loan {
    /// 借用结果绑定到的引用变量（borrow_var 的 base_id）。
    borrow_var: usize,
    /// 被借用的目标 place（保留完整路径，做字段级冲突判定用）。
    place: MirPlace,
    kind: LoanKind,
}

// ---- place 的展开形式：根 + 投影链（从根到叶）----
// 关键修复（Sym/Static 搬家）：mir.rs 把 MirPlace::Static 挪去了
// MirOperand（见 mir.rs 里 MirPlace 定义处的说明）——MirPlace 现在只有
// Ssa/Field/Index/Deref/EnumPayload 五种真正的"位置"，全部递归到底都
// 落在 Ssa 上，不会再有"根是一个静态常量路径"这种情况。Root::Static
// 这个变体因此变成了永远不会被构造的死变体，删掉；Root 目前只剩
// Local 一种，暂时保留 enum 的形状（没有收窄成裸 usize），是为了不
// 牵动 places_overlap 里 `ra != rb` 这处比较逻辑和 flatten 的返回类型
// ——这是一次独立的、比这次修改范围更大的简化，这次不做。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Root {
    Local(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Proj {
    /// 字段投影，只有字段名不同才不重叠。
    Field(String),
    /// Deref 投影。保守处理：一旦出现，之后的路径不做精确区分。
    Deref,
    /// 不透明投影：Index / EnumPayload，保守起见按"之后可能重叠"处理。
    Opaque,
}

impl BorrowChecker {
    // ============================================================
    // 入口
    // ============================================================
    pub fn check(program: &MirProgram) -> Result<(), String> {
        for f in &program.fns {
            Self::check_fn(f)?;
        }
        Ok(())
    }

    fn check_fn(f: &MirFn) -> Result<(), String> {
        if f.body.blocks.is_empty() {
            return Ok(());
        }

        // 后向 liveness：live_after[block][i] = block 里第 i 条语句执行
        // 完之后仍然活跃的 base_id 集合。
        let (live_after, live_out) = Self::compute_liveness(&f.body);

        // 前向初始化检查（独立 pass）。
        Self::init_check(f, &live_after)?;

        // 前向借用冲突检查（独立 pass，用 live_after 判定活跃 loan）。
        Self::borrow_check(f, &live_after, &live_out)?;

        Ok(())
    }

    // ============================================================
    // 后向 liveness
    // ============================================================
    fn compute_liveness(body: &MirBody) -> (Vec<Vec<BTreeSet<usize>>>, Vec<BTreeSet<usize>>) {
        let n = body.blocks.len();
        let mut live_out: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        let mut live_after: Vec<Vec<BTreeSet<usize>>> = body
            .blocks
            .iter()
            .map(|b| vec![BTreeSet::new(); b.stmts.len()])
            .collect();

        let preds = Self::compute_preds(body);

        // 标准后向 worklist：每个块被重新处理时，用它当前的 live_out 加
        // 上终结指令的 uses 作为起点，逆序遍历语句逐条把 live 集合往前
        // 推；块入口得到的 live_in 再并进所有前驱的 live_out。
        let mut queue: VecDeque<usize> = (0..n).collect();
        let mut in_queue = vec![true; n];

        while let Some(block_id) = queue.pop_front() {
            in_queue[block_id] = false;
            let block = &body.blocks[block_id];

            // live 从"块结束时 + 终结指令读到的变量"起步。
            let mut live = live_out[block_id].clone();
            for u in Self::term_uses(&block.terminator) {
                live.insert(u);
            }

            for i in (0..block.stmts.len()).rev() {
                live_after[block_id][i] = live.clone();
                let uses = Self::stmt_uses(&block.stmts[i]);
                let defs = Self::stmt_defs(&block.stmts[i]);
                for d in &defs {
                    live.remove(d);
                }
                for u in &uses {
                    live.insert(*u);
                }
            }

            // 此刻 live 就是 live_in。
            for &pred in &preds[block_id] {
                let before = live_out[pred].len();
                live_out[pred].extend(live.iter().copied());
                if live_out[pred].len() != before && !in_queue[pred] {
                    in_queue[pred] = true;
                    queue.push_back(pred);
                }
            }
        }

        (live_after, live_out)
    }

    fn compute_preds(body: &MirBody) -> Vec<Vec<usize>> {
        let n = body.blocks.len();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (src, block) in body.blocks.iter().enumerate() {
            for succ in Self::successors(block) {
                if succ < n {
                    preds[succ].push(src);
                }
            }
        }
        preds
    }

    fn successors(block: &MirBlock) -> Vec<usize> {
        match &block.terminator {
            MirTerminator::Goto(t) => vec![*t],
            MirTerminator::If { then_block, else_block, .. } => vec![*then_block, *else_block],
            MirTerminator::Switch { targets, default, .. } => {
                let mut s: Vec<usize> = targets.iter().map(|(_, t)| *t).collect();
                s.push(*default);
                s
            }
            _ => Vec::new(),
        }
    }

    // ============================================================
    // uses / defs 提取（base_id 粒度，供 liveness 与 init 使用）
    // ============================================================
    fn stmt_defs(stmt: &MirStmt) -> BTreeSet<usize> {
        let mut defs = BTreeSet::new();
        if let MirStmt::Assign { dest: MirPlace::Ssa(s), .. } = stmt {
            defs.insert(s.base_id);
        }
        defs
    }

    fn stmt_uses(stmt: &MirStmt) -> BTreeSet<usize> {
        let mut uses = BTreeSet::new();
        match stmt {
            MirStmt::Assign { dest, value } => {
                Self::collect_rvalue_uses(value, &mut uses);
                // 写一个带投影的 place（p.x / p[i] / *p）要先读 base 再改，
                // 所以把 dest 的 base 也算作 use；纯 Ssa dest 是全新 def，
                // 不算 use。
                match dest {
                    MirPlace::Ssa(_) => {}
                    _ => Self::collect_place_uses(dest, &mut uses),
                }
            }
            MirStmt::ExprStmt(value) => Self::collect_rvalue_uses(value, &mut uses),
            MirStmt::Drop { place } => Self::collect_place_uses(place, &mut uses),
            MirStmt::SetMetadata { place, .. } => Self::collect_place_uses(place, &mut uses),
            MirStmt::EffectCheck { .. } => {}
        }
        uses
    }

    fn term_uses(term: &MirTerminator) -> BTreeSet<usize> {
        let mut uses = BTreeSet::new();
        match term {
            MirTerminator::If { cond, .. } => Self::collect_operand_uses(cond, &mut uses),
            MirTerminator::Return(Some(op)) => Self::collect_operand_uses(op, &mut uses),
            MirTerminator::Switch { discr, .. } => Self::collect_operand_uses(discr, &mut uses),
            _ => {}
        }
        uses
    }

    fn collect_place_uses(place: &MirPlace, out: &mut BTreeSet<usize>) {
        match place {
            MirPlace::Ssa(s) => {
                out.insert(s.base_id);
            }
            MirPlace::Field { base, .. } => Self::collect_place_uses(base, out),
            MirPlace::Index { base, index } => {
                Self::collect_place_uses(base, out);
                Self::collect_operand_uses(index, out);
            }
            MirPlace::Deref(base) => Self::collect_place_uses(base, out),
            MirPlace::EnumPayload { base, .. } => Self::collect_place_uses(base, out),
        }
    }

    fn collect_operand_uses(op: &MirOperand, out: &mut BTreeSet<usize>) {
        match op {
            MirOperand::Copy(p) | MirOperand::Move(p) => Self::collect_place_uses(p, out),
            MirOperand::Constant(_) => {}
            // 关键修复（Sym/Static 搬家）：mir.rs 把这两个变体从
            // MirPlace 挪去了 MirOperand（它们是值，不是"位置"，见
            // mir.rs 里 MirPlace 定义处的说明）。它们不对应任何局部
            // 变量，跟原来 MirPlace::Static 分支的处理是同一个道理：
            // 不产生任何"用到了哪个 base_id"的信息。
            MirOperand::Sym(_) => {}
            MirOperand::Static(_) => {}
        }
    }

    fn collect_rvalue_uses(rv: &MirRvalue, out: &mut BTreeSet<usize>) {
        match rv {
            MirRvalue::Use(op) => Self::collect_operand_uses(op, out),
            MirRvalue::BinaryOp(_, l, r) => {
                Self::collect_operand_uses(l, out);
                Self::collect_operand_uses(r, out);
            }
            MirRvalue::UnaryOp(_, op) => Self::collect_operand_uses(op, out),
            MirRvalue::Cast(op, _) => Self::collect_operand_uses(op, out),
            MirRvalue::Call { args, .. } => {
                for a in args {
                    Self::collect_operand_uses(a, out);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::collect_operand_uses(receiver, out);
                for a in args {
                    Self::collect_operand_uses(a, out);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::collect_operand_uses(op, out);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for a in args {
                    Self::collect_operand_uses(a, out);
                }
            }
            MirRvalue::Ref { place, .. } => Self::collect_place_uses(place, out),
            MirRvalue::ArrayLiteral(elems) => {
                for e in elems {
                    Self::collect_operand_uses(e, out);
                }
            }
            MirRvalue::Discriminant { value, .. } => Self::collect_operand_uses(value, out),
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::collect_operand_uses(op, out);
                }
            }
        }
    }

    // ============================================================
    // 初始化检查（前向 must）
    // ============================================================
    fn init_check(f: &MirFn, live_after: &[Vec<BTreeSet<usize>>]) -> Result<(), String> {
        let body = &f.body;
        let n = body.blocks.len();
        let mut in_init: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for local in &body.locals {
            if local.is_param {
                in_init[0].insert(local.id);
            }
        }
        let mut seeded = vec![false; n];
        seeded[0] = true;

        // must（全称）分析：前向传播，多前驱取交集。
        let mut changed = true;
        while changed {
            changed = false;
            for block_id in 0..n {
                let block = &body.blocks[block_id];
                let mut state = in_init[block_id].clone();

                for stmt in &block.stmts {
                    Self::exec_init_stmt(stmt, &mut state, f)?;
                }
                // 终结指令的 uses 也必须在块内已初始化。
                for u in Self::term_uses(&block.terminator) {
                    if !state.contains(&u) {
                        return Err(format!(
                            "use of possibly-uninitialized variable `{}`",
                            Self::get_var_name(f, u)
                        ));
                    }
                }

                for succ in Self::successors(block) {
                    if succ >= n {
                        continue;
                    }
                    if Self::merge_into_init(&mut in_init[succ], &state, &mut seeded[succ]) {
                        changed = true;
                    }
                }
            }
        }

        let _ = live_after; // init 不依赖 liveness，签名保留以便将来做须活跃性判定的检查。
        Ok(())
    }

    fn exec_init_stmt(stmt: &MirStmt, state: &mut BTreeSet<usize>, f: &MirFn) -> Result<(), String> {
        // 1. 这条语句读到的所有变量都必须已初始化。
        for u in Self::stmt_uses(stmt) {
            if !state.contains(&u) {
                return Err(format!(
                    "use of possibly-uninitialized variable `{}`",
                    Self::get_var_name(f, u)
                ));
            }
        }

        // 2. Move 副作用：整体移动一个非 Copy 变量后，它变未初始化。
        let moves = Self::moves_in_stmt(stmt);
        for base in moves {
            let ty = Self::get_var_type(f, base);
            if !ty.is_copy() {
                state.remove(&base);
            }
        }

        // 3. 定义 / 销毁。
        match stmt {
            MirStmt::Assign { dest: MirPlace::Ssa(s), .. } => {
                state.insert(s.base_id);
            }
            MirStmt::Drop { place } => {
                if let Some(base) = Self::place_base_id(place) {
                    state.remove(&base);
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// 收集语句里所有"整体 Move 一个局部变量"的 base_id。
    /// 部分移动（Move 一个字段，如 `let v = p.x;`）这里不收集，见文件头
    /// "已知边界"。
    fn moves_in_stmt(stmt: &MirStmt) -> Vec<usize> {
        let mut out = Vec::new();
        match stmt {
            MirStmt::Assign { value, .. } | MirStmt::ExprStmt(value) => {
                Self::collect_moves_in_rvalue(value, &mut out);
            }
            _ => {}
        }
        out
    }

    fn collect_moves_in_rvalue(rv: &MirRvalue, out: &mut Vec<usize>) {
        match rv {
            MirRvalue::Use(op) => Self::collect_moves_in_operand(op, out),
            MirRvalue::BinaryOp(_, l, r) => {
                Self::collect_moves_in_operand(l, out);
                Self::collect_moves_in_operand(r, out);
            }
            MirRvalue::UnaryOp(_, op) => Self::collect_moves_in_operand(op, out),
            MirRvalue::Cast(op, _) => Self::collect_moves_in_operand(op, out),
            MirRvalue::Call { args, .. } => {
                for a in args {
                    Self::collect_moves_in_operand(a, out);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::collect_moves_in_operand(receiver, out);
                for a in args {
                    Self::collect_moves_in_operand(a, out);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::collect_moves_in_operand(op, out);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for a in args {
                    Self::collect_moves_in_operand(a, out);
                }
            }
            MirRvalue::ArrayLiteral(elems) => {
                for e in elems {
                    Self::collect_moves_in_operand(e, out);
                }
            }
            MirRvalue::Discriminant { value, .. } => Self::collect_moves_in_operand(value, out),
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::collect_moves_in_operand(op, out);
                }
            }
            MirRvalue::Ref { .. } => {}
        }
    }

    fn collect_moves_in_operand(op: &MirOperand, out: &mut Vec<usize>) {
        if let MirOperand::Move(place) = op {
            if let MirPlace::Ssa(s) = place {
                out.push(s.base_id);
            }
        }
    }

    /// must 交集合并。seeded 含义：一个块"还没合并过任何前驱"时，第一
    /// 个前驱直接整体拷贝当作近似值（不能用空表做 AND，否则第一个前驱
    /// 的贡献会被空表吃掉）；从第二个前驱开始才真正做交集收窄。
    fn merge_into_init(
        target: &mut BTreeSet<usize>,
        source: &BTreeSet<usize>,
        seeded: &mut bool,
    ) -> bool {
        if !*seeded {
            *seeded = true;
            if *target != *source {
                *target = source.clone();
                return true;
            }
            return false;
        }
        let keys: Vec<usize> = target.iter().copied().collect();
        let mut changed = false;
        for k in keys {
            if !source.contains(&k) {
                target.remove(&k);
                changed = true;
            }
        }
        changed
    }

    // ============================================================
    // 借用冲突检查（前向，用 liveness 判定活跃 loan）
    // ============================================================
    fn collect_loans(f: &MirFn) -> Vec<Loan> {
        let mut loans = Vec::new();
        for block in &f.body.blocks {
            for stmt in &block.stmts {
                if let MirStmt::Assign { dest, value: MirRvalue::Ref { mutable, place } } = stmt {
                    if let Some(var) = Self::place_base_id(dest) {
                        loans.push(Loan {
                            borrow_var: var,
                            place: place.clone(),
                            kind: if *mutable { LoanKind::Mut } else { LoanKind::Shared },
                        });
                    }
                }
            }
        }
        loans
    }

    fn borrow_check(
        f: &MirFn,
        live_after: &[Vec<BTreeSet<usize>>],
        live_out: &[BTreeSet<usize>],
    ) -> Result<(), String> {
        let loans = Self::collect_loans(f);

        for (bi, block) in f.body.blocks.iter().enumerate() {
            for (j, stmt) in block.stmts.iter().enumerate() {
                // 活跃 loan = 那条借用的 borrow_var（引用变量）在这条语句
                // 执行期间仍然 live——要么这条语句之后还会被用
                // （live_after），要么这条语句本身就在用它（uses）。借用
                // 的活跃期完全由引用变量的 liveness 决定，跟原变量的词法
                // 作用域无关，这正是 NLL 的"最后一次使用处释放"。
                let mut active: Vec<&Loan> = loans
                    .iter()
                    .filter(|l| {
                        let b = l.borrow_var;
                        live_after[bi][j].contains(&b) || Self::stmt_uses(stmt).contains(&b)
                    })
                    .collect();

                // 借用创建语句要排除"这条语句自己刚创建出的那条 loan"，
                // 否则 `&mut x` 会跟自己撞上（borrow_var 若后续仍被使用，
                // 它也会出现在 active 里）。同 base_id 的旧 loan 一并排除：
                // 往同一个引用变量重新赋值，等价于旧引用已经不再被使用。
                if let MirStmt::Assign { dest, value: MirRvalue::Ref { .. } } = stmt {
                    if let Some(base) = Self::place_base_id(dest) {
                        active.retain(|l| l.borrow_var != base);
                    }
                }

                Self::check_stmt(stmt, &active, f)?;
            }

            // 终结指令里的操作数同样受借用约束：If/Switch 的条件读、
            // Return 的返回值，都必须跟"块出口处仍然活跃"的借用不冲突。
            // 块出口活跃集合 = live_out ∪ 终结指令自身的 uses。
            let mut exit_live = live_out[bi].clone();
            for u in Self::term_uses(&block.terminator) {
                exit_live.insert(u);
            }
            let active: Vec<&Loan> = loans
                .iter()
                .filter(|l| exit_live.contains(&l.borrow_var))
                .collect();
            Self::check_terminator(&block.terminator, &active, f)?;
        }

        Ok(())
    }

    fn check_stmt(stmt: &MirStmt, active: &[&Loan], f: &MirFn) -> Result<(), String> {
        match stmt {
            MirStmt::Assign { dest, value } => {
                // 1. 右值检查：区分"只读（Copy）"和"消费（Move）"。
                //    Copy 只是读，只跟可变借用冲突；Move 会消费，跟任何
                //    借用（共享或可变）都冲突。
                let mut copies = Vec::new();
                let mut moves = Vec::new();
                Self::rvalue_places(value, &mut copies, &mut moves);
                for p in &copies {
                    Self::check_read(p, active, f)?;
                }
                for p in &moves {
                    Self::check_move(p, active, f)?;
                }
                // 2. 借用创建检查（对 `&`/`&mut` 单独做，比读更严格）。
                if let MirRvalue::Ref { mutable, place } = value {
                    Self::check_borrow_creation(place, *mutable, active, f)?;
                }
                // 3. 写检查：dest 不能与任何活跃借用重叠。
                Self::check_write(dest, active, f)?;
            }
            MirStmt::ExprStmt(value) => {
                let mut copies = Vec::new();
                let mut moves = Vec::new();
                Self::rvalue_places(value, &mut copies, &mut moves);
                for p in &copies {
                    Self::check_read(p, active, f)?;
                }
                for p in &moves {
                    Self::check_move(p, active, f)?;
                }
            }
            MirStmt::Drop { place } => {
                // 关键修复：原来这里直接调用 check_write，报错信息会说
                // "cannot write `x` because ..."——但这条语句明明是
                // Drop，不是写，报错文本对不上实际发生的事。逻辑（"不能
                // 跟任何活跃借用重叠"）跟 write/move 完全一样，走同一个
                // check_exclusive helper，只是动作动词换成 "drop"。
                Self::check_exclusive(place, active, f, "drop")?;
            }
            MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
        }
        Ok(())
    }

    fn check_terminator(term: &MirTerminator, active: &[&Loan], f: &MirFn) -> Result<(), String> {
        match term {
            MirTerminator::Goto(_) | MirTerminator::Unreachable => Ok(()),
            MirTerminator::If { cond, .. } => Self::check_operand_use(cond, active, f),
            MirTerminator::Return(op) => match op {
                Some(op) => Self::check_operand_use(op, active, f),
                None => Ok(()),
            },
            MirTerminator::Switch { discr, .. } => Self::check_operand_use(discr, active, f),
            // 关键修复（Placeholder/Unreachable 拆分）：mir.rs 新增的
            // Placeholder 专门表示"这个块的终止器还没被真正设置过"（见
            // mir.rs 里 MirTerminator 定义处的说明）。借用检查在流水线
            // 里跑在 control::simplify 之后（pipeline.rs：lower →
            // control::simplify → borrow_check），任何真正跑完构建的
            // 函数，到这一步理应不再有 Placeholder 残留——如果真的
            // 看到一个，说明 mir_builder.rs 某条路径漏了给某个块设置
            // 终止器，是编译器自身的构建缺陷，不是用户程序的错，不该
            // 悄悄放过（当成 Unreachable 处理）掩盖这个信号。
            MirTerminator::Placeholder => Err(
                "internal error: block terminator is still Placeholder at borrow-check time \
                 (mir_builder.rs 某条路径构建完没有给这个块设置真正的终止器)".to_string()
            ),
        }
    }

    fn check_operand_use(op: &MirOperand, active: &[&Loan], f: &MirFn) -> Result<(), String> {
        match op {
            MirOperand::Copy(p) => Self::check_read(p, active, f),
            MirOperand::Move(p) => Self::check_move(p, active, f),
            MirOperand::Constant(_) => Ok(()),
            // 关键修复（Sym/Static 搬家）：跟 collect_operand_uses 那处
            // 同一个道理——Sym/Static 是编译期已知/待 JIT 期绑定的值，
            // 不指向任何局部变量的内存，谈不上"读"或"移动"，没有借用
            // 规则可检查。
            MirOperand::Sym(_) => Ok(()),
            MirOperand::Static(_) => Ok(()),
        }
    }

    fn check_read(place: &MirPlace, active: &[&Loan], f: &MirFn) -> Result<(), String> {
        for loan in active {
            if loan.kind == LoanKind::Mut && Self::places_overlap(place, &loan.place) {
                return Err(format!(
                    "cannot read `{}` because it is mutably borrowed as `{}`",
                    Self::describe_place(place, f),
                    Self::get_var_name(f, loan.borrow_var)
                ));
            }
        }
        Ok(())
    }

    /// check_write / check_move / Drop 三处逻辑完全一样——只要有任何一条
    /// 活跃 loan 跟这个 place 重叠就报错，唯一区别是错误信息里的动作
    /// 动词（write / move / drop）。抽成一个 helper，避免三份几乎一样
    /// 的循环体各自维护一份。
    fn check_exclusive(place: &MirPlace, active: &[&Loan], f: &MirFn, action: &str) -> Result<(), String> {
        for loan in active {
            if Self::places_overlap(place, &loan.place) {
                return Err(format!(
                    "cannot {} `{}` because it is {} borrowed as `{}`",
                    action,
                    Self::describe_place(place, f),
                    loan.kind,
                    Self::get_var_name(f, loan.borrow_var)
                ));
            }
        }
        Ok(())
    }

    fn check_write(place: &MirPlace, active: &[&Loan], f: &MirFn) -> Result<(), String> {
        Self::check_exclusive(place, active, f, "write")
    }

    fn check_move(place: &MirPlace, active: &[&Loan], f: &MirFn) -> Result<(), String> {
        Self::check_exclusive(place, active, f, "move")
    }

    fn check_borrow_creation(
        place: &MirPlace,
        mutable: bool,
        active: &[&Loan],
        f: &MirFn,
    ) -> Result<(), String> {
        for loan in active {
            if !Self::places_overlap(place, &loan.place) {
                continue;
            }
            if mutable {
                return Err(format!(
                    "cannot mutably borrow `{}` because it is already {} borrowed as `{}`",
                    Self::describe_place(place, f),
                    loan.kind,
                    Self::get_var_name(f, loan.borrow_var)
                ));
            } else if loan.kind == LoanKind::Mut {
                return Err(format!(
                    "cannot shared-borrow `{}` because it is mutably borrowed as `{}`",
                    Self::describe_place(place, f),
                    Self::get_var_name(f, loan.borrow_var)
                ));
            }
            // `&` 与已有 `&` 可以共存，不报错。
        }
        Ok(())
    }

    /// 把右值里被"读"的 place 拆成两组：Copy（只读）与 Move（消费）。
    /// 注意：`Ref` 不在这里返回目标 place——借用的目标由
    /// `check_borrow_creation` 单独检查，`Ref` 本身只是取地址，不读值。
    fn rvalue_places(rv: &MirRvalue, copies: &mut Vec<MirPlace>, moves: &mut Vec<MirPlace>) {
        match rv {
            MirRvalue::Use(op) => Self::operand_place(op, copies, moves),
            MirRvalue::BinaryOp(_, l, r) => {
                Self::operand_place(l, copies, moves);
                Self::operand_place(r, copies, moves);
            }
            MirRvalue::UnaryOp(_, op) => Self::operand_place(op, copies, moves),
            MirRvalue::Cast(op, _) => Self::operand_place(op, copies, moves),
            MirRvalue::Call { args, .. } => {
                for a in args {
                    Self::operand_place(a, copies, moves);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::operand_place(receiver, copies, moves);
                for a in args {
                    Self::operand_place(a, copies, moves);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::operand_place(op, copies, moves);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for a in args {
                    Self::operand_place(a, copies, moves);
                }
            }
            // 关键修复：`Ref` 本身确实不读"目标"这个值（借用只是取
            // 地址，注释里已经说明），但如果目标 place 内部嵌了 Index
            // （比如 `&arr[i]`），下标操作数 `i` 是要被求值的——这是
            // 一次真正的读/移动，跟"借用 arr[i] 这件事本身"是两回事：
            // 如果 `i` 当下正处于一个活跃的可变借用之下，求值 `i`
            // 本身就该冲突，不能因为它出现在借用目标内部就被跳过
            // 检查。原来这里整个返回空，连 `i` 都没检查。
            MirRvalue::Ref { place, .. } => Self::collect_nested_index_operands(place, copies, moves),
            MirRvalue::ArrayLiteral(elems) => {
                for e in elems {
                    Self::operand_place(e, copies, moves);
                }
            }
            MirRvalue::Discriminant { value, .. } => Self::operand_place(value, copies, moves),
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::operand_place(op, copies, moves);
                }
            }
        }
    }

    fn operand_place(op: &MirOperand, copies: &mut Vec<MirPlace>, moves: &mut Vec<MirPlace>) {
        match op {
            MirOperand::Copy(p) => {
                // 关键修复：同 Ref 那处，`p` 内部如果嵌了 Index（比如
                // `arr[i]`），下标操作数 `i` 本身也是一次读/移动，得
                // 单独走一遍检查——不能只检查 `arr[i]` 这个 place 整体
                // 跟活跃借用重不重叠，却漏掉"读 i 这件事本身"是否安全
                // （比如 i 当下正被可变借用着）。
                Self::collect_nested_index_operands(p, copies, moves);
                copies.push(p.clone());
            }
            MirOperand::Move(p) => {
                Self::collect_nested_index_operands(p, copies, moves);
                moves.push(p.clone());
            }
            MirOperand::Constant(_) => {}
            // 关键修复（Sym/Static 搬家）：不指向任何局部变量的内存，
            // 不产生 copies/moves。
            MirOperand::Sym(_) => {}
            MirOperand::Static(_) => {}
        }
    }

    /// 递归收集一个 place 内部所有 Index 投影用到的下标操作数，按它们
    /// 各自原本的 Copy/Move 标记分别归类——用于 Ref/Copy/Move 处理
    /// place 时，把"读这个 place 需要连带求值哪些下标"这件事也纳入
    /// 借用冲突检查，不会因为下标操作数嵌在另一个 place 内部就被漏查。
    fn collect_nested_index_operands(
        place: &MirPlace,
        copies: &mut Vec<MirPlace>,
        moves: &mut Vec<MirPlace>,
    ) {
        match place {
            MirPlace::Ssa(_) => {}
            MirPlace::Field { base, .. } => Self::collect_nested_index_operands(base, copies, moves),
            MirPlace::Index { base, index } => {
                Self::collect_nested_index_operands(base, copies, moves);
                Self::operand_place(index, copies, moves);
            }
            MirPlace::Deref(base) => Self::collect_nested_index_operands(base, copies, moves),
            MirPlace::EnumPayload { base, .. } => Self::collect_nested_index_operands(base, copies, moves),
        }
    }

    // ============================================================
    // place 重叠判定（字段级，保守安全）
    // ============================================================
    fn places_overlap(a: &MirPlace, b: &MirPlace) -> bool {
        let (ra, pa) = Self::flatten(a);
        let (rb, pb) = Self::flatten(b);
        if ra != rb {
            return false;
        }
        let n = pa.len().min(pb.len());
        for i in 0..n {
            match (&pa[i], &pb[i]) {
                (Proj::Field(x), Proj::Field(y)) => {
                    if x != y {
                        return false;
                    }
                }
                // 出现 Deref / Opaque / 类型不匹配，退化为"可能重叠"。
                _ => return true,
            }
        }
        // 较短的投影链是较长的那一个的前缀，重叠。
        true
    }

    fn flatten(place: &MirPlace) -> (Root, Vec<Proj>) {
        match place {
            MirPlace::Ssa(s) => (Root::Local(s.base_id), Vec::new()),
            MirPlace::Field { base, field } => {
                let (r, mut ps) = Self::flatten(base);
                ps.push(Proj::Field(field.clone()));
                (r, ps)
            }
            MirPlace::Index { base, .. } => {
                let (r, mut ps) = Self::flatten(base);
                ps.push(Proj::Opaque);
                (r, ps)
            }
            MirPlace::Deref(base) => {
                let (r, mut ps) = Self::flatten(base);
                ps.push(Proj::Deref);
                (r, ps)
            }
            MirPlace::EnumPayload { base, .. } => {
                let (r, mut ps) = Self::flatten(base);
                ps.push(Proj::Opaque);
                (r, ps)
            }
        }
    }

    // ============================================================
    // 辅助
    // ============================================================
    // 关键说明：Static 分支删掉之后，这个函数其实再也不会返回 None
    // 了——MirPlace 现在只有 Ssa/Field/Index/Deref/EnumPayload 五种，
    // 全部递归到底都落在 Ssa 上。保留 Option<usize> 这个签名不变（没有
    // 收紧成直接返回 usize），是因为改签名要牵动所有调用点，这次不做，
    // 只求把 E0599（Static 变体不存在）消掉。
    fn place_base_id(place: &MirPlace) -> Option<usize> {
        match place {
            MirPlace::Ssa(s) => Some(s.base_id),
            MirPlace::Field { base, .. } => Self::place_base_id(base),
            MirPlace::Index { base, .. } => Self::place_base_id(base),
            MirPlace::Deref(base) => Self::place_base_id(base),
            MirPlace::EnumPayload { base, .. } => Self::place_base_id(base),
        }
    }

    // 关键重构：原来这里有一份 is_copy_type，跟 simplify.rs 里的另一份
    // 各自维护、判断标准还悄悄不一致（simplify.rs 那边一直没跟上这里
    // 后来补的 Privacy/Tuple/Array 递归处理）。现在统一收进
    // ast.rs::Type::is_copy()，两边都改成调用 `ty.is_copy()`，不用再
    // 记得"改 Copy 类型要两处一起改"这件事。

    fn get_var_type(f: &MirFn, base_id: usize) -> Type {
        f.body
            .locals
            .iter()
            .find(|l| l.id == base_id)
            .map(|l| l.ty.clone())
            .unwrap_or(Type::Unit)
    }

    fn get_var_name(f: &MirFn, base_id: usize) -> String {
        f.body
            .locals
            .iter()
            .find(|l| l.id == base_id)
            .and_then(|l| l.name.clone())
            .unwrap_or_else(|| format!("_var_{}", base_id))
    }

    fn describe_place(place: &MirPlace, f: &MirFn) -> String {
        match place {
            MirPlace::Ssa(s) => Self::get_var_name(f, s.base_id),
            MirPlace::Field { base, field } => format!("{}.{}", Self::describe_place(base, f), field),
            MirPlace::Index { base, .. } => format!("{}[..]", Self::describe_place(base, f)),
            MirPlace::Deref(base) => format!("*{}", Self::describe_place(base, f)),
            MirPlace::EnumPayload { base, .. } => format!("{}.payload", Self::describe_place(base, f)),
        }
    }
}
