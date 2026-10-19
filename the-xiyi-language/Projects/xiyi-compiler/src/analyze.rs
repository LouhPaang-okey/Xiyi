// analyze.rs
//
// 整函数分析的数据结构：CFG/支配树/可达性、SCCP 格与结果、GVN 的 key
// 与结果、循环信息、别名信息、函数纯度表、以及把它们打包起来的
// AnalysisResults。只依赖 mir 和 detect（底层查询库），不依赖 simplify
// 等优化 pass；具体分析的求解与编排在各 pass 里（见 Simplify::analyze）。
//
// 依赖方向：simplify/control -> analyze -> detect。detect 不能反过来
// use analyze，所以依赖 LoopInfo 的 find_loop_invariants 放在这里。
use crate::detect::{
    binop_may_trap, block_phi_info, call_purity, copy_local_ids, int_literal_value, place_has_deref,
    place_may_trap, purity_of_fn, rvalue_may_trap, scan_place_deref, scan_rvalue_deref,
    scan_terminator_facts, stmt_is_phi, Purity,
};
use crate::mir::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

// ==================== 控制流图信息（CFG Info）====================
//
// 关键设计：前驱/后继表和支配树跟具体做哪一种分析无关，只依赖
// MirBody 本身的形状，所以单独抽成 CfgInfo，作为后面所有分析共用的
// L0 基础设施。SCCP 靠 exec_edges 判断可执行性，用不上支配树；GVN
// 要判断"某个已有的计算结果能不能替代这一处"，靠的就是 dominates 和
// rpo（见下面 rpo 字段的说明）。
//
// 边表不在这里重新扫描终止器，直接用 MirBody::edge_tables——全项目
// 只有那一处知道每种终止器指向哪些块。
//
// 支配树算法用的是 Cooper-Harvey-Kennedy 的迭代版本，不是经典的
// Lengauer-Tarjan——渐进复杂度差一些，但不需要额外的并查集结构，
// 实现和验证都简单得多，函数体量通常几十个块，这点差距可以忽略。
#[allow(dead_code)]
pub struct CfgInfo {
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
    /// 直接支配者。入口块为 None（无支配者）。不可达块保持 None。
    idom: Vec<Option<usize>>,
    /// 在支配树里的深度，入口为 0。配合 idom 把 dominates 查询加速到
    /// O(depth)，不用每次都从头沿着 idom 链爬到根。
    dom_depth: Vec<usize>,
    /// reverse post-order，只含从入口可达的块。支配树有一条经典性质：
    /// 若 A 支配 B，则 A 在 RPO 里一定先于 B。GVN 靠这条性质做到"按
    /// RPO 线性扫一遍块 + 一张全局候选表 + dominates 判断"，不需要按
    /// 支配树做"进入块时复制作用域表、退出块时还原"那一套。
    rpo: Vec<usize>,
    /// 建立这份快照时 MirBody.revision 的值。块的集合或编号变了之后修订号
    /// 会变，据此识别"拿着过期的快照"——只比块数识别不了"块数没变、编号
    /// 却被打乱"的情形（见 MirBody::revision）。
    revision: u64,
}

#[allow(dead_code)]
impl CfgInfo {
    pub fn build(body: &MirBody) -> Self {
        let n = body.blocks.len();
        // 这里只如实反映 MIR 本身声明的控制流边（越界目标、重复边的处理
        // 见 MirBody::edge_tables），不判断"这个块是否可达"。基于这份
        // 边表的纯结构可达性见下面的 compute_reachable；SCCP 那种"沿着
        // 常量条件才能走到"的可执行性是另一回事，由 SCCP 自己算。
        let (successors, predecessors) = body.edge_tables();

        let rpo = Self::compute_rpo(&successors, n);
        let (idom, dom_depth) = Self::compute_dominators(&predecessors, &rpo, n);

        CfgInfo { successors, predecessors, idom, dom_depth, rpo, revision: body.revision }
    }

    /// 从入口可达的块，按 reverse post-order 排列。性质：若 A 支配 B，
    /// A 一定排在 B 前面。
    pub fn rpo(&self) -> &[usize] {
        &self.rpo
    }

    /// 后继表/前驱表（去重、已过滤越界，见 MirBody::edge_tables）。给
    /// LoopInfo 这类建在 CFG 之上的分析直接用，免得每个分析再扫一遍终止器。
    pub fn successors(&self) -> &[Vec<usize>] {
        &self.successors
    }

    pub fn predecessors(&self) -> &[Vec<usize>] {
        &self.predecessors
    }

    /// 迭代式 DFS 求 reverse post-order，只包含从入口可达的块——支配
    /// 树只对可达块有意义，不可达块的 idom 原样留 None。
    fn compute_rpo(successors: &[Vec<usize>], n: usize) -> Vec<usize> {
        if n == 0 { return Vec::new(); }
        let mut visited = vec![false; n];
        let mut post_order = Vec::new();
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        visited[0] = true;

        while let Some((node, idx)) = stack.pop() {
            if idx < successors[node].len() {
                stack.push((node, idx + 1));
                let succ = successors[node][idx];
                if !visited[succ] {
                    visited[succ] = true;
                    stack.push((succ, 0));
                }
            } else {
                post_order.push(node);
            }
        }
        post_order.reverse();
        post_order
    }

    /// Cooper-Harvey-Kennedy 迭代版支配树计算。
    fn compute_dominators(
        predecessors: &[Vec<usize>],
        rpo: &[usize],
        n: usize,
    ) -> (Vec<Option<usize>>, Vec<usize>) {
        let mut idom: Vec<Option<usize>> = vec![None; n];
        let mut dom_depth: Vec<usize> = vec![0; n];
        if rpo.is_empty() { return (idom, dom_depth); }

        // rpo_idx[b] = b 在 RPO 里的位置，给 intersect 判断"谁在 RPO
        // 里更靠后"用，避免每次都线性查找。
        let mut rpo_idx = vec![usize::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_idx[b] = i;
        }

        let entry = rpo[0];
        // 哨兵：让 entry 支配自己，intersect 沿 idom 链向上爬时才有
        // 一个确定的终点。计算结束后会在下面把它去掉，不对外暴露。
        idom[entry] = Some(entry);

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new_idom: Option<usize> = None;
                for &p in &predecessors[b] {
                    if idom[p].is_none() { continue; }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => Self::intersect(p, cur, &idom, &rpo_idx),
                    });
                }
                let new_idom = match new_idom {
                    Some(x) => x,
                    None => continue, // 还没有已处理的前驱（多半是不可达块），先跳过，等下一轮
                };
                if idom[b] != Some(new_idom) {
                    idom[b] = Some(new_idom);
                    changed = true;
                }
            }
        }

        // RPO 顺序保证处理 b 时 idom[b] 一定已经处理过，可以放心地用
        // 父节点的深度 + 1，不需要额外排序。
        for &b in rpo.iter().skip(1) {
            if let Some(parent) = idom[b] {
                if parent != b {
                    dom_depth[b] = dom_depth[parent] + 1;
                }
            }
        }

        // 去掉哨兵：entry 没有支配者。
        idom[entry] = None;

        (idom, dom_depth)
    }

    /// a 和 b 的最近公共支配者：沿 idom 链向上爬，RPO 位置更靠后的
    /// 那一个先往上走一步，直到两边相遇。
    fn intersect(
        a: usize, b: usize,
        idom: &[Option<usize>],
        rpo_idx: &[usize],
    ) -> usize {
        let mut a = a;
        let mut b = b;
        while a != b {
            while rpo_idx[a] > rpo_idx[b] {
                match idom[a] {
                    Some(p) if p != a => a = p,
                    _ => return a,
                }
            }
            while rpo_idx[b] > rpo_idx[a] {
                match idom[b] {
                    Some(p) if p != b => b = p,
                    _ => return b,
                }
            }
        }
        a
    }

    /// a 是否支配 b。先用 dom_depth 把 b 提到跟 a 相同的深度，再一起
    /// 往上走到相等为止，复杂度 O(depth)。
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        if a == b { return true; }
        let a_depth = self.dom_depth[a];
        let mut b = b;
        let mut b_depth = self.dom_depth[b];
        if a_depth > b_depth { return false; }
        while b_depth > a_depth {
            match self.idom[b] {
                Some(p) => { b = p; b_depth = self.dom_depth[b]; }
                None => return false,
            }
        }
        b == a
    }

    /// 建立这份快照时 MirBody 的修订号。
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// 建立这份快照时的块数。
    pub fn block_count(&self) -> usize {
        self.successors.len()
    }

    /// 从入口可达的块。基于 self.successors 做 DFS，不再回头读
    /// body.blocks[id].terminator——CfgInfo::build 已经把每个块的后继边
    /// 展开成边表了，可达性只是这张边表的一个属性，直接复用即可。
    /// 顺带消除了旧版 simplify.rs 里那份对 Goto/If/Switch 各写一遍的
    /// 重复展开逻辑（现在只有 MirTerminator::for_each_target 一处知道
    /// 每种终止器有哪些后继，新增终止器变体时不会再漏改）。
    pub fn compute_reachable(&self, n: usize) -> Vec<bool> {
        // 关键说明：CfgInfo 是按建立时的块编号索引的快照。调用方传进来的
        // n 必须等于建立时的块数，否则说明拿着过期的 CfgInfo 去查已经重
        // 编号过的 body——这是内部 bug，debug 下直接断言；release 下按
        // 快照自己的块数算，保证不会越界 panic。
        debug_assert_eq!(n, self.successors.len(), "stale CfgInfo: block count changed");
        let n = self.successors.len();
        if n == 0 {
            return Vec::new();
        }
        let mut reachable = vec![false; n];
        let mut stack = vec![0];
        reachable[0] = true;

        while let Some(id) = stack.pop() {
            for &succ in &self.successors[id] {
                if !reachable[succ] {
                    reachable[succ] = true;
                    stack.push(succ);
                }
            }
        }
        reachable
    }
}

// ==================== SSA 粒度后向活跃分析（Liveness）====================
//
// 关键重构：这份分析原来在三个地方各有一份（borrow.rs 的 compute_liveness、
// simplify.rs 的"全局已使用集合"死赋值判断、analyze.rs 里别处的扫描），逻辑
// 重复，而且口径不一样。现在只有这里一份，其它文件都调用它。
//
// 用 SSA 值而不是变量 base_id：一个 SSA 值只被定义一次、只被读若干次，它的
// 活跃区间就是从 def 到它所有 use 之间在 CFG 上可达的所有点。这样
// `r = &x; r = &y; use(r);` 里 r@1 的活跃区间只到 `r = &y` 为止，不会因为
// r 是同一个 base_id 就被强行续命到 `use(r)`。
//
//   live_after[b][i] = 块 b 的第 i 条语句执行完之后仍然活跃的 SSA 值
//   live_out[b]      = 块 b 出口处活跃的 SSA 值
//
// 关键设计（Phi 是边敏感的）：Phi 的 incoming (pred, v) 只在 pred -> 本块
// 这条边上活跃：v 加进 live_out[pred]，不会加进别的前驱的 live_out；Phi 自己
// 定义的值在块首被"定义掉"，不会往前驱传播。旧版把 Phi 的所有 incoming 当成
// Phi 所在块里的普通读取，每个 incoming 值在所有前驱路径上都显得活跃——对
// 借用检查是多余的保守（loan 被无谓续命），对死赋值消除则是漏判。
//
// 块内逆向扫描遇到 Phi 语句只记录 live_after，不做 gen/kill：它们的读写由
// 上面的边规则统一处理，块内再走一遍会把 incoming 当成块内读取，又退回
// 旧行为。
#[derive(Debug)]
pub struct Liveness {
    pub live_after: Vec<Vec<BTreeSet<SsaLocal>>>,
    pub live_out: Vec<BTreeSet<SsaLocal>>,
}

impl Liveness {
    /// 用 CFG 里已有的前驱表。
    pub fn build(body: &MirBody, cfg: &CfgInfo) -> Self {
        Self::build_from_preds(body, cfg.predecessors())
    }

    /// 只有前驱表就够了的调用方（没建 CfgInfo，比如借用检查、死赋值消除）
    /// 用这个——前驱表来自 MirBody::predecessor_table 即可。
    pub fn build_from_preds(body: &MirBody, preds: &[Vec<usize>]) -> Self {
        let n = body.blocks.len();
        let mut live_out: Vec<BTreeSet<SsaLocal>> = vec![BTreeSet::new(); n];
        let mut live_after: Vec<Vec<BTreeSet<SsaLocal>>> = Vec::with_capacity(n);
        for b in &body.blocks {
            let mut per_stmt: Vec<BTreeSet<SsaLocal>> = Vec::with_capacity(b.stmts.len());
            for _ in 0..b.stmts.len() {
                per_stmt.push(BTreeSet::new());
            }
            live_after.push(per_stmt);
        }
        if n == 0 {
            return Liveness { live_after, live_out };
        }

        let mut phi_info: Vec<(Vec<SsaLocal>, Vec<(usize, SsaLocal)>)> = Vec::with_capacity(n);
        for b in &body.blocks {
            phi_info.push(block_phi_info(b));
        }

        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);
        for b in 0..n {
            queue.push_back(b);
        }
        let mut in_queue = vec![true; n];
        while let Some(bi) = queue.pop_front() {
            in_queue[bi] = false;
            let block = &body.blocks[bi];

            // live 从"块出口活跃集合 + 终结器自身读到的值"起步。
            let mut live = live_out[bi].clone();
            for u in term_ssa_uses(&block.terminator) {
                live.insert(u);
            }
            for i in (0..block.stmts.len()).rev() {
                live_after[bi][i] = live.clone();
                if stmt_is_phi(&block.stmts[i]) {
                    continue;
                }
                for d in stmt_ssa_defs(&block.stmts[i]) {
                    live.remove(&d);
                }
                for u in stmt_ssa_uses(&block.stmts[i]) {
                    live.insert(u);
                }
            }
            // Phi 定义的值在块首才开始存在，往前驱传播之前先去掉。
            for d in &phi_info[bi].0 {
                live.remove(d);
            }

            for &p in &preds[bi] {
                let before = live_out[p].len();
                live_out[p].extend(live.iter().copied());
                if live_out[p].len() != before && !in_queue[p] {
                    in_queue[p] = true;
                    queue.push_back(p);
                }
            }

            // Phi 的 incoming 只在对应的那条边上活跃。越界的前驱 id（畸形
            // MIR）直接略过，不让这一步 panic。
            for &(pred_block, incoming_ssa) in &phi_info[bi].1 {
                if pred_block >= n {
                    continue;
                }
                if live_out[pred_block].insert(incoming_ssa) && !in_queue[pred_block] {
                    in_queue[pred_block] = true;
                    queue.push_back(pred_block);
                }
            }
        }

        Liveness { live_after, live_out }
    }
}

// ---- SSA use/def 提取 ----
//
// 关键说明：这里的"use/def"是 SSA 值粒度（SsaLocal，带版本号）的，供活跃
// 分析用。borrow.rs 的 init_check 追踪的是"变量的哪条字段路径还被初始化
// 着"（PathKey），名字像但语义不同，两边各自保留，不要往一起合。

pub fn stmt_ssa_defs(stmt: &MirStmt) -> Vec<SsaLocal> {
    let mut out = Vec::new();
    match stmt {
        MirStmt::Assign { dest, .. } => match dest {
            MirPlace::Ssa(s) => out.push(*s),
            // 写 p.x / p[i] / *p 不是定义一个新的 SSA 值。
            MirPlace::Field { .. }
            | MirPlace::Index { .. }
            | MirPlace::Deref(_)
            | MirPlace::EnumPayload { .. } => {}
        },
        MirStmt::ExprStmt(_)
        | MirStmt::Drop { .. }
        | MirStmt::SetMetadata { .. }
        | MirStmt::EffectCheck { .. } => {}
    }
    out
}

pub fn stmt_ssa_uses(stmt: &MirStmt) -> Vec<SsaLocal> {
    let mut out = Vec::new();
    match stmt {
        MirStmt::Assign { dest, value } => {
            ssa_uses_in_rvalue(value, &mut out);
            // 写一个带投影的 place（p.x / p[i] / *p）要先读 base 再改，所以把
            // dest 的 base 也算作 use；纯 Ssa dest 是全新 def，不算 use。
            match dest {
                MirPlace::Ssa(_) => {}
                MirPlace::Field { .. }
                | MirPlace::Index { .. }
                | MirPlace::Deref(_)
                | MirPlace::EnumPayload { .. } => ssa_uses_in_place(dest, &mut out),
            }
        }
        MirStmt::ExprStmt(value) => ssa_uses_in_rvalue(value, &mut out),
        MirStmt::Drop { place } => ssa_uses_in_place(place, &mut out),
        MirStmt::SetMetadata { place, .. } => ssa_uses_in_place(place, &mut out),
        MirStmt::EffectCheck { .. } => {}
    }
    out
}

pub fn term_ssa_uses(term: &MirTerminator) -> Vec<SsaLocal> {
    let mut out = Vec::new();
    match term {
        MirTerminator::If { cond, .. } => ssa_uses_in_operand(cond, &mut out),
        MirTerminator::Return(Some(op)) => ssa_uses_in_operand(op, &mut out),
        MirTerminator::Switch { discr, .. } => ssa_uses_in_operand(discr, &mut out),
        // Placeholder 在借用检查入口已经早退，到不了这里。显式列出每个没有
        // 操作数的变体，不用通配：将来 MirTerminator 新增一个带操作数的变体
        // 时，这里会编译失败，而不是静默漏掉它的 use（漏掉的后果是活跃分析
        // 少算一个活跃的值——借用检查里 loan 提前"死亡"，死赋值消除里把
        // 还有人读的定义删掉）。
        MirTerminator::Goto(_)
        | MirTerminator::Unreachable
        | MirTerminator::Return(None)
        | MirTerminator::Placeholder => {}
    }
    out
}

pub fn ssa_uses_in_place(place: &MirPlace, out: &mut Vec<SsaLocal>) {
    match place {
        MirPlace::Ssa(s) => out.push(*s),
        MirPlace::Field { base, .. } => ssa_uses_in_place(base, out),
        MirPlace::Index { base, index } => {
            ssa_uses_in_place(base, out);
            ssa_uses_in_operand(index, out);
        }
        MirPlace::Deref(base) => ssa_uses_in_place(base, out),
        MirPlace::EnumPayload { base, .. } => ssa_uses_in_place(base, out),
    }
}

fn ssa_uses_in_operand(op: &MirOperand, out: &mut Vec<SsaLocal>) {
    match op {
        MirOperand::Copy(p) | MirOperand::Move(p) => ssa_uses_in_place(p, out),
        MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => {}
    }
}

fn ssa_uses_in_rvalue(rv: &MirRvalue, out: &mut Vec<SsaLocal>) {
    match rv {
        MirRvalue::Use(op) => ssa_uses_in_operand(op, out),
        MirRvalue::BinaryOp(_, l, r) => {
            ssa_uses_in_operand(l, out);
            ssa_uses_in_operand(r, out);
        }
        MirRvalue::UnaryOp(_, o) => ssa_uses_in_operand(o, out),
        MirRvalue::Cast(o, _) => ssa_uses_in_operand(o, out),
        MirRvalue::Call { args, .. } => {
            for a in args {
                ssa_uses_in_operand(a, out);
            }
        }
        MirRvalue::MethodCall { receiver, args, .. } => {
            ssa_uses_in_operand(receiver, out);
            for a in args {
                ssa_uses_in_operand(a, out);
            }
        }
        MirRvalue::StructInit { fields, .. } => {
            for (_, op) in fields {
                ssa_uses_in_operand(op, out);
            }
        }
        MirRvalue::EnumVariantConstruction { args, .. } => {
            for a in args {
                ssa_uses_in_operand(a, out);
            }
        }
        MirRvalue::Ref { place, .. } => ssa_uses_in_place(place, out),
        MirRvalue::ArrayLiteral(elems) => {
            for e in elems {
                ssa_uses_in_operand(e, out);
            }
        }
        MirRvalue::Discriminant { value, .. } => ssa_uses_in_operand(value, out),
        MirRvalue::Phi { values } => {
            for (_, op) in values {
                ssa_uses_in_operand(op, out);
            }
        }
    }
}

// ==================== 可用表达式分析（Available Expressions）====================
//
// 前向"必经"分析：`available_in[b]` 是进入块 b 时，从入口走到 b 的*每一条*
// 路径上都已经算出过的表达式 key 的集合。
//
// 为什么 SSA 之下它没有"kill"：表达式的操作数是 SSA 值（只定义一次、永远
// 不变），一旦在某条路径上算过 `a + b`，这条路径之后它的结果就一直成立，
// 没有任何语句能让它失效。所以 gen 就是块里算过的所有 key，不需要 kill 集，
// available_out = available_in ∪ gen。key 里只放不读内存、不依赖纯度表的
// 形态（见 pure_expr_key）；读内存的表达式根本不进来，不存在"中间被写掉"。
//
// 跟 GVN 的分工：GVN 是"有一个支配当前点的定义可以直接拿来用"（基于单个
// SSA 名字），可用表达式回答的是"这个值在所有路径上都已经算过了，只是在
// 不同路径上可能是不同的 SSA 名字"（比如 if/else 两边各算了一遍 `a+b`，
// 汇合点没有任何一个定义支配它——GVN 看不见，这里看得见）。这里的 key 是
// 纯结构的（操作数就是 SSA 值本身），不吃 SCCP 的常量结论、也不跟着 GVN
// 的改写表走——它描述的是"程序现在长什么样"，不是"优化之后会长什么样"，
// 这样它能独立用来检查/引导 GVN，不会跟它互相依赖。
//
// 初值与不动点：入口块的 in 是空集；其它块一开始是"全集"（⊤），用 is_top
// 标记代替真的去构造一个全集。块 b 的 in = 所有前驱 out 的交集，还没算出来
// （仍是 ⊤）的前驱在交集里直接跳过——乐观假设，回边那一路的 out 还没算出来时
// 不拖累循环头。集合只会缩小，一定收敛。只处理从入口可达的块，不可达块的
// 集合保持空。
//
// holder_at_out：块出口处，每个可用 key 由哪个 SSA 持有（"这个值现成躺在
// 哪个变量里"）。块入口的 holder 只有在*所有*已算出的前驱都报告同一个 SSA
// 时才继承，否则丢弃——合并点上不同分支各算一遍的 key，没有任何现成变量
// 能代表它，需要插 phi 才能合并，这里保守放弃。块内新算的 key，holder 取块里
// 第一个定义它的 SSA；入口继承来的 key 保持原 holder（不被块内重算覆盖）。
//
// 保持"纯数据流、不含支配信息"：holder 的定义块到底支配不支配某个块的入口，
// 是消费者（simplify.rs::pick_holder）拿 ssa_def_blocks 和 dominates 去
// 判的事，不在这里做。
#[derive(Debug)]
pub struct AvailableExprs {
    pub available_in: Vec<HashSet<ExprKey>>,
    pub available_out: Vec<HashSet<ExprKey>>,
    pub holder_at_out: Vec<HashMap<ExprKey, SsaLocal>>,
}

impl AvailableExprs {
    pub fn is_available_at_entry(&self, block: usize, key: &ExprKey) -> bool {
        match self.available_in.get(block) {
            Some(s) => s.contains(key),
            None => false,
        }
    }

    pub fn build(body: &MirBody, cfg: &CfgInfo) -> Self {
        let n = body.blocks.len();
        let mut available_in: Vec<HashSet<ExprKey>> = vec![HashSet::new(); n];
        let mut available_out: Vec<HashSet<ExprKey>> = vec![HashSet::new(); n];
        let mut holder_at_out: Vec<HashMap<ExprKey, SsaLocal>> = vec![HashMap::new(); n];
        if n == 0 {
            return AvailableExprs { available_in, available_out, holder_at_out };
        }

        let copy_base = copy_local_ids(body);
        let mut gen_per_block: Vec<HashSet<ExprKey>> = Vec::with_capacity(n);
        for b in &body.blocks {
            gen_per_block.push(Self::block_gen(b, &copy_base));
        }

        let rpo = cfg.rpo();
        let preds_table = cfg.predecessors();

        // 入口块的初值：available_out 和 holder_at_out 一起从 gen 算出。
        // 关键：入口块在主循环里被跳过（它的 in 恒为空），holder_at_out[0]
        // 必须在这里就初始化——否则"唯一前驱是入口块"的块查不到入口块定义的
        // key 的 holder，所有替换都会被保守跳过。
        let mut is_top = vec![true; n];
        is_top[0] = false;
        available_out[0] = gen_per_block[0].clone();
        for stmt in &body.blocks[0].stmts {
            if let MirStmt::Assign { dest: MirPlace::Ssa(s), value } = stmt {
                if let Some(k) = Self::rvalue_key(value, &copy_base) {
                    holder_at_out[0].entry(k).or_insert(*s);
                }
            }
        }

        // 主不动点：available 和 holder 在同一个 while-changed 里一起迭代，
        // 不能"available 收敛之后再单独扫一遍 holder"——单独的一遍 RPO 扫描
        // 里，回边的源块排在循环头后面，那时它的 holder_at_out 还是初始空值，
        // 循环头的 key 会被误丢。
        //
        // 收敛性：available 只会缩小（交集、gen 固定）。holder 的取值是
        // "未算出 -> 某个 SSA -> 因分歧被丢弃"，加上块内首次定义这一档，
        // 变化次数有限。为了不依赖这个论证，下面还设了轮数上限：万一超限，
        // 不拿一个可能没稳定的 holder 去指导改写，直接清空全部 holder
        // （消费者查不到就保守跳过，只是少优化）。
        let max_rounds = 4 * n + 16;
        let mut rounds = 0usize;
        let mut converged = false;
        while rounds < max_rounds {
            rounds += 1;
            let mut changed = false;
            for &b in rpo {
                if b == 0 {
                    continue;
                }
                let preds = &preds_table[b];

                let mut new_in: Option<HashSet<ExprKey>> = None;
                for &p in preds {
                    if is_top[p] {
                        continue;
                    }
                    new_in = Some(match new_in {
                        None => available_out[p].clone(),
                        Some(cur) => cur.intersection(&available_out[p]).cloned().collect(),
                    });
                }
                // 所有前驱都还是 ⊤：这个块这一轮还算不出来，等前驱先算。
                let new_in = match new_in {
                    Some(s) => s,
                    None => continue,
                };

                // 入口 holder：所有已算出的前驱都同意才继承。
                let mut new_holder_in: HashMap<ExprKey, SsaLocal> = HashMap::new();
                if let Some(&first) = preds.iter().find(|&&p| !is_top[p]) {
                    'next_key: for (key, h) in &holder_at_out[first] {
                        if !new_in.contains(key) {
                            continue;
                        }
                        for &p in preds {
                            if is_top[p] {
                                continue;
                            }
                            match holder_at_out[p].get(key) {
                                Some(h2) if h2 == h => {}
                                _ => continue 'next_key,
                            }
                        }
                        new_holder_in.insert(key.clone(), *h);
                    }
                }

                let mut new_out = new_in.clone();
                let mut new_holder_out = new_holder_in.clone();
                for stmt in &body.blocks[b].stmts {
                    if let MirStmt::Assign { dest: MirPlace::Ssa(s), value } = stmt {
                        if let Some(k) = Self::rvalue_key(value, &copy_base) {
                            new_out.insert(k.clone());
                            new_holder_out.entry(k).or_insert(*s);
                        }
                    }
                }

                if is_top[b]
                    || available_in[b] != new_in
                    || available_out[b] != new_out
                    || holder_at_out[b] != new_holder_out
                {
                    is_top[b] = false;
                    available_in[b] = new_in;
                    available_out[b] = new_out;
                    holder_at_out[b] = new_holder_out;
                    changed = true;
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged {
            debug_assert!(false, "AvailableExprs::build: 没有在轮数上限内收敛");
            for h in holder_at_out.iter_mut() {
                h.clear();
            }
        }

        AvailableExprs { available_in, available_out, holder_at_out }
    }

    fn block_gen(block: &MirBlock, copy_base: &HashSet<usize>) -> HashSet<ExprKey> {
        let mut out: HashSet<ExprKey> = HashSet::new();
        for stmt in &block.stmts {
            if let MirStmt::Assign { dest: MirPlace::Ssa(_), value } = stmt {
                if let Some(k) = Self::rvalue_key(value, copy_base) {
                    out.insert(k);
                }
            }
        }
        out
    }

    /// 纯结构的表达式 key：操作数就是常量或 Copy 类型的 SSA 值本身。key 的
    /// 形态由 pure_expr_key 统一给出（交换律规范化也在那里），跟 GVN 共用，
    /// 不在这里再手抄一份。
    pub(crate) fn rvalue_key(rv: &MirRvalue, copy_base: &HashSet<usize>) -> Option<ExprKey> {
        pure_expr_key(rv, |op| Self::operand_key(op, copy_base))
    }

    fn operand_key(op: &MirOperand, copy_base: &HashSet<usize>) -> Option<OperandKey> {
        match op {
            MirOperand::Constant(lit) => Some(OperandKey::Const(LiteralKey::from_literal(lit))),
            MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => {
                if copy_base.contains(&s.base_id) {
                    Some(OperandKey::Ssa(*s))
                } else {
                    None
                }
            }
            MirOperand::Copy(_)
            | MirOperand::Move(_)
            | MirOperand::Sym(_)
            | MirOperand::Static(_) => None,
        }
    }
}

// ==================== SCCP 数据结构 ====================
//
// 关键设计：Lattice 只有三层（Undefined ⊑ Constant ⊑ Overdefined），
// 不追踪"两个不同的 SsaLocal 是否始终相等"这种更细的信息——那属于
// 值编号（GVN）要回答的问题，SCCP 只回答"这个 SsaLocal 是不是一个
// 编译期已知的字面量常量"。
#[derive(Clone, Debug, PartialEq)]
pub enum Lattice {
    /// 乐观初始值：还没有证据，先假设它有可能是常量。
    Undefined,
    /// 已知等于某个具体字面量。
    Constant(Literal),
    /// 保守顶：确认不可能是单一常量（多个不同的值汇合到了一起，或者
    /// 来源本身没法在编译期求值，比如一次函数调用的返回值）。
    Overdefined,
}

impl Lattice {
    /// 把 other 合进 self（格上的 join：Undefined < Constant <
    /// Overdefined，只能往上走不能往下退），返回 self 是否因此发生了
    /// 变化——调用方用这个返回值驱动不动点迭代的 changed 标记。
    pub fn join(&mut self, other: &Lattice) -> bool {
        match (&*self, other) {
            (Lattice::Overdefined, _) | (_, Lattice::Undefined) => false,
            (Lattice::Undefined, _) => {
                *self = other.clone();
                true
            }
            (Lattice::Constant(a), Lattice::Constant(b)) => {
                if a == b {
                    false
                } else {
                    *self = Lattice::Overdefined;
                    true
                }
            }
            (Lattice::Constant(_), Lattice::Overdefined) => {
                *self = Lattice::Overdefined;
                true
            }
        }
    }
}

#[allow(dead_code)]
pub struct SccpResult {
    pub values: HashMap<SsaLocal, Lattice>,
    /// 哪些块被判定为可执行（沿着 SCCP 自己求出的条件分支结果能走到，
    /// 不是单纯的 CFG 可达性——条件恒为假的分支即使在 CFG 上连着，这
    /// 里也是 false）。不可执行块里的语句不参与常量推导。
    pub exec_blocks: Vec<bool>,
    /// 哪些边被判定为可执行。Phi 只应该 join 来自可执行边的 incoming，
    /// 这是 SCCP 相对于朴素常量传播能在条件分支和循环上都算对的关键，
    /// 见 simplify.rs 里 sccp_eval_rvalue 的 Phi 分支说明。
    pub exec_edges: HashSet<(usize, usize)>,
}

impl SccpResult {
    /// 块是否被 SCCP 判定为可执行。比 CfgInfo::compute_reachable 精确：
    /// 后者只看结构上有没有边连着，这里还要求沿途的条件分支在常量结论
    /// 下真的会走过来。越界一律当作不可执行（调用方应先用
    /// AnalysisResults::matches 确认快照没过期）。
    pub fn block_executable(&self, id: usize) -> bool {
        self.exec_blocks.get(id).copied().unwrap_or(false)
    }

    /// 边 from -> to 是否被判定为可执行。Phi 的 incoming 裁剪应该看
    /// 边而不是看前驱块：前驱块可执行，不代表它一定会跳到这个块。
    pub fn edge_executable(&self, from: usize, to: usize) -> bool {
        self.exec_edges.contains(&(from, to))
    }
}

// ==================== GVN 数据结构 ====================
//
// 关键设计：GVN 的"两个计算是不是同一个值"靠表达式 key 判定。key 必须
// 能哈希、能比较，还要在运算满足交换律时让 `a + b` 和 `b + a` 落到同
// 一个 key 上——所以操作数 key 需要一个全序（派生 Ord），才能把两个
// 操作数"排一下"再组 key。
//
// LiteralKey：把 Literal 归一化成可哈希、可比较的形式。不能直接给
// Literal 派生 Hash/Eq/Ord：浮点变体没有 Eq/Hash（NaN != NaN），必须
// 转成位模式。GVN 对浮点要的是"这两次计算的操作数是不是同一个 bit
// pattern"，跟 to_bits() 的语义一致。Float16 在 AST 里是拿 f32 模拟的
// （Literal::Float16(f32)），所以这里存的是 f32 的 u32 位模式，不是
// 真的 16 位。

#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum LiteralKey {
    Int8(i8), Int16(i16), Int32(i32), Int64(i64), Int128(i128),
    UInt8(u8), UInt16(u16), UInt32(u32), UInt64(u64), UInt128(u128),
    Isize(isize), Usize(usize),
    Float16(u32), Float32(u32), Float64(u64),
    Bool(bool),
    Char(char),
    String(String),
    Unit,
    ByteString(Vec<u8>),
}

impl LiteralKey {
    // 显式列出全部变体、不用通配：ast::Literal 新增变体时这里会编译失败，
    // 逼着当场决定新变体怎么归一化——用通配兜底的话，新变体会悄悄落到
    // 某个不该落的 key 上，把不相等的两个字面量误判成同一个值。
    pub fn from_literal(lit: &Literal) -> LiteralKey {
        match lit {
            Literal::Int8(v) => LiteralKey::Int8(*v),
            Literal::Int16(v) => LiteralKey::Int16(*v),
            Literal::Int32(v) => LiteralKey::Int32(*v),
            Literal::Int64(v) => LiteralKey::Int64(*v),
            Literal::Int128(v) => LiteralKey::Int128(*v),
            Literal::UInt8(v) => LiteralKey::UInt8(*v),
            Literal::UInt16(v) => LiteralKey::UInt16(*v),
            Literal::UInt32(v) => LiteralKey::UInt32(*v),
            Literal::UInt64(v) => LiteralKey::UInt64(*v),
            Literal::UInt128(v) => LiteralKey::UInt128(*v),
            Literal::Isize(v) => LiteralKey::Isize(*v),
            Literal::Usize(v) => LiteralKey::Usize(*v),
            Literal::Float16(v) => LiteralKey::Float16(v.to_bits()),
            Literal::Float32(v) => LiteralKey::Float32(v.to_bits()),
            Literal::Float64(v) => LiteralKey::Float64(v.to_bits()),
            Literal::Bool(v) => LiteralKey::Bool(*v),
            Literal::Char(v) => LiteralKey::Char(*v),
            Literal::String(v) => LiteralKey::String(v.clone()),
            Literal::Unit => LiteralKey::Unit,
            Literal::ByteString(v) => LiteralKey::ByteString(v.clone()),
        }
    }
}

// 变体顺序决定交换律规范化之后的操作数排列：常量排在 SSA 前面，所以
// `x + 1` 和 `1 + x` 都规范成 (Const(1), Ssa(x))。
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum OperandKey {
    Const(LiteralKey),
    Ssa(SsaLocal),
}

// 关键设计：key 只覆盖"相同输入一定得到相同值、重复计算可以只留一份"的
// 右值，哪些右值能进来由 Simplify::gvn_expr_key 逐变体把关：
//   Binary/Unary/Cast/Discriminant：纯计算；
//   Use：Copy 类型的 SSA 复制，两次复制同一个源只留一份；
//   Call：只有 Purity::Pure 的调用（intrinsic 查 is_pure 表、用户函数查
//         纯度表）。key 带函数名、实参 key 列表和泛型实参——同名函数不同
//         泛型实参是不同的实例，不能合并。ReadOnly 的调用不进来：相同
//         参数的两次调用中间内存可能被改过，不保证同一个值。
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum ExprKey {
    Binary(BinaryOp, OperandKey, OperandKey),
    Unary(UnaryOp, OperandKey),
    Use(OperandKey),
    Cast(OperandKey, Type),
    Discriminant(OperandKey, String),
    Call(String, Vec<OperandKey>, Vec<Type>),
}

// 关键设计：把一个右值翻译成 ExprKey 的唯一入口（只覆盖"相同输入一定得到
// 相同值"、不依赖函数纯度表的形态：二元/一元运算、类型转换、Copy 类型 SSA 的
// 复制、枚举判别式）。形态判定和交换律规范化（`a+b` 与 `b+a` 落在同一个 key）
// 只写在这里；操作数 key 怎么来由调用方决定，用闭包传进来：
//   - GVN 的闭包：先检查操作数是 Copy 的 SSA/常量，再按 SCCP 常量结论和
//     GVN 改写表归一化（见 Simplify::gvn_operand_key）；
//   - 可用表达式的闭包：纯结构，操作数就是 SSA 值本身。
// 两边各写一份的话，新增一种可编号的右值时很容易只改到一边，让"可用表达式
// 认为算过了"和"GVN 认为是同一个值"对不上。
//
// Call 不在这里：它要查纯度表，只有 GVN 需要，由 Simplify::gvn_expr_key 自己
// 处理。显式列出每个变体、不用通配：新增右值种类时这里会编译失败。
pub fn pure_expr_key<F>(rv: &MirRvalue, mut operand_key: F) -> Option<ExprKey>
where
    F: FnMut(&MirOperand) -> Option<OperandKey>,
{
    match rv {
        MirRvalue::BinaryOp(op, l, r) => {
            let lk = operand_key(l)?;
            let rk = operand_key(r)?;
            let (lk, rk) = if is_commutative(*op) && lk > rk { (rk, lk) } else { (lk, rk) };
            Some(ExprKey::Binary(*op, lk, rk))
        }
        MirRvalue::UnaryOp(op, o) => {
            let ok = operand_key(o)?;
            Some(ExprKey::Unary(*op, ok))
        }
        MirRvalue::Cast(o, ty) => {
            let ok = operand_key(o)?;
            Some(ExprKey::Cast(ok, ty.clone()))
        }
        // Copy 类型 SSA 的复制：同一个源被复制两次，后一次直接读前一次的结果。
        MirRvalue::Use(MirOperand::Copy(MirPlace::Ssa(s)))
        | MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(s))) => {
            let ok = operand_key(&MirOperand::Copy(MirPlace::Ssa(*s)))?;
            Some(ExprKey::Use(ok))
        }
        MirRvalue::Discriminant { value, enum_name } => {
            let ok = operand_key(value)?;
            Some(ExprKey::Discriminant(ok, enum_name.clone()))
        }
        MirRvalue::Use(MirOperand::Constant(_))
        | MirRvalue::Use(MirOperand::Copy(_))
        | MirRvalue::Use(MirOperand::Move(_))
        | MirRvalue::Use(MirOperand::Sym(_))
        | MirRvalue::Use(MirOperand::Static(_))
        | MirRvalue::Call { .. }
        | MirRvalue::MethodCall { .. }
        | MirRvalue::StructInit { .. }
        | MirRvalue::EnumVariantConstruction { .. }
        | MirRvalue::Ref { .. }
        | MirRvalue::ArrayLiteral(_)
        | MirRvalue::Phi { .. } => None,
    }
}

// 二元运算是否满足交换律（`a op b == b op a`）。放在这里而不是 simplify.rs：
// 它描述的是 BinaryOp 本身的性质，跟 ExprKey 属于同一层。
//
// 显式列出全部变体、不用通配：新增运算符时必须当场表态"它满足交换律吗"，
// 通配兜底成 false 只是少一点优化，兜底成 true 就会把 `a - b` 和 `b - a`
// 这种不同的值合并成一个，是静默的错误结果。
//
// And/Or 在 MIR 里的操作数都是已经求好值的操作数（短路求值在构建
// 阶段就展开成了控制流，不会以 BinaryOp 的形式出现），所以对值来说
// 满足交换律。浮点 Add/Mul 交换后结果的数值位模式相同，仅在两个操作数
// 都是 NaN 时 NaN 的载荷位可能不同——跟 LLVM 对 fadd/fmul 的处理一致，
// 这里接受。
pub fn is_commutative(op: BinaryOp) -> bool {
    match op {
        BinaryOp::Add
        | BinaryOp::Mul
        | BinaryOp::Eq
        | BinaryOp::Neq
        | BinaryOp::And
        | BinaryOp::Or => true,

        BinaryOp::Sub
        | BinaryOp::Div
        | BinaryOp::Mod
        | BinaryOp::Lt
        | BinaryOp::Gt
        | BinaryOp::Le
        | BinaryOp::Ge => false,
    }
}

// GVN 的输出：dest -> rep，含义是"dest 这个 SSA 值和 rep 是同一个值，
// 而且 rep 的定义处支配 dest 的定义处，所以凡是读 dest 的地方都可以
// 改成读 rep"。
//
// 关键设计：表达式候选表（key -> 已有的计算）只是 gvn() 内部的临时
// 工作台，算完就没用了，不放进结果里——留着它只会占内存，还会让"结果里
// 哪些字段是对外承诺的契约"变得含糊。结果只保留对外的契约：rewrite，
// 以及为了能在分析刚算完时验证契约而记下的定义块（见 sites）。
//
// 写入统一走 record_rewrite，字段不对外可写：dest == rep 的自环和记不
// 全定义块的条目在这里就拒掉，不留给使用方去防。
#[derive(Debug)]
pub struct GvnResult {
    /// dest -> rep。契约（由 AnalysisResults::is_consistent 检查）：
    /// 无自环、无链（rep 自己不能再是某个 dest）、rep 的定义块支配
    /// dest 的定义块。
    pub rewrite: HashMap<SsaLocal, SsaLocal>,
    /// rewrite 里每一项的 (dest 所在块, rep 所在块)，只给一致性检查用。
    sites: HashMap<SsaLocal, (usize, usize)>,
}

impl GvnResult {
    pub fn empty() -> Self {
        GvnResult {
            rewrite: HashMap::new(),
            sites: HashMap::new(),
        }
    }

    /// 登记一条 dest -> rep。返回 false 表示被拒绝（dest == rep，登记
    /// 一个自环没有意义，也会让 fold_operand 里"读代表"变成死循环的
    /// 候选），调用方按"没有命中"处理即可。
    pub fn record_rewrite(
        &mut self,
        dest: SsaLocal,
        dest_block: usize,
        rep: SsaLocal,
        rep_block: usize,
    ) -> bool {
        if dest == rep {
            return false;
        }
        self.rewrite.insert(dest, rep);
        self.sites.insert(dest, (dest_block, rep_block));
        true
    }
}

// ==================== 函数纯度表 ====================
//
// 关键设计：Purity 这个概念和"从调用点/签名读出它"的函数在 detect.rs，
// 这里负责回答"每个用户函数到底属于哪一级"——它需要跨函数的不动点，
// 属于分析，不是单点查询。
//
// 为什么不能只信 EffectSet：hir_builder.rs 里一个 Call 表达式的 effects
// 只合并了它*实参*的 effects，被调函数自己的 effects 并没有合并进来；
// 而且只有函数名字面恰好是 "print" 才会置 has_io，panic/rng/ffi 在 HIR
// 层根本没有来源。后果：g 调用了 print，g.has_io 是 true，但 g 的调用者 f
// 的 EffectSet 全是 false——如果直接拿 EffectSet 当纯度，f 会被判成 Pure，
// 死赋值消除就会把 `let _ = f();` 整条删掉，连带丢掉 g 里的打印。
// 所以纯度必须看 MIR 函数体：函数体里出现的每个调用都按被调函数当前的
// 纯度 join 上来，Deref 写入直接算 Impure，Deref 读取至少是 ReadOnly（严格
// 只读，定义见 detect.rs 的 Purity）；
// EffectSet 只用来给初值（保留 HIR 层能看到的信息）。
//
// 不动点：初值是 EffectSet/签名给出的纯度，之后每轮把函数体里的调用按
// 当前表 join 进来，只升不降，所以一定收敛（每个函数至多升两级）。递归
// 和互递归自然成立——纯函数自己调用自己不会让自己变不纯。这里不证明
// 函数会终止：一个无限递归的"纯"函数在结果没被使用时也会被当成可删，
// 跟 LLVM 在没有 willreturn 证明时的立场不同，是这一轮明确接受的取舍。
//
// Drop 语句不计入副作用：目前编译器里没有用户自定义析构（Drop 的实现）
// 的建模，Drop 只回收函数自己拥有的内存。将来加了用户 Drop 实现，这里
// 要改成按被 Drop 类型的析构函数纯度 join。
pub fn compute_purity_table(
    fns: &[MirFn],
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> HashMap<String, Purity> {
    let mut table: HashMap<String, Purity> = HashMap::new();
    for f in fns {
        table.insert(f.name.clone(), purity_of_fn(&f.effect_set, &f.params));
    }

    // 格高度是 3、只升不降：每个函数至多升两级，所以有变化的轮数不会
    // 超过 2 * 函数数。上界纯属防御，正常一定提前收敛。
    let max_rounds = 2 * fns.len() + 2;
    let mut rounds = 0usize;
    let mut changed = true;
    while changed && rounds < max_rounds {
        changed = false;
        rounds += 1;
        for f in fns {
            let current = match table.get(&f.name) {
                Some(p) => *p,
                None => Purity::Impure,
            };
            if current == Purity::Impure {
                continue;
            }
            let next = current.join(body_purity(&f.body, &table, structs, enums));
            if next != current {
                table.insert(f.name.clone(), next);
                changed = true;
            }
        }
    }
    debug_assert!(!changed, "compute_purity_table: 纯度不动点没有在上界内收敛");
    table
}

fn body_purity(
    body: &MirBody,
    table: &HashMap<String, Purity>,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> Purity {
    let mut result = Purity::Pure;
    for block in &body.blocks {
        for stmt in &block.stmts {
            match stmt {
                MirStmt::Assign { dest, value } => {
                    // 通过 Deref 写入：写的是调用者可见的内存。
                    if place_has_deref(dest) {
                        result = result.join(Purity::Impure);
                    }
                    // 目标 place 自己也可能陷入（`a[i] = x` 越界）或读内存
                    // （下标操作数里嵌着 Deref），两者都只升到 ReadOnly。
                    if place_may_trap(dest, body, structs, enums) {
                        result = result.join(Purity::ReadOnly);
                    }
                    let mut dest_reads = false;
                    let mut dest_writes = false;
                    scan_place_deref(dest, &mut dest_reads, &mut dest_writes);
                    if dest_reads {
                        result = result.join(Purity::ReadOnly);
                    }
                    result = result.join(rvalue_purity(value, table, body, structs, enums));
                }
                MirStmt::ExprStmt(value) => {
                    result = result.join(rvalue_purity(value, table, body, structs, enums));
                }
                MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
            }
            if result == Purity::Impure {
                return result;
            }
        }
        // 终止器里的操作数（If 条件、Switch 判别式、返回值）也可能读 Deref
        // 或者带着越界下标。
        let (reads, _writes, traps) = scan_terminator_facts(&block.terminator, body, structs, enums);
        if reads {
            result = result.join(Purity::ReadOnly);
        }
        if traps {
            result = result.join(Purity::ReadOnly);
        }
    }
    result
}

fn rvalue_purity(
    rv: &MirRvalue,
    table: &HashMap<String, Purity>,
    body: &MirBody,
    structs: &[MirStruct],
    enums: &[MirEnum],
) -> Purity {
    let mut purity = Purity::Pure;
    // 第一步：只有调用才带来"被调函数自己的"纯度。
    match rv {
        MirRvalue::Call { func, is_intrinsic, intrinsic_name, .. } => {
            purity = purity.join(call_purity(func, *is_intrinsic, *intrinsic_name, table));
        }
        // MethodCall 要 receiver 类型的 trait 解析才知道调的是哪个方法，保守。
        MirRvalue::MethodCall { .. } => return Purity::Impure,
        MirRvalue::BinaryOp(_, _, _)
        | MirRvalue::Cast(_, _)
        | MirRvalue::UnaryOp(_, _)
        | MirRvalue::Use(_)
        | MirRvalue::StructInit { .. }
        | MirRvalue::EnumVariantConstruction { .. }
        | MirRvalue::Ref { .. }
        | MirRvalue::ArrayLiteral(_)
        | MirRvalue::Discriminant { .. }
        | MirRvalue::Phi { .. } => {}
    }
    // 第二步：对所有右值统一收口——不在每个分支里零散地查。
    // 任何操作数读 Deref：读的是调用者可见的内存，至少是 ReadOnly。
    let mut reads = false;
    let mut writes = false;
    scan_rvalue_deref(rv, &mut reads, &mut writes);
    if reads {
        purity = purity.join(Purity::ReadOnly);
    }
    // 运算本身（含嵌在操作数里的 Index 越界、除法）可能陷入：不改内存，
    // 只升到 ReadOnly（不能删、不能投机），不是 Impure。"哪些运算会陷入"
    // 的唯一定义在 detect.rs 的 rvalue_may_trap。
    if rvalue_may_trap(rv, body, structs, enums) {
        purity = purity.join(Purity::ReadOnly);
    }
    purity
}

// ==================== 别名分析 ====================
//
// 给"循环里的写入会不会影响循环里某个读取"提供判据。当前判据是"循环里
// 有没有通过 Deref 的写入"，只为将来 ReadOnly 调用的外提服务。
//
// 做了这些简化：
//   1. SSA 局部变量之间没有别名，每个 SsaLocal 只被赋值一次，读 x_v1 跟写
//      x_v2 无关。
//   2. 唯一可能产生别名的是通过 Deref 的内存访问。两个 Deref 是否指向同一
//      块内存需要 points-to 分析，这一轮没有，所以退化成按块的二值分类：
//      这个块有没有通过 Deref 写入/读取。查询"循环里有没有 Deref 写入"
//      就是所有循环块的 OR。
//
// 关键补充：调用也可能通过 Deref 写入——把 `&mut` 传给一个 Impure 的
// 被调函数，它就能改写调用者的内存，而这在调用者的 MIR 里一个 Deref 都
// 看不到。所以只要块里有 Impure 调用（MethodCall 一律算），就当作这个块
// 有 Deref 写入。ReadOnly/Pure 的调用按定义不写，不计入。没有这一条，将来
// 外提 ReadOnly 调用时会把它搬到一个"藏着 Impure 调用写同一块内存"的
// 循环外面，静默改变程序行为。
//
// 预留的扩展点：block_reads_deref/block_writes_deref 是按块的位图，将来
// 做 points-to 时把它们换成按 place 的集合，查询方法的签名不用变。
#[derive(Debug)]
pub struct AliasInfo {
    block_writes_deref: Vec<bool>,
    block_reads_deref: Vec<bool>,
}

impl AliasInfo {
    pub fn build(
        body: &MirBody,
        purity: &HashMap<String, Purity>,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> Self {
        let n = body.blocks.len();
        let mut writes = vec![false; n];
        let mut reads = vec![false; n];
        for (bi, block) in body.blocks.iter().enumerate() {
            for stmt in &block.stmts {
                match stmt {
                    MirStmt::Assign { dest, value } => {
                        if place_has_deref(dest) {
                            writes[bi] = true;
                        }
                        scan_place_deref(dest, &mut reads[bi], &mut writes[bi]);
                        scan_rvalue_deref(value, &mut reads[bi], &mut writes[bi]);
                        Self::scan_call_writes(value, purity, &mut writes[bi]);
                    }
                    MirStmt::ExprStmt(value) => {
                        scan_rvalue_deref(value, &mut reads[bi], &mut writes[bi]);
                        Self::scan_call_writes(value, purity, &mut writes[bi]);
                    }
                    MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
                }
            }
            // 局部变量名避开 reads/writes 两个按块的位图，免得遮蔽。
            let (term_reads, term_writes, _term_traps) = scan_terminator_facts(&block.terminator, body, structs, enums);
            if term_reads {
                reads[bi] = true;
            }
            if term_writes {
                writes[bi] = true;
            }
        }
        AliasInfo { block_writes_deref: writes, block_reads_deref: reads }
    }

    pub fn block_writes_deref(&self, block: usize) -> bool {
        self.block_writes_deref.get(block).copied().unwrap_or(false)
    }

    pub fn block_reads_deref(&self, block: usize) -> bool {
        self.block_reads_deref.get(block).copied().unwrap_or(false)
    }

    pub fn block_writes_len(&self) -> usize {
        self.block_writes_deref.len()
    }

    pub fn block_reads_len(&self) -> usize {
        self.block_reads_deref.len()
    }

    /// 循环里有没有通过 Deref 的写入。查不到这个循环时保守地回答"有"。
    pub fn loop_writes_deref(&self, loops: &LoopInfo, loop_idx: usize) -> bool {
        match loops.loops.get(loop_idx) {
            Some(l) => l.blocks.iter().any(|&b| self.block_writes_deref(b)),
            None => true,
        }
    }

    // Deref 读写的扫描（scan_place_deref 等）是纯查询，在 detect.rs；
    // 这里只保留依赖纯度表的 scan_call_writes。
    // 调用可能通过传进去的引用写调用者的内存，见上面"关键补充"。
    fn scan_call_writes(rv: &MirRvalue, purity: &HashMap<String, Purity>, writes: &mut bool) {
        match rv {
            MirRvalue::Call { func, is_intrinsic, intrinsic_name, .. } => {
                if call_purity(func, *is_intrinsic, *intrinsic_name, purity) == Purity::Impure {
                    *writes = true;
                }
            }
            MirRvalue::MethodCall { .. } => *writes = true,
            MirRvalue::Use(_)
            | MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Cast(_, _)
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. } => {}
        }
    }
}

// ==================== 循环信息（Loop Info）====================
//
// 自然循环识别与循环树。基础是"回边就是目标支配源的边"这条经典定义
// （Allen & Cocke 1972）：配上支配树一次扫完所有边找到全部循环头，再用
// 反向可达把循环体圈出来。同一个循环头的多条回边合并成一个循环，多条回边
// 就是同一个循环的多个 latch。依赖 CfgInfo（dominates / rpo / 边表），与
// SCCP / GVN 平行。
//
// 关键设计（只看可达块）：从入口不可达的块不参与循环识别——它们的支配关系
// 没有定义，反向圈循环体时如果把"某个不可达块恰好跳进循环"的边也算进去，
// 会把一个根本不被循环头支配的块圈进循环体，自己的一致性检查就过不去。
// 不可达块会在后面的 remove_unreachable_blocks 里被清掉，这里不需要为它们
// 操心。
//
// 关键设计（preheader 的严格定义）：preheader 不等于"唯一的循环外前驱"，
// 而是"唯一的循环外前驱，并且它的唯一后继就是循环头"。如果那个前驱还有
// 别的出口（比如以 If 结尾，一条边进循环、一条边绕开），把循环不变量外提
// 到它里面，会让绕开循环的那条路径也执行一遍——外提的运算可能 panic（除
// 零）、可能很贵，是错误的变换。所以这种情况 preheader 是 None，由
// control.rs 的 insert_preheaders 补一个专用块。

#[derive(Debug)]
pub struct Loop {
    pub header: usize,
    pub latches: Vec<usize>,
    /// 循环体的所有块（含 header），按块 id 升序。
    pub blocks: Vec<usize>,
    blocks_set: HashSet<usize>,
    pub exit_blocks: Vec<usize>,
    pub exit_edges: Vec<(usize, usize)>,
    /// 循环头的、来自循环外且从入口可达的前驱块（升序）。
    pub entering_preds: Vec<usize>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub preheader: Option<usize>,
}

impl Loop {
    /// 块是否在循环内（含 header）。
    pub fn contains(&self, block: usize) -> bool {
        self.blocks_set.contains(&block)
    }
}

#[derive(Debug)]
pub struct LoopInfo {
    pub loops: Vec<Loop>,
    pub innermost_loop_of: Vec<Option<usize>>,
}

impl LoopInfo {
    pub fn empty(n: usize) -> Self {
        LoopInfo { loops: Vec::new(), innermost_loop_of: vec![None; n] }
    }

    pub fn build(body: &MirBody, cfg: &CfgInfo) -> Self {
        let n = body.blocks.len();
        let mut rpo_idx = vec![usize::MAX; n];
        let mut innermost_loop_of: Vec<Option<usize>> = vec![None; n];
        // 用 BTreeMap：按循环头 id 升序遍历，生成的 Loop 顺序不依赖哈希
        // 迭代顺序，MIR 和后续诊断输出每次运行都一样。
        let mut latches_of: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut loops: Vec<Loop> = Vec::new();
        if n == 0 {
            return Self::empty(0);
        }

        let successors = cfg.successors();
        let predecessors = cfg.predecessors();
        let reachable = cfg.compute_reachable(n);

        for u in 0..n {
            if !reachable[u] {
                continue;
            }
            for &v in &successors[u] {
                if cfg.dominates(v, u) {
                    latches_of.entry(v).or_insert_with(Vec::new).push(u);
                }
            }
        }
        if latches_of.is_empty() {
            return Self::empty(n);
        }

        for (&header, latches) in &latches_of {
            let mut blocks_set: HashSet<usize> = HashSet::new();
            blocks_set.insert(header);
            let mut stack: Vec<usize> = latches.clone();
            while let Some(b) = stack.pop() {
                // header 已经在集合里，所以 latch 就是 header 自己（自环）
                // 时这里直接跳过；回溯永远不会越过 header 往循环外走。
                if !blocks_set.insert(b) {
                    continue;
                }
                for &p in &predecessors[b] {
                    if reachable[p] && !blocks_set.contains(&p) {
                        stack.push(p);
                    }
                }
            }
            let mut blocks: Vec<usize> = blocks_set.iter().copied().collect();
            blocks.sort_unstable();

            let mut exit_blocks: Vec<usize> = Vec::new();
            let mut exit_edges: Vec<(usize, usize)> = Vec::new();
            for &b in &blocks {
                for &s in &successors[b] {
                    if !blocks_set.contains(&s) {
                        if exit_blocks.last() != Some(&b) {
                            exit_blocks.push(b);
                        }
                        exit_edges.push((b, s));
                    }
                }
            }

            let mut entering_preds: Vec<usize> = Vec::new();
            for &p in &predecessors[header] {
                if reachable[p] && !blocks_set.contains(&p) {
                    entering_preds.push(p);
                }
            }
            // preheader：唯一的循环外前驱，且它只通向循环头（后继表已去重，
            // 长度为 1 就说明唯一的后继是 header）。
            let mut preheader: Option<usize> = None;
            if entering_preds.len() == 1 && successors[entering_preds[0]].len() == 1 {
                preheader = Some(entering_preds[0]);
            }

            loops.push(Loop {
                header,
                latches: latches.clone(),
                blocks,
                blocks_set,
                exit_blocks,
                exit_edges,
                entering_preds,
                parent: None,
                children: Vec::new(),
                preheader,
            });
        }

        // 外层循环排在前面（循环头 RPO 位置靠前）。
        for (i, &b) in cfg.rpo().iter().enumerate() {
            rpo_idx[b] = i;
        }
        loops.sort_by_key(|l| rpo_idx[l.header]);

        // 循环树：包含 i 的循环头、且体积最小的那个其它循环就是 i 的父亲。
        let mut parent_of: Vec<Option<usize>> = Vec::with_capacity(loops.len());
        for i in 0..loops.len() {
            let h = loops[i].header;
            let mut best: Option<usize> = None;
            let mut best_size = usize::MAX;
            for j in 0..loops.len() {
                if i == j {
                    continue;
                }
                if loops[j].blocks_set.contains(&h) && loops[j].blocks.len() < best_size {
                    best_size = loops[j].blocks.len();
                    best = Some(j);
                }
            }
            parent_of.push(best);
        }
        for i in 0..loops.len() {
            loops[i].parent = parent_of[i];
            if let Some(p) = parent_of[i] {
                loops[p].children.push(i);
            }
        }

        // 每个块的最内层循环：按体积从小到大分配，先到先得。
        let mut order: Vec<usize> = Vec::with_capacity(loops.len());
        for i in 0..loops.len() {
            order.push(i);
        }
        order.sort_by_key(|&i| loops[i].blocks.len());
        for i in order {
            for &b in &loops[i].blocks {
                if innermost_loop_of[b].is_none() {
                    innermost_loop_of[b] = Some(i);
                }
            }
        }

        LoopInfo { loops, innermost_loop_of }
    }

    pub fn in_loop(&self, block: usize) -> bool {
        self.innermost_loop_of.get(block).copied().flatten().is_some()
    }

    pub fn contains(&self, loop_idx: usize, block: usize) -> bool {
        match self.loops.get(loop_idx) {
            Some(l) => l.contains(block),
            None => false,
        }
    }

    pub fn enclosing_loops_innermost_first(&self, block: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = self.innermost_loop_of.get(block).copied().flatten();
        while let Some(l) = cur {
            out.push(l);
            cur = self.loops[l].parent;
        }
        out
    }

    /// 所有"专用 preheader"块的集合。control.rs/simplify.rs 里会穿过或合并
    /// 空块的 pass 要用它做保护：一个只含 `Goto(循环头)` 的空 preheader，
    /// 在它们眼里跟任何别的空跳板没有区别，会被当场绕过/折叠掉，而
    /// insert_preheaders 下一轮又会把它补回来——两边互相撤销，不动点循环
    /// 永远不收敛（每轮都报告 changed），最终形状取决于轮数的奇偶。
    pub fn preheader_blocks(&self) -> HashSet<usize> {
        let mut out: HashSet<usize> = HashSet::new();
        for l in &self.loops {
            if let Some(p) = l.preheader {
                out.insert(p);
            }
        }
        out
    }

    /// 循环信息自身的一致性检查（给 debug_assert 用）。
    pub fn check_consistency(&self, cfg: &CfgInfo) -> bool {
        let n = cfg.block_count();
        if self.innermost_loop_of.len() != n {
            return false;
        }
        let successors = cfg.successors();
        for (i, l) in self.loops.iter().enumerate() {
            if l.header >= n || !l.blocks_set.contains(&l.header) {
                return false;
            }
            for &b in &l.blocks {
                if b >= n || !l.blocks_set.contains(&b) {
                    return false;
                }
                // 循环头支配循环里的每一个块——自然循环的定义。
                if b != l.header && !cfg.dominates(l.header, b) {
                    return false;
                }
            }
            if l.blocks.len() != l.blocks_set.len() {
                return false;
            }
            for &latch in &l.latches {
                if !l.blocks_set.contains(&latch) || !successors[latch].contains(&l.header) {
                    return false;
                }
            }
            for &(from, to) in &l.exit_edges {
                if !l.blocks_set.contains(&from) || l.blocks_set.contains(&to) {
                    return false;
                }
            }
            for &p in &l.entering_preds {
                if p >= n || l.blocks_set.contains(&p) || !successors[p].contains(&l.header) {
                    return false;
                }
            }
            if let Some(ph) = l.preheader {
                if l.entering_preds.len() != 1 || l.entering_preds[0] != ph {
                    return false;
                }
                if successors[ph].len() != 1 {
                    return false;
                }
            }
            if let Some(p) = l.parent {
                if p >= self.loops.len() || !self.loops[p].children.contains(&i) {
                    return false;
                }
            }
            for &c in &l.children {
                if c >= self.loops.len() {
                    return false;
                }
                if self.loops[c].parent != Some(i) {
                    return false;
                }
                if !l.blocks_set.contains(&self.loops[c].header) {
                    return false;
                }
            }
        }
        for (b, l) in self.innermost_loop_of.iter().enumerate() {
            if let Some(idx) = l {
                if *idx >= self.loops.len() || !self.loops[*idx].blocks_set.contains(&b) {
                    return false;
                }
            }
        }
        true
    }
}

// ==================== 循环不变量检测 ====================
//
// 只报告，不改 IR（外提由 simplify.rs 的 licm 做）。判定条件（保守）：
//   1. 右值是 BinaryOp/UnaryOp/Cast（纯计算，没有 Call 之类的副作用）
//   2. 所有操作数要么是常量，要么是循环外定义的 SSA，要么是已经确认是不变量
//      的 SSA（SSA 只定义一次，"循环外定义"就等价于"循环内无重定义"）
//   3. 除法/取模只有在除数是非零整数常量时才算——它们可能 panic（除零），
//      把一个"循环一次都没跑就不会执行"的除法外提到循环前面，等于让它
//      无条件执行，是错误的投机。
//
// 关键重构：做成不动点。`x = a + 1; y = x * 2` 里 x 不变之后 y 也不变，单轮
// 扫描会漏掉 y，需要反复扫到没有新的不变量为止。每一轮的 not_yet =
// 循环内定义、但还没确认是不变量的 SSA 值；操作数不在 not_yet 里，就是
// 循环外定义，或者已经确认是不变量。
//
// 返回 (循环下标, dest 的 SSA)，按循环分组，组内按块的 RPO 顺序、块内按
// 语句顺序排列：支配者的块排在被支配者前面，依赖的定义排在使用前面，搬到
// preheader 之后依赖顺序天然成立，不需要再排序。嵌套循环里的不变量会对
// 每个包含它的循环各报一次。
//
// 这里只做"数据流意义上的不变"。内存读（`Use(Copy(a.f))`）只算到"地址不变"
// 这一层，内容会不会被循环里的写入改掉要别名信息，不在这里。能不能真的搬走还
// 要看是否安全（结果是不是 Copy 类型、运算会不会陷入、内存读有没有被循环内
// 写入覆盖、有没有专用 preheader），那些依赖类型信息、别名信息和循环结构，
// 由 licm 判断。它依赖 LoopInfo，因此放在 analyze.rs 而不是
// detect.rs（detect 不能依赖 analyze，见文件头）。
pub fn find_loop_invariants(
    body: &MirBody,
    cfg: &CfgInfo,
    loops: &LoopInfo,
) -> Vec<(usize, SsaLocal)> {
    let n = body.blocks.len();
    let mut rpo_idx = vec![usize::MAX; n];
    for (i, &b) in cfg.rpo().iter().enumerate() {
        rpo_idx[b] = i;
    }

    let mut local_tys: HashMap<usize, &Type> = HashMap::new();
    for local in &body.locals {
        local_tys.insert(local.id, &local.ty);
    }

    let mut out = Vec::new();
    for (loop_idx, l) in loops.loops.iter().enumerate() {
        let mut loop_blocks: Vec<usize> = l.blocks.clone();
        loop_blocks.sort_by_key(|&b| rpo_idx[b]);

        let mut defined_inside: HashSet<SsaLocal> = HashSet::new();
        for &b in &loop_blocks {
            for stmt in &body.blocks[b].stmts {
                if let MirStmt::Assign { dest: MirPlace::Ssa(s), .. } = stmt {
                    defined_inside.insert(*s);
                }
            }
        }
        if defined_inside.is_empty() {
            continue;
        }

        // 归纳变量：供"同步长 IV 之差不变"这条规则使用。
        let ivs = find_induction_vars(body, loops, loop_idx, &defined_inside, &local_tys);

        let mut invariant_set: HashSet<SsaLocal> = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            let mut not_yet: HashSet<SsaLocal> = HashSet::new();
            for s in &defined_inside {
                if !invariant_set.contains(s) {
                    not_yet.insert(*s);
                }
            }
            for &b in &loop_blocks {
                for stmt in &body.blocks[b].stmts {
                    let (dest, rv) = match stmt {
                        MirStmt::Assign { dest: MirPlace::Ssa(s), value } => (*s, value),
                        MirStmt::Assign { .. }
                        | MirStmt::ExprStmt(_)
                        | MirStmt::Drop { .. }
                        | MirStmt::SetMetadata { .. }
                        | MirStmt::EffectCheck { .. } => continue,
                    };
                    if invariant_set.contains(&dest) {
                        continue;
                    }
                    if rvalue_is_loop_invariant(rv, &not_yet, &ivs) {
                        invariant_set.insert(dest);
                        changed = true;
                    }
                }
            }
        }

        for &b in &loop_blocks {
            for stmt in &body.blocks[b].stmts {
                if let MirStmt::Assign { dest: MirPlace::Ssa(s), .. } = stmt {
                    if invariant_set.contains(s) {
                        out.push((loop_idx, *s));
                    }
                }
            }
        }
    }
    out
}

// not_yet：循环内定义、还没确认是不变量的 SSA。操作数不在里面 = 循环外定义
// 或者已经确认是不变量。ivs：当前循环的归纳变量表，只用于下面的"同步长 IV
// 之差"规则。
//
// 陷入的判定统一交给 detect::binop_may_trap（Div/Mod 的除数必须是非零且
// 非 -1 的整数常量）——这里不再自己写一份"v != 0"：-1 会让 MIN / -1 溢出，
// 与外提时 licm_speculation_safe 的口径必须一致。Cast 不陷入（见
// detect::cast_may_trap），只要操作数不变就是不变量。
fn rvalue_is_loop_invariant(
    rv: &MirRvalue,
    not_yet: &HashSet<SsaLocal>,
    ivs: &[InductionVar],
) -> bool {
    let op_ok = |op: &MirOperand| -> bool {
        match op {
            MirOperand::Constant(_) => true,
            MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => {
                !not_yet.contains(s)
            }
            MirOperand::Copy(_)
            | MirOperand::Move(_)
            | MirOperand::Sym(_)
            | MirOperand::Static(_) => false,
        }
    };
    match rv {
        MirRvalue::BinaryOp(op, l, r) => {
            if op_ok(l) && op_ok(r) {
                return !binop_may_trap(*op, r);
            }
            // 操作数不都是不变量时，只有一种形态仍可能不变：两个同步长
            // 归纳变量相减。i_k = i0 + k·s，j_k = j0 + k·s，差 = i0 - j0，
            // 与迭代次数无关。
            match op {
                BinaryOp::Sub => iv_difference_is_invariant(l, r, ivs),
                BinaryOp::Add
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
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
        MirRvalue::UnaryOp(_, o) => op_ok(o),
        MirRvalue::Cast(o, _) => op_ok(o),
        // 内存读 `Use(Copy/Move(Field/Index/EnumPayload ...))`：只在"地址不变"
        // 这一层算不变——place 里读到的每个 SSA（根变量、下标操作数）都不是
        // 循环内还没确认不变的值，并且不经过 Deref（不知道指向哪）。至于
        // 这块内存的*内容*会不会被循环里的写入改掉，这里不回答（要别名信息），
        // 由 simplify.rs 的 licm_hoistable 判；传递依赖于这条读的运算虽然也会被
        // 报告为不变，同样会被 licm_hoist 的传递闭包挡住。
        // 单纯的 SSA 拷贝 `y = x` 不在这里处理（交给复制传播）。
        MirRvalue::Use(op) => match op {
            MirOperand::Copy(MirPlace::Ssa(_)) | MirOperand::Move(MirPlace::Ssa(_)) => false,
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                if place_has_deref(place) {
                    return false;
                }
                let mut uses: Vec<SsaLocal> = Vec::new();
                ssa_uses_in_place(place, &mut uses);
                !uses.is_empty() && uses.iter().all(|s| !not_yet.contains(s))
            }
            MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => false,
        },
        MirRvalue::Call { .. }
        | MirRvalue::MethodCall { .. }
        | MirRvalue::StructInit { .. }
        | MirRvalue::EnumVariantConstruction { .. }
        | MirRvalue::Ref { .. }
        | MirRvalue::ArrayLiteral(_)
        | MirRvalue::Discriminant { .. }
        | MirRvalue::Phi { .. } => false,
    }
}

// ==================== 归纳变量分析（IVA） ====================
//
// 识别最简单也最常见的一类基本归纳变量：循环头里的 Phi
//     i = phi(init(循环外进来), next(所有回边))
// 其中 next 在循环内由 `next = i + stride` / `next = stride + i` /
// `next = i - stride` 唯一定义，stride 是循环不变量（整数常量，或者在
// 循环外定义的 SSA）。
//
// 保守约束（每一条都是为了让"差值不变"这个结论在语义上站得住）：
//   - 只收整数类型的 IV。浮点加法不满足结合律，(i0+s+s) - (j0+s+s) 和
//     i0 - j0 在舍入下不一定相等。
//   - 整数 Add/Sub 按回绕处理（detect.rs 的语言语义假设），所以模 2^n 下
//     i_k - j_k ≡ i0 - j0 严格成立。如果将来整数溢出改成陷入，这里的
//     "差值不变"规则要连同 binop_may_trap 一起重新审视。
//   - 所有回边必须送来同一个 next，且 next 在整个函数里只定义一次。
//   - 只在 Phi 本身（或它的 next）上谈论差值：i_next - j 这种一个超前一个
//     不超前的混合并不是不变量，不接受。
//
// 这一层只"识别"，不改 IR。IV 的差即使被报告为不变量，也不会被 LICM 外提
// ——它读了循环头的 Phi（循环内定义、不可外提），会被 licm_hoist 的传递闭包
// 挡住；真正想把它变成 `init_i - init_j` 需要另一个 pass 去改写。

/// 步长的可比较形态：整数常量，或者循环外定义的 SSA。`negated` 表示
/// next = i - stride（有效步长取反）。两个 IV 的有效步长相同当且仅当
/// StrideKey 与 negated 都相同。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrideKey {
    Const(i128),
    Ssa(SsaLocal),
}

#[derive(Debug, Clone)]
pub struct InductionVar {
    /// 循环头里定义这个归纳变量的 Phi 的目标。
    pub phi: SsaLocal,
    /// 从循环外进来的初值。多个入口前驱送来不同值时为 None。
    pub init: Option<MirOperand>,
    /// 步长操作数（原样保留，供改写 pass 使用）。
    pub stride: MirOperand,
    /// 步长的可比较形态，同步长判定只看它和 negated。
    pub stride_key: StrideKey,
    /// next = phi - stride 时为 true。
    pub negated: bool,
    /// 回边送回的下一个值（`next = phi ± stride` 的目标）。
    pub step: SsaLocal,
    /// 所属循环在 LoopInfo.loops 里的下标。
    pub loop_idx: usize,
}

fn operand_is_ssa(op: &MirOperand) -> Option<SsaLocal> {
    match op {
        MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => Some(*s),
        MirOperand::Copy(_)
        | MirOperand::Move(_)
        | MirOperand::Constant(_)
        | MirOperand::Sym(_)
        | MirOperand::Static(_) => None,
    }
}

/// 步长是不是循环不变量：整数常量，或者不在循环内定义的 SSA（函数参数、
/// 循环外的定义）。返回可比较形态；不满足返回 None。
fn stride_is_invariant(op: &MirOperand, defined_inside: &HashSet<SsaLocal>) -> Option<StrideKey> {
    if let Some(v) = int_literal_value(op) {
        return Some(StrideKey::Const(v));
    }
    match operand_is_ssa(op) {
        Some(s) => {
            if defined_inside.contains(&s) {
                None
            } else {
                Some(StrideKey::Ssa(s))
            }
        }
        None => None,
    }
}

/// 操作数是不是某个 IV 的 Phi（返回那个 IV）。
fn operand_is_iv_phi<'a>(op: &MirOperand, ivs: &'a [InductionVar]) -> Option<&'a InductionVar> {
    let s = operand_is_ssa(op)?;
    ivs.iter().find(|iv| iv.phi == s)
}

/// 操作数是不是某个 IV 的 next（回边送回的那个值）。
fn operand_is_iv_step<'a>(op: &MirOperand, ivs: &'a [InductionVar]) -> Option<&'a InductionVar> {
    let s = operand_is_ssa(op)?;
    ivs.iter().find(|iv| iv.step == s)
}

/// `l - r` 是不是两个同步长归纳变量的差：两边同为 Phi，或两边同为 next，
/// 且有效步长相同。
fn iv_difference_is_invariant(l: &MirOperand, r: &MirOperand, ivs: &[InductionVar]) -> bool {
    if let (Some(a), Some(b)) = (operand_is_iv_phi(l, ivs), operand_is_iv_phi(r, ivs)) {
        return a.stride_key == b.stride_key && a.negated == b.negated;
    }
    if let (Some(a), Some(b)) = (operand_is_iv_step(l, ivs), operand_is_iv_step(r, ivs)) {
        return a.stride_key == b.stride_key && a.negated == b.negated;
    }
    false
}

/// 本类型是不是整数（剥掉隐私包装）。Type 变体很多，除了这十个整数类型和
/// 隐私包装之外都不是整数，通配在这里是对它们的正确回答（项目里
/// simplify.rs::type_is_float 同样处理）。
fn type_is_int(ty: &Type) -> bool {
    match ty {
        Type::I8
        | Type::I16
        | Type::I32
        | Type::I64
        | Type::I128
        | Type::U8
        | Type::U16
        | Type::U32
        | Type::U64
        | Type::U128 => true,
        Type::Privacy(inner, _) => type_is_int(inner),
        _ => false,
    }
}

/// 找出 loop_idx 这个循环的所有基本归纳变量。defined_inside 是该循环内定义
/// 的全部 SSA（调用方已经算好，避免重复遍历）。
pub fn find_induction_vars(
    body: &MirBody,
    loops: &LoopInfo,
    loop_idx: usize,
    defined_inside: &HashSet<SsaLocal>,
    local_tys: &HashMap<usize, &Type>,
) -> Vec<InductionVar> {
    let mut out: Vec<InductionVar> = Vec::new();
    let l = match loops.loops.get(loop_idx) {
        Some(l) => l,
        None => return out,
    };
    if l.header >= body.blocks.len() {
        return out;
    }

    // 循环内每个 SSA 的定义语句；定义次数大于一次的记入 multi，不当 next。
    let mut def_of: HashMap<SsaLocal, &MirRvalue> = HashMap::new();
    let mut multi: HashSet<SsaLocal> = HashSet::new();
    for &b in &l.blocks {
        for stmt in &body.blocks[b].stmts {
            if let MirStmt::Assign { dest: MirPlace::Ssa(s), value } = stmt {
                if def_of.insert(*s, value).is_some() {
                    multi.insert(*s);
                }
            }
        }
    }

    for stmt in &body.blocks[l.header].stmts {
        let (phi, values) = match stmt {
            MirStmt::Assign { dest: MirPlace::Ssa(p), value: MirRvalue::Phi { values } } => (*p, values),
            MirStmt::Assign { .. }
            | MirStmt::ExprStmt(_)
            | MirStmt::Drop { .. }
            | MirStmt::SetMetadata { .. }
            | MirStmt::EffectCheck { .. } => continue,
        };
        // 只收整数 IV。
        match local_tys.get(&phi.base_id) {
            Some(ty) => {
                if !type_is_int(ty) {
                    continue;
                }
            }
            None => continue,
        }

        // 把 Phi 的入边按"来自循环内/外"分开。
        let mut outside: Vec<&MirOperand> = Vec::new();
        let mut back_next: Option<SsaLocal> = None;
        let mut ok = true;
        for (pred, op) in values {
            if l.contains(*pred) {
                let s = match operand_is_ssa(op) {
                    Some(s) => s,
                    None => {
                        ok = false;
                        break;
                    }
                };
                match back_next {
                    None => back_next = Some(s),
                    Some(prev) => {
                        if prev != s {
                            ok = false;
                            break;
                        }
                    }
                }
            } else {
                outside.push(op);
            }
        }
        let next = match (ok, back_next) {
            (true, Some(n)) => n,
            (true, None) | (false, _) => continue,
        };
        if outside.is_empty() || multi.contains(&next) {
            continue;
        }
        let def = match def_of.get(&next) {
            Some(d) => *d,
            None => continue,
        };
        // next = phi + stride / stride + phi / phi - stride
        let (stride_op, negated) = match def {
            MirRvalue::BinaryOp(BinaryOp::Add, a, b) => {
                if operand_is_ssa(a) == Some(phi) {
                    (b, false)
                } else if operand_is_ssa(b) == Some(phi) {
                    (a, false)
                } else {
                    continue;
                }
            }
            MirRvalue::BinaryOp(BinaryOp::Sub, a, b) => {
                if operand_is_ssa(a) == Some(phi) {
                    (b, true)
                } else {
                    continue;
                }
            }
            MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Cast(_, _)
            | MirRvalue::Use(_)
            | MirRvalue::Call { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. } => continue,
        };
        let stride_key = match stride_is_invariant(stride_op, defined_inside) {
            Some(k) => k,
            None => continue,
        };
        // 步长不能就是 phi 自己（`i + i` 是倍增，不是线性归纳变量）。
        if operand_is_ssa(stride_op) == Some(phi) {
            continue;
        }
        // 初值：所有循环外入边送来同一个操作数才记录。
        let first = outside[0];
        let init = if outside.iter().all(|o| *o == first) { Some(first.clone()) } else { None };
        out.push(InductionVar {
            phi,
            init,
            stride: stride_op.clone(),
            stride_key,
            negated,
            step: next,
            loop_idx,
        });
    }
    out
}

// ==================== 分析结果打包 ====================
//
// 关键设计：这里只是一个纯数据容器，没有 build 方法。build 要调用
// Simplify::sccp（将来还有 gvn），如果它跟着 AnalysisResults 搬进
// analyze.rs，analyze 就得 use simplify::Simplify，而 simplify 又要
// use analyze::AnalysisResults，形成循环引用。所以分析结果的"长什么
// 样"放在这里，"怎么算出来、按什么顺序编排"留在 simplify.rs 的
// Simplify::analyze——analyze.rs 只依赖 mir，不依赖任何优化 pass，
// 依赖方向始终是 simplify -> analyze，单向。
//
// 以后每加一种新的整函数分析（活跃变量……），都往这个 struct 加一个
// 字段，逐块重写的 fold_xxx 系列函数统一收 &AnalysisResults，签名不用
// 跟着变。目前这几层：
//   cfg      —— L0 基础设施（前驱/后继/支配树/rpo/可达性），不依赖其它分析
//   sccp     —— L1 分析：哪些 SSA 是常量、哪些块/边可执行
//   gvn      —— L1 分析：哪些 SSA 值和另一个支配它的 SSA 值相等；依赖
//               cfg 的 rpo 和 dominates，以及 sccp 的常量结论
//   loops    —— L1 分析：自然循环、循环树、preheader；依赖 cfg
//   alias    —— L1 分析：按块的 Deref 读/写位图；依赖函数纯度表（调用
//               可能经由引用写调用者的内存）
//   purity—— 全程序的函数纯度表，由 Simplify::run 一次性算好，不随
//               单个函数的 body 变化。用 Arc 共享：每个函数、每轮迭代都要
//               构造一份 AnalysisResults，直接 clone HashMap 会让总开销
//               随"函数数 × 函数数 × 迭代数"增长。
#[allow(dead_code)]
pub struct AnalysisResults {
    pub cfg: CfgInfo,
    pub sccp: SccpResult,
    pub gvn: GvnResult,
    pub loops: LoopInfo,
    pub alias: AliasInfo,
    pub available: AvailableExprs,
    pub purity: Arc<HashMap<String, Purity>>,
}

impl AnalysisResults {
    /// 这份快照是否还对得上 body。所有分析结果都按块编号索引，任何会
    /// 重编号/增删块的 pass（collapse_empty_blocks、remove_dead_blocks）
    /// 执行之后，快照就作废了。依赖快照的 pass 入口先调这个，对不上就
    /// 放弃优化，而不是拿错位的编号去查——错位查出来的不是 panic 就是
    /// 悄悄误删活块。
    ///
    /// 先比修订号，再比各张表的长度：修订号拦住"块数没变、编号被打乱"，
    /// 长度检查拦住"某张表自己没跟上"。只改边、只挪语句的 pass 不会让
    /// 修订号变化（见 MirBody::revision），这里不会误判失效。
    pub fn matches(&self, body: &MirBody) -> bool {
        let n = body.blocks.len();
        self.cfg.revision() == body.revision
            && self.cfg.block_count() == n
            && self.sccp.exec_blocks.len() == n
            && self.loops.innermost_loop_of.len() == n
            && self.alias.block_writes_len() == n
            && self.alias.block_reads_len() == n
            && self.available.available_in.len() == n
            && self.available.available_out.len() == n
            && self.available.holder_at_out.len() == n
    }

    /// 各项分析之间的内部一致性检查（给 debug_assert 用，release 不跑）。
    /// 每加一种新分析（比如 GVN），把它跟 cfg/sccp 之间必须成立的关系
    /// 也补在这里，出问题时能在分析刚算完的地方就暴露，而不是等到重写
    /// 阶段才以怪异的优化结果表现出来。
    pub fn is_consistent(&self) -> bool {
        let n = self.cfg.block_count();
        if self.sccp.exec_blocks.len() != n {
            return false;
        }
        if self.loops.innermost_loop_of.len() != n {
            return false;
        }
        if self.alias.block_writes_len() != n || self.alias.block_reads_len() != n {
            return false;
        }
        if self.available.available_in.len() != n || self.available.available_out.len() != n {
            return false;
        }
        if self.available.holder_at_out.len() != n {
            return false;
        }
        if n == 0 {
            return true;
        }
        // 入口块恒可执行。
        if !self.sccp.exec_blocks[0] {
            return false;
        }
        // SCCP 可执行的块一定在结构上可达（可执行是可达的子集）。
        let reachable = self.cfg.compute_reachable(n);
        for b in 0..n {
            if self.sccp.exec_blocks[b] && !reachable[b] {
                return false;
            }
        }
        // 可执行边必须是 CFG 里真实存在的边，且两端都可执行。
        for &(from, to) in &self.sccp.exec_edges {
            if from >= n || to >= n {
                return false;
            }
            if !self.sccp.exec_blocks[from] || !self.sccp.exec_blocks[to] {
                return false;
            }
            if !self.cfg.successors[from].contains(&to) {
                return false;
            }
        }

        // GVN 的契约。这几条里"无链"是靠 gvn() 的遍历顺序隐式保证的
        // （只有没被改写的值才会进候选表），一旦将来改了遍历顺序或
        // 放宽了候选条件就会悄悄破掉——写成显式检查，出问题时在分析
        // 刚算完的地方就暴露，而不是等到 fold_operand 读到一个"其实还能
        // 继续往下换"的代表时才以奇怪的结果表现出来。
        for (dest, rep) in &self.gvn.rewrite {
            if dest == rep {
                return false;
            }
            if self.gvn.rewrite.contains_key(rep) {
                return false;
            }
            // 每一项都必须记着定义块，并且代表的定义块支配 dest 的定义块
            // ——这是"凡是读 dest 的地方都能改成读 rep"的全部依据。
            match self.gvn.sites.get(dest) {
                Some(&(dest_block, rep_block)) => {
                    if dest_block >= n || rep_block >= n {
                        return false;
                    }
                    if !self.cfg.dominates(rep_block, dest_block) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        if self.gvn.sites.len() != self.gvn.rewrite.len() {
            return false;
        }
        // 循环信息自己的一致性（循环头支配循环体、出口边/前驱边合法、循环树
        // 父子互相指对等）。检查本身在 LoopInfo 里，这里只是调用一次。
        if !self.loops.check_consistency(&self.cfg) {
            return false;
        }
        true
    }
}

