// control.rs
use crate::analyze::{CfgInfo, LoopInfo};
use crate::detect::{
    block_contains_phi, is_marker_block, multiply_defined_ssa, ssa_def_blocks, stmt_is_phi,
    terminators_equivalent,
};
use crate::mir::*;
use std::collections::{HashMap, HashSet};

pub struct Control;

impl Control {
    pub fn simplify(program: &mut MirProgram) {
        for f in &mut program.fns {
            Self::simplify_function(f);
        }
    }

    // 关键修复（健壮性）：原来这个循环没有任何轮数上限——只要每一轮
    // block 数量还在变就会一直跑下去。加一个保守的上限纯粹是防御性
    // 的，参照 simplify.rs 那边 MAX_OPT_ITERATIONS 的同一个考虑。
    const MAX_ITERATIONS: usize = 64;

    // 关键重构：收敛判断原来是"比较这一轮前后的块数"——这漏掉了一大类
    // 真实发生了的化简：simplify_switch 把 Switch 退化成 Goto、
    // fold_single_pred_phis 把 Phi 退化成 Use、tail_merge 把重复块的
    // 引用改指向代表块，这些都不改变块数，却是实打实的改写，后续的
    // pass（乃至下一轮的这些 pass 自己）可能因此获得新的化简机会。用
    // 块数当收敛信号，会让这类改写只做一轮就被判定为"没变化"提前退出。
    //
    // 现在每个子过程自己汇报"这一轮里是不是真的动过 IR"（bool），只要
    // 有一个动过，这一轮就不算收敛，继续跑下一轮；全部报告"没动"才真正
    // 停下来。
    //
    // 顺序上 remove_unreachable_blocks 排在最后：它是唯一一个会重新
    // 编号块的 pass，前面几个 pass 都按当前的块 id 认路（Goto 目标、Phi
    // 的 incoming 前驱 id……），必须先让它们把这一轮能做的改写都做完，
    // 再统一重新编号，否则前面几个 pass 用的 block id 和重新编号之后的
    // 状态对不上。
    fn simplify_function(f: &mut MirFn) {
        for _ in 0..Self::MAX_ITERATIONS {
            let mut changed = false;
            changed |= Self::simplify_switch(&mut f.body);
            changed |= Self::simplify_redundant_if(&mut f.body);
            changed |= Self::thread_single_pred_branches(&mut f.body);
            changed |= Self::fold_single_pred_phis(&mut f.body);
            changed |= Self::fold_marker_blocks(&mut f.body);
            changed |= Self::chain_gotos(&mut f.body);
            changed |= Self::merge_blocks(&mut f.body);
            changed |= Self::tail_merge(&mut f.body);
            changed |= Self::extract_common_suffix(&mut f.body);
            // 关键说明：放在所有会穿过/合并空块的 pass 之后、重编号的
            // remove_unreachable_blocks 之前。preheader 本身是一个空的
            // `Goto(循环头)` 块，chain_gotos/tail_merge（以及 simplify.rs 的
            // collapse_empty_blocks）都会把它当普通空跳板处理——它们靠
            // LoopInfo::preheader_blocks 识别并跳过专用 preheader，否则这里
            // 补、它们拆，每轮都报告 changed，不动点循环永远不收敛。
            changed |= Self::insert_preheaders(&mut f.body);

            let cfg = CfgInfo::build(&f.body);
            changed |= Self::remove_unreachable_blocks(&mut f.body, &cfg);

            if !changed {
                break;
            }
        }
    }

    fn simplify_switch(body: &mut MirBody) -> bool {
        let mut changed = false;
        for block in &mut body.blocks {
            // 关键修复：`MirTerminator::Switch` 这一轮多了一个
            // `discr_ty` 字段，这里原来解构的三个字段（targets/default/
            // discr）没有把它列进来，也没有用 `..` 兜底剩余字段——
            // struct 模式必须覆盖所有字段，缺一个就编译不过（E0027）。
            if let MirTerminator::Switch { targets, default, discr: _, discr_ty: _ } = &block.terminator {
                if targets.is_empty() {
                    block.terminator = MirTerminator::Goto(*default);
                    changed = true;
                } else if targets.iter().all(|(_, t)| *t == targets[0].1) && targets[0].1 == *default {
                    block.terminator = MirTerminator::Goto(targets[0].1);
                    changed = true;
                }
            }
        }
        changed
    }

    // ---- 冗余的 If ----
    // 关键新增：`if c then A else B`，而 A、B 都是不含任何语句、直接
    // `Goto(X)` 的空块——不管 c 是什么，最后都到 X，这个 If 是多余的，换成
    // `Goto(X)`。then_block == else_block 是它的退化特例（同一个块当然
    // 同一个目标）。跟 simplify_switch 同一层：纯结构化简，跟常量无关，
    // 所以放在 control.rs，不放在依赖 SCCP 的 simplify.rs::fold_terminator
    // 里。
    //
    // 关键修复（示例方案漏掉的前提）：X 带 Phi 时不能这样化简。X 的 Phi
    // 里 incoming 是按前驱 A、B 分别记的，`(A, v1), (B, v2)` 在 v1 != v2 时
    // 表示"取决于 c 走了哪条路"；把 If 换成 Goto 之后 A、B 变成不可达，
    // 删块时它们名下的 incoming 被一起丢掉，c 选值的语义就凭空消失了，Phi
    // 要么没有 incoming、要么悄悄只剩一边。严格说 v1 == v2 时仍然可以化简，
    // 但那需要把两条 incoming 合并成一条记在当前块名下，这里不做——X 有
    // Phi 一律跳过，宁可少优化。then_block == else_block 的特例不受此限：
    // 当前块到 X 的边一条没少，Phi 里记的前驱不变。
    fn simplify_redundant_if(body: &mut MirBody) -> bool {
        let n = body.blocks.len();
        if n == 0 {
            return false;
        }

        // 每个块"如果是空的纯 Goto 跳板"，记下它的目标。
        let mut goto_target: Vec<Option<usize>> = Vec::with_capacity(n);
        for block in &body.blocks {
            if !block.stmts.is_empty() {
                goto_target.push(None);
                continue;
            }
            match &block.terminator {
                MirTerminator::Goto(t) => goto_target.push(Some(*t)),
                MirTerminator::If { .. }
                | MirTerminator::Switch { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable
                | MirTerminator::Placeholder => goto_target.push(None),
            }
        }
        // 每个块有没有 Phi，提前算好：下面的循环对 body.blocks 是可变遍历，
        // 不能一边改一边再借用 body.blocks 去查别的块。
        let mut has_phi: Vec<bool> = Vec::with_capacity(n);
        for block in &body.blocks {
            has_phi.push(block_contains_phi(block));
        }

        let mut changed = false;
        for block in body.blocks.iter_mut() {
            let (then_block, else_block) = match &block.terminator {
                MirTerminator::If { then_block, else_block, .. } => (*then_block, *else_block),
                MirTerminator::Goto(_)
                | MirTerminator::Switch { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable
                | MirTerminator::Placeholder => continue,
            };
            if then_block == else_block {
                block.terminator = MirTerminator::Goto(then_block);
                changed = true;
                continue;
            }
            let t = goto_target.get(then_block).copied().flatten();
            let e = goto_target.get(else_block).copied().flatten();
            if let (Some(t), Some(e)) = (t, e) {
                // 目标越界（畸形 MIR）按"有 Phi"同样处理：查不到就不动。
                let target_has_phi = match has_phi.get(t) {
                    Some(p) => *p,
                    None => true,
                };
                if t == e && !target_has_phi {
                    block.terminator = MirTerminator::Goto(t);
                    changed = true;
                }
            }
        }
        changed
    }

    // ---- 单前驱分支的跳转线程化 ----
    // 块 b 只有一个前驱 p，p 的终止器已经决定了"进入 b 时某个值是什么"，而 b
    // 自己的终止器又在对同一个值做分支：那么 b 的分支只有一个方向可能发生，
    // 直接改成 `Goto`。两种形态：
    //   - If 套 If：两个条件解析到同一个布尔根（只差若干层 Copy / `!`），
    //     走的是 p 的 then 边还是 else 边就决定了 b 的条件真假（thread_if_if）；
    //   - Switch 套 Switch：两个判别式解析到同一个 SSA，p 经由唯一一个
    //     `值 -> b` 的 case 进来，就知道判别式等于那个值，直接查 b 的
    //     case 表（thread_switch_switch）。
    //
    // 为什么靠 SSA 就够：SSA 值只定义一次、之后不变，p 的出口和 b 的入口之间
    // 同一个名字读到的一定是同一个值。
    //
    // 只改终止器、不增删块，修订号不变。b 原来的各个目标里，凡是不再是新目标的，
    // 都要摘掉它们 Phi 里记着的"来自 b"的 incoming——这条边已经不存在了，
    // 留着会让 Phi 的形状和前驱表对不上。
    //
    // 保守的地方：
    //   - 入口块（块 0）除了前驱表里的边还有一条"函数入口"的隐式进入，不当
    //     单前驱处理；
    //   - p == b 跳过；
    //   - 条件/判别式的追溯只在规范 SSA 上做：同名多定义的 SSA 追溯到的定义
    //     不可信，追溯过程中遇到就放弃。
    fn thread_single_pred_branches(body: &mut MirBody) -> bool {
        let n = body.blocks.len();
        if n <= 1 {
            return false;
        }

        let def_block = ssa_def_blocks(body);
        let multi = multiply_defined_ssa(body);
        let preds = body.predecessor_table();

        let mut changed = false;
        for b in 1..n {
            if preds[b].len() != 1 {
                continue;
            }
            let p = preds[b][0];
            if p == b || p >= n {
                continue;
            }

            // 先只读地决定目标，再改写——判定函数读整个 body（追溯定义）。
            let target = match Self::thread_if_if(
                &body.blocks[p].terminator,
                &body.blocks[b].terminator,
                b,
                &def_block,
                &multi,
                body,
            ) {
                Some(t) => Some(t),
                None => Self::thread_switch_switch(
                    &body.blocks[p].terminator,
                    &body.blocks[b].terminator,
                    b,
                    &def_block,
                    &multi,
                    body,
                ),
            };
            let target = match target {
                Some(t) => t,
                None => continue,
            };
            // 目标块越界（畸形 MIR）不动。
            if target >= n {
                continue;
            }

            let b_old_targets: Vec<usize> = body.blocks[b].terminator.targets();

            if body.blocks[b].terminator != MirTerminator::Goto(target) {
                body.blocks[b].terminator = MirTerminator::Goto(target);
                changed = true;
            }

            // b 不再通向的那些块：摘掉它们 Phi 里来自 b 的 incoming。
            let mut seen: HashSet<usize> = HashSet::new();
            for t in b_old_targets {
                if t == target || t >= n || !seen.insert(t) {
                    continue;
                }
                for stmt in &mut body.blocks[t].stmts {
                    if let MirStmt::Assign { value: MirRvalue::Phi { values }, .. } = stmt {
                        values.retain(|(pred, _)| *pred != b);
                    }
                }
            }
        }
        changed
    }

    // If 套 If。返回 b 的终止器应该改成跳去的目标块。
    fn thread_if_if(
        p_term: &MirTerminator,
        b_term: &MirTerminator,
        b: usize,
        def_block: &HashMap<SsaLocal, usize>,
        multi: &HashSet<SsaLocal>,
        body: &MirBody,
    ) -> Option<usize> {
        let (p_cond, p_then, p_else) = match p_term {
            MirTerminator::If { cond, then_block, else_block } => (cond, *then_block, *else_block),
            MirTerminator::Goto(_)
            | MirTerminator::Switch { .. }
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => return None,
        };
        // p 的两条边都指向 b 时，进来不说明任何事。
        let from_then = p_then == b && p_else != b;
        let from_else = p_else == b && p_then != b;
        if !from_then && !from_else {
            return None;
        }

        let (b_cond, b_then, b_else) = match b_term {
            MirTerminator::If { cond, then_block, else_block } => (cond, *then_block, *else_block),
            MirTerminator::Goto(_)
            | MirTerminator::Switch { .. }
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => return None,
        };

        let p_norm = Self::resolve_boolean(p_cond, def_block, multi, body);
        let b_norm = Self::resolve_boolean(b_cond, def_block, multi, body);
        match (p_norm, b_norm) {
            (Some((p_pol, p_base)), Some((b_pol, b_base))) if p_base == b_base => {
                // 进入 b 时 p 的条件为真（from_then）或为假（from_else）；
                // 两边极性相同则 b 的条件同值，否则相反。
                let p_cond_true = from_then;
                let b_cond_true = if p_pol == b_pol { p_cond_true } else { !p_cond_true };
                Some(if b_cond_true { b_then } else { b_else })
            }
            _ => None,
        }
    }

    // Switch 套 Switch。返回 b 的终止器应该改成跳去的目标块。
    //
    // p 必须恰好通过一个 case 值进入 b，并且 b 不是 p 的 default：default 意味着
    // "不等于任何 case 值"，说明不了判别式是什么；同一个 b 对应多个 case 值时
    // 也无法确定是哪一个。两个 Switch 的 discr_ty 必须相同，这样两边的字面量
    // 表示一致，才能直接用 `==` 在 b 的 case 表里查。
    fn thread_switch_switch(
        p_term: &MirTerminator,
        b_term: &MirTerminator,
        b: usize,
        def_block: &HashMap<SsaLocal, usize>,
        multi: &HashSet<SsaLocal>,
        body: &MirBody,
    ) -> Option<usize> {
        let (p_discr, p_ty, p_targets, p_default) = match p_term {
            MirTerminator::Switch { discr, discr_ty, targets, default } => {
                (discr, discr_ty, targets, *default)
            }
            MirTerminator::Goto(_)
            | MirTerminator::If { .. }
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => return None,
        };
        if p_default == b {
            return None;
        }

        let mut from_value: Option<&Literal> = None;
        let mut count = 0usize;
        for (v, t) in p_targets {
            if *t == b {
                count += 1;
                from_value = Some(v);
            }
        }
        if count != 1 {
            return None;
        }
        let from_value = from_value?;

        let (b_discr, b_ty, b_targets, b_default) = match b_term {
            MirTerminator::Switch { discr, discr_ty, targets, default } => {
                (discr, discr_ty, targets, *default)
            }
            MirTerminator::Goto(_)
            | MirTerminator::If { .. }
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => return None,
        };
        if p_ty != b_ty {
            return None;
        }

        let p_root = Self::resolve_value(p_discr, def_block, multi, body)?;
        let b_root = Self::resolve_value(b_discr, def_block, multi, body)?;
        if p_root != b_root {
            return None;
        }
        // 在 b 的 case 表里查这个值；表里没有就落到 b 的 default。
        for (val, t) in b_targets {
            if val == from_value {
                return Some(*t);
            }
        }
        Some(b_default)
    }

    // 沿 `y = x`（Copy/Move 的 Use）追溯到值的根 SSA。不翻转任何东西，所以
    // 对任意类型的判别式都成立。走到同名多定义的 SSA、找不到定义块、有环
    // 一律返回 None；参数（没有定义语句）自己就是根。
    fn resolve_value(
        op: &MirOperand,
        def_block: &HashMap<SsaLocal, usize>,
        multi: &HashSet<SsaLocal>,
        body: &MirBody,
    ) -> Option<SsaLocal> {
        let mut ssa = Self::operand_as_ssa(op)?;
        let mut visited: HashSet<SsaLocal> = HashSet::new();
        loop {
            if multi.contains(&ssa) || !visited.insert(ssa) {
                return None;
            }
            let db = match def_block.get(&ssa) {
                Some(&b) => b,
                None => return Some(ssa),
            };
            let block = body.blocks.get(db)?;

            let mut def: Option<&MirRvalue> = None;
            for stmt in &block.stmts {
                if let MirStmt::Assign { dest: MirPlace::Ssa(s), value } = stmt {
                    if *s == ssa {
                        def = Some(value);
                        break;
                    }
                }
            }
            let value = def?;
            match value {
                MirRvalue::Use(MirOperand::Copy(MirPlace::Ssa(t)))
                | MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(t))) => {
                    ssa = *t;
                }
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
                | MirRvalue::Discriminant { .. }
                | MirRvalue::Phi { .. } => return Some(ssa),
            }
        }
    }

    fn operand_as_ssa(op: &MirOperand) -> Option<SsaLocal> {
        match op {
            MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => Some(*s),
            MirOperand::Copy(_)
            | MirOperand::Move(_)
            | MirOperand::Constant(_)
            | MirOperand::Sym(_)
            | MirOperand::Static(_) => None,
        }
    }

    // 把一个布尔条件追溯到它的"根"：返回 (极性, 根 SSA)，条件 == 根（极性为
    // true）或 == !根（极性为 false）。沿 `y = x`（Copy/Move 的 Use）和
    // `y = !x` 往上走；遇到别的定义形态、参数（没有定义语句）就把当前 SSA
    // 当根。走到同名多定义的 SSA、找不到定义块、有环，一律返回 None（放弃）。
    //
    // `!` 只会出现在 bool 上：条件本身是 bool，`!` 的结果是 bool，操作数就只能
    // 是 bool，所以翻转极性是对的（整数的按位取反不会出现在条件链里）。
    fn resolve_boolean(
        op: &MirOperand,
        def_block: &HashMap<SsaLocal, usize>,
        multi: &HashSet<SsaLocal>,
        body: &MirBody,
    ) -> Option<(bool, SsaLocal)> {
        let mut ssa = Self::operand_as_ssa(op)?;
        let mut polarity = true;
        let mut visited: HashSet<SsaLocal> = HashSet::new();
        loop {
            if multi.contains(&ssa) || !visited.insert(ssa) {
                return None;
            }
            let db = match def_block.get(&ssa) {
                Some(&b) => b,
                // 没有定义语句：函数参数之类，它自己就是根。
                None => return Some((polarity, ssa)),
            };
            let block = body.blocks.get(db)?;

            let mut def: Option<&MirRvalue> = None;
            for stmt in &block.stmts {
                if let MirStmt::Assign { dest: MirPlace::Ssa(s), value } = stmt {
                    if *s == ssa {
                        def = Some(value);
                        break;
                    }
                }
            }
            // def_block 说它在这个块里定义，却找不到——表和 body 对不上，放弃。
            let value = def?;
            match value {
                MirRvalue::Use(MirOperand::Copy(MirPlace::Ssa(t)))
                | MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(t))) => {
                    ssa = *t;
                }
                MirRvalue::UnaryOp(UnaryOp::Not, inner) => match Self::operand_as_ssa(inner) {
                    Some(t) => {
                        polarity = !polarity;
                        ssa = t;
                    }
                    None => return Some((polarity, ssa)),
                },
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
                | MirRvalue::Discriminant { .. }
                | MirRvalue::Phi { .. } => return Some((polarity, ssa)),
            }
        }
    }

    // ---- 单前驱块的 Phi 退化 ----
    // 关键新增：只有一个前驱的块，它的 Phi 只有一条 incoming，本质上
    // 就是"从那一个前驱搬一个值过来"，退化成普通的 Use 语义完全等价。
    // 这一步必须跑在 merge_blocks 之前：merge_blocks 会把"唯一后继是
    // 自己的那个前驱"整个并进来，如果 succ 自己还带着 Phi，Phi 会被原样
    // 搬进 pred 语句列表的中间——但 Phi 只有出现在块开头才有意义（它的
    // 语义是"进入这个块的那一刻，根据来路取值"），出现在块中间会让
    // 后续假定"Phi 只在块首"的 pass（比如 simplify.rs 的 simplify_phi）
    // 判断出错。跑完这一步之后，单前驱块里不会再剩下 Phi，merge_blocks
    // 不需要对此单独处理。
    fn fold_single_pred_phis(body: &mut MirBody) -> bool {
        let n = body.blocks.len();
        if n <= 1 {
            return false;
        }

        // 这里只需要"每个块有几个不同的直接前驱"，用不着完整的前驱表，
        // 所以保留一份自己的计数遍历——但"终止器指向哪些块"这件事交给
        // MirTerminator::for_each_target，不再自己展开。
        //
        // 同一个前驱块用多条边指向同一个目标（If 两边同一目标、Switch
        // 多个值落到同一目标），只算一个前驱：同一个块的所有边是连续
        // 访问的，只需要记住"上一个来源是谁"就能去重。越界目标用
        // `.get_mut` 直接跳过。
        let mut pred_count = vec![0usize; n];
        let mut last_pred: Vec<Option<usize>> = vec![None; n];
        for (src, block) in body.blocks.iter().enumerate() {
            block.terminator.for_each_target(|t| {
                let (Some(count), Some(last)) = (pred_count.get_mut(t), last_pred.get_mut(t)) else {
                    return;
                };
                if *last == Some(src) {
                    return;
                }
                *count += 1;
                *last = Some(src);
            });
        }

        let mut changed = false;
        for i in 0..n {
            if pred_count[i] != 1 {
                continue;
            }
            let stmts = std::mem::take(&mut body.blocks[i].stmts);
            let mut new_stmts = Vec::with_capacity(stmts.len());
            for stmt in stmts {
                match stmt {
                    MirStmt::Assign { dest, value: MirRvalue::Phi { mut values } } if values.len() == 1 => {
                        let (_, op) = values.pop().unwrap();
                        new_stmts.push(MirStmt::Assign { dest, value: MirRvalue::Use(op) });
                        changed = true;
                    }
                    other => new_stmts.push(other),
                }
            }
            body.blocks[i].stmts = new_stmts;
        }
        changed
    }

    // ---- 标记块下沉 ----
    // 关键新增：有些块除了 SetMetadata/EffectCheck 这类"标记"语句之外
    // 什么都没有，终结指令又是纯 Goto——这种块本该被 chain_gotos 直接
    // 穿透，但 chain_gotos 判断"能不能穿透"用的是 `stmts.is_empty()`
    // （跟 simplify.rs 的 collapse_empty_blocks 同一条件），只要块里还有
    // 语句就不穿。与其放宽 chain_gotos 的穿透条件（那样职责就模糊了：
    // 一个只管"追链"的函数还要顺带决定"哪些语句可以被跳过"），不如让
    // 这一步先把标记语句挪到目标块开头（Phi 之后），把标记块清空成
    // 真正的空块——chain_gotos 不用改，自然就能穿透了。
    //
    // 只在目标块*只有这一个标记块作为前驱*时才挪：如果目标块还有别的
    // 前驱，那些路径原本不会执行这条标记（标记只属于"经过标记块"的
    // 那条路径），挪过去会让它们凭空多执行一次，语义就变了。链式标记
    // （M1 -> M2 -> T，两个都是标记块）不奢求一轮挪到底，只挪一层，
    // 剩下的交给外层循环下一轮处理——`is_marker[target]` 的检查把这一点
    // 明确固定下来，不在一次调用里递归下沉。
    fn fold_marker_blocks(body: &mut MirBody) -> bool {
        let n = body.blocks.len();
        if n <= 1 {
            return false;
        }

        // 前驱表统一来自 MirBody::predecessor_table：去重、过滤越界、按块
        // id 升序，不在这里再扫一遍终止器。
        let preds = body.predecessor_table();

        // 入口块（id 0）不当标记块：它是函数的起点，不存在"前驱"这个
        // 概念，把它的语句挪去别处没有意义。
        // "块自身是不是标记块"的判定在 detect::is_marker_block；"入口块不算"
        // 是这里的调用点语义，不是块自身的属性，所以在这里加 `i != 0`。
        let mut is_marker: Vec<bool> = Vec::with_capacity(n);
        for i in 0..n {
            is_marker.push(i != 0 && is_marker_block(&body.blocks[i]));
        }

        let mut changed = false;
        for i in 0..n {
            if !is_marker[i] {
                continue;
            }
            let target = match body.blocks[i].terminator {
                MirTerminator::Goto(t) => t,
                _ => continue,
            };
            // target == i 理论上不会发生（is_marker 要求终结指令是纯
            // Goto，一个块 Goto 自己会在 chain_gotos 里被环检测挡住，不
            // 会一直是这个形状），但下标操作前照旧防一手越界/自环。
            if target >= n || target == i {
                continue;
            }
            // 目标本身也是标记块：留给外层循环下一轮，先处理完这一层。
            if is_marker[target] {
                continue;
            }
            if preds[target].len() != 1 || preds[target][0] != i {
                continue;
            }

            let marker_stmts: Vec<MirStmt> = body.blocks[i].stmts.drain(..).collect();
            // 插在目标块所有 Phi 之后——Phi 只能出现在块首，标记语句插进
            // 两条 Phi 中间会破坏这条不变量。
            let insert_at = {
                let b = &body.blocks[target];
                b.stmts.iter().take_while(|s| stmt_is_phi(s)).count()
            };
            body.blocks[target].stmts.splice(insert_at..insert_at, marker_stmts);
            changed = true;
        }
        changed
    }

    // 关键修复（会悄悄删代码的 bug）：原来这里只看 target_block 的
    // terminator 是不是 Goto，完全没检查 target_block 自己有没有真实
    // 语句——如果 A -> B -> C，B 里其实有真实语句（哪怕只是一句
    // print），原来的写法照样会把 A 直接改成 Goto(C)，绕过 B、永远不
    // 再执行 B 的那些语句。这不是优化，是丢代码：只有当 B 是一个
    // "纯跳转跳板"（自己没有任何语句，terminator 又是 Goto）时，跳过
    // B 直接指向它的目标才是安全的——这跟 simplify.rs 的
    // collapse_empty_blocks 判断"能不能折叠"用的是同一条件
    // （`block.stmts.is_empty()`），这里补上。带标记的"假空块"由
    // fold_marker_blocks 先清空，这里不需要关心标记语句。
    //
    // 关键修复（原来只追一跳，链条追不完整）：A -> B -> C -> D 这种
    // 三跳以上的链，改成在这里就顺着链一次追到底（带环检测，防止 Goto
    // 成环时死循环——环本身是合法的控制流，比如空 `loop {}`，不应该被
    // 这里破坏掉），不依赖外层循环帮忙兜底。
    fn chain_gotos(body: &mut MirBody) -> bool {
        let block_count = body.blocks.len();
        // 关键修复：专用 preheader 是一个空的 `Goto(循环头)` 块，这里不能把
        // 它当普通跳板穿过去——否则 `p(If) -> ph -> header` 被改回
        // `p(If) -> header`，preheader 就没了，insert_preheaders 下一轮又补
        // 回来，两边互相撤销、永不收敛。现算一份循环信息（本函数前面的
        // pass 可能已经改过边，不用上一轮的）。
        let protected = Self::preheader_set(body);
        let mut visited = HashSet::new();
        let mut changed = false;
        for i in 0..block_count {
            let original = if let MirTerminator::Goto(t) = &body.blocks[i].terminator {
                *t
            } else {
                continue;
            };

            let mut current = original;
            visited.clear();
            visited.insert(i);
            loop {
                if current >= block_count || !visited.insert(current) {
                    break;
                }
                let next_block = &body.blocks[current];
                if !next_block.stmts.is_empty() || protected.contains(&current) {
                    break;
                }
                if let MirTerminator::Goto(next) = &next_block.terminator {
                    current = *next;
                } else {
                    break;
                }
            }
            // 关键修复：只有真的追到了比原来更远的目标才写回去——原来
            // 这里无条件赋值，即使 current == original（比如 i 本来就
            // 指向一个有语句的块，一步都没追出去）也要重写一次
            // terminator，纯属浪费；现在顺手把"changed 该不该置 true"和
            // "值该不该写回去"用同一个条件表达。
            if current != original {
                body.blocks[i].terminator = MirTerminator::Goto(current);
                changed = true;
            }
        }
        changed
    }

    fn merge_blocks(body: &mut MirBody) -> bool {
        if body.blocks.len() <= 1 {
            return false;
        }

        // 关键重构：前驱表统一来自 MirBody::predecessor_table，不再自己
        // 扫一遍终止器。顺带修掉了一处不一致：这里原来的表没有去重，
        // 同一个块的两条边指向同一个目标（If 两边同一目标）会被算成两个
        // 前驱，"单前驱"判断因此漏掉；而 fold_marker_blocks 等其它 pass
        // 用的是去重的口径。现在四个 pass 用同一个口径。
        let preds = body.predecessor_table();

        // 关键修复（move-out-of-index，编译不过）：`MirTerminator` 不是
        // Copy，从一个 `Vec` 的下标位置按值把它匹配出来会报 E0507，改成
        // 借用 `&body.blocks[pred_id].terminator`，只需要读，不需要拿走。
        let mut merged = Vec::new();
        for (id, _block) in body.blocks.iter().enumerate() {
            if preds[id].len() == 1 && id != 0 {
                let pred_id = preds[id][0];
                if let MirTerminator::Goto(target) = &body.blocks[pred_id].terminator {
                    if *target == id {
                        merged.push((pred_id, id));
                    }
                }
            }
        }

        if merged.is_empty() {
            return false;
        }

        // 关键说明：succ（这里的 id）单前驱不代表它没有 Phi——它仍然可能
        // 带一条只有一个 incoming 的"单路 Phi"。但 fold_single_pred_phis
        // 排在这个函数之前跑过一轮，单前驱块里的 Phi 已经全部退化成了
        // Use，这里不需要再单独处理 Phi 的问题。
        //
        // 关键修复（合并方向反了，会丢代码）：正确方向是 pred 保留自己
        // 的 block id 不变（因为可能有别的前驱依赖"跳到 pred_id"这件事
        // 本身），把 succ 的语句追加到 pred 后面，pred 的终结指令换成
        // succ 原来的终结指令；succ 这边才是可以清空、标记 Unreachable
        // 的那个——这是安全的，因为 succ 已经确认只有一个前驱（pred），
        // pred 合并完之后不再指向 succ_id，succ_id 自然变得彻底不可达，
        // 交给 remove_unreachable_blocks 收尾。
        //
        // 关键修复（同一批合并列表里的链式依赖）：如果这一批 merged 里
        // 同时出现 (A, B) 和 (B, C)，用 touched 集合防住：一对
        // (pred, succ) 只有在两边这一轮都还没被动过时才真正执行；被
        // 跳过的那一对留给下一轮——succ 变成不可达之后
        // remove_unreachable_blocks 会让块数发生变化，外层循环会再跑
        // 一轮，那时候 merge_blocks 重新从头统计前驱，链条剩下的部分
        // 自然会在下一轮被正确合并。
        let mut touched = std::collections::HashSet::new();
        for (pred, succ) in merged {
            if touched.contains(&pred) || touched.contains(&succ) {
                continue;
            }
            touched.insert(pred);
            touched.insert(succ);

            let succ_stmts: Vec<MirStmt> = body.blocks[succ].stmts.drain(..).collect();
            body.blocks[pred].stmts.extend(succ_stmts);
            body.blocks[pred].terminator = body.blocks[succ].terminator.clone();
            body.blocks[succ].terminator = MirTerminator::Unreachable;
        }
        // 关键修复：判断"这一轮是否真的做了合并"要看 touched，不能看
        // merged 是否为空——merged 里的每一对都可能因为链式依赖被
        // touched 挡住而跳过，出现"merged 非空、但一次真正的合并都没
        // 执行"的情况，那种情况下如果仍然报告 changed，外层循环会
        // 误以为还在推进，白白多跑一轮。
        !touched.is_empty()
    }

    // 现算一份"专用 preheader"集合。只有会穿过/合并空块的 pass 需要它。
    fn preheader_set(body: &MirBody) -> HashSet<usize> {
        let cfg = CfgInfo::build(body);
        LoopInfo::build(body, &cfg).preheader_blocks()
    }

    // ---- Tail merge：把等价的块合并成一个 ----
    //
    // merge_blocks 只处理"pred 是单前驱的纯 Goto"这一种情况。这里处理
    // 的是另一类重复：好几个毫不相干的块（比如 match 展开出来的不同
    // 分支）末尾恰好是逐条相同的语句 + 等价的终结指令，彼此之间没有
    // "谁是谁的唯一前驱"这种关系，merge_blocks 覆盖不到。
    //
    // 关键设计（等价类而不是逐字相等）：原来判断终结指令"相同"用的是
    // `==`，只能识别 A: Goto(10)、B: Goto(10) 这种目标字面量一样的情况；
    // A: Goto(10)、B: Goto(20) 即使 10 和 20 在这一轮已经被判定为等价块
    // （比如它们自己又都是 Goto(30)），A 和 B 也不会被认成一对，代码
    // 体积没能被压到最简。改成先把每个目标块顺着 `redirect` 链爬到它
    // 现在的"代表"（canonical_target），再比较代表是否相同——这样只要
    // 目标最终等价，中间差着几层重定向也认得出来。
    //
    // `redirect` 是在同一次调用里边发现等价边填的，所以 canonical_target
    // 在比较某一对 (i, j) 时，只能看到*在它们之前*已经确定下来的等价
    // 关系；比 i、j 靠后的块这一轮还没轮到，即使后面会被判等价，这里
    // 也看不见。这会让某些等价机会要多跑一轮外层循环才被发现，不会
    // 得出错误结论——只是保守，不是错误。
    //
    // 两个限制不变：入口块（id 0）永远不参与合并；含 Phi 的块不参与
    // 合并（Phi 的 incoming 按前驱 id 记录，重定向会让代表块的 Phi 对
    // 不上新增的前驱，真要支持需要把两边 Phi 项做并集，属于更大的
    // 改动，这里如实排除）。
    fn tail_merge(body: &mut MirBody) -> bool {
        let block_count = body.blocks.len();
        if block_count <= 1 {
            return false;
        }

        // redirect[j] = Some(i) 表示"块 j 是块 i 的重复，所有跳到 j 的
        // 边都应该改跳到 i"。i 本身保证不会再被重定向（outer 循环用
        // `redirect[i].is_some()` 挡掉），只会指向更小的下标，所以
        // detect::redirect_canonical 最多爬一层就到底，不存在多级链。
        //
        // 关键修复：专用 preheader 不参与合并。一个空的 preheader 跟某个
        // 空的 latch（同样是 `Goto(循环头)`）语句相同、终止器相同，会被判成
        // 等价而合并——合并之后循环外的入口边和循环内的回边落在同一个块上，
        // preheader 不复存在，insert_preheaders 再补一个，下一轮又被合并，
        // 永不收敛。
        let protected = Self::preheader_set(body);
        let mut redirect: Vec<Option<usize>> = vec![None; block_count];
        for i in 1..block_count {
            if redirect[i].is_some() || protected.contains(&i) || block_contains_phi(&body.blocks[i]) {
                continue;
            }
            for j in (i + 1)..block_count {
                if redirect[j].is_some() || protected.contains(&j) || block_contains_phi(&body.blocks[j]) {
                    continue;
                }
                if body.blocks[i].stmts == body.blocks[j].stmts
                    && terminators_equivalent(&body.blocks[i].terminator, &body.blocks[j].terminator, &redirect)
                {
                    redirect[j] = Some(i);
                }
            }
        }

        if redirect.iter().all(Option::is_none) {
            return false;
        }

        // 关键重构：把每个目标改成它的代表块。遍历"有哪些目标"由
        // MirTerminator::for_each_target_mut 统一负责；原来这里对 Goto/
        // If/Switch 各写一遍，而且直接 `redirect[*target]` 裸下标——目标
        // 越界的畸形 MIR 会 panic。现在用 `.get()`，越界的目标原样保留
        // （重定向只该改"确实被合并掉的块"，不该替畸形输入做决定）。
        for block in &mut body.blocks {
            block.terminator.for_each_target_mut(|t| {
                if let Some(canon) = redirect.get(*t).copied().flatten() {
                    *t = canon;
                }
            });
        }
        // 重复块此刻已经没有任何边指向它们了，块数暂时没变——交给紧接着
        // 的 remove_unreachable_blocks 去掉。
        true
    }

    // ---- 公共后缀提取 ----
    // 关键新增：tail_merge 解决的是"两个块从头到尾逐字相同"，这里解决
    // 的是更常见的一种重复——两个块末尾若干条语句相同，但各自前面的
    // 部分不同，tail_merge 的整体相等判断根本不会命中它们。做法：把
    // 公共的尾巴搬到它们共同的目标块开头，两个源块各自截掉这一段，
    // 代码只保留一份。
    //
    // 为了不引入错误的合并，限定了几个条件：
    //   1. 两个块 A、B 的终结指令都得是 Goto(T)，指向同一个 T——公共
    //      后缀提取的前提是"它们本来就要去同一个地方"。
    //   2. T 的直接前驱只能是 A、B 这两个（去重后恰好 2 个）。T 还有
    //      别的前驱的话，把公共后缀挪到 T 开头会让那些路径也多执行
    //      一遍这几条语句，语义变了。
    //   3. T 不含 Phi。T 的 Phi incoming 是按 A、B 的哪个前驱记录的；
    //      公共后缀里如果定义了 Phi 引用的 SSA 值，把这些语句挪到 T
    //      之后，定义就跑到了它们本该支配的那条边的另一侧，位置不对。
    //      为简单起见，T 有 Phi 就跳过，不做半吊子处理。
    //   4. 公共后缀只含 ExprStmt / Drop / SetMetadata / EffectCheck，
    //      一旦遇到 Assign 立刻停止比较——Assign 会引入新的 SSA 定义，
    //      挪动定义位置可能影响它在 A、B 内部剩余语句里已经生效的
    //      作用域，不像上面几种纯副作用语句那样"挪到哪里执行效果都
    //      一样"。
    //   5. 提取出来的公共后缀非空，一条都没有就没什么好做的。
    //
    // 一轮只处理每个 T 至多一对 (A, B)：处理完 T 的语句已经变了，其它
    // 以 T 为目标的块对是否仍然满足条件需要重新判断，交给外层循环的
    // 下一轮。
    fn extract_common_suffix(body: &mut MirBody) -> bool {
        let n = body.blocks.len();
        if n <= 2 {
            return false;
        }

        // 前驱表统一来自 MirBody::predecessor_table：去重、过滤越界、按块
        // id 升序（所以下面取到的 (a, b) 已经是 a < b，不需要再排序）。
        let preds = body.predecessor_table();

        let mut changed = false;
        for t in 0..n {
            if preds[t].len() != 2 {
                continue;
            }
            let (a, b) = (preds[t][0], preds[t][1]);

            let a_goes_to_t = match &body.blocks[a].terminator {
                MirTerminator::Goto(x) => *x == t,
                MirTerminator::If { .. }
                | MirTerminator::Switch { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable
                | MirTerminator::Placeholder => false,
            };
            let b_goes_to_t = match &body.blocks[b].terminator {
                MirTerminator::Goto(x) => *x == t,
                MirTerminator::If { .. }
                | MirTerminator::Switch { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable
                | MirTerminator::Placeholder => false,
            };
            if !a_goes_to_t || !b_goes_to_t {
                continue;
            }
            if block_contains_phi(&body.blocks[t]) {
                continue;
            }

            let k = {
                let a_stmts = &body.blocks[a].stmts;
                let b_stmts = &body.blocks[b].stmts;
                let max_k = a_stmts.len().min(b_stmts.len());
                let mut k = 0;
                while k < max_k {
                    let ai = a_stmts.len() - 1 - k;
                    let bi = b_stmts.len() - 1 - k;
                    if a_stmts[ai] != b_stmts[bi] {
                        break;
                    }
                    // 遇到 Assign 立即停止：只允许挪动无定义、只有副作用
                    // 的语句，见上面条件 4 的说明。
                    let is_assign = match &a_stmts[ai] {
                        MirStmt::Assign { .. } => true,
                        MirStmt::ExprStmt(_)
                        | MirStmt::Drop { .. }
                        | MirStmt::SetMetadata { .. }
                        | MirStmt::EffectCheck { .. } => false,
                    };
                    if is_assign {
                        break;
                    }
                    k += 1;
                }
                k
            };
            if k == 0 {
                continue;
            }

            let a_len = body.blocks[a].stmts.len();
            let b_len = body.blocks[b].stmts.len();
            let suffix: Vec<MirStmt> = body.blocks[a].stmts[a_len - k..].to_vec();
            body.blocks[a].stmts.truncate(a_len - k);
            body.blocks[b].stmts.truncate(b_len - k);
            body.blocks[t].stmts.splice(0..0, suffix);
            changed = true;
        }
        changed
    }

    // ---- 删除不可达块 ----
    // 关键重命名：simplify.rs 里也有一个 Simplify::remove_dead_blocks，
    // 但那边用的是 SCCP 算出的 exec_blocks（要求沿途的条件分支在常量
    // 结论下真的走得到，是"条件可达性"）；这里用的是 CfgInfo 提供的
    // 结构可达性（只要有边连着就算，不管条件的值）。两者是不同的分析、
    // 不同的删除依据，虽然常常删掉同一批块，名字撞在一起容易让人以为
    // 是同一个函数，改名 remove_unreachable_blocks 把"这是结构可达性"
    // 这件事写进名字里。
    //
    // 关键重构：这里只负责回答"保留哪些块"（结构可达的块）。"怎么删"——
    // 原地压缩、重编号终止器目标和 Phi 前驱、目标被删时终止器怎么降级
    // （含 Switch 缺 default 时标 Unreachable 而不是挑一个显式分支）——
    // 原来跟 simplify.rs 的 remove_dead_blocks 各有一份逐字相同的实现，
    // 现在统一走 MirBody::retain_and_renumber_blocks，兜底策略只有那一处。
    fn remove_unreachable_blocks(body: &mut MirBody, cfg: &CfgInfo) -> bool {
        let block_count = body.blocks.len();
        if block_count == 0 {
            return false;
        }
        // cfg 必须是针对当前 body 刚建的：它按块编号索引，块数对不上说明
        // 拿着过期的 CfgInfo。compute_reachable 自己也会 debug 断言，这里
        // 在 release 下也不冒险，直接放弃这一轮。
        if cfg.block_count() != block_count || cfg.revision() != body.revision {
            debug_assert!(false, "remove_unreachable_blocks: stale CfgInfo");
            return false;
        }
        let keep = cfg.compute_reachable(block_count);
        body.retain_and_renumber_blocks(&keep)
    }

    // ---- 补 preheader ----
    // 关键新增：给还没有"专用 preheader"的自然循环补一个——一个只含
    // `Goto(循环头)` 的新块，所有来自循环外的入口边都改指它。这是将来
    // 循环不变量外提（LICM）的前提：外提的计算要放在"只有进入循环这条
    // 路径才会执行"的地方，循环头的循环外前驱如果还有别的出口（比如一个
    // 以 If 结尾、一边进循环一边绕开的块），把计算放进它里面会让绕开循环
    // 的路径也执行一遍。
    //
    // 只处理下面这些情形，其余原样跳过（保守，不是遗漏）：
    //   1. 循环已经有专用 preheader（LoopInfo::preheader 是 Some）：不用补。
    //   2. 循环头没有任何来自循环外的、从入口可达的前驱：循环头是入口块，
    //      或者整个循环不可达。入口块当循环头时补 preheader 要新建一个入口
    //      块并重编号所有块，这一轮不做。
    //   3. 循环头带 Phi 时：只有"恰好一个循环外前驱，且每条 Phi 恰好有
    //      一条来自它的 incoming"才处理——这时把那条 incoming 的前驱换成
    //      新块即可，值的定义本来就支配那个前驱，也就支配新块。循环外前驱
    //      有多个、Phi 因此有多条循环外 incoming 时，要在新块里为每条 Phi
    //      合成一个汇合 Phi（还得分配新的 SSA 版本号，MirBody 里没有这份
    //      分配器），这是下一轮的目标，这里跳过。循环头没有 Phi 时不受这条
    //      限制：多个循环外前驱直接都改指新块就行。
    //
    // 只改"循环外的、可达的"前驱的终止器，循环内的回边不动——回边必须
    // 继续指向循环头。
    fn insert_preheaders(body: &mut MirBody) -> bool {
        if body.blocks.len() < 2 {
            return false;
        }
        let cfg = CfgInfo::build(body);
        let loops = LoopInfo::build(body, &cfg);
        if loops.loops.is_empty() {
            return false;
        }

        let mut changed = false;
        for l in &loops.loops {
            if l.preheader.is_some() || l.entering_preds.is_empty() {
                continue;
            }
            let header = l.header;

            let mut phi_ok = true;
            for stmt in &body.blocks[header].stmts {
                if let MirStmt::Assign { value: MirRvalue::Phi { values }, .. } = stmt {
                    let mut external_incoming = 0usize;
                    for (pred, _) in values {
                        if l.entering_preds.contains(pred) {
                            external_incoming += 1;
                        }
                    }
                    if external_incoming != 1 || l.entering_preds.len() != 1 {
                        phi_ok = false;
                        break;
                    }
                }
            }
            if !phi_ok {
                continue;
            }

            let ph = body.blocks.len();
            body.blocks.push(MirBlock {
                id: ph,
                stmts: Vec::new(),
                terminator: MirTerminator::Goto(header),
            });
            for &p in &l.entering_preds {
                body.blocks[p].terminator.for_each_target_mut(|t| {
                    if *t == header {
                        *t = ph;
                    }
                });
            }
            for stmt in body.blocks[header].stmts.iter_mut() {
                if let MirStmt::Assign { value: MirRvalue::Phi { values }, .. } = stmt {
                    for (pred, _) in values.iter_mut() {
                        if l.entering_preds.contains(pred) {
                            *pred = ph;
                        }
                    }
                }
            }
            changed = true;
        }
        // 补了 preheader：块的集合变了（多了新块），按块编号索引的分析快照作废。
        // 这是这个文件里唯一一个会增删块的 pass——其它几个只改终止器（改边）、
        // 只挪语句，块的集合和编号都没变，不能 bump：bump 会让依赖快照的
        // pass 误判快照失效、跳过本该做的工作（见 MirBody::revision）。
        // remove_unreachable_blocks 走 retain_and_renumber_blocks，它自己会 bump。
        if changed {
            body.bump_revision();
        }
        changed
    }
}
