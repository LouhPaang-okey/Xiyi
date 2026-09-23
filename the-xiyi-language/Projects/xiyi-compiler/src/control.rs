// control.rs
use crate::mir::*;
// 关键修复：HashMap/VecDeque 这两个原来的 import 在全文件都没有被用到
// 过——preds 用的是 Vec<Vec<usize>>，remove_dead_blocks 的可达性遍历
// 用 Vec 当栈（push/pop），没有任何地方真正需要哈希表或双端队列。死
// import 只会产生编译警告，不影响功能，但顺手清理掉。

pub struct Control;

impl Control {
    pub fn simplify(program: &mut MirProgram) {
        for f in &mut program.fns {
            Self::simplify_function(f);
        }
    }

    // 关键修复（健壮性）：原来这个循环没有任何轮数上限——只要每一轮
    // block 数量还在变就会一直跑下去。四个子过程都已经确认是收敛的
    // （每轮要么块数变少、要么不变），正常情况下不会真的跑到这个
    // 上限；加一个保守的上限纯粹是防御性的，避免以后哪个子过程一旦
    // 引入新 bug（比如误判成"块数变了"但实际在两个等价状态之间来回
    // 摆动），这里不会变成一个真正意义上的死循环——参照 simplify.rs
    // 那边 MAX_OPT_ITERATIONS 的同一个考虑。
    const MAX_ITERATIONS: usize = 64;

    fn simplify_function(f: &mut MirFn) {
        for _ in 0..Self::MAX_ITERATIONS {
            let before = f.body.blocks.len();
            Self::simplify_switch(&mut f.body);
            Self::chain_gotos(&mut f.body);
            Self::merge_blocks(&mut f.body);
            Self::tail_merge(&mut f.body);
            Self::remove_dead_blocks(&mut f.body);
            if f.body.blocks.len() == before {
                break;
            }
        }
    }

    fn simplify_switch(body: &mut MirBody) {
        for block in &mut body.blocks {
            // 关键修复：`MirTerminator::Switch` 这一轮多了一个
            // `discr_ty` 字段，这里原来解构的三个字段（targets/default/
            // discr）没有把它列进来，也没有用 `..` 兜底剩余字段——
            // struct 模式必须覆盖所有字段，缺一个就编译不过（E0027）。
            if let MirTerminator::Switch { targets, default, discr: _, discr_ty: _ } = &block.terminator {
                if targets.is_empty() {
                    block.terminator = MirTerminator::Goto(*default);
                }
                else if targets.iter().all(|(_, t)| *t == targets[0].1) && targets[0].1 == *default {
                    block.terminator = MirTerminator::Goto(targets[0].1);
                }
            }
        }
    }

    // 关键修复（会悄悄删代码的 bug）：原来这里只看 target_block 的
    // terminator 是不是 Goto，完全没检查 target_block 自己有没有真实
    // 语句——如果 A -> B -> C，B 里其实有真实语句（哪怕只是一句
    // print），原来的写法照样会把 A 直接改成 Goto(C)，绕过 B、永远不
    // 再执行 B 的那些语句。这不是优化，是丢代码：只有当 B 是一个
    // "纯跳转跳板"（自己没有任何语句，terminator 又是 Goto）时，跳过
    // B 直接指向它的目标才是安全的——这跟 simplify.rs 的
    // collapse_empty_blocks 判断"能不能折叠"用的是同一条件
    // （`block.stmts.is_empty()`），这里补上。
    //
    // 关键修复（原来只追一跳，链条追不完整）：A -> B -> C -> D 这种
    // 三跳以上的链，原来单次调用只把 A 改成 Goto(C)（用的是处理 A 那
    // 一刻 B 的旧 terminator），到 Goto(D) 还得等外层 simplify_function
    // 的循环因为别的原因（比如某个块被删导致块数变化）再跑一轮才会
    // 继续往下追——如果那一轮恰好没有块被删（block 数没变），循环会
    // 提前退出，链条就停在没追完的中间状态。改成在这里就顺着链一次
    // 追到底（带环检测，防止 Goto 成环时死循环——环本身是合法的控制
    // 流，比如空 `loop {}`，不应该被这里破坏掉），不依赖外层循环帮忙
    // 兜底。
    fn chain_gotos(body: &mut MirBody) {
        let block_count = body.blocks.len();
        // 关键修复（性能）：原来每一轮外层循环都在循环体里
        // `std::collections::HashSet::new()` 一个新的 set，只是用来在
        // 追链的过程中做"环检测"，各轮之间互不依赖彼此的内容——挪到
        // 外层只分配一次，每轮开始追链前 `clear()` 复用同一块内存，
        // 省掉 block 数量多时的重复分配开销。
        let mut visited = std::collections::HashSet::new();
        for i in 0..block_count {
            let mut current = if let MirTerminator::Goto(t) = &body.blocks[i].terminator {
                *t
            } else {
                continue;
            };

            visited.clear();
            visited.insert(i);
            loop {
                if current >= block_count || !visited.insert(current) {
                    // 越界（防御性，理论上不会发生）或者追到了已经在
                    // 链上出现过的块（成环），停在当前位置，不再往下追。
                    break;
                }
                let next_block = &body.blocks[current];
                if !next_block.stmts.is_empty() {
                    break;
                }
                if let MirTerminator::Goto(next) = &next_block.terminator {
                    current = *next;
                } else {
                    break;
                }
            }
            body.blocks[i].terminator = MirTerminator::Goto(current);
        }
    }

    fn merge_blocks(body: &mut MirBody) {
        if body.blocks.len() <= 1 {
            return;
        }

        let block_count = body.blocks.len();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); block_count];
        for (src_id, block) in body.blocks.iter().enumerate() {
            match &block.terminator {
                MirTerminator::Goto(target) if *target < block_count => {
                    preds[*target].push(src_id);
                }
                MirTerminator::If { then_block, else_block, .. } => {
                    if *then_block < block_count { preds[*then_block].push(src_id); }
                    if *else_block < block_count { preds[*else_block].push(src_id); }
                }
                MirTerminator::Switch { targets, default, .. } => {
                    if *default < block_count { preds[*default].push(src_id); }
                    for (_, target) in targets {
                        if *target < block_count { preds[*target].push(src_id); }
                    }
                }
                _ => {}
            }
        }

        // 关键修复（move-out-of-index，编译不过）：`body.blocks[pred_id]
        // .terminator` 没有 `&`，是按值匹配——`MirTerminator` 不是
        // Copy（内部经过 MirOperand 最终会带上 String/Vec），从一个
        // `Vec` 的下标位置按值把它匹配出来，等于要把它从 Vec 的元素里
        // 移走，这对 `Vec` 的元素是不允许的（E0507）。改成借用
        // `&body.blocks[pred_id].terminator`，只需要读，不需要拿走。
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
            return;
        }

        // 关键修复（合并方向反了，会丢代码）：原来的实现是把 pred 的
        // 语句搬进 succ、然后把 pred 清空、terminator 改成
        // Unreachable。这个方向只有在"没有别的块会跳到 pred_id"这个
        // 前提下才是对的，但上面判断的是 succ（这里的 id）只有一个
        // 前驱（就是 pred），完全没检查 pred 自己有没有别的前驱。如果
        // 真有别的块跳到 pred_id（这很常见，比如 pred 是好几条分支
        // 汇合之后的一个公共块），把 pred 清空 + 标记 Unreachable 会
        // 让所有跳到 pred_id 的路径全部变成"执行空语句然后
        // Unreachable"——pred 原来的代码彻底丢了，不是优化，是正确性
        // bug。
        //
        // 正确方向反过来：pred 保留自己的 block id 不变（因为可能有
        // 别的前驱依赖"跳到 pred_id"这件事本身），把 succ 的语句追加
        // 到 pred 后面，pred 的终结指令换成 succ 原来的终结指令；succ
        // 这边才是可以清空、标记 Unreachable 的那个——这是安全的，
        // 因为 succ 已经确认只有一个前驱（pred），pred 合并完之后不再
        // 指向 succ_id，succ_id 自然变得彻底不可达，交给
        // remove_dead_blocks 收尾。
        //
        // 关键修复（同一批合并列表里的链式依赖）：如果这一批 merged
        // 里同时出现 (A, B) 和 (B, C)（B 既是 A 的 succ、又是 C 的
        // pred），这两条是在"谁都还没被合并"的状态下一起统计出来的；
        // 如果不加防护，先处理 (A, B) 会把 B 清空、terminator 改成
        // Unreachable，这时候再处理 (B, C) 就是在一个已经空了、谁都
        // 跳不到的块上瞎搬——C 原本的内容会被合并进这个死块，永远
        // 不可达，等于也丢了。用一个 touched 集合防住：一对
        // (pred, succ) 只有在两边这一轮都还没被动过时才真正执行；被
        // 跳过的那一对不会丢——succ 变成不可达之后 remove_dead_blocks
        // 会让块数发生变化，外层 simplify_function 的循环会再跑一轮，
        // 那时候 merge_blocks 重新从头统计前驱，链条剩下的部分自然会
        // 在下一轮被正确合并。
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
    }

    // ---- Tail merge：把语句+终结指令完全相同的多个块合并成一个 ----
    //
    // merge_blocks 只处理"pred 是单前驱的纯 Goto"这一种情况——A 只有
    // B 一个前驱、且 B 就是通过 Goto 跳过来的，才把 A 并进 B。但还有
    // 一种很常见、merge_blocks 完全覆盖不到的重复：好几个毫不相干的
    // 块（比如 match 展开出来的不同分支）末尾恰好是逐条相同的语句 +
    // 相同的终结指令（典型的比如都以同一句 `return` 或跳到同一个
    // `end` 块收尾），这些块彼此之间没有"谁是谁的唯一前驱"这种关系，
    // merge_blocks 的判断条件根本不会命中它们，代码体积会随着分支数
    // 线性重复。
    //
    // 做法：两两比较块的 `stmts` + `terminator`（现在都已经是
    // PartialEq 了，见 mir.rs 的改动），完全相同就只留一个当代表，把
    // 所有跳到重复块的边都改指向代表块，重复块自然变得不可达，交给
    // remove_dead_blocks 收尾——不需要新建块，只是重新接线，改动面小。
    //
    // 两个限制：
    //   1. 入口块（id 0）永远不参与合并，既不当代表也不当被合并的一方
    //      ——它的身份是"函数从这里开始"，不是随便一个可替换的普通块。
    //   2. 含 Phi 语句的块不参与合并。Phi 的 `values` 里按前驱块 id
    //      记录了"从哪个前驱来、取哪个值"；一旦把某个块的所有前驱
    //      都改指向代表块，代表块自己的 Phi 并不知道这些"新"前驱，会
    //      丢失对应分支的取值信息。这不是"再多做一步 id 重映射"就能
    //      简单补上的事——真要支持，需要把两个块的 Phi 项做并集，属于
    //      更大的改动，这里先如实排除，不做半吊子的合并。
    //
    // 只做"整块完全相同"，不做"公共后缀"式的更激进合并（两个块前半段
    // 不同、只有尾巴几条语句相同，把尾巴拆成一个新块）——那种做法要
    // 新增块、要在 If/Switch 里插入新的跳转目标，改动面大得多，留到
    // 以后。
    fn stmt_is_phi(stmt: &MirStmt) -> bool {
        match stmt {
            MirStmt::Assign { value: MirRvalue::Phi { .. }, .. } => true,
            _ => false,
        }
    }

    fn contains_phi(block: &MirBlock) -> bool {
        block.stmts.iter().any(Self::stmt_is_phi)
    }

    fn tail_merge(body: &mut MirBody) {
        let block_count = body.blocks.len();
        if block_count <= 1 {
            return;
        }

        // redirect[j] = Some(i) 表示"块 j 是块 i 的重复，所有跳到 j 的
        // 边都应该改跳到 i"。i 本身保证不会再被重定向（下面 outer 循环
        // 用 `redirect[i].is_some()` 挡掉），所以这里最多一层，不会有
        // 需要多次追链才能找到最终代表的情况。
        let mut redirect: Vec<Option<usize>> = vec![None; block_count];
        for i in 1..block_count {
            if redirect[i].is_some() || Self::contains_phi(&body.blocks[i]) {
                continue;
            }
            for j in (i + 1)..block_count {
                if redirect[j].is_some() || Self::contains_phi(&body.blocks[j]) {
                    continue;
                }
                if body.blocks[i].stmts == body.blocks[j].stmts
                    && body.blocks[i].terminator == body.blocks[j].terminator
                {
                    redirect[j] = Some(i);
                }
            }
        }

        if redirect.iter().all(Option::is_none) {
            return;
        }

        for block in &mut body.blocks {
            match &mut block.terminator {
                MirTerminator::Goto(target) => {
                    if let Some(canon) = redirect[*target] {
                        *target = canon;
                    }
                }
                MirTerminator::If { then_block, else_block, .. } => {
                    if let Some(canon) = redirect[*then_block] {
                        *then_block = canon;
                    }
                    if let Some(canon) = redirect[*else_block] {
                        *else_block = canon;
                    }
                }
                MirTerminator::Switch { targets, default, .. } => {
                    if let Some(canon) = redirect[*default] {
                        *default = canon;
                    }
                    for (_, target) in targets.iter_mut() {
                        if let Some(canon) = redirect[*target] {
                            *target = canon;
                        }
                    }
                }
                _ => {}
            }
        }
        // 重复块此刻已经没有任何边指向它们了（自己的出边没动，但反正
        // 不可达），块数暂时没变——交给紧接着的 remove_dead_blocks 去
        // 掉，块数变化会让外层 simplify_function 的循环继续跑下一轮。
    }

    // 删除不可达块
    fn remove_dead_blocks(body: &mut MirBody) {
        let block_count = body.blocks.len();
        if block_count == 0 {
            return;
        }

        // 1. 标记可达块
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

        // 3. 更新所有保留块的跳转目标，同时处理 Phi 中的前驱 ID
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
                    match (new_then, new_else) {
                        (Some(nt), Some(ne)) => {
                            *then_block = nt;
                            *else_block = ne;
                        }
                        (Some(nt), None) => block.terminator = MirTerminator::Goto(nt),
                        (None, Some(ne)) => block.terminator = MirTerminator::Goto(ne),
                        (None, None) => block.terminator = MirTerminator::Unreachable,
                    }
                }
                // discr/discr_ty 这里用不上（新的 default 只有
                // Unreachable 或复用旧值两种结局，不需要重新构造整个
                // Switch 去带上它们），用 `..` 忽略——比之前"显式列出
                // discr/discr_ty 但从没用过"更干净。（历史备注：这里曾
                // 经因为 struct 模式没列全字段漏了 discr_ty 报过 E0027，
                // `..` 天然把这类"新增字段忘记加进模式"的问题也一并
                // 挡掉了，不用每次 MirTerminator::Switch 加字段都跟着
                // 改这里。）
                MirTerminator::Switch { targets, default, .. } => {
                    let new_default = id_map[*default];
                    let mut new_targets = Vec::new();
                    for (val, target) in targets.iter_mut() {
                        if let Some(new_target) = id_map[*target] {
                            *target = new_target;
                            // 关键修复：`*val` 要求 Literal: Copy——这
                            // 一轮 ast.rs 改动之前 targets 存的是
                            // `i64`（Copy），现在存的是 `Literal`（只
                            // derive 了 Clone，不是 Copy，因为它内部
                            // 可能带 String 之类的变体，虽然这里实际
                            // 只会出现 Int*/UInt*/Bool/Char，但类型系统
                            // 只看声明，不看"实际会出现哪些变体"），
                            // `*val` 编译不过，改成 `.clone()`。
                            new_targets.push((val.clone(), *target));
                        }
                    }
                    if new_targets.is_empty() {
                        if let Some(nd) = new_default {
                            block.terminator = MirTerminator::Goto(nd);
                        } else {
                            block.terminator = MirTerminator::Unreachable;
                        }
                    } else {
                        *targets = new_targets;
                        if let Some(nd) = new_default {
                            *default = nd;
                        } else {
                            // 关键修复（错误的兜底值，静默走错分支）：
                            // 原来这里挑 `targets[0].1`（幸存的第一个
                            // 显式分支）当新的 default——但 default 对应
                            // 的是"判别式不匹配任何显式 target 时走哪"，
                            // 跟"随便一个显式分支"是完全不同的语义。如果
                            // 这里真的被执行到，等于让运行时一部分本该走
                            // default 的取值，静默地跑进了 targets[0] 那
                            // 条分支的代码——不报错，但执行的是错的分支，
                            // 比崩溃更难查。
                            //
                            // 而且这个分支按当前的可达性算法根本不可能被
                            // 真正走到：这个 Switch 语句所在的块能出现在
                            // `new_blocks` 里，本身就意味着它在上面第 1
                            // 步的可达性标记阶段被访问过——而那一步对
                            // *每一个*被访问到的 Switch 块，都会无条件把
                            // 它的 `default` 目标标成可达并压栈（见上面
                            // "1. 标记可达块"那段），所以只要这个块本身
                            // 可达，它的 default 目标必然也可达，
                            // `id_map[*default]` 必然是 `Some`，不可能走
                            // 到这个 else 分支。
                            //
                            // 既然按道理走不到，就没必要去猜一个"看起来
                            // 合理"的替代值——猜错了后果（悄悄执行错误
                            // 分支）比直接报错严重得多，用一个假的 block
                            // id 去凑一个 Switch 也不行（mir.rs 里
                            // Placeholder 那条注释已经讲过类似的道理：不
                            // 能拿哨兵值硬凑一个假的 id，这里是同一个
                            // 道理）。既然这个分支理论上不可能被真的走
                            // 到，直接标 Unreachable——万一以后可达性
                            // 算法被改出 bug、这个假设不再成立，运行时
                            // 应该是"炸出来"而不是"静默算错"。
                            block.terminator = MirTerminator::Unreachable;
                        }
                    }
                }
                _ => {}
            }

            let mut new_stmts = Vec::with_capacity(block.stmts.len());
            for stmt in &block.stmts {
                match stmt {
                    MirStmt::Assign { dest, value: MirRvalue::Phi { values } } => {
                        let mut new_values = Vec::new();
                        for (old_pred, op) in values {
                            if let Some(new_pred) = id_map[*old_pred] {
                                new_values.push((new_pred, op.clone()));
                            }
                        }
                        if new_values.is_empty() {
                            continue;
                        } else if new_values.len() == 1 {
                            // 关键修复：`new_values[0]` 按值索引再解构，
                            // 要求 `MirOperand: Copy`——同上，不成立
                            // （内部经 Literal 可能带 String）。改成借
                            // 用取出来，下面已经在用 `.clone()`。
                            let (_, op) = &new_values[0];
                            new_stmts.push(MirStmt::Assign {
                                dest: dest.clone(),
                                value: MirRvalue::Use(op.clone()),
                            });
                        } else {
                            new_stmts.push(MirStmt::Assign {
                                dest: dest.clone(),
                                value: MirRvalue::Phi { values: new_values },
                            });
                        }
                    }
                    _ => new_stmts.push(stmt.clone()),
                }
            }
            block.stmts = new_stmts;
        }

        body.blocks = new_blocks;
    }
}