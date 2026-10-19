// simplify.rs
use crate::mir::*;
use crate::calc::Calc;
use crate::analyze::{
    compute_purity_table, find_loop_invariants, pure_expr_key, ssa_uses_in_place, AliasInfo,
    AnalysisResults,
    AvailableExprs, CfgInfo, ExprKey, GvnResult, Lattice, LiteralKey, Liveness, Loop, LoopInfo,
    OperandKey, SccpResult,
};
use crate::detect::{
    binop_may_trap, block_contains_phi, call_purity, cast_may_trap, copy_local_ids, is_bool_false,
    is_bool_true, is_one, is_zero, multiply_defined_ssa, must_not_alias,
    operand_is_copy_ssa_or_const, place_has_deref, place_has_enum_payload, place_may_trap,
    rvalue_has_side_effect, ssa_def_blocks, stmt_is_phi, Purity,
};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};

const MAX_OPT_ITERATIONS: usize = 20; // 收敛性控制

pub struct Simplify;

impl Simplify {
    pub fn run(mut program: MirProgram) -> MirProgram {
        // 一次性算好全程序的函数纯度表——它不随单个函数的 body 变化。
        // 前提：program.fns 里每个函数的 name 已经是 mangled 名（monomorphic
        // 已经跑过），Call.func 用的也是 mangled 名，两边才对得上；对不上的
        // 调用在 call_purity 里按 Impure 处理，只会少优化，不会误删。
        //
        // 关键说明：这张表是从 MIR 函数体算出来的不动点（见 analyze.rs 的
        // compute_purity_table），不是直接拿每个函数的 EffectSet 映射——
        // EffectSet 不是传递闭包，直接用会把"间接调用了 print 的函数"误判成
        // Pure。表在优化开始前算一次：优化只会让函数更纯或保持不变（删掉的
        // 语句不会引入新的副作用），所以用优化前的结果是保守的。
        let purity: Arc<HashMap<String, Purity>> =
            Arc::new(compute_purity_table(&program.fns, &program.structs, &program.enums));
        // fns 可变借用、structs/enums 只读借用，是 program 的不同字段，不冲突。
        for f in &mut program.fns {
            Self::optimize_function(f, &purity, &program.structs, &program.enums);
        }
        program
    }

    // ===== 优化主循环 =====
    // 关键修复：eliminate_dead_stores 和 fold_terminator 这两个函数在
    // 旧版本里已经写好了，但从来没有被这里调用过——旧版 fold_block 的
    // 签名/实现本身也编译不过（下面会细说），大概率是上一轮改 SSA 改
    // 到一半的中间状态。这一轮全部接回主循环，并且把原来"边扫边攒
    // const_map"的块内常量传播换成了先分析、再重写两个阶段：
    //   0) 分析（Self::analyze）：建 CFG/支配树，跑一次 SCCP（常量
    //      事实、可执行块/边），再跑一次 GVN（哪些 SSA 值和某个支配它
    //      的值相等）——这一步只读 IR，不改 IR。
    //   1) 重写（fold_block）：用分析结果替换常量、做复制传播、代数
    //      恒等式化简、终止器折叠。
    //   2) Phi 简化：按 SCCP 的可执行边裁剪 incoming
    //   3) 死块消除：按 SCCP 的可执行块删除
    //   4) 死赋值消除（需要整个函数范围的"谁被读到了"信息，比逐块
    //      重写更全局，所以单独一趟；放在删块之后，这样已删除的死块里
    //      的读取不会再让定义显得"还被用着"）
    //   5) 空块消除
    // 每轮迭代都重新做一次分析——上一轮重写/消除可能让更多分支的条件
    // 变成常量、让更多块变得不可达，分析结果需要跟着刷新，不能只在第
    // 一轮算一次。
    //
    // 关键约束：analysis 里所有东西都按块编号索引，是这一轮开头的快照。
    // fold_block / simplify_phi / eliminate_dead_stores 不改块编号，快照
    // 一直有效；remove_dead_blocks 是第一个会重编号的 pass，用完快照即
    // 作废，它之后的 collapse_empty_blocks 是纯结构性的，不读 analysis。
    // 所以这几步的顺序不能随便换：会重编号的 pass 必须排在所有依赖快照
    // 的 pass 之后。依赖快照的入口另有 analysis.matches(body) 兜底。
    //
    // 关键修复（收敛判断失效）：原来判断"这一轮有没有变化"是比较前后的
    // 块数和语句总数。块数确实只在有真实合并/删除时才变，可以信；但
    // 语句总数不行——fold_block 每次都会用 fold_stmt 的结果重新拼一个
    // 等长的新 Vec 整体替换 block.stmts，不管折叠有没有真的发生，"总
    // 语句数"永远不变。SSA 强化和 fold_terminator 的重写同理：改的是
    // 语句/终结指令的*内容*，不是数量。这意味着只要没有 pass 删块或删
    // 整条语句，收敛判断永远命中，实际效果是"最多跑一轮就退出"——
    // 注释里"每轮迭代都重新做一次分析"的意图从来没有真正生效过。
    //
    // 现在改成每个子过程各自汇报"这一轮是不是真的改写了 IR"（bool），
    // 只要有一个报告 true 就继续下一轮，全部 false 才停。fold_block/
    // fold_terminator/simplify_phi 内容可能原地改写但长度不变，用
    // "新内容和旧内容是否相等"来判断（MirStmt/MirTerminator 都已经是
    // PartialEq，参见 control.rs 的 tail_merge）；eliminate_dead_stores
    // 只做删除不做原地改写，比较语句数前后即可，不需要整体做内容对比。
    fn optimize_function(
        f: &mut MirFn,
        purity: &Arc<HashMap<String, Purity>>,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) {
        // 关键修复（这一轮加上真正的 Drop 语义之后才会暴露出来的
        // 问题，见下面 fold_stmt 里的详细说明）：预先算出"哪些局部
        // 变量是 Copy 类型"，传给 fold_block/fold_stmt——复制传播
        // （copy_map）只有在来源是 Copy 类型时才能安全折叠。
        let copy_types = copy_local_ids(&f.body);

        for _ in 0..MAX_OPT_ITERATIONS {
            let mut changed = false;

            // ===== 分析阶段：只读 IR，产出这一轮重写要用的事实 =====
            let analysis = Self::analyze(&f.body, purity, structs, enums);
            // SSA -> 定义块，供 fold_block 判"可用表达式的 holder 是否支配块
            // 入口"。每轮重建：上一轮的 fold / licm / collapse 可能增删过
            // SSA 定义，复用旧表会漏掉新定义。同名多定义的 SSA（不是规范
            // SSA）的"首次定义块"不可信，直接从表里剔掉——pick_holder 查不到
            // 定义块就拒绝这个候选，宁可少优化。
            let mut def_block = ssa_def_blocks(&f.body);
            for s in multiply_defined_ssa(&f.body) {
                def_block.remove(&s);
            }

            // ===== 重写阶段：copy_map 仍然按 block 现开现用（见
            // fold_block 上面的说明），常量部分统一查 analysis =====
            // 跨块公共子表达式在合并点插 Phi 时要给新 Phi 分配版本号；分配器按这一轮
            // 开头的 body 建，整轮 fold 共用一个，保证各块插入的 Phi 互不撞名。
            let mut version_alloc = VersionAllocator::new(&f.body);

            for (block_id, block) in f.body.blocks.iter_mut().enumerate() {
                let mut copy_map: HashMap<SsaLocal, MirOperand> = HashMap::new();
                changed |= Self::fold_block(
                    block_id,
                    block,
                    &analysis,
                    &mut copy_map,
                    &copy_types,
                    &def_block,
                    &mut version_alloc,
                );
            }

            // 关键修复：上一版这里把 analysis.cfg 传给 simplify_phi 和
            // remove_dead_blocks 做可达性——一是 SCCP 已经算出了精确得多的
            // 可执行性却不用，二是 remove_dead_blocks 排在会重编号的
            // collapse_empty_blocks 之后，拿建立于重编号之前的快照按新的
            // 块数去查，轻则越界 panic，重则错位后悄悄删掉活块。现在两步
            // 都改用 SCCP 的 exec_edges/exec_blocks，并且把删块提到所有
            // 依赖快照的 pass 之后、重编号的 collapse 之前。
            changed |= Self::simplify_phi(&mut f.body, &analysis);
            changed |= Self::remove_dead_blocks(&mut f.body, &analysis);
            // 循环不变量外提：放在删块之后（块编号稳定）、死赋值消除之前
            // （外提后原位置留下的东西、以及外提出来但没人用的计算交给它清）。
            // licm 自己现建 CFG/循环信息，不读 analysis。
            changed |= Self::licm(&mut f.body, purity, structs, enums);
            // 关键说明：这里只传 purity，不传整个 analysis——前面的
            // remove_dead_blocks 已经重编号了块，analysis 里按块编号索引的
            // 东西（cfg/sccp/loops/alias）此刻都是过期的，把整个快照递给
            // 后面的 pass，等于留一个"将来谁顺手读了 analysis.cfg 就读到错位
            // 数据"的陷阱。纯度表不按块索引，不受影响。
            changed |= Self::eliminate_dead_stores(&mut f.body, purity, structs, enums);
            changed |= Self::collapse_empty_blocks(&mut f.body);

            if !changed {
                break;
            }
        }
    }

    // ==================== LICM（循环不变量外提）====================
    //
    // 把循环里"每次迭代算出来都一样"的纯计算搬到循环的专用 preheader 里，
    // 只算一次。前置条件：循环有专用 preheader（LoopInfo::preheader 是 Some，
    // control.rs 的 insert_preheaders 负责补）；没有的循环原样跳过。
    //
    // 一轮扫描只能搬走"当前就能确认不变"的那些：外提之后，原来依赖它们的
    // 语句才会在下一轮看起来是循环外定义的；嵌套循环里内层外提到内层
    // preheader（它在外层循环里面）之后，外层的下一轮才有机会把它继续往外
    // 搬。所以反复做，每轮都重建 CFG/循环信息，直到没有东西可搬（带轮数上限）。
    //
    // 位置在 remove_dead_blocks 之后：块编号已经稳定；这里自己建
    // CfgInfo/LoopInfo，不用 optimize_function 开头那份快照（它是删块之前的）。
    // 外提只移动语句、不改块结构，所以每一轮内的 CFG/循环信息一直有效。
    fn licm(
        body: &mut MirBody,
        purity: &HashMap<String, Purity>,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> bool {
        const MAX_ROUNDS: usize = 8;
        let mut any_changed = false;
        for _ in 0..MAX_ROUNDS {
            let cfg = CfgInfo::build(body);
            let loops = LoopInfo::build(body, &cfg);
            if loops.loops.is_empty() {
                break;
            }
            // 内存读外提要问"循环里有没有通过 Deref/Impure 调用的写"，每轮按当前
            // body 现建别名信息（上一轮外提移动过语句）。
            let alias = AliasInfo::build(body, purity, structs, enums);
            if !Self::licm_round(body, &cfg, &loops, &alias, structs, enums) {
                break;
            }
            any_changed = true;
        }
        any_changed
    }

    fn licm_round(
        body: &mut MirBody,
        cfg: &CfgInfo,
        loops: &LoopInfo,
        alias: &AliasInfo,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> bool {
        let invariants = find_loop_invariants(body, cfg, loops);
        if invariants.is_empty() {
            return false;
        }

        // find_loop_invariants 的结果按循环分组、同一个循环的条目连续排列。
        let mut by_loop: Vec<(usize, Vec<SsaLocal>)> = Vec::new();
        for (loop_idx, ssa) in invariants {
            let mut appended = false;
            if let Some((idx, list)) = by_loop.last_mut() {
                if *idx == loop_idx {
                    list.push(ssa);
                    appended = true;
                }
            }
            if !appended {
                let mut list: Vec<SsaLocal> = Vec::with_capacity(1);
                list.push(ssa);
                by_loop.push((loop_idx, list));
            }
        }

        let n = body.blocks.len();
        let mut rpo_idx = vec![usize::MAX; n];
        for (i, &b) in cfg.rpo().iter().enumerate() {
            rpo_idx[b] = i;
        }
        let copy_base = copy_local_ids(body);

        // 循环按循环头的 RPO 位置排序，外层在前；倒着走就是内层先处理。内层先
        // 搬到内层 preheader（外层循环的一个块），这一轮轮到外层时，那些语句
        // 已经在外层循环的块里，可以接着往外搬。上面算出的不变量清单是这一轮
        // 开头的，外层清单里本来就包含内层的那些语句，搬走之后位置变了但
        // 仍然是不变量，licm_hoist 按 dest 在当前循环块里找，不依赖旧位置。
        let mut changed = false;
        for (loop_idx, ssas) in by_loop.iter().rev() {
            let l = match loops.loops.get(*loop_idx) {
                Some(l) => l,
                None => continue,
            };
            let preheader = match l.preheader {
                Some(p) => p,
                None => continue,
            };
            if Self::licm_hoist(
                body, &rpo_idx, l, preheader, ssas, &copy_base, alias, structs, enums,
            ) {
                changed = true;
            }
        }
        changed
    }

    // 把 invariant_ssas 里"真的可以搬"的那些语句搬进 preheader。
    //
    // 数据流上不变不等于能搬，每一条还要同时满足（见 licm_hoistable）：
    //   1. 结果是 Copy 类型，操作数是常量或 Copy 类型的 SSA。外提之后这个值
    //      只算一次、在循环里被多次读到；非 Copy 的值每被读一次（Move）就消耗
    //      一次所有权，原来每轮迭代各算一个新的，现在第二轮就是 use-after-move。
    //   2. 运算本身可以投机执行（licm_speculation_safe）：循环一次都没跑时，
    //      外提后的计算也会被执行，它不能陷入。
    //   3. 内存读（`Use(Copy/Move(place))`）另走 licm_memory_read_hoistable：
    //      除了"不陷入"，还要证明循环里没有任何写入会改到这块内存。
    // 并且搬走之后依赖要完整：一条语句如果读了某个循环内定义、却没有被搬走
    // 的值，它也不能搬——搬走后它在 preheader 里读一个还没定义的值。这一条
    // 靠下面的"传递闭包"保证（先按条件 1、2 筛，再反复剔除依赖了没搬的
    // 循环内定义的语句，直到稳定）。
    fn licm_hoist(
        body: &mut MirBody,
        rpo_idx: &[usize],
        l: &Loop,
        preheader: usize,
        invariant_ssas: &[SsaLocal],
        copy_base: &HashSet<usize>,
        alias: &AliasInfo,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> bool {
        let candidates: HashSet<SsaLocal> = invariant_ssas.iter().copied().collect();

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

        // 第一遍：按条件 1、2 筛，并记下每条候选读了哪些循环内定义的值。
        let mut hoistable: HashSet<SsaLocal> = HashSet::new();
        let mut inside_operands: Vec<(SsaLocal, Vec<SsaLocal>)> = Vec::new();
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
                if !candidates.contains(&dest) {
                    continue;
                }
                if !Self::licm_hoistable(body, l, dest, rv, copy_base, alias, structs, enums) {
                    continue;
                }
                let mut deps: Vec<SsaLocal> = Vec::new();
                if let Some(ops) = Self::licm_operands(rv) {
                    for op in ops {
                        match op {
                            MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => {
                                if defined_inside.contains(s) {
                                    deps.push(*s);
                                }
                            }
                            MirOperand::Copy(_)
                            | MirOperand::Move(_)
                            | MirOperand::Constant(_)
                            | MirOperand::Sym(_)
                            | MirOperand::Static(_) => {}
                        }
                    }
                }
                // 内存读 `Use(place)` 的依赖单独收集：place 里读到的每个 SSA
                // （根变量、下标操作数）如果是循环内定义的，也必须先被搬走。
                // licm_operands 对 Use 返回 None，它是"纯计算操作数"的视图，
                // 不在这里混用。
                if let MirRvalue::Use(MirOperand::Copy(place))
                | MirRvalue::Use(MirOperand::Move(place)) = rv
                {
                    let mut uses: Vec<SsaLocal> = Vec::new();
                    ssa_uses_in_place(place, &mut uses);
                    for u in uses {
                        if defined_inside.contains(&u) {
                            deps.push(u);
                        }
                    }
                }
                hoistable.insert(dest);
                inside_operands.push((dest, deps));
            }
        }

        // 第二遍：传递闭包——依赖了"循环内定义、又没有搬走"的值的语句也不能搬。
        loop {
            let mut removed = false;
            for (dest, deps) in &inside_operands {
                if !hoistable.contains(dest) {
                    continue;
                }
                let mut blocked = false;
                for d in deps {
                    if !hoistable.contains(d) {
                        blocked = true;
                        break;
                    }
                }
                if blocked {
                    hoistable.remove(dest);
                    removed = true;
                }
            }
            if !removed {
                break;
            }
        }
        if hoistable.is_empty() {
            return false;
        }

        // 按块的 RPO 顺序、块内语句顺序收集：支配者在前、依赖的定义在使用之前，
        // 在 preheader 里的先后顺序天然满足依赖。
        let mut to_hoist: Vec<MirStmt> = Vec::new();
        for &b in &loop_blocks {
            for stmt in &body.blocks[b].stmts {
                if let MirStmt::Assign { dest: MirPlace::Ssa(s), .. } = stmt {
                    if hoistable.contains(s) {
                        to_hoist.push(stmt.clone());
                    }
                }
            }
        }
        if to_hoist.is_empty() {
            return false;
        }

        for &b in &loop_blocks {
            let mut kept: Vec<MirStmt> = Vec::with_capacity(body.blocks[b].stmts.len());
            for stmt in body.blocks[b].stmts.drain(..) {
                let moved = match &stmt {
                    MirStmt::Assign { dest: MirPlace::Ssa(s), .. } => hoistable.contains(s),
                    MirStmt::Assign { .. }
                    | MirStmt::ExprStmt(_)
                    | MirStmt::Drop { .. }
                    | MirStmt::SetMetadata { .. }
                    | MirStmt::EffectCheck { .. } => false,
                };
                if !moved {
                    kept.push(stmt);
                }
            }
            body.blocks[b].stmts = kept;
        }

        // preheader 是专用的（唯一后继就是循环头），语句追加在末尾、终止器之前。
        body.blocks[preheader].stmts.extend(to_hoist);
        true
    }

    // 外提要求的运算形态：BinaryOp / UnaryOp / Cast，没有 Call 之类的副作用。
    // 返回它读取的操作数；别的右值不外提。
    fn licm_operands(rv: &MirRvalue) -> Option<Vec<&MirOperand>> {
        let mut ops: Vec<&MirOperand> = Vec::with_capacity(2);
        match rv {
            MirRvalue::BinaryOp(_, l, r) => {
                ops.push(l);
                ops.push(r);
            }
            MirRvalue::UnaryOp(_, o) => ops.push(o),
            MirRvalue::Cast(o, _) => ops.push(o),
            MirRvalue::Use(_)
            | MirRvalue::Call { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. } => return None,
        }
        Some(ops)
    }

    // 外提要求的右值形态（两类共用第 1 条"结果是 Copy 类型"，其余条件分开讲）：
    //   - 纯计算 BinaryOp / UnaryOp / Cast：操作数是常量或 Copy 类型的 SSA，
    //     运算本身可以投机执行；
    //   - 内存读 Use(Copy/Move(place))：见 licm_memory_read_hoistable。
    fn licm_hoistable(
        body: &MirBody,
        l: &Loop,
        dest: SsaLocal,
        rv: &MirRvalue,
        copy_base: &HashSet<usize>,
        alias: &AliasInfo,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> bool {
        if !copy_base.contains(&dest.base_id) {
            return false;
        }

        if let MirRvalue::Use(op) = rv {
            return match op {
                MirOperand::Copy(place) | MirOperand::Move(place) => {
                    Self::licm_memory_read_hoistable(body, l, place, alias, structs, enums)
                }
                MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => false,
            };
        }

        let ops = match Self::licm_operands(rv) {
            Some(o) => o,
            None => return false,
        };
        for op in ops {
            if !operand_is_copy_ssa_or_const(op, copy_base) {
                return false;
            }
        }
        let dest_ty = match body.locals.iter().find(|l| l.id == dest.base_id) {
            Some(l) => &l.ty,
            None => return false,
        };
        Self::licm_speculation_safe(rv, dest_ty)
    }

    // 内存读能不能外提到循环前面。全部条件都要满足：
    //   1. place 不是单纯的 SSA（那是拷贝，交给复制传播）；
    //   2. 不经过 Deref——不知道指向哪块内存，无从证明没有别名写；
    //   3. 不含 EnumPayload——读它只有在判别式检查之后才合法，挪出去就是投机读
    //      一个可能不对的变体；
    //   4. 读本身不会陷入（place_may_trap：越界下标）——外提后循环一次都没跑
    //      的路径上也会执行这次读；
    //   5. 循环里没有"通过 Deref 的写 / Impure 调用 / MethodCall"（AliasInfo
    //      的 loop 级判据）：它们可能经由别名写到这块内存；
    //   6. 循环里每个直接写入的目标 place 都"一定不重叠"于这次读（must_not_alias）。
    //      写目标是整个 SSA 的（定义一个新 SSA 值）不影响读：读的根 SSA 是循环外
    //      定义的，那个值不会变。
    fn licm_memory_read_hoistable(
        body: &MirBody,
        l: &Loop,
        place: &MirPlace,
        alias: &AliasInfo,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> bool {
        match place {
            MirPlace::Ssa(_) => return false,
            MirPlace::Field { .. }
            | MirPlace::Index { .. }
            | MirPlace::Deref(_)
            | MirPlace::EnumPayload { .. } => {}
        }
        if place_has_deref(place) || place_has_enum_payload(place) {
            return false;
        }
        if place_may_trap(place, body, structs, enums) {
            return false;
        }
        for &b in &l.blocks {
            if alias.block_writes_deref(b) {
                return false;
            }
        }
        for &b in &l.blocks {
            let block = match body.blocks.get(b) {
                Some(bl) => bl,
                None => return false,
            };
            for stmt in &block.stmts {
                match stmt {
                    MirStmt::Assign { dest: MirPlace::Ssa(_), .. } => {}
                    MirStmt::Assign { dest: dest_place, .. } => {
                        // 带 Deref 的写已经被上面的 block_writes_deref 整体拒绝过，
                        // 走到这里的都是直接写；仍然防御性地处理。
                        if place_has_deref(dest_place) {
                            return false;
                        }
                        if !must_not_alias(dest_place, place) {
                            return false;
                        }
                    }
                    // Drop 会使整个 place 的存储失效，保守地当作写。
                    MirStmt::Drop { place: dropped } => {
                        if !must_not_alias(dropped, place) {
                            return false;
                        }
                    }
                    MirStmt::ExprStmt(_)
                    | MirStmt::SetMetadata { .. }
                    | MirStmt::EffectCheck { .. } => {}
                }
            }
        }
        true
    }

    // 运算能不能被投机执行：外提之后，循环一次都没跑的路径上它也会被执行，
    // 所以它不能陷入。"哪些运算可能陷入"的唯一定义在 detect.rs（binop_may_trap /
    // cast_may_trap；place 里的 Index 越界由 place_may_trap 管），跟函数纯度
    // 分析共用同一份，两边对"这个运算会不会 panic"的看法不会不一致。
    // 整数溢出是否陷入的语言语义假设也只写在那里。
    //
    // 关键前提（别删）：本函数只对 licm_operands 放行的三种右值——BinaryOp /
    // UnaryOp / Cast——有意义。这三种的操作数在进入这里之前已被 licm_hoistable
    // 里的 operand_is_copy_ssa_or_const 限定为"常量或 Copy 类型的 SSA"，不可能
    // 带 Index / Deref 之类会越界的 place，只剩运算本身会不会陷入这一个问题。
    // 内存读 Use(place) 不走这里，由 licm_memory_read_hoistable 单独判（含
    // place_may_trap）。如果将来放宽 licm_operands（带 place 操作数的运算），
    // 必须同时改用 detect::rvalue_may_trap 判断整条右值，否则越界下标会被悄悄
    // 投机到循环外面。其余形态在这里一律返回 false。
    //
    // 浮点除法/取模（IEEE，不陷入）单独放行：binop_may_trap 拿不到操作数类型，
    // 对浮点除法保守地回答"可能"，这里有 dest 的类型，可以更精确。
    fn licm_speculation_safe(rv: &MirRvalue, dest_ty: &Type) -> bool {
        match rv {
            MirRvalue::BinaryOp(op, _, r) => match op {
                BinaryOp::Div | BinaryOp::Mod => {
                    Self::type_is_float(dest_ty) || !binop_may_trap(*op, r)
                }
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
                | BinaryOp::Or => !binop_may_trap(*op, r),
            },
            MirRvalue::UnaryOp(_, _) => true,
            MirRvalue::Cast(o, ty) => !cast_may_trap(o, ty),
            MirRvalue::Use(_)
            | MirRvalue::Call { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. } => false,
        }
    }

    fn type_is_float(ty: &Type) -> bool {
        match ty {
            Type::F16 | Type::F32 | Type::F64 => true,
            Type::Privacy(inner, _) => Self::type_is_float(inner),
            // Type 有二十多个变体，除了浮点类型和它的隐私包装以外都不是
            // 浮点，通配在这里正是对它们的正确回答。
            _ => false,
        }
    }

    // ==================== 分析编排 ====================
    //
    // 关键设计：AnalysisResults 在 analyze.rs 里只是纯数据容器，"按什么
    // 顺序算各项分析"放在这里——sccp、gvn 都是 Simplify 的关联函数，
    // 编排逻辑放在 analyze.rs 会造成 analyze <-> simplify 循环引用。
    // 顺序体现依赖：cfg 不依赖别人，先建；sccp 只看 body；gvn 要用 cfg
    // 的 rpo/dominates，也要用 sccp 的结论（常量操作数按常量进 key，
    // 不可执行的块直接跳过）。
    fn analyze(
        body: &MirBody,
        purity: &Arc<HashMap<String, Purity>>,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> AnalysisResults {
        // 关键说明：下游函数（gvn / AliasInfo::build）要的是 `&HashMap`，而这里
        // 手里是 `&Arc<HashMap>`。这个转换是合法的——`&Arc<T>` 在函数参数、
        // 带类型标注的 let 这类强制转换点上会自动解引用成 `&T`（Arc<T> 实现了
        // Deref<Target = T>）。这里显式写一次带类型的 let，让转换发生在一个
        // 一眼能看到的地方，而不是散落在三个调用点各自隐式发生；下面的
        // Arc::clone(purity) 才是真正需要 Arc 本身的地方。
        let table: &HashMap<String, Purity> = purity;
        let cfg = CfgInfo::build(body);
        let sccp = Self::sccp(body);
        let gvn = Self::gvn(body, &cfg, &sccp, table);
        let loops = LoopInfo::build(body, &cfg);
        let alias = AliasInfo::build(body, table, structs, enums);
        let available = AvailableExprs::build(body, &cfg);
        let results = AnalysisResults {
            cfg,
            sccp,
            gvn,
            loops,
            alias,
            available,
            purity: Arc::clone(purity),
        };
        debug_assert!(
            results.is_consistent(),
            "analysis results are internally inconsistent (CFG/SCCP/GVN/Loop/Alias/Available)"
        );
        results
    }

    // ==================== SCCP ====================
    //
    // 关键设计：稀疏条件常量传播（Sparse Conditional Constant
    // Propagation）。跟原来 fold_stmt 里那套"每进一个 block 就重开一
    // 张空表"的块内常量传播比，SCCP 把整个函数当成一张网求不动点，
    // 天然能吃跨块常量，包括循环回边——不是靠先建支配树、按拓扑顺序
    // 处理来保证正确性，而是靠 exec_blocks/exec_edges：一个块要先被
    // 证明可执行，它里面的赋值才参与求值；一条边要先被证明可执行，
    // Phi 才会 join 它带来的值。循环体第一次还没跑到的时候，回边根本
    // 不在 exec_edges 里，不会把只在某次迭代里成立的值错误地带到前面
    // 去（这正是旧版块内常量传播为了避免算错、干脆放弃跨块传播时留下
    // 的那个限制，见 fold_block 上面的说明）。
    //
    // 用的是反复全量扫描到不动点，不是经典的 worklist 算法——函数规模
    // 通常几十个块，不动点一般几轮以内收敛，扫描开销可以忽略，换来
    // 实现和验证都简单。SccpResult 的字段是这个分析对外的稳定接口，
    // 以后真要因为规模变大而升级成 worklist，只需要换掉这个函数的
    // 内部实现，fold_operand 这些消费者不用跟着改。
    fn sccp(body: &MirBody) -> SccpResult {
        let n = body.blocks.len();
        if n == 0 {
            return SccpResult {
                values: HashMap::new(),
                exec_blocks: Vec::new(),
                exec_edges: HashSet::new(),
            };
        }

        let mut values: HashMap<SsaLocal, Lattice> = HashMap::new();
        let mut exec_blocks = vec![false; n];
        let mut exec_edges: HashSet<(usize, usize)> = HashSet::new();
        exec_blocks[0] = true;

        let mut changed = true;
        while changed {
            changed = false;
            for block_id in 0..n {
                if !exec_blocks[block_id] { continue; }
                let block = &body.blocks[block_id];

                for stmt in &block.stmts {
                    if let MirStmt::Assign { dest: MirPlace::Ssa(ssa), value } = stmt {
                        let new_val = Self::sccp_eval_rvalue(value, &values, &exec_edges, block_id);
                        let slot = values.entry(*ssa).or_insert(Lattice::Undefined);
                        if slot.join(&new_val) {
                            changed = true;
                        }
                    }
                }

                for succ in Self::sccp_successors(&block.terminator, &values) {
                    // 跟 CfgInfo::build 一致：目标块 id 越界的边（畸形 MIR）
                    // 直接忽略，不让分析阶段因为输入不合法而越界 panic。
                    if succ >= n {
                        continue;
                    }
                    if exec_edges.insert((block_id, succ)) {
                        changed = true;
                    }
                    if !exec_blocks[succ] {
                        exec_blocks[succ] = true;
                        changed = true;
                    }
                }
            }
        }

        SccpResult { values, exec_blocks, exec_edges }
    }

    // 关键说明：跟下面重写阶段用的 fold_rvalue 不是同一个函数，虽然
    // 分支结构很像——这里只回答"这个右值按当前格状态求出来是什么"，
    // 不产出新的 MirRvalue，也不做代数恒等式那一套化简（那是重写阶段
    // 的事，分析阶段只管值，不动 IR 形状）。BinaryOp/UnaryOp 复用的是
    // Calc 里跟 fold_rvalue 完全相同的求值函数，保证"分析阶段判定为
    // 常量"和"重写阶段真的能折叠成那个常量"两边不会对不上。
    fn sccp_eval_rvalue(
        rv: &MirRvalue,
        values: &HashMap<SsaLocal, Lattice>,
        exec_edges: &HashSet<(usize, usize)>,
        block_id: usize,
    ) -> Lattice {
        match rv {
            MirRvalue::Use(op) => Self::sccp_eval_operand(op, values),
            // 关键修复：操作数是 Undefined（"还没算到"）时，结果必须保持
            // Undefined，不能直接掉成 Overdefined。格只能往上走、不能退
            // 回来，一旦在"某个操作数恰好还没被本轮扫描算到"的时刻被判成
            // Overdefined，之后操作数变成常量了这里也回不去了——而扫描是
            // 按块 id 顺序、不是按支配顺序，使用者的块 id 比定义者小完全
            // 可能发生，尤其是 GVN 之前块顺序一旦被别的 pass 打乱之后。
            // 只有操作数里真的有 Overdefined，结果才是 Overdefined。
            MirRvalue::BinaryOp(op, l, r) => {
                let lhs = Self::sccp_eval_operand(l, values);
                let rhs = Self::sccp_eval_operand(r, values);
                match (lhs, rhs) {
                    (Lattice::Overdefined, _) | (_, Lattice::Overdefined) => Lattice::Overdefined,
                    (Lattice::Undefined, _) | (_, Lattice::Undefined) => Lattice::Undefined,
                    (Lattice::Constant(a), Lattice::Constant(b)) => {
                        match Calc::eval_binary_op(*op, a, b) {
                            Some(c) => Lattice::Constant(c),
                            None => Lattice::Overdefined,
                        }
                    }
                }
            }
            MirRvalue::UnaryOp(op, o) => match Self::sccp_eval_operand(o, values) {
                Lattice::Constant(lit) => match Calc::eval_unary_op(*op, lit) {
                    Some(c) => Lattice::Constant(c),
                    None => Lattice::Overdefined,
                },
                Lattice::Undefined => Lattice::Undefined,
                Lattice::Overdefined => Lattice::Overdefined,
            },
            MirRvalue::Phi { values: phi_vals } => {
                // 关键：只 join 来自可执行前驱边的 incoming。循环体第
                // 一次还没执行过时，指向循环体的那条 incoming 边不在
                // exec_edges 里，不参与 join；循环体真的执行过一次
                // 之后，这条边才会被 sccp_successors 标记为可执行，它
                // 带来的值才会被吸收进来。这是 SCCP 相对朴素常量传播
                // 真正的核心，也是它不需要先建支配树、仅凭反复迭代就
                // 能在循环上算对的原因。
                let mut acc = Lattice::Undefined;
                for (pred, op) in phi_vals {
                    if exec_edges.contains(&(*pred, block_id)) {
                        acc.join(&Self::sccp_eval_operand(op, values));
                    }
                }
                acc
            }
            // 其余右值（Call/MethodCall/Cast/StructInit/...）没有编译期
            // 求值语义，一律 Overdefined——跟 fold_rvalue 里"这些变体
            // 只折叠内部操作数、整体不产出常量"是同一个判断，只是这里
            // 落在格上表现为顶元素。
            _ => Lattice::Overdefined,
        }
    }

    fn sccp_eval_operand(op: &MirOperand, values: &HashMap<SsaLocal, Lattice>) -> Lattice {
        match op {
            MirOperand::Constant(l) => Lattice::Constant(l.clone()),
            MirOperand::Copy(MirPlace::Ssa(ssa)) | MirOperand::Move(MirPlace::Ssa(ssa)) => {
                values.get(ssa).cloned().unwrap_or(Lattice::Undefined)
            }
            // 非 Ssa 的 place（Field/Index/Deref/EnumPayload）可能被
            // 另一条路径的写入改变，Sym/Static 又根本不指向局部变量——
            // 这两种都不在 SCCP 的追踪范围内，一律当作已知不是常量，
            // 不要盲目乐观地留 Undefined 拖慢收敛。跟 fold_operand 一样
            // 显式列出操作数变体，新增变体时编译器会逼我们表态。
            MirOperand::Copy(_)
            | MirOperand::Move(_)
            | MirOperand::Sym(_)
            | MirOperand::Static(_) => Lattice::Overdefined,
        }
    }

    // 关键说明：跟重写阶段的 fold_terminator 分工类似但不是一回事——
    // 这里只回答"按当前格状态，这个终止器实际可能走到哪些后继"，用来
    // 驱动 exec_blocks/exec_edges 的传播；真正把 If/Switch 改写成
    // Goto，是重写阶段 fold_terminator 该做的事，两者故意分开。
    fn sccp_successors(term: &MirTerminator, values: &HashMap<SsaLocal, Lattice>) -> Vec<usize> {
        // 先问"按当前格状态，能不能唯一确定走哪一个后继"；确定不了就
        // 回退到该终止器的全部目标。
        //
        // 关键设计："全部目标"这条回退路径走的是 MirTerminator::targets，
        // 它背后的 for_each_target 是必须显式列出所有变体的那个唯一位置，
        // 新增变体时那里会编译失败、逼着回答"它指向哪些块"。下面这个 match
        // 也照样显式列出每个变体，不用通配：新变体如果也需要"按常量条件
        // 确定唯一后继"，编译器会在这里提醒，而不是让它悄悄落进"拿不准、
        // 全部都走"的保守分支。
        let decided = match term {
            MirTerminator::If { cond, then_block, else_block } => {
                match Self::sccp_eval_operand(cond, values) {
                    Lattice::Constant(Literal::Bool(true)) => Some(*then_block),
                    Lattice::Constant(Literal::Bool(false)) => Some(*else_block),
                    // Overdefined，或者理论上不该出现的"非布尔常量"（sema
                    // 已经保证 cond 是 bool，这里只是防御性兜底）：两边
                    // 都可能走，都标记成可执行。
                    _ => None,
                }
            }
            MirTerminator::Switch { discr, targets, default, .. } => {
                match Self::sccp_eval_operand(discr, values) {
                    Lattice::Constant(lit) => Some(
                        targets
                            .iter()
                            .find(|(v, _)| *v == lit)
                            .map(|(_, t)| *t)
                            .unwrap_or(*default),
                    ),
                    _ => None,
                }
            }
            MirTerminator::Goto(_)
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => None,
        };
        match decided {
            Some(t) => {
                let mut only = Vec::with_capacity(1);
                only.push(t);
                only
            }
            None => term.targets(),
        }
    }

    // ==================== GVN ====================
    //
    // 关键设计：全局值编号（Global Value Numbering），基于支配树的
    // 冗余消除。按 reverse post-order 线性扫一遍所有块、维护一张全局的
    // "表达式 key -> 已经算出这个表达式的 SSA 值（及其所在块）"候选表：
    // 遇到一条新计算，如果候选表里已经有同 key 的、且*定义所在块支配当前
    // 块*的候选，就记一条 dest -> rep，让后面读 dest 的地方改读 rep。
    //
    // 为什么一次线性扫描 + dominates 判断就够（不用按支配树递归、进块
    // 复制作用域表、出块还原）：RPO 保证支配者先于被支配者出现，所以
    // 扫到某个块时，所有可能支配它的块都已经把自己的候选登记过了；
    // "候选真的支配当前块"这件事交给 cfg.dominates 逐个验证，不依赖
    // 表里只放了"有效"的候选。同一块内靠语句顺序：候选只会在它自己那条
    // 语句处理完之后才进表，所以同块命中的候选一定在当前语句之前。
    //
    // 只处理满足下面全部条件的语句，其余一概跳过（保守，不是遗漏）：
    //   1. `dest` 是 SSA 且局部变量类型是 Copy。非 Copy 类型读第二次会
    //      消耗所有权，"把这一处读取换成读另一个 SSA"不成立；Copy 类型
    //      读多少次都行，是唯一能保证安全的场景。fold_operand 里把读取
    //      换成 Copy（而不是 Move）也正是依赖这一条，两处必须同步。
    //   2. 右值是 BinaryOp/UnaryOp/Cast/Discriminant/Use，或者 Purity::Pure
    //      的 Call，且操作数全是常量或 Copy 类型的 SSA（逐变体的检查见
    //      gvn_expr_key）。MethodCall 要靠 receiver 类型才知道调的是谁，查
    //      不了纯度；ReadOnly 的调用两次之间内存可能被改过，不保证同一个值；
    //      StructInit/EnumVariantConstruction/ArrayLiteral/Ref 每次都构造
    //      新的聚合值/新的借用，合并等于把两个独立的值变成共享同一个，
    //      不合法；Phi 的值取决于来路，不是"同样输入同样输出"。
    //   3. `dest` 不是 SCCP 已经判定的常量——那种值 fold_operand 会直接
    //      换成字面量，比换成另一个 SSA 更好，也不需要占候选表的位置。
    //   4. `dest` 和操作数里出现的 SSA 在整个函数里只被定义过一次。GVN
    //      的正确性建立在"一个 SSA 名字对应唯一一处定义"上，万一上游
    //      产出了重复定义（SCCP 遇到这种情况会自己 join 成 Overdefined，
    //      GVN 没有这层保护），宁可跳过也不要把两个不同的值当成同一个。
    //
    // 只处理 SCCP 判定为可执行的块：不可执行的块马上会被 remove_dead_blocks
    // 整块删掉，在里面登记候选/记录改写都是白费；而且它们不可能支配任何
    // 可执行块（支配者不可执行则被支配者也不可执行），不会污染候选。
    fn gvn(
        body: &MirBody,
        cfg: &CfgInfo,
        sccp: &SccpResult,
        purity: &HashMap<String, Purity>,
    ) -> GvnResult {
        let mut result = GvnResult::empty();
        if body.blocks.is_empty() {
            return result;
        }

        let copy_base = copy_local_ids(body);
        let multi_def = multiply_defined_ssa(body);
        // 候选表只是这个函数内部的工作台，不放进结果里（见 GvnResult 的说明）。
        let mut candidates: HashMap<ExprKey, Vec<(SsaLocal, usize)>> = HashMap::new();

        for &block_id in cfg.rpo() {
            if block_id >= body.blocks.len() || !sccp.block_executable(block_id) {
                continue;
            }

            for stmt in &body.blocks[block_id].stmts {
                let (dest, value) = match stmt {
                    MirStmt::Assign { dest: MirPlace::Ssa(ssa), value } => (*ssa, value),
                    _ => continue,
                };
                if !copy_base.contains(&dest.base_id) || multi_def.contains(&dest) {
                    continue;
                }
                if let Some(Lattice::Constant(_)) = sccp.values.get(&dest) {
                    continue;
                }
                // 操作数是不是 Copy 的 SSA/常量，由 gvn_expr_key 按右值的种类各自
                // 检查——Use/Cast/Discriminant/Call 的操作数个数和位置跟二元运算
                // 不一样，在这里统一卡一道只会卡掉后面这几种。
                let key = match Self::gvn_expr_key(
                    value,
                    sccp,
                    &result.rewrite,
                    &multi_def,
                    &copy_base,
                    purity,
                ) {
                    Some(k) => k,
                    None => continue,
                };

                let hit = candidates.get(&key).and_then(|cands| {
                    cands
                        .iter()
                        .find(|(_, cand_block)| cfg.dominates(*cand_block, block_id))
                        .copied()
                });
                match hit {
                    Some((rep, rep_block)) => {
                        result.record_rewrite(dest, block_id, rep, rep_block);
                    }
                    // 只有*没被改写*的值才会进候选表——这是"rewrite 表里
                    // 没有链"这条契约的来源（AnalysisResults::is_consistent
                    // 会显式检查它，别只依赖这里的遍历顺序）。
                    None => candidates.entry(key).or_default().push((dest, block_id)),
                }
            }
        }

        result
    }

    // 构造表达式 key。形态判定和交换律规范化（`a + b` 与 `b + a`、`x + 1` 与
    // `1 + x` 落在同一个 key）统一在 analyze.rs 的 pure_expr_key，跟可用表达式
    // 分析共用；这里只负责：
    //   - 提供 GVN 自己的操作数 key：先检查操作数是"常量或 Copy 类型的 SSA"
    //     （operand_is_copy_ssa_or_const），再按 SCCP 常量结论和改写表归一化
    //     （gvn_operand_key）；任何一步返回 None 都表示这条右值不参与编号，调用
    //     方直接跳过，不是错误；
    //   - 处理 Call——它要查纯度表，只有 GVN 需要编号它。
    //
    // 这里的 match 显式列出 MirRvalue 的每个变体、不用通配：新增右值种类时
    // 编译器会在这里报错，逼着当场决定"它能不能编号"。
    fn gvn_expr_key(
        rv: &MirRvalue,
        sccp: &SccpResult,
        rewrite: &HashMap<SsaLocal, SsaLocal>,
        multi_def: &HashSet<SsaLocal>,
        copy_base: &HashSet<usize>,
        purity: &HashMap<String, Purity>,
    ) -> Option<ExprKey> {
        match rv {
            // 只有 Pure 的调用才能合并。ReadOnly 不行：它读的内存在两次调用
            // 之间可能被改过，相同参数不保证相同结果；也可能陷入（见
            // detect::Purity）。
            MirRvalue::Call { func, args, is_intrinsic, intrinsic_name, generic_args } => {
                if call_purity(func, *is_intrinsic, *intrinsic_name, purity) != Purity::Pure {
                    return None;
                }
                let mut arg_keys: Vec<OperandKey> = Vec::with_capacity(args.len());
                for a in args {
                    if !operand_is_copy_ssa_or_const(a, copy_base) {
                        return None;
                    }
                    arg_keys.push(Self::gvn_operand_key(a, sccp, rewrite, multi_def)?);
                }
                Some(ExprKey::Call(func.clone(), arg_keys, generic_args.clone()))
            }
            MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Cast(_, _)
            | MirRvalue::Use(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Phi { .. } => pure_expr_key(rv, |op| {
                if !operand_is_copy_ssa_or_const(op, copy_base) {
                    return None;
                }
                Self::gvn_operand_key(op, sccp, rewrite, multi_def)
            }),
        }
    }

    // 构造操作数 key：SCCP 已判定为常量的 SSA 按常量算（`b = 5` 之后的
    // `b + 1` 和 `5 + 1` 是同一个表达式）；其余的 SSA 先顺着 rewrite 换成
    // 代表——这样 `t2 = dest + 1` 里的 dest 如果已经被判定等于 rep，
    // key 就是 `rep + 1`，能跟别处的 `rep + 1` 合并，一层层传下去。
    //
    // rewrite 里没有链（rep 自己不会再被改写，见 gvn() 里候选表的入表
    // 条件和 is_consistent 的检查），所以一次查表就到终点，不需要循环。
    fn gvn_operand_key(
        op: &MirOperand,
        sccp: &SccpResult,
        rewrite: &HashMap<SsaLocal, SsaLocal>,
        multi_def: &HashSet<SsaLocal>,
    ) -> Option<OperandKey> {
        match op {
            MirOperand::Constant(lit) => Some(OperandKey::Const(LiteralKey::from_literal(lit))),
            MirOperand::Copy(MirPlace::Ssa(s)) | MirOperand::Move(MirPlace::Ssa(s)) => {
                if multi_def.contains(s) {
                    return None;
                }
                if let Some(Lattice::Constant(lit)) = sccp.values.get(s) {
                    return Some(OperandKey::Const(LiteralKey::from_literal(lit)));
                }
                let rep = rewrite.get(s).copied().unwrap_or(*s);
                Some(OperandKey::Ssa(rep))
            }
            _ => None,
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

    // ===== 块内重写：SCCP 常量替换 + 复制传播 =====
    // 关键设计：常量传播和复制传播现在是两条不同的路——常量的事实来自
    // analysis.sccp.values，是 Self::analyze 阶段对整个函数
    // 一次性求出的不动点结果（见上面 SCCP 一节），这里只管查、不再现
    // 算现攒；copy_map 还是老样子，只按 SsaLocal 记事实，而且每进一个
    // block 就重开一张空表，不带上一个 block 的结论。原因分两层：
    //
    // 1）为什么只认 SsaLocal，不认 MirPlace::Local：
    //    SsaLocal 才有"整个函数里最多被赋值一次"这条静态保证（builder
    //    那边每次 let/赋值都会分配一个新的 version）。MirPlace::Local
    //    在现在的 MIR 里专门留给"会被多次写"的场景——最典型的就是
    //    If/Match 的汇合点：then_block 和 else_block 各自都会往同一个
    //    dest local 里 Assign 一次，随便挑一次的值记进按"这个 id 现在
    //    等于什么"建模的表里，会把另一条分支的事实错误地带到不该带到
    //    的地方。cond_temp/disc_temp 这类 mir_builder.rs 内部临时变量
    //    虽然实际只写一次，但类型上没有 SsaLocal 这个身份，也放不进
    //    copy_map——干脆不追踪，读的时候原样保留，不算错误，只是错过
    //    一点点折叠机会。
    //
    // 2）为什么 copy_map 依然不做跨 block 传播：
    //    常量传播已经交给 SCCP——它靠 exec_edges 判断哪些前驱真的能走
    //    到、Phi 只 join 可执行边带来的值，这套机制不需要先排出支配
    //    顺序也能在循环回边上算对（见 SCCP 一节的说明）。但"y 是不是
    //    x 的别名"这类复制传播事实不一样：判断"在某处读到它时，它的
    //    定义是否已经执行过"仍然要靠支配关系，不是 block id 的数值
    //    顺序。CfgInfo 现在已经把支配树建出来了（主要是为将来的 GVN
    //    准备），但复制传播本身还没有接上 cfg.dominates，贸然跨块沿用
    //    同一张 copy_map，遇到循环回边还是会把只在某次迭代里成立的
    //    别名关系错误地套用到之前的迭代。这次没有被要求把 copy_map 也
    //    做成支配感知的，所以复制传播继续按 block-local 来：传播范围
    //    小一点，但一定不会算错。
    //
    // 可用表达式（跨块的公共子表达式）：块入口处 available_in 里的 key，如果
    // 能挑出一个"定义支配本块入口"的现成 holder（pick_holder），块内对同一个
    // key 的重新计算就直接改成读那个 holder。holder 表按"块内顺序"逐步补充：
    // 入口继承来的在最前面，块内新算出来的紧跟着登记，这样块内第二次算同一个
    // key 也能命中。这一步是跨块的，跟上面讲的 block-local 的 copy_map 是两回事
    // ——它的正确性来自支配关系，不是来自 block id 的顺序。
    fn fold_block(
        block_id: usize,
        block: &mut MirBlock,
        analysis: &AnalysisResults,
        copy_map: &mut HashMap<SsaLocal, MirOperand>,
        copy_types: &HashSet<usize>,
        def_block: &HashMap<SsaLocal, usize>,
        version_alloc: &mut VersionAllocator,
    ) -> bool {
        let mut holder: HashMap<ExprKey, SsaLocal> = HashMap::new();
        let mut inserted_phi = false;

        if let (Some(entry_keys), Some(preds)) = (
            analysis.available.available_in.get(block_id),
            analysis.cfg.predecessors().get(block_id),
        ) {
            if !preds.is_empty() {
                // 第一步：有现成的、定义严格支配本块入口的 holder 就直接读它。
                for key in entry_keys {
                    if let Some(h) = Self::pick_holder(key, block_id, preds, analysis, def_block) {
                        holder.insert(key.clone(), h);
                    }
                }

                // 第二步：剩下的 key 里，本块真的会重新计算它、而各前驱各有各的 holder
                // （if/else 两边各算一遍）——在块首插一个 Phi 把各前驱的 holder 合起来，
                // 本块里的重算就改成读这个 Phi。这是完全冗余消除（所有路径上都
                // 已经算过）能做的那一部分，不是投机：没有在任何路径上多算。
                //
                // 只给"本块里会被 fold_rvalue 真的替换"的 key 建 Phi：块里根本不重算的
                // key 建了 Phi 也没人读，下一轮被死代码消除删掉、又会被重建，永远
                // 不收敛。按语句在块里的先后顺序收集（不遍历 HashSet），Phi 的版本号
                // 分配顺序每次运行都一样。
                let mut seen_keys: HashSet<ExprKey> = HashSet::new();
                let mut phi_plans: Vec<(ExprKey, Vec<SsaLocal>)> = Vec::new();
                for stmt in &block.stmts {
                    let value = match stmt {
                        MirStmt::Assign { dest: MirPlace::Ssa(_), value } => value,
                        MirStmt::Assign { .. }
                        | MirStmt::ExprStmt(_)
                        | MirStmt::Drop { .. }
                        | MirStmt::SetMetadata { .. }
                        | MirStmt::EffectCheck { .. } => continue,
                    };
                    if !Self::rvalue_is_holder_replaceable(value) {
                        continue;
                    }
                    let key = match AvailableExprs::rvalue_key(value, copy_types) {
                        Some(k) => k,
                        None => continue,
                    };
                    if !seen_keys.insert(key.clone()) {
                        continue;
                    }
                    if holder.contains_key(&key) || !entry_keys.contains(&key) {
                        continue;
                    }
                    if let Some(hs) =
                        Self::phi_holders(&key, block_id, preds, analysis, def_block, copy_types)
                    {
                        phi_plans.push((key, hs));
                    }
                }

                if !phi_plans.is_empty() {
                    let insert_at = block.stmts.iter().take_while(|s| stmt_is_phi(s)).count();
                    let mut new_phis: Vec<MirStmt> = Vec::with_capacity(phi_plans.len());
                    for (key, hs) in phi_plans {
                        let phi_ssa = version_alloc.alloc(hs[0].base_id);
                        let values: Vec<(usize, MirOperand)> = preds
                            .iter()
                            .zip(hs.iter())
                            .map(|(&p, &h)| (p, MirOperand::Copy(MirPlace::Ssa(h))))
                            .collect();
                        new_phis.push(MirStmt::Assign {
                            dest: MirPlace::Ssa(phi_ssa),
                            value: MirRvalue::Phi { values },
                        });
                        holder.insert(key, phi_ssa);
                    }
                    block.stmts.splice(insert_at..insert_at, new_phis);
                    inserted_phi = true;
                }
            }
        }

        let mut new_stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            new_stmts.push(Self::fold_stmt(stmt, analysis, copy_map, copy_types, &mut holder));
        }
        // 关键说明：fold_stmt 是原地重写，折叠没发生时也会返回一份内容
        // 相同的新语句——用于判断"这一轮真的改写过吗"的不是长度（重写
        // 从不增删语句），是内容对比，在覆盖 block.stmts 之前，趁旧值
        // 还在就地比较一次。
        let stmts_changed = new_stmts != block.stmts || inserted_phi;
        block.stmts = new_stmts;
        let terminator_changed = Self::fold_terminator(block, analysis, copy_map);
        stmts_changed || terminator_changed
    }

    // fold_rvalue 里会按 holder 替换的右值形态：化简后仍是真正的运算的那几种。
    // 必须跟 fold_rvalue 的判定保持一致，否则会建出没人读的 Phi。
    fn rvalue_is_holder_replaceable(rv: &MirRvalue) -> bool {
        match rv {
            MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Cast(_, _)
            | MirRvalue::Discriminant { .. } => true,
            MirRvalue::Use(_)
            | MirRvalue::Call { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Phi { .. } => false,
        }
    }

    // 合并点上要给一个 key 插 Phi 时，收集每个前驱出口处的 holder（按前驱顺序）。
    // 全部满足才返回 Some：
    //   - 每个前驱都有这个 key 的 holder，且是 Copy 类型（Phi 的结果会被读多次）；
    //   - holder 的定义块能查到，并且支配*该前驱的出口*（不是支配本块入口——
    //     if/else 两边的 holder 本来就不支配汇合点，Phi 的每个 incoming 只要求
    //     在对应前驱的末尾可读）；
    //   - holder 不能定义在本块里：那是在拿本块里的定义当本块 Phi 的输入，
    //     可能把正在被改写的语句变成自己的输入，保守放弃。
    // 至少要有两个前驱，否则不需要合并。
    fn phi_holders(
        key: &ExprKey,
        block_id: usize,
        preds: &[usize],
        analysis: &AnalysisResults,
        def_block: &HashMap<SsaLocal, usize>,
        copy_types: &HashSet<usize>,
    ) -> Option<Vec<SsaLocal>> {
        if preds.len() < 2 {
            return None;
        }
        let mut hs: Vec<SsaLocal> = Vec::with_capacity(preds.len());
        for &p in preds {
            let h = *analysis.available.holder_at_out.get(p)?.get(key)?;
            if !copy_types.contains(&h.base_id) {
                return None;
            }
            let db = *def_block.get(&h)?;
            if db == block_id || !analysis.cfg.dominates(db, p) {
                return None;
            }
            hs.push(h);
        }
        Some(hs)
    }

    // 为块入口处的一个 key 选一个可以直接读的 holder。
    //
    // 收集所有前驱出口处这个 key 的 holder：
    //   - 全部相同：直接用它。所有路径都经过同一个 SSA 的定义，定义必然支配
    //     汇合点，不需要再查支配关系；
    //   - 不一致：挑第一个定义块支配本块入口的。典型是循环头：入口边的
    //     holder 定义在循环外（支配循环头），回边的 holder 定义在循环体内
    //     （不支配），前者被选中，不需要 phi；
    //   - 都不行：返回 None。这是真的要插 phi 才能合并的情形（if/else 两边各
    //     算一遍），保守跳过。
    // 任何一个前驱的表里没有这个 key（available_in 保证了正常情况不会发生，
    // 畸形输入下才可能）也返回 None。
    //
    // 关键防护（两条路径都要过）：候选 holder 的定义块必须能在 def_block 里查
    // 到（查不到：参数、或同名多定义被剔除），并且不能就是本块自己——holder 若
    // 定义在本块内，要么就是正在被改写的这条语句自己（改成 `x = x`），要么
    // 定义在使用点之后，都是 use-before-def。dominates 对 a == b 返回 true，
    // 所以"定义块支配本块入口"不能直接用 dominates，必须是严格支配。
    fn pick_holder(
        key: &ExprKey,
        block_id: usize,
        preds: &[usize],
        analysis: &AnalysisResults,
        def_block: &HashMap<SsaLocal, usize>,
    ) -> Option<SsaLocal> {
        let mut holders: Vec<SsaLocal> = Vec::with_capacity(preds.len());
        for &p in preds {
            match analysis.available.holder_at_out.get(p).and_then(|m| m.get(key)) {
                Some(&h) => holders.push(h),
                None => return None,
            }
        }
        if holders.is_empty() {
            return None;
        }
        let strictly_dominates_entry = |h: SsaLocal| -> bool {
            match def_block.get(&h) {
                Some(&db) => db != block_id && analysis.cfg.dominates(db, block_id),
                None => false,
            }
        };
        if holders.iter().all(|h| *h == holders[0]) {
            if strictly_dominates_entry(holders[0]) {
                return Some(holders[0]);
            }
            return None;
        }
        for &h in &holders {
            if strictly_dominates_entry(h) {
                return Some(h);
            }
        }
        None
    }

    // ===== 语句折叠：维护 copy_map（常量已由 SCCP 统一求出）=====
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
        analysis: &AnalysisResults,
        copy_map: &mut HashMap<SsaLocal, MirOperand>,
        copy_types: &HashSet<usize>,
        holder: &mut HashMap<ExprKey, SsaLocal>,
    ) -> MirStmt {
        match stmt {
            MirStmt::Assign { dest, value } => {
                // key 取自*折叠之前*的原始右值：available_in / holder 表都是从
                // 这一轮开头的 body 算出来的，用同一份口径查才对得上。
                let original_key = AvailableExprs::rvalue_key(value, copy_types);
                let folded_value = Self::fold_rvalue(value, analysis, copy_map, holder, copy_types);

                // 关键设计：现在只往 copy_map 里记事实，不再像旧版
                // 那样顺手把折叠出的 Use(Constant(_)) 也塞进一张
                // const_map——常量已经是 SCCP 在分析阶段对整个函数
                // 求好的不动点结果，这里是重写阶段，重新往一张
                // block-local 的表里记一遍不会让 fold_operand 查得
                // 更全（该查的都在 analysis.sccp.values 里），反而会
                // 让"常量事实到底以哪张表为准"变得含糊。
                if let MirPlace::Ssa(ssa) = dest {
                    if let MirRvalue::Use(
                        op @ (MirOperand::Copy(MirPlace::Ssa(_))
                            | MirOperand::Move(MirPlace::Ssa(_))),
                    ) = &folded_value
                    {
                        // folded_value 是 fold_rvalue 算出来的，内部
                        // 操作数已经在 fold_operand 里顺着 SCCP 常量
                        // 和 copy_map 尽量解析过一轮了——走到这里还是
                        // Copy/Move(Ssa(src))，说明 src 目前两边都没有
                        // 已知的结论，这就是"最终来源"，直接存，不需要
                        // 在查表那一侧再递归链式往下追一次。
                        if copy_types.contains(&ssa.base_id) {
                            copy_map.insert(*ssa, op.clone());
                        }
                    }
                }

                // 登记块内新算出来的 key：已经有 holder（入口继承、或块内更早算
                // 过）就保持原来的——这条语句多半刚被改写成读它。
                if let MirPlace::Ssa(dest_ssa) = dest {
                    if let Some(k) = original_key {
                        holder.entry(k).or_insert(*dest_ssa);
                    }
                }

                MirStmt::Assign {
                    dest: dest.clone(),
                    value: folded_value,
                }
            }
            MirStmt::ExprStmt(value) => {
                MirStmt::ExprStmt(Self::fold_rvalue(value, analysis, copy_map, holder, copy_types))
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
    // 关键设计："哪些后继可行"只有一个权威来源——sccp_successors（跟
    // SCCP 自己传播 exec_edges 时用的是同一个函数）。旧版这里自己再判
    // 一遍 cond/discr 是不是常量，两份逻辑一旦出现细微分歧（比如 Switch
    // 的字面量比较口径不同），就会出现"SCCP 认为某条边不可行、终止器却
    // 还连着它"的状态：Phi 按 exec_edges 裁掉了 incoming，CFG 上却还有
    // 这条边，SSA 结构对不上。现在先问 SCCP：只剩唯一可行后继就直接改成
    // Goto；剩下没法折叠的情况才只做操作数替换（别名/常量）。
    //
    // 关键修复（then_block == else_block 的退化被漏掉）：两个分支目标
    // 本来就是同一个块时，不管 cond 求值成什么，走的都是那一个地方——
    // 这是结构性事实，不需要问 SCCP。但 sccp_successors 对拿不准的 cond
    // （Overdefined）统一返回 `vec![then_block, else_block]`，两个元素
    // 相同也不去重，长度仍然是 2，不会被下面"只剩一个可行后继"的判断
    // 命中。`if c { x } else { x }` 这种写法在源码里并不少见（尤其是
    // 生成式代码），单独判一次成本很低，放在问 SCCP 之前。
    fn fold_terminator(
        block: &mut MirBlock,
        analysis: &AnalysisResults,
        copy_map: &HashMap<SsaLocal, MirOperand>,
    ) -> bool {
        // 关键说明：不管走哪条分支，最后都靠"改写前后是否相等"统一
        // 判断有没有变化，不需要在每个分支各自维护一份"是否改写过"的
        // 标记——MirTerminator 已经是 PartialEq（tail_merge 等处已经在
        // 用），这里克隆一份原始值，函数末尾比较一次即可。
        let original = block.terminator.clone();

        if let MirTerminator::If { then_block, else_block, .. } = &block.terminator {
            if then_block == else_block {
                block.terminator = MirTerminator::Goto(*then_block);
                return block.terminator != original;
            }
        }

        match &block.terminator {
            MirTerminator::If { .. } | MirTerminator::Switch { .. } => {
                let feasible = Self::sccp_successors(&block.terminator, &analysis.sccp.values);
                if feasible.len() == 1 {
                    block.terminator = MirTerminator::Goto(feasible[0]);
                    return block.terminator != original;
                }
            }
            MirTerminator::Goto(_)
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => return false,
        }

        match &block.terminator {
            MirTerminator::If { cond, then_block, else_block } => {
                let folded_cond = Self::fold_operand(cond, analysis, copy_map);
                block.terminator = MirTerminator::If {
                    cond: folded_cond,
                    then_block: *then_block,
                    else_block: *else_block,
                };
            }
            MirTerminator::Switch { discr, discr_ty, targets, default } => {
                let folded_discr = Self::fold_operand(discr, analysis, copy_map);
                block.terminator = MirTerminator::Switch {
                    discr: folded_discr,
                    discr_ty: discr_ty.clone(),
                    targets: targets.clone(),
                    default: *default,
                };
            }
            MirTerminator::Goto(_)
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => {}
        }

        block.terminator != original
    }

    // ===== 右值折叠 =====
    // 先做操作数/常量/代数化简（fold_rvalue_operands）；化简后仍然是一个
    // 需要真算的运算（没有被折成常量、也没被化成单纯的 Use）时，再查 holder：
    // 这个 key 如果已经有现成的变量持有，就改成读它。顺序不能反——常量折叠
    // 比"读另一个变量"更好，且不依赖任何支配关系。
    //
    // 只有 holder 是 Copy 类型才替换：被替换的值之后可能被读很多次，非 Copy 的
    // 值读一次就被消耗了，会造成 use-after-move（key 里本来就只放 Copy 操作数，
    // 这里是对 holder 自己类型的再确认）。
    fn fold_rvalue(
        rv: &MirRvalue,
        analysis: &AnalysisResults,
        copy_map: &HashMap<SsaLocal, MirOperand>,
        holder: &HashMap<ExprKey, SsaLocal>,
        copy_types: &HashSet<usize>,
    ) -> MirRvalue {
        let folded = Self::fold_rvalue_operands(rv, analysis, copy_map);
        match folded {
            MirRvalue::Use(_) => return folded,
            MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Cast(_, _)
            | MirRvalue::Discriminant { .. } => {
                if let Some(key) = AvailableExprs::rvalue_key(rv, copy_types) {
                    if let Some(&h) = holder.get(&key) {
                        if copy_types.contains(&h.base_id) {
                            return MirRvalue::Use(MirOperand::Copy(MirPlace::Ssa(h)));
                        }
                    }
                }
            }
            MirRvalue::Call { .. }
            | MirRvalue::MethodCall { .. }
            | MirRvalue::StructInit { .. }
            | MirRvalue::EnumVariantConstruction { .. }
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Phi { .. } => {}
        }
        folded
    }

    fn fold_rvalue_operands(
        rv: &MirRvalue,
        analysis: &AnalysisResults,
        copy_map: &HashMap<SsaLocal, MirOperand>,
    ) -> MirRvalue {
        match rv {
            MirRvalue::BinaryOp(op, left, right) => {
                let l = Self::fold_operand(left, analysis, copy_map);
                let r = Self::fold_operand(right, analysis, copy_map);
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
                let folded = Self::fold_operand(operand, analysis, copy_map);
                if let MirOperand::Constant(lit) = &folded {
                    if let Some(result) = Calc::eval_unary_op(*op, lit.clone()) {
                        return MirRvalue::Use(MirOperand::Constant(result));
                    }
                }
                MirRvalue::UnaryOp(*op, folded)
            }
            MirRvalue::Use(operand) => {
                MirRvalue::Use(Self::fold_operand(operand, analysis, copy_map))
            }
            MirRvalue::Cast(operand, ty) => {
                MirRvalue::Cast(Self::fold_operand(operand, analysis, copy_map), ty.clone())
            }
            MirRvalue::Call { func, args, is_intrinsic, intrinsic_name, generic_args } => {
                MirRvalue::Call {
                    func: func.clone(),
                    args: args
                        .iter()
                        .map(|a| Self::fold_operand(a, analysis, copy_map))
                        .collect(),
                    is_intrinsic: *is_intrinsic,
                    intrinsic_name: intrinsic_name.clone(),
                    generic_args: generic_args.clone(),
                }
            }
            MirRvalue::MethodCall { receiver, method, args, generic_args } => {
                MirRvalue::MethodCall {
                    receiver: Self::fold_operand(receiver, analysis, copy_map),
                    method: method.clone(),
                    args: args
                        .iter()
                        .map(|a| Self::fold_operand(a, analysis, copy_map))
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
                        .map(|(name, op)| (name.clone(), Self::fold_operand(op, analysis, copy_map)))
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
                        .map(|a| Self::fold_operand(a, analysis, copy_map))
                        .collect(),
                }
            }
            MirRvalue::ArrayLiteral(elements) => {
                MirRvalue::ArrayLiteral(
                    elements
                        .iter()
                        .map(|e| Self::fold_operand(e, analysis, copy_map))
                        .collect(),
                )
            }
            MirRvalue::Discriminant { value, enum_name } => {
                MirRvalue::Discriminant {
                    value: Self::fold_operand(value, analysis, copy_map),
                    enum_name: enum_name.clone(),
                }
            }
            // 关键修复：Phi 的 incoming 原来被原样透传，操作数不经过
            // fold_operand——被 SCCP 判成常量的值、被 GVN 判成重复计算的值，
            // 在 Phi 里读到的还是旧名字：既没吃到常量，那条被改写掉的重复
            // 计算也因为"还被 Phi 引用着"而删不掉。现在 Phi 的每个 incoming
            // 操作数也走 fold_operand。
            //
            // 这样做是安全的，不需要额外的支配检查：
            //   - SCCP 常量在任何位置都成立，直接换。
            //   - GVN 只在"代表的定义块支配 dest 的定义块"时才登记改写；而
            //     Phi 的 incoming (pred, v) 要求 v 的定义支配 pred 块的末尾，
            //     所以代表同样支配 pred 块末尾，在那里读代表是合法的。
            //   - copy_map 是 block-local 的别名表，Phi 都在块首，此刻通常
            //     还是空的，即使不空，SSA 别名关系本身也是全局成立的。
            // 只替换操作数，pred 块 id 原样保留，Phi 的形状（有几条 incoming、
            // 各自来自哪个前驱）由 simplify_phi 按可执行边去裁剪，不在这里动。
            MirRvalue::Phi { values } => {
                let mut folded: Vec<(usize, MirOperand)> = Vec::with_capacity(values.len());
                for (pred, op) in values {
                    folded.push((*pred, Self::fold_operand(op, analysis, copy_map)));
                }
                MirRvalue::Phi { values: folded }
            }
            // Ref 借用的是一个地址本身（place），"把它替换成常量"没有
            // 意义，原样透传。
            MirRvalue::Ref { .. } => rv.clone(),
        }
    }

    // ===== 操作数折叠：SCCP 常量 -> GVN 代表 -> copy_map 别名 =====
    // 关键设计：三级查表，优先级从高到低，越靠前越通用、越不依赖位置：
    //   1. SCCP 常量：编译期常量在任何位置都能直接用，不需要支配检查。
    //   2. GVN 代表：gvn() 只在"代表的定义块支配 dest 的定义块"时才登记
    //      改写（AnalysisResults::is_consistent 也会验证这一点），而 dest
    //      的每一处读取又被 dest 的定义支配，所以读取处一定能看到代表。
    //   3. copy_map：block-local 的别名兜底，前两级都没命中才查。
    //
    // GVN 这一级为什么改成 Copy 而不是保留原来的 Move：能这样做，*仅仅*
    // 因为 gvn() 的入口条件要求 dest 必须是 Copy 类型，于是 rep 也必然是
    // Copy 类型，读多少次都合法；如果换成 Move，会把 rep 消耗掉，破坏它
    // 之后的所有使用。这是跟 gvn() 硬耦合的：将来如果放宽了"dest 必须
    // 是 Copy"，这里必须同步改成"保留原来的 Copy/Move 属性"或者引入临时
    // 变量，否则会静默地产生 use-after-move。
    //
    // 关键说明：Phi 的 incoming 也走这里（fold_rvalue 的 Phi 分支逐个调用
    // fold_operand），所以被改写的 dest 一旦所有读取（含 Phi 里的）都换成
    // 了代表，它自己的定义语句就成了死赋值，由 eliminate_dead_stores 清掉。
    fn fold_operand(
        op: &MirOperand,
        analysis: &AnalysisResults,
        copy_map: &HashMap<SsaLocal, MirOperand>,
    ) -> MirOperand {
        match op {
            MirOperand::Copy(MirPlace::Ssa(ssa)) | MirOperand::Move(MirPlace::Ssa(ssa)) => {
                if let Some(Lattice::Constant(lit)) = analysis.sccp.values.get(ssa) {
                    return MirOperand::Constant(lit.clone());
                }
                if let Some(&rep) = analysis.gvn.rewrite.get(ssa) {
                    return MirOperand::Copy(MirPlace::Ssa(rep));
                }
                if let Some(resolved) = copy_map.get(ssa) {
                    // resolved 在写入 copy_map 的时候已经是解析过的
                    // 结果（见 fold_stmt 里的说明），这里一次查表就够，
                    // 不需要再递归链式解析。
                    return resolved.clone();
                }
                op.clone()
            }
            // 非 Ssa 的 place（Field/Index/Deref/EnumPayload）不在
            // SCCP/GVN/copy_map 的追踪范围内，原样返回——理由见 fold_block
            // 顶部的说明。Sym/Static 同样原样返回：它们是 mir.rs 这一轮
            // 从 MirPlace 搬到 MirOperand 的两个变体，不指向任何局部
            // 变量，根本不可能出现在这几张表里，没有什么可折叠的。
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
                if is_zero(left) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if is_zero(right) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if is_one(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if is_one(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            Add => {
                if is_zero(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if is_zero(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            Sub => {
                if is_zero(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
            }
            And => {
                if is_bool_true(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if is_bool_true(right) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if is_bool_false(left) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if is_bool_false(right) {
                    return Some(MirRvalue::Use(right.clone()));
                }
            }
            Or => {
                if is_bool_true(left) {
                    return Some(MirRvalue::Use(left.clone()));
                }
                if is_bool_true(right) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if is_bool_false(left) {
                    return Some(MirRvalue::Use(right.clone()));
                }
                if is_bool_false(right) {
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

    // ===== 空块消除（不涉及 Ssa/Local 的读写事实，跟 SSA 改造本身无关；
    // 下面第二个循环里 If/Switch 分支的写法后来发现有借用检查过不去的
    // 问题，已经修——见那两个分支上面的说明） =====
    fn collapse_empty_blocks(body: &mut MirBody) -> bool {
        if body.blocks.len() <= 1 {
            return false;
        }

        let block_count = body.blocks.len();
        // 关键修复：目标块里有 Phi 的空块不能折叠。Phi 的 incoming 是按
        // 前驱块 id 记的，空块 E 被折叠掉之后，E 的所有前驱会直接跳到
        // 目标块，Phi 里原来记在 E 名下的那个值就没有着落了（应该拆成
        // 每个前驱各一份）——老代码不处理这一点，Phi 一旦出现就会悄悄丢
        // 值。这里选择保守：目标块有 Phi 就不折叠这个空块。
        let mut has_phi: Vec<bool> = Vec::with_capacity(block_count);
        for b in &body.blocks {
            has_phi.push(block_contains_phi(b));
        }
        // 关键修复（跟 control.rs 的 insert_preheaders 互相撤销）：专用
        // preheader 通常就是一个只含 `Goto(循环头)` 的空块，在这里跟任何别的
        // 空跳板没有区别，会被折叠掉；insert_preheaders 下一轮又把它补回来，
        // 两边来回拉扯，外层循环永远报告 changed、永远不收敛。所以专用
        // preheader 不参与折叠。这里现算一份 CFG/循环信息而不用 analysis
        // 里那份：analysis 是这一轮开头的快照，前面的 remove_dead_blocks
        // 已经重编号过块，快照的块编号此刻是错位的。
        let protected = {
            let cfg = CfgInfo::build(body);
            LoopInfo::build(body, &cfg).preheader_blocks()
        };
        let mut replaced_targets: HashMap<usize, usize> = HashMap::new();
        for (block_id, block) in body.blocks.iter().enumerate() {
            // 直接 if let / match 取出 Goto 的目标，不先问一遍"是不是
            // Goto"再解一遍：同一件事不问两次。
            if block_id == 0 || !block.stmts.is_empty() || protected.contains(&block_id) {
                continue;
            }
            let target = match &block.terminator {
                MirTerminator::Goto(t) => *t,
                _ => continue,
            };
            if target == block_id {
                continue;
            }
            if target >= block_count || has_phi[target] {
                continue;
            }
            replaced_targets.insert(block_id, target);
        }

        // 关键修复：空块之间可能互相 Goto 成环（E1 -> E2 -> E1，比如
        // 循环头和空循环体各自是一个空块）。下面重定向用的是
        // `while let Some(..) = replaced_targets.get(..)` 顺着链一路追，
        // 遇到环就永远退不出来——编译器直接卡死。这里先把"顺着链走
        // block_count 步还没走出 replaced_targets"的项（环上的，以及
        // 指向环的）都剔除，它们保持原样不折叠；剩下的链一定是有限的。
        let cyclic: Vec<usize> = replaced_targets
            .keys()
            .copied()
            .filter(|&start| {
                let mut cur = start;
                for _ in 0..block_count {
                    match replaced_targets.get(&cur) {
                        Some(&next) => cur = next,
                        None => return false,
                    }
                }
                true
            })
            .collect();
        for k in cyclic {
            replaced_targets.remove(&k);
        }

        if replaced_targets.is_empty() {
            return false;
        }

        // 关键重构：终止器的每个目标顺着 replaced_targets 追到链尾。
        // 遍历"有哪些目标"由 MirTerminator::for_each_target_mut 统一负责，
        // 这里不再自己对 Goto/If/Switch 各写一遍 match——上面已经剔除了成
        // 环的项，所以这个 while 一定会停。
        for block in &mut body.blocks {
            block.terminator.for_each_target_mut(|t| {
                while let Some(&new_target) = replaced_targets.get(&*t) {
                    *t = new_target;
                }
            });
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

        // 关键重构：块重编号时，终止器的每个目标都换成新编号；只要有任何
        // 一个目标解析不到（理论上不会——上面已经把所有目标重定向到不会
        // 被折叠掉的块了；但畸形 MIR 里目标本身可能越界），整个终止器
        // 退化成 Unreachable，宁可安全地"到不了"，也不带着一个不存在的
        // 块 id 继续跑。
        //
        // 这里用 for_each_target_mut 逐个改写目标，再由"是否全部解析
        // 成功"的标记决定要不要整体降级，替代原来 Goto/If/Switch 三段
        // 各自展开的 match：那三段既重复，又对 id_map 做了裸下标（越界
        // 会 panic），现在统一走 `.get()`，越界和"被删掉"按同一种情况
        // 处理。改写发生在终止器内部，整体降级是最后一步的整体赋值，
        // 不会跟前面的借用冲突（这曾经是 E0506 的来源）。
        for block in &mut new_blocks {
            let mut all_resolved = true;
            block.terminator.for_each_target_mut(|t| match id_map.get(*t).copied().flatten() {
                Some(new_target) => *t = new_target,
                None => all_resolved = false,
            });
            if !all_resolved {
                block.terminator = MirTerminator::Unreachable;
            }
        }

        // 块重编号之后，Phi 里记的前驱块 id 也要跟着换。
        MirBody::remap_phi_preds(&mut new_blocks, &id_map);
        // 块自带的 id 字段也要跟着换成新下标（见 MirBody::retain_and_renumber_blocks）。
        for (new_id, block) in new_blocks.iter_mut().enumerate() {
            block.id = new_id;
        }

        body.blocks = new_blocks;
        // 折叠掉空块之后块的集合和编号都变了：按块编号索引的分析快照作废。
        body.bump_revision();
        // 关键说明：走到这里一定是真的改写过——上面两处 replaced_targets
        // 判空的早退都已经用 `return false` 拦在前面了，剩下能走到这里
        // 的路径必然至少折叠了一个空块。
        true
    }

    // ===== 死赋值消除 =====
    // 关键重构：判断"这条赋值的结果有没有人读"改用 analyze.rs 的 Liveness——
    // 语句执行完之后，dest 这个 SSA 值在活跃集合里吗。原来是先扫一遍整个
    // 函数收集"出现过读取的 SSA 值"，再把没出现的删掉：那是一个跟程序点
    // 无关的全局集合，而且这个"收集读取"的逻辑跟借用检查里的活跃分析是
    // 各写各的两份。现在两边共用同一份分析和同一套 use/def 提取。
    //
    // 一次调用只做一轮：删掉一条死赋值之后，它读取的那些值可能跟着变成死的，
    // 要靠外层不动点循环（optimize_function 的 changed）再来一轮才能删掉。
    //
    // 这个函数只做删除，不做原地改写——一条语句要么原样保留、要么整条丢弃，
    // 所以比较改动前后的语句数就足够判断"这一轮是否真的删过东西"。
    //
    // 关键说明：前驱表现算一份（body.predecessor_table），不依赖
    // optimize_function 里那份 analysis 快照——前面的 remove_dead_blocks 已经
    // 重编号了块，快照里按块编号索引的东西此刻都是错位的。
    //
    // 只删 dest 是 Ssa 的赋值。dest 是 Field/Index/Deref/EnumPayload 的写入是
    // 写进内存里某个更复杂的位置，这里不追踪"这个具体位置有没有被读到"
    // （可能被另一条路径别名读取），一律保留。
    fn eliminate_dead_stores(
        body: &mut MirBody,
        purity: &HashMap<String, Purity>,
        structs: &[MirStruct],
        enums: &[MirEnum],
    ) -> bool {
        if body.blocks.is_empty() {
            return false;
        }
        let preds = body.predecessor_table();
        let liveness = Liveness::build_from_preds(body, &preds);

        // 两阶段：rvalue_has_side_effect 要读整个 body（查局部变量的静态类型，
        // 判断常量下标会不会越界），而改写阶段要对 blocks 可变遍历，两者同时
        // 借用会冲突。所以先只读地决定每条语句留不留，再统一执行。
        let mut plans: Vec<Vec<bool>> = Vec::with_capacity(body.blocks.len());
        for (bi, block) in body.blocks.iter().enumerate() {
            let mut keep: Vec<bool> = Vec::with_capacity(block.stmts.len());
            for (si, stmt) in block.stmts.iter().enumerate() {
                let should_keep = match stmt {
                    MirStmt::Assign { dest: MirPlace::Ssa(s), value } => {
                        let is_dead = !liveness.live_after[bi][si].contains(s);
                        !(is_dead && !rvalue_has_side_effect(value, purity, body, structs, enums))
                    }
                    MirStmt::Assign { .. }
                    | MirStmt::ExprStmt(_)
                    | MirStmt::Drop { .. }
                    | MirStmt::SetMetadata { .. }
                    | MirStmt::EffectCheck { .. } => true,
                };
                keep.push(should_keep);
            }
            plans.push(keep);
        }

        // plans 和 body.blocks 在上面同一次遍历里按同样的顺序、同样的长度构造，
        // zip 两边必然等长。
        let mut changed = false;
        for (block, keep) in body.blocks.iter_mut().zip(plans.into_iter()) {
            let before = block.stmts.len();
            let old_stmts = std::mem::take(&mut block.stmts);
            let mut new_stmts: Vec<MirStmt> = Vec::with_capacity(before);
            for (stmt, k) in old_stmts.into_iter().zip(keep.into_iter()) {
                if k {
                    new_stmts.push(stmt);
                }
            }
            if new_stmts.len() != before {
                changed = true;
            }
            block.stmts = new_stmts;
        }
        changed
    }

    // ===== 死块消除 =====
    // 关键设计：删除依据是 SCCP 的"可执行块"，而不是单纯结构上的可达
    // ——后者只要有一条边连着就算可达，条件恒为假的分支目标也算；前者
    // 已经把常量条件的结论算进去了。这也是本函数会重编号的原因，它用
    // 完 analysis 之后，快照就作废了（见 optimize_function 的说明）。
    //
    // 关键重构：这里只负责回答"保留哪些块"。"怎么删"——压缩块数组、
    // 重编号终止器目标和 Phi 前驱、以及目标被删时终止器怎么降级——原来在
    // 这里和 control.rs 各有一份近乎逐字相同的实现，Switch 缺 default 的
    // 兜底就是只在一边出过错。现在统一走 MirBody::retain_and_renumber_blocks，兜底策略
    // 只有那一处。
    fn remove_dead_blocks(body: &mut MirBody, analysis: &AnalysisResults) -> bool {
        let block_count = body.blocks.len();
        if block_count == 0 {
            return false;
        }
        if !analysis.matches(body) {
            debug_assert!(false, "remove_dead_blocks: stale analysis snapshot");
            return false;
        }

        // 入口块恒可执行（is_consistent 里有检查），retain_and_renumber_blocks 的入口块
        // 不变量因此一定成立。
        let mut keep: Vec<bool> = Vec::with_capacity(block_count);
        for b in 0..block_count {
            keep.push(analysis.sccp.block_executable(b));
        }
        body.retain_and_renumber_blocks(&keep)
    }

    // ===== Phi 简化 =====
    // 关键设计：按 SCCP 的可执行**边**裁剪 incoming，而不是按前驱块是否
    // 可达。前驱块可执行不代表它会跳到这个块（比如它的 If 条件恒为真、
    // 走的是另一侧），只有 (pred -> 本块) 这条边可执行，它带来的值才
    // 可能真的流进来。前驱以 Unreachable 结尾的情况也自然被覆盖——它
    // 没有后继，不会产生任何可执行边，不需要再单独预先算一份
    // "每个块的终止器是不是 Unreachable"（旧版为了绕开借用检查才有
    // 那份辅助数组，现在整个循环体里都不再需要回头读 body.blocks）。
    fn simplify_phi(body: &mut MirBody, analysis: &AnalysisResults) -> bool {
        if !analysis.matches(body) {
            debug_assert!(false, "simplify_phi: stale analysis snapshot");
            return false;
        }

        // 关键说明：跟 fold_block 同样的道理——Phi 退化成 Use、裁剪
        // incoming 之后语句数量不变，只能靠内容对比判断有没有真的
        // 改写过。
        let mut changed = false;
        for (block_id, block) in body.blocks.iter_mut().enumerate() {
            let mut new_stmts = Vec::with_capacity(block.stmts.len());
            for stmt in &block.stmts {
                let simplified = match stmt {
                    MirStmt::Assign { dest, value: MirRvalue::Phi { values } } => {
                        // 只保留来自可执行边的 incoming
                        let mut live_values: Vec<(usize, &MirOperand)> = Vec::new();
                        for (pred, op) in values.iter() {
                            if analysis.sccp.edge_executable(*pred, block_id) {
                                live_values.push((*pred, op));
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
            if new_stmts != block.stmts {
                changed = true;
            }
            block.stmts = new_stmts;
        }
        changed
    }
}
