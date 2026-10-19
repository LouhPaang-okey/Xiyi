// borrow.rs
//
// 完全 Polonius 化重构：这套借用检查建立在三层之上——
//
//   1. SSA 粒度 liveness：每个 SSA 值 (base_id, version) 的存活区间。
//      （分析本身和 SSA use/def 提取在 analyze.rs 的 Liveness，借用检查和
//      死赋值消除共用同一份，这里只消费它的结果。）
//      这个粒度是相对老版本（变量 base_id 粒度）的关键升级：只有它才能
//      正确区分 `r = &x; r = &y; use(r);` 里旧版本 r@1 已经死亡、新版本
//      r@2 仍然活跃——用 base_id 判两个 loan 都会在 use 处被误判活跃。
//
//   2. Origin 收集：每个 SSA 值分配一个 origin。origin 的"活跃点集合"
//      初始化自 liveness 给出的存活区间，再通过子集约束（SubsetConstraint）
//      向前传播。
//
//   3. 子集约束：目前有两条来源——
//        · `dst = src`（引用别名/移动/复制）：dst 的活跃点都在 src 的
//          活跃点里，因为 dst 在那些点解引用得到的就是 src 的 target。
//        · `dst = phi(src_i...)`：汇合点上，输出的活跃性必须覆盖到每
//          一个入边——运行期具体走哪条 branch 不确定，保守起见每个入边
//          的 origin 都要能在那时被"叫醒"。
//      不动点求解后每个 origin 得到最终点集合，再投影成
//      `loan_live_at`（Polonius 术语：程序点 → 该点活跃的 loan 列表）。
//
// 借用检查主体只做一次 `loan_live_at` 的哈希表查询，不再每次从头遍历
// loan 列表并反推活跃性。这跟 Polonius 论文的
// `origin_contains_loan_on_entry` / `loan_live_at` 是同一个数据模型，
// 只是用 BTreeSet 不动点代替了 datalog。
//
// 历史修复（问题清单一~五，仍然成立）：
//   一、Placeholder terminator 的策略统一：check_func_status 入口先扫
//       一遍所有 block 的 terminator，发现 Placeholder 直接早退报内部
//       错误——init_check 的 term_uses 和 borrow_check 的
//       check_terminator 都不会再碰到它，两处策略不再打架。
//   二、`*r` 的 flatten 不再错误地以 r 自己为 root：引用变量指向哪些
//       place 由 points-to 表（RefFacts.pt）给出，flatten 遇
//       Deref(Ssa(s)) 优先展开成 s 的 target。查不到（如函数参数传入的
//       引用）时退回 root = Local(s.base_id) + [Deref]，保守但不错。
//   三、init_check 与 borrow_check 不再互相短路：两个 pass 都跑完，
//       错误合并后一次性报给用户；check_fn 更名为 check_func_status，
//       避免与 semantic 的文件重名。
//   四、mark_live 批量化：先把 (SSA -> 活跃点列表) 聚合成 HashMap，
//       每个 SSA 只做一次 origin_for 哈希 + 一次 seeds.entry B 树查找。
//   五、ProgramPoint.stmt 越界防护：新增 stmt_point 构造器，
//       debug_assert!(stmt <= stmts_len)；收集与检查两处统一走构造器。
//
// 本版新增：
//   六、通过引用访问它自己的 target 不再自撞。`*r = 5`、`let v = *r;`、
//       `let x = &mut *r; *x = 5;` 里，访问路径上的那个引用变量就是这次
//       访问的授权来源，它携带的 loan 不算冲突源。授权按"引用变量携带了
//       哪些 loan"（RefFacts.carried，沿别名/Phi/重借出传递）判定，而不是只比较
//       loan.borrow_var 是不是同一个变量——后者在 `r2 = r; *r2 = 5;` 这种
//       经过一次拷贝/移动的写法上会误报。check_read / check_exclusive /
//       check_borrow_creation 三处统一处理。
//   七、points-to 取代单一 ref_target：flow-insensitive 不动点，一个引用
//       变量可以指向多个 place（Phi 汇合时取并集），部分来源未知时保留
//       "还可能指向别处"的标记，flatten 对未知部分退回保守展开——旧的
//       ref_target 在 Phi 汇合处没有条目，`*dst = 5` 会退回一个跟真实
//       target 毫无关系的 root，检测不到跟已有 loan 的冲突。
//   八、place 投影精确化：常量下标 a[0] / a[1] 不重叠；枚举不同变体的
//       payload 不重叠；`*x` 的两次访问若字段不同，同样判为不重叠由
//       places_overlap 的前缀比较处理。
//   九、局部移动：init_check 改成"可能未初始化路径集合"（前向 may 分析，
//       合并取并集），移动 `p.a` 之后 `p.b` 仍然可用，`p` 整体和 `p.a`
//       不可用，重新给 `p.a` 赋值即可恢复。只有 Ssa / Field 链上的移动
//       才记录；经过 Index / Deref / EnumPayload 的读取不是对基址的移动。
//   十、两阶段借用（two-phase borrow）：`v.push(v.len())` 里，为方法
//       调用自动取的 `&mut v` 在"创建 -> 作为 receiver 被调用"之间处于预留
//       阶段，允许读取和共享借用；真正调用（激活）时再检查有没有别的活跃
//       借用冲突。只对编译器生成的临时引用生效（见 two_phase 的判定）。
//   十一、`term_use_paths` 等函数里的 `_ => ...` 通配全部改成显式列出变体，
//       不再有 matches! 宏；前驱/后继表统一来自 MirBody::edge_tables，
//       不再自己扫一遍终止器；活跃分析与 SSA use/def 提取抽到 analyze.rs，
//       不再在这里留一份（跟死赋值消除各写各的，口径不一致）。
//   十二、重借出（`x = &mut *r`）继承父引用携带的 loan：`x = &mut *r;
//       *x = 5; use(r);` 里 r 的 loan 在 x 死后仍然活跃，写 `*x` 展开后落在
//       r 借来的位置上，只有 x 自己的 loan 被授权会误报。pt 与 carried
//       的传播规则因此拆开（flow_pt / flow_carried）。
//
// 已知边界（如实标注）：
//   - Phi 汇合点上每个入边的 origin 都会在输出活跃处被叫醒——比
//     "运行期真的走这条边才活跃"更保守。完全精确需要按路径敏感的
//     Polonius subscription 级别处理，留到后续。
//   - points-to 是 flow-insensitive 的，只覆盖 Ref / 别名 / Phi 三种来源；
//     函数参数、调用返回值、从结构体字段里读出来的引用，target 未知，
//     退回保守展开。
//   - 局部移动的类型判定只覆盖非泛型结构体字段和元组；泛型结构体的字段、
//     经过引用的字段无法确定类型，一律不当作移动（跟以前"局部移动完全
//     不记录"的行为一致，不会因此多出误报）。
//   - 两阶段借用只认方法调用的 receiver（Rust 里自动引用的那一类）；
//     普通函数调用里显式写的 `&mut v` 实参不是两阶段借用，`f(&mut v, v.len())`
//     仍然报错，跟 Rust 一致。
//   - 常量下标只认直接写在 MIR 里的常量，不认 SCCP 的结论（borrow.rs
//     不依赖优化分析），经过 SSA 变量的下标一律按动态处理。

use crate::analyze::{term_ssa_uses, Liveness};
use crate::detect::{authorized_borrow_var, int_literal_value};
use crate::mir::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// ============================================================
// 路径类型（places_overlap / init_check 共用的模块级定义）
// ============================================================
//
// 这几个类型原来写在 `impl BorrowChecker { ... }` 里面——impl 块里不能
// 定义 enum/struct，只能放函数、关联常量和关联类型，挪到模块级。

/// place 展开之后的"根"。Any 表示"可能是任何位置"，用于展开深度/数量
/// 超限时的保守兜底，跟所有路径都重叠。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Root {
    Local(usize),
    Any,
}

/// place 上的一步投影。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Proj {
    Field(String),
    /// 常量下标，用 i128 装得下所有整数字面量的值，不做有损截断。
    ConstIndex(i128),
    /// 动态下标。
    DynIndex,
    /// 枚举变体的 payload。
    Variant(String),
    Deref,
    /// 以后要加新的不透明投影时用。
    #[allow(dead_code)]
    Opaque,
}

/// 展开后的一条候选路径：根 + 从根往外的投影序列。
type Flat = (Root, Vec<Proj>);

/// 局部移动的粒度：一个局部变量，加上从它往里的字段链。`fields` 为空表示
/// 整个变量。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PathKey {
    base: usize,
    fields: Vec<String>,
}

impl PathKey {
    fn root(base: usize) -> PathKey {
        PathKey { base, fields: Vec::new() }
    }
}

pub struct BorrowChecker;

// ============================================================
// Polonius 风格类型
// ============================================================

/// 区域变量（Polonius 术语）。语义：一组程序点；"这个引用在这些点必须
/// 有效"。跟 SSA 值一对一分配——每个 SSA 值有且仅有一个 origin。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OriginId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(dead_code)]
pub struct LoanId(pub u32);

/// 程序点。`stmt == block.stmts.len()` 表示终结器位置。
///
/// 修复五：一律通过 stmt_point / terminator 构造器创建，stmt 越界
/// （> stmts.len()）在 debug 构建下会直接断言失败，不再静默查不到
/// 活跃点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramPoint {
    pub block: usize,
    pub stmt: usize,
}

impl ProgramPoint {
    /// 语句位置构造器。`stmts_len` 是所在 block 的语句数，仅用于
    /// 越界断言。
    fn stmt_point(block: usize, stmt: usize, stmts_len: usize) -> Self {
        debug_assert!(
            stmt <= stmts_len,
            "ProgramPoint.stmt {} out of range for block {} (stmts.len() = {})",
            stmt,
            block,
            stmts_len
        );
        Self { block, stmt }
    }

    fn terminator(block: usize, stmts_len: usize) -> Self {
        Self { block, stmt: stmts_len }
    }
}

/// 子集约束 `superset ⊇ subset`——`subset` 里出现的所有点，`superset`
/// 也必须活跃。这是 Polonius 里唯一一种需要不动点迭代的约束形式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsetConstraint {
    pub superset: OriginId,
    pub subset: OriginId,
}

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

/// 一次借用。`origin` 决定它什么时候活跃——不再是"借用的 borrow_var
/// 什么时候活跃"这种间接关系，而是 origin 直接给出点集合。
#[derive(Debug, Clone)]
struct Loan {
    #[allow(dead_code)]
    id: LoanId,
    origin: OriginId,
    /// 持有这个借用的引用变量（SSA 值，含版本）。
    borrow_var: SsaLocal,
    /// 被借用的目标 place（保留完整路径，做字段级冲突判定用）。
    place: MirPlace,
    kind: LoanKind,
    /// 这条 loan 在哪条语句被创建。用于在创建语句自身的检查里排除掉
    /// "刚刚创建的这条 loan"——否则 `&mut x` 会跟自己撞上。
    created_at: ProgramPoint,
    /// 是不是两阶段借用的候选：可变借用，并且持有它的引用变量是编译器
    /// 生成的临时（MirLocal.name 为 None）——也就是方法调用自动取的
    /// `&mut recv`。用户自己写的 `let r = &mut v;` 绑定了名字，不是
    /// 两阶段借用，`let r = &mut v; v.len(); r.push(1);` 必须报错。
    /// 候选不等于一定有预留阶段：还要真的找到一个方法调用把它当 receiver
    /// 用掉（见 collect_two_phase）。
    two_phase: bool,
}

/// 某个程序点上活跃的一条 loan。`reserved` 表示这条 loan 在这个点上处于
/// 两阶段借用的预留阶段：行为上等同共享借用——允许读取、允许再取共享
/// 借用，写入/移动/再取可变借用仍然冲突。
#[derive(Debug, Clone, Copy)]
struct Active {
    idx: usize,
    reserved: bool,
}

/// 一个引用变量可能指向哪些 place。`unknown == true` 表示除了 `targets`
/// 里列出的，它还可能指向别处（比如汇合进来的某条路径上是函数参数、调用
/// 返回值），flatten 对这一部分退回保守展开。
#[derive(Debug, Clone, Default)]
struct PtSet {
    targets: Vec<MirPlace>,
    unknown: bool,
}

/// 引用变量的两张流不敏感事实表：
///   pt      —— 引用变量 -> 可能指向的 place 集合（给 flatten 展开 `*r`）
///   carried —— 引用变量 -> 它可能携带的 loan 下标（给授权/两阶段借用判定）
/// pt 沿别名（`dst = src`）和 Phi 传递；carried 在此之上再沿重借出
/// （`x = &mut *r`，x 继承 r 携带的 loan）传递。都是求不动点，不依赖
/// 语句顺序。
#[derive(Debug, Default)]
struct RefFacts {
    pt: HashMap<SsaLocal, PtSet>,
    carried: HashMap<SsaLocal, BTreeSet<usize>>,
}

/// 两阶段借用的结果：
///   activating_at —— 程序点 -> 在这个点被"激活"的 (loan 下标, receiver
///                    引用变量)，也就是方法调用真正使用 `&mut` receiver 的地方
///   reserved      —— loan 下标 -> 它的预留阶段包含的程序点（从创建之后、
///                    沿任意路径一直到激活点之前）
#[derive(Debug, Default)]
struct TwoPhaseInfo {
    activating_at: HashMap<ProgramPoint, Vec<(usize, SsaLocal)>>,
    reserved: HashMap<usize, BTreeSet<ProgramPoint>>,
}

impl TwoPhaseInfo {
    fn is_reserved(&self, idx: usize, p: ProgramPoint) -> bool {
        match self.reserved.get(&idx) {
            Some(points) => points.contains(&p),
            None => false,
        }
    }
}

/// 冲突检查的只读上下文。以前每个 check_* 都带着 f / active / ref_target
/// 三四个参数一路往下传，再加 loans、points-to、授权信息就太多了，收进
/// 一个结构体。
struct BorrowCtx<'a> {
    f: &'a MirFn,
    loans: &'a [Loan],
    facts: &'a RefFacts,
}

// ============================================================
// Origin 收集器
// ============================================================

struct OriginCollector {
    next_origin: u32,
    next_loan: u32,
    /// SSA 值 -> origin。跟 SSA 值一对一。
    ssa_origin: HashMap<SsaLocal, OriginId>,
    /// origin 的活跃点集合种子。求解前只包含 liveness 直接给出的点；
    /// 求解时子集约束会把点沿 superset 方向传播。
    seeds: BTreeMap<OriginId, BTreeSet<ProgramPoint>>,
    constraints: Vec<SubsetConstraint>,
    loans: Vec<Loan>,
}

impl OriginCollector {
    fn new() -> Self {
        Self {
            next_origin: 0,
            next_loan: 0,
            ssa_origin: HashMap::new(),
            seeds: BTreeMap::new(),
            constraints: Vec::new(),
            loans: Vec::new(),
        }
    }

    fn origin_for(&mut self, ssa: SsaLocal) -> OriginId {
        if let Some(&o) = self.ssa_origin.get(&ssa) {
            return o;
        }
        let o = OriginId(self.next_origin);
        self.next_origin += 1;
        self.ssa_origin.insert(ssa, o);
        o
    }

    /// 修复四：批量喂入活跃点。调用方先把 (SSA -> 点列表) 聚合好，
    /// 这里每个 SSA 只做一次 origin_for 哈希查找 + 一次 seeds.entry
    /// B 树查找，点集合用 extend 一次性并入。
    fn mark_live_batch(&mut self, live_points: HashMap<SsaLocal, Vec<ProgramPoint>>) {
        for (ssa, points) in live_points {
            let o = self.origin_for(ssa);
            self.seeds.entry(o).or_default().extend(points);
        }
    }

    fn add_loan(
        &mut self,
        borrow_var: SsaLocal,
        place: MirPlace,
        kind: LoanKind,
        at: ProgramPoint,
        two_phase: bool,
    ) {
        let origin = self.origin_for(borrow_var);
        let id = LoanId(self.next_loan);
        self.next_loan += 1;
        self.loans.push(Loan { id, origin, borrow_var, place, kind, created_at: at, two_phase });
    }

    /// `dst = src`（引用复制/移动）——dst 在哪个点活跃，src 的 origin
    /// 也必须在那个点活跃，因为 dst 在那些点解引用得到的 target 就是
    /// src 的 target。
    fn add_alias(&mut self, src: SsaLocal, dst: SsaLocal) {
        let so = self.origin_for(src);
        let do_ = self.origin_for(dst);
        self.constraints.push(SubsetConstraint { superset: so, subset: do_ });
    }

    /// `phi` 的一个入边：汇合点的输出 origin 活跃的地方，每个入边的
    /// origin 也必须活跃——因为输出的值可能是任何一个入边来的。
    fn add_phi_input(&mut self, input: SsaLocal, output: SsaLocal) {
        let io = self.origin_for(input);
        let oo = self.origin_for(output);
        self.constraints.push(SubsetConstraint { superset: io, subset: oo });
    }

    fn finish(self) -> OriginSolution {
        OriginSolution {
            loans: self.loans,
            constraints: self.constraints,
            seeds: self.seeds,
        }
    }
}

struct OriginSolution {
    loans: Vec<Loan>,
    constraints: Vec<SubsetConstraint>,
    seeds: BTreeMap<OriginId, BTreeSet<ProgramPoint>>,
}

impl OriginSolution {
    /// 不动点求解：反复执行 `points[superset] ⊇ points[subset]`，直到
    /// 稳定。复杂度 O(C · P)，C 约束数、P 点数；对本项目的函数规模
    /// 足够。真要上规模可以换成 worklist 半朴素迭代，接口不变。
    fn solve(&self) -> BTreeMap<OriginId, BTreeSet<ProgramPoint>> {
        let mut points = self.seeds.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for c in &self.constraints {
                let sub_points: Vec<ProgramPoint> = points
                    .get(&c.subset)
                    .map(|s| s.iter().copied().collect())
                    .unwrap_or_default();
                let entry = points.entry(c.superset).or_default();
                for p in sub_points {
                    if entry.insert(p) {
                        changed = true;
                    }
                }
            }
        }
        points
    }

    /// 投影成 Polonius 的 `loan_live_at`：程序点 -> 该点上活跃的 loan
    /// 索引（指向 self.loans）。这一步是纯查表，O(活跃 loan 总数)，
    /// 借用检查主体不再每次从头遍历 loans。
    fn loan_live_at(
        &self,
        origin_points: &BTreeMap<OriginId, BTreeSet<ProgramPoint>>,
    ) -> BTreeMap<ProgramPoint, Vec<usize>> {
        let mut m: BTreeMap<ProgramPoint, Vec<usize>> = BTreeMap::new();
        for (idx, loan) in self.loans.iter().enumerate() {
            if let Some(pts) = origin_points.get(&loan.origin) {
                for &p in pts {
                    m.entry(p).or_default().push(idx);
                }
            }
        }
        m
    }
}

// ============================================================
// 入口
// ============================================================

impl BorrowChecker {
    pub fn check(program: &MirProgram) -> Result<(), String> {
        // 局部移动要知道结构体字段的类型（`Move(p.x)` 移动的是 x，不是 p）：
        // 一次性建好名字 -> 定义的表，传给每个函数。
        let mut structs: HashMap<&str, &MirStruct> = HashMap::new();
        for s in &program.structs {
            structs.insert(s.name.as_str(), s);
        }
        for f in &program.fns {
            Self::check_func_status(f, &structs)?;
        }
        Ok(())
    }

    /// 修复三：原 check_fn 更名，避免与 semantic 的文件重名。
    fn check_func_status(f: &MirFn, structs: &HashMap<&str, &MirStruct>) -> Result<(), String> {
        if f.body.blocks.is_empty() {
            return Ok(());
        }

        // 0. 修复一：Placeholder 早退。统一在入口扫描所有 block 的
        //    terminator——只要还有 Placeholder，说明 mir_builder 有
        //    路径没设真正的终止器，属于内部错误，直接报出来。后面
        //    init_check 的 term_uses 和 borrow_check 的 check_terminator
        //    都不会再碰到它，两处策略不再不一致。
        for (bi, block) in f.body.blocks.iter().enumerate() {
            match &block.terminator {
                MirTerminator::Placeholder => {
                    return Err(format!(
                        "internal error: block {} terminator is still Placeholder at borrow-check time \
                         (mir_builder.rs 某条路径构建完没有给这个块设置真正的终止器)",
                        bi
                    ));
                }
                MirTerminator::Goto(_)
                | MirTerminator::If { .. }
                | MirTerminator::Switch { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable => {}
            }
        }

        // 前驱/后继表只建一次，liveness、init_check、两阶段借用共用——
        // 它们原来各自扫一遍终止器，逻辑重复，新增终止器变体时要同步改
        // 好几处。MirBody::edge_tables 是全项目唯一做这件事的地方。
        let (succs, preds) = f.body.edge_tables();

        // 修复三：两个 pass 都跑完，错误合并后一次性上报——
        // 一个函数同时有 uninit 和 borrow 冲突时，用户能同时看到。
        let mut errors: Vec<String> = Vec::new();

        // 1. 初始化检查：独立的前向数据流，跟借用检查解耦。
        if let Err(e) = Self::init_check(f, &succs, structs) {
            errors.push(e);
        }

        // 2. SSA 粒度 liveness 分析：用 analyze.rs 里统一的那一份（以前这里
        //    自己有一份，跟死赋值消除用的"已使用集合"各写各的）。
        let Liveness { live_after, live_out } = Liveness::build_from_preds(&f.body, &preds);

        // 3. 收集 origin / loan / 子集约束。
        let solution = Self::collect_origins(f, &live_after, &live_out);

        // 3.5 引用变量的 points-to / 携带的 loan（流不敏感不动点）。
        let facts = Self::collect_ref_facts(f, &solution.loans);

        // 4. 不动点求解每个 origin 的点集合（= Polonius 的
        //    origin_live_on_entry）。
        let origin_points = solution.solve();

        // 5. 投影成 loan_live_at。
        let loan_live = solution.loan_live_at(&origin_points);

        // 5.5 两阶段借用：激活点 + 每条候选 loan 的预留阶段。
        let two_phase = Self::collect_two_phase(f, &solution.loans, &facts, &succs);

        // 6. 借用冲突检查：只查表。
        let ctx = BorrowCtx { f, loans: &solution.loans, facts: &facts };
        if let Err(e) = Self::borrow_check(&ctx, &loan_live, &two_phase) {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    // ============================================================
    // Origin / Loan / 约束收集
    // ============================================================

    fn collect_origins(
        f: &MirFn,
        live_after: &[Vec<BTreeSet<SsaLocal>>],
        live_out: &[BTreeSet<SsaLocal>],
    ) -> OriginSolution {
        let mut c = OriginCollector::new();

        // 1. 把 liveness 结果喂进 origin。修复四：先聚合成
        //    (SSA -> 活跃点列表)，再一次性批量喂入——每个 SSA 只做
        //    一次 origin_for + 一次 seeds.entry。
        let mut live_points: HashMap<SsaLocal, Vec<ProgramPoint>> = HashMap::new();
        for (bi, block) in f.body.blocks.iter().enumerate() {
            let len = block.stmts.len();
            for j in 0..len {
                // 修复五：统一走构造器，debug 下断言 j <= len。
                let p = ProgramPoint::stmt_point(bi, j, len);
                for ssa in &live_after[bi][j] {
                    live_points.entry(*ssa).or_default().push(p);
                }
            }
            // 终结器点：块出口活跃集合 + 终结器自身读到的 SSA 值。
            let tp = ProgramPoint::terminator(bi, len);
            for ssa in &live_out[bi] {
                live_points.entry(*ssa).or_default().push(tp);
            }
            for ssa in term_ssa_uses(&block.terminator) {
                live_points.entry(ssa).or_default().push(tp);
            }
        }
        c.mark_live_batch(live_points);

        // 2. 收集 loan、引用别名、Phi 约束。引用变量指向哪里、携带哪些
        //    loan 不在这里算（语句顺序和块顺序不等于支配顺序，循环头的
        //    Phi 会引用后面块里才定义的值），统一交给 collect_ref_facts
        //    做不动点。
        for (bi, block) in f.body.blocks.iter().enumerate() {
            let len = block.stmts.len();
            for (j, stmt) in block.stmts.iter().enumerate() {
                let p = ProgramPoint::stmt_point(bi, j, len);

                // Loan 创建：`dest = &place` / `dest = &mut place`
                if let MirStmt::Assign {
                    dest: MirPlace::Ssa(dest_ssa),
                    value: MirRvalue::Ref { mutable, place },
                } = stmt
                {
                    let kind = if *mutable { LoanKind::Mut } else { LoanKind::Shared };
                    // 两阶段借用的候选：可变借用 + 持有者是编译器临时（没有
                    // 名字）。见 Loan.two_phase 的说明。
                    let two_phase = kind == LoanKind::Mut && Self::local_is_temp(f, dest_ssa.base_id);
                    c.add_loan(*dest_ssa, place.clone(), kind, p, two_phase);
                }

                // 引用别名：`dst = src`（Move/Copy 一个 SSA 引用）
                if let MirStmt::Assign {
                    dest: MirPlace::Ssa(dst),
                    value: MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(src)))
                         | MirRvalue::Use(MirOperand::Copy(MirPlace::Ssa(src))),
                } = stmt
                {
                    c.add_alias(*src, *dst);
                }

                // Phi：`dst = phi(..., src_i, ...)`
                if let MirStmt::Assign {
                    dest: MirPlace::Ssa(dst),
                    value: MirRvalue::Phi { values },
                } = stmt
                {
                    for (_, op) in values {
                        match op {
                            MirOperand::Move(MirPlace::Ssa(src))
                            | MirOperand::Copy(MirPlace::Ssa(src)) => {
                                c.add_phi_input(*src, *dst);
                            }
                            MirOperand::Move(_)
                            | MirOperand::Copy(_)
                            | MirOperand::Constant(_)
                            | MirOperand::Sym(_)
                            | MirOperand::Static(_) => {}
                        }
                    }
                }
            }
        }

        c.finish()
    }

    // ============================================================
    // 引用变量的 points-to 与携带的 loan（流不敏感不动点）
    // ============================================================
    //
    // 关键设计：两张表一起算、但传播规则不同——
    //   pt      —— 引用变量可能指向哪些 place，给 flatten 展开 `*r` 用；
    //              沿别名（`dst = src`）和 Phi 传递；
    //   carried —— 引用变量可能携带哪些 loan，给"访问授权"和两阶段借用的
    //              激活点判定用；沿别名、Phi、重借出三种关系传递。
    //
    // 为什么 carried 要多一条"重借出"：`x = &mut *r` 创建了一条新 loan（x 自己
    // 的），但 x 是从 r 派生出来的——通过 x 访问，同时也是在 r 授权的范围里
    // 访问。只有 x 自己那条的话，`x = &mut *r; *x = 5; use(r);` 里 r 的
    // loan 在 x 死了之后仍然活跃（r 后面还要用），写 `*x` 展开后恰好落在
    // r 借来的那个位置上，会被当成冲突误报。pt 没有这个问题：`*x` 展开成
    // r 的 target 本来就是对的，不需要继承。两张表的规则因此拆开，不能再用
    // 同一个函数传播。
    //
    // 为什么要不动点而不是收集 loan 时顺手记：语句在块里的顺序、块在函数里
    // 的编号都不等于支配顺序——循环头的 Phi 会引用循环体里（编号更大的块）
    // 才定义的值，按顺序扫一遍会漏掉这些来源。旧的 ref_target 就是按顺序
    // 记的，而且 Phi 汇合处干脆没有条目。
    //
    // 来源只有三种：`dst = &place`（Ref，给出一个具体的 target 和一条 loan；
    // place 的访问路径上如果有 `Deref(Ssa(r))`，同时是 r 的重借出）、
    // `dst = src`（别名，整张表照搬）、`dst = phi(...)`（汇合，取并集）。
    // 其它来源（函数参数、调用返回值、从结构体字段读出来的引用）的 target 未知：
    // 汇合到 dst 时给它打上 unknown 标记，flatten 对这一部分退回保守展开。
    //
    // 单调性：targets 只增、unknown 只会变成 true、carried 只增，而可能出现
    // 的 target 只来自程序里的 Ref 语句，是有限的，所以一定收敛。
    //
    // 关于"源还没算出来"和"源根本算不出来"的区别：源 SSA 如果是某条
    // Ref/别名/Phi 的目标（producible），暂时没有条目只表示这一轮还没轮到，
    // 跳过、等下一轮；如果它根本不是这三种语句的目标（函数参数等），才算
    // 未知。不区分的话，循环里的 Phi 会因为"回边那一路的值还没算出来"被过早
    // 地标成 unknown，而且这个标记再也不会消失。
    fn collect_ref_facts(f: &MirFn, loans: &[Loan]) -> RefFacts {
        let mut facts = RefFacts::default();
        let mut producible: HashSet<SsaLocal> = HashSet::new();
        let mut aliases: Vec<(SsaLocal, SsaLocal)> = Vec::new();
        let mut phis: Vec<(SsaLocal, Vec<Option<SsaLocal>>)> = Vec::new();
        // 重借出：(新引用, 它从哪个引用派生)。
        let mut inherited: Vec<(SsaLocal, SsaLocal)> = Vec::new();

        for block in &f.body.blocks {
            for stmt in &block.stmts {
                let (dst, value) = match stmt {
                    MirStmt::Assign { dest: MirPlace::Ssa(dst), value } => (*dst, value),
                    MirStmt::Assign { .. }
                    | MirStmt::ExprStmt(_)
                    | MirStmt::Drop { .. }
                    | MirStmt::SetMetadata { .. }
                    | MirStmt::EffectCheck { .. } => continue,
                };
                match value {
                    MirRvalue::Ref { place, .. } => {
                        producible.insert(dst);
                        let entry = facts.pt.entry(dst).or_default();
                        if !entry.targets.contains(place) {
                            entry.targets.push(place.clone());
                        }
                        // `&mut *r` / `&(*r).f`：被借的 place 经过解引用 r，
                        // 新引用是 r 的重借出，要继承 r 携带的 loan。
                        if let Some(parent) = authorized_borrow_var(place) {
                            inherited.push((dst, parent));
                        }
                    }
                    MirRvalue::Use(MirOperand::Move(MirPlace::Ssa(src)))
                    | MirRvalue::Use(MirOperand::Copy(MirPlace::Ssa(src))) => {
                        producible.insert(dst);
                        aliases.push((dst, *src));
                    }
                    MirRvalue::Phi { values } => {
                        producible.insert(dst);
                        let mut incoming: Vec<Option<SsaLocal>> = Vec::with_capacity(values.len());
                        for (_, op) in values {
                            match op {
                                MirOperand::Move(MirPlace::Ssa(src))
                                | MirOperand::Copy(MirPlace::Ssa(src)) => incoming.push(Some(*src)),
                                // 常量、字段读出来的值之类：不是我们追踪的引用来源。
                                MirOperand::Move(_)
                                | MirOperand::Copy(_)
                                | MirOperand::Constant(_)
                                | MirOperand::Sym(_)
                                | MirOperand::Static(_) => incoming.push(None),
                            }
                        }
                        phis.push((dst, incoming));
                    }
                    MirRvalue::Use(_)
                    | MirRvalue::BinaryOp(_, _, _)
                    | MirRvalue::UnaryOp(_, _)
                    | MirRvalue::Cast(_, _)
                    | MirRvalue::Call { .. }
                    | MirRvalue::MethodCall { .. }
                    | MirRvalue::StructInit { .. }
                    | MirRvalue::EnumVariantConstruction { .. }
                    | MirRvalue::ArrayLiteral(_)
                    | MirRvalue::Discriminant { .. } => {}
                }
            }
        }

        // 每条 loan 的持有者一开始就携带它自己这条 loan。
        for (idx, loan) in loans.iter().enumerate() {
            facts.carried.entry(loan.borrow_var).or_default().insert(idx);
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (dst, src) in &aliases {
                let pt_changed = Self::flow_pt(&mut facts, &producible, *dst, Some(*src));
                let carried_changed = Self::flow_carried(&mut facts, *dst, Some(*src));
                if pt_changed || carried_changed {
                    changed = true;
                }
            }
            for (dst, incoming) in &phis {
                for src in incoming {
                    let pt_changed = Self::flow_pt(&mut facts, &producible, *dst, *src);
                    let carried_changed = Self::flow_carried(&mut facts, *dst, *src);
                    if pt_changed || carried_changed {
                        changed = true;
                    }
                }
            }
            // 重借出只传 carried，不传 pt：`*x` 的展开靠 x 自己的 target
            // （Ref 语句里已经记了），不需要把父引用的 target 并进来。
            for (dst, parent) in &inherited {
                if Self::flow_carried(&mut facts, *dst, Some(*parent)) {
                    changed = true;
                }
            }
        }
        facts
    }

    /// 把 `src` 的 points-to 并进 `dst`。返回 dst 的 pt 是否发生了变化。
    /// `src` 是 None 表示这一路来源不是 SSA 引用变量，target 未知。
    fn flow_pt(
        facts: &mut RefFacts,
        producible: &HashSet<SsaLocal>,
        dst: SsaLocal,
        src: Option<SsaLocal>,
    ) -> bool {
        let src_set: Option<PtSet> = match src {
            Some(s) => facts.pt.get(&s).cloned(),
            None => None,
        };
        // 源既没有条目、又不是任何 Ref/别名/Phi 的目标：这一轮没轮到的
        // 情形不算（producible），根本算不出来的才是"未知"。
        let src_unknown_forever = match src {
            Some(s) => src_set.is_none() && !producible.contains(&s),
            None => true,
        };

        let existed = facts.pt.contains_key(&dst);
        let entry = facts.pt.entry(dst).or_default();
        let mut changed = !existed;
        if let Some(set) = src_set {
            for t in set.targets {
                if !entry.targets.contains(&t) {
                    entry.targets.push(t);
                    changed = true;
                }
            }
            if set.unknown && !entry.unknown {
                entry.unknown = true;
                changed = true;
            }
        } else if src_unknown_forever && !entry.unknown {
            entry.unknown = true;
            changed = true;
        }
        changed
    }

    /// 把 `src` 携带的 loan 并进 `dst`。返回 dst 的 carried 是否发生了变化。
    /// `src` 是 None（来源不是 SSA 引用变量）或者 src 还没有携带任何 loan 时
    /// 什么都不做——"未知"只对 pt 有意义，一个携带不了 loan 的来源不会授权
    /// 任何访问。
    fn flow_carried(facts: &mut RefFacts, dst: SsaLocal, src: Option<SsaLocal>) -> bool {
        let src_loans: BTreeSet<usize> = match src {
            Some(s) => match facts.carried.get(&s) {
                Some(set) => set.clone(),
                None => return false,
            },
            None => return false,
        };
        let dst_loans = facts.carried.entry(dst).or_default();
        let mut changed = false;
        for loan_idx in src_loans {
            if dst_loans.insert(loan_idx) {
                changed = true;
            }
        }
        changed
    }

    /// 持有者是不是编译器生成的临时（没有用户名字）。
    fn local_is_temp(f: &MirFn, base_id: usize) -> bool {
        for l in &f.body.locals {
            if l.id == base_id {
                return l.name.is_none();
            }
        }
        false
    }

    // ============================================================
    // 两阶段借用（two-phase borrow）
    // ============================================================
    //
    // 问题：`v.push(v.len())` 里，方法调用的 receiver 是自动取的 `&mut v`，
    // 求值顺序是"先取 &mut v，再算实参 v.len()，最后调用"。按普通借用规则，
    // `&mut v` 活跃期间读 v（v.len()）就是冲突——可这是完全合理的程序。
    //
    // 两阶段借用把可变借用分成两个阶段：
    //   预留（reservation）：从创建到真正被调用之前，行为等同共享借用——
    //       允许读取，允许再取共享借用，但仍然不允许写入、移动、再取可变借用；
    //   激活（activation）：receiver 被方法调用真正使用的那一刻，这时它
    //       才是独占的，要检查此刻有没有别的活跃借用跟它冲突。
    //
    // 候选条件（Loan.two_phase）：可变借用，且持有者是编译器生成的临时。
    // 用户自己写 `let r = &mut v;` 绑定了名字，不享受预留阶段——Rust 里也是
    // 这样，`let r = &mut v; v.len(); r.push(1);` 是错误。
    //
    // 激活点：MethodCall 的 receiver 是一个引用变量，而候选 loan 在它携带的
    // loan 集合里。receiver 在 MIR 里是 `Move(Ssa(t))`（构建器对所有标识符
    // 读取都发 Move，不是 Copy），所以 Move 和 Copy 两种都认。普通 Call 的
    // 实参不是激活点：函数调用里显式写的 `&mut v` 不是自动引用，`f(&mut v,
    // v.len())` 在 Rust 里报错，这里也报错。
    //
    // 预留阶段的范围：从创建语句之后开始、沿任意路径走到激活点之前经过的
    // 所有程序点（receiver 之后还有别的实参里含控制流时，会跨好几个块）。
    // 不能用 ProgramPoint 的大小比较来判断"在激活之前"——块编号跟控制流
    // 顺序没有关系，必须沿着 CFG 走。
    fn collect_two_phase(
        f: &MirFn,
        loans: &[Loan],
        facts: &RefFacts,
        succs: &[Vec<usize>],
    ) -> TwoPhaseInfo {
        let mut info = TwoPhaseInfo::default();
        let mut act_points: HashMap<usize, HashSet<ProgramPoint>> = HashMap::new();

        for (bi, block) in f.body.blocks.iter().enumerate() {
            let len = block.stmts.len();
            for (j, stmt) in block.stmts.iter().enumerate() {
                let value = match stmt {
                    MirStmt::Assign { value, .. } => value,
                    MirStmt::ExprStmt(value) => value,
                    MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => continue,
                };
                let receiver = match value {
                    MirRvalue::MethodCall { receiver, .. } => receiver,
                    MirRvalue::Use(_)
                    | MirRvalue::BinaryOp(_, _, _)
                    | MirRvalue::UnaryOp(_, _)
                    | MirRvalue::Cast(_, _)
                    | MirRvalue::Call { .. }
                    | MirRvalue::StructInit { .. }
                    | MirRvalue::EnumVariantConstruction { .. }
                    | MirRvalue::Ref { .. }
                    | MirRvalue::ArrayLiteral(_)
                    | MirRvalue::Discriminant { .. }
                    | MirRvalue::Phi { .. } => continue,
                };
                let r = match receiver {
                    MirOperand::Move(MirPlace::Ssa(r)) | MirOperand::Copy(MirPlace::Ssa(r)) => *r,
                    MirOperand::Move(_)
                    | MirOperand::Copy(_)
                    | MirOperand::Constant(_)
                    | MirOperand::Sym(_)
                    | MirOperand::Static(_) => continue,
                };
                let carried = match facts.carried.get(&r) {
                    Some(set) => set,
                    None => continue,
                };
                let p = ProgramPoint::stmt_point(bi, j, len);
                for &idx in carried {
                    if loans[idx].two_phase {
                        info.activating_at.entry(p).or_default().push((idx, r));
                        act_points.entry(idx).or_default().insert(p);
                    }
                }
            }
        }

        // 按 loan 下标升序处理，结果不依赖哈希迭代顺序。
        let mut idxs: Vec<usize> = act_points.keys().copied().collect();
        idxs.sort_unstable();
        for idx in idxs {
            let acts = match act_points.get(&idx) {
                Some(a) => a,
                None => continue,
            };
            let created = loans[idx].created_at;
            let mut region: BTreeSet<ProgramPoint> = BTreeSet::new();
            // 以"从语句 0 进入"的方式走进来过的块，每个块只展开一次。
            let mut entered: HashSet<usize> = HashSet::new();
            let mut work: Vec<(usize, usize)> = Vec::new();
            work.push((created.block, created.stmt + 1));
            while let Some((b, start)) = work.pop() {
                let len = f.body.blocks[b].stmts.len();
                let mut reached_end = true;
                let mut k = start;
                while k <= len {
                    let p = ProgramPoint::stmt_point(b, k, len);
                    if acts.contains(&p) {
                        // 走到激活点就停：这条路径上激活点之后不再是预留阶段。
                        reached_end = false;
                        break;
                    }
                    region.insert(p);
                    k += 1;
                }
                if reached_end {
                    for &next in &succs[b] {
                        if entered.insert(next) {
                            work.push((next, 0));
                        }
                    }
                }
            }
            info.reserved.insert(idx, region);
        }
        info
    }

    // ============================================================
    // 初始化检查（前向 may，独立 pass）
    // ============================================================
    //
    // 关键重构：状态从"一定已初始化的变量集合"（must，合并取交集）改成
    // "可能未初始化的路径集合"（may，合并取并集）——两者互为补集，但只有
    // 后者能自然地表达局部移动：路径可以是整个变量 `p`，也可以是字段链
    // `p.a`、`p.a.b`。
    //
    // 规则：
    //   - 移动 `p.a`（该字段类型不是 Copy）：把路径 p.a 加进集合。之后：
    //       读 p.b → 没问题（路径不重叠）；读 p.a / p.a.x / p / 把 p 整个
    //       传出去 → 报错（p.a 是 p.a.x 的祖先，也是 p 的后代，都算重叠）；
    //       给 p.a 重新赋值 → 恢复（删掉 p.a 及其后代）；
    //       给 p 整个重新赋值 → 同样恢复。
    //   - 给 p.a.b 赋值而它的某个祖先（p、p.a）整体被移走了 → 报错：往一个
    //     已经被移空的值里面写字段没有意义。祖先只是"部分被移走"
    //     （比如 p.c 被移走）不算——p.a.b 本身还在。
    //   - 只有 Ssa 和 Field 链（`p.a.b`）上的移动/赋值才改变状态。经过 Index、
    //     Deref、EnumPayload 的移动不是对基址的移动（元素/解引用目标不能被
    //     "搬走"基址），经过它们的赋值也不能让任何东西变成已初始化——这两种
    //     读取在 use 检查里仍然保守地当作读基址，但不进入状态更新。
    //   - 只检查从入口可达的块。从入口走不到的块（比如 return 之后的死代码）
    //     里的"未初始化"没有意义，旧版把它们的输入状态当成空集，会在死代码
    //     里误报。
    //
    // 单调性与收敛：每个块的输入状态只增不减（并集），转移函数对输入状态
    // 单调，所以不动点迭代一定收敛，而且迭代中途发现的错误在最终状态里
    // 同样成立（状态只会变大），可以立刻返回。
    fn init_check(
        f: &MirFn,
        succs: &[Vec<usize>],
        structs: &HashMap<&str, &MirStruct>,
    ) -> Result<(), String> {
        let body = &f.body;
        let n = body.blocks.len();
        let mut in_uninit: Vec<BTreeSet<PathKey>> = vec![BTreeSet::new(); n];
        let mut reached: Vec<bool> = vec![false; n];

        // 入口：所有非参数的局部变量一开始都是未初始化的；参数由调用者给值。
        for local in &body.locals {
            if !local.is_param {
                in_uninit[0].insert(PathKey::root(local.id));
            }
        }
        reached[0] = true;

        let mut changed = true;
        while changed {
            changed = false;
            for block_id in 0..n {
                if !reached[block_id] {
                    continue;
                }
                let block = &body.blocks[block_id];
                let mut state = in_uninit[block_id].clone();
                for stmt in &block.stmts {
                    Self::exec_init_stmt(stmt, &mut state, f, structs)?;
                }
                let mut term_uses: Vec<PathKey> = Vec::new();
                Self::term_use_paths(&block.terminator, &mut term_uses);
                for path in &term_uses {
                    if Self::is_uninit(&state, path) {
                        return Err(format!(
                            "use of possibly-uninitialized variable `{}`",
                            Self::describe_path(f, path)
                        ));
                    }
                }

                for &succ in &succs[block_id] {
                    if !reached[succ] {
                        reached[succ] = true;
                        changed = true;
                    }
                    let before = in_uninit[succ].len();
                    in_uninit[succ].extend(state.iter().cloned());
                    if in_uninit[succ].len() != before {
                        changed = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// 在 state（可能未初始化的路径集合）上执行一条语句：先检查读，再检查
    /// 赋值目标，再记录移动，最后处理定义。顺序跟旧版一致——同一条语句里的
    /// 移动不影响它自己的读检查（`x = f(Move(x))` 里读 x 发生在移动之前）。
    fn exec_init_stmt(
        stmt: &MirStmt,
        state: &mut BTreeSet<PathKey>,
        f: &MirFn,
        structs: &HashMap<&str, &MirStruct>,
    ) -> Result<(), String> {
        let mut uses: Vec<PathKey> = Vec::new();
        Self::stmt_use_paths(stmt, &mut uses);
        for path in &uses {
            if Self::is_uninit(state, path) {
                return Err(format!(
                    "use of possibly-uninitialized variable `{}`",
                    Self::describe_path(f, path)
                ));
            }
        }

        if let MirStmt::Assign { dest, .. } = stmt {
            if let Some(path) = Self::field_path(dest) {
                if Self::has_uninit_ancestor(state, &path) {
                    return Err(format!(
                        "assign to part of moved value `{}`",
                        Self::describe_path(f, &path)
                    ));
                }
            }
        }

        let mut moved: Vec<PathKey> = Vec::new();
        Self::stmt_move_paths(stmt, f, structs, &mut moved);
        for path in moved {
            state.insert(path);
        }

        match stmt {
            MirStmt::Assign { dest, .. } => {
                if let Some(path) = Self::field_path(dest) {
                    Self::reinitialize(state, &path);
                }
            }
            MirStmt::Drop { place } => {
                // 对一个变量 Drop 之后，整个变量（包括所有字段）都不能再用。
                if let Some(base) = Self::place_base_id(place) {
                    state.insert(PathKey::root(base));
                }
            }
            MirStmt::ExprStmt(_) | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
        }
        Ok(())
    }

    // ---- 路径与状态的基本操作 ----

    /// 路径 a、b 是否重叠：同一个变量，并且其中一条是另一条的前缀
    /// （`p` 与 `p.a` 重叠，`p.a` 与 `p.b` 不重叠）。
    fn paths_overlap(a: &PathKey, b: &PathKey) -> bool {
        if a.base != b.base {
            return false;
        }
        let common = a.fields.len().min(b.fields.len());
        a.fields[..common] == b.fields[..common]
    }

    fn is_uninit(state: &BTreeSet<PathKey>, path: &PathKey) -> bool {
        state.iter().any(|u| Self::paths_overlap(u, path))
    }

    /// state 里有没有 path 的"严格祖先"路径（比 path 短、是它的前缀）。
    fn has_uninit_ancestor(state: &BTreeSet<PathKey>, path: &PathKey) -> bool {
        state.iter().any(|u| {
            u.base == path.base
                && u.fields.len() < path.fields.len()
                && path.fields[..u.fields.len()] == u.fields[..]
        })
    }

    /// 给 path 赋了新值：path 本身以及它的所有后代都重新变成已初始化。
    fn reinitialize(state: &mut BTreeSet<PathKey>, path: &PathKey) {
        state.retain(|u| {
            !(u.base == path.base
                && u.fields.len() >= path.fields.len()
                && u.fields[..path.fields.len()] == path.fields[..])
        });
    }

    /// 纯粹由 Ssa / Field 构成的 place 对应的路径；只要中间经过 Index、
    /// Deref 或 EnumPayload，就不是一个"字段路径"，返回 None。
    fn field_path(place: &MirPlace) -> Option<PathKey> {
        let mut fields: Vec<String> = Vec::new();
        let mut cur = place;
        loop {
            match cur {
                MirPlace::Ssa(s) => {
                    // 从最外层往里走，字段是倒着收集的，还原成从基址往外的顺序。
                    fields.reverse();
                    return Some(PathKey { base: s.base_id, fields });
                }
                MirPlace::Field { base, field } => {
                    fields.push(field.clone());
                    cur = base;
                }
                MirPlace::Index { .. } | MirPlace::Deref(_) | MirPlace::EnumPayload { .. } => {
                    return None;
                }
            }
        }
    }

    /// 这条路径上的移动是不是真的移动：路径对应的值的类型不是 Copy。
    ///
    /// 整个变量用局部变量自己的类型；字段路径要顺着结构体定义/元组找到
    /// 字段本身的类型——`Move(p.x)` 搬走的是 x，不是 p：p 是非 Copy 的
    /// 结构体、x 是 f32 时，这不是移动。构建器对所有字段读取都发 Move，
    /// 不区分 Copy 与否，所以这里必须解析字段类型，否则任何一次读 f32 字段
    /// 都会被当成移动、第二次读就报错。
    ///
    /// 泛型结构体的实例（`Pair<i32, String>`）会按实参替换字段类型里的类型
    /// 参数再判 Copy。类型仍然解析不出来（经过引用的字段、查不到的结构体、
    /// 泛型实参个数对不上）一律按"不是移动"处理——跟旧版"局部移动完全不记录"
    /// 的行为一致，不会因此多出误报，代价是这几种情形下的局部移动检测不到。
    fn path_move_is_real(path: &PathKey, f: &MirFn, structs: &HashMap<&str, &MirStruct>) -> bool {
        match Self::resolve_path_type(path, f, structs) {
            Some(ty) => !ty.is_copy(),
            None => false,
        }
    }

    fn resolve_path_type(
        path: &PathKey,
        f: &MirFn,
        structs: &HashMap<&str, &MirStruct>,
    ) -> Option<Type> {
        let mut ty = Self::get_var_type(f, path.base);
        for field in &path.fields {
            while let Type::Privacy(inner, _) = ty {
                ty = *inner;
            }
            ty = match ty {
                Type::Struct(name) => {
                    let def = structs.get(name.as_str())?;
                    if !def.generic_params.is_empty() {
                        return None;
                    }
                    let mut found: Option<Type> = None;
                    for (fname, fty) in &def.fields {
                        if fname == field {
                            found = Some(fty.clone());
                            break;
                        }
                    }
                    found?
                }
                // 泛型结构体的实例化（`Pair<i32, String>`）：用定义里的字段类型，
                // 把里面的类型参数替换成实参。原来泛型结构体整个解析不出来，
                // 局部移动（`let a = p.first;`）检测不到，是漏报。
                // 泛型参数个数对不上（畸形输入）、找不到字段，一律解析失败。
                Type::Generic(name, args) => {
                    let def = structs.get(name.as_str())?;
                    if def.generic_params.len() != args.len() {
                        return None;
                    }
                    let mut found: Option<Type> = None;
                    for (fname, fty) in &def.fields {
                        if fname == field {
                            found = Some(Self::substitute_type_params(
                                fty,
                                &def.generic_params,
                                &args,
                            ));
                            break;
                        }
                    }
                    found?
                }
                // 元组字段在 MIR 里用十进制下标当字段名。
                Type::Tuple(elems) => {
                    let idx: usize = field.parse().ok()?;
                    elems.get(idx)?.clone()
                }
                // 这里的通配是有意的：Type 有二十多个变体，除了结构体、泛型结构体
                // 和元组以外都没有"字段"可以往里走，"解析不出来"正是对它们的
                // 正确回答（调用方把 None 当成"不是移动"，保守）。真新增了一种
                // 带字段的聚合类型，退化成"局部移动检测不到"，不会误报。
                _ => return None,
            };
        }
        Some(ty)
    }

    // 把类型里的类型参数按 (参数名 -> 实参) 替换掉。params 和 args 按位置一一对应。
    fn substitute_type_params(ty: &Type, params: &[HirGenericParam], args: &[Type]) -> Type {
        let mut subst: HashMap<&str, &Type> = HashMap::new();
        for (p, a) in params.iter().zip(args.iter()) {
            subst.insert(p.name.as_str(), a);
        }
        Self::substitute_type_params_inner(ty, &subst)
    }

    fn substitute_type_params_inner(ty: &Type, subst: &HashMap<&str, &Type>) -> Type {
        match ty {
            Type::TypeParam(name) => match subst.get(name.as_str()) {
                Some(t) => (*t).clone(),
                None => ty.clone(),
            },
            Type::Generic(name, args) => Type::Generic(
                name.clone(),
                args.iter().map(|a| Self::substitute_type_params_inner(a, subst)).collect(),
            ),
            Type::Tuple(elems) => Type::Tuple(
                elems.iter().map(|t| Self::substitute_type_params_inner(t, subst)).collect(),
            ),
            Type::Array(elem, n) => {
                Type::Array(Box::new(Self::substitute_type_params_inner(elem, subst)), *n)
            }
            Type::Ref { mutable, inner } => Type::Ref {
                mutable: *mutable,
                inner: Box::new(Self::substitute_type_params_inner(inner, subst)),
            },
            Type::Slice(inner) => {
                Type::Slice(Box::new(Self::substitute_type_params_inner(inner, subst)))
            }
            Type::Tensor { dtype, shape } => Type::Tensor {
                dtype: Box::new(Self::substitute_type_params_inner(dtype, subst)),
                shape: shape.clone(),
            },
            Type::Privacy(inner, tag) => Type::Privacy(
                Box::new(Self::substitute_type_params_inner(inner, subst)),
                tag.clone(),
            ),
            // 其余类型（标量、具名 Struct/Enum、SelfType 等）里没有嵌套的类型
            // 参数可替换；新增带子类型的变体时必须在上面补分支。
            _ => ty.clone(),
        }
    }

    // ---- 语句里的读、移动 ----

    fn stmt_use_paths(stmt: &MirStmt, out: &mut Vec<PathKey>) {
        match stmt {
            MirStmt::Assign { dest, value } => {
                Self::rvalue_use_paths(value, out);
                match dest {
                    MirPlace::Ssa(_) => {}
                    // 写 Field 链不是读：它的前置条件（祖先没被整体移走）由
                    // exec_init_stmt 单独检查。其余带投影的目标（p[i] = v、
                    // *p = v、经过 Deref 的字段写）要先读基址。
                    MirPlace::Field { .. } => {
                        if Self::field_path(dest).is_none() {
                            Self::place_use_paths(dest, out);
                        }
                    }
                    MirPlace::Index { .. } | MirPlace::Deref(_) | MirPlace::EnumPayload { .. } => {
                        Self::place_use_paths(dest, out);
                    }
                }
            }
            MirStmt::ExprStmt(value) => Self::rvalue_use_paths(value, out),
            MirStmt::Drop { place } => Self::place_use_paths(place, out),
            MirStmt::SetMetadata { place, .. } => Self::place_use_paths(place, out),
            MirStmt::EffectCheck { .. } => {}
        }
    }

    fn term_use_paths(term: &MirTerminator, out: &mut Vec<PathKey>) {
        match term {
            MirTerminator::If { cond, .. } => Self::operand_use_paths(cond, out),
            MirTerminator::Return(Some(op)) => Self::operand_use_paths(op, out),
            MirTerminator::Switch { discr, .. } => Self::operand_use_paths(discr, out),
            // Placeholder 已在 check_func_status 入口早退，到不了这里。显式
            // 列出每个没有操作数的变体，不用通配：将来 MirTerminator 新增
            // 一个带操作数的变体时，这里会编译失败，而不是静默漏掉它的读
            // （漏掉的后果是未初始化的变量被放过去）。
            MirTerminator::Goto(_)
            | MirTerminator::Unreachable
            | MirTerminator::Return(None)
            | MirTerminator::Placeholder => {}
        }
    }

    /// 读一个 place 用到的路径：纯字段链读的是那条具体的字段路径；经过
    /// Index/Deref/EnumPayload 的读取保守地当作读基址（路径是整个变量），
    /// 下标操作数本身也是读。
    fn place_use_paths(place: &MirPlace, out: &mut Vec<PathKey>) {
        match place {
            MirPlace::Ssa(s) => out.push(PathKey::root(s.base_id)),
            MirPlace::Field { base, .. } => match Self::field_path(place) {
                Some(path) => out.push(path),
                None => Self::place_use_paths(base, out),
            },
            MirPlace::Index { base, index } => {
                Self::place_use_paths(base, out);
                Self::operand_use_paths(index, out);
            }
            MirPlace::Deref(base) => Self::place_use_paths(base, out),
            MirPlace::EnumPayload { base, .. } => Self::place_use_paths(base, out),
        }
    }

    fn operand_use_paths(op: &MirOperand, out: &mut Vec<PathKey>) {
        match op {
            MirOperand::Copy(p) | MirOperand::Move(p) => Self::place_use_paths(p, out),
            MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => {}
        }
    }

    fn rvalue_use_paths(rv: &MirRvalue, out: &mut Vec<PathKey>) {
        match rv {
            MirRvalue::Use(op) => Self::operand_use_paths(op, out),
            MirRvalue::BinaryOp(_, l, r) => {
                Self::operand_use_paths(l, out);
                Self::operand_use_paths(r, out);
            }
            MirRvalue::UnaryOp(_, o) => Self::operand_use_paths(o, out),
            MirRvalue::Cast(o, _) => Self::operand_use_paths(o, out),
            MirRvalue::Call { args, .. } => {
                for a in args {
                    Self::operand_use_paths(a, out);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::operand_use_paths(receiver, out);
                for a in args {
                    Self::operand_use_paths(a, out);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::operand_use_paths(op, out);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for a in args {
                    Self::operand_use_paths(a, out);
                }
            }
            MirRvalue::Ref { place, .. } => Self::place_use_paths(place, out),
            MirRvalue::ArrayLiteral(elems) => {
                for e in elems {
                    Self::operand_use_paths(e, out);
                }
            }
            MirRvalue::Discriminant { value, .. } => Self::operand_use_paths(value, out),
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::operand_use_paths(op, out);
                }
            }
        }
    }

    fn stmt_move_paths(
        stmt: &MirStmt,
        f: &MirFn,
        structs: &HashMap<&str, &MirStruct>,
        out: &mut Vec<PathKey>,
    ) {
        match stmt {
            MirStmt::Assign { value, .. } => Self::rvalue_move_paths(value, f, structs, out),
            MirStmt::ExprStmt(value) => Self::rvalue_move_paths(value, f, structs, out),
            MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
        }
    }

    fn operand_move_path(
        op: &MirOperand,
        f: &MirFn,
        structs: &HashMap<&str, &MirStruct>,
        out: &mut Vec<PathKey>,
    ) {
        match op {
            MirOperand::Move(place) => {
                if let Some(path) = Self::field_path(place) {
                    if Self::path_move_is_real(&path, f, structs) {
                        out.push(path);
                    }
                }
            }
            MirOperand::Copy(_) | MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => {}
        }
    }

    fn rvalue_move_paths(
        rv: &MirRvalue,
        f: &MirFn,
        structs: &HashMap<&str, &MirStruct>,
        out: &mut Vec<PathKey>,
    ) {
        match rv {
            MirRvalue::Use(op) => Self::operand_move_path(op, f, structs, out),
            MirRvalue::BinaryOp(_, l, r) => {
                Self::operand_move_path(l, f, structs, out);
                Self::operand_move_path(r, f, structs, out);
            }
            MirRvalue::UnaryOp(_, o) => Self::operand_move_path(o, f, structs, out),
            MirRvalue::Cast(o, _) => Self::operand_move_path(o, f, structs, out),
            MirRvalue::Call { args, .. } => {
                for a in args {
                    Self::operand_move_path(a, f, structs, out);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::operand_move_path(receiver, f, structs, out);
                for a in args {
                    Self::operand_move_path(a, f, structs, out);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::operand_move_path(op, f, structs, out);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for a in args {
                    Self::operand_move_path(a, f, structs, out);
                }
            }
            MirRvalue::ArrayLiteral(elems) => {
                for e in elems {
                    Self::operand_move_path(e, f, structs, out);
                }
            }
            MirRvalue::Discriminant { value, .. } => Self::operand_move_path(value, f, structs, out),
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::operand_move_path(op, f, structs, out);
                }
            }
            // Ref 借用一个位置，不移动它。
            MirRvalue::Ref { .. } => {}
        }
    }

    // ============================================================
    // 借用冲突检查（查 loan_live_at 表）
    // ============================================================

    fn borrow_check(
        ctx: &BorrowCtx,
        live: &BTreeMap<ProgramPoint, Vec<usize>>,
        two_phase: &TwoPhaseInfo,
    ) -> Result<(), String> {
        for (bi, block) in ctx.f.body.blocks.iter().enumerate() {
            let len = block.stmts.len();
            for (j, stmt) in block.stmts.iter().enumerate() {
                // 修复五：统一走构造器，debug 下断言 j <= len。
                let p = ProgramPoint::stmt_point(bi, j, len);
                let active = Self::active_at(ctx, live, two_phase, p);

                // 方法调用真正使用 `&mut` receiver 的这一刻：两阶段借用在这里
                // 激活，检查此刻有没有别的活跃借用跟它冲突。
                if let Some(acts) = two_phase.activating_at.get(&p) {
                    for (idx, receiver) in acts {
                        Self::check_activation(ctx, *idx, *receiver, p, live)?;
                    }
                }
                Self::check_stmt(ctx, stmt, &active)?;
            }

            let tp = ProgramPoint::terminator(bi, len);
            let active = Self::active_at(ctx, live, two_phase, tp);
            Self::check_terminator(ctx, &block.terminator, &active)?;
        }
        Ok(())
    }

    /// 程序点 p 上活跃的 loan，并标注哪些处于两阶段借用的预留阶段。
    /// 这条语句自己刚创建出的那条 loan 要排除掉，否则 `&mut x` 会跟自己撞上。
    fn active_at(
        ctx: &BorrowCtx,
        live: &BTreeMap<ProgramPoint, Vec<usize>>,
        two_phase: &TwoPhaseInfo,
        p: ProgramPoint,
    ) -> Vec<Active> {
        let mut out: Vec<Active> = Vec::new();
        if let Some(indices) = live.get(&p) {
            for &i in indices {
                if ctx.loans[i].created_at == p {
                    continue;
                }
                out.push(Active { idx: i, reserved: two_phase.is_reserved(i, p) });
            }
        }
        out
    }

    /// 两阶段借用的激活检查：loan `idx` 在 p 处被方法调用真正使用，此刻它
    /// 要独占 `loan.place`——不能还有别的活跃借用指向重叠的位置。
    ///
    /// loan_live_at 记录的是"这条语句执行之后还活跃的 loan"，而对 receiver
    /// 自己和作为实参传进来的引用，这条语句恰恰是它们最后一次使用——它们
    /// 在语句之后已经不活跃，只看 p 处的表会漏掉。所以候选取 p 处和同一个
    /// 块里前一条语句处的并集，近似"语句执行之前活跃的 loan"。（p 是块里的
    /// 第一条语句时拿不到前一个点，只用 p 处的表，会漏掉恰好在这条语句
    /// 里死掉的实参借用——已知的小缺口。）
    ///
    /// receiver 自己携带的 loan（别名、Phi 带过来的）不算冲突源。
    fn check_activation(
        ctx: &BorrowCtx,
        idx: usize,
        receiver: SsaLocal,
        p: ProgramPoint,
        live: &BTreeMap<ProgramPoint, Vec<usize>>,
    ) -> Result<(), String> {
        let loan = &ctx.loans[idx];
        let own = ctx.facts.carried.get(&receiver);

        let mut candidates: BTreeSet<usize> = BTreeSet::new();
        if let Some(v) = live.get(&p) {
            candidates.extend(v.iter().copied());
        }
        if p.stmt > 0 {
            let prev = ProgramPoint { block: p.block, stmt: p.stmt - 1 };
            if let Some(v) = live.get(&prev) {
                candidates.extend(v.iter().copied());
            }
        }

        for m in candidates {
            if m == idx {
                continue;
            }
            if let Some(own) = own {
                if own.contains(&m) {
                    continue;
                }
            }
            let other = &ctx.loans[m];
            if other.created_at == p {
                continue;
            }
            if Self::places_overlap(&loan.place, &other.place, &ctx.facts.pt) {
                return Err(format!(
                    "cannot mutably borrow `{}` because it is already {} borrowed as `{}`",
                    Self::describe_place(&loan.place, ctx.f),
                    other.kind,
                    Self::get_var_name(ctx.f, other.borrow_var.base_id)
                ));
            }
        }
        Ok(())
    }

    fn check_stmt(ctx: &BorrowCtx, stmt: &MirStmt, active: &[Active]) -> Result<(), String> {
        match stmt {
            MirStmt::Assign { dest, value } => {
                let mut copies = Vec::new();
                let mut moves = Vec::new();
                Self::rvalue_places(value, &mut copies, &mut moves);
                for p in &copies {
                    Self::check_read(ctx, p, active)?;
                }
                for p in &moves {
                    Self::check_move(ctx, p, active)?;
                }
                if let MirRvalue::Ref { mutable, place } = value {
                    Self::check_borrow_creation(ctx, place, *mutable, active)?;
                }
                Self::check_write(ctx, dest, active)?;
            }
            MirStmt::ExprStmt(value) => {
                let mut copies = Vec::new();
                let mut moves = Vec::new();
                Self::rvalue_places(value, &mut copies, &mut moves);
                for p in &copies {
                    Self::check_read(ctx, p, active)?;
                }
                for p in &moves {
                    Self::check_move(ctx, p, active)?;
                }
            }
            MirStmt::Drop { place } => {
                Self::check_exclusive(ctx, place, active, "drop")?;
            }
            MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
        }
        Ok(())
    }

    fn check_terminator(ctx: &BorrowCtx, term: &MirTerminator, active: &[Active]) -> Result<(), String> {
        match term {
            MirTerminator::Goto(_) | MirTerminator::Unreachable => Ok(()),
            MirTerminator::If { cond, .. } => Self::check_operand_use(ctx, cond, active),
            MirTerminator::Return(op) => match op {
                Some(op) => Self::check_operand_use(ctx, op, active),
                None => Ok(()),
            },
            MirTerminator::Switch { discr, .. } => Self::check_operand_use(ctx, discr, active),
            // 修复一：Placeholder 已在 check_func_status 入口统一早退，
            // 正常到不了这个分支；保留作防御性兜底，语义与入口一致。
            MirTerminator::Placeholder => Err(
                "internal error: block terminator is still Placeholder at borrow-check time \
                 (mir_builder.rs 某条路径构建完没有给这个块设置真正的终止器)".to_string()
            ),
        }
    }

    fn check_operand_use(ctx: &BorrowCtx, op: &MirOperand, active: &[Active]) -> Result<(), String> {
        match op {
            MirOperand::Copy(p) => Self::check_read(ctx, p, active),
            MirOperand::Move(p) => Self::check_move(ctx, p, active),
            MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => Ok(()),
        }
    }

    /// 修复六：通过引用 s 访问它自己指向的位置时（place 的访问路径上有
    /// `Deref(Ssa(s))`），s 携带的 loan 是这次访问的授权来源，不是冲突源。
    /// 不跳过它，`*r = 5; use(r);` 里 r 的 loan（place = x）会跟写 `*r`
    /// （展开后也是 x）撞上，通过一个借用访问它自己借来的东西被报成冲突。
    ///
    /// 判据是"loan 在不在 s 携带的集合里"，不是"loan.borrow_var == s"：
    /// `r2 = r; *r2 = 5;` 里 r2 本身没有创建任何 loan，只是从 r 那里带来
    /// 了 r 的 loan，只比 borrow_var 会误报。别的、没被 s 携带的活跃 loan
    /// 仍然照常检查——两个不同的借用同时指向同一个位置就是冲突。
    fn is_authorized(ctx: &BorrowCtx, place: &MirPlace, loan_idx: usize) -> bool {
        match authorized_borrow_var(place) {
            Some(s) => match ctx.facts.carried.get(&s) {
                Some(set) => set.contains(&loan_idx),
                None => false,
            },
            None => false,
        }
    }

    /// 读：只跟可变借用冲突；处于预留阶段的两阶段借用行为等同共享借用，
    /// 读它借走的位置没问题（`v.push(v.len())` 里读 v）。
    fn check_read(ctx: &BorrowCtx, place: &MirPlace, active: &[Active]) -> Result<(), String> {
        for a in active {
            let loan = &ctx.loans[a.idx];
            if loan.kind != LoanKind::Mut || a.reserved {
                continue;
            }
            if Self::is_authorized(ctx, place, a.idx) {
                continue;
            }
            if Self::places_overlap(place, &loan.place, &ctx.facts.pt) {
                return Err(format!(
                    "cannot read `{}` because it is mutably borrowed as `{}`",
                    Self::describe_place(place, ctx.f),
                    Self::get_var_name(ctx.f, loan.borrow_var.base_id)
                ));
            }
        }
        Ok(())
    }

    /// 写/移动/Drop：跟任何活跃借用冲突，预留阶段也不例外。
    fn check_exclusive(
        ctx: &BorrowCtx,
        place: &MirPlace,
        active: &[Active],
        action: &str,
    ) -> Result<(), String> {
        for a in active {
            let loan = &ctx.loans[a.idx];
            if Self::is_authorized(ctx, place, a.idx) {
                continue;
            }
            if Self::places_overlap(place, &loan.place, &ctx.facts.pt) {
                return Err(format!(
                    "cannot {} `{}` because it is {} borrowed as `{}`",
                    action,
                    Self::describe_place(place, ctx.f),
                    loan.kind,
                    Self::get_var_name(ctx.f, loan.borrow_var.base_id)
                ));
            }
        }
        Ok(())
    }

    fn check_write(ctx: &BorrowCtx, place: &MirPlace, active: &[Active]) -> Result<(), String> {
        Self::check_exclusive(ctx, place, active, "write")
    }

    fn check_move(ctx: &BorrowCtx, place: &MirPlace, active: &[Active]) -> Result<(), String> {
        Self::check_exclusive(ctx, place, active, "move")
    }

    /// 取借用：可变借用跟任何重叠的活跃借用冲突；共享借用只跟可变借用冲突，
    /// 但预留阶段的两阶段借用除外。
    ///
    /// 授权跳过在这里同样需要：`let y = &mut *x;`（对 x 做重借用）里，被借的
    /// 位置是 `*x`，而 x 携带的 loan（place = 它自己借的那个位置）展开后跟它
    /// 重叠，没有授权判定会把"通过 x 重借用"报成跟 x 自己冲突。
    fn check_borrow_creation(
        ctx: &BorrowCtx,
        place: &MirPlace,
        mutable: bool,
        active: &[Active],
    ) -> Result<(), String> {
        for a in active {
            let loan = &ctx.loans[a.idx];
            if Self::is_authorized(ctx, place, a.idx) {
                continue;
            }
            if !Self::places_overlap(place, &loan.place, &ctx.facts.pt) {
                continue;
            }
            if mutable {
                return Err(format!(
                    "cannot mutably borrow `{}` because it is already {} borrowed as `{}`",
                    Self::describe_place(place, ctx.f),
                    loan.kind,
                    Self::get_var_name(ctx.f, loan.borrow_var.base_id)
                ));
            } else if loan.kind == LoanKind::Mut && !a.reserved {
                return Err(format!(
                    "cannot shared-borrow `{}` because it is mutably borrowed as `{}`",
                    Self::describe_place(place, ctx.f),
                    Self::get_var_name(ctx.f, loan.borrow_var.base_id)
                ));
            }
        }
        Ok(())
    }

    // ============================================================
    // 右值 place 拆解（供冲突检查用）
    // ============================================================

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
                for a in args { Self::operand_place(a, copies, moves); }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::operand_place(receiver, copies, moves);
                for a in args { Self::operand_place(a, copies, moves); }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields { Self::operand_place(op, copies, moves); }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for a in args { Self::operand_place(a, copies, moves); }
            }
            MirRvalue::Ref { place, .. } => Self::collect_nested_index_operands(place, copies, moves),
            MirRvalue::ArrayLiteral(elems) => {
                for e in elems { Self::operand_place(e, copies, moves); }
            }
            MirRvalue::Discriminant { value, .. } => Self::operand_place(value, copies, moves),
            MirRvalue::Phi { values } => {
                for (_, op) in values { Self::operand_place(op, copies, moves); }
            }
        }
    }

    fn operand_place(op: &MirOperand, copies: &mut Vec<MirPlace>, moves: &mut Vec<MirPlace>) {
        match op {
            MirOperand::Copy(p) => {
                Self::collect_nested_index_operands(p, copies, moves);
                copies.push(p.clone());
            }
            MirOperand::Move(p) => {
                Self::collect_nested_index_operands(p, copies, moves);
                moves.push(p.clone());
            }
            MirOperand::Constant(_) | MirOperand::Sym(_) | MirOperand::Static(_) => {}
        }
    }

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
    //
    // 一个 place 展开成若干条"根 + 投影序列"的候选路径（Flat）：points-to
    // 给出多个 target 时，`*r` 有多种可能的展开，两个 place 只要有任意一对
    // 候选路径重叠就判为重叠（保守——宁可误报，不能漏报）。

    /// 展开的最大深度守卫：引用互相指向（`a = &mut *b; b = &mut *a`）会让
    /// points-to 形成环。超过深度时不再往下展开，直接退回 Root::Any——
    /// "可能跟任何东西重叠"，保守但安全。
    const FLATTEN_MAX_DEPTH: usize = 32;

    /// 候选路径数量的上限：Phi 汇合出很多 target、再叠上多层字段时展开
    /// 结果会指数增长。超过上限同样退回 Root::Any。
    const FLATTEN_MAX_ALTS: usize = 64;

    fn places_overlap(a: &MirPlace, b: &MirPlace, pt: &HashMap<SsaLocal, PtSet>) -> bool {
        let fa = Self::flatten(a, pt);
        let fb = Self::flatten(b, pt);
        for x in &fa {
            for y in &fb {
                if Self::flat_overlap(x, y) {
                    return true;
                }
            }
        }
        false
    }

    /// 两条候选路径是否重叠：根要相同（Any 跟什么都重叠），然后逐个投影比较，
    /// 一旦发现"一定指向不同位置"的投影就判不重叠；比较到较短的一条结束
    /// 仍没分出来，说明一条是另一条的前缀（`x` 与 `x.a`），重叠。
    fn flat_overlap(a: &Flat, b: &Flat) -> bool {
        let (ra, pa) = a;
        let (rb, pb) = b;
        if *ra == Root::Any || *rb == Root::Any {
            return true;
        }
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
                // 两个不同的常量下标是同一个数组里的不同元素。
                (Proj::ConstIndex(x), Proj::ConstIndex(y)) => {
                    if x != y {
                        return false;
                    }
                }
                // 同一个枚举值的不同变体的 payload 不重叠。
                (Proj::Variant(x), Proj::Variant(y)) => {
                    if x != y {
                        return false;
                    }
                }
                // 其余组合保守地视为重叠，包括：动态下标与任何下标、
                // Deref 与 Deref（前缀相同的两个"解引用"只在指针值相同
                // 时才指向同一处——按 base_id 归并的根不区分 SSA 版本，
                // 无法确认，不敢据此判不重叠）、不同种类的投影。
                (Proj::Field(_), _)
                | (Proj::ConstIndex(_), _)
                | (Proj::DynIndex, _)
                | (Proj::Variant(_), _)
                | (Proj::Deref, _)
                | (Proj::Opaque, _) => return true,
            }
        }
        true
    }

    fn flatten(place: &MirPlace, pt: &HashMap<SsaLocal, PtSet>) -> Vec<Flat> {
        Self::flatten_at(place, pt, 0)
    }

    /// 下标投影：只认直接写在 MIR 里的整数常量（不认 SCCP 的结论，borrow.rs
    /// 不依赖优化分析）。常量值用 detect::int_literal_value 统一取出，不在
    /// 这里把十二种整数字面量变体各列一遍、各自做可能截断的类型转换。
    fn index_proj(index: &MirOperand) -> Proj {
        match int_literal_value(index) {
            Some(v) => Proj::ConstIndex(v),
            None => Proj::DynIndex,
        }
    }

    fn flatten_at(place: &MirPlace, pt: &HashMap<SsaLocal, PtSet>, depth: usize) -> Vec<Flat> {
        let mut out: Vec<Flat> = Vec::new();
        match place {
            MirPlace::Ssa(s) => out.push((Root::Local(s.base_id), Vec::new())),
            MirPlace::Field { base, field } => {
                for (r, mut ps) in Self::flatten_at(base, pt, depth) {
                    ps.push(Proj::Field(field.clone()));
                    out.push((r, ps));
                }
            }
            MirPlace::Index { base, index } => {
                let proj = Self::index_proj(index);
                for (r, mut ps) in Self::flatten_at(base, pt, depth) {
                    ps.push(proj.clone());
                    out.push((r, ps));
                }
            }
            MirPlace::EnumPayload { base, variant_name, .. } => {
                for (r, mut ps) in Self::flatten_at(base, pt, depth) {
                    ps.push(Proj::Variant(variant_name.clone()));
                    out.push((r, ps));
                }
            }
            MirPlace::Deref(base) => {
                if depth >= Self::FLATTEN_MAX_DEPTH {
                    out.push((Root::Any, Vec::new()));
                    return out;
                }
                // `*base` 优先展开成 base 的 target（points-to 里记的所有
                // 可能 target）。base 不在表里（函数参数传入的引用）、
                // 或者表里标了 unknown（汇合进来的某条路径来源未知）时，
                // 还要加上保守展开：以 base 自己为根、再接一层 Deref。
                let mut need_fallback = true;
                if let MirPlace::Ssa(s) = base.as_ref() {
                    if let Some(set) = pt.get(s) {
                        for target in &set.targets {
                            out.extend(Self::flatten_at(target, pt, depth + 1));
                        }
                        need_fallback = set.unknown || set.targets.is_empty();
                    }
                }
                if need_fallback {
                    for (r, mut ps) in Self::flatten_at(base, pt, depth) {
                        ps.push(Proj::Deref);
                        out.push((r, ps));
                    }
                }
            }
        }
        if out.len() > Self::FLATTEN_MAX_ALTS {
            out.clear();
            out.push((Root::Any, Vec::new()));
        }
        out
    }

    fn place_base_id(place: &MirPlace) -> Option<usize> {
        match place {
            MirPlace::Ssa(s) => Some(s.base_id),
            MirPlace::Field { base, .. } => Self::place_base_id(base),
            MirPlace::Index { base, .. } => Self::place_base_id(base),
            MirPlace::Deref(base) => Self::place_base_id(base),
            MirPlace::EnumPayload { base, .. } => Self::place_base_id(base),
        }
    }

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

    /// 路径的可读形式：变量名，加上 `.字段` 链（整个变量就只是变量名）。
    fn describe_path(f: &MirFn, path: &PathKey) -> String {
        let mut s = Self::get_var_name(f, path.base);
        for field in &path.fields {
            s.push('.');
            s.push_str(field);
        }
        s
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
