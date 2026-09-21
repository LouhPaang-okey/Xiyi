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
use crate::mir_builder::{LoopCtx, MirBuilder};
use std::collections::HashMap;

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

    pub(crate) fn new_version(&mut self, base_id: usize) -> u32 {
        let ver = self.ssa_versions.get(&base_id).copied().unwrap_or(0) + 1;
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

    // 关键修复：FieldAccess/Index 读一个字段/下标（`let x = p.x;`/
    // `let x = arr[i];`）之前完全没往 moved 里登记任何东西——build_place
    // 递归拆出来的最终 MirPlace 是 Field{base:...}/Index{base:...}，
    // 只存了"读的是哪个字段/下标"，从没往上找过"这次操作到底 touch 的
    // 是哪个变量"，导致 pop_scope 在 p 离开作用域时照常给它插一条
    // Drop——对一个已经被（哪怕只是部分）移动过的值调用 drop()，生成
    // 的 Rust 编译不过。这个 helper 就是补上"往上找到底"这一步：Field/
    // Index/Deref/EnumPayload 都是"在某个 place 上面套一层投影"，顺着
    // base 一路往下找，找到 Ssa 就是这次操作真正touch 到的那个变量；
    // Static 没有对应的局部变量，返回 None。
    pub(crate) fn place_base_ssa(place: &MirPlace) -> Option<SsaLocal> {
        match place {
            MirPlace::Ssa(s) => Some(*s),
            MirPlace::Field { base, .. } => Self::place_base_ssa(base),
            MirPlace::Index { base, .. } => Self::place_base_ssa(base),
            MirPlace::Deref(base) => Self::place_base_ssa(base),
            MirPlace::EnumPayload { base, .. } => Self::place_base_ssa(base),
            MirPlace::Static(_) => None,
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

    // -------- 循环栈（给 break 用） --------
    // 关键新增：跟 push_scope/pop_scope 配套，在进入 While/Loop 的循环
    // 体之前调用——此时循环体自己的 push_scope 还没发生，记下的
    // scope_depth 就是"循环体之外"的作用域层数，break 时用它切出
    // "循环体内部、该被跳过并补 Drop"的那一段 scope_vars（见 LoopCtx
    // 定义处的注释）。
    pub(crate) fn push_loop(&mut self, break_target: usize) {
        self.loop_stack.push(LoopCtx {
            break_target,
            scope_depth: self.scope_vars.len(),
        });
    }

    // 循环体构建完毕（不管成功还是中途报错）都要弹栈，避免栈里留着
    // 一层已经不存在的循环，把外层同名/同层级的另一个循环的 break
    // 误导到这层失效的目标上。调用点用"先 pop 再 `?` 传播错误"的顺序
    // 保证这一点，见 mir_builder.rs 里 While/Loop 分支的写法。
    pub(crate) fn pop_loop(&mut self) {
        self.loop_stack.pop();
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
    // 关键设计：块在用到之前就先创建好（占位终止器是 Unreachable），
    // 之后随时可以用 id 引用它、往里面塞语句，最后再补上真正的终止器。
    // 这是为了支持 if/while 这类需要"提前知道 then/else 块的 id 才能
    // 设置当前块的跳转目标"的控制流——不这样做的话，构建顺序会陷入
    // "先有鸡还是先有蛋"的死结。
    pub(crate) fn new_block(&mut self) -> usize {
        let id = self.blocks.len();
        self.blocks.push(MirBlock {
            id,
            stmts: Vec::new(),
            terminator: MirTerminator::Unreachable,
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

    pub(crate) fn current_terminator_is_placeholder(&self) -> bool {
        match self.blocks[self.current_block].terminator {
            MirTerminator::Unreachable => true,
            _ => false,
        }
    }
}
