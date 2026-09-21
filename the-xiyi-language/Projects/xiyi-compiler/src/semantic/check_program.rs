// src/semantic/check_program.rs
use std::collections::{HashMap, HashSet};
use crate::ast::*;
use crate::hir;
use crate::hir_builder;
use super::lookup::MethodInfo;

pub struct TypeChecker {
    pub scopes: Vec<HashMap<String, Type>>,
    pub structs: HashMap<String, StructDef>,
    pub enums: HashMap<String, EnumDef>,
    pub consts: HashMap<String, Type>,
    pub functions: HashMap<String, FnDef>,
    // 类型名 -> 方法名 -> MethodInfo。之前 implement 块从来没被收集过，
    // 方法调用只能靠"查不到就返回 I32"这种兜底，这张表补上之后，方法调用
    // 才能真正按方法自己的签名做类型检查。
    // 关键修复（拆文件引入）：这个字段原来是私有的（没有 pub）——当
    // TypeChecker 的方法全部挤在同一个 impl 块、同一个文件里时无所谓，
    // 但现在 lookup.rs（方法表的读写逻辑）、check_expr.rs（方法调用/
    // 限定路径静态调用）都要在各自的文件里通过 `self.methods` 访问它，
    // 分属不同的子模块（semantic::check_program vs semantic::lookup /
    // semantic::check_expr 是平级模块，不是父子关系），私有字段过不了
    // 编译。这个结构体上其它字段本来就都是 pub，这里补 pub 只是让它
    // 跟兄弟字段保持一致，不是新引入的例外。
    pub methods: HashMap<String, HashMap<String, MethodInfo>>,
    pub in_model: bool,
    pub fn_stack: Vec<String>,
    pub model_names: HashSet<String>,
    pub model_return_types: HashMap<String, Type>,
    pub model_sensitivities: HashMap<String, f64>,
    pub current_self_type: Option<Type>,
    // 新增：当前正在检查的函数的声明返回类型，供 Stmt::Return 用来给
    // return 语句里的表达式（比如裸 Err(...)）传递期望类型提示。
    pub current_return_type: Option<Type>,
    pub expr_types: HashMap<usize, Type>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            structs: HashMap::new(),
            enums: HashMap::new(),
            consts: HashMap::new(),
            functions: HashMap::new(),
            methods: HashMap::new(),
            in_model: false,
            fn_stack: Vec::new(),
            model_names: HashSet::new(),
            model_return_types: HashMap::new(),
            model_sensitivities: HashMap::new(),
            current_self_type: None,
            current_return_type: None,
            expr_types: HashMap::new(),
        }
    }

    // 关键重构：以前 check_program 里直接堆着两个几乎一样结构的
    // `for item in &program.items { match item { ... } }`——第一遍收集
    // 签名、第二遍检查函数体，这个"先收集再检查"的两阶段结构本身是
    // 对的（函数体互相调用前必须先把所有签名收齐），但两次 match 各自
    // 罗列一遍所有 Item 变体，以后加一个新的 Item 种类（比如
    // Item::TraitDef），两处都要记得加，忘了一处也不会立刻报错——
    // 只会在某个用到新变体的地方悄悄走空分支。拆成 collect_item（只
    // 登记、不检查）和 check_item（只检查、不登记）两个方法后，新增
    // 一种 Item 只需要各自去这两个方法里补一条 match 分支，属于
    // "看代码就知道要不要补"的地方，不再是两处分散、容易漏掉的重复。
    pub fn check_program(&mut self, program: &Program) -> Result<hir::HirProgram, String> {
        for item in &program.items {
            self.collect_item(item)?;
        }

        for item in &program.items {
            self.check_item(item)?;
        }

        let hir = hir_builder::HirBuilder::build(program, &self.expr_types)?;
        Ok(hir)
    }

    // ===== 第一遍遍历：只登记签名，不检查函数体 =====
    fn collect_item(&mut self, item: &Item) -> Result<(), String> {
        match item {
            Item::FnDef(f) => {
                self.functions.insert(f.name.clone(), f.clone());
            }
            Item::StructDef(s) => {
                self.structs.insert(s.name.clone(), s.clone());
            }
            Item::EnumDef(e) => {
                self.enums.insert(e.name.clone(), e.clone());
            }
            Item::ConstDef(c) => {
                self.consts.insert(c.name.clone(), c.ty.clone());
            }
            // model 相关的收集逻辑（登记 model 名字、把 model 的字段
            // 转成一个同名 struct、记录 forward 的返回类型/sensitivity
            // 属性）挪到了 check_model.rs 的 collect_model_def 里，这里
            // 只是委托调用。collect_model_def 现在会返回 Err（比如
            // #[sensitivity(const = ...)] 里写了个解析不出来的有理数），
            // 用 `?` 照常传播，不能吞掉。
            Item::ModelDef(m) => self.collect_model_def(m)?,
            Item::ProtoDef(_) => {}
            Item::Use(_) => {}
            // implement 块登记进方法表这件事挪到了 lookup.rs 的
            // register_impl 里，这里只是委托调用。
            //
            // register_impl 现在会检测同名方法的重叠实现（规范
            // §3.0.6，error[IMP001]），撞名时返回 Err 而不是像以前
            // 那样静默覆盖。这里用 `?` 把错误原样传播出去。
            Item::Implement(imp) => self.register_impl(imp)?,
            _ => {}
        }
        Ok(())
    }

    // ===== 第二遍遍历：真正检查函数体/常量初始值 =====
    fn check_item(&mut self, item: &Item) -> Result<(), String> {
        match item {
            Item::FnDef(f) => self.check_func(f)?,
            Item::ConstDef(c) => {
                let value_type = self.check_expr(&c.value)?;
                if !self.types_equal(&value_type, &c.ty) {
                    return Err(format!("const type mismatch: expected {:?}, got {:?}", c.ty, value_type));
                }
            }
            // model 块本身"必须有 forward"、检查每个函数体这些逻辑，
            // 挪到了 check_model.rs 的 check_model_def 里。
            Item::ModelDef(m) => self.check_model_def(m)?,
            Item::ProtoDef(_) => {}
            Item::Use(_) => {}
            // implement 块里的方法体在这里真正被检查一遍（`self` 的
            // 类型设成这个 implement 块的 target_type，跟 ModelDef 那边
            // 的处理方式一致）。
            Item::Implement(imp) => {
                self.current_self_type = Some(imp.target_type.clone());
                for fn_def in &imp.functions {
                    self.check_func(fn_def)?;
                }
                self.current_self_type = None;
            }
            _ => {}
        }
        Ok(())
    }
}
