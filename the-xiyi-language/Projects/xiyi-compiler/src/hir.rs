use crate::ast::{Type, PrivacyTag, ShapeDim, Literal, Pattern, BinaryOp, UnsafeKind, UnaryOp, IfKind, WhereClause, Attribute};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct EffectSet {
    pub has_io: bool,
    pub has_rng: bool,
    pub has_ai: bool,
    pub has_ffi: bool,
    pub has_panic: bool,
}

impl EffectSet {
    // 关键修复：merge 和 merge_with 原来是两份各自把五个字段抄一遍的
    // OR 逻辑——内容完全重复，只是一个在循环里累加多个 EffectSet，
    // 一个只处理一个。这种重复本身就是个隐患：以后 EffectSet 再加一个
    // has_xxx 字段（这门语言目前已经有 io/rng/ai/ffi/panic 五种效应，
    // 不难想见还会再加），改的人很容易只改其中一个函数就以为完事了，
    // 另一个悄悄漏改，从此 merge 和 merge_with 在语义上不再等价，却没
    // 有任何编译错误能提醒——这类"两处手抄同一份逻辑"的漏改，在这个
    // 项目里已经反复出现过好几次了。把 merge 改成基于 merge_with 实现，
    // 让"哪些字段要参与合并"只有一处需要维护，加字段时唯一要改的地方
    // 就是 merge_with。
    pub fn merge(sets: &[&EffectSet]) -> Self {
        let mut merged = EffectSet::default();
        for set in sets {
            merged.merge_with(set);
        }
        merged
    }

    pub fn merge_with(&mut self, other: &EffectSet) {
        self.has_io |= other.has_io;
        self.has_rng |= other.has_rng;
        self.has_ai |= other.has_ai;
        self.has_ffi |= other.has_ffi;
        self.has_panic |= other.has_panic;
    }

    // 关键新增：所有副作用标志位都是 false 才算"干净"。has_panic 也算
    // 副作用——一个可能 panic 的调用就算结果没人用也不能删，因为 panic
    // 的异常传播会跳过后面的 Drop，是可观察行为。
    //
    // 跟 merge/merge_with 收在同一个 impl 里是有意的：EffectSet 再加
    // 一个 has_xxx 字段时，这三个函数必须一起改，放在一起才不容易漏。
    //
    // 注意这个函数只回答"这份 EffectSet 里记了没有副作用"，不保证
    // EffectSet 本身记全了：hir_builder.rs 目前只有字面函数名是 print 的
    // 调用才会置 has_io，调用别的函数并不会合并被调函数的 effects（见
    // analyze.rs::compute_purity_table 里对这一点的处理）。
    pub fn is_pure(&self) -> bool {
        !self.has_io && !self.has_rng && !self.has_ai && !self.has_ffi && !self.has_panic
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Sensitivity {
    Const(f64),
    Symbolic(String),
    Dynamic { f: String, bound: f64 },
    Manual(f64),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

impl Default for Span {
    fn default() -> Self {
        Span { start: 0, end: 0, line: 0, col: 0 }
    }
}

// ===== HIR 里统一表示泛型参数，对应 ast::GenericParam::Type { name, bounds } =====
#[derive(Debug, PartialEq, Clone)]
pub struct HirGenericParam {
    pub name: String,
    pub bounds: Vec<String>,
}

// ===== HIR 表示 implement 块（补上 generic_params，对应 ast::ImplementDef 已有的字段）=====
#[derive(Debug, Clone)]
pub struct HirImplement {
    // 关键新增：ast::ImplementDef 早就有 attributes/where_clause 这两个
    // 字段（parser.rs 已经在填），HIR 这层一直没跟上，把它们丢在了
    // build_implement 里——跟 generic_params 当年的坑是同一个模式。
    // Attribute/WhereClause 都是纯数据结构，HIR 直接复用 ast 里的定义，
    // 不需要另起一套 Hir 版本。
    pub attributes: Vec<Attribute>,
    pub generic_params: Vec<HirGenericParam>,
    pub target_type: Type,
    pub interface_name: Option<String>,
    pub where_clause: Vec<WhereClause>,
    pub functions: Vec<HirFn>,
}

// ===== HIR 表示 interface 定义（同上，补 generic_params）=====
#[derive(Debug, Clone)]
pub struct HirInterface {
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub methods: Vec<HirFnSig>,
}

#[derive(Debug, Clone)]
pub struct HirFnSig {
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub params: Vec<HirParam>,
    pub return_type: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct HirProgram {
    pub models: Vec<HirModel>,
    pub fns: Vec<HirFn>,
    pub structs: Vec<HirStruct>,
    pub enums: Vec<HirEnum>,
    pub consts: Vec<HirConst>,
    pub protos: Vec<HirProto>,
    pub impls: Vec<HirImplement>,
    pub interfaces: Vec<HirInterface>,
}

#[derive(Debug, Clone)]
pub struct HirModel {
    // 关键新增：同 HirImplement，ast::ModelDef 已有 attributes，HIR
    // 这层之前一直没接。
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub fields: Vec<HirField>,
    pub functions: Vec<HirFn>,
    pub sensitivity: Option<Sensitivity>,
    pub training_context_required: bool,
    pub privacy_eps: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HirField {
    pub name: String,
    pub ty: Type,
}

// ===== 补上 generic_params，对应 ast::FnDef 已有的字段，之前 HIR 这层把它丢了 =====
#[derive(Debug, Clone)]
pub struct HirFn {
    // 关键新增：同上，ast::FnDef 已有 attributes（比如 #[effect(...)]、
    // #[sensitivity(...)] 这些标注最终都要落在这里），HIR 这层之前
    // 一直没接，效果系统/敏感度分析拿不到这份信息。
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub params: Vec<HirParam>,
    pub return_type: Option<Type>,
    pub body: HirBlock,
    pub effects: EffectSet,
    pub sensitivity: Option<Sensitivity>,
    pub is_forward: bool,
}

#[derive(Debug, Clone)]
pub struct HirParam {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct HirBlock {
    pub stmts: Vec<HirStmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirCallArg {
    Positional(HirExpr),
    Named(String, HirExpr),
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    Let {
        name: String,
        ty: Option<Type>,
        init: HirExpr,
        mutable: bool,
        persist: bool,
        span: Span,
    },
    Expr { expr: HirExpr, span: Span },
    Return { expr: Option<HirExpr>, span: Span },
    While { cond: HirExpr, body: HirBlock, span: Span },
    For { var: String, iterable: HirExpr, body: HirBlock, span: Span },
    // 同 ast::AssignStmt 的改动：name -> target，支持字段/索引作为赋值目标
    Assign { target: Box<HirExpr>, expr: HirExpr, span: Span },
    Loop { body: HirBlock, span: Span },
    Break { span: Span },
    // 关键新增：对应 ast::Stmt::Continue——语句本身不携带数据，只是
    // "跳到循环头，进入下一轮"的标记。真正"该跳到哪个块、该给循环头
    // 哪些 Phi 补一条 incoming"这些语义都在 mir_builder.rs 里落地，
    // HIR 这一层只负责如实转述"这里有一个 continue"这件事。
    Continue { span: Span },
    UnsafeBlock { kind: UnsafeKind, body: HirBlock, span: Span },
}

#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
    pub privacy_tag: Option<PrivacyTag>,
    pub sensitivity: Sensitivity,
    pub effects: EffectSet,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirExprKind {
    Literal(Literal),
    Ident(String),
    Sym(String),
    BinaryOp {
        op: BinaryOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },
    // ===== 补 generic_args，支持 identity::<i32>(1) 这种显式实例化 =====
    // 注：在 sema 真正实现泛型替换之前，这个字段允许为空 Vec，先占位、不阻塞现有调用
    Call {
        // 同 ast::ExprKind::Call 的 qualifier，透传即可
        qualifier: Option<String>,
        func: String,
        generic_args: Vec<Type>,
        args: Vec<HirCallArg>,
        is_method: bool,
    },
    Block(HirBlock),
    StructInit {
        struct_name: String,
        generic_args: Vec<Type>,
        fields: Vec<(String, HirExpr)>,
    },
    FieldAccess {
        struct_expr: Box<HirExpr>,
        field_name: String,
    },
    Range {
        start: Box<HirExpr>,
        end: Box<HirExpr>,
    },
    EnumVariantAccess {
        enum_name: String,
        variant_name: String,
    },
    EnumVariantConstruction {
        enum_name: String,
        generic_args: Vec<Type>,
        variant_name: String,
        args: Vec<HirCallArg>,
    },
    Match {
        cond: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
    },
    Closure {
        param: String,
        body: Box<HirExpr>,
    },
    If {
        // 同 ast::ExprKind::If 的 kind，直接复用 ast::IfKind，透传即可
        kind: IfKind,
        cond: Box<HirExpr>,
        then_expr: Box<HirExpr>,
        else_expr: Option<Box<HirExpr>>,
    },
    ArrayLiteral(Vec<HirExpr>),
    UnsafeBlock {
        kind: UnsafeKind,
        body: HirBlock,
        span: Span,
    },
    // ===== 新增：一元运算符，对应 ast::ExprKind::Unary =====
    Unary {
        op: UnaryOp,
        expr: Box<HirExpr>,
    },
    // ===== 新增：as 类型转换，对应 ast::ExprKind::Cast =====
    Cast {
        expr: Box<HirExpr>,
        ty: Type,
    },
    // ===== 新增：索引表达式，对应 ast::ExprKind::Index =====
    Index {
        expr: Box<HirExpr>,
        index: Box<HirExpr>,
    },
    // ===== 新增：lack &[T] 空切片字面量，对应 ast::ExprKind::LackSlice =====
    LackSlice(Type),
    // ===== 新增：取地址 &x / &mut x，对应 ast::ExprKind::Ref =====
    Ref {
        mutable: bool,
        expr: Box<HirExpr>,
    },
    // ===== 新增：解引用 *p，对应 ast::ExprKind::Deref =====
    Deref(Box<HirExpr>),
}

#[derive(Debug, Clone)]
pub struct HirMatchArm {
    pub pattern: Pattern,
    pub expr: HirExpr,
}

#[derive(Debug, Clone)]
pub struct HirStruct {
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub fields: Vec<HirField>,
}

#[derive(Debug, Clone)]
pub struct HirEnumVariant {
    pub name: String,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct HirEnum {
    pub name: String,
    pub generic_params: Vec<HirGenericParam>,
    pub variants: Vec<HirEnumVariant>,
}

impl HirEnum {
    /// 派生自 generic_params，不单独存字段，避免和 generic_params 不同步
    pub fn is_generic(&self) -> bool {
        !self.generic_params.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct HirConst {
    pub name: String,
    pub ty: Type,
    pub value: HirExpr,
}

#[derive(Debug, Clone)]
pub struct HirProto {
    pub name: String,
    pub variants: Vec<HirProtoVariant>,
}

#[derive(Debug, Clone)]
pub struct HirProtoVariant {
    pub name: String,
    pub ty: Option<Type>,
}