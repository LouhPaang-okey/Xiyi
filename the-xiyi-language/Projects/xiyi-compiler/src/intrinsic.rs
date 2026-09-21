// intrinsic.rs
use crate::ast::{BinaryOp, CallArg, Expr, ExprKind, Literal, ShapeDim, Type, UnaryOp};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum IntrinsicFn {
    Print,
    Panic,
    FromUtf8Unchecked,
    TensorCond,
    TensorWhileLoop,
    Linear,
    Conv2d,
    MaxPool2d,
    Flatten,
    Reshape,
    Relu,
    Dropout,
    LayerNorm,
    Sum,
    // 关键修复：embedding 原来在 check_expr.rs 里被当作内建函数直接
    // 手写 `if func == "embedding"` 处理，但从来没有登记进这张表——
    // 跟这里其它十几个内建函数比，它是唯一一个"有实现、没身份"的。
    // 补上变体本身还不够，ALL / from_str / build_fn_table 三处必须
    // 同步加，见下面 ALL 那条注释。
    Embedding,
}

impl IntrinsicFn {
    // 手动维护的全部变体列表，替代 std::mem::variant_count::<IntrinsicFn>()。
    //
    // 关键修复：variant_count 到现在（2026）还是 nightly-only 的 unstable
    // API（tracking issue #73662，标着 B-unstable，还挂着
    // S-tracking-design-concerns，没有稳定的迹象），需要
    // `#![feature(variant_count)]` 才能用。这个 crate 要在 stable 工具链
    // 上编译，用了它直接编译不过。
    //
    // 这里不用宏（比如 strum 那类 derive 宏）去自动生成变体数量或列表——
    // 一是不想为了这一件事引入额外依赖，二是自举以后我们自己的语言大概率
    // 也没有对应的反射/宏能力，Rust 这边的实现风格越"手写直白"，以后照着
    // 搬过去就越省事。所以选择手写这份列表：新增/删除 IntrinsicFn 变体时，
    // 这里、from_str、以及 build_fn_table 里的 insert 都要跟着改——忘改
    // 的话，多出来的变体会在 `FN_TABLE.get(name as usize)` 那里悄悄返回
    // None（数组越界访问用的是 `.get`，不会 panic），表现为"这个内建函数
    // 查不到"，而不是编译错误。这是手动维护无法完全消除的代价，只能靠这
    // 条注释和三处改动挨在一起提醒。
    pub const ALL: [IntrinsicFn; 15] = [
        Self::Print,
        Self::Panic,
        Self::FromUtf8Unchecked,
        Self::TensorCond,
        Self::TensorWhileLoop,
        Self::Linear,
        Self::Conv2d,
        Self::MaxPool2d,
        Self::Flatten,
        Self::Reshape,
        Self::Relu,
        Self::Dropout,
        Self::LayerNorm,
        Self::Sum,
        Self::Embedding,
    ];

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "print" => Some(Self::Print),
            "panic" => Some(Self::Panic),
            "from_utf8_unchecked" => Some(Self::FromUtf8Unchecked),
            "tensor.cond" => Some(Self::TensorCond),
            "tensor.while_loop" => Some(Self::TensorWhileLoop),
            "linear" => Some(Self::Linear),
            "conv2d" => Some(Self::Conv2d),
            "max_pool2d" => Some(Self::MaxPool2d),
            "flatten" => Some(Self::Flatten),
            "reshape" => Some(Self::Reshape),
            "relu" => Some(Self::Relu),
            "dropout" => Some(Self::Dropout),
            "layer_norm" => Some(Self::LayerNorm),
            "sum" => Some(Self::Sum),
            "embedding" => Some(Self::Embedding),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum IntrinsicConst {
    I8Max,
    I16Max,
    I32Max,
    I64Max,
    I128Max,
    U8Max,
    U16Max,
    U32Max,
    U64Max,
    U128Max,
    F32Infinity,
    F32Nan,
    F64Infinity,
    F64Nan,
}

impl IntrinsicConst {
    // 同 IntrinsicFn::ALL，手动维护，理由见那边的注释。
    pub const ALL: [IntrinsicConst; 14] = [
        Self::I8Max,
        Self::I16Max,
        Self::I32Max,
        Self::I64Max,
        Self::I128Max,
        Self::U8Max,
        Self::U16Max,
        Self::U32Max,
        Self::U64Max,
        Self::U128Max,
        Self::F32Infinity,
        Self::F32Nan,
        Self::F64Infinity,
        Self::F64Nan,
    ];

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "i8::MAX" => Some(Self::I8Max),
            "i16::MAX" => Some(Self::I16Max),
            "i32::MAX" => Some(Self::I32Max),
            "i64::MAX" => Some(Self::I64Max),
            "i128::MAX" => Some(Self::I128Max),
            "u8::MAX" => Some(Self::U8Max),
            "u16::MAX" => Some(Self::U16Max),
            "u32::MAX" => Some(Self::U32Max),
            "u64::MAX" => Some(Self::U64Max),
            "u128::MAX" => Some(Self::U128Max),
            "f32::INFINITY" => Some(Self::F32Infinity),
            "f32::NAN" => Some(Self::F32Nan),
            "f64::INFINITY" => Some(Self::F64Infinity),
            "f64::NAN" => Some(Self::F64Nan),
            _ => None,
        }
    }
}

const FN_COUNT: usize = IntrinsicFn::ALL.len();
const CONST_COUNT: usize = IntrinsicConst::ALL.len();

// ===== 内建函数元数据 =====

#[derive(Debug, Clone, PartialEq)]
pub enum IntrinsicParam {
    Value(Type),
    Ref(bool, Type),
    Fn(Vec<Type>, Box<Type>),
    VarArgs,
}

#[derive(Debug, Clone)]
pub struct IntrinsicSignature {
    pub params: Vec<IntrinsicParam>,
    pub return_type: Type,
}

#[derive(Debug, Clone)]
pub struct Intrinsic {
    pub kind: IntrinsicKind,
    pub allowed_in_model: bool,
    pub is_pure: bool,
    pub requires_unsafe: bool,
    pub signature: Option<IntrinsicSignature>,
    pub link_name: &'static str,
    pub doc: &'static str,
}

impl Intrinsic {
    pub fn diverges(&self) -> bool {
        // 改成显式 match，不用 matches! 宏（哪怕它是标准库自带的基础
        // 宏）——你说了不想用宏，这里干脆一并去掉，别留个例外。
        match &self.signature {
            Some(IntrinsicSignature { return_type: Type::Never, .. }) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicKind {
    Function,
    Special,
}

// ===== 常量元数据 =====

#[derive(Debug, Clone)]
pub struct Constant {
    pub name: &'static str,
    pub ty: Type,
    pub value: Option<Literal>,
    pub doc: &'static str,
}

impl Constant {
    pub fn has_known_value(&self) -> bool {
        self.value.is_some()
    }
}

// ===== 注册表（数组 + OnceLock） =====

// 与 HashMap 方案不同：用数组替代 HashMap，长度由 IntrinsicFn::ALL /
// IntrinsicConst::ALL 这两份手写列表的长度决定（原因见上面 ALL 的注释：
// std::mem::variant_count 是 nightly-only 的 unstable API，不能用）。
// 查找从 O(1) 哈希变成 O(1) 直接索引，无哈希开销。
static FN_TABLE: OnceLock<[Option<Intrinsic>; FN_COUNT]> = OnceLock::new();
static CONST_TABLE: OnceLock<[Option<Constant>; CONST_COUNT]> = OnceLock::new();

fn build_fn_table() -> [Option<Intrinsic>; FN_COUNT] {
    let mut arr: [Option<Intrinsic>; FN_COUNT] = std::array::from_fn(|_| None);
    let mut insert = |name: IntrinsicFn, intrinsic: Intrinsic| {
        arr[name as usize] = Some(intrinsic);
    };

    // ---- 基础 IO ----
    insert(IntrinsicFn::Print, Intrinsic {
        kind: IntrinsicKind::Function,
        allowed_in_model: false,
        is_pure: false,
        requires_unsafe: false,
        signature: Some(IntrinsicSignature {
            params: vec![IntrinsicParam::Value(Type::Str)],
            return_type: Type::Unit,
        }),
        link_name: "xiyi::io::print",
        doc: "Prints a value to standard output",
    });
    insert(IntrinsicFn::Panic, Intrinsic {
        kind: IntrinsicKind::Function,
        allowed_in_model: false,
        is_pure: false,
        requires_unsafe: false,
        signature: Some(IntrinsicSignature {
            params: vec![IntrinsicParam::Value(Type::Str)],
            return_type: Type::Never,
        }),
        link_name: "core::panic",
        doc: "Panics with a given message",
    });
    insert(IntrinsicFn::FromUtf8Unchecked, Intrinsic {
        kind: IntrinsicKind::Function,
        allowed_in_model: true,
        is_pure: true,
        requires_unsafe: true,
        signature: Some(IntrinsicSignature {
            params: vec![IntrinsicParam::Ref(false, Type::Slice(Box::new(Type::U8)))],
            return_type: Type::Ref { mutable: false, inner: Box::new(Type::Str) },
        }),
        link_name: "core::str::from_utf8_unchecked",
        doc: "Converts &[u8] to &str without validation (unsafe)",
    });

    // ---- 图域控制流 ----
    insert(IntrinsicFn::TensorCond, Intrinsic {
        kind: IntrinsicKind::Special,
        allowed_in_model: true,
        is_pure: true,
        requires_unsafe: false,
        signature: None,
        link_name: "xiyi_tensor::cond",
        doc: "Dynamic conditional in graph domain",
    });
    insert(IntrinsicFn::TensorWhileLoop, Intrinsic {
        kind: IntrinsicKind::Special,
        allowed_in_model: true,
        is_pure: true,
        requires_unsafe: false,
        signature: None,
        link_name: "xiyi_tensor::while_loop",
        doc: "Dynamic while loop in graph domain",
    });

    // ---- 张量算子（签名复杂，由 sema 检查） ----
    for (name, link_name, doc) in [
        (IntrinsicFn::Embedding, "xiyi_math::embedding", "Embedding lookup"),
        (IntrinsicFn::Linear, "xiyi_math::linear", "Linear transformation"),
        (IntrinsicFn::Conv2d, "xiyi_math::conv2d", "2D convolution"),
        (IntrinsicFn::MaxPool2d, "xiyi_math::max_pool2d", "2D max pooling"),
        (IntrinsicFn::Flatten, "xiyi_math::flatten", "Flatten tensor"),
        (IntrinsicFn::Reshape, "xiyi_math::reshape", "Reshape tensor"),
        (IntrinsicFn::Relu, "xiyi_math::relu", "ReLU activation"),
        (IntrinsicFn::Dropout, "xiyi_math::dropout", "Dropout"),
        (IntrinsicFn::LayerNorm, "xiyi_math::layer_norm", "Layer normalization"),
        (IntrinsicFn::Sum, "xiyi_math::sum", "Sum reduction"),
    ] {
        insert(name, Intrinsic {
            kind: IntrinsicKind::Function,
            allowed_in_model: true,
            is_pure: true,
            requires_unsafe: false,
            signature: None,
            link_name,
            doc,
        });
    }

    arr
}

fn build_const_table() -> [Option<Constant>; CONST_COUNT] {
    let mut arr: [Option<Constant>; CONST_COUNT] = std::array::from_fn(|_| None);
    let mut insert = |name: IntrinsicConst, constant: Constant| {
        arr[name as usize] = Some(constant);
    };

    // 注：所有常量的 value 目前都是 None，因为常量折叠尚未实现。
    // 等实现后，将 None 替换为对应的 Literal 值即可。
    insert(IntrinsicConst::I8Max, Constant {
        name: "i8::MAX",
        ty: Type::I8,
        value: None,
        doc: "Maximum value of i8",
    });
    insert(IntrinsicConst::I16Max, Constant {
        name: "i16::MAX",
        ty: Type::I16,
        value: None,
        doc: "Maximum value of i16",
    });
    insert(IntrinsicConst::I32Max, Constant {
        name: "i32::MAX",
        ty: Type::I32,
        value: None,
        doc: "Maximum value of i32",
    });
    insert(IntrinsicConst::I64Max, Constant {
        name: "i64::MAX",
        ty: Type::I64,
        value: None,
        doc: "Maximum value of i64",
    });
    insert(IntrinsicConst::I128Max, Constant {
        name: "i128::MAX",
        ty: Type::I128,
        value: None,
        doc: "Maximum value of i128",
    });
    insert(IntrinsicConst::U8Max, Constant {
        name: "u8::MAX",
        ty: Type::U8,
        value: None,
        doc: "Maximum value of u8",
    });
    insert(IntrinsicConst::U16Max, Constant {
        name: "u16::MAX",
        ty: Type::U16,
        value: None,
        doc: "Maximum value of u16",
    });
    insert(IntrinsicConst::U32Max, Constant {
        name: "u32::MAX",
        ty: Type::U32,
        value: None,
        doc: "Maximum value of u32",
    });
    insert(IntrinsicConst::U64Max, Constant {
        name: "u64::MAX",
        ty: Type::U64,
        value: None,
        doc: "Maximum value of u64",
    });
    insert(IntrinsicConst::U128Max, Constant {
        name: "u128::MAX",
        ty: Type::U128,
        value: None,
        doc: "Maximum value of u128",
    });
    insert(IntrinsicConst::F32Infinity, Constant {
        name: "f32::INFINITY",
        ty: Type::F32,
        value: None,
        doc: "Infinity for f32",
    });
    insert(IntrinsicConst::F32Nan, Constant {
        name: "f32::NAN",
        ty: Type::F32,
        value: None,
        doc: "NaN for f32",
    });
    insert(IntrinsicConst::F64Infinity, Constant {
        name: "f64::INFINITY",
        ty: Type::F64,
        value: None,
        doc: "Infinity for f64",
    });
    insert(IntrinsicConst::F64Nan, Constant {
        name: "f64::NAN",
        ty: Type::F64,
        value: None,
        doc: "NaN for f64",
    });

    arr
}

// ===== 公共 API =====
pub fn get_intrinsic(name: IntrinsicFn) -> Option<&'static Intrinsic> {
    FN_TABLE
        .get_or_init(build_fn_table)
        .get(name as usize)?
        .as_ref()
}

pub fn get_constant(name: IntrinsicConst) -> Option<&'static Constant> {
    CONST_TABLE
        .get_or_init(build_const_table)
        .get(name as usize)?
        .as_ref()
}

pub fn is_intrinsic(s: &str) -> bool {
    IntrinsicFn::from_str(s).is_some() || IntrinsicConst::from_str(s).is_some()
}

// ===== 从 mir_builder.rs 搬过来的内建调用识别逻辑 =====
//
// 这两个函数原来是 mir_builder.rs 的 build_expr_rvalue 里
// HirExprKind::Call 分支内联的两段代码。之所以能搬过来而不用把
// MirBuilder 整个搬过来、也不用让这个文件反过来认识 mir.rs 的类型：
// 这两段逻辑其实只需要"这次调用长什么样"（func/qualifier/is_method/
// 参数个数）和 MirBuilder 当时的两个状态位（in_forward/unsafe_depth），
// 不需要真的持有 &mut MirBuilder，也不产出任何 MIR 值——常量引用只
// 返回一个名字字符串（由调用方自己包成 MirPlace::Static），内建函数
// 检测只返回"是不是、是哪个"，真正构造 MirRvalue::Call 的代码留在
// mir_builder.rs（那部分对内建调用和普通调用是共用的，拆出来意义不大）。

/// 尝试把一次零参裸调用识别成内建关联常量的引用（比如 `i128::MAX`）。
/// 命中返回这个常量的规范名字（用来构造 MirPlace::Static），命中不了
/// （不满足零参裸调用的形状，或者名字根本不是已知常量）返回 None，
/// 调用方按普通调用继续处理——常量引用语法上就是一个裸名字，不可能
/// 同时是方法调用或带参数，所以先拿这两个条件筛一遍。
pub fn try_resolve_constant_ref(
    qualifier: &Option<String>,
    func: &str,
    is_method: bool,
    arg_count: usize,
) -> Option<&'static str> {
    if qualifier.is_some() || is_method || arg_count != 0 {
        return None;
    }
    let name = IntrinsicConst::from_str(func)?;
    get_constant(name).map(|c| c.name)
}

/// 校验并识别一次调用是不是真正的内建/固有函数。
/// - 不是内建函数（用户自定义函数/普通调用）→ `Ok((false, None))`，
///   调用方按普通函数调用继续处理。
/// - 是内建函数但触发了限制（model 块里用了不允许的副作用函数、
///   unsafe 函数在 unsafe 块外被调用）→ `Err(...)`。
/// - 是内建函数且检查通过 → `Ok((true, Some(名字)))`。
///
/// `in_forward`/`unsafe_depth` 是 MirBuilder 自己的构建期状态，这两条
/// 检查天生依赖它们，所以按值传进来，而不是把 MirBuilder 整个传进来
/// ——这个文件不需要认识 MirBuilder 长什么样。`expr_diverges` 同理，
/// 是调用方已经从 HIR 表达式的 `ty` 字段（`Type::Never`）算好的结果，
/// 传一个 bool 进来即可，不用为了这一条 debug 断言让这个文件认识
/// hir.rs 的类型。
pub fn resolve_intrinsic_call(
    qualifier: &Option<String>,
    func: &str,
    is_method: bool,
    in_forward: bool,
    unsafe_depth: usize,
    expr_diverges: bool,
) -> Result<(bool, Option<IntrinsicFn>), String> {
    if qualifier.is_some() || is_method {
        return Ok((false, None));
    }
    let name = match IntrinsicFn::from_str(func) {
        Some(n) => n,
        None => return Ok((false, None)),
    };
    let intrinsic = match get_intrinsic(name) {
        Some(i) => i,
        None => return Ok((false, None)),
    };

    if in_forward && !intrinsic.allowed_in_model {
        return Err(format!(
            "error[MD001]: side-effect `{}` is not allowed in model block (forward method)",
            func
        ));
    }
    if intrinsic.requires_unsafe && unsafe_depth == 0 {
        return Err(format!(
            "call to unsafe intrinsic `{}` requires an `unsafe` block or `verify unsafe`",
            func
        ));
    }

    // 交叉校验：sema 给这个调用表达式算出来的发散性和 intrinsic 注册表
    // 自己声明的签名，理论上必须给出一致的答案（比如 panic() 应该是
    // Type::Never，跟 Intrinsic::diverges() 一致）。只在 debug 构建里
    // 检查，正常运行零开销。
    debug_assert_eq!(
        expr_diverges,
        intrinsic.diverges(),
        "intrinsic `{}` 的签名声明的发散性跟 sema 算出的表达式类型对不上",
        func
    );

    Ok((true, Some(name)))
}

// ==================== 内建函数类型检查（sema 侧使用） ====================
//
// 跟上面 resolve_intrinsic_call 那一段是同一个 IntrinsicFn，但服务的
// 是完全不同的编译阶段：resolve_intrinsic_call 是 mir_builder.rs 在
// "降到 MIR"这一步用的（判断这是不是内建调用、要不要拦下副作用/
// unsafe 违规）；这里是 semantic/check_expr.rs 在"类型检查"这一步用的
// （每个内建函数的参数长什么样、接收者必须是什么类型、结果类型该
// 推导成什么）。
//
// 纯函数：不依赖 TypeChecker，不调用 sema 方法——check_expr.rs 只把
// "各参数已经检查出来的类型"传进来，这里只管按每个内建函数自己的
// 规则做形状/类型推导，不倒回去问 self.xxx。唯一不能纯函数化的是
// tensor.cond/tensor.while_loop：它们的闭包参数需要 sema 用接收者的
// 类型（input_ty/init_ty）做上下文去递归检查闭包体，这一步天然离不开
// TypeChecker（check_closure 是 TypeChecker 的方法），所以这两个走
// is_dynamic_intrinsic 单独分支，参数布局的提取（哪个位置/哪个具名
// 参数对应 input/condition/then/else）仍然放在这个文件里，跟其它
// 内建函数的参数布局知识放在一起。

/// 一次内建函数调用的上下文：参数表达式本身，和调用方已经
/// check_call_arg 过一遍算出来的类型（顺序一一对应）。
pub struct CallCtx<'a> {
    pub args: &'a [CallArg],
    pub arg_types: &'a [Type],
}

/// 静态内建函数的检查结果。
pub enum CheckedResult {
    /// 直接用这个类型作为结果，不需要调用方再套接收者的隐私标签
    /// （print → Unit、panic → Never、from_utf8_unchecked → &str，
    /// 这几个的结果类型跟接收者的隐私标签无关）。
    Plain(Type),
    /// 结果类型已经剥掉了隐私标签的"基础形态"，调用方需要从
    /// `ctx.arg_types[0]`（接收者）提取隐私标签，套回这个类型上再
    /// 返回——embedding/linear/conv2d/max_pool2d/flatten/reshape/
    /// relu/dropout/layer_norm/sum 都是"接收者带什么隐私标签，结果
    /// 就带什么隐私标签"，这条规则统一由调用方处理，这里只管算出
    /// 不带标签的那部分类型。
    Receiver(Type),
}

/// 哪些内建函数是"动态"的——参数里带闭包、需要 sema 用接收者类型做
/// 上下文递归检查闭包体，走 check_tensor_dynamic_call 那条单独路径，
/// 不能塞进这个文件的纯函数 check_intrinsic_call 里。
pub fn is_dynamic_intrinsic(name: IntrinsicFn) -> bool {
    match name {
        IntrinsicFn::TensorCond | IntrinsicFn::TensorWhileLoop => true,
        _ => false,
    }
}

/// 静态内建函数的类型检查总入口。调用方（check_expr.rs 的
/// check_static_intrinsic_call）已经过滤掉了 is_dynamic_intrinsic
/// 为 true 的两个，这里的 match 对它们只放一个 unreachable 兜底。
pub fn check_intrinsic_call(name: IntrinsicFn, ctx: &CallCtx) -> Result<CheckedResult, String> {
    match name {
        IntrinsicFn::Print => check_print(ctx),
        IntrinsicFn::Panic => check_panic(ctx),
        IntrinsicFn::FromUtf8Unchecked => check_from_utf8_unchecked(ctx),
        IntrinsicFn::TensorCond | IntrinsicFn::TensorWhileLoop => {
            unreachable!("dynamic intrinsic 走 check_tensor_dynamic_call，不走 check_intrinsic_call")
        }
        IntrinsicFn::Embedding => check_embedding(ctx),
        IntrinsicFn::Linear => check_linear(ctx),
        IntrinsicFn::Conv2d => check_conv2d(ctx),
        IntrinsicFn::MaxPool2d => check_max_pool2d(ctx),
        IntrinsicFn::Flatten => check_flatten(ctx),
        IntrinsicFn::Reshape => check_reshape(ctx),
        IntrinsicFn::Relu | IntrinsicFn::Dropout | IntrinsicFn::LayerNorm => check_passthrough(ctx),
        IntrinsicFn::Sum => check_sum(ctx),
    }
}

// ===== 本文件内部的纯函数版本 strip_privacy/is_str_type =====
// 跟 semantic/privacy.rs 的 TypeChecker::strip_privacy 逻辑完全一样，
// 但这里不能借 &self 用那个方法——这个文件的设计前提就是不依赖
// TypeChecker，所以本地另存一份。两边如果以后要改隐私标签的剥离
// 规则（比如加新的 Type 变体需要一并剥离），要记得两处一起改。

fn strip_privacy_type(ty: &Type) -> Type {
    match ty {
        Type::Privacy(inner, _) => strip_privacy_type(inner),
        Type::Ref { mutable, inner } => Type::Ref {
            mutable: *mutable,
            inner: Box::new(strip_privacy_type(inner)),
        },
        _ => ty.clone(),
    }
}

fn is_str_type(ty: &Type) -> bool {
    strip_privacy_type(ty) == Type::Str
}

fn check_print(ctx: &CallCtx) -> Result<CheckedResult, String> {
    // print 对参数类型没有限制；调用方（check_static_intrinsic_call）
    // 已经把所有参数都 check_call_arg 过一遍，这里不需要再检查什么。
    // model 块内的副作用限制也已经在调用方通过 Intrinsic::
    // allowed_in_model 统一处理，不用在这里重复判断 in_model。
    let _ = ctx;
    Ok(CheckedResult::Plain(Type::Unit))
}

fn check_panic(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.len() != 1 {
        return Err("`panic` expects exactly 1 argument (a message)".to_string());
    }
    if !is_str_type(&ctx.arg_types[0]) {
        return Err(format!(
            "`panic` expects a string argument, got {:?}",
            ctx.arg_types[0]
        ));
    }
    Ok(CheckedResult::Plain(Type::Never))
}

fn check_from_utf8_unchecked(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.len() != 1 {
        return Err("`from_utf8_unchecked` expects exactly 1 argument".to_string());
    }
    let expected_arg_ty = Type::Ref {
        mutable: false,
        inner: Box::new(Type::Slice(Box::new(Type::U8))),
    };
    if strip_privacy_type(&ctx.arg_types[0]) != expected_arg_ty {
        return Err(format!(
            "`from_utf8_unchecked` expects &[u8], got {:?}",
            ctx.arg_types[0]
        ));
    }
    Ok(CheckedResult::Plain(Type::Ref {
        mutable: false,
        inner: Box::new(Type::Str),
    }))
}

fn check_embedding(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.len() < 2 {
        return Err("embedding requires at least two arguments".to_string());
    }
    // num_embeddings 目前只用于校验"确实传了这个具名参数、而且是
    // 常量整数"，形状推算用不上它的值。
    let _num_embeddings = extract_int_arg(ctx.args, "num_embeddings")?;
    let embedding_dim = extract_int_arg(ctx.args, "embedding_dim")?;
    // 关键修复（P1-8）：embedding_dim 下面会被 `as usize` 用来扩展
    // 形状——负数在这一步会静默变成一个巨大的正数（比如 -1 在 64 位
    // 平台变成 usize::MAX），产出一个荒谬的张量形状，而不是报出
    // "你传了个负数"这种一眼能看懂的错误。
    if embedding_dim <= 0 {
        return Err(format!(
            "embedding `embedding_dim` must be positive, got {}",
            embedding_dim
        ));
    }
    match strip_privacy_type(&ctx.arg_types[0]) {
        Type::Tensor { shape, .. } => {
            let mut new_shape = shape.clone();
            new_shape.push(ShapeDim::Const(embedding_dim as usize));
            Ok(CheckedResult::Receiver(Type::Tensor {
                dtype: Box::new(Type::F32),
                shape: new_shape,
            }))
        }
        _ => Err(format!("embedding expects a tensor, got {:?}", ctx.arg_types[0])),
    }
}

fn check_linear(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.len() < 2 {
        return Err("linear requires at least two arguments".to_string());
    }
    let in_val = extract_int_arg(ctx.args, "in")?;
    let out_val = extract_int_arg(ctx.args, "out")?;
    // 关键修复（P1-8）：同 embedding_dim，in_val/out_val 会被
    // `as usize` 转换用来核对/改写形状维度。
    if in_val <= 0 {
        return Err(format!("linear `in` must be positive, got {}", in_val));
    }
    if out_val <= 0 {
        return Err(format!("linear `out` must be positive, got {}", out_val));
    }
    match strip_privacy_type(&ctx.arg_types[0]) {
        Type::Tensor { dtype, shape } => {
            if let Some(last) = shape.last() {
                if let ShapeDim::Const(c) = last {
                    if *c != in_val as usize {
                        return Err(format!(
                            "shape mismatch in linear: input last dim {} does not match 'in' value {}",
                            c, in_val
                        ));
                    }
                }
                let mut new_shape = shape.clone();
                if let Some(last) = new_shape.last_mut() {
                    *last = ShapeDim::Const(out_val as usize);
                } else {
                    return Err("tensor must have at least one dimension".to_string());
                }
                Ok(CheckedResult::Receiver(Type::Tensor { dtype, shape: new_shape }))
            } else {
                Err("tensor must have at least one dimension for linear".to_string())
            }
        }
        _ => Err(format!("linear expects a tensor, got {:?}", ctx.arg_types[0])),
    }
}

fn check_conv2d(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.len() < 2 {
        return Err("conv2d requires at least 2 arguments".to_string());
    }
    let (dtype, shape) = match strip_privacy_type(&ctx.arg_types[0]) {
        Type::Tensor { dtype, shape } => (dtype, shape),
        _ => return Err(format!("conv2d expects a tensor, got {:?}", ctx.arg_types[0])),
    };
    if shape.len() != 3 && shape.len() != 4 {
        return Err("conv2d expects 3D [C, H, W] or 4D [B, C, H, W] tensor".to_string());
    }
    let (c_idx, h_idx, w_idx) = if shape.len() == 4 { (1, 2, 3) } else { (0, 1, 2) };

    let _in_channels = extract_int_arg(ctx.args, "in")?;
    let out_channels = extract_int_arg(ctx.args, "out")?;
    let kernel = extract_int_arg(ctx.args, "kernel")?;
    let stride = extract_int_arg(ctx.args, "stride").unwrap_or(1);
    let padding = extract_int_arg(ctx.args, "padding").unwrap_or(0);

    // 关键修复（P1-8）：这四个值后面都会参与 `as usize` 转换或除法
    // 运算，负数/零会静默产出荒谬结果（usize 溢出、除以零 panic）
    // 而不是报出一条看得懂的错误。
    if out_channels <= 0 {
        return Err(format!("conv2d `out` must be positive, got {}", out_channels));
    }
    if kernel <= 0 {
        return Err(format!("conv2d `kernel` must be positive, got {}", kernel));
    }
    if stride <= 0 {
        return Err(format!("conv2d `stride` must be positive, got {}", stride));
    }
    if padding < 0 {
        return Err(format!("conv2d `padding` must be non-negative, got {}", padding));
    }

    let h_out = match &shape[h_idx] {
        ShapeDim::Const(h) => {
            let h = *h as i64;
            let h_out = (h + 2 * padding - kernel) / stride + 1;
            if h_out <= 0 {
                return Err("conv2d output height non-positive".to_string());
            }
            ShapeDim::Const(h_out as usize)
        }
        _ => ShapeDim::Dyn,
    };
    let w_out = match &shape[w_idx] {
        ShapeDim::Const(w) => {
            let w = *w as i64;
            let w_out = (w + 2 * padding - kernel) / stride + 1;
            if w_out <= 0 {
                return Err("conv2d output width non-positive".to_string());
            }
            ShapeDim::Const(w_out as usize)
        }
        _ => ShapeDim::Dyn,
    };

    let mut new_shape = shape.clone();
    new_shape[c_idx] = ShapeDim::Const(out_channels as usize);
    new_shape[h_idx] = h_out;
    new_shape[w_idx] = w_out;

    Ok(CheckedResult::Receiver(Type::Tensor { dtype, shape: new_shape }))
}

fn check_max_pool2d(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.is_empty() {
        return Err("max_pool2d requires at least 1 argument".to_string());
    }
    let kernel = extract_int_arg(ctx.args, "kernel")?;
    if kernel <= 0 {
        return Err(format!("max_pool2d `kernel` must be positive, got {}", kernel));
    }
    let stride = extract_int_arg(ctx.args, "stride").unwrap_or(kernel);
    if stride <= 0 {
        return Err(format!("max_pool2d `stride` must be positive, got {}", stride));
    }
    match strip_privacy_type(&ctx.arg_types[0]) {
        Type::Tensor { dtype, shape } => {
            if shape.len() != 3 && shape.len() != 4 {
                return Err("max_pool2d expects 3D or 4D tensor".to_string());
            }
            let (h_idx, w_idx) = if shape.len() == 4 { (2, 3) } else { (1, 2) };

            let h_out = match &shape[h_idx] {
                ShapeDim::Const(h) => {
                    let h = *h as i64;
                    let h_out = (h - kernel) / stride + 1;
                    if h_out <= 0 {
                        return Err("max_pool2d output height non-positive".to_string());
                    }
                    ShapeDim::Const(h_out as usize)
                }
                _ => ShapeDim::Dyn,
            };
            let w_out = match &shape[w_idx] {
                ShapeDim::Const(w) => {
                    let w = *w as i64;
                    let w_out = (w - kernel) / stride + 1;
                    if w_out <= 0 {
                        return Err("max_pool2d output width non-positive".to_string());
                    }
                    ShapeDim::Const(w_out as usize)
                }
                _ => ShapeDim::Dyn,
            };
            let mut new_shape = shape.clone();
            new_shape[h_idx] = h_out;
            new_shape[w_idx] = w_out;
            Ok(CheckedResult::Receiver(Type::Tensor { dtype, shape: new_shape }))
        }
        _ => Err(format!("max_pool2d expects a tensor, got {:?}", ctx.arg_types[0])),
    }
}

fn check_flatten(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.is_empty() {
        return Err("flatten requires at least 1 argument".to_string());
    }
    match strip_privacy_type(&ctx.arg_types[0]) {
        Type::Tensor { dtype, shape } => {
            let mut total = 1usize;
            let mut all_const = true;
            for dim in &shape {
                if let ShapeDim::Const(c) = dim {
                    total *= *c;
                } else {
                    all_const = false;
                    break;
                }
            }
            let result_ty = if all_const {
                Type::Tensor { dtype, shape: vec![ShapeDim::Const(total)] }
            } else {
                Type::Tensor { dtype, shape }
            };
            Ok(CheckedResult::Receiver(result_ty))
        }
        _ => Err(format!("flatten expects a tensor, got {:?}", ctx.arg_types[0])),
    }
}

fn check_reshape(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.len() < 2 {
        return Err("reshape requires target shape".to_string());
    }
    let shape_vals = match &ctx.arg_types[1] {
        Type::ConstIntArray(vals) => vals.clone(),
        _ => return Err("reshape expects a constant integer array for shape".to_string()),
    };
    let mut new_shape = Vec::with_capacity(shape_vals.len());
    for v in shape_vals {
        // 关键修复（P1-8）：v 是 i64，直接 `as usize` 在 v 为负数时会
        // 静默变成一个巨大的正数，产出荒谬的目标形状。
        if v < 0 {
            return Err(format!(
                "reshape target dimension must be non-negative, got {}",
                v
            ));
        }
        new_shape.push(ShapeDim::Const(v as usize));
    }
    let dtype = match strip_privacy_type(&ctx.arg_types[0]) {
        Type::Tensor { dtype, .. } => dtype,
        _ => return Err("reshape expects a tensor".to_string()),
    };
    Ok(CheckedResult::Receiver(Type::Tensor { dtype, shape: new_shape }))
}

fn check_passthrough(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.is_empty() {
        return Err("passthrough intrinsic requires at least 1 argument".to_string());
    }
    // 关键修复（P1-5）：以前这三个函数（relu/dropout/layer_norm）
    // 直接把参数类型原样返回，不检查它到底是不是张量——`relu(42)`
    // 会被静默放行，返回 I32，这个 I32 混进后面的张量运算链条后，
    // 报错的位置会跟真实错误原因（一开始就不该传非张量进来）完全
    // 对不上。
    let stripped = strip_privacy_type(&ctx.arg_types[0]);
    match stripped {
        Type::Tensor { .. } => Ok(CheckedResult::Receiver(stripped)),
        _ => Err(format!("expects a tensor, got {:?}", ctx.arg_types[0])),
    }
}

fn check_sum(ctx: &CallCtx) -> Result<CheckedResult, String> {
    if ctx.args.is_empty() {
        return Err("sum requires at least 1 argument".to_string());
    }
    match strip_privacy_type(&ctx.arg_types[0]) {
        // 关键修复（P1-5，顺带发现的另一个问题）：以前这里完全不看
        // 参数类型，直接返回 Type::F32，而且用的是不套隐私标签的
        // 路径——一个 dp(1/1) 张量求和之后会静默变成不带隐私标签的
        // 公开 F32，等于凭空抹掉了隐私保护。这在一门把隐私标签当
        // 一等公民的语言里是相当危险的疏漏。现在既检查接收者必须是
        // 张量，又改用 Receiver 而不是 Plain，让调用方按跟其它张量
        // 算子一样的规则把原来的隐私标签套回结果上。
        Type::Tensor { .. } => Ok(CheckedResult::Receiver(Type::F32)),
        _ => Err(format!("sum expects a tensor, got {:?}", ctx.arg_types[0])),
    }
}

// ==================== 从 helpers.rs 搬过来的纯函数 ====================
// 这三个原来是 TypeChecker 的方法（&self），但函数体从来没有用过
// self 的任何字段——只是因为历史上跟其它需要 &self 的方法写在同一个
// impl 块里，才带着一个用不上的 &self。它们的调用方现在也大多是这个
// 文件里的 check_xxx（extract_int_arg），搬过来之后不用再绕一层
// self.xxx(...)。check_expr.rs 里仅剩的一处调用点（ArrayLiteral 分支）
// 改成 `crate::intrinsic::eval_const_int_expr(...)`。

pub(crate) fn eval_const_int_expr(expr: &Expr) -> Option<i64> {
    match &expr.kind {
        ExprKind::Literal(Literal::Int8(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::Int16(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::Int32(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::Int64(v)) => Some(*v),
        ExprKind::Literal(Literal::Int128(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::UInt8(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::UInt16(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::UInt32(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::UInt64(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::UInt128(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::Isize(v)) => Some(*v as i64),
        ExprKind::Literal(Literal::Usize(v)) => Some(*v as i64),
        ExprKind::Unary { op: UnaryOp::Neg, expr } => eval_const_int_expr(expr).map(|v| -v),
        ExprKind::BinaryOp { op, left, right } => {
            let l = eval_const_int_expr(left)?;
            let r = eval_const_int_expr(right)?;
            match op {
                BinaryOp::Add => Some(l + r),
                BinaryOp::Sub => Some(l - r),
                BinaryOp::Mul => Some(l * r),
                BinaryOp::Div => {
                    if r == 0 { None } else { Some(l / r) }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

pub(crate) fn extract_int_arg(args: &[CallArg], name: &str) -> Result<i64, String> {
    for arg in args {
        if let CallArg::Named(n, expr) = arg {
            if n == name {
                return eval_const_int_expr(expr).ok_or_else(|| {
                    format!("parameter '{}' is not a constant integer expression", name)
                });
            }
        }
    }
    Err(format!("parameter '{}' not found", name))
}

pub(crate) fn get_arg_by_pos_or_name<'a>(
    args: &'a [CallArg],
    pos: usize,
    name: &str,
) -> Result<&'a Expr, String> {
    if let Some(arg) = args.get(pos) {
        match arg {
            CallArg::Positional(e) => return Ok(e),
            CallArg::Named(n, e) => {
                if n == name {
                    return Ok(e);
                }
            }
        }
    }
    for arg in args {
        if let CallArg::Named(n, e) = arg {
            if n == name {
                return Ok(e);
            }
        }
    }
    Err(format!(
        "argument '{}' not found (tried position {} and name '{}')",
        name, pos + 1, name
    ))
}

// ===== 动态内建函数的参数布局 =====
// tensor.cond / tensor.while_loop 的闭包参数需要 sema 拿接收者类型
// 做上下文递归检查，参数本身的提取（哪个位置/具名参数对应
// input/condition/then/else）不需要 TypeChecker，跟其它内建函数的
// 参数布局知识放在同一个文件里。

pub struct TensorCondArgs<'a> {
    pub input: &'a Expr,
    pub condition: &'a Expr,
    pub then_expr: &'a Expr,
    pub else_expr: &'a Expr,
}

pub fn extract_tensor_cond_args<'a>(args: &'a [CallArg]) -> Result<TensorCondArgs<'a>, String> {
    Ok(TensorCondArgs {
        input: get_arg_by_pos_or_name(args, 0, "input")?,
        condition: get_arg_by_pos_or_name(args, 1, "condition")?,
        then_expr: get_arg_by_pos_or_name(args, 2, "then")?,
        else_expr: get_arg_by_pos_or_name(args, 3, "else")?,
    })
}

pub struct TensorWhileLoopArgs<'a> {
    pub init: &'a Expr,
    pub cond: &'a Expr,
    pub body: &'a Expr,
}

pub fn extract_tensor_while_loop_args<'a>(
    args: &'a [CallArg],
) -> Result<TensorWhileLoopArgs<'a>, String> {
    Ok(TensorWhileLoopArgs {
        init: get_arg_by_pos_or_name(args, 0, "init")?,
        cond: get_arg_by_pos_or_name(args, 1, "cond")?,
        body: get_arg_by_pos_or_name(args, 2, "body")?,
    })
}