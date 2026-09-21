// src/semantic/lookup.rs
use std::collections::HashMap;
use crate::ast::*;
use super::check_program::TypeChecker;

// 存一个 implement 块里的某个方法：方法本身的定义，加上这个 implement 块
// 的 target_type（比如 `implement<T> Option<T> { fn is_some(self) -> bool }`
// 里的 `Option<T>`，里面的 T 会是 Type::TypeParam("T")）。调用点要靠
// target_type 把 receiver 的具体类型（比如 Option<i32>）跟方法签名里的
// 类型变量对上号。
//
// 结构体本身要 pub（check_program.rs 的 `methods` 字段类型要点名它），
// 但 fn_def/target_type 这两个字段不需要 pub——真正读写它们的代码
// （register_impl、check_method_call、check_qualified_static_call）
// 都在这个文件里，没有其它文件需要直接摸这两个字段。
#[derive(Clone)]
pub struct MethodInfo {
    pub(super) fn_def: FnDef,
    pub(super) target_type: Type,
}

impl TypeChecker {
    // 从一个类型里提取"用来查 methods 表的 key"——Struct/Enum/Generic 都是
    // 拿类型名本身当 key（Generic 也只取名字，不含泛型实参，因为同一个
    // `implement<T> Option<T>` 要覆盖所有 Option<具体类型>，不分开存）。
    // 其他类型（Tensor、I32 这些内置标量）暂时没有方法表，返回 None，
    // 调用点会退回旧的兜底行为。
    pub fn type_key_for_impl_target(ty: &Type) -> Option<String> {
        match ty {
            Type::Struct(name) => Some(name.clone()),
            Type::Enum(name) => Some(name.clone()),
            Type::Generic(name, _) => Some(name.clone()),
            _ => None,
        }
    }
    
    // ===== 把一个 implement 块登记进方法表 =====
    // 从 check_program.rs 第一遍遍历的 `Item::Implement(imp) => { ... }`
    // 分支搬过来。
    pub fn register_impl(&mut self, imp: &ImplementDef) -> Result<(), String> {
        let Some(key) = Self::type_key_for_impl_target(&imp.target_type) else {
            return Ok(());
        };
        let entry = self.methods.entry(key.clone()).or_insert_with(HashMap::new);
        for fn_def in &imp.functions {
            if entry.contains_key(&fn_def.name) {
                return Err(format!(
                    "error[IMP001]: conflicting implementations of method `{}` for type `{}`",
                    fn_def.name, key
                ));
            }
            entry.insert(
                fn_def.name.clone(),
                MethodInfo {
                    fn_def: fn_def.clone(),
                    target_type: imp.target_type.clone(),
                },
            );
        }
        Ok(())
    }

    // ===== 内建方法表：基础类型（str/整数等）身上"自带"的方法 =====
    // 这些类型是原语，没有对应的 implement 块能被查到（str 用户没法给它
    // implement，标准库也没这么做）——不补这张表，任何调用都会落进"查不到
    // 就默认 I32"的兜底，产出一个几乎总是错的类型。这里不是想做一套完整
    // 的基础类型方法体系，只覆盖标准库已经实际用到、会踩坑的这几个；
    // 以后再冒出新的（比如 i32.to_string()），照这个格式加一行就行。
    pub fn builtin_primitive_method_return_type(&self, receiver: &Type, method: &str) -> Option<Type> {
        match (receiver, method) {
            (Type::Str, "len") => Some(Type::U64),
            (Type::Str, "is_empty") => Some(Type::Bool),
            (Type::Str, "as_bytes") => Some(Type::Ref {
                mutable: false,
                inner: Box::new(Type::Slice(Box::new(Type::U8))),
            }),
            // .abs() 只对有符号数值类型有意义，返回类型跟接收者一致
            (t, "abs") if self.is_signed_numeric_type(t) => Some(t.clone()),
            _ => None,
        }
    }

    // ===== 方法调用：查 methods 表，按方法自己的签名（含泛型）检查 =====
    // 从 check_expr 的 `ExprKind::Call { .. }` 分支里 `if *is_method { ... }`
    // 那一大段搬过来，独立成方法，check_expr.rs 那边只是一句委托调用。
    pub fn check_method_call(&mut self, func: &str, args: &[CallArg]) -> Result<Type, String> {
        if args.is_empty() {
            return Err("method call requires a receiver".to_string());
        }
        let receiver_ty = self.check_call_arg(&args[0])?;
        let receiver_stripped = self.strip_privacy(&receiver_ty);

        // 内建方法表，先查这个，再查 self.methods。str/整数这类基础类型
        // 没有对应的 implement 块，之前查不到就无差别回退成 I32——这正是
        // `s.len()` 被判成 I32、跟 Vec::with_capacity 要求的 U64 对不上号
        // 的根源。这里 strip_ref 是因为 receiver 常常是 &str 这种带引用的
        // 形式，方法调用要先看穿引用查到底层类型（就像 Rust 的自动解引用）。
        let receiver_base = self.strip_ref(&receiver_stripped);
        if let Some(ret_ty) =
            self.builtin_primitive_method_return_type(&receiver_base, func)
        {
            for arg in &args[1..] {
                self.check_call_arg(arg)?;
            }
            return Ok(ret_ty);
        }

        if let Some(key) = Self::type_key_for_impl_target(&receiver_stripped) {
            if let Some(method_info) = self.methods.get(&key).and_then(|m| m.get(func)).cloned() {
                let mut bindings: HashMap<String, Type> = HashMap::new();
                // 先把 receiver 的具体类型（比如 Option<i32>）跟这个
                // implement 块的 target_type（Option<T>）合一，解出
                // T 绑定成了什么，再用这份绑定去对方法自己的参数/
                // 返回类型做替换。
                if !self.unify_type(&receiver_stripped, &method_info.target_type, &mut bindings) {
                    return Err(format!(
                        "receiver type {:?} does not match implement target {:?} for method `{}`",
                        receiver_ty, method_info.target_type, func
                    ));
                }

                let non_self_params: Vec<&Param> = method_info
                    .fn_def
                    .params
                    .iter()
                    .filter(|p| p.name != "self")
                    .collect();
                let extra_args = &args[1..];
                if non_self_params.len() != extra_args.len() {
                    return Err(format!(
                        "method `{}` expects {} argument(s), got {}",
                        func, non_self_params.len(), extra_args.len()
                    ));
                }
                for (param, arg) in non_self_params.iter().zip(extra_args) {
                    let arg_ty = self.check_call_arg(arg)?;
                    if !self.unify_type(&arg_ty, &param.ty, &mut bindings) {
                        return Err(format!(
                            "type mismatch in call to `{}`: parameter `{}` expected {:?}, got {:?}",
                            func, param.name, param.ty, arg_ty
                        ));
                    }
                }

                // 同上：方法没声明返回类型时该是 Unit，不是 I32
                let result_ty = method_info
                    .fn_def
                    .return_type
                    .clone()
                    .map(|ret| self.substitute_type(&ret, &bindings))
                    .unwrap_or(Type::Unit);
                return Ok(result_ty);
            }
        }

        // 查不到已注册的方法（比如 receiver 是内置标量/张量类型，
        // 或者方法确实没被任何 implement 块定义过）——退回旧的
        // 兜底行为，只检查剩余参数、返回 I32，不阻断这类调用。
        for arg in &args[1..] {
            self.check_call_arg(arg)?;
        }
        Ok(Type::I32)
    }

    // ===== 限定路径静态调用：`TypeName::func(args)`，没有 self 接收者 =====
    // 跟方法调用解析逻辑（check_method_call，查 self.methods 表 → unify
    // 参数 → 替换返回类型）共享同一套思路，唯一区别是没有 receiver，
    // 不需要先做"receiver 类型 vs impl 块 target_type"的合一，所有形参
    // 直接按位置对实参。
    pub fn check_qualified_static_call(
        &mut self,
        type_name: &str,
        func_name: &str,
        args: &[CallArg],
        expected: Option<&Type>,
    ) -> Result<Type, String> {
        let method_info = self
            .methods
            .get(type_name)
            .and_then(|m| m.get(func_name))
            .cloned()
            .ok_or_else(|| format!("undefined function `{}::{}`", type_name, func_name))?;

        let mut bindings: HashMap<String, Type> = HashMap::new();

        // 用外部期望类型预置绑定。Vec::new()/Vec::with_capacity() 这类
        // "返回值带泛型参数 T，但参数列表里完全看不到 T"的静态函数，光靠
        // 参数没法推出 T 到底是什么——`String { vec: Vec::new() }` 报的
        // "expected Generic(Vec,[U8]), got Generic(Vec,[TypeParam(T)])"
        // 就是这么来的：T 全程没人告诉它该是 U8。这里从"这个返回值将被
        // 用在什么类型的位置"反推。尽力而为，unify 不上就放着不报错——
        // 真正的类型不匹配，交给外层（比如 check_struct_init 的字段
        // 比较）在真正比较的时候报错，不在这一步抢先报错。
        if let (Some(expected_ty), Some(ret_ty)) = (expected, &method_info.fn_def.return_type) {
            self.unify_type(expected_ty, ret_ty, &mut bindings);
        }

        let params: Vec<&Param> = method_info
            .fn_def
            .params
            .iter()
            .filter(|p| p.name != "self")
            .collect();
        if params.len() != args.len() {
            return Err(format!(
                "`{}::{}` expects {} argument(s), got {}",
                type_name, func_name, params.len(), args.len()
            ));
        }
        for (param, arg) in params.iter().zip(args) {
            let arg_ty = self.check_call_arg(arg)?;
            if !self.unify_type(&arg_ty, &param.ty, &mut bindings) {
                return Err(format!(
                    "type mismatch in call to `{}::{}`: parameter `{}` expected {:?}, got {:?}",
                    type_name, func_name, param.name, param.ty, arg_ty
                ));
            }
        }

        Ok(method_info
            .fn_def
            .return_type
            .clone()
            .map(|ret| self.substitute_type(&ret, &bindings))
            .unwrap_or(Type::Unit))
    }
}
