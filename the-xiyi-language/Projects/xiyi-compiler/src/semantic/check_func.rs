// src/semantic/check_func.rs
use std::mem;
use std::collections::HashMap;
use crate::ast::*;
use super::check_program::TypeChecker;

impl TypeChecker {
    // ===== "进入新上下文、退出时恢复"的通用 combinator =====
    // check_func 原来手写"保存旧值 -> 设置新值 -> 跑一段可能失败的
    // 检查 -> 不管成败都先恢复旧值 -> 再决定要不要把错误往上抛"，注释
    // 里专门解释过为什么不能直接用 `?`（会跳过恢复）。这套模式不止
    // current_return_type 一处要用——check_closure 里对 self.scopes
    // 的保存/恢复是同一个模式，抽出来一次写对，别处都不用再手写一遍
    // 、也不用在每处新增的地方重新验证"恢复有没有漏掉某条失败路径"。
    fn with_current_return_type<R>(
        &mut self,
        new_ty: Option<Type>,
        f: impl FnOnce(&mut Self) -> Result<R, String>,
    ) -> Result<R, String> {
        let prev = mem::replace(&mut self.current_return_type, new_ty);
        let result = f(self);
        self.current_return_type = prev;
        result
    }

    // 同上，管的是 self.scopes：整个作用域栈临时替换成只装着给定绑定
    // 的一个新栈（闭包体检查不应该看见外层函数的局部变量），检查完
    // 不管成败都恢复原来的作用域栈。
    fn with_isolated_scope<R>(
        &mut self,
        bindings: Vec<(String, Type)>,
        f: impl FnOnce(&mut Self) -> Result<R, String>,
    ) -> Result<R, String> {
        let old_scopes = mem::take(&mut self.scopes);
        let mut scope = HashMap::new();
        for (name, ty) in bindings {
            scope.insert(name, ty);
        }
        self.scopes.push(scope);
        let result = f(self);
        self.scopes = old_scopes;
        result
    }

    // 原名 check_fn_def，按要求改名为 check_func。
    pub fn check_func(&mut self, fn_def: &FnDef) -> Result<(), String> {
        if self.fn_stack.contains(&fn_def.name) {
            return Err("error[MD002]: recursion not allowed in model block; graph must be topologically sortable".to_string());
        }
        self.fn_stack.push(fn_def.name.clone());

        self.scopes.push(HashMap::new());
        for param in &fn_def.params {
            let ty = if param.name == "self" {
                self.current_self_type.clone().unwrap_or(Type::SelfType)
            } else {
                param.ty.clone()
            };
            self.scopes.last_mut().unwrap().insert(param.name.clone(), ty);
        }

        let body_type = self.with_current_return_type(fn_def.return_type.clone(), |s| {
            s.check_block_with_expected(&fn_def.body, fn_def.return_type.as_ref())
        })?;

        if let Some(expected) = &fn_def.return_type {
            // 关键修复：never（return/break/continue/panic(...) 的类型）
            // 按规范 §6.1 可以强制转换成任意类型——函数体最后一句恰好
            // 是这类发散表达式时（比如 `fn f() -> i32 { panic("boom") }`），
            // 不该被"函数体类型必须匹配声明的返回类型"这条规则拦下来。
            // 这条兼容规则不放进 types_equal 本身（那样会让 Never 在
            // 任何地方都悄悄跟一切类型相等，参见 check_type.rs 的
            // 说明），只在这种"函数体这里确实需要产出一个具体类型值"
            // 的位置显式处理。
            let body_is_never = self.strip_privacy(&body_type) == Type::Never;
            if !body_is_never && !self.types_equal(&body_type, expected) {
                return Err(format!("expected return type {:?}, got {:?}", expected, body_type));
            }
            // 关键修复：types_equal_with_privacy 现在返回 Result（因为
            // 它内部经过 rational.rs 的有理数比较，不再允许吞错），
            // 用 `?` 照常传播。
            if !body_is_never && !self.types_equal_with_privacy(&body_type, expected)? {
                return Err(format!("privacy label mismatch: expected {:?}, got {:?}", expected, body_type));
            }
        }

        // forward 函数带差分隐私参数时，必须同时收一个 &mut
        // TrainingContext——这条检查是 model 领域知识，实现挪到了
        // check_model.rs，这里只是调用一下。
        self.check_forward_dp_requirement(fn_def)?;

        self.scopes.pop();
        self.fn_stack.pop();
        Ok(())
    }

    pub fn check_closure(
        &mut self,
        closure_expr: &Expr,
        param_ty: &Type,
        expected_ret: Option<&Type>,
    ) -> Result<Type, String> {
        let (param_name, body) = match &closure_expr.kind {
            ExprKind::Closure { param, body } => (param, body),
            _ => return Err("Expected a closure".to_string()),
        };

        let ret_ty = self.with_isolated_scope(
            vec![(param_name.clone(), param_ty.clone())],
            |s| s.check_expr(body),
        )?;

        if let Some(expected) = expected_ret {
            // 同 check_func：闭包体是 never（比如 `|t| panic("...")`）时
            // 不该被返回类型不匹配拦下来。
            let ret_is_never = self.strip_privacy(&ret_ty) == Type::Never;
            if !ret_is_never && !self.types_equal_with_privacy(&ret_ty, expected)? {
                return Err(format!(
                    "Closure return type {:?} does not match expected {:?}",
                    ret_ty, expected
                ));
            }
        }
        Ok(ret_ty)
    }
}
