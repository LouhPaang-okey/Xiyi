// src/semantic/helpers.rs
use std::fs;
use crate::ast::*;
use super::check_program::TypeChecker;

impl TypeChecker {
    pub fn resolve_import(&self, module_path: &str, stdlib_path: &str) -> Result<String, String> {
        let path = format!("{}/xiyi-core/src/{}.xiyi", stdlib_path, module_path);
        if fs::metadata(&path).is_ok() {
            Ok(path)
        } else {
            Err(format!("module not found: {}", module_path))
        }
    }

    pub fn is_compile_time_constant(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Literal(_) => true,
            ExprKind::Sym(_) => true,
            ExprKind::Ident(name) => self.consts.contains_key(name),
            ExprKind::BinaryOp { left, right, .. } => {
                self.is_compile_time_constant(left) && self.is_compile_time_constant(right)
            }
            // 一元负号也可能出现在常量表达式里（比如 -1 as ConstIntArray 元素）
            ExprKind::Unary { op: UnaryOp::Neg, expr } => self.is_compile_time_constant(expr),
            // lack &[T] 规范里明确写着"均为编译期常量"
            ExprKind::LackSlice(_) => true,
            _ => false,
        }
    }

    #[allow(dead_code)]
    pub fn shape_dim_to_i64(&self, dim: &ShapeDim) -> Result<i64, String> {
        match dim {
            ShapeDim::Const(c) => Ok(*c as i64),
            _ => Err("expected constant dimension".to_string()),
        }
    }

    // 关键修复：以前这里还有 eval_const_int_expr / extract_int_arg /
    // get_call_arg_by_pos_or_name 三个方法，签名都是 `&self`，但函数体
    // 从来没用过 self 的任何字段——只是历史上凑巧跟其它需要 &self 的
    // 方法写在同一个 impl 块里，才带着一个用不上的接收者。这三个是
    // "内建函数（linear/conv2d/tensor.cond 等）的参数怎么解析"这类领域
    // 知识，不是 TypeChecker 本身的状态操作，现在跟着内建函数检查逻辑
    // 一起搬进了 intrinsic.rs（作为自由函数，去掉了用不上的 &self），
    // 调用点相应改成 `crate::intrinsic::eval_const_int_expr(...)` 等。
    //
    // 还删掉了 get_arg_type：它跟 check_call_arg 逐字节相同（都是
    // "CallArg::Positional/Named 两种情况都转发给 check_expr"），只留
    // 语义更贴切的 check_call_arg 一个名字，所有调用点已统一改过来。

    pub fn check_call_arg(&mut self, arg: &CallArg) -> Result<Type, String> {
        match arg {
            CallArg::Positional(expr) => self.check_expr(expr),
            CallArg::Named(_, expr) => self.check_expr(expr),
        }
    }
}
