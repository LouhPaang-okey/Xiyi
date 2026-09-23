// calc.rs
// 常量折叠：对字面量执行编译期二元 / 一元运算。
//
// 返回 `None` 表示"无法折叠"（溢出、除零、类型不匹配、运算符不支持）。

use crate::ast::{BinaryOp, Literal, UnaryOp};

pub struct Calc;

impl Calc {
    // ============================================================
    // 对外接口：只负责按类型分发
    // ============================================================

    pub fn eval_binary_op(op: BinaryOp, left: Literal, right: Literal) -> Option<Literal> {
        match (left, right) {
            // 有符号整数
            (Literal::Int8(l), Literal::Int8(r)) => Self::fold_i8(op, l, r),
            (Literal::Int16(l), Literal::Int16(r)) => Self::fold_i16(op, l, r),
            (Literal::Int32(l), Literal::Int32(r)) => Self::fold_i32(op, l, r),
            (Literal::Int64(l), Literal::Int64(r)) => Self::fold_i64(op, l, r),
            (Literal::Int128(l), Literal::Int128(r)) => Self::fold_i128(op, l, r),
            (Literal::Isize(l), Literal::Isize(r)) => Self::fold_isize(op, l, r),

            // 无符号整数
            (Literal::UInt8(l), Literal::UInt8(r)) => Self::fold_u8(op, l, r),
            (Literal::UInt16(l), Literal::UInt16(r)) => Self::fold_u16(op, l, r),
            (Literal::UInt32(l), Literal::UInt32(r)) => Self::fold_u32(op, l, r),
            (Literal::UInt64(l), Literal::UInt64(r)) => Self::fold_u64(op, l, r),
            (Literal::UInt128(l), Literal::UInt128(r)) => Self::fold_u128(op, l, r),
            (Literal::Usize(l), Literal::Usize(r)) => Self::fold_usize(op, l, r),

            // 浮点数（Float16 内部用 f32 存储）
            (Literal::Float16(l), Literal::Float16(r)) => Self::fold_f16(op, l, r),
            (Literal::Float32(l), Literal::Float32(r)) => Self::fold_f32(op, l, r),
            (Literal::Float64(l), Literal::Float64(r)) => Self::fold_f64(op, l, r),

            // 布尔值
            (Literal::Bool(l), Literal::Bool(r)) => Self::fold_bool(op, l, r),

            // 类型不匹配，或暂不支持的类型（Char/String/Unit/ByteString）
            _ => None,
        }
    }

    /// 一元运算常量折叠（`!` 和 `-`）
    pub fn eval_unary_op(op: UnaryOp, operand: Literal) -> Option<Literal> {
        match (op, operand) {
            // 有符号整数取反（MIN 取反溢出 -> None）
            (UnaryOp::Neg, Literal::Int8(v)) => v.checked_neg().map(Literal::Int8),
            (UnaryOp::Neg, Literal::Int16(v)) => v.checked_neg().map(Literal::Int16),
            (UnaryOp::Neg, Literal::Int32(v)) => v.checked_neg().map(Literal::Int32),
            (UnaryOp::Neg, Literal::Int64(v)) => v.checked_neg().map(Literal::Int64),
            (UnaryOp::Neg, Literal::Int128(v)) => v.checked_neg().map(Literal::Int128),
            (UnaryOp::Neg, Literal::Isize(v)) => v.checked_neg().map(Literal::Isize),

            // 浮点数取反
            (UnaryOp::Neg, Literal::Float16(v)) => Some(Literal::Float16(-v)),
            (UnaryOp::Neg, Literal::Float32(v)) => Some(Literal::Float32(-v)),
            (UnaryOp::Neg, Literal::Float64(v)) => Some(Literal::Float64(-v)),

            // 布尔取反
            (UnaryOp::Not, Literal::Bool(v)) => Some(Literal::Bool(!v)),

            // 无符号整数取反等其余组合：不折叠
            _ => None,
        }
    }

    // ============================================================
    // 比较运算：所有可比较类型共用（非比较运算符返回 None）
    // ============================================================

    fn cmp<T: PartialOrd>(op: BinaryOp, l: T, r: T) -> Option<Literal> {
        match op {
            BinaryOp::Eq => Some(Literal::Bool(l == r)),
            BinaryOp::Neq => Some(Literal::Bool(l != r)),
            BinaryOp::Lt => Some(Literal::Bool(l < r)),
            BinaryOp::Gt => Some(Literal::Bool(l > r)),
            BinaryOp::Le => Some(Literal::Bool(l <= r)),
            BinaryOp::Ge => Some(Literal::Bool(l >= r)),
            _ => None,
        }
    }

    // ============================================================
    // 有符号整数：受检算术 + 比较
    // ============================================================

    fn fold_i8(op: BinaryOp, l: i8, r: i8) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Int8),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Int8),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Int8),
            BinaryOp::Div => l.checked_div(r).map(Literal::Int8),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Int8),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_i16(op: BinaryOp, l: i16, r: i16) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Int16),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Int16),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Int16),
            BinaryOp::Div => l.checked_div(r).map(Literal::Int16),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Int16),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_i32(op: BinaryOp, l: i32, r: i32) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Int32),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Int32),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Int32),
            BinaryOp::Div => l.checked_div(r).map(Literal::Int32),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Int32),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_i64(op: BinaryOp, l: i64, r: i64) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Int64),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Int64),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Int64),
            BinaryOp::Div => l.checked_div(r).map(Literal::Int64),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Int64),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_i128(op: BinaryOp, l: i128, r: i128) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Int128),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Int128),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Int128),
            BinaryOp::Div => l.checked_div(r).map(Literal::Int128),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Int128),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_isize(op: BinaryOp, l: isize, r: isize) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Isize),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Isize),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Isize),
            BinaryOp::Div => l.checked_div(r).map(Literal::Isize),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Isize),
            _ => Self::cmp(op, l, r),
        }
    }

    // ============================================================
    // 无符号整数：受检算术 + 比较
    // ============================================================

    fn fold_u8(op: BinaryOp, l: u8, r: u8) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::UInt8),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::UInt8),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::UInt8),
            BinaryOp::Div => l.checked_div(r).map(Literal::UInt8),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::UInt8),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_u16(op: BinaryOp, l: u16, r: u16) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::UInt16),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::UInt16),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::UInt16),
            BinaryOp::Div => l.checked_div(r).map(Literal::UInt16),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::UInt16),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_u32(op: BinaryOp, l: u32, r: u32) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::UInt32),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::UInt32),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::UInt32),
            BinaryOp::Div => l.checked_div(r).map(Literal::UInt32),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::UInt32),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_u64(op: BinaryOp, l: u64, r: u64) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::UInt64),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::UInt64),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::UInt64),
            BinaryOp::Div => l.checked_div(r).map(Literal::UInt64),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::UInt64),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_u128(op: BinaryOp, l: u128, r: u128) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::UInt128),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::UInt128),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::UInt128),
            BinaryOp::Div => l.checked_div(r).map(Literal::UInt128),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::UInt128),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_usize(op: BinaryOp, l: usize, r: usize) -> Option<Literal> {
        match op {
            BinaryOp::Add => l.checked_add(r).map(Literal::Usize),
            BinaryOp::Sub => l.checked_sub(r).map(Literal::Usize),
            BinaryOp::Mul => l.checked_mul(r).map(Literal::Usize),
            BinaryOp::Div => l.checked_div(r).map(Literal::Usize),
            BinaryOp::Mod => l.checked_rem(r).map(Literal::Usize),
            _ => Self::cmp(op, l, r),
        }
    }

    // ============================================================
    // 浮点数：四则 + 比较（浮点不折叠取模）
    // ============================================================

    /// Float16 内部用 f32 存储
    fn fold_f16(op: BinaryOp, l: f32, r: f32) -> Option<Literal> {
        match op {
            BinaryOp::Add => Some(Literal::Float16(l + r)),
            BinaryOp::Sub => Some(Literal::Float16(l - r)),
            BinaryOp::Mul => Some(Literal::Float16(l * r)),
            BinaryOp::Div => Some(Literal::Float16(l / r)),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_f32(op: BinaryOp, l: f32, r: f32) -> Option<Literal> {
        match op {
            BinaryOp::Add => Some(Literal::Float32(l + r)),
            BinaryOp::Sub => Some(Literal::Float32(l - r)),
            BinaryOp::Mul => Some(Literal::Float32(l * r)),
            BinaryOp::Div => Some(Literal::Float32(l / r)),
            _ => Self::cmp(op, l, r),
        }
    }

    fn fold_f64(op: BinaryOp, l: f64, r: f64) -> Option<Literal> {
        match op {
            BinaryOp::Add => Some(Literal::Float64(l + r)),
            BinaryOp::Sub => Some(Literal::Float64(l - r)),
            BinaryOp::Mul => Some(Literal::Float64(l * r)),
            BinaryOp::Div => Some(Literal::Float64(l / r)),
            _ => Self::cmp(op, l, r),
        }
    }

    // ============================================================
    // 布尔值：逻辑 + 相等性（不支持大小比较）
    // ============================================================

    fn fold_bool(op: BinaryOp, l: bool, r: bool) -> Option<Literal> {
        match op {
            BinaryOp::And => Some(Literal::Bool(l && r)),
            BinaryOp::Or => Some(Literal::Bool(l || r)),
            BinaryOp::Eq => Some(Literal::Bool(l == r)),
            BinaryOp::Neq => Some(Literal::Bool(l != r)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin(op: BinaryOp, l: Literal, r: Literal) -> Option<Literal> {
        Calc::eval_binary_op(op, l, r)
    }

    #[test]
    fn int_arithmetic_folds() {
        let r = bin(BinaryOp::Add, Literal::Int32(2), Literal::Int32(3));
        assert!(matches!(r, Some(Literal::Int32(5))));
        let r = bin(BinaryOp::Mod, Literal::UInt64(10), Literal::UInt64(4));
        assert!(matches!(r, Some(Literal::UInt64(2))));
    }

    #[test]
    fn int_overflow_and_div_zero_do_not_fold() {
        assert!(bin(BinaryOp::Add, Literal::Int8(127), Literal::Int8(1)).is_none());
        assert!(bin(BinaryOp::Sub, Literal::UInt8(0), Literal::UInt8(1)).is_none());
        assert!(bin(BinaryOp::Div, Literal::Int32(7), Literal::Int32(0)).is_none());
        assert!(bin(BinaryOp::Mod, Literal::Int32(7), Literal::Int32(0)).is_none());
        assert!(bin(BinaryOp::Div, Literal::Int32(i32::MIN), Literal::Int32(-1)).is_none());
        assert!(bin(BinaryOp::Mod, Literal::Int32(i32::MIN), Literal::Int32(-1)).is_none());
    }

    #[test]
    fn int_comparison_folds() {
        let r = bin(BinaryOp::Lt, Literal::UInt128(1), Literal::UInt128(2));
        assert!(matches!(r, Some(Literal::Bool(true))));
        let r = bin(BinaryOp::Ge, Literal::Isize(-1), Literal::Isize(0));
        assert!(matches!(r, Some(Literal::Bool(false))));
        // 有符号 / 无符号边界值比较
        let r = bin(BinaryOp::Lt, Literal::Int8(-1), Literal::Int8(1));
        assert!(matches!(r, Some(Literal::Bool(true))));
        let r = bin(BinaryOp::Gt, Literal::UInt64(u64::MAX), Literal::UInt64(0));
        assert!(matches!(r, Some(Literal::Bool(true))));
    }

    #[test]
    fn float_ieee_semantics_preserved() {
        let r = bin(BinaryOp::Div, Literal::Float64(1.0), Literal::Float64(0.0));
        assert!(matches!(r, Some(Literal::Float64(v)) if v.is_infinite()));
        let r = bin(BinaryOp::Eq, Literal::Float32(f32::NAN), Literal::Float32(f32::NAN));
        assert!(matches!(r, Some(Literal::Bool(false))));
        let r = bin(BinaryOp::Neq, Literal::Float32(f32::NAN), Literal::Float32(f32::NAN));
        assert!(matches!(r, Some(Literal::Bool(true))));
        // 浮点不折叠取模
        assert!(bin(BinaryOp::Mod, Literal::Float64(5.0), Literal::Float64(2.0)).is_none());
    }

    #[test]
    fn bool_ops() {
        let r = bin(BinaryOp::And, Literal::Bool(true), Literal::Bool(false));
        assert!(matches!(r, Some(Literal::Bool(false))));
        let r = bin(BinaryOp::Neq, Literal::Bool(true), Literal::Bool(false));
        assert!(matches!(r, Some(Literal::Bool(true))));
        assert!(bin(BinaryOp::Lt, Literal::Bool(false), Literal::Bool(true)).is_none());
    }

    #[test]
    fn type_mismatch_does_not_fold() {
        assert!(bin(BinaryOp::Add, Literal::Int32(1), Literal::Int64(1)).is_none());
    }

    #[test]
    fn unary_ops() {
        let r = Calc::eval_unary_op(UnaryOp::Neg, Literal::Int32(5));
        assert!(matches!(r, Some(Literal::Int32(-5))));
        assert!(Calc::eval_unary_op(UnaryOp::Neg, Literal::Int8(i8::MIN)).is_none());
        assert!(Calc::eval_unary_op(UnaryOp::Neg, Literal::UInt8(1)).is_none());
        let r = Calc::eval_unary_op(UnaryOp::Not, Literal::Bool(true));
        assert!(matches!(r, Some(Literal::Bool(false))));
    }
}
