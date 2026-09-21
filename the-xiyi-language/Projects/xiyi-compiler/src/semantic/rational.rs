// src/semantic/rational.rs
use super::check_program::TypeChecker;

/// 一个精确有理数：num/den，不变量 den > 0（永远是正数，符号完全由
/// num 承担）。对应规范里的 Rational128——这里用 i128 装分子（可以是
/// 负数）、u128 装分母（不变量：非零、非负）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub num: i128,
    pub den: u128, // 不变量：den > 0
}

/// parse_rational 以及依赖它的比较函数的失败原因。以前这些函数遇到
/// 解析失败就用 `unwrap_or((0, 1))` 把"这根本不是一个合法有理数"悄悄
/// 当成"这是 0"处理，等于在编译期把用户写错的隐私预算/敏感度字面量
/// 悄悄判定为相等或有序，而不是报错——这是本次要修的核心问题，所以
/// 这里把失败原因做成一个具体的枚举，不能被随手 unwrap_or 掉。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalError {
    /// 输入是空串（trim 之后）
    Empty,
    /// 不是整数、"N/D"、"N.M" 三种已知形态中的任何一种
    InvalidFormat,
    /// 分子段解析失败（比如 "N/D" 里 N 不是合法整数）
    InvalidNumerator,
    /// 分母段解析失败（比如 "N/D" 里 D 不是合法整数）
    InvalidDenominator,
    /// 分母为 0
    ZeroDenominator,
    /// 小数形态（"N.M"）解析失败
    InvalidDecimal,
    /// 比较时交叉乘法溢出（两个数值差距过大，u128 装不下）
    Overflow,
}

impl TypeChecker {
    /// 解析一个有理数字面量：整数（"3"）、分数（"3/2"）、小数（"0.1"）
    /// 三种形态，跟原来支持的合法输入语义完全一致——只是把"解析不出来"
    /// 从"悄悄返回 None、调用方再悄悄兜底成 0"改成"明确说出到底哪里
    /// 不对"，不允许任何吞错。
    pub fn parse_rational(s: &str) -> Result<Rational, RationalError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(RationalError::Empty);
        }
        if s.contains('/') {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() != 2 {
                return Err(RationalError::InvalidFormat);
            }
            let num = parts[0]
                .trim()
                .parse::<i128>()
                .map_err(|_| RationalError::InvalidNumerator)?;
            let den = parts[1]
                .trim()
                .parse::<u128>()
                .map_err(|_| RationalError::InvalidDenominator)?;
            if den == 0 {
                return Err(RationalError::ZeroDenominator);
            }
            Ok(Rational { num, den })
        } else if let Ok(num) = s.parse::<i128>() {
            Ok(Rational { num, den: 1 })
        } else if let Some((int_part, frac_part)) = s.split_once('.') {
            let sign = if s.starts_with('-') { -1i128 } else { 1i128 };
            let int_val = if int_part.is_empty() || int_part == "-" {
                0
            } else {
                int_part
                    .parse::<i128>()
                    .map_err(|_| RationalError::InvalidDecimal)?
            };
            let frac_str = frac_part.trim_end_matches('0');
            let den = 10u128.pow(frac_str.len() as u32);
            let frac_val = if frac_str.is_empty() {
                0
            } else {
                frac_str
                    .parse::<i128>()
                    .map_err(|_| RationalError::InvalidDecimal)?
            };
            let num = int_val * den as i128 + sign * frac_val;
            Ok(Rational { num, den })
        } else {
            Err(RationalError::InvalidFormat)
        }
    }

    // ===== 安全的交叉比较 =====
    // 两个不变量必须同时满足：
    // 1. 同号负数要反转比较结果——|a| < |b| 但两者都是负数时，a 实际上
    //    比 b 更大（比如 -1 比 -5 大），不能直接拿绝对值的 cmp 结果
    //    当作 a 和 b 的 cmp 结果。
    // 2. 交叉乘法必须在 u128 上用 checked_mul 做，不能把 den（u128）转成
    //    i128 再乘——den 可能超出 i128::MAX（i128::MAX 还不到 u128::MAX
    //    的一半），转换本身就可能截断或变成负数，静默产生错误结果。
    fn cross_cmp(a: &Rational, b: &Rational) -> Result<std::cmp::Ordering, RationalError> {
        use std::cmp::Ordering;
        let a_sign = a.num.signum();
        let b_sign = b.num.signum();
        if a_sign != b_sign {
            // 符号不同（含一边为 0 的情况），谁的符号更大谁就更大，
            // 不需要也不能再比较绝对值。
            return Ok(a_sign.cmp(&b_sign));
        }
        // 走到这里，a_sign == b_sign：要么都是正数，要么都是负数，要么
        // 都恰好是 0（0 的 signum 是 0）。都是 0 时 num 也都是 0，下面
        // 算出的 abs_cmp 必然是 Equal，符号翻不翻转都无所谓。
        let lhs = a.num.unsigned_abs().checked_mul(b.den);
        let rhs = b.num.unsigned_abs().checked_mul(a.den);
        let (l, r) = match (lhs, rhs) {
            (Some(l), Some(r)) => (l, r),
            _ => return Err(RationalError::Overflow),
        };
        let abs_cmp = l.cmp(&r);
        if a_sign < 0 {
            // 两个负数：绝对值更大的那个，实际数值反而更小（-5 < -1）。
            Ok(abs_cmp.reverse())
        } else {
            Ok(abs_cmp)
        }
    }

    fn compare_rational(a: &str, b: &str) -> Result<std::cmp::Ordering, RationalError> {
        let ra = Self::parse_rational(a)?;
        let rb = Self::parse_rational(b)?;
        Self::cross_cmp(&ra, &rb)
    }

    pub fn rational_le(a: &str, b: &str) -> Result<bool, RationalError> {
        Ok(Self::compare_rational(a, b)? != std::cmp::Ordering::Greater)
    }

    pub fn rational_eq(a: &str, b: &str) -> Result<bool, RationalError> {
        Ok(Self::compare_rational(a, b)? == std::cmp::Ordering::Equal)
    }

    pub fn rational_min(a: &str, b: &str) -> Result<String, RationalError> {
        if Self::compare_rational(a, b)? != std::cmp::Ordering::Greater {
            Ok(a.to_string())
        } else {
            Ok(b.to_string())
        }
    }
}
