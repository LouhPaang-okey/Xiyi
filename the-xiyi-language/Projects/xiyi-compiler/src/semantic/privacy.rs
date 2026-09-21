// src/semantic/privacy.rs
use crate::ast::*;
use super::check_program::TypeChecker;

impl TypeChecker {
    // 关键修复：rational.rs 那几个比较函数不再吞错、改成返回
    // Result<_, RationalError> 之后，这里所有间接依赖它们的函数
    // （privacy_tags_equal / join_privacy_tags / join_privacy_labels /
    // types_equal_with_privacy）也必须跟着变成 Result——不然唯一的
    // 出路又是在这一层用 unwrap_or 把错误吞掉，等于把问题从 rational.rs
    // 搬到这里，没有真正解决。sema 上层（check_expr.rs/check_func.rs
    // 等调用方）统一只关心 Result<_, String>，不需要知道 rational.rs
    // 内部具体的 RationalError 变体，所以在这一层把 RationalError 转成
    // 带原始字符串（方便定位是哪个字面量出的问题）的 String，往上只
    // 暴露 String。
    pub fn types_equal_with_privacy(&self, a: &Type, b: &Type) -> Result<bool, String> {
        match (a, b) {
            (Type::Privacy(inner1, tag1), Type::Privacy(inner2, tag2)) => {
                Ok(self.types_equal(inner1, inner2) && self.privacy_tags_equal(tag1, tag2)?)
            }
            (Type::Privacy(inner, _), other) => Ok(self.types_equal(inner, other)),
            (other, Type::Privacy(inner, _)) => Ok(self.types_equal(other, inner)),
            _ => Ok(self.types_equal(a, b)),
        }
    }

    pub fn privacy_tags_equal(&self, a: &PrivacyTag, b: &PrivacyTag) -> Result<bool, String> {
        match (a, b) {
            (PrivacyTag::Public, PrivacyTag::Public) => Ok(true),
            (PrivacyTag::Private, PrivacyTag::Private) => Ok(true),
            (PrivacyTag::Differential { eps: e1, delta: d1 }, PrivacyTag::Differential { eps: e2, delta: d2 }) => {
                let eps_eq = Self::rational_eq(e1, e2)
                    .map_err(|err| format!("invalid differential eps `{}`/`{}`: {:?}", e1, e2, err))?;
                let delta_eq = match (d1, d2) {
                    (Some(d1), Some(d2)) => Self::rational_eq(d1, d2)
                        .map_err(|err| format!("invalid differential delta `{}`/`{}`: {:?}", d1, d2, err))?,
                    (None, None) => true,
                    _ => false,
                };
                Ok(eps_eq && delta_eq)
            }
            _ => Ok(false),
        }
    }

    pub fn join_privacy_tags(&self, a: &PrivacyTag, b: &PrivacyTag) -> Result<PrivacyTag, String> {
        match (a, b) {
            (PrivacyTag::Public, x) => Ok(x.clone()),
            (x, PrivacyTag::Public) => Ok(x.clone()),
            (PrivacyTag::Private, _) => Ok(PrivacyTag::Private),
            (_, PrivacyTag::Private) => Ok(PrivacyTag::Private),
            (PrivacyTag::Differential { eps: e1, delta: d1 }, PrivacyTag::Differential { eps: e2, delta: d2 }) => {
                let eps = Self::rational_min(e1, e2)
                    .map_err(|err| format!("invalid differential eps `{}`/`{}`: {:?}", e1, e2, err))?;
                let delta = match (d1, d2) {
                    (Some(d1), Some(d2)) => {
                        let le = Self::rational_le(d1, d2)
                            .map_err(|err| format!("invalid differential delta `{}`/`{}`: {:?}", d1, d2, err))?;
                        Some(if le { d2.clone() } else { d1.clone() })
                    }
                    (Some(d), None) | (None, Some(d)) => Some(d.clone()),
                    (None, None) => None,
                };
                Ok(PrivacyTag::Differential { eps, delta })
            }
        }
    }

    pub fn extract_privacy_tag(&self, ty: &Type) -> Option<PrivacyTag> {
        match ty {
            Type::Privacy(_, tag) => Some(tag.clone()),
            Type::Ref { inner, .. } => self.extract_privacy_tag(inner),
            _ => None,
        }
    }

    pub fn apply_privacy_tag(&self, ty: Type, tag: Option<PrivacyTag>) -> Type {
        match tag {
            Some(t) => Type::Privacy(Box::new(ty), t),
            None => ty,
        }
    }

    pub fn join_privacy_labels(&self, a: &Type, b: &Type) -> Result<Option<PrivacyTag>, String> {
        let tag_a = self.extract_privacy_tag(a);
        let tag_b = self.extract_privacy_tag(b);
        match (tag_a, tag_b) {
            (Some(t1), Some(t2)) => Ok(Some(self.join_privacy_tags(&t1, &t2)?)),
            (Some(t), None) => Ok(Some(t)),
            (None, Some(t)) => Ok(Some(t)),
            (None, None) => Ok(None),
        }
    }

    pub fn strip_privacy(&self, ty: &Type) -> Type {
        match ty {
            Type::Privacy(inner, _) => self.strip_privacy(inner),
            Type::Ref { mutable, inner } => Type::Ref {
                mutable: *mutable,
                inner: Box::new(self.strip_privacy(inner)),
            },
            _ => ty.clone(),
        }
    }
}
