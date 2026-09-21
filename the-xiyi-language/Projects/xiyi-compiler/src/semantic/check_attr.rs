// src/semantic/check_attr.rs
//
// 属性（`#[name(args...)]`）解析相关的小工具函数。从 check_model.rs 里
// 挖 sensitivity 属性值的那几个 helper 通用化搬过来——原来那几个函数
// （find_sensitivity_attr/find_const_value_in_attr）名字里就带着
// "sensitivity"/"const"，是完全针对这一个属性写死的。以后再加别的属性
// （不管是 model 块还是别的什么地方要用），照抄一遍"找属性名、找
// key-value 参数、把值转成某个具体类型"这三步的话，又是同一类重复，
// 干脆先把这三步收成通用工具，"哪个属性、哪个 key"作为参数传进来，
// 不再是写死在函数名里的一部分——这就是"以绝后患"要解决的问题：不是
// 现在有第二个属性要处理，而是不想在第二个属性真的出现时，又在某个
// check_xxx.rs 里重新写一遍同样的三步。

use crate::ast::*;
use super::check_program::TypeChecker;

impl TypeChecker {
    /// 在一组属性里按名字找一个（比如 `#[sensitivity(...)]` 的
    /// "sensitivity"）。属性名允许重复出现属于语言设计层面的问题，这里
    /// 只返回第一个匹配的——目前没有任何一种属性设计成允许重复标注，
    /// 真出现重复也不是这个查找函数该负责报错的事。
    pub fn find_attr<'a>(attrs: &'a [Attribute], name: &str) -> Option<&'a Attribute> {
        attrs.iter().find(|a| a.name == name)
    }

    /// 在一个属性的参数列表里找 `key = value` 形式的某个 key 对应的值
    /// （比如 `#[sensitivity(const = "1.5")]` 里的 "const"）。
    pub fn find_key_value_arg<'a>(attr: &'a Attribute, key: &str) -> Option<&'a AttributeArg> {
        for arg in &attr.args {
            if let AttributeArg::KeyValue(k, val) = arg {
                if k == key {
                    return Some(val);
                }
            }
        }
        None
    }

    /// 把一个 `AttributeArg::Rational`（有理数字符串，比如 "3/2"、
    /// "1.5"）转成 f64。真正的字符串解析在 rational.rs 的
    /// parse_rational 里，这里只是把结果从 Rational 转成 f64，不重新
    /// 写一遍解析逻辑。
    ///
    /// 关键修复：parse_rational 现在返回 Result 而不是 Option（不再
    /// 允许吞错，见 rational.rs），这里跟着改成 Result<f64, String>——
    /// 不是这个变体、或者字符串解析失败，都要把原因如实报出去，不能
    /// 再像原来那样一律返回 None，让调用方分不清"这根本不是有理数
    /// 参数"和"是有理数参数但写错了"这两种情况。
    pub fn rational_arg_to_f64(arg: &AttributeArg) -> Result<f64, String> {
        if let AttributeArg::Rational(r) = arg {
            let parsed = Self::parse_rational(r)
                .map_err(|err| format!("invalid rational literal `{}`: {:?}", r, err))?;
            Ok(parsed.num as f64 / parsed.den as f64)
        } else {
            Err("expected rational argument".to_string())
        }
    }
}
