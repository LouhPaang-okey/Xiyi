// src/syntactic/parse_type.rs
use crate::ast::*;
use crate::token::Token;
use super::Parser;

impl Parser {
    // ===== 类型解析 =====
    pub(crate) fn parse_type(&mut self) -> Result<Type, String> {
        if let Some((Token::Amp, _)) = self.peek() {
            self.next();
            let mutable = if let Some((Token::Mut, _)) = self.peek() {
                self.next();
                true
            } else {
                false
            };
            // &[T] / &mut [T] 切片类型
            if let Some((Token::LBracket, _)) = self.peek() {
                self.next();
                let elem_ty = self.parse_type()?;
                self.expect(Token::RBracket)?;
                return Ok(Type::Ref {
                    mutable,
                    inner: Box::new(Type::Slice(Box::new(elem_ty))),
                });
            }
            let inner = Box::new(self.parse_type()?);
            return Ok(Type::Ref { mutable, inner });
        }
        let base = self.parse_base_type()?;
        // 关键修复：不再由调用方先 peek 一次 '<'、函数内部再 peek 一次——
        // 现在 parse_privacy_tag 自己开头就是 self.expect(Token::Lt)?，
        // 不是 '<' 直接 Err，交给 try_parse 判定失败并原样还原 pos，
        // 不用在这里重复判断一遍。
        if let Some(tag) = self.try_parse(|p| p.parse_privacy_tag()) {
            return Ok(Type::Privacy(Box::new(base), tag));
        }
        Ok(base)
    }

    // ===== parse_base_type（支持泛型参数识别） =====
    pub(crate) fn parse_base_type(&mut self) -> Result<Type, String> {
        // 首先检查 Tensor
        if let Some((Token::Ident, name)) = self.peek() {
            if name == "Tensor" {
                self.next();
                self.expect(Token::Lt)?;
                let dtype = Box::new(self.parse_base_type()?);
                self.expect(Token::Comma)?;
                let shape = self.parse_shape()?;
                self.expect(Token::Gt)?;
                return Ok(Type::Tensor { dtype, shape });
            }
        }
        match self.peek() {
            Some((Token::I8, _)) => { self.next(); Ok(Type::I8) }
            Some((Token::I16, _)) => { self.next(); Ok(Type::I16) }
            Some((Token::I32, _)) => { self.next(); Ok(Type::I32) }
            Some((Token::I64, _)) => { self.next(); Ok(Type::I64) }
            Some((Token::I128, _)) => { self.next(); Ok(Type::I128) }
            Some((Token::U8, _)) => { self.next(); Ok(Type::U8) }
            Some((Token::U16, _)) => { self.next(); Ok(Type::U16) }
            Some((Token::U32, _)) => { self.next(); Ok(Type::U32) }
            Some((Token::U64, _)) => { self.next(); Ok(Type::U64) }
            Some((Token::U128, _)) => { self.next(); Ok(Type::U128) }
            Some((Token::F16, _)) => { self.next(); Ok(Type::F16) }
            Some((Token::F32, _)) => { self.next(); Ok(Type::F32) }
            Some((Token::F64, _)) => { self.next(); Ok(Type::F64) }
            Some((Token::Bool, _)) => { self.next(); Ok(Type::Bool) }
            Some((Token::Char, _)) => { self.next(); Ok(Type::Char) }
            Some((Token::Str, _)) => { self.next(); Ok(Type::Str) }
            Some((Token::Never, _)) => { self.next(); Ok(Type::Never) }
            Some((Token::SelfType, _)) => { self.next(); Ok(Type::SelfType) }
            Some((Token::LParen, _)) => {
                self.next(); // consume '('
                if let Some((Token::RParen, _)) = self.peek() {
                    self.next();
                    Ok(Type::Unit)
                } else {
                    Err("Tuple types not supported yet".to_string())
                }
            }
            Some((Token::Ident, name)) => {
                let name_clone = name.clone();
                self.next();
                if let Some((Token::Lt, _)) = self.peek() {
                    // 泛型类型实例化：Option<T> → Type::Generic
                    self.next();
                    let mut args = Vec::new();
                    while let Some((token, _)) = self.peek() {
                        if *token == Token::Gt { break; }
                        let ty = self.parse_base_type()?;
                        args.push(ty);
                        if let Some((Token::Comma, _)) = self.peek() {
                            self.next();
                        } else if let Some((Token::Gt, _)) = self.peek() {
                            break;
                        } else {
                            return Err("Expected ',' or '>' in generic type".to_string());
                        }
                    }
                    self.expect(Token::Gt)?;
                    Ok(Type::Generic(name_clone, args))
                } else {
                    // ===== 判断是否是泛型参数 =====
                    // 不靠"名字是不是 T/U/E/K/V"这种硬编码猜测——改成查真正
                    // 的作用域栈：当前站在哪个 fn/struct/enum/implement/interface
                    // 声明内部，这些声明各自把自己的 generic_params 压栈在
                    // push_generic_scope 里，这里只要查名字在不在里面。
                    // 这样任何合法的泛型参数名都认得出来，也不会误伤一个真的
                    // 叫 "T" 的普通 struct（只要它不是在某个声明了 T 作为泛型
                    // 参数的作用域内被引用）。
                    if name_clone == "usize" {
                        Ok(Type::U64)
                    } else if name_clone == "isize" {
                        Ok(Type::I64)
                    } else if self.is_active_generic(&name_clone) {
                        Ok(Type::TypeParam(name_clone))
                    } else {
                        Ok(Type::Struct(name_clone))
                    }
                }
            }
            _ => Err("Expected type".to_string()),
        }
    }

    // ===== 隐私标签 =====
    // 改名去掉 try_ 前缀，同时去掉所有手动 self.pos = pos——回滚统一
    // 交给 module.rs 的 try_parse combinator（调用点见 parse_type 里的
    // `self.try_parse(|p| p.parse_privacy_tag())`）。这个函数现在只管
    // "隐私标签长什么样"：失败了直接 Err/`?`，不用再操心"这条分支
    // 是不是忘了还原 pos"——旧版本 expect(Token::Gt)? 失败时会直接把
    // Err 网上抛，绕过所有手写还原，是真正的 bug；现在这类失败也会
    // 被 try_parse 的唯一还原点接住，不会再漏。
    pub(crate) fn parse_privacy_tag(&mut self) -> Result<PrivacyTag, String> {
        self.expect(Token::Lt)?;
        let tag = match self.peek() {
            Some((Token::Ident, name)) if name == "public" => {
                self.next();
                PrivacyTag::Public
            }
            Some((Token::Ident, name)) if name == "private" => {
                self.next();
                PrivacyTag::Private
            }
            Some((Token::Ident, name)) if name == "dp" => {
                self.next();
                self.expect(Token::LParen)?;
                if let Some((Token::Ident, key)) = self.next() {
                    if key != "eps" {
                        return Err(format!("Expected 'eps', got '{}'", key));
                    }
                } else {
                    return Err("Expected 'eps'".to_string());
                }
                self.expect(Token::Colon)?;
                let eps = self.parse_rational_literal()?;
                let delta = if let Some((Token::Comma, _)) = self.peek() {
                    self.next();
                    if let Some((Token::Ident, key)) = self.next() {
                        if key != "delta" {
                            return Err(format!("Expected 'delta', got '{}'", key));
                        }
                    } else {
                        return Err("Expected 'delta'".to_string());
                    }
                    self.expect(Token::Colon)?;
                    Some(self.parse_rational_literal()?)
                } else {
                    None
                };
                self.expect(Token::RParen)?;
                PrivacyTag::Differential { eps, delta }
            }
            _ => return Err("Expected privacy tag".to_string()),
        };
        self.expect(Token::Gt)?;
        Ok(tag)
    }

    pub(crate) fn parse_rational_literal(&mut self) -> Result<String, String> {
        match self.peek() {
            Some((Token::Integer, v)) => {
                let int_part = v.clone();
                self.next();
                if let Some((Token::Slash, _)) = self.peek() {
                    self.next();
                    if let Some((Token::Integer, den)) = self.next() {
                        Ok(format!("{}/{}", int_part, den))
                    } else {
                        Err("Expected integer after '/'".to_string())
                    }
                } else {
                    Ok(int_part)
                }
            }
            Some((Token::Float, v)) => {
                let v = v.clone();
                self.next();
                Ok(v)
            }
            _ => Err("Expected rational literal".to_string()),
        }
    }

    // ===== 张量形状 =====
    pub(crate) fn parse_shape(&mut self) -> Result<Vec<ShapeDim>, String> {
        self.expect(Token::LBracket)?;
        let mut dims = Vec::new();
        while let Some((token, _)) = self.peek() {
            if *token == Token::RBracket { break; }
            // 这里的 unwrap 是安全的：上面 `while let Some((token, _))
            // = self.peek()` 刚确认过当前位置是 Some，中间没有任何
            // 代码会消费 token，所以 next() 在这里不可能是 None——跟
            // 下面 Token::Integer 分支里"字符串长得像数字但 parse
            // 不出来"这种真正有 panic 风险的 unwrap 不是一回事，不用
            // 跟着改。
            let (token, value) = self.next().unwrap();
            let dim = match token {
                Token::Ident => {
                    if value == "Dyn" {
                        ShapeDim::Dyn
                    } else if value == "Sym" {
                        self.expect(Token::Lt)?;
                        let sym_name = self.parse_ident()?;
                        self.expect(Token::Gt)?;
                        ShapeDim::Sym(sym_name)
                    } else {
                        ShapeDim::Sym(value)
                    }
                }
                Token::Integer => {
                    // 词法层只保证这是"看起来像整数"的 token，不保证它
                    // 能塞进 usize（比如超出范围）。解析失败时如实报错，
                    // 不能 unwrap 让编译器直接 panic。
                    let num = value.parse::<usize>().map_err(|_| {
                        format!("Invalid shape dimension literal: {}", value)
                    })?;
                    ShapeDim::Const(num)
                }
                _ => return Err("Expected dimension in shape".to_string()),
            };
            dims.push(dim);
            match self.peek() {
                Some((Token::Comma, _)) => {
                    self.next();
                    if let Some((Token::RBracket, _)) = self.peek() { break; }
                }
                Some((Token::RBracket, _)) => break,
                _ => return Err("Expected comma or closing bracket in shape".to_string()),
            }
        }
        self.expect(Token::RBracket)?;
        Ok(dims)
    }
}
