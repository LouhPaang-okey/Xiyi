// src/syntactic/parse_func.rs
use crate::ast::*;
use crate::token::Token;
use super::Parser;

// parse_func_sig（interface 里的方法声明）和 parse_func_def（真正带函数体
// 的定义）前 90% 完全一样：可选 pub/priv、fn、名字、泛型参数、参数列表、
// 可选返回类型——只有收尾不同（签名收在 `;`，定义收在函数体）。这个
// 结构体就是那段共同前缀解析出来的结果，两个函数各自拿到它之后只需要
// 再补自己那半段，不用整段复制一遍。
struct FnHeader {
    name: String,
    generic_params: Vec<GenericParam>,
    params: Vec<Param>,
    return_type: Option<Type>,
}

impl Parser {
    // ===== parse_func_sig / parse_func_def 共享的头部解析 =====
    // 注意：泛型作用域在这里被 push，但故意不在这个函数结尾 pop——
    // 签名的 `;` 不需要作用域，但函数体（parse_func_def 那边）需要，
    // 调用方各自决定什么时候该弹出去，跟原来两份代码里 pop 的时机
    // 保持一致（签名是拿到 header 后立刻 pop；定义是解析完函数体
    // 再 pop）。
    //
    // 但这就带来一个不对称：如果 header 自己内部（push 之后的
    // LParen/参数列表/RParen/返回类型这几步）解析失败，header 会直接
    // 把 Err 网上抛，调用方根本拿不到 Ok(header)——没有任何人会替这
    // 一层作用域收尾。原来的代码就是这样：push 完之后一路 `?`，中间
    // 任何一步出错，Parser 的 generic_scopes 就会永久多出这一层脏
    // 数据，污染后面所有解析（比如后面某个同名 T 被误认成活跃泛型
    // 参数）。所以这里把 push 之后的部分拆成 parse_func_signature_tail，
    // 用 match 接住结果：失败就自己 pop 再把 Err 传出去，成功则把
    // pop 的责任正常移交给调用方。
    fn parse_func_header(&mut self) -> Result<FnHeader, String> {
        // 可选可见性修饰符
        if let Some((Token::Pub, _)) = self.peek() {
            self.next();
        } else if let Some((Token::Priv, _)) = self.peek() {
            self.next();
        }
        self.expect(Token::Fn)?;
        let name = self.parse_ident()?;
        let generic_params = if let Some((Token::Lt, _)) = self.peek() {
            self.parse_generic_params()?
        } else {
            Vec::new()
        };
        self.push_generic_scope(&generic_params);

        match self.parse_func_signature_tail() {
            Ok((params, return_type)) => Ok(FnHeader {
                name,
                generic_params,
                params,
                return_type,
            }),
            Err(e) => {
                // header 自己失败：没有调用方能替这层作用域收尾，必须
                // 在这里自己弹出去，否则永久泄漏。
                self.pop_generic_scope();
                Err(e)
            }
        }
    }

    // ===== 参数列表 + 可选返回类型（parse_func_header 的可失败部分） =====
    fn parse_func_signature_tail(&mut self) -> Result<(Vec<Param>, Option<Type>), String> {
        self.expect(Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(Token::RParen)?;
        let return_type = if let Some((Token::Arrow, _)) = self.peek() {
            self.next();
            Some(self.parse_type()?)
        } else {
            None
        };
        Ok((params, return_type))
    }

    // 原名 parse_fn_sig，按要求改名为 parse_func_sig。
    // ===== 函数签名（支持可见性修饰符） =====
    pub(crate) fn parse_func_sig(&mut self) -> Result<FnSig, String> {
        let header = self.parse_func_header()?;
        // header 成功返回时，作用域仍然压着（header 只有自己失败时才
        // 会自己弹出）。签名的收尾只是 `;`，不需要作用域，所以在这里
        // 立刻 pop——且 pop 发生在 expect(Semicolon) 之前，不管分号
        // 检查成不成功，这层作用域都已经弹干净了，不会泄漏。
        self.pop_generic_scope();
        self.expect(Token::Semicolon)?;
        Ok(FnSig {
            name: header.name,
            params: header.params,
            return_type: header.return_type,
            generic_params: header.generic_params,
        })
    }

    // 原名 parse_fn_def，按要求改名为 parse_func_def。
    // ===== 函数定义（支持可见性修饰符） =====
    pub(crate) fn parse_func_def(&mut self, attributes: Vec<Attribute>) -> Result<FnDef, String> {
        let header = self.parse_func_header()?;
        // 跟 parse_func_sig 不一样：函数体本身要引用刚声明的泛型参数，
        // 不能在解析函数体之前就把作用域弹掉。但 parse_block() 有可能
        // 失败——如果直接 `let body = self.parse_block()?;`，失败时
        // `?` 会跳过下面的 pop_generic_scope()，让这层作用域永久残留，
        // 这正是这次要修的 bug。改成先接住结果（不管成功失败），
        // 无条件 pop 一次，再决定是把错误继续往上抛还是正常返回。
        let body = self.parse_block();
        self.pop_generic_scope();
        let body = body?;
        Ok(FnDef {
            attributes,
            name: header.name,
            generic_params: header.generic_params,
            params: header.params,
            return_type: header.return_type,
            body,
        })
    }

    // ===== parse_params（支持 &self / &mut self / 裸 self） =====
    pub(crate) fn parse_params(&mut self) -> Result<Vec<Param>, String> {
        let mut params = Vec::new();
        while let Some((token, _)) = self.peek() {
            if *token == Token::RParen { break; }

            // ===== 处理 &mut self / &self =====
            let handled_self = if let Some((Token::Amp, _)) = self.peek() {
                self.next(); // consume '&'
                let is_mut = if let Some((Token::Mut, _)) = self.peek() {
                    self.next();
                    true
                } else {
                    false
                };
                // 检查是否是 self（现在是 Token::SelfLower）
                if let Some((Token::SelfLower, _)) = self.peek() {
                    self.next(); // consume 'self'
                    let ty = Type::Ref {
                        mutable: is_mut,
                        inner: Box::new(Type::SelfType),
                    };
                    params.push(Param {
                        name: "self".to_string(),
                        ty,
                    });
                    if let Some((Token::Comma, _)) = self.peek() {
                        self.next();
                    }
                    true
                } else {
                    return Err("Expected 'self' after '&'".to_string());
                }
            } else if let Some((Token::SelfLower, _)) = self.peek() {
                // ===== 处理裸 self =====
                self.next(); // consume 'self'
                params.push(Param {
                    name: "self".to_string(),
                    ty: Type::SelfType,
                });
                if let Some((Token::Comma, _)) = self.peek() {
                    self.next();
                }
                true
            } else {
                false
            };

            if handled_self {
                continue;
            }

            // ===== 普通参数 =====
            let name = self.parse_ident()?;
            self.expect(Token::Colon)?;
            let ty = self.parse_type()?;
            params.push(Param { name, ty });

            match self.peek() {
                Some((Token::Comma, _)) => { self.next(); continue; }
                _ => break,
            }
        }
        Ok(params)
    }
}
