// src/syntactic/parse_attr.rs
use crate::ast::*;
use crate::token::Token;
use super::Parser;

impl Parser {
    // ===== 属性解析 =====
    pub(crate) fn parse_attributes(&mut self) -> Result<Vec<Attribute>, String> {
        let mut attrs = Vec::new();
        while let Some((Token::Pound, _)) = self.peek() {
            self.next();
            self.expect(Token::LBracket)?;
            let name = self.parse_ident()?;
            let mut args = Vec::new();
            if let Some((Token::LParen, _)) = self.peek() {
                self.next();
                while let Some((token, _)) = self.peek() {
                    if *token == Token::RParen { break; }
                    args.push(self.parse_attribute_arg()?);
                    match self.peek() {
                        Some((Token::Comma, _)) => { self.next(); }
                        _ => break,
                    }
                }
                self.expect(Token::RParen)?;
            }
            self.expect(Token::RBracket)?;
            attrs.push(Attribute { name, args });
        }
        Ok(attrs)
    }

    pub(crate) fn parse_attribute_arg(&mut self) -> Result<AttributeArg, String> {
        if let Some((_, key)) = self.peek() {
            let key = key.clone();
            if let Some((Token::Eq, _)) = self.peek_nth(1) {
                self.next();
                self.next();
                let value = self.parse_attribute_arg_value()?;
                return Ok(AttributeArg::KeyValue(key, Box::new(value)));
            }
        }
        self.parse_attribute_arg_value()
    }

    pub(crate) fn parse_attribute_arg_value(&mut self) -> Result<AttributeArg, String> {
        match self.peek() {
            Some((Token::Ident, v)) => {
                let v = v.clone();
                self.next();
                Ok(AttributeArg::Ident(v))
            }
            Some((Token::String, v)) => {
                let v = v.clone();
                self.next();
                Ok(AttributeArg::StringLit(v))
            }
            Some((Token::Integer, v)) => {
                let int_part = v.clone();
                if let Some((Token::Slash, _)) = self.peek_nth(1) {
                    self.next();
                    self.next();
                    if let Some((Token::Integer, den)) = self.next() {
                        let rational = format!("{}/{}", int_part, den);
                        return Ok(AttributeArg::Rational(rational));
                    } else {
                        return Err("Expected integer after '/' in rational".to_string());
                    }
                } else {
                    self.next();
                    // 词法层只保证这一串字符"看起来像整数"，不保证真的
                    // parse 得出来（比如位数超出目标整数类型能表示的
                    // 范围）。unwrap 会让这种情况直接 panic 掉整个编译器
                    // 进程，改成如实报错。
                    let num = int_part.parse().map_err(|_| {
                        format!("Invalid integer attribute value: {}", int_part)
                    })?;
                    Ok(AttributeArg::Int(num))
                }
            }
            Some((Token::Float, v)) => {
                let v = v.clone();
                self.next();
                let num = v.parse().map_err(|_| {
                    format!("Invalid float attribute value: {}", v)
                })?;
                Ok(AttributeArg::Float(num))
            }
            _ => Err("Expected attribute argument value".to_string()),
        }
    }
}
