// src/syntactic/parse_pattern.rs
use crate::ast::*;
use crate::token::Token;
use super::Parser;

impl Parser {
    // ===== parse_pattern（支持绑定） =====
    pub(crate) fn parse_pattern(&mut self) -> Result<Pattern, String> {
        if let Some((Token::Ident, name)) = self.peek() {
            if name == "_" {
                self.next();
                return Ok(Pattern::Wildcard);
            }
        }

        let name = self.parse_ident()?;
        match self.peek() {
            Some((Token::PathSep, _)) => {
                self.next();
                let variant_name = self.parse_ident()?;

                // ===== 检查是否带绑定：Enum::Variant(binding) =====
                if let Some((Token::LParen, _)) = self.peek() {
                    self.next(); // consume '('
                    let binding = self.parse_ident()?;
                    self.expect(Token::RParen)?;
                    return Ok(Pattern::EnumVariantWithBinding {
                        enum_name: name,
                        variant_name,
                        binding,
                    });
                }

                Ok(Pattern::EnumVariant {
                    enum_name: name,
                    variant_name,
                })
            }
            Some((Token::Colon, _)) => Err("Unexpected ':' in pattern, expected '::'".to_string()),
            _ => Err("expected '::' after enum name in pattern".to_string()),
        }
    }
}
