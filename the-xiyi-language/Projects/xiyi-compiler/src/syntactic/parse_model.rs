// src/syntactic/parse_model.rs
use crate::ast::*;
use crate::token::Token;
use super::Parser;

impl Parser {
    // 原名 parse_model_def，按要求改名为 parse_model。
    pub(crate) fn parse_model(&mut self, attributes: Vec<Attribute>) -> Result<ModelDef, String> {
        self.expect(Token::Model)?;
        let name = self.parse_ident()?;
        let generic_params = if let Some((Token::Lt, _)) = self.peek() {
            self.parse_generic_params()?
        } else {
            Vec::new()
        };
        self.expect(Token::LBrace)?;

        let mut fields = Vec::new();
        let mut functions = Vec::new();

        while let Some((token, _)) = self.peek() {
            if *token == Token::RBrace { break; }

            if *token == Token::Var || *token == Token::Let {
                self.next();
                let field_name = self.parse_ident()?;
                self.expect(Token::Colon)?;
                let ty = self.parse_type()?;
                if let Some((Token::Eq, _)) = self.peek() {
                    self.next();
                    self.parse_expr()?;
                }
                self.expect(Token::Semicolon)?;
                fields.push(ModelField { name: field_name, ty });
            } else if let Some((Token::Fn, _)) = self.peek() {
                let fn_attrs = self.parse_attributes()?;
                functions.push(self.parse_func_def(fn_attrs)?);
            } else {
                return Err("Expected field (var/let) or function (fn) definition in model block".to_string());
            }
        }

        self.expect(Token::RBrace)?;
        Ok(ModelDef {
            attributes,
            name,
            generic_params,
            fields,
            functions,
        })
    }
}
