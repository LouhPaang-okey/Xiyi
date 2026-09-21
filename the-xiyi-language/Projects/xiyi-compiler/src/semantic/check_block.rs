// src/semantic/check_block.rs
use crate::ast::*;
use super::check_program::TypeChecker;

impl TypeChecker {
    pub fn check_block(&mut self, block: &Block) -> Result<Type, String> {
        self.check_block_with_expected(block, None)
    }

    // ===== check_block 的"带期望类型提示"版本 =====
    // 目前只有 check_func 会传 expected（函数声明的返回类型），用来让
    // block 最后一句是 Ok(...)/Err(...) 这类裸枚举变体构造、或结构体初始化
    // 时，能正确推导出泛型参数，而不是留一个没绑定的 TypeParam 卡在那儿
    // 跟声明的返回类型对不上。中间的语句该怎么检查还怎么检查，只有最后
    // 一条、且是不带分号的尾随表达式（Stmt::ExprStmt）时才用这个提示。
    pub fn check_block_with_expected(&mut self, block: &Block, expected: Option<&Type>) -> Result<Type, String> {
        // 关键修复：空 block（`{}` / `{ }`）语义上是 Unit——循环体一次
        // 都不执行，直接落到这里返回。原来默认值是 Type::I32，会让
        // `fn f() -> Unit { }` 这类写法报"expected Unit, got I32"，
        // 明明什么都没做，却报出一个跟"什么都没做"这件事完全不相关
        // 的类型错误。Unit 才是"这个 block 什么值都没产出"的自然默认。
        let mut last_type = Type::Unit;
        let last_index = block.stmts.len().checked_sub(1);
        for (i, stmt) in block.stmts.iter().enumerate() {
            if Some(i) == last_index {
                if let Stmt::ExprStmt(expr) = stmt {
                    last_type = self.check_expr_with_expected(expr, expected)?;
                    continue;
                }
            }
            last_type = self.check_stmt(stmt)?;
        }
        Ok(last_type)
    }
}
