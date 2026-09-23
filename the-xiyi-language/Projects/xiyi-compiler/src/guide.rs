// guide.rs
//
// 从 mir_builder.rs 拆出来的一部分：把一个 HIR 表达式解析成"能被赋值/
// 取地址的位置"（MirPlace）——变量、字段、索引。跟 build_expr（求值一
// 个表达式、拿到它的*值*）是相反方向的问题："这个表达式指的是哪个
// 可写的位置"，只有 Assign 的左边、以及取地址/取字段/取下标这几种
// 场景会用到，分开之后 build_expr/build_expr_rvalue 那种"只关心怎么
// 求值"的代码不用跟这部分夹在一起看。

use crate::ast::Type;
use crate::hir::*;
use crate::mir::*;
use crate::mir_builder::{propagated, Diverging, MirBuilder, SharedContext};

impl MirBuilder {
    // -------- Place（左值） --------
    // 关键重构：build_place 现在也走跟 build_expr/build_expr_rvalue/
    // build_block 同一套 Diverging<T> 传播——下标表达式本身可能发散
    // （`arr[panic()] = 5;`），这里跟别处一样，见到 Diverged 就不再
    // 往下求值，原样传播给调用方（目前唯一的调用方是 mir_builder.rs
    // 的 HirStmt::Assign）。项目里不用宏，`propagated` 是
    // mir_builder.rs 里的一个普通 pub(crate) 函数（负责把
    // `Diverging<T>` 分类成 `Ok(v)`/`Err(())`），不是宏，这里直接
    // `use` 过来跟那边用同一套写法：`match propagated(...) { Ok(v) =>
    // v, Err(()) => return Ok(Diverging::Diverged) }`。
    pub(crate) fn build_place(&mut self, expr: &HirExpr, shared: &SharedContext) -> Result<Diverging<MirPlace>, String> {
        match &expr.kind {
            HirExprKind::Ident(name) => {
                let id = self.lookup(name).ok_or_else(|| format!("undefined variable `{}`", name))?;
                Ok(Diverging::Value(MirPlace::Ssa(self.current_ssa(id))))
            }
            HirExprKind::FieldAccess { struct_expr, field_name } => {
                let base = match propagated(self.build_place(struct_expr, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                Ok(Diverging::Value(MirPlace::Field { base: Box::new(base), field: field_name.clone() }))
            }
            HirExprKind::Index { expr: base, index } => {
                let base_place = match propagated(self.build_place(base, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                let index_operand = match propagated(self.build_expr(index, shared)?) {
                    Ok(v) => v,
                    Err(()) => return Ok(Diverging::Diverged),
                };
                Ok(Diverging::Value(MirPlace::Index { base: Box::new(base_place), index: Box::new(index_operand) }))
            }
            _ => Err(format!(
                "internal error: {:?} is not a valid assignment target \
                 (sema.rs 的 is_assignable 应该已经拦住了这种情况，走到这里说明两边检查不一致)",
                expr.kind
            )),
        }
    }

    /// 从 FieldAccess 的 struct_expr 自身携带的类型信息（sema 阶段已经
    /// 解析好），查这个结构体真正定义里的哪个字段名（目前 build_place
    /// 没有用到这个查找结果——MirPlace::Field 只存字段名字符串，类型
    /// 由 codegen 需要时再去 MirProgram.structs 里查，不在 Place 上
    /// 冗余缓存一份，避免跟结构体定义各说各话）。这个函数先保留，等
    /// codegen 真正需要"构建期就确定字段类型"的场景时再启用。
    #[allow(dead_code)]
    pub(crate) fn lookup_field_type(&self, struct_expr: &HirExpr, field_name: &str, shared: &SharedContext) -> Type {
        let struct_name = match &struct_expr.ty {
            Type::Struct(name) => name.clone(),
            Type::Generic(name, _) => name.clone(),
            Type::Ref { inner, .. } => match inner.as_ref() {
                Type::Struct(name) => name.clone(),
                Type::Generic(name, _) => name.clone(),
                _ => return Type::Unit,
            },
            _ => return Type::Unit,
        };
        shared
            .struct_fields
            .get(&struct_name)
            .and_then(|fields| fields.get(field_name))
            .cloned()
            .unwrap_or(Type::Unit)
    }
}
