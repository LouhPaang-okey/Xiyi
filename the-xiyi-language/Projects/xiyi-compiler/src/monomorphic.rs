// monomorphic.rs
use crate::mir::*;
use crate::ast::{Type, ShapeDim};
use std::collections::{HashMap, HashSet};

pub struct Monomorphic;

impl Monomorphic {
    pub fn run(mut program: MirProgram) -> MirProgram {
        let all_fns = program.fns.clone();
        let all_structs = program.structs.clone();
        let all_enums = program.enums.clone();

        let mut new_fns = Vec::new();
        let mut generated_fn_names = HashSet::new();
        let mut call_sites = Vec::new();
        let mut struct_instances: Vec<(String, Vec<Type>)> = Vec::new();
        let mut enum_instances: Vec<(String, Vec<Type>)> = Vec::new();

        // 先从所有 MIR 类型位置和构造点收集实例。不能只看 StructInit /
        // EnumVariantConstruction，否则“只出现在字段/参数/返回值里的泛型类型”
        // 会漏掉。
        for f in &all_fns {
            Self::collect_calls(&f.body, &mut call_sites);
            for (_, ty) in &f.params {
                Self::collect_generic_types(ty, &mut struct_instances, &mut enum_instances);
            }
            if let Some(ty) = &f.return_type {
                Self::collect_generic_types(ty, &mut struct_instances, &mut enum_instances);
            }
            for local in &f.body.locals {
                Self::collect_generic_types(&local.ty, &mut struct_instances, &mut enum_instances);
            }
            Self::collect_generic_instances(&f.body, &mut struct_instances, &mut enum_instances);
        }

        // 函数单态化工作队列：处理一个实例后，再扫描它的具体函数体，
        // 发现新的具体泛型调用，直到队列耗尽。
        //
        // call_sites 只增不减。generated_fn_names.insert(mangled) 能挡住
        // "同一个 mangled 名字被重复生成"，但挡不住不同 mangled 名字之间
        // 的无限链——例如一个泛型函数在体内递归调用自己并且每次都把类型
        // 参数包一层（T -> Wrapper<T> -> Wrapper<Wrapper<T>> -> ...），
        // 每次产生的 mangled 名字都是新的，永远不会被 generated_fn_names
        // 挡住，队列会无限增长。这在源语言层面是非法的（违反
        // "单态化后代码规模必须有限"这条前提），但不能指望前面的阶段一定
        // 已经拒绝掉它；这里必须有自己的硬上限，超限就报错退出，而不是
        // 让编译器无限跑下去。
        const MAX_MONOMORPHIZATION_ITERATIONS: usize = 100_000;

        let mut call_index = 0;
        while call_index < call_sites.len() {
            if call_index >= MAX_MONOMORPHIZATION_ITERATIONS {
                panic!(
                    "monomorphization error: exceeded {} iterations while expanding generic function calls; \
                     this usually means unbounded generic recursion (e.g. a generic function that calls itself \
                     with an ever-growing type argument)",
                    MAX_MONOMORPHIZATION_ITERATIONS
                );
            }

            let (func_name, generic_args) = call_sites[call_index].clone();
            call_index += 1;

            // 泛型模板中的调用可能仍然含有外层 TypeParam；这种调用不能
            // 在这里提前实例化，等外层函数本身被具体化后再重新收集。
            if Self::contains_unresolved_type(&generic_args) {
                continue;
            }

            let Some(base_fn) = all_fns.iter().find(|f| f.name == func_name) else {
                continue;
            };

            if base_fn.generic_params.is_empty() {
                panic!(
                    "monomorphization error: function `{}` received {} generic arguments, but is not generic",
                    func_name,
                    generic_args.len()
                );
            }
            if base_fn.generic_params.len() != generic_args.len() {
                panic!(
                    "monomorphization error: generic arity mismatch for `{}`: expected {}, got {}",
                    func_name,
                    base_fn.generic_params.len(),
                    generic_args.len()
                );
            }

            let mangled = Self::mangled_name(&base_fn.name, &generic_args);
            if !generated_fn_names.insert(mangled.clone()) {
                continue;
            }

            let subst: HashMap<String, Type> = base_fn
                .generic_params
                .iter()
                .zip(generic_args.iter())
                .map(|(param, concrete)| (param.name.clone(), concrete.clone()))
                .collect();
            let mut cloned = base_fn.clone();
            cloned.name = mangled;
            cloned.generic_params.clear();
            cloned.params = cloned
                .params
                .into_iter()
                .map(|(name, ty)| (name, Self::replace_type(&ty, &subst)))
                .collect();
            cloned.return_type = cloned.return_type.map(|ty| Self::replace_type(&ty, &subst));
            cloned.body = Self::replace_body(&cloned.body, &subst);

            Self::assert_no_unresolved_types_in_fn(&cloned, "function monomorphization");

            Self::collect_calls(&cloned.body, &mut call_sites);
            Self::collect_generic_instances(&cloned.body, &mut struct_instances, &mut enum_instances);
            for (_, ty) in &cloned.params {
                Self::collect_generic_types(ty, &mut struct_instances, &mut enum_instances);
            }
            if let Some(ty) = &cloned.return_type {
                Self::collect_generic_types(ty, &mut struct_instances, &mut enum_instances);
            }
            for local in &cloned.body.locals {
                Self::collect_generic_types(&local.ty, &mut struct_instances, &mut enum_instances);
            }
            new_fns.push(cloned);
        }

        // 结构体和枚举使用同一个工作队列思想：实例化一个具体类型后，
        // 继续扫描其字段/变体，直到没有新的具体泛型类型出现。
        let mut struct_index = 0;
        let mut enum_index = 0;
        let mut generated_struct_names = HashSet::new();
        let mut generated_enum_names = HashSet::new();
        let mut new_structs = Vec::new();
        let mut new_enums = Vec::new();

        loop {
            let mut progressed = false;

            while struct_index < struct_instances.len() {
                let (struct_name, type_args) = struct_instances[struct_index].clone();
                struct_index += 1;
                if Self::contains_unresolved_type(&type_args) {
                    continue;
                }

                let Some(base_struct) = all_structs.iter().find(|s| s.name == struct_name) else {
                    continue;
                };
                if base_struct.generic_params.is_empty() {
                    continue;
                }
                if base_struct.generic_params.len() != type_args.len() {
                    panic!(
                        "monomorphization error: generic arity mismatch for struct `{}`: expected {}, got {}",
                        struct_name,
                        base_struct.generic_params.len(),
                        type_args.len()
                    );
                }

                let mangled = Self::mangled_name(&base_struct.name, &type_args);
                if !generated_struct_names.insert(mangled.clone()) {
                    continue;
                }

                let subst: HashMap<String, Type> = base_struct
                    .generic_params
                    .iter()
                    .zip(type_args.iter())
                    .map(|(param, concrete)| (param.name.clone(), concrete.clone()))
                    .collect();
                let mut cloned = base_struct.clone();
                cloned.name = mangled;
                cloned.generic_params.clear();
                cloned.fields = cloned
                    .fields
                    .into_iter()
                    .map(|(name, ty)| (name, Self::replace_type(&ty, &subst)))
                    .collect();

                for (_, ty) in &cloned.fields {
                    Self::collect_generic_types(ty, &mut struct_instances, &mut enum_instances);
                }
                Self::assert_no_unresolved_types_in_struct(&cloned, "struct monomorphization");
                new_structs.push(cloned);
                progressed = true;
            }

            while enum_index < enum_instances.len() {
                let (enum_name, type_args) = enum_instances[enum_index].clone();
                enum_index += 1;
                if Self::contains_unresolved_type(&type_args) {
                    continue;
                }

                let Some(base_enum) = all_enums.iter().find(|e| e.name == enum_name) else {
                    continue;
                };
                if base_enum.generic_params.is_empty() {
                    continue;
                }
                if base_enum.generic_params.len() != type_args.len() {
                    panic!(
                        "monomorphization error: generic arity mismatch for enum `{}`: expected {}, got {}",
                        enum_name,
                        base_enum.generic_params.len(),
                        type_args.len()
                    );
                }

                let mangled = Self::mangled_name(&base_enum.name, &type_args);
                if !generated_enum_names.insert(mangled.clone()) {
                    continue;
                }

                let subst: HashMap<String, Type> = base_enum
                    .generic_params
                    .iter()
                    .zip(type_args.iter())
                    .map(|(param, concrete)| (param.name.clone(), concrete.clone()))
                    .collect();
                let mut cloned = base_enum.clone();
                cloned.name = mangled;
                cloned.generic_params.clear();
                cloned.variants = cloned
                    .variants
                    .into_iter()
                    .map(|(name, ty)| {
                        let ty = ty.map(|ty| Self::replace_type(&ty, &subst));
                        if let Some(ref concrete_ty) = ty {
                            Self::collect_generic_types(
                                concrete_ty,
                                &mut struct_instances,
                                &mut enum_instances,
                            );
                        }
                        (name, ty)
                    })
                    .collect();

                Self::assert_no_unresolved_types_in_enum(&cloned, "enum monomorphization");
                new_enums.push(cloned);
                progressed = true;
            }

            if !progressed {
                break;
            }
        }

        // 原始泛型模板不能进入最终 MIR；最终产物必须是具体的。
        for f in all_fns {
            if f.generic_params.is_empty() {
                new_fns.push(f);
            }
        }
        for s in program.structs.drain(..) {
            if s.generic_params.is_empty() {
                new_structs.push(s);
            }
        }
        for e in program.enums.drain(..) {
            if e.generic_params.is_empty() {
                new_enums.push(e);
            }
        }

        let struct_instance_names: HashSet<String> =
            new_structs.iter().map(|s| s.name.clone()).collect();
        let enum_instance_names: HashSet<String> =
            new_enums.iter().map(|e| e.name.clone()).collect();

        for f in &mut new_fns {
            for (_, ty) in f.params.iter_mut() {
                Self::replace_generic_with_concrete(ty, &struct_instance_names, &enum_instance_names);
            }
            if let Some(ty) = f.return_type.as_mut() {
                Self::replace_generic_with_concrete(ty, &struct_instance_names, &enum_instance_names);
            }
            for local in &mut f.body.locals {
                Self::replace_generic_with_concrete(&mut local.ty, &struct_instance_names, &enum_instance_names);
            }
            for block in &mut f.body.blocks {
                for stmt in &mut block.stmts {
                    if let MirStmt::Assign { value, .. } | MirStmt::ExprStmt(value) = stmt {
                        Self::replace_type_in_rvalue(
                            value,
                            &generated_fn_names,
                            &struct_instance_names,
                            &enum_instance_names,
                        );
                    }
                }
                // Switch 的判别式类型也可能是泛型实例，要同样改写成具体名字。
                if let MirTerminator::Switch { discr_ty, .. } = &mut block.terminator {
                    Self::replace_generic_with_concrete(discr_ty, &struct_instance_names, &enum_instance_names);
                }
            }
            Self::assert_no_unresolved_types_in_fn(f, "final monomorphization");
        }

        for s in &mut new_structs {
            for (_, ty) in s.fields.iter_mut() {
                Self::replace_generic_with_concrete(ty, &struct_instance_names, &enum_instance_names);
            }
            Self::assert_no_unresolved_types_in_struct(s, "final monomorphization");
        }
        for e in &mut new_enums {
            for (_, ty_opt) in e.variants.iter_mut() {
                if let Some(ty) = ty_opt {
                    Self::replace_generic_with_concrete(ty, &struct_instance_names, &enum_instance_names);
                }
            }
            Self::assert_no_unresolved_types_in_enum(e, "final monomorphization");
        }

        Self::update_enum_names_in_fns(&mut new_fns);

        MirProgram {
            structs: new_structs,
            enums: new_enums,
            consts: program.consts,
            protos: program.protos,
            interfaces: program.interfaces,
            fns: new_fns,
            intrinsics_used: program.intrinsics_used,
        }
    }

    fn contains_unresolved_type(tys: &[Type]) -> bool {
        tys.iter().any(Self::type_contains_unresolved)
    }

    fn type_contains_unresolved(ty: &Type) -> bool {
        match ty {
            Type::TypeParam(_) | Type::SelfType => true,
            Type::Generic(_, args) => args.iter().any(Self::type_contains_unresolved),
            Type::Tuple(elems) => elems.iter().any(Self::type_contains_unresolved),
            Type::Array(elem, _) => Self::type_contains_unresolved(elem),
            Type::Ref { inner, .. }
            | Type::Privacy(inner, _)
            | Type::Slice(inner) => Self::type_contains_unresolved(inner),
            Type::Tensor { dtype, .. } => Self::type_contains_unresolved(dtype),
            _ => false,
        }
    }

    fn collect_generic_types(
        ty: &Type,
        struct_out: &mut Vec<(String, Vec<Type>)>,
        enum_out: &mut Vec<(String, Vec<Type>)>,
    ) {
        match ty {
            Type::Generic(name, args) => {
                if !Self::contains_unresolved_type(args) {
                    // 这里不提前判断它究竟是 struct 还是 enum；两张表都收集，
                    // 后续按定义表决定实际种类。
                    Self::push_unique_instance(struct_out, name, args);
                    Self::push_unique_instance(enum_out, name, args);
                }
                for arg in args {
                    Self::collect_generic_types(arg, struct_out, enum_out);
                }
            }
            Type::Tuple(elems) => {
                for elem in elems {
                    Self::collect_generic_types(elem, struct_out, enum_out);
                }
            }
            Type::Array(elem, _) => Self::collect_generic_types(elem, struct_out, enum_out),
            Type::Ref { inner, .. }
            | Type::Privacy(inner, _)
            | Type::Slice(inner) => Self::collect_generic_types(inner, struct_out, enum_out),
            Type::Tensor { dtype, .. } => Self::collect_generic_types(dtype, struct_out, enum_out),
            _ => {}
        }
    }

    fn push_unique_instance(
        out: &mut Vec<(String, Vec<Type>)>,
        name: &str,
        args: &[Type],
    ) {
        if !out.iter().any(|(n, a)| n == name && a == args) {
            out.push((name.to_string(), args.to_vec()));
        }
    }

    fn assert_no_unresolved_types_in_fn(f: &MirFn, phase: &str) {
        for (_, ty) in &f.params {
            Self::assert_concrete_type(ty, phase, &format!("function `{}` parameter", f.name));
        }
        if let Some(ty) = &f.return_type {
            Self::assert_concrete_type(ty, phase, &format!("function `{}` return", f.name));
        }
        for local in &f.body.locals {
            Self::assert_concrete_type(&local.ty, phase, &format!("function `{}` local", f.name));
        }
        for block in &f.body.blocks {
            for stmt in &block.stmts {
                match stmt {
                    MirStmt::Assign { value, .. } | MirStmt::ExprStmt(value) => {
                        Self::assert_no_unresolved_types_in_rvalue(value, phase, &f.name);
                    }
                    MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => {}
                }
            }
            // 终止器里的 Switch.discr_ty 同样不能残留类型参数。
            match &block.terminator {
                MirTerminator::Switch { discr_ty, .. } => {
                    Self::assert_concrete_type(discr_ty, phase, &format!("function `{}` switch discriminant", f.name));
                }
                MirTerminator::Goto(_)
                | MirTerminator::If { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable
                | MirTerminator::Placeholder => {}
            }
        }
    }

    fn assert_no_unresolved_types_in_rvalue(rv: &MirRvalue, phase: &str, fn_name: &str) {
        match rv {
            MirRvalue::Cast(_, ty) => {
                Self::assert_concrete_type(ty, phase, &format!("function `{}` cast", fn_name));
            }
            MirRvalue::StructInit { generic_args, .. }
            | MirRvalue::EnumVariantConstruction { generic_args, .. }
            | MirRvalue::Call { generic_args, .. }
            | MirRvalue::MethodCall { generic_args, .. } => {
                for ty in generic_args {
                    Self::assert_concrete_type(
                        ty,
                        phase,
                        &format!("function `{}` generic argument", fn_name),
                    );
                }
            }
            MirRvalue::Use(_)
            | MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. } => {}
        }
    }

    fn assert_no_unresolved_types_in_struct(s: &MirStruct, phase: &str) {
        for (_, ty) in &s.fields {
            Self::assert_concrete_type(ty, phase, &format!("struct `{}` field", s.name));
        }
    }

    fn assert_no_unresolved_types_in_enum(e: &MirEnum, phase: &str) {
        for (_, ty) in &e.variants {
            if let Some(ty) = ty {
                Self::assert_concrete_type(ty, phase, &format!("enum `{}` variant", e.name));
            }
        }
    }

    fn assert_concrete_type(ty: &Type, phase: &str, context: &str) {
        if Self::type_contains_unresolved(ty) {
            panic!(
                "monomorphization error during {}: unresolved generic type in {}: {:?}",
                phase, context, ty
            );
        }
    }

    // ---- 收集所有泛型调用 ----
    fn collect_calls(body: &MirBody, out: &mut Vec<(String, Vec<Type>)>) {
        for block in &body.blocks {
            for stmt in &block.stmts {
                match stmt {
                    MirStmt::Assign { value, .. } | MirStmt::ExprStmt(value) => {
                        Self::collect_rvalue(value, out);
                    }
                    _ => {}
                }
            }
        }
    }

    fn collect_rvalue(rv: &MirRvalue, out: &mut Vec<(String, Vec<Type>)>) {
        match rv {
            MirRvalue::Call { func, generic_args, .. } if !generic_args.is_empty() => {
                out.push((func.clone(), generic_args.clone()));
            }
            MirRvalue::MethodCall { method, generic_args, .. } if !generic_args.is_empty() => {
                // 不再检查 receiver 形式，直接收集
                out.push((method.clone(), generic_args.clone()));
            }
            _ => {}
        }
    }

    // ---- 收集泛型结构体/枚举实例 ----
    // 关键修复：原来这里（collect_struct_rvalue）对着 BinaryOp/UnaryOp/
    // Cast/Call/MethodCall/ArrayLiteral/Ref 这些不含 StructInit 的变体
    // 挨个递归下钻，试图从"复杂表达式内部"挖出嵌套的 StructInit——
    // 但这在这份 MIR 里是不可能发生的：mir.rs 开头的设计说明写得很
    // 清楚，MirRvalue 右边只允许"一层"操作，任何子表达式在构建阶段
    // 就必须先落进一个临时局部变量，再引用那个临时变量。也就是说
    // `Point { x: 1, y: f(g()) }` 这种嵌套，`f(g())` 早就被拆成了
    // 自己的临时变量和自己独立的 Assign 语句，`StructInit` 的
    // `fields` 里出现的永远只是 `MirOperand::Copy/Move(某个已经算好
    // 的临时变量)`，不可能直接嵌一个 `MirRvalue::StructInit`——而
    // `MirOperand` 本身根本没有"内嵌一个 MirRvalue"这个可能性（它只
    // 有 Copy(Place)/Move(Place)/Constant(Literal) 三种）。所以原来
    // 那一整套递归下钻从设计上就打不到任何真实目标，是在解决一个
    // 不存在的问题（并且顺带因为想"通用地处理所有变体"，反而漏掉了
    // 真正需要覆盖的 EnumVariantConstruction.args）。
    //
    // 真正需要做的只是：对每条语句的最外层右值直接判断"这是不是一次
    // 带 generic_args 的 StructInit/EnumVariantConstruction"——跟
    // `collect_rvalue`（收集泛型函数调用的那个，一直是这么写的）用的
    // 是同一个模式，不需要递归。
    fn collect_generic_instances(
        body: &MirBody,
        struct_out: &mut Vec<(String, Vec<Type>)>,
        enum_out: &mut Vec<(String, Vec<Type>)>,
    ) {
        for block in &body.blocks {
            for stmt in &block.stmts {
                let value = match stmt {
                    MirStmt::Assign { value, .. } | MirStmt::ExprStmt(value) => value,
                    MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => continue,
                };
                match value {
                    MirRvalue::StructInit { struct_name, generic_args, .. }
                        if !generic_args.is_empty() =>
                    {
                        if !Self::contains_unresolved_type(generic_args) {
                            Self::push_unique_instance(struct_out, struct_name, generic_args);
                        }
                        for arg in generic_args {
                            Self::collect_generic_types(arg, struct_out, enum_out);
                        }
                    }
                    MirRvalue::EnumVariantConstruction { enum_name, generic_args, .. }
                        if !generic_args.is_empty() =>
                    {
                        if !Self::contains_unresolved_type(generic_args) {
                            Self::push_unique_instance(enum_out, enum_name, generic_args);
                        }
                        for arg in generic_args {
                            Self::collect_generic_types(arg, struct_out, enum_out);
                        }
                    }
                    MirRvalue::Cast(_, ty) => {
                        Self::collect_generic_types(ty, struct_out, enum_out);
                    }
                    MirRvalue::Call { generic_args, .. }
                    | MirRvalue::MethodCall { generic_args, .. } => {
                        for arg in generic_args {
                            Self::collect_generic_types(arg, struct_out, enum_out);
                        }
                    }
                    // 带 generic_args 但为空的 StructInit/EnumVariantConstruction（守卫
                    // 没通过）以及不携带泛型实例的其余右值，没有要收集的东西。
                    MirRvalue::StructInit { .. }
                    | MirRvalue::EnumVariantConstruction { .. }
                    | MirRvalue::Use(_)
                    | MirRvalue::BinaryOp(_, _, _)
                    | MirRvalue::UnaryOp(_, _)
                    | MirRvalue::Ref { .. }
                    | MirRvalue::ArrayLiteral(_)
                    | MirRvalue::Discriminant { .. }
                    | MirRvalue::Phi { .. } => {}
                }
            }
            // Switch 的判别式类型里也可能首次出现某个泛型实例。
            match &block.terminator {
                MirTerminator::Switch { discr_ty, .. } => {
                    Self::collect_generic_types(discr_ty, struct_out, enum_out);
                }
                MirTerminator::Goto(_)
                | MirTerminator::If { .. }
                | MirTerminator::Return(_)
                | MirTerminator::Unreachable
                | MirTerminator::Placeholder => {}
            }
        }
    }

    // ---- 类型替换 ----
    fn replace_type(ty: &Type, subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::TypeParam(name) => subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
            Type::Generic(name, args) => {
                let new_args = args.iter().map(|a| Self::replace_type(a, subst)).collect();
                Type::Generic(name.clone(), new_args)
            }
            Type::Tuple(elems) => {
                let new_elems = elems.iter().map(|e| Self::replace_type(e, subst)).collect();
                Type::Tuple(new_elems)
            }
            Type::Array(elem_ty, len) => {
                Type::Array(Box::new(Self::replace_type(elem_ty, subst)), *len)
            }
            Type::Ref { mutable, inner } => {
                Type::Ref {
                    mutable: *mutable,
                    inner: Box::new(Self::replace_type(inner, subst)),
                }
            }
            Type::Privacy(inner, tag) => {
                Type::Privacy(Box::new(Self::replace_type(inner, subst)), tag.clone())
            }
            Type::Slice(inner) => {
                Type::Slice(Box::new(Self::replace_type(inner, subst)))
            }
            Type::Tensor { dtype, shape } => Type::Tensor {
                dtype: Box::new(Self::replace_type(dtype, subst)),
                shape: shape.clone(), // Sym 不参与类型替换
            },
            _ => ty.clone(),
        }
    }

    // ---- 替换函数体中的所有类型 ----
    fn replace_body(body: &MirBody, subst: &HashMap<String, Type>) -> MirBody {
        let mut new_locals = Vec::new();
        for local in &body.locals {
            let mut new_local = local.clone();
            new_local.ty = Self::replace_type(&local.ty, subst);
            new_locals.push(new_local);
        }

        let mut new_blocks = Vec::new();
        for block in &body.blocks {
            let mut new_stmts = Vec::new();
            for stmt in &block.stmts {
                new_stmts.push(Self::replace_stmt(stmt, subst));
            }
            let mut new_block = block.clone();
            new_block.stmts = new_stmts;
            new_block.terminator = Self::replace_terminator(&block.terminator, subst);
            new_blocks.push(new_block);
        }

        // 关键修复：MirBody 新增了 revision 字段，这里必须带上。块的集合、
        // 编号都没变（只替换了类型），所以原样继承 revision，不自增——
        // 自增只在增删块/重编号时发生（见 MirBody::revision 的说明）。
        MirBody {
            locals: new_locals,
            blocks: new_blocks,
            revision: body.revision,
        }
    }

    fn replace_stmt(stmt: &MirStmt, subst: &HashMap<String, Type>) -> MirStmt {
        match stmt {
            MirStmt::Assign { dest, value } => {
                MirStmt::Assign {
                    dest: dest.clone(), // Place 中的类型在 locals 里已经替换了
                    value: Self::replace_rvalue(value, subst),
                }
            }
            MirStmt::ExprStmt(value) => MirStmt::ExprStmt(Self::replace_rvalue(value, subst)),
            // 这三种不携带需要替换的类型；显式列出，新增语句变体时编译器会提醒。
            MirStmt::Drop { .. } | MirStmt::SetMetadata { .. } | MirStmt::EffectCheck { .. } => stmt.clone(),
        }
    }

    fn replace_rvalue(rv: &MirRvalue, subst: &HashMap<String, Type>) -> MirRvalue {
        match rv {
            MirRvalue::Cast(op, ty) => MirRvalue::Cast(op.clone(), Self::replace_type(ty, subst)),
            MirRvalue::StructInit { struct_name, generic_args, fields } => {
                // 关键修复：generic_args 本身也是类型，函数在把 T 换成
                // 具体类型的这一步里，如果某个 StructInit 的 generic_args
                // 用到了这个函数自己的类型参数（比如 `fn wrap<T>(x: T)
                // { Box { value: x } }` 里 Box 的 generic_args 是
                // `[T]`），这里不把 T 替换成具体类型的话，后面
                // collect_generic_instances 收集到的就还是抽象的 T，
                // 没法据此生成正确的 `Box_i32`。原来这里漏了这一步，
                // 只是简单地把 fields/struct_name 原样搬过去。
                let new_generic_args = generic_args.iter().map(|t| Self::replace_type(t, subst)).collect();
                let new_fields = fields.iter()
                    .map(|(name, op)| (name.clone(), op.clone()))
                    .collect();
                MirRvalue::StructInit {
                    struct_name: struct_name.clone(),
                    generic_args: new_generic_args,
                    fields: new_fields,
                }
            }
            MirRvalue::EnumVariantConstruction { enum_name, generic_args, variant_name, args } => {
                // 同上。
                let new_generic_args = generic_args.iter().map(|t| Self::replace_type(t, subst)).collect();
                MirRvalue::EnumVariantConstruction {
                    enum_name: enum_name.clone(),
                    generic_args: new_generic_args,
                    variant_name: variant_name.clone(),
                    args: args.clone(),
                }
            }
            MirRvalue::Call { func, args, is_intrinsic, intrinsic_name, generic_args } => {
                // 同上：Call 自己的 generic_args 也可能引用外层函数的
                // 类型参数，一并替换——原来这里落在 `_ => rv.clone()`
                // 兜底分支里，完全没替换，等于 identity::<T>(x) 这种
                // 写法在单态化之后 generic_args 里还留着抽象的 T。
                MirRvalue::Call {
                    func: func.clone(),
                    args: args.clone(),
                    is_intrinsic: *is_intrinsic,
                    intrinsic_name: intrinsic_name.clone(),
                    generic_args: generic_args.iter().map(|t| Self::replace_type(t, subst)).collect(),
                }
            }
            MirRvalue::MethodCall { receiver, method, args, generic_args } => {
                MirRvalue::MethodCall {
                    receiver: receiver.clone(),
                    method: method.clone(),
                    args: args.clone(),
                    generic_args: generic_args.iter().map(|t| Self::replace_type(t, subst)).collect(),
                }
            }
            // 关键修复：原来这 7 个变体（Use/BinaryOp/UnaryOp/Ref/
            // ArrayLiteral/Discriminant/Phi）全靠最后一句 `_ => rv.clone()`
            // 兜底——这几个变体确实都没有需要在这里替换的类型（它们只
            // 携带 MirOperand，具体类型信息挂在 MirLocal.ty 上，由
            // replace_body 那边单独整体替换过一轮了，这里不用重复做），
            // 所以行为不用改。但用一个通配符盖住"这 7 个变体不需要处理"
            // 和"以后 MirRvalue 新增变体、这里应该默认克隆"这两件本质不
            // 同的事，等于放弃了穷尽匹配的保护——新变体如果其实需要替换
            // （比如以后加一个直接携带 Type 的变体），编译器不会提醒，
            // 只会在这里悄悄漏做替换。展开成显式分支，效果不变，但以后
            // MirRvalue 加变体时这里会因非穷尽编译不过，逼着回来决定。
            MirRvalue::Use(_)
            | MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::Ref { .. }
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. } => rv.clone(),
        }
    }

    // 终止器里的操作数（If.cond / Switch.discr / Return 的值）是 MirOperand，
    // 不携带类型；但 Switch 自己带着 discr_ty（判别式的类型），它可能引用
    // 函数的类型参数（`match x` 里 x: T 单态化成 Option<T> 时），必须
    // 随函数体一起替换。原来这里整个 clone，discr_ty 里的 TypeParam 会
    // 原样留到 codegen。
    fn replace_terminator(term: &MirTerminator, subst: &HashMap<String, Type>) -> MirTerminator {
        match term {
            MirTerminator::Switch { discr, discr_ty, targets, default } => MirTerminator::Switch {
                discr: discr.clone(),
                discr_ty: Self::replace_type(discr_ty, subst),
                targets: targets.clone(),
                default: *default,
            },
            MirTerminator::Goto(_)
            | MirTerminator::If { .. }
            | MirTerminator::Return(_)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => term.clone(),
        }
    }

    // 关键修复：改名 + 扩展成同时认 struct 和 enum 两套具体实例名字。
    // 原来只查 struct_instances 一张表，`Type::Generic("Option", [I32])`
    // 这种指向泛型枚举的类型永远查不到、永远走不到 `*ty = Type::Struct(..)`
    // 那条改写路径，泛型枚举类型在整个替换阶段完全没有被处理过。
    fn replace_generic_with_concrete(
        ty: &mut Type,
        struct_instances: &HashSet<String>,
        enum_instances: &HashSet<String>,
    ) {
        match ty {
            Type::Generic(name, args) => {
                let candidate = Self::mangled_name(name, args);
                if struct_instances.contains(&candidate) {
                    *ty = Type::Struct(candidate);
                } else if enum_instances.contains(&candidate) {
                    *ty = Type::Enum(candidate);
                } else {
                    for arg in args.iter_mut() {
                        Self::replace_generic_with_concrete(arg, struct_instances, enum_instances);
                    }
                }
            }
            Type::Tuple(elems) => {
                for elem in elems.iter_mut() {
                    Self::replace_generic_with_concrete(elem, struct_instances, enum_instances);
                }
            }
            Type::Array(elem_ty, _) => {
                Self::replace_generic_with_concrete(elem_ty.as_mut(), struct_instances, enum_instances);
            }
            Type::Ref { inner, .. } => {
                Self::replace_generic_with_concrete(inner.as_mut(), struct_instances, enum_instances);
            }
            Type::Privacy(inner, _) => {
                Self::replace_generic_with_concrete(inner.as_mut(), struct_instances, enum_instances);
            }
            Type::Slice(inner) => {
                Self::replace_generic_with_concrete(inner.as_mut(), struct_instances, enum_instances);
            }
            // 张量的元素类型也可能是泛型实例（Tensor<Wrapper<f32>>），
            // 跟 replace_type / collect_generic_types 的覆盖范围对齐。
            Type::Tensor { dtype, .. } => {
                Self::replace_generic_with_concrete(dtype.as_mut(), struct_instances, enum_instances);
            }
            // 其余类型（标量、Struct/Enum 具名类型、TypeParam 等）里没有
            // 嵌套的 Generic 可改写。Type 变体很多，这里的通配就是对它们的
            // 正确回答；新增带子类型的变体时必须在上面补分支。
            _ => {}
        }
    }

    fn replace_type_in_rvalue(
        rv: &mut MirRvalue,
        fn_instances: &HashSet<String>,
        struct_instances: &HashSet<String>,
        enum_instances: &HashSet<String>,
    ) {
        match rv {
            MirRvalue::Cast(_, ty) => {
                Self::replace_generic_with_concrete(ty, struct_instances, enum_instances);
            }
            MirRvalue::Call { func, generic_args, .. } => {
                for ty in generic_args.iter_mut() {
                    Self::replace_generic_with_concrete(ty, struct_instances, enum_instances);
                }
                if !generic_args.is_empty() {
                    let mangled = Self::mangled_name(func, generic_args);
                    if fn_instances.contains(&mangled) {
                        *func = mangled;
                        generic_args.clear();
                    } else if !Self::contains_unresolved_type(generic_args) {
                        panic!(
                            "monomorphization error: generic function instance `{}` was not generated",
                            mangled
                        );
                    }
                }
            }
            MirRvalue::MethodCall { method, generic_args, .. } => {
                for ty in generic_args.iter_mut() {
                    Self::replace_generic_with_concrete(ty, struct_instances, enum_instances);
                }
                if !generic_args.is_empty() {
                    let mangled = Self::mangled_name(method, generic_args);
                    if fn_instances.contains(&mangled) {
                        *method = mangled;
                        generic_args.clear();
                    } else if !Self::contains_unresolved_type(generic_args) {
                        panic!(
                            "monomorphization error: generic method instance `{}` was not generated",
                            mangled
                        );
                    }
                }
            }
            // 关键新增：StructInit/EnumVariantConstruction 原来完全没
            // 在这个函数里出现过（旧代码的注释写着"StructInit 的
            // struct_name 已经在生成时替换，但如果你后续又引入新的，
            // 可能需要处理"——但生成阶段（上面第 3 步）压根没有改写
            // 调用点的 struct_name，这条注释描述的事情从来没发生过）。
            // 这里补上：如果这次 StructInit/EnumVariantConstruction 的
            // generic_args 精确对应一个已经生成好的具体实例，就把
            // struct_name/enum_name 改写成那个具体实例的名字（比如
            // "Point" + [I32] -> "Point_i32"），并清空 generic_args——
            // 清空是为了跟 Call 那边的约定保持一致（Call 的注释里也
            // 提过："单态化之后应该清空，但可能残留"），生成的具体
            // 结构体/枚举名字本身已经不含类型参数了，构造点自然也不
            // 应该再带着一份 generic_args。
            MirRvalue::StructInit { struct_name, generic_args, .. } => {
                if !generic_args.is_empty() {
                    let mangled = Self::mangled_name(struct_name, generic_args);
                    if struct_instances.contains(&mangled) {
                        *struct_name = mangled;
                        generic_args.clear();
                    } else if !Self::contains_unresolved_type(generic_args) {
                        panic!(
                            "monomorphization error: generic struct instance `{}` was not generated",
                            mangled
                        );
                    }
                }
            }
            MirRvalue::EnumVariantConstruction { enum_name, generic_args, .. } => {
                if !generic_args.is_empty() {
                    let mangled = Self::mangled_name(enum_name, generic_args);
                    if enum_instances.contains(&mangled) {
                        *enum_name = mangled;
                        generic_args.clear();
                    } else if !Self::contains_unresolved_type(generic_args) {
                        panic!(
                            "monomorphization error: generic enum instance `{}` was not generated",
                            mangled
                        );
                    }
                }
            }
            MirRvalue::Use(_)
            | MirRvalue::BinaryOp(_, _, _)
            | MirRvalue::UnaryOp(_, _)
            | MirRvalue::ArrayLiteral(_)
            | MirRvalue::Discriminant { .. }
            | MirRvalue::Phi { .. }
            | MirRvalue::Ref { .. } => {}
        }
    }

    // ---- 生成唯一函数名 ----
    fn mangled_name(base: &str, args: &[Type]) -> String {
        let args_str = args.iter()
            .map(Self::type_to_string)
            .collect::<Vec<_>>()
            .join("_");
        format!("{}_{}", base, args_str)
    }

    fn type_to_string(ty: &Type) -> String {
        match ty {
            // 基础类型
            Type::I8 => "i8".to_string(),
            Type::I16 => "i16".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::I128 => "i128".to_string(),
            Type::U8 => "u8".to_string(),
            Type::U16 => "u16".to_string(),
            Type::U32 => "u32".to_string(),
            Type::U64 => "u64".to_string(),
            Type::U128 => "u128".to_string(),
            Type::F16 => "f16".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Char => "char".to_string(),
            Type::Str => "str".to_string(),
            Type::Unit => "unit".to_string(),
            Type::Never => "never".to_string(),
            Type::SymInt => "symint".to_string(),

            // 复合类型
            Type::Struct(name) => name.clone(),
            Type::Enum(name) => name.clone(),
            Type::TypeParam(name) => name.clone(),
            Type::SelfType => "Self".to_string(),

            // 泛型
            Type::Generic(name, args) => {
                let args_str = args.iter()
                    .map(Self::type_to_string)
                    .collect::<Vec<_>>()
                    .join("_");
                format!("{}_{}", name, args_str)
            }

            // 元组
            Type::Tuple(elems) => {
                let elems_str = elems.iter()
                    .map(Self::type_to_string)
                    .collect::<Vec<_>>()
                    .join("_");
                format!("tuple_{}", elems_str)
            }

            // 数组
            Type::Array(elem_ty, len) => {
                format!("arr_{}_{}", Self::type_to_string(elem_ty), len)
            }

            // 引用
            Type::Ref { mutable, inner } => {
                let mut_str = if *mutable { "mut" } else { "immut" };
                format!("ref_{}_{}", mut_str, Self::type_to_string(inner))
            }

            // 切片
            Type::Slice(inner) => {
                format!("slice_{}", Self::type_to_string(inner))
            }

            // 隐私类型：标签必须进名字。原来剥掉标签，`Box<Private<i32>>` 和
            // `Box<i32>` 会得到同一个 mangled 名，两个不同的实例（实例表按完整
            // Type 去重）生成出重名的结构体/函数。Debug 输出里的符号统一压成
            // 下划线，保证是合法标识符片段。
            Type::Privacy(inner, tag) => {
                let tag_str: String = format!("{:?}", tag)
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                format!("priv_{}_{}", tag_str, Self::type_to_string(inner))
            }

            // 张量
            Type::Tensor { dtype, shape } => {
                let dtype_str = Self::type_to_string(dtype);
                let shape_str = shape.iter()
                    .map(|d| match d {
                        ShapeDim::Const(c) => c.to_string(),
                        ShapeDim::Sym(s) => s.clone(),
                        ShapeDim::Dyn => "Dyn".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("_");
                format!("tensor_{}_{}", dtype_str, shape_str)
            }

            // 常量整数数组
            Type::ConstIntArray(vals) => {
                let vals_str = vals.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("_");
                format!("constarr_{}", vals_str)
            }
        }
    }

    // ===== 更新泛型枚举的 enum_name =====
    fn update_enum_names_in_fns(fns: &mut [MirFn]) {
        for f in fns {
            // 关键修复：原来是
            // `update_enum_names_in_body(&mut f.body, &f.body.locals)`
            // ——同一次函数调用里，第一个参数问 `f.body` 整体要一份
            // 独占的可变借用，第二个参数又要对 `f.body.locals`（`f.body`
            // 的一部分）借一份共享借用，两者在这次调用期间同时存活，
            // 违反了"可变借用必须独占"这条规则，编译不过（E0502：
            // cannot borrow `f.body.locals` as immutable because
            // `f.body` is also borrowed as mutable）。
            //
            // 改成直接借 `f.body` 的两个不同字段——`blocks` 要改、
            // `locals` 只读——Rust 允许对同一个结构体的不同字段分别
            // 借用（借用检查是按字段粒度做的，不是整个结构体一刀切），
            // 这两个借用不冲突。
            Self::update_enum_names_in_body(&mut f.body.blocks, &f.body.locals);
        }
    }

    fn update_enum_names_in_body(blocks: &mut [MirBlock], locals: &[MirLocal]) {
        for block in blocks {
            for stmt in &mut block.stmts {
                match stmt {
                    MirStmt::Assign { dest, value } => {
                        Self::update_enum_names_in_place(dest, locals);
                        Self::update_enum_names_in_rvalue(value, locals);
                    }
                    MirStmt::ExprStmt(value) => {
                        Self::update_enum_names_in_rvalue(value, locals);
                    }
                    MirStmt::Drop { place } => {
                        Self::update_enum_names_in_place(place, locals);
                    }
                    MirStmt::SetMetadata { place, .. } => {
                        Self::update_enum_names_in_place(place, locals);
                    }
                    MirStmt::EffectCheck { .. } => {}
                }
            }
            Self::update_enum_names_in_terminator(&mut block.terminator, locals);
        }
    }

    // 终止器里的 cond / discr 是 MirOperand，可以是
    // MirOperand::Copy(MirPlace::EnumPayload { base, enum_name, .. })——
    // 跟语句里的 Rvalue 一样需要把 enum_name 更新成单态化后的具体枚举名，
    // 不能因为 Discriminant 已经在 Rvalue 里处理过就跳过这里。
    fn update_enum_names_in_terminator(term: &mut MirTerminator, locals: &[MirLocal]) {
        match term {
            MirTerminator::If { cond, .. } => Self::update_enum_names_in_operand(cond, locals),
            MirTerminator::Switch { discr, .. } => Self::update_enum_names_in_operand(discr, locals),
            MirTerminator::Return(Some(op)) => Self::update_enum_names_in_operand(op, locals),
            MirTerminator::Goto(_)
            | MirTerminator::Return(None)
            | MirTerminator::Unreachable
            | MirTerminator::Placeholder => {}
        }
    }

    fn update_enum_names_in_rvalue(rv: &mut MirRvalue, locals: &[MirLocal]) {
        match rv {
            MirRvalue::Use(op) => Self::update_enum_names_in_operand(op, locals),
            MirRvalue::BinaryOp(_, l, r) => {
                Self::update_enum_names_in_operand(l, locals);
                Self::update_enum_names_in_operand(r, locals);
            }
            MirRvalue::UnaryOp(_, op) => Self::update_enum_names_in_operand(op, locals),
            MirRvalue::Cast(op, _) => Self::update_enum_names_in_operand(op, locals),
            MirRvalue::Call { args, .. } => {
                for arg in args {
                    Self::update_enum_names_in_operand(arg, locals);
                }
            }
            MirRvalue::MethodCall { receiver, args, .. } => {
                Self::update_enum_names_in_operand(receiver, locals);
                for arg in args {
                    Self::update_enum_names_in_operand(arg, locals);
                }
            }
            MirRvalue::StructInit { fields, .. } => {
                for (_, op) in fields {
                    Self::update_enum_names_in_operand(op, locals);
                }
            }
            MirRvalue::EnumVariantConstruction { args, .. } => {
                for arg in args {
                    Self::update_enum_names_in_operand(arg, locals);
                }
            }
            MirRvalue::ArrayLiteral(elems) => {
                for elem in elems {
                    Self::update_enum_names_in_operand(elem, locals);
                }
            }
            MirRvalue::Ref { place, .. } => {
                Self::update_enum_names_in_place(place, locals);
            }
            MirRvalue::Discriminant { value, enum_name } => {
                if let Some(base_id) = Self::extract_local_id_from_operand(value) {
                    if let Some(ty) = locals.iter().find(|l| l.id == base_id).map(|l| &l.ty) {
                        if let Type::Enum(name) = ty {
                            *enum_name = name.clone();
                        }
                    }
                }
            }
            MirRvalue::Phi { values } => {
                for (_, op) in values {
                    Self::update_enum_names_in_operand(op, locals);
                }
            }
            // 关键修复：原来这里还有一句 `_ => {}` 兜底——但上面这些
            // 分支（Use/BinaryOp/UnaryOp/Cast/Call/MethodCall/
            // StructInit/EnumVariantConstruction/ArrayLiteral/Ref/
            // Discriminant/Phi）已经是 MirRvalue 全部 12 个变体，一个不
            // 落。`_ => {}` 在这里纯属多余，而且是有害的多余——它悄悄
            // 关掉了"以后 mir.rs 给 MirRvalue 新增第 13 个变体时，这里
            // 会因为非穷尽编译不过、逼你回来决定要不要处理"这层保护。
            // 删掉它，让这个 match 恢复真正的穷尽匹配。
        }
    }

    fn update_enum_names_in_operand(op: &mut MirOperand, locals: &[MirLocal]) {
        match op {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                Self::update_enum_names_in_place(place, locals);
            }
            MirOperand::Constant(_) => {}
            // 关键修复（Sym/Static 搬家）：这两个变体从 MirPlace 挪来
            // MirOperand 之后，不包含任何 MirPlace，没有 enum_name 需要
            // 更新，跟 Constant 是同一个道理。
            MirOperand::Sym(_) => {}
            MirOperand::Static(_) => {}
        }
    }

    fn update_enum_names_in_place(place: &mut MirPlace, locals: &[MirLocal]) {
        match place {
            // 关键修复：`MirPlace::Local` 这个变体已经被删掉了（"SSA
            // 完整版，Local 被废弃"），对着一个不存在的枚举变体匹配，
            // 编译不过（E0599）。剩下的 Ssa 分支已经覆盖了"这个变体本身
            // 不包含 enum_name，不需要处理"这件事。
            MirPlace::Ssa(_) => {
                // 这个变体本身不包含 enum_name，不需要处理
            }
            MirPlace::Field { base, .. } => Self::update_enum_names_in_place(base, locals),
            MirPlace::Index { base, index, .. } => {
                Self::update_enum_names_in_place(base, locals);
                Self::update_enum_names_in_operand(index, locals);
            }
            MirPlace::Deref(base) => Self::update_enum_names_in_place(base, locals),
            MirPlace::EnumPayload { base, enum_name, .. } => {
                if let Some(base_id) = Self::extract_local_id_from_place(base) {
                    if let Some(ty) = locals.iter().find(|l| l.id == base_id).map(|l| &l.ty) {
                        if let Type::Enum(name) = ty {
                            *enum_name = name.clone();
                        }
                    }
                }
            }
            // 关键修复（穷尽列出，去掉 `_` 兜底）：上面 Ssa/Field/Index/
            // Deref/EnumPayload 五条分支已经覆盖了 MirPlace 现在的全部
            // 变体（Sym/Static 已经搬去 MirOperand，不会出现在这里），
            // 原来收尾的 `_ => {}` 因此从来不会被真的走到，纯粹是给
            // 已经删除的 Local 变体留的痕迹。删掉它让这个 match 变成
            // 真正的穷尽匹配——以后 mir.rs 给 MirPlace 加新变体时，这里
            // 会因为"non-exhaustive match"编译不过，逼着回来决定这个
            // 新变体要不要携带 enum_name、需不需要递归，而不是被通配符
            // 悄悄吞掉、留下一个"看起来处理了、其实什么都没做"的假象。
        }
    }

    // 辅助：从 MirPlace 中提取底层 local id（如果有）
    fn extract_local_id_from_place(place: &MirPlace) -> Option<usize> {
        match place {
            // 关键修复：同上，`MirPlace::Local` 不存在了，删掉这一支——
            // 下面 `MirPlace::Ssa(ssa) => Some(ssa.base_id)` 已经覆盖了
            // "取出底层变量 id"这件事，不需要另外处理 Local。
            MirPlace::Ssa(ssa) => Some(ssa.base_id),
            MirPlace::Field { base, .. } => Self::extract_local_id_from_place(base),
            MirPlace::Index { base, .. } => Self::extract_local_id_from_place(base),
            MirPlace::Deref(base) => Self::extract_local_id_from_place(base),
            MirPlace::EnumPayload { base, .. } => Self::extract_local_id_from_place(base),
            // 关键修复（穷尽列出，去掉 `_` 兜底）：同
            // update_enum_names_in_place 那处——五条分支已经穷尽了
            // MirPlace 现在的全部变体，原来的 `_ => None` 从来不会被
            // 真的走到，删掉它，让新增变体时的遗漏在这里编译不过，而
            // 不是被悄悄归到 None 里。
        }
    }

    // 辅助：从 MirOperand 中提取 local id
    fn extract_local_id_from_operand(op: &MirOperand) -> Option<usize> {
        match op {
            MirOperand::Copy(place) | MirOperand::Move(place) => Self::extract_local_id_from_place(place),
            _ => None,
        }
    }
}
