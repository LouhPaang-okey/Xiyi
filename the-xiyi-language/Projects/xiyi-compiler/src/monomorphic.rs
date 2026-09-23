// monomorphic.rs
use crate::mir::*;
use crate::ast::{Type, ShapeDim};
use std::collections::{HashMap, HashSet};

pub struct Monomorphic;

impl Monomorphic {
    pub fn run(mut program: MirProgram) -> MirProgram {
        let all_fns = program.fns;
        let mut new_fns = Vec::new();
        let mut used_fn_names = HashSet::new();
        let mut call_sites = Vec::new();

        // 关键修复：原来 struct 实例是收集进
        // `HashMap<String, Vec<Type>>`——同一个泛型结构体名字只能存一份
        // 类型实参，如果程序里同时用到 `Point<i32>` 和 `Point<String>`，
        // 后收集到的会直接覆盖掉先收集到的，只有一个会被真正生成，
        // 另一个悄悄丢失。这跟 `call_sites`（函数那边）的做法不一致——
        // 函数那边一直是 `Vec<(String, Vec<Type>)>`，同一个名字的多次
        // 不同实例化都能各自留一条记录，靠后面按"单态化之后的名字"
        // （比如 `identity_i32`）去重，而不是靠原始名字去重。这里改成
        // 同样的 Vec 形式，struct 和 enum 都是。
        let mut struct_instances: Vec<(String, Vec<Type>)> = Vec::new();
        let mut enum_instances: Vec<(String, Vec<Type>)> = Vec::new();

        // 初始收集所有函数的调用 + 泛型结构体/枚举实例
        for f in &all_fns {
            Self::collect_calls(&f.body, &mut call_sites);
            Self::collect_generic_instances(&f.body, &mut struct_instances, &mut enum_instances);
        }

        // 循环直到没有新的调用需要处理
        while !call_sites.is_empty() {
            let mut new_call_sites = Vec::new();
            for (func_name, generic_args) in call_sites.drain(..) {
                if let Some(base_fn) = all_fns.iter().find(|f| f.name == func_name) {
                    if base_fn.generic_params.is_empty() {
                        continue;
                    }
                    let mut subst = HashMap::new();
                    for (param, concrete) in base_fn.generic_params.iter().zip(generic_args.iter()) {
                        subst.insert(param.name.clone(), concrete.clone());
                    }
                    let mut cloned = base_fn.clone();
                    let mangled = Self::mangled_name(&base_fn.name, &generic_args);
                    cloned.name = mangled.clone();
                    cloned.generic_params.clear();
                    cloned.params = cloned.params
                        .into_iter()
                        .map(|(name, ty)| (name, Self::replace_type(&ty, &subst)))
                        .collect();
                    cloned.return_type = cloned.return_type.map(|ty| Self::replace_type(&ty, &subst));
                    cloned.body = Self::replace_body(&cloned.body, &subst);

                    if !used_fn_names.contains(&mangled) {
                        Self::collect_calls(&cloned.body, &mut new_call_sites);
                        // 关键修复：新生成的这份函数体是把 T 换成具体
                        // 类型之后的版本，它内部对泛型结构体/枚举的
                        // 实例化（比如 `fn wrap<T>(x: T) -> Box<T> { Box
                        // { value: x } }` 里的 `Box { value: x }`，T 换成
                        // i32 之后就是 `Box<i32>` 的一次实例化）只有在
                        // 替换完之后才看得出具体是哪个类型——原来这里
                        // 没有对 cloned.body 重新收集一次，这类"藏在
                        // 泛型函数体里、只有单态化之后才能确定具体类型"
                        // 的结构体/枚举实例化会被完全漏掉。
                        Self::collect_generic_instances(&cloned.body, &mut struct_instances, &mut enum_instances);
                        new_fns.push(cloned);
                        used_fn_names.insert(mangled);
                    }
                }
            }
            call_sites = new_call_sites;
        }

        // 3. 生成具体结构体
        // 关键修复：原来遍历的是 `used_struct_instances: HashMap<...>`，
        // 一个名字只有一份类型实参；现在遍历 Vec，同一个结构体的多次
        // 不同实例化都会各自走一遍，靠 `generated_struct_names`（存的
        // 是单态化之后的名字，如 `Point_i32`）去重，避免同一个具体
        // 实例被重复生成。
        let mut new_structs = Vec::new();
        let mut generated_struct_names: HashSet<String> = HashSet::new();
        for (struct_name, type_args) in &struct_instances {
            if let Some(base_struct) = program.structs.iter().find(|s| &s.name == struct_name) {
                if base_struct.generic_params.is_empty() {
                    continue;
                }
                let mangled = Self::mangled_name(&base_struct.name, type_args);
                if generated_struct_names.contains(&mangled) {
                    continue;
                }
                let mut subst = HashMap::new();
                for (param, concrete) in base_struct.generic_params.iter().zip(type_args.iter()) {
                    subst.insert(param.name.clone(), concrete.clone());
                }
                let mut cloned = base_struct.clone();
                cloned.name = mangled.clone();
                cloned.generic_params.clear();
                cloned.fields = cloned.fields
                    .into_iter()
                    .map(|(name, ty)| (name, Self::replace_type(&ty, &subst)))
                    .collect();
                generated_struct_names.insert(mangled);
                new_structs.push(cloned);
            }
        }

        // 3.5 生成具体枚举
        // 关键新增：这一整段之前完全不存在——mir.rs 里 MirEnum 本来就带
        // generic_params（跟 MirStruct 是同一个设计），意味着这门语言
        // 的枚举（很可能包括 Option<T>/Result<T,E> 这类核心类型）本来
        // 就可以是泛型的，但 monomorphic.rs 只处理了 struct，压根没碰
        // enum——任何用到泛型枚举的程序，枚举定义会带着没法翻译成
        // Rust 的 TypeParam 类型原样流到后面的阶段。这里补上跟 struct
        // 完全对称的一套流程。
        let mut new_enums = Vec::new();
        let mut generated_enum_names: HashSet<String> = HashSet::new();
        for (enum_name, type_args) in &enum_instances {
            if let Some(base_enum) = program.enums.iter().find(|e| &e.name == enum_name) {
                if base_enum.generic_params.is_empty() {
                    continue;
                }
                let mangled = Self::mangled_name(&base_enum.name, type_args);
                if generated_enum_names.contains(&mangled) {
                    continue;
                }
                let mut subst = HashMap::new();
                for (param, concrete) in base_enum.generic_params.iter().zip(type_args.iter()) {
                    subst.insert(param.name.clone(), concrete.clone());
                }
                let mut cloned = base_enum.clone();
                cloned.name = mangled.clone();
                cloned.generic_params.clear();
                cloned.variants = cloned.variants
                    .into_iter()
                    .map(|(name, ty_opt)| (name, ty_opt.map(|ty| Self::replace_type(&ty, &subst))))
                    .collect();
                generated_enum_names.insert(mangled);
                new_enums.push(cloned);
            }
        }

        // 4. 保留非泛型函数
        // 5. 保留非泛型结构体/枚举
        //
        // 关键修复（这三处是同一个 bug，抄了三遍）：原来的条件是
        // `x.generic_params.is_empty() || !used_xxx_names.contains(&x.name)`。
        // `used_xxx_names`/`generated_xxx_names` 存的都是单态化之后的
        // 名字（`identity_i32`、`Point_i32`），这里的 `x.name` 遍历的
        // 却是原始、没改过的名字（`identity`、`Point`）——两者永远对
        // 不上，`!contains(&x.name)` 恒为 true，等于这个"要不要保留
        // 原始泛型模板"的判断从来没有生效过：不管一个泛型函数/结构体/
        // 枚举有没有被用到、有没有生成具体版本，原始模板（参数/字段/
        // 变体类型里还带着 TypeParam）都会原封不动地混进最终产物。
        // TypeParam 不是一个 codegen 能翻译成具体 Rust 代码的类型，
        // 这些模板留在最终产物里迟早在更后面的阶段炸掉。
        //
        // 正确的判断其实不需要"是否被用到"这个信息：泛型模板本身永远
        // 不该原样进最终产物——用到的实例已经在上面单独生成过具体版本
        // 了，没用到的就是死代码，两种情况下原始模板都该被丢弃，
        // 直接按 `generic_params.is_empty()` 二选一即可。
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

        // 6. 替换所有函数体中的类型
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
                        Self::replace_type_in_rvalue(value, &struct_instance_names, &enum_instance_names);
                    }
                }
            }
        }

        for s in &mut new_structs {
            for (_, ty) in s.fields.iter_mut() {
                Self::replace_generic_with_concrete(ty, &struct_instance_names, &enum_instance_names);
            }
        }
        for e in &mut new_enums {
            for (_, ty_opt) in e.variants.iter_mut() {
                if let Some(ty) = ty_opt {
                    Self::replace_generic_with_concrete(ty, &struct_instance_names, &enum_instance_names);
                }
            }
        }

        // 关键修复（过期注释，跟下面的代码自相矛盾）：这段注释原来
        // 说"这一轮先没做，需要在这之后单独一轮补上"——但下面紧接着
        // 就调用了 `update_enum_names_in_fns`，这正是这一轮新加的、
        // 用来补上这个缺口的实现：按每个 Discriminant/EnumPayload 的
        // `base`/`value` 操作数对应的局部变量类型（第 6 步已经把泛型
        // 枚举类型替换成具体的 `Type::Enum("Option_i32")` 这类名字了），
        // 反查出正确的 enum_name 回填进去。留着"还没做"的说明会让下一
        // 个读到这里的人误以为这个缺口还在，所以删掉，只留下面真正
        // 在做这件事的调用和它自己的实现。
        Self::update_enum_names_in_fns(&mut new_fns);

        MirProgram {
            structs: new_structs,
            enums: new_enums,
            fns: new_fns,
            intrinsics_used: program.intrinsics_used,
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
                    _ => continue,
                };
                match value {
                    MirRvalue::StructInit { struct_name, generic_args, .. } if !generic_args.is_empty() => {
                        struct_out.push((struct_name.clone(), generic_args.clone()));
                    }
                    MirRvalue::EnumVariantConstruction { enum_name, generic_args, .. } if !generic_args.is_empty() => {
                        enum_out.push((enum_name.clone(), generic_args.clone()));
                    }
                    _ => {}
                }
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

        MirBody {
            locals: new_locals,
            blocks: new_blocks,
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
            _ => stmt.clone(),
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

    fn replace_terminator(term: &MirTerminator, _subst: &HashMap<String, Type>) -> MirTerminator {
        // 终结指令里的操作数（If.cond / Switch.discr / Return 的值）
        // 都是 MirOperand，只可能是 Copy/Move/Constant，不携带类型
        // 信息，这里没有类型可替换，原样 clone。
        term.clone()
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
            _ => {}
        }
    }

    fn replace_type_in_rvalue(
        rv: &mut MirRvalue,
        struct_instances: &HashSet<String>,
        enum_instances: &HashSet<String>,
    ) {
        match rv {
            MirRvalue::Cast(_, ty) => {
                Self::replace_generic_with_concrete(ty, struct_instances, enum_instances);
            }
            MirRvalue::Call { generic_args, .. } => {
                for ty in generic_args.iter_mut() {
                    Self::replace_generic_with_concrete(ty, struct_instances, enum_instances);
                }
            }
            MirRvalue::MethodCall { generic_args, .. } => {
                for ty in generic_args.iter_mut() {
                    Self::replace_generic_with_concrete(ty, struct_instances, enum_instances);
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
                    }
                }
            }
            MirRvalue::EnumVariantConstruction { enum_name, generic_args, .. } => {
                if !generic_args.is_empty() {
                    let mangled = Self::mangled_name(enum_name, generic_args);
                    if enum_instances.contains(&mangled) {
                        *enum_name = mangled;
                        generic_args.clear();
                    }
                }
            }
            _ => {}
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

            // 隐私类型（剥离标签，只保留内部类型）
            Type::Privacy(inner, _) => Self::type_to_string(inner),

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
                    _ => {}
                }
            }
            // 终止器中的 cond 和 discr 是 MirOperand，但 Discriminant 已经在 Rvalue 中处理了，
            // 所以不需要再处理终止器。
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
