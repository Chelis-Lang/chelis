use crate::ast::*;

const WIDTH: usize = 80;

pub fn format_program(decls: &[Decl]) -> String {
    let mut out = Vec::new();
    for decl in decls {
        out.push(format_decl(decl));
    }
    if out.is_empty() {
        String::new()
    } else {
        format!("{}\n", out.join("\n"))
    }
}

fn format_decl(decl: &Decl) -> String {
    match decl {
        Decl::Module { name, decls, .. } => {
            let mut out = format!("module {name}");
            if !decls.is_empty() {
                out.push('\n');
                out.push_str(&decls.iter().map(format_decl).collect::<Vec<_>>().join("\n"));
            }
            out
        }
        Decl::Import { module, kind, .. } => match kind {
            ImportKind::Qualified => format!("import {module}"),
            ImportKind::All => format!("import {module} (..)"),
            ImportKind::Names(names) => format!("import {module} ({})", names.join(", ")),
        },
        Decl::Sig {
            name, ty, effects, ..
        } => {
            let effects = format_effects(effects.as_deref());
            format!("sig {name}: {}{effects}", format_type(ty))
        }
        Decl::Dim { names, .. } => format!("dim {}", names.join(", ")),
        Decl::TypeDef {
            name,
            params,
            variants,
            ..
        } => {
            let params = format_type_params(params);
            if variants.is_empty() {
                format!("type {name}{params}")
            } else {
                let variants = variants
                    .iter()
                    .map(format_variant)
                    .collect::<Vec<_>>()
                    .join("\n  | ");
                format!("type {name}{params} =\n  | {variants}")
            }
        }
        Decl::TypeAlias {
            name, params, ty, ..
        } => {
            let params = format_type_params(params);
            format!("type {name}{params} = {}", format_type(ty))
        }
        Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            effects,
            body,
            ..
        } => {
            let dims = if dim_params.is_empty() {
                String::new()
            } else {
                format!("[{}]", dim_params.join(", "))
            };
            let params = params
                .iter()
                .map(format_param)
                .collect::<Vec<_>>()
                .join(", ");
            let ret = ret_ty
                .as_ref()
                .map(|ty| format!(" -> {}", format_type(ty)))
                .unwrap_or_default();
            let effects = format_effects(effects.as_deref());
            let body = format_function_body(body);
            format!("def {name}{dims}({params}){ret}{effects} = {body}")
        }
        Decl::LetDef {
            name, ty, value, ..
        } => {
            if let Some(ty) = ty {
                format!("{name}: {} = {}", format_type(ty), format_expr(value))
            } else {
                format!("{name} = {}", format_expr(value))
            }
        }
        Decl::MacroDef {
            name, params, body, ..
        } => {
            format!(
                "macro {name}({}) = {}",
                params.join(", "),
                format_expr(body)
            )
        }
        Decl::Export { names, .. } => format!("export ({})", names.join(", ")),
    }
}

fn format_variant(variant: &Variant) -> String {
    match &variant.fields {
        VariantFields::Positional(fields) if fields.is_empty() => variant.name.clone(),
        VariantFields::Positional(fields) => format!(
            "{}({})",
            variant.name,
            fields
                .iter()
                .map(format_type)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        VariantFields::Record(fields) if fields.is_empty() => variant.name.clone(),
        VariantFields::Record(fields) => format!(
            "{} {{ {} }}",
            variant.name,
            fields
                .iter()
                .map(|(name, ty)| format!("{name}: {}", format_type(ty)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn format_type_params(params: &[String]) -> String {
    if params.is_empty() {
        String::new()
    } else {
        format!("[{}]", params.join(", "))
    }
}

fn format_param(param: &Param) -> String {
    match &param.ty {
        Some(ty) => format!("{}: {}", param.name, format_type(ty)),
        None => param.name.clone(),
    }
}

fn format_effects(effects: Option<&[EffectExpr]>) -> String {
    let Some(effects) = effects else {
        return String::new();
    };
    if effects.is_empty() {
        return String::new();
    }
    format!(
        " ! {{ {} }}",
        effects
            .iter()
            .map(format_effect)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_effect(effect: &EffectExpr) -> String {
    match effect {
        EffectExpr::Diff(_) => "Diff".to_string(),
        EffectExpr::Random(_) => "Random".to_string(),
        EffectExpr::Accum(_) => "Accum".to_string(),
        EffectExpr::Io(_) => "IO".to_string(),
        EffectExpr::Test(_) => "Test".to_string(),
        EffectExpr::Resource(device, _) => format!("Resource(\"{device}\")"),
    }
}

fn format_type(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named(name, _) => name.clone(),
        TypeExpr::Tensor(parts, precision, _) => {
            let mut elems = parts.iter().map(format_type).collect::<Vec<_>>();
            elems.push(precision.clone());
            format!("tensor[{}]", elems.join(", "))
        }
        TypeExpr::Arrow(args, ret, _) => {
            let mut parts = args.iter().map(format_type).collect::<Vec<_>>();
            parts.push(format_type(ret));
            parts.join(" -> ")
        }
        TypeExpr::App(name, args, _) if args.is_empty() => name.clone(),
        TypeExpr::App(name, args, _) => {
            format!(
                "{name}[{}]",
                args.iter().map(format_type).collect::<Vec<_>>().join(", ")
            )
        }
        TypeExpr::Tuple(parts, _) => {
            format!(
                "({})",
                parts.iter().map(format_type).collect::<Vec<_>>().join(", ")
            )
        }
        TypeExpr::Infer(_) => "_".to_string(),
    }
}

fn format_function_body(expr: &Expr) -> String {
    if matches!(expr, Expr::Block(_, _, _)) {
        format_expr(expr)
    } else {
        let rendered = format_expr(expr);
        if rendered.contains('\n') {
            format!("{{\n{}\n}}", indent_lines(&rendered, 2))
        } else {
            rendered
        }
    }
}

fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Lit(lit, _) => format_lit(lit),
        Expr::Var(name, _) | Expr::Constructor(name, _) => name.clone(),
        Expr::Apply(func, args, _) => format!("{}({})", format_expr(func), format_args(args)),
        Expr::List(items, _) => format!(
            "[{}]",
            items.iter().map(format_expr).collect::<Vec<_>>().join(", ")
        ),
        Expr::Record(name, fields, _) => format!(
            "{name} {{ {} }}",
            fields
                .iter()
                .map(|(field, expr)| format!("{field}: {}", format_expr(expr)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Access(expr, field, _) => format!("{}.{}", wrap_simple(expr), field),
        Expr::TupleGet(expr, index, _) => format!("{}.{}", wrap_simple(expr), index),
        Expr::Binary(op, left, right, _) => {
            format!(
                "({} {} {})",
                format_expr(left),
                format_binop(*op),
                format_expr(right)
            )
        }
        Expr::Unary(op, expr, _) => format!("{}{}", format_unary(*op), wrap_simple(expr)),
        Expr::Pipe(seed, stages, _) => format_pipe_expr(seed, stages),
        Expr::If(cond, then_expr, else_expr, _) => format!(
            "if {} then {} else {}",
            format_expr(cond),
            format_expr(then_expr),
            format_expr(else_expr)
        ),
        Expr::Match(scrutinee, arms, _) => {
            let arms = arms.iter().map(format_arm).collect::<Vec<_>>().join("\n  ");
            format!("match {} with {{\n  {}\n}}", format_expr(scrutinee), arms)
        }
        Expr::Lambda(params, body, _) => format!(
            "fn ({}) -> {}",
            params
                .iter()
                .map(format_param)
                .collect::<Vec<_>>()
                .join(", "),
            format_expr(body)
        ),
        Expr::Tuple(parts, _) => {
            format!(
                "({})",
                parts.iter().map(format_expr).collect::<Vec<_>>().join(", ")
            )
        }
        Expr::Cast(expr, ty, _) => format!("cast({}, {})", format_expr(expr), ty),
        Expr::Grad(expr, wrt, _) => match wrt {
            None => format!("grad({})", format_expr(expr)),
            Some(names) if names.len() == 1 => {
                format!("grad({}, wrt={})", format_expr(expr), names[0])
            }
            Some(names) => format!("grad({}, wrt=({}))", format_expr(expr), names.join(", ")),
        },
        Expr::Vmap(expr, axis, _) => match axis {
            Some(axis) => format!("vmap({}, axis={axis})", format_expr(expr)),
            None => format!("vmap({})", format_expr(expr)),
        },
        Expr::Jit(expr, _) => format!("jit({})", format_expr(expr)),
        Expr::Realize(expr, _) => format!("realize({})", format_expr(expr)),
        Expr::Copy(expr, _) => format!("copy({})", format_expr(expr)),
        Expr::Borrow(expr, _) => format!("&{}", wrap_simple(expr)),
        Expr::WithSeed(seed, body, _) => format!(
            "with seed({}) {}",
            format_expr(seed),
            format_handler_body(body)
        ),
        Expr::WithDevice(device, body, _) => {
            format!(
                "with device({}) {}",
                format_expr(device),
                format_handler_body(body)
            )
        }
        Expr::Par(exprs, _) => {
            format!(
                "par {{ {} }}",
                exprs.iter().map(format_expr).collect::<Vec<_>>().join("; ")
            )
        }
        Expr::Annotate(expr, ty, _) => format!("({} : {})", format_expr(expr), format_type(ty)),
        Expr::Block(bindings, body, _) => format_block(bindings, body),
    }
}

fn format_lit(lit: &Literal) -> String {
    match lit {
        Literal::Int(value) => value.to_string(),
        Literal::Float(value) => {
            let text = value.to_string();
            if text.contains('.') {
                text
            } else {
                format!("{text}.0")
            }
        }
        Literal::Str(value) => format!("{value:?}"),
        Literal::Bool(value) => value.to_string(),
    }
}

fn format_args(args: &[Expr]) -> String {
    args.iter().map(format_expr).collect::<Vec<_>>().join(", ")
}

fn format_arm(arm: &MatchArm) -> String {
    let guard = arm
        .guard
        .as_ref()
        .map(|guard| format!(" if {}", format_expr(guard)))
        .unwrap_or_default();
    format!(
        "| {}{} => {}",
        format_pattern(&arm.pattern),
        guard,
        format_expr(&arm.body)
    )
}

fn format_pattern(pattern: &Pattern) -> String {
    match pattern {
        Pattern::Wildcard(_) => "_".to_string(),
        Pattern::Var(name, _) => name.clone(),
        Pattern::Lit(lit, _) => format_lit(lit),
        Pattern::Constructor(name, args, _) if args.is_empty() => name.clone(),
        Pattern::Constructor(name, args, _) => format!(
            "{name}({})",
            args.iter()
                .map(format_pattern)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pattern::Tuple(parts, _) => format!(
            "({})",
            parts
                .iter()
                .map(format_pattern)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pattern::Record(name, fields, _) => format!(
            "{name} {{ {} }}",
            fields
                .iter()
                .map(|(field, pattern)| format!("{field}: {}", format_pattern(pattern)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pattern::As(name, pattern, _) => format!("{name} @ {}", format_pattern(pattern)),
    }
}

fn format_let_pattern(pattern: &LetPattern) -> String {
    match pattern {
        LetPattern::Var(name, _) => name.clone(),
        LetPattern::Wildcard(_) => "_".to_string(),
        LetPattern::Tuple(parts, _) => format!(
            "({})",
            parts
                .iter()
                .map(format_let_pattern)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn format_binding_line(binding: &LetBinding) -> String {
    let mut head = format_let_pattern(&binding.pattern);
    if let Some(ty) = &binding.ty {
        head.push_str(&format!(": {}", format_type(ty)));
    }
    format_binding_assignment(&head, &binding.value)
}

fn format_binding_assignment(head: &str, value: &Expr) -> String {
    if let Expr::Pipe(seed, stages, _) = value {
        return format_pipe_with_binding(head, seed, stages);
    }
    let rendered = format_expr(value);
    format!("{head} = {rendered}")
}

fn format_handler_body(expr: &Expr) -> String {
    match expr {
        Expr::Block(_, _, _) => format_expr(expr),
        _ => format!("{{\n  {}\n}}", format_expr(expr)),
    }
}

fn format_block(bindings: &[LetBinding], body: &Expr) -> String {
    if bindings.is_empty() {
        return format!("{{ {} }}", format_expr(body));
    }
    let mut lines = Vec::new();
    for binding in bindings {
        for line in format_binding_line(binding).lines() {
            lines.push(format!("  {line}"));
        }
    }
    for line in format_expr(body).lines() {
        lines.push(format!("  {line}"));
    }
    format!("{{\n{}\n}}", lines.join("\n"))
}

fn format_pipe_expr(seed: &Expr, stages: &[Expr]) -> String {
    format_pipe_layout(
        None,
        format_expr(seed),
        stages.iter().map(format_expr).collect(),
    )
}

fn format_pipe_with_binding(head: &str, seed: &Expr, stages: &[Expr]) -> String {
    format_pipe_layout(
        Some(head),
        format_expr(seed),
        stages.iter().map(format_expr).collect(),
    )
}

fn format_pipe_layout(binding_head: Option<&str>, seed: String, stages: Vec<String>) -> String {
    if stages.is_empty() {
        return match binding_head {
            Some(head) => format!("{head} = {seed}"),
            None => seed,
        };
    }

    let total_stages = 1 + stages.len();
    let flat_chain = std::iter::once(seed.clone())
        .chain(stages.iter().map(|stage| format!("|> {stage}")))
        .collect::<Vec<_>>()
        .join(" ");
    let flat = match binding_head {
        Some(head) => format!("{head} = {flat_chain}"),
        None => flat_chain.clone(),
    };

    if total_stages <= 3 && flat.chars().count() <= WIDTH {
        return flat;
    }

    match binding_head {
        Some(head) => {
            let first_line = format!("{head} = {seed}");
            if first_line.chars().count() <= WIDTH && total_stages <= 2 {
                let mut lines = vec![first_line];
                lines.extend(stages.iter().map(|stage| format!("  |> {stage}")));
                lines.join("\n")
            } else {
                let mut lines = vec![format!("{head} ="), format!("  {seed}")];
                lines.extend(stages.iter().map(|stage| format!("  |> {stage}")));
                lines.join("\n")
            }
        }
        None => {
            let mut lines = vec![seed];
            lines.extend(stages.iter().map(|stage| format!("|> {stage}")));
            lines.join("\n")
        }
    }
}

fn wrap_simple(expr: &Expr) -> String {
    match expr {
        Expr::Lit(_, _)
        | Expr::Var(_, _)
        | Expr::Constructor(_, _)
        | Expr::Apply(_, _, _)
        | Expr::List(_, _)
        | Expr::Access(_, _, _)
        | Expr::TupleGet(_, _, _)
        | Expr::Tuple(_, _) => format_expr(expr),
        _ => format!("({})", format_expr(expr)),
    }
}

fn format_binop(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

fn format_unary(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Not => "!",
    }
}

fn indent_lines(text: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_stage_pipe_chain_breaks_even_under_width() {
        let rendered = format_pipe_layout(
            Some("value"),
            "add(x, y)".to_string(),
            vec!["relu".to_string(), "log".to_string(), "neg".to_string()],
        );

        assert_eq!(
            rendered,
            "value =\n  add(x, y)\n  |> relu\n  |> log\n  |> neg"
        );
    }

    #[test]
    fn formatted_program_ends_with_trailing_newline() {
        let span = chelis_deep::Span::new(0, 0);
        let program = [Decl::LetDef {
            name: "x".to_string(),
            ty: None,
            value: Expr::Lit(Literal::Int(1), span),
            span,
        }];

        assert_eq!(format_program(&program), "x = 1\n");
    }
}
