use crate::ast::*;
use crate::lexer::{self, Comment, LexError};
use crate::parser::{self, ParseError};

const WIDTH: usize = 80;

/// Error from [`format_source`]: the input failed to lex or parse, so
/// there is no canonical form to produce.
#[derive(Debug)]
pub enum FormatError {
    Lex(LexError),
    Parse(ParseError),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::Lex(e) => write!(f, "{e}"),
            FormatError::Parse(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<LexError> for FormatError {
    fn from(e: LexError) -> Self {
        FormatError::Lex(e)
    }
}

impl From<ParseError> for FormatError {
    fn from(e: ParseError) -> Self {
        FormatError::Parse(e)
    }
}

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

/// Canonically format Surf `source`, preserving its comments.
///
/// [`format_program`] works purely from the parsed AST, which carries
/// no comments — it is the right entry point for synthesized decls
/// (decompiler output, prove fixtures) that never had source comments.
/// `format_source` is the entry point for `chelis fmt` on a real
/// on-disk `.ch` file: it re-lexes to recover comments and splices
/// them back into the formatted output at their source position.
///
/// Comments are placed between the formatted declarations whose source
/// offsets bracket them. A comment that falls inside a declaration's
/// span (e.g. inside a function body) is emitted immediately before
/// that declaration rather than being dropped — body-internal comment
/// placement is not yet position-exact, but no comment is lost.
pub fn format_source(source: &str) -> Result<String, FormatError> {
    let (tokens, comments) = lexer::lex_with_comments(source)?;
    let decls = parser::parse(&tokens)?;
    let mut lines: Vec<String> = Vec::new();
    let mut next = 0usize;
    emit_decls_with_comments(&decls, &comments, &mut next, &mut lines);
    // Any comments after the last declaration.
    for comment in &comments[next..] {
        lines.push(comment.text.clone());
    }
    if lines.is_empty() {
        Ok(String::new())
    } else {
        Ok(format!("{}\n", lines.join("\n")))
    }
}

/// Walk `decls` in source order, emitting each pending comment whose
/// source offset precedes the current declaration's start before the
/// declaration itself. `next` is the index of the first not-yet-emitted
/// comment in `comments` (which is in source order). Recurses into
/// `Module` bodies so comments land in the right nesting scope.
fn emit_decls_with_comments(
    decls: &[Decl],
    comments: &[Comment],
    next: &mut usize,
    lines: &mut Vec<String>,
) {
    for decl in decls {
        let decl_start = decl.span().offset;
        while *next < comments.len() && comments[*next].span.offset < decl_start {
            lines.push(comments[*next].text.clone());
            *next += 1;
        }
        match decl {
            Decl::Module {
                name, decls: inner, ..
            } => {
                let mut header = format!("module {name}");
                let decl_end = decl.span().end();
                if inner.is_empty() {
                    // Comments inside an otherwise-empty module body.
                    let mut body: Vec<String> = Vec::new();
                    while *next < comments.len() && comments[*next].span.offset < decl_end {
                        body.push(comments[*next].text.clone());
                        *next += 1;
                    }
                    if !body.is_empty() {
                        header.push('\n');
                        header.push_str(&body.join("\n"));
                    }
                    lines.push(header);
                } else {
                    let mut body: Vec<String> = Vec::new();
                    emit_decls_with_comments(inner, comments, next, &mut body);
                    // Trailing comments still inside the module span.
                    while *next < comments.len() && comments[*next].span.offset < decl_end {
                        body.push(comments[*next].text.clone());
                        *next += 1;
                    }
                    header.push('\n');
                    header.push_str(&body.join("\n"));
                    lines.push(header);
                }
            }
            _ => {
                // A comment whose offset falls within this declaration's
                // own span (e.g. inside a function body) is emitted just
                // before the declaration so it is never lost.
                let decl_end = decl.span().end();
                while *next < comments.len() && comments[*next].span.offset < decl_end {
                    lines.push(comments[*next].text.clone());
                    *next += 1;
                }
                lines.push(format_decl(decl));
            }
        }
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
            opaque,
            ..
        } => {
            let params = format_type_params(params);
            let body = if variants.is_empty() {
                format!("type {name}{params}")
            } else {
                let variants = variants
                    .iter()
                    .map(format_variant)
                    .collect::<Vec<_>>()
                    .join("\n  | ");
                format!("type {name}{params} =\n  | {variants}")
            };
            if *opaque {
                format!("@opaque\n{body}")
            } else {
                body
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
        Decl::Property {
            name,
            params,
            preconditions,
            body,
            options,
            ..
        } => format_property(name, params, preconditions, body, options),
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

fn format_property(
    name: &str,
    params: &[Param],
    preconditions: &[Expr],
    body: &Expr,
    options: &[PropertyOption],
) -> String {
    let params = params
        .iter()
        .map(format_param)
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = format!("@property {name} forall({params})");
    if !preconditions.is_empty() {
        out.push_str(" where ");
        out.push_str(
            &preconditions
                .iter()
                .map(format_expr)
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    out.push_str(":\n  ");
    out.push_str(&format_expr(body).replace('\n', "\n  "));
    for option in options {
        out.push('\n');
        out.push_str(&format_property_option(option));
    }
    out
}

fn format_property_option(option: &PropertyOption) -> String {
    match option {
        PropertyOption::Tolerance(value, _) => {
            format!("  with tolerance = {}", format_expr(value))
        }
        PropertyOption::Seed(value, _) => format!("  with seed = {}", format_expr(value)),
        PropertyOption::Samples(value, _) => format!("  with samples = {}", format_expr(value)),
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
        return " ! {}".to_string();
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
        TypeExpr::Ref(inner, _) => format!("&{}", format_type(inner)),
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
        // Typed-suffix literals (spec/02-surf-syntax.md §P10a): the
        // canonical formatter preserves the suffix on the literal token
        // since dropping it would change the program's typing.
        Literal::TypedInt(value, suffix) => format!("{value}{}", suffix.as_str()),
        Literal::TypedFloat(value, suffix) => {
            let text = value.to_string();
            let body = if text.contains('.') || text.contains('e') || text.contains('E') {
                text
            } else {
                format!("{text}.0")
            };
            format!("{body}{}", suffix.as_str())
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
        // Single-line `{ body }` is only safe when `format_expr(body)` is
        // itself single-line. For multi-line bodies (notably 3+ stage
        // pipe chains, which format multi-line per `format_pipe_layout`),
        // the single-line template glues the opening `{` to the seed and
        // strips per-stage indentation, breaking `fmt` idempotency on
        // re-parse. Mirror the non-empty path: indent each body line by
        // two spaces and emit the braces on their own lines.
        // Finding 3a (PR #51).
        let rendered = format_expr(body);
        if !rendered.contains('\n') {
            return format!("{{ {rendered} }}");
        }
        let lines: Vec<String> = rendered.lines().map(|line| format!("  {line}")).collect();
        return format!("{{\n{}\n}}", lines.join("\n"));
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
        stages.iter().map(format_pipe_stage).collect(),
    )
}

fn format_pipe_with_binding(head: &str, seed: &Expr, stages: &[Expr]) -> String {
    format_pipe_layout(
        Some(head),
        format_expr(seed),
        stages.iter().map(format_pipe_stage).collect(),
    )
}

/// Format a single pipe stage, compacting synthesized unary-builtin
/// lambdas (`fn (v) -> realize(v)`, `fn (v) -> copy(v)`) back to the
/// bare keyword form. Mirrors spec `01-nomenclature.md` §3.6: the
/// decompiler/formatter may compact a lambda stage to call-stage sugar
/// when the carried value is the only argument. Item 2b round-trip.
fn format_pipe_stage(stage: &Expr) -> String {
    if let Some(compacted) = compact_bare_unary_builtin_stage(stage) {
        return compacted;
    }
    format_expr(stage)
}

fn compact_bare_unary_builtin_stage(stage: &Expr) -> Option<String> {
    let Expr::Lambda(params, body, _) = stage else {
        return None;
    };
    let [only_param] = params.as_slice() else {
        return None;
    };
    if only_param.ty.is_some() {
        return None;
    }
    let name = &only_param.name;
    match body.as_ref() {
        Expr::Realize(inner, _) if matches!(inner.as_ref(), Expr::Var(v, _) if v == name) => {
            Some("realize".to_string())
        }
        Expr::Copy(inner, _) if matches!(inner.as_ref(), Expr::Var(v, _) if v == name) => {
            Some("copy".to_string())
        }
        _ => None,
    }
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

    #[test]
    fn explicit_empty_effect_row_is_preserved() {
        // Regression: `! {}` is a meaningful annotation (declared-pure) and must survive
        // a format round-trip. Erasing it silently downgrades the effect contract and
        // lets assertions leak into declared-pure functions.
        let source = "def f() -> unit ! {} = ()\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            rendered.contains("! {}"),
            "empty effect row must be preserved; got: {rendered}"
        );
    }

    #[test]
    fn explicit_effect_row_survives_round_trip() {
        let source = "def f() -> unit ! { Test } = ()\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            rendered.contains("! { Test }"),
            "Test effect row must be preserved; got: {rendered}"
        );
    }

    #[test]
    fn unannotated_def_stays_unannotated() {
        // The absence of an effect row is distinct from an empty row and must survive.
        let source = "def f() -> unit = ()\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            !rendered.contains("! {"),
            "unannotated def must not grow an effect row; got: {rendered}"
        );
    }

    // ── format_source comment preservation (#144) ────────────────

    #[test]
    fn format_source_preserves_line_and_block_comments() {
        let source = "module School.Comment_Test\n\
                       -- regular comment 1\n\
                       --- triple-dash\n\
                       -- expect:\n\
                       {- block comment -}\n\
                       def main() -> f32 = cast(1.0, f32)\n";
        let rendered = format_source(source).expect("format");
        assert!(rendered.contains("-- regular comment 1"), "got: {rendered}");
        assert!(rendered.contains("--- triple-dash"), "got: {rendered}");
        assert!(rendered.contains("-- expect:"), "got: {rendered}");
        assert!(rendered.contains("{- block comment -}"), "got: {rendered}");
        assert!(rendered.contains("def main"), "got: {rendered}");
    }

    #[test]
    fn format_source_is_idempotent_with_comments() {
        let source = "module Foo\n\
                       {- multi-line\n   block -}\n\
                       def a() -> f32 = cast(1.0, f32)\n\
                       -- between defs\n\
                       def b() -> f32 = cast(2.0, f32)\n";
        let once = format_source(source).expect("format once");
        let twice = format_source(&once).expect("format twice");
        assert_eq!(once, twice, "format_source must be idempotent");
    }

    #[test]
    fn format_source_preserves_comment_only_module_body() {
        // An illustrative file whose body is 100% documentation must
        // not collapse to a bare `module Foo`.
        let source = "module Empty\n\
                       -- this file is all documentation\n\
                       -- explaining a concept\n";
        let rendered = format_source(source).expect("format");
        assert!(
            rendered.contains("-- this file is all documentation"),
            "comment-only module body was dropped; got: {rendered}"
        );
        assert!(
            rendered.contains("-- explaining a concept"),
            "got: {rendered}"
        );
    }

    #[test]
    fn format_source_preserves_file_header_comment_above_module() {
        let source = "-- file header above the module\n\
                       module Top\n\
                       def x() -> f32 = cast(3.0, f32)\n";
        let rendered = format_source(source).expect("format");
        assert!(
            rendered.starts_with("-- file header above the module"),
            "header comment lost or moved; got: {rendered}"
        );
    }

    #[test]
    fn format_source_reformats_non_canonical_while_keeping_comments() {
        let source = "module Bar\n-- keep me\ndef   f()  ->  f32  =  cast(1.0,f32)\n";
        let rendered = format_source(source).expect("format");
        assert!(rendered.contains("-- keep me"), "got: {rendered}");
        assert!(
            rendered.contains("def f() -> f32 = cast(1.0, f32)"),
            "non-canonical spacing was not normalized; got: {rendered}"
        );
    }

    #[test]
    fn format_source_without_comments_matches_format_program() {
        let source = "module Bare\ndef f() -> f32 = cast(1.0, f32)\n";
        let via_source = format_source(source).expect("format_source");
        let decls = crate::parser::parse_str(source).expect("parse");
        let via_program = format_program(&decls);
        assert_eq!(
            via_source, via_program,
            "comment-free input must format identically through both paths"
        );
    }
}
