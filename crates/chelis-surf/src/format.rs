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

/// Canonically format a single Surf expression to its one-expression
/// rendering. This is the same renderer the whole-program formatter uses for
/// expression positions, exposed so a consumer that holds one `Expr` (not a
/// `Decl`) can produce its canonical text directly. The chelis#436 prove-JSON
/// `goal` field uses it to emit the discharged proposition (a property body)
/// in the record, so a consumer displays exactly what was discharged rather
/// than re-parsing it out of source.
pub fn format_expression(expr: &Expr) -> String {
    format_expr(expr)
}

/// Canonically format a property's discharged PROPOSITION: its body when it
/// has no preconditions, or the full quantified, guarded form
/// `forall(<params>) where <preconds>: <body>` when it does (chelis#436,
/// MED-1). The prover discharges `(/\ preconditions) => body`, so a guarded
/// property's proposition is NOT its bare body -- emitting only the body
/// over-claims an unconditional result (`where x > 0.0: x <= x` is not the
/// unconditional `x <= x`). This renders the same `where`-guarded text the
/// whole-property formatter produces (minus the `@property name` prefix), so
/// the `goal` field is exactly what was discharged and re-parses to it. An
/// unguarded property keeps the bare-body form so the common case stays the
/// single proposition a consumer displays directly.
pub fn format_proposition(params: &[Param], preconditions: &[Expr], body: &Expr) -> String {
    if preconditions.is_empty() {
        return format_expr(body);
    }
    let params = params
        .iter()
        .map(format_param)
        .collect::<Vec<_>>()
        .join(", ");
    let preconds = preconditions
        .iter()
        .map(format_expr)
        .collect::<Vec<_>>()
        .join(", ");
    format!("forall({params}) where {preconds}: {}", format_expr(body))
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
            invariant,
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
                // `@opaque`, then the optional `@invariant(binder) <expr>`
                // block, then `type ...` (RFC D-SYNTAX). The invariant
                // body uses the Surf expression formatter so the result
                // is idempotent under `chelis fmt`.
                let mut out = String::from("@opaque\n");
                if let Some(inv) = invariant {
                    out.push_str(&format!(
                        "@invariant({}) {}\n",
                        inv.binder,
                        format_expr(&inv.body)
                    ));
                }
                out.push_str(&body);
                out
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
        PropertyOption::Contract(id, _) => format!("  with contract = {id:?}"),
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
        TypeExpr::RankSpread(name, _) => format!("..{name}"),
        TypeExpr::Tensor(parts, precision, _) => {
            let mut elems = parts.iter().map(format_type).collect::<Vec<_>>();
            elems.push(precision.clone());
            format!("tensor[{}]", elems.join(", "))
        }
        TypeExpr::Arrow(args, ret, _) => {
            // Arrow types are right-associative, so `a -> b -> c` means
            // `a -> (b -> c)`. An arrow that sits in *argument* (left) position
            // of another arrow — `(a -> b) -> c` — is a distinct type (one
            // function-typed argument, not a curried 3-ary). It must be
            // parenthesized to survive a format round-trip (#290). The return
            // type is in right position, where the parens are redundant under
            // right-associativity, so it is printed without grouping.
            let mut parts = args.iter().map(format_type_arg).collect::<Vec<_>>();
            parts.push(format_type(ret));
            parts.join(" -> ")
        }
        // `&` binds tighter than `->`, so a reference to a function type must
        // group the arrow: `&(a -> b)` is distinct from `&a -> b`
        // (`(&a) -> b`). Reuse the arrow-argument grouping helper (#290).
        TypeExpr::Ref(inner, _) => format!("&{}", format_type_arg(inner)),
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

/// Format a type that appears in *argument* (left) position of an arrow.
///
/// A nested arrow here is a function-typed argument and must be grouped so the
/// printed form reparses to the same arity. Right-position (return) types do
/// not need this because arrow is right-associative — see `format_type`'s
/// `TypeExpr::Arrow` arm (#290).
fn format_type_arg(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Arrow(..) => format!("({})", format_type(ty)),
        _ => format_type(ty),
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
            // Operands that are compound, non-self-delimiting expressions
            // (`if`/`match`/`fn`/`|>`/block) MUST be parenthesized, or the
            // operator binds into the operand's tail and the formatted text
            // re-parses to a DIFFERENT proposition than was written
            // (chelis#461): `(if c then a else b) + 1` would render
            // `(if c then a else b + 1)`, in which `+ 1` joins the else-branch.
            // A nested binary already self-parenthesizes, so it stays
            // single-wrapped.
            format!(
                "({} {} {})",
                wrap_operand(left),
                format_binop(*op),
                wrap_operand(right)
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

/// Render `expr` as a binary operand, parenthesizing it when its formatted
/// text is NOT self-delimiting -- i.e. a compound expression whose tail would
/// otherwise absorb the surrounding operator and change the parse
/// (chelis#461). The self-delimiting forms render as a single bracketed or
/// atomic token (a literal/var, a `f(...)` / `[...]` / `N { ... }` / `(...)`
/// form, a postfix `x.f`, an already-parenthesized nested binary, a
/// keyword-call like `cast(...)`/`grad(...)`, or a `( e : T )` ascription), so
/// they need no extra wrap and stay byte-identical (no double parens on a
/// nested binary, which the formatter corpus and idempotency depend on). The
/// keyword-led / prefix forms (`if`, `match`, `fn`, `|>`, block, `with`,
/// `par`) are NOT self-delimiting and are wrapped.
fn wrap_operand(expr: &Expr) -> String {
    match expr {
        Expr::Lit(_, _)
        | Expr::Var(_, _)
        | Expr::Constructor(_, _)
        | Expr::Apply(_, _, _)
        | Expr::List(_, _)
        | Expr::Record(_, _, _)
        | Expr::Access(_, _, _)
        | Expr::TupleGet(_, _, _)
        | Expr::Tuple(_, _)
        | Expr::Binary(_, _, _, _)
        | Expr::Unary(_, _, _)
        | Expr::Cast(_, _, _)
        | Expr::Grad(_, _, _)
        | Expr::Vmap(_, _, _)
        | Expr::Jit(_, _)
        | Expr::Realize(_, _)
        | Expr::Copy(_, _)
        | Expr::Borrow(_, _)
        | Expr::Annotate(_, _, _) => format_expr(expr),
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
    fn format_expression_renders_a_property_body_canonically() {
        // chelis#436: the public expression formatter renders a single Expr
        // (a property body) to its canonical one-expression text, the goal a
        // prove-JSON record carries. It must match the whole-program
        // formatter's expression rendering exactly.
        let source = "@property p forall(x: f32) where x > 0.0:\n  (x - 1.0) <= x\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let Decl::Property { body, .. } = &program[0] else {
            panic!("expected a property decl");
        };
        assert_eq!(format_expression(body), "((x - 1.0) <= x)");
        // Parity with the internal renderer used in expression positions.
        assert_eq!(format_expression(body), format_expr(body));
    }

    fn property_parts(source: &str) -> (Vec<Param>, Vec<Expr>, Expr) {
        let program = crate::parser::parse_str(source).expect("parse");
        let Decl::Property {
            params,
            preconditions,
            body,
            ..
        } = &program[0]
        else {
            panic!("expected a property decl");
        };
        (params.clone(), preconditions.clone(), body.clone())
    }

    #[test]
    fn compound_expr_as_binary_operand_is_parenthesized_and_round_trips() {
        // chelis#461 (the bug #436's goal exposed): a non-self-delimiting
        // compound operand (here an `if`) must be wrapped, or the operator binds
        // into the operand's tail and the text re-parses to a DIFFERENT
        // proposition. `(if x>y then x else y) + 1.0 >= x` must NOT render
        // `((if (x > y) then x else y + 1.0) >= x)` (in which `+ 1.0` joins the
        // else-branch).
        let source = "@property p forall(x: f32, y: f32):\n  (if x > y then x else y) + 1.0 >= x\n";
        let (_, _, body) = property_parts(source);
        let rendered = format_expression(&body);
        assert_eq!(rendered, "(((if (x > y) then x else y) + 1.0) >= x)");
        // Round-trip: the rendered goal re-parses to the SAME AST, so the
        // displayed proposition equals the discharged one (idempotent formatter).
        let reparsed = property_parts(&format!(
            "@property p forall(x: f32, y: f32):\n  {rendered}\n"
        ))
        .2;
        assert_eq!(
            format_expression(&reparsed),
            rendered,
            "the rendered operand-wrapped goal must re-parse to itself"
        );
    }

    #[test]
    fn nested_binary_operand_is_not_double_wrapped() {
        // A nested binary operand already self-parenthesizes, so it stays
        // SINGLE-wrapped -- the fix must not regress this into `((x - 1.0))`,
        // which would break the formatter corpus and idempotency.
        let (_, _, nested) =
            property_parts("@property p forall(x: f32):\n  (x - 1.0) + 2.0 >= x\n");
        assert_eq!(
            format_expression(&nested),
            "(((x - 1.0) + 2.0) >= x)",
            "a nested binary operand is not double-wrapped"
        );
    }

    #[test]
    fn if_in_either_operand_position_is_parenthesized() {
        // The wrap applies to BOTH operands: an `if` on the right of a
        // comparison must be wrapped too, or the comparison binds into its
        // condition/then on re-parse.
        let (_, _, right) = property_parts(
            "@property p forall(x: f32, y: f32):\n  x <= (if x > y then y else x)\n",
        );
        assert_eq!(
            format_expression(&right),
            "(x <= (if (x > y) then y else x))",
            "an if as the right operand is wrapped: {}",
            format_expression(&right)
        );
    }

    #[test]
    fn format_proposition_includes_the_guard_but_keeps_unguarded_bare() {
        // chelis#436 MED-1: a guarded property's discharged proposition is the
        // full `forall ... where ...: body`, not the bare body (which would
        // over-claim an unconditional result). An unguarded property keeps the
        // bare-body form.
        let (params, pre, body) =
            property_parts("@property g forall(x: f32) where x > 0.0:\n  x <= x\n");
        assert_eq!(
            format_proposition(&params, &pre, &body),
            "forall(x: f32) where (x > 0.0): (x <= x)"
        );
        let (uparams, upre, ubody) = property_parts("@property u forall(x: f32):\n  x <= x\n");
        assert!(upre.is_empty());
        assert_eq!(
            format_proposition(&uparams, &upre, &ubody),
            "(x <= x)",
            "an unguarded proposition is the bare body"
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
    fn property_contract_option_formats_round_trip() {
        let source = r#"@property reflected forall(x: f32):
  x == x
  with contract = "std.normal_cdf.reflection"
"#;
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            rendered.contains(r#"with contract = "std.normal_cdf.reflection""#),
            "contract option lost: {rendered}"
        );
        let reparsed = crate::parser::parse_str(&rendered).expect("reparse formatted contract");
        assert_eq!(
            format_program(&reparsed),
            rendered,
            "formatted contract property must be idempotent"
        );
        let [Decl::Property { options, .. }] = reparsed.as_slice() else {
            panic!("formatted source should parse back to one property: {reparsed:?}");
        };
        assert!(matches!(
            options.as_slice(),
            [PropertyOption::Contract(contract, _)] if contract == "std.normal_cdf.reflection"
        ));
    }

    #[test]
    fn opaque_invariant_formats_with_invariant_line() {
        // RFC D-SYNTAX: `chelis fmt` renders the `@invariant(binder)
        // <expr>` block between `@opaque` and `type`.
        let source = "module M\n@opaque\n\
             @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
             type Probability =\n  | Probability { value: f32 }\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(rendered.contains("@opaque"), "opaque lost: {rendered}");
        assert!(
            rendered.contains("@invariant(p)"),
            "invariant line lost: {rendered}"
        );
    }

    #[test]
    fn opaque_invariant_format_is_idempotent() {
        // `chelis fmt` is idempotent on an invariant-bearing opaque type:
        // formatting the formatted output reproduces it.
        let source = "module M\n@opaque\n\
             @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
             type Probability =\n  | Probability { value: f32 }\n";
        let once = format_source(source).expect("format once");
        let twice = format_source(&once).expect("format twice");
        assert_eq!(
            once, twice,
            "fmt not idempotent:\n--once--\n{once}\n--twice--\n{twice}"
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

    // Module-qualified references (chelis#316) must survive a format round
    // trip and stay parseable — the formatter-output-is-parseable invariant.
    // A qualified constructor and a qualified applied value both render back
    // to their dotted-path form.
    #[test]
    fn qualified_reference_round_trips() {
        let source = "def f(m) = add(Demo.Dropout.use(m), Demo.Sd.use(Demo.Sd.Eval))\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            rendered.contains("Demo.Dropout.use(m)"),
            "qualified call must render as a dotted path; got: {rendered}"
        );
        assert!(
            rendered.contains("Demo.Sd.Eval"),
            "qualified constructor must render as a dotted path; got: {rendered}"
        );
        // The rendered text must re-parse — the canonical invariant.
        crate::parser::parse_str(&rendered).expect("formatted qualified reference must re-parse");
    }

    // A module-qualified constructor *pattern* (chelis#316) must likewise
    // survive a format round trip and re-parse.
    #[test]
    fn qualified_constructor_pattern_round_trips() {
        let source =
            "def f(m) = match m with { | Demo.Dropout.Train => 1 | Demo.Dropout.Eval => 0 }\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            rendered.contains("Demo.Dropout.Train") && rendered.contains("Demo.Dropout.Eval"),
            "qualified constructor patterns must render as dotted paths; got: {rendered}"
        );
        crate::parser::parse_str(&rendered)
            .expect("formatted qualified constructor pattern must re-parse");
    }

    // A module-qualified *type* name (chelis#316) must survive a format round
    // trip and re-parse.
    #[test]
    fn qualified_type_name_round_trips() {
        let source = "def f(m: Demo.Dropout.Mode) -> i64 = Demo.Dropout.use(m)\n";
        let program = crate::parser::parse_str(source).expect("parse");
        let rendered = format_program(&program);
        assert!(
            rendered.contains("Demo.Dropout.Mode"),
            "qualified type name must render as a dotted path; got: {rendered}"
        );
        crate::parser::parse_str(&rendered).expect("formatted qualified type name must re-parse");
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

    // ── higher-order function-type parens in arrow argument position (#290) ──
    //
    // Arrow types are right-associative: `a -> b -> c` is `a -> (b -> c)`,
    // a curried 3-ary function. An arrow that appears in *argument* (left)
    // position — `(a -> b) -> c` — is a different type: one argument that is
    // itself a function. The formatter must keep the grouping parens around a
    // left-position arrow so the printed form reparses to the same arity, and
    // must NOT add parens to right-position (return) arrows, where they are
    // redundant under right-associativity.

    /// Find the first `Sig` declaration, descending into a wrapping `Module`.
    fn find_sig_ty(decls: &[Decl]) -> Option<&TypeExpr> {
        for decl in decls {
            match decl {
                Decl::Sig { ty, .. } => return Some(ty),
                Decl::Module { decls, .. } => {
                    if let Some(ty) = find_sig_ty(decls) {
                        return Some(ty);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Format the single `Sig` declaration in `source` and return only the
    /// type portion of the `sig <name>: <type>` line.
    fn sig_type_str(source: &str) -> String {
        let decls = crate::parser::parse_str(source).expect("parse sig");
        let ty =
            find_sig_ty(&decls).unwrap_or_else(|| panic!("no Sig declaration found in: {source}"));
        format_type(ty)
    }

    fn sig_type_ast(source: &str) -> TypeExpr {
        let decls = crate::parser::parse_str(source).expect("parse sig");
        find_sig_ty(&decls)
            .cloned()
            .unwrap_or_else(|| panic!("no Sig declaration found in: {source}"))
    }

    #[test]
    fn hof_arg_arrow_keeps_parens() {
        // `(a -> b) -> c`: the function-typed argument must stay parenthesized.
        assert_eq!(
            sig_type_str("module T\nsig f: (a -> b) -> c"),
            "(a -> b) -> c"
        );
    }

    #[test]
    fn hof_arg_arrow_format_is_idempotent() {
        // A second format pass over the issue reproducer must be stable.
        let source = "module T\nsig f: (a -> b) -> c\ndef f(g, x) = x\n";
        let once = format_source(source).expect("format once");
        let twice = format_source(&once).expect("format twice");
        assert_eq!(once, twice, "HOF sig formatting must be idempotent");
        assert!(
            once.contains("sig f: (a -> b) -> c"),
            "grouping parens around the function-typed argument were dropped; got: {once}"
        );
    }

    #[test]
    fn curried_arrow_gets_no_spurious_parens() {
        // Plain curried `a -> b -> c` must NOT grow parens.
        assert_eq!(sig_type_str("module T\nsig f: a -> b -> c"), "a -> b -> c");
    }

    #[test]
    fn right_nested_arrow_canonicalizes_without_parens() {
        // Right-position arrow parens are redundant; `a -> (b -> c)` is the
        // same type as `a -> b -> c` and canonicalizes to the bare form.
        assert_eq!(
            sig_type_str("module T\nsig f: a -> (b -> c)"),
            "a -> b -> c"
        );
    }

    #[test]
    fn hof_arg_and_curried_are_distinct_asts() {
        // The two sources must parse to *different* ASTs — proof that the
        // grouping is semantically meaningful, not cosmetic.
        let hof = sig_type_ast("module T\nsig f: (a -> b) -> c");
        let curried = sig_type_ast("module T\nsig f: a -> b -> c");
        assert_ne!(
            hof, curried,
            "`(a -> b) -> c` and `a -> b -> c` must be distinct types"
        );
        // And each must format to its own canonical, non-equal string.
        assert_ne!(
            format_type(&hof),
            format_type(&curried),
            "distinct arrow types must render to distinct strings"
        );
    }

    #[test]
    fn hof_arg_arrow_round_trips_through_parser() {
        // Format → reparse → format must be stable and arity-preserving.
        let src = "module T\nsig f: (a -> b) -> c";
        let first = sig_type_str(src);
        let reparsed = sig_type_str(&format!("module T\nsig f: {first}"));
        assert_eq!(first, reparsed, "arrow-arg sig must round-trip");
        // The reparsed AST must still be the one-argument HOF shape.
        let ast = sig_type_ast(&format!("module T\nsig f: {first}"));
        match ast {
            TypeExpr::Arrow(args, _, _) => {
                assert_eq!(args.len(), 1, "must remain a 1-argument function type");
                assert!(
                    matches!(args[0], TypeExpr::Arrow(..)),
                    "the single argument must itself be a function type"
                );
            }
            other => panic!("expected Arrow, got {other:?}"),
        }
    }

    #[test]
    fn multi_arg_hof_round_trips() {
        // `(a -> b) -> (c -> d) -> e`: two function-typed arguments. The
        // first is in left position (needs parens); the second is also in
        // argument position of the outer arrow (needs parens); `e` is the
        // return.
        assert_eq!(
            sig_type_str("module T\nsig f: (a -> b) -> (c -> d) -> e"),
            "(a -> b) -> (c -> d) -> e"
        );
    }

    #[test]
    fn nested_hof_arg_round_trips() {
        // `((a -> b) -> c) -> d`: a function-typed argument whose own argument
        // is a function. Both layers of grouping must survive.
        let src = "module T\nsig f: ((a -> b) -> c) -> d";
        assert_eq!(sig_type_str(src), "((a -> b) -> c) -> d");
        let once = sig_type_str(src);
        let twice = sig_type_str(&format!("module T\nsig f: {once}"));
        assert_eq!(once, twice, "nested HOF arg must be idempotent");
    }

    // These two cases build the `TypeExpr` AST directly so they exercise the
    // formatter's grouping rules independently of the parser's surface grammar
    // for nested groups.

    fn named(n: &str) -> TypeExpr {
        TypeExpr::Named(n.to_string(), chelis_deep::Span::new(0, 0))
    }

    fn arrow(args: Vec<TypeExpr>, ret: TypeExpr) -> TypeExpr {
        TypeExpr::Arrow(args, Box::new(ret), chelis_deep::Span::new(0, 0))
    }

    #[test]
    fn arrow_in_tuple_element_needs_no_parens() {
        // Tuple commas already delimit an arrow element, so an arrow inside a
        // tuple is unambiguous and must NOT gain grouping parens. This guards
        // against the fix over-parenthesizing arrows that are NOT in
        // arrow-argument position: `(a -> b, c) -> d`.
        let tuple = TypeExpr::Tuple(
            vec![arrow(vec![named("a")], named("b")), named("c")],
            chelis_deep::Span::new(0, 0),
        );
        let ty = arrow(vec![tuple], named("d"));
        assert_eq!(format_type(&ty), "(a -> b, c) -> d");
    }

    #[test]
    fn arrow_under_ref_in_arg_position_keeps_inner_parens() {
        // `&(a -> b) -> c`: the argument is a reference to a function type.
        // The arrow under the `&` must stay grouped — `&` binds tighter than
        // `->`, so `&(a -> b)` is distinct from `&a -> b` (`(&a) -> b`).
        let inner = TypeExpr::Ref(
            Box::new(arrow(vec![named("a")], named("b"))),
            chelis_deep::Span::new(0, 0),
        );
        let ty = arrow(vec![inner], named("c"));
        assert_eq!(format_type(&ty), "&(a -> b) -> c");
    }
}
