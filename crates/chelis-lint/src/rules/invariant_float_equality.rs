//! Rule `invariant-float-equality` — exact `==` over a representation
//! field in an opaque-type invariant predicate starves Tier C generation
//! by design (RFC D-WF / D-STARVE). The documented idiom for float
//! aggregates is a tolerance band over a module constant
//! (`sum >= 1.0 - eps && sum <= 1.0 + eps`).
//!
//! Advisory-only: wired through `registry::non_blocking_rules`, so it
//! never fails the style gate or `lint --check`. It is the lint home for
//! the D-WF `==` advisory; the checker has no non-blocking warning
//! channel.

use crate::{Context, Rule, Severity, Surface, Violation};
use chelis_surf::ast::{BinOp, Decl, Expr};

pub struct InvariantFloatEquality;

impl Rule for InvariantFloatEquality {
    fn id(&self) -> &str {
        "invariant-float-equality"
    }

    fn spec_ref(&self) -> &str {
        "§12.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "Exact `==` over a representation field in an invariant starves generation; use a tolerance band"
    }

    fn severity(&self) -> Severity {
        Severity::Advisory
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let Ok(decls) = chelis_surf::parser::parse_str(source) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        collect(&decls, source, ctx, &mut out);
        out
    }
}

fn collect(decls: &[Decl], source: &str, ctx: &Context<'_>, out: &mut Vec<Violation>) {
    for decl in decls {
        match decl {
            Decl::TypeDef {
                name,
                invariant: Some(inv),
                ..
            } if expr_has_field_equality(&inv.body) => {
                let (line, col) = line_col(source, inv.span.offset);
                out.push(Violation {
                    rule_id: "invariant-float-equality".to_string(),
                    spec_ref: "§12.1".to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line),
                    col: Some(col),
                    message: format!(
                        "invariant on `{name}` uses exact `==` over a representation field; \
                         exact float equality starves generation by design \
                         (use a tolerance band over a module constant)"
                    ),
                });
            }
            Decl::Module { decls, .. } => collect(decls, source, ctx, out),
            _ => {}
        }
    }
}

/// Whether an expression contains an `==` whose left or right operand is
/// a field access (`p.field`) — the float-field-equality shape.
fn expr_has_field_equality(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinOp::Eq, lhs, rhs, _) => {
            is_field_access(lhs)
                || is_field_access(rhs)
                || expr_has_field_equality(lhs)
                || expr_has_field_equality(rhs)
        }
        Expr::Binary(_, lhs, rhs, _) => {
            expr_has_field_equality(lhs) || expr_has_field_equality(rhs)
        }
        Expr::Unary(_, inner, _) => expr_has_field_equality(inner),
        Expr::Apply(func, args, _) => {
            expr_has_field_equality(func) || args.iter().any(expr_has_field_equality)
        }
        Expr::If(c, t, e, _) => {
            expr_has_field_equality(c) || expr_has_field_equality(t) || expr_has_field_equality(e)
        }
        Expr::Access(inner, _, _) => expr_has_field_equality(inner),
        _ => false,
    }
}

/// Whether `expr` is a field access on some base (`base.field`), or a
/// `sum(base.field)`-style call over one.
fn is_field_access(expr: &Expr) -> bool {
    match expr {
        Expr::Access(_, _, _) => true,
        Expr::Apply(_, args, _) => args.iter().any(is_field_access),
        _ => false,
    }
}

/// 1-based line and column of `offset` in `source`.
fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rule, Severity};
    use std::path::Path;

    fn run(src: &str) -> Vec<Violation> {
        let ctx = Context {
            root: Path::new("/"),
            path: Path::new("test.ch"),
            source: Some(src),
            surface: Surface::SurfSource,
        };
        InvariantFloatEquality.check(&ctx)
    }

    #[test]
    fn is_advisory() {
        assert_eq!(InvariantFloatEquality.severity(), Severity::Advisory);
    }

    #[test]
    fn flags_exact_field_equality() {
        let v = run("module M\n@opaque\n@invariant(p) p.value == 0.5\n\
             type Probability = | Probability { value: f32 }\n");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_id, "invariant-float-equality");
        assert!(v[0].message.contains("Probability"));
    }

    #[test]
    fn tolerance_band_is_silent() {
        // The documented idiom: no `==`, only bounds. No advisory.
        let v = run("module M\neps = 0.001\n@opaque\n\
             @invariant(p) (sum(p.weights) >= (1.0 - eps)) && (sum(p.weights) <= (1.0 + eps))\n\
             type Simplex = | Simplex { weights: tensor[3, f32] }\n");
        assert!(v.is_empty(), "tolerance band must not warn: {v:?}");
    }

    #[test]
    fn range_predicate_is_silent() {
        // Comparisons (no `==`) do not trip the advisory.
        let v = run(
            "module M\n@opaque\n@invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
             type Probability = | Probability { value: f32 }\n",
        );
        assert!(v.is_empty(), "range predicate must not warn: {v:?}");
    }
}
