//! Rule `opaque-escape-site` — the complete argument-egress enumeration
//! for invariant-carrying opaque types (RFC D-SOUND / D-LINT).
//!
//! Return-egress of an opaque type is mechanically obligated (the derived
//! producer obligations). Argument-egress is NOT: in-module code that
//! passes a value of the type (or a function value capable of producing
//! it) as an argument to an out-of-module callee is trusted by module
//! audit. This lint is that audit surface: it enumerates **every**
//! in-module argument-egress site, at two levels:
//!
//! - **note** — the value is locally attested: it traces (one hop) to a
//!   producer call or to a type-T input parameter of the enclosing
//!   function.
//! - **warning** — the value is unattested: it traces to a raw
//!   construction (`Ctor { ... }`) or a representation update.
//!
//! A call through a function-typed value of unknown provenance (a
//! function parameter, a stored closure) counts as out-of-module —
//! fail-closed. Local attestation is one-hop by design (RFC D-SOUND): a
//! helper `g(p: T) = outside.f(p)` attests its own egress and the audit
//! follows callers of `g`; transitive flows are the module audit's job.
//!
//! Enumeration is complete over the per-function-body local dataflow:
//! every `Apply` whose callee is out-of-module (or an unknown function
//! value) and that passes a T-valued argument or a T-producing function
//! value is reported. The provenance labels are local one-hop, as the RFC
//! specifies; they are advisory direction for the audit, never blocking.

use std::collections::HashMap;

use crate::{Context, Rule, Severity, Surface, Violation};
use chelis_surf::ast::{Decl, Expr, Param, TypeExpr, VariantFields};

pub struct OpaqueEscapeSite;

impl Rule for OpaqueEscapeSite {
    fn id(&self) -> &str {
        "opaque-escape-site"
    }

    fn spec_ref(&self) -> &str {
        "§12.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "Enumerates in-module argument-egress sites of an opaque type passed to out-of-module callees"
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
        check_scope(&decls, source, ctx, &mut out);
        out
    }
}

/// Provenance of a tracked T-value within one function body.
#[derive(Clone, Copy, PartialEq)]
enum Provenance {
    /// Traces to a producer call or a type-T input parameter (one hop).
    Attested,
    /// Traces to a raw construction or representation update.
    Unattested,
}

struct Scope<'a> {
    /// The invariant-carrying opaque type names declared in this module.
    opaque_types: Vec<String>,
    /// Constructor name -> opaque type it constructs (raw construction).
    ctor_to_type: HashMap<String, String>,
    /// Names of defs DEFINED in this module (callees not here are
    /// out-of-module).
    local_defs: Vec<String>,
    /// Names of module producers: defs whose declared return mentions an
    /// opaque type (a producer call attests its result).
    producer_defs: Vec<String>,
    source: &'a str,
}

fn check_scope(decls: &[Decl], source: &str, ctx: &Context<'_>, out: &mut Vec<Violation>) {
    for decl in decls {
        if let Decl::Module { decls: inner, .. } = decl {
            check_scope(inner, source, ctx, out);
        }
    }

    let opaque_types: Vec<String> = decls
        .iter()
        .filter_map(|d| match d {
            Decl::TypeDef {
                name,
                opaque: true,
                invariant: Some(_),
                ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect();
    if opaque_types.is_empty() {
        return;
    }

    let mut ctor_to_type = HashMap::new();
    for d in decls {
        if let Decl::TypeDef {
            name,
            opaque: true,
            invariant: Some(_),
            variants,
            ..
        } = d
        {
            for v in variants {
                if matches!(v.fields, VariantFields::Record(_)) {
                    ctor_to_type.insert(v.name.clone(), name.clone());
                }
            }
        }
    }

    let local_defs: Vec<String> = decls
        .iter()
        .filter_map(|d| match d {
            Decl::FunDef { name, .. } | Decl::LetDef { name, .. } | Decl::Sig { name, .. } => {
                Some(name.clone())
            }
            _ => None,
        })
        .collect();

    let producer_defs: Vec<String> = decls
        .iter()
        .filter_map(|d| match d {
            Decl::FunDef {
                name,
                ret_ty: Some(rt),
                ..
            } if opaque_types.iter().any(|t| type_mentions(rt, t)) => Some(name.clone()),
            Decl::Sig { name, ty, .. } if opaque_types.iter().any(|t| type_mentions(ty, t)) => {
                Some(name.clone())
            }
            _ => None,
        })
        .collect();

    let scope = Scope {
        opaque_types,
        ctor_to_type,
        local_defs,
        producer_defs,
        source,
    };

    for d in decls {
        if let Decl::FunDef { params, body, .. } = d {
            let mut env: HashMap<String, Provenance> = HashMap::new();
            // Type-T input parameters are attested.
            for p in params {
                if param_is_opaque(p, &scope) {
                    env.insert(p.name.clone(), Provenance::Attested);
                }
            }
            walk_expr(body, &mut env, &scope, ctx, out);
        }
    }
}

/// Walk a function-body expression, threading the local T-value
/// provenance environment and emitting egress sites.
fn walk_expr(
    expr: &Expr,
    env: &mut HashMap<String, Provenance>,
    scope: &Scope<'_>,
    ctx: &Context<'_>,
    out: &mut Vec<Violation>,
) {
    match expr {
        Expr::Block(bindings, body, _) => {
            for b in bindings {
                // First evaluate the binding value for nested egress.
                walk_expr(&b.value, env, scope, ctx, out);
                if let chelis_surf::ast::LetPattern::Var(name, _) = &b.pattern
                    && let Some(prov) = value_provenance(&b.value, env, scope)
                {
                    env.insert(name.clone(), prov);
                }
            }
            walk_expr(body, env, scope, ctx, out);
        }
        Expr::Apply(callee, args, span) => {
            // Recurse first so nested egress sites are also reported.
            walk_expr(callee, env, scope, ctx, out);
            for a in args {
                walk_expr(a, env, scope, ctx, out);
            }
            if callee_is_out_of_module(callee, scope) {
                for a in args {
                    if let Some(prov) = egressing_value(a, env, scope) {
                        emit_site(span.offset, prov, scope, ctx, out);
                    }
                }
            }
        }
        Expr::Pipe(head, stages, _) => {
            walk_expr(head, env, scope, ctx, out);
            for s in stages {
                walk_expr(s, env, scope, ctx, out);
                // A pipe `x |> outside.f` passes x as the first arg of f.
                if callee_is_out_of_module(s, scope)
                    && let Some(prov) = egressing_value(head, env, scope)
                {
                    emit_site(expr_offset(s), prov, scope, ctx, out);
                }
            }
        }
        Expr::If(c, t, e, _) => {
            walk_expr(c, env, scope, ctx, out);
            walk_expr(t, env, scope, ctx, out);
            walk_expr(e, env, scope, ctx, out);
        }
        Expr::Match(scrut, arms, _) => {
            walk_expr(scrut, env, scope, ctx, out);
            for arm in arms {
                walk_expr(&arm.body, env, scope, ctx, out);
            }
        }
        Expr::Binary(_, l, r, _) => {
            walk_expr(l, env, scope, ctx, out);
            walk_expr(r, env, scope, ctx, out);
        }
        Expr::Unary(_, e, _)
        | Expr::Access(e, _, _)
        | Expr::TupleGet(e, _, _)
        | Expr::Cast(e, _, _)
        | Expr::Copy(e, _)
        | Expr::Borrow(e, _)
        | Expr::Annotate(e, _, _) => walk_expr(e, env, scope, ctx, out),
        Expr::Tuple(items, _) | Expr::List(items, _) | Expr::Par(items, _) => {
            for i in items {
                walk_expr(i, env, scope, ctx, out);
            }
        }
        Expr::Record(_, fields, _) => {
            for (_, v) in fields {
                walk_expr(v, env, scope, ctx, out);
            }
        }
        Expr::Lambda(_, body, _) => walk_expr(body, env, scope, ctx, out),
        _ => {}
    }
}

/// Provenance of a value-producing expression, when it is a T-value.
fn value_provenance(
    expr: &Expr,
    env: &HashMap<String, Provenance>,
    scope: &Scope<'_>,
) -> Option<Provenance> {
    match expr {
        // Raw construction => unattested.
        Expr::Record(name, _, _) if scope.ctor_to_type.contains_key(name) => {
            Some(Provenance::Unattested)
        }
        // A producer call => attested.
        Expr::Apply(callee, _, _) => {
            if let Expr::Var(n, _) = callee.as_ref()
                && scope.producer_defs.contains(n)
            {
                return Some(Provenance::Attested);
            }
            // `Some(Ctor { .. })` wrapping a raw construction stays
            // unattested (the rep is raw).
            None
        }
        // A reference to a tracked T-value carries its provenance.
        Expr::Var(name, _) => env.get(name).copied(),
        Expr::Annotate(inner, _, _) | Expr::Copy(inner, _) | Expr::Borrow(inner, _) => {
            value_provenance(inner, env, scope)
        }
        _ => None,
    }
}

/// The provenance with which an argument egresses, if it is a T-value or a
/// T-producing function value. Tracked variables, raw constructions, bare
/// constructors of T, and producer-def references all egress.
fn egressing_value(
    arg: &Expr,
    env: &HashMap<String, Provenance>,
    scope: &Scope<'_>,
) -> Option<Provenance> {
    match arg {
        Expr::Var(name, _) => {
            if let Some(p) = env.get(name) {
                return Some(*p);
            }
            // A producer-def name passed as a function VALUE is a
            // T-producing function value (egress, attested — it traces to
            // a producer).
            if scope.producer_defs.contains(name) {
                return Some(Provenance::Attested);
            }
            None
        }
        // The bare constructor of T passed as a function value: a
        // T-producing function value whose result is a raw construction
        // (unattested).
        Expr::Constructor(name, _) if scope.ctor_to_type.contains_key(name) => {
            Some(Provenance::Unattested)
        }
        // An inline construction or producer call passed directly.
        _ => value_provenance(arg, env, scope),
    }
}

/// Whether a call's callee is out-of-module (fail-closed): a name not
/// defined locally, or a non-name callee (a function-typed value of
/// unknown provenance — a stored closure, a higher-order result).
fn callee_is_out_of_module(callee: &Expr, scope: &Scope<'_>) -> bool {
    match callee {
        Expr::Var(name, _) => !scope.local_defs.contains(name),
        // A bare constructor callee is in-module construction, not an
        // out-of-module call.
        Expr::Constructor(_, _) => false,
        // Anything else (a call through a closure, an applied lambda, a
        // field-access callee) is unknown provenance => out-of-module.
        _ => true,
    }
}

fn param_is_opaque(p: &Param, scope: &Scope<'_>) -> bool {
    p.ty.as_ref()
        .is_some_and(|t| scope.opaque_types.iter().any(|name| type_mentions(t, name)))
}

fn emit_site(
    offset: usize,
    prov: Provenance,
    scope: &Scope<'_>,
    ctx: &Context<'_>,
    out: &mut Vec<Violation>,
) {
    let (line, col) = line_col(scope.source, offset);
    let message = match prov {
        Provenance::Attested => {
            "opaque-type value (or a T-producing function value) egresses to an \
             out-of-module callee here; locally attested (traces to a producer call \
             or a type-T input): covered by module audit, not by machine obligations"
                .to_string()
        }
        Provenance::Unattested => {
            "opaque-type value (or a T-producing function value) egresses to an \
             out-of-module callee here; UNATTESTED (traces to a raw construction or \
             representation update): audit this leak path"
                .to_string()
        }
    };
    out.push(Violation {
        rule_id: "opaque-escape-site".to_string(),
        spec_ref: "§12.1".to_string(),
        path: ctx.path.to_path_buf(),
        line: Some(line),
        col: Some(col),
        message,
    });
}

/// The byte offset of an expression's span. `Expr` has no `span()`
/// accessor, so match the span field positionally (it is the last field
/// of every variant).
fn expr_offset(expr: &Expr) -> usize {
    match expr {
        Expr::Lit(_, s)
        | Expr::Var(_, s)
        | Expr::Constructor(_, s)
        | Expr::Apply(_, _, s)
        | Expr::List(_, s)
        | Expr::Record(_, _, s)
        | Expr::Access(_, _, s)
        | Expr::TupleGet(_, _, s)
        | Expr::Binary(_, _, _, s)
        | Expr::Unary(_, _, s)
        | Expr::Pipe(_, _, s)
        | Expr::If(_, _, _, s)
        | Expr::Match(_, _, s)
        | Expr::Lambda(_, _, s)
        | Expr::Tuple(_, s)
        | Expr::Cast(_, _, s)
        | Expr::Grad(_, _, s)
        | Expr::Vmap(_, _, s)
        | Expr::Jit(_, s)
        | Expr::Realize(_, s)
        | Expr::Copy(_, s)
        | Expr::Borrow(_, s)
        | Expr::WithSeed(_, _, s)
        | Expr::WithDevice(_, _, s)
        | Expr::Par(_, s)
        | Expr::Annotate(_, _, s)
        | Expr::Block(_, _, s) => s.offset,
    }
}

fn type_mentions(ty: &TypeExpr, type_name: &str) -> bool {
    match ty {
        TypeExpr::Named(name, _) => name == type_name,
        TypeExpr::Tensor(dims, _, _) => dims.iter().any(|d| type_mentions(d, type_name)),
        TypeExpr::App(name, args, _) => {
            name == type_name || args.iter().any(|a| type_mentions(a, type_name))
        }
        TypeExpr::Arrow(args, ret, _) => {
            args.iter().any(|a| type_mentions(a, type_name)) || type_mentions(ret, type_name)
        }
        TypeExpr::Ref(inner, _) => type_mentions(inner, type_name),
        TypeExpr::Tuple(items, _) => items.iter().any(|t| type_mentions(t, type_name)),
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
    use crate::Rule;
    use std::path::Path;

    fn run(src: &str) -> Vec<Violation> {
        let ctx = Context {
            root: Path::new("/"),
            path: Path::new("test.ch"),
            source: Some(src),
            surface: Surface::SurfSource,
        };
        OpaqueEscapeSite.check(&ctx)
    }

    #[test]
    fn is_advisory() {
        assert_eq!(OpaqueEscapeSite.severity(), Severity::Advisory);
    }

    #[test]
    fn naive_direct_leak_of_raw_construction_is_unattested_warning() {
        // A raw construction passed directly to an out-of-module callee.
        let src = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def leak(x: f32) -> f32 = outside_sink(Probability { value: x })
";
        let v = run(src);
        assert_eq!(v.len(), 1, "exactly one egress site");
        assert!(v[0].message.contains("UNATTESTED"));
    }

    #[test]
    fn one_hop_laundered_leak_through_attested_helper() {
        // g(p: T) = outside.f(p): p is a type-T input => attested egress.
        // The audit then follows callers of g (transitive, not this lint).
        let src = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def launder(p: Probability) -> f32 = outside_sink(p)
";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("locally attested"));
    }

    #[test]
    fn producer_call_result_egress_is_attested() {
        let src = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
def use_it(x: f32) -> f32 = { p = probability(x); outside_sink(p) }
";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("locally attested"));
    }

    #[test]
    fn in_module_call_is_not_an_egress_site() {
        // Passing T to a LOCAL def is not argument-egress.
        let src = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def prob_value(p: Probability) -> f32 = p.value
def use_it(p: Probability) -> f32 = prob_value(p)
";
        assert!(run(src).is_empty(), "local call is not egress");
    }

    #[test]
    fn bare_constructor_passed_as_function_value_egresses() {
        // The bare constructor (a T-producing function value) handed to an
        // out-of-module HOF is an egress site (unattested — raw rep).
        let src = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def hof_leak(x: f32) -> f32 = outside_apply(Probability)
";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("UNATTESTED"));
    }

    #[test]
    fn no_sites_for_non_opaque_or_invariant_free() {
        let src = "module M.Open
type Pair =
  | Pair { a: f32 }
def leak(x: f32) -> f32 = outside_sink(Pair { a: x })
";
        assert!(run(src).is_empty(), "only invariant-carrying opaque types");
    }
}
