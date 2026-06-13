//! Rule `unreachable-producer` — an invariant-carrying opaque type with
//! no exported producers (RFC D-PRODUCER / D-LINT). Such a type is fully
//! sealed (the sixth rejection makes internal producers uncallable from
//! outside) and its derived-obligation set is empty: the declaration is
//! effectively dead. The lint flags it so the author either exports a
//! producer or removes the type.
//!
//! Advisory-only and best-effort syntactic (local, no type context): a
//! "producer" here is an exported `def` whose declared return type
//! mentions the opaque type, or whose body syntactically constructs it
//! (a `Constructor { ... }` record literal of the type's variant). When
//! the producer is unannotated and constructs through a helper, the lint
//! may under-report — advisory, never blocking. The authoritative
//! producer set is `chelis prove`'s (which uses inferred types).

use crate::{Context, Rule, Severity, Surface, Violation};
use chelis_surf::ast::{Decl, Expr, TypeExpr};

pub struct UnreachableProducer;

impl Rule for UnreachableProducer {
    fn id(&self) -> &str {
        "unreachable-producer"
    }

    fn spec_ref(&self) -> &str {
        "§12.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "An opaque type with no exported producers is fully sealed and has an empty obligation set"
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
        // The lint reasons per module scope: collect the module's decls,
        // exports, and opaque types together.
        check_scope(&decls, source, ctx, &mut out);
        out
    }
}

fn check_scope(decls: &[Decl], source: &str, ctx: &Context<'_>, out: &mut Vec<Violation>) {
    // Recurse into nested modules as their own scopes first.
    for decl in decls {
        if let Decl::Module { decls: inner, .. } = decl {
            check_scope(inner, source, ctx, out);
        }
    }

    // Exported names in this scope.
    let mut exports = Vec::new();
    for decl in decls {
        if let Decl::Export { names, .. } = decl {
            exports.extend(names.iter().cloned());
        }
    }

    // Opaque invariant-carrying types declared in this scope, with their
    // variant constructor name.
    for decl in decls {
        let Decl::TypeDef {
            name,
            opaque: true,
            invariant: Some(_),
            variants,
            span,
            ..
        } = decl
        else {
            continue;
        };
        let ctor_names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
        if has_exported_producer(decls, &exports, name, &ctor_names) {
            continue;
        }
        let (line, col) = line_col(source, span.offset);
        out.push(Violation {
            rule_id: "unreachable-producer".to_string(),
            spec_ref: "§12.1".to_string(),
            path: ctx.path.to_path_buf(),
            line: Some(line),
            col: Some(col),
            message: format!(
                "opaque type `{name}` has no exported producers; it is fully sealed \
                 and its derived-obligation set is empty (export a producer or remove the type)"
            ),
        });
    }
}

/// Whether any exported `def` in the scope is a producer of `type_name`:
/// its declared return mentions the type, or its body constructs one of
/// the type's variants.
fn has_exported_producer(
    decls: &[Decl],
    exports: &[String],
    type_name: &str,
    ctor_names: &[&str],
) -> bool {
    for decl in decls {
        let Decl::FunDef {
            name, ret_ty, body, ..
        } = decl
        else {
            continue;
        };
        if !exports.iter().any(|e| e == name) {
            continue;
        }
        if ret_ty.as_ref().is_some_and(|t| type_mentions(t, type_name)) {
            return true;
        }
        if body_constructs(body, ctor_names) {
            return true;
        }
    }
    // An exported non-function binding of the type is also a producer.
    for decl in decls {
        if let Decl::LetDef { name, ty, .. } = decl
            && exports.iter().any(|e| e == name)
            && ty.as_ref().is_some_and(|t| type_mentions(t, type_name))
        {
            return true;
        }
    }
    false
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

/// Whether an expression syntactically constructs one of the named
/// constructors (a `Ctor { ... }` record literal). Best-effort: walks the
/// common producing shapes (block bodies, if-branches, match arms).
fn body_constructs(expr: &Expr, ctor_names: &[&str]) -> bool {
    match expr {
        // A record literal of the type's ctor, OR a record (e.g. a wrapper
        // `Wrapper { inner: T { .. } }`) whose field values construct it.
        Expr::Record(name, fields, _) => {
            ctor_names.contains(&name.as_str())
                || fields.iter().any(|(_, v)| body_constructs(v, ctor_names))
        }
        Expr::Apply(func, args, _) => {
            // `Some(Ctor { .. })` and other wrapping applications.
            (matches!(func.as_ref(), Expr::Var(n, _) if ctor_names.contains(&n.as_str())))
                || body_constructs(func, ctor_names)
                || args.iter().any(|a| body_constructs(a, ctor_names))
        }
        Expr::Block(bindings, body, _) => {
            bindings
                .iter()
                .any(|b| body_constructs(&b.value, ctor_names))
                || body_constructs(body, ctor_names)
        }
        Expr::If(_, t, e, _) => body_constructs(t, ctor_names) || body_constructs(e, ctor_names),
        Expr::Match(_, arms, _) => arms.iter().any(|a| body_constructs(&a.body, ctor_names)),
        Expr::Binary(_, l, r, _) => {
            body_constructs(l, ctor_names) || body_constructs(r, ctor_names)
        }
        Expr::Tuple(items, _) => items.iter().any(|i| body_constructs(i, ctor_names)),
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
        UnreachableProducer.check(&ctx)
    }

    #[test]
    fn is_advisory() {
        assert_eq!(UnreachableProducer.severity(), Severity::Advisory);
    }

    #[test]
    fn flags_opaque_with_no_exported_producer() {
        // `probability` is NOT exported => no exported producer.
        let src = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_id, "unreachable-producer");
    }

    #[test]
    fn no_flag_when_producer_exported_by_return_annotation() {
        let src = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";
        assert!(run(src).is_empty());
    }

    #[test]
    fn no_flag_when_producer_exported_via_constructing_body() {
        // Unannotated return, but the exported body constructs the type.
        let src = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) = Probability { value: x }
";
        assert!(run(src).is_empty());
    }

    #[test]
    fn no_flag_when_producer_wraps_the_type_in_a_record_field() {
        // RT-2: an exported producer that constructs the opaque type INSIDE
        // a non-generic wrapper record IS a producer; the lint must not
        // falsely assert the type is sealed with an empty obligation set.
        let src = "module M
export (make_wrapped)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper = | Wrapper { inner: T }
def make_wrapped(x: f32) -> Wrapper = Wrapper { inner: T { value: x } }
";
        assert!(
            run(src).is_empty(),
            "wrapper-constructing producer is reachable; no unreachable flag"
        );
    }

    #[test]
    fn no_flag_for_opaque_without_invariant() {
        // Plain opacity (no @invariant) is out of scope for this lint.
        let src = "module M.Plain
@opaque
type Token =
  | Token { id: int32 }
def make(i: int32) -> Token = Token { id: i }
";
        assert!(run(src).is_empty());
    }
}
