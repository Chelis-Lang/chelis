//! Rule `opaque-without-invariant` — an opaque type with no declared
//! `@invariant` is informational (RFC D-LINT). Adding an invariant lets
//! `chelis prove` derive producer obligations; without one, the opaque
//! type carries only the construction guarantee.
//!
//! Advisory-only: wired through `registry::non_blocking_rules`, so it
//! never fails the style gate or `lint --check`.

use crate::{Context, Rule, Severity, Surface, Violation};
use chelis_surf::ast::Decl;

pub struct OpaqueWithoutInvariant;

impl Rule for OpaqueWithoutInvariant {
    fn id(&self) -> &'static str {
        "opaque-without-invariant"
    }

    fn spec_ref(&self) -> &'static str {
        "§12.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &'static str {
        "An opaque type without a declared @invariant carries only the construction guarantee"
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
                opaque: true,
                invariant: None,
                span,
                ..
            } => {
                let (line, col) = line_col(source, span.offset);
                out.push(Violation {
                    rule_id: "opaque-without-invariant".to_string(),
                    spec_ref: "§12.1".to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line),
                    col: Some(col),
                    message: format!(
                        "opaque type `{name}` has no declared `@invariant`; \
                         consider adding one so `chelis prove` can derive producer obligations"
                    ),
                });
            }
            Decl::Module { decls, .. } => collect(decls, source, ctx, out),
            _ => {}
        }
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
        OpaqueWithoutInvariant.check(&ctx)
    }

    #[test]
    fn is_advisory() {
        assert_eq!(OpaqueWithoutInvariant.severity(), Severity::Advisory);
    }

    #[test]
    fn flags_opaque_without_invariant() {
        let v = run("module M\n@opaque\ntype Probability = | Probability { value: f32 }\n");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_id, "opaque-without-invariant");
        assert!(v[0].message.contains("Probability"));
    }

    #[test]
    fn opaque_with_invariant_is_silent() {
        let v = run("module M\n@opaque\n@invariant(p) p.value >= 0.0\n\
             type Probability = | Probability { value: f32 }\n");
        assert!(
            v.is_empty(),
            "invariant-bearing opaque must not warn: {v:?}"
        );
    }

    #[test]
    fn non_opaque_type_is_silent() {
        let v = run("module M\ntype Point = | Point { x: f32, y: f32 }\n");
        assert!(v.is_empty(), "non-opaque type must not warn: {v:?}");
    }
}
