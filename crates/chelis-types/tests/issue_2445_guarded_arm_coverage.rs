//! A guarded arm covers nothing: spec/04-type-system.md section 2.4 and
//! [04-PAT-2].
//!
//! A guard can be `false`, and then matching continues with the next arm. So
//! a guarded arm cannot be what makes a `match` exhaustive: counting it would
//! accept a program that runs out of arms at run time. Before the guard was
//! evaluated (chelis#2445), counting it was merely wrong on paper, because a
//! guarded arm was selected whenever its pattern matched.
//!
//! Each program is checked on both checker ingresses, which must agree. The
//! claim is the programs listed here.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("surf must parse: {source}\n{e:?}"));
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn expanded(source: &str) -> Vec<Expr> {
    expand_program(&desugared(source), &ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs()
}

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

/// Every diagnostic for `source`, having first asserted that the stamped
/// ingress and the normalizing ingress agree on it.
fn agreed_diagnostics(source: &str) -> Vec<String> {
    let typed = match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    assert_eq!(
        typed, ir,
        "both checker ingresses must return the same diagnostics for:\n{source}"
    );
    typed
}

const SHAPE: &str = "type Shape =\n  | Circle(f32)\n  | Square(f32)\n";

/// `(program, variant the checker must report missing)`: each leaves a
/// variant covered only by a guarded arm.
fn guarded_only_programs() -> Vec<(String, &'static str)> {
    vec![
        (
            "def big(o: Option[i64]) -> i32 =\n  match o with {\n    | Some(v) if gt(v, 10i64) => 1\n    | None => 3\n  }\n"
                .to_string(),
            "\"Some\"",
        ),
        (
            format!(
                "{SHAPE}def classify(s: Shape) -> i32 =\n  match s with {{\n    | Circle(r) if gt(r, 1.0) => 1\n    | Square(w) => 3\n  }}\n"
            ),
            "\"Circle\"",
        ),
        (
            format!(
                "{SHAPE}def any_shape(s: Shape, c: bool) -> i32 =\n  match s with {{\n    | whole if c => 1\n    | Square(w) => 3\n  }}\n"
            ),
            "\"Circle\"",
        ),
        // A guarded irrefutable arm alone covers nothing at all.
        (
            format!(
                "{SHAPE}def any_shape(s: Shape, c: bool) -> i32 =\n  match s with {{\n    | whole if c => 1\n  }}\n"
            ),
            "\"Circle\", \"Square\"",
        ),
    ]
}

/// The same program with an unguarded wildcard arm appended before the
/// closing brace of its `match`.
fn with_fallback(program: &str) -> String {
    let close = program.rfind("  }\n").expect("the match closes");
    format!("{}    | _ => 0\n{}", &program[..close], &program[close..])
}

/// REGRESSION TEST. Each program leaves a variant covered only by a guarded
/// arm, and each checked with no diagnostic on the base sha (6d1a9d513).
#[test]
fn a_variant_covered_only_by_a_guarded_arm_is_missing() {
    for (source, missing) in guarded_only_programs() {
        let diagnostics = agreed_diagnostics(&source);
        assert!(
            diagnostics.iter().any(|message| {
                message.starts_with("[NonExhaustiveMatch]") && message.contains(missing)
            }),
            "a guarded arm must not cover {missing}:\n{source}\ngot {diagnostics:?}"
        );
    }
}

/// Disposition lock. The same programs check clean once an unguarded
/// wildcard arm follows, as they did on the base sha, so the guard is still
/// typed and the guarded arm still binds for its guard and body.
#[test]
fn the_same_programs_with_an_unguarded_fallback_check_clean() {
    for (source, _) in guarded_only_programs() {
        let source = with_fallback(&source);
        let diagnostics = agreed_diagnostics(&source);
        assert!(diagnostics.is_empty(), "{source}\ngot {diagnostics:?}");
    }
}

/// Disposition lock. A guarded arm followed by an unguarded arm for the same
/// constructor checks clean, as it did on the base sha.
#[test]
fn a_guarded_arm_then_an_unguarded_arm_for_the_same_constructor_checks_clean() {
    let cases = [
        "def big(o: Option[i64]) -> i32 =\n  match o with {\n    | Some(v) if gt(v, 10i64) => 1\n    | Some(v) => 2\n    | None => 3\n  }\n"
            .to_string(),
        format!(
            "{SHAPE}def classify(s: Shape) -> i32 =\n  match s with {{\n    | Circle(r) if gt(r, 1.0) => 1\n    | Circle(r) => 2\n    | Square(w) => 3\n  }}\n"
        ),
    ];
    for source in cases {
        let diagnostics = agreed_diagnostics(&source);
        assert!(diagnostics.is_empty(), "{source}\ngot {diagnostics:?}");
    }
}
