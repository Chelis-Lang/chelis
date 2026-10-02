//! [05-OP-75]: `clock_wall_read` and `clock_monotonic_read` take no argument
//! and return exactly `(i64, i64)`.
//!
//! Every program is asserted on both checker ingresses (chelis#1107), and each
//! rejection has an accepted twin. The `IO` effect is checked separately, in
//! `chelis-effects`.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{BUILTIN_NAMES, check_ir_program, check_typed_program};

const CLOCKS: [&str; 2] = ["clock_wall_read", "clock_monotonic_read"];

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

fn accepts(source: &str) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics.is_empty(),
        "must type-check:\n{source}\ngot {diagnostics:#?}"
    );
}

/// A rejection carrying a diagnostic that contains every fragment.
fn rejects_with(source: &str, fragments: &[&str]) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics
            .iter()
            .any(|message| fragments.iter().all(|fragment| message.contains(fragment))),
        "expected a diagnostic containing all of {fragments:?} for:\n{source}\ngot {diagnostics:#?}"
    );
}

/// Both reads are registered builtins; an unregistered lookalike is not, so
/// the membership assertion is not vacuous.
#[test]
fn the_clock_reads_are_closed_vocabulary_builtins() {
    for clock in CLOCKS {
        assert!(BUILTIN_NAMES.contains(&clock), "{clock} must be a builtin");
    }
    assert!(!BUILTIN_NAMES.contains(&"clock_now"));
    rejects_with(
        "def f() -> (i64, i64) ! {IO} = clock_now()",
        &["[UnboundVariable]", "clock_now"],
    );
}

/// REGRESSION TEST: the result is exactly `(i64, i64)`, destructurable into
/// two `i64` halves.
#[test]
fn the_result_is_exactly_two_i64_halves() {
    for clock in CLOCKS {
        accepts(&format!("def ok() -> (i64, i64) ! {{IO}} = {clock}()"));
        accepts(&format!(
            "def seconds() -> i64 ! {{IO}} = {{\n  (s, n) = {clock}()\n  s\n}}"
        ));
        for declared in ["(i64, i32)", "i64", "(i64, i64, i64)", "(f64, i64)"] {
            rejects_with(
                &format!("def bad() -> {declared} ! {{IO}} = {clock}()"),
                &[
                    "[TypeMismatch]",
                    "body has type `() -> (i64, i64)`",
                    &format!("declared type is `() -> {declared}`"),
                ],
            );
        }
    }
}

/// REGRESSION TEST: a clock read takes no argument, so any argument is an
/// arity error rather than an ignored value.
#[test]
fn a_clock_read_takes_no_argument() {
    for clock in CLOCKS {
        for argument in ["0i64", "\"wall\"", "()"] {
            rejects_with(
                &format!("def bad() -> (i64, i64) ! {{IO}} = {clock}({argument})"),
                &["[ArityMismatch]", "expected 0 args, got 1"],
            );
        }
    }
}

/// A bare reference is the function value, not a read: it is typed as the
/// nullary function and is not a `(i64, i64)`.
#[test]
fn a_bare_reference_is_not_a_reading() {
    for clock in CLOCKS {
        rejects_with(
            &format!("def bad() -> (i64, i64) ! {{IO}} = {clock}"),
            &["body has type `() -> () -> (i64, i64)`"],
        );
    }
}
