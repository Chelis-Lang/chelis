//! chelis#2524: [05-OP-60] and [05-OP-38] make every argument and result
//! type of the file, mapped-file and process builtins exact.
//!
//! Each builtin carried a generic scheme and an inference arm that returned
//! its result type without looking at its arguments, so an `f32` path, or a
//! path typed by a `Float` binder, checked at score 1 on `main` `32e4122e4`.
//! Each now carries its exact signature, and ordinary unification rejects any
//! other argument.
//!
//! Every program is asserted on both checker ingresses (chelis#1107), and each
//! rejection has an accepted twin.

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

/// `(call over the parameter k, the declared result)` for each builtin whose
/// first argument is a string path.
const PATH_CALLS: &[(&str, &str)] = &[
    ("read_file(k)", "string"),
    ("write_file(k, \"text\")", "unit"),
    ("read_lines(k)", "List[string]"),
    ("read_bytes(k)", "List[i64]"),
    ("file_exists(k)", "bool"),
    ("list_dir(k)", "List[string]"),
    ("mmap_file(k)", "MappedFile"),
    ("process_run(k, [\"--version\"])", "(i64, string, string)"),
];

/// REGRESSION TEST: a concrete non-string path is rejected.
#[test]
fn a_path_must_be_a_string() {
    for (call, result) in PATH_CALLS {
        rejects_with(
            &format!("def bad(k: f32) -> {result} ! {{IO}} = {call}"),
            &["[PrecisionMismatch] precision mismatch: expected string, got f32"],
        );
        accepts(&format!("def ok(k: string) -> {result} ! {{IO}} = {call}"));
    }
}

/// REGRESSION TEST: a path typed by an authored `Float` binder is rejected,
/// since no member of the family is a string.
#[test]
fn a_path_typed_by_a_float_binder_is_rejected() {
    for (call, result) in PATH_CALLS {
        rejects_with(
            &format!("def bad[p: Float](k: p) -> {result} ! {{IO}} = {call}"),
            &["dtype family `Float`", "cannot be instantiated at `string`"],
        );
    }
}

/// REGRESSION TEST: the mapped-file reads take the handle and exact `i64`
/// offsets and counts, and `process_run` a list of string arguments.
#[test]
fn the_remaining_arguments_are_exact() {
    rejects_with(
        "def bad(m: MappedFile) -> List[i64] ! {IO} = mmap_read(m, 0i32, 4i64)",
        &["expected i64, got i32"],
    );
    accepts("def ok(m: MappedFile) -> List[i64] ! {IO} = mmap_read(m, 0i64, 4i64)");
    rejects_with(
        "def bad(m: f32) -> i64 ! {IO} = mmap_len(m)",
        &["MappedFile"],
    );
    accepts("def ok(m: MappedFile) -> i64 ! {IO} = mmap_len(m)");
    rejects_with(
        "def bad(k: string) -> (i64, string, string) ! {IO} = process_run(k, \"hi\")",
        &["List string vs string"],
    );
}
