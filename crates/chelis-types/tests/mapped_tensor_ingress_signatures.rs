//! [05-OP-79] and [05-OP-80]: the signatures of the mapped-range reads.
//!
//! `mmap_tensor(mapped, offset, count, T)` returns a rank-one tensor of the
//! dtype its reserved fourth argument states; `mmap_text` and `mmap_sha256`
//! return a string. All three take the handle and exact `i64` offsets and
//! counts, and none carries an effect. Every program is asserted on both
//! checker ingresses (chelis#1107), and each rejection has an accepted twin.

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

fn rejects_with(source: &str, fragments: &[&str]) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics
            .iter()
            .any(|message| fragments.iter().all(|fragment| message.contains(fragment))),
        "expected a diagnostic containing all of {fragments:?} for:\n{source}\ngot {diagnostics:#?}"
    );
}

const DATA_DTYPES: [&str; 9] = [
    "f64", "f32", "f16", "bf16", "i64", "i32", "i16", "i8", "bool",
];

/// Every active data element dtype is a result dtype, and the result has
/// that dtype; a declared result of another dtype rejects.
#[test]
fn the_dtype_argument_states_the_result_dtype() {
    for dtype in DATA_DTYPES {
        accepts(&format!(
            "def ok(m: MappedFile, n: i64) -> tensor[*, {dtype}] = mmap_tensor(m, 0i64, n, {dtype})"
        ));
        let other = if dtype == "f32" { "f64" } else { "f32" };
        rejects_with(
            &format!(
                "def bad(m: MappedFile, n: i64) -> tensor[*, {other}] = mmap_tensor(m, 0i64, n, {dtype})"
            ),
            &[dtype, other],
        );
    }
}

/// A literal count is a literal extent, checked statically; a computed count
/// is a fresh extent that a declared extent guards at run time.
#[test]
fn a_literal_count_is_a_literal_extent() {
    accepts("def ok(m: MappedFile) -> tensor[3, f32] = mmap_tensor(m, 0i64, 3i64, f32)");
    rejects_with(
        "def bad(m: MappedFile) -> tensor[4, f32] = mmap_tensor(m, 0i64, 3i64, f32)",
        &["DimensionMismatch"],
    );
    accepts("def ok(m: MappedFile, n: i64) -> tensor[4, f32] = mmap_tensor(m, 0i64, n, f32)");
}

/// `key` and `string` are not data element dtypes.
#[test]
fn key_and_string_are_not_result_dtypes() {
    for dtype in ["key", "string"] {
        rejects_with(
            &format!(
                "def bad(m: MappedFile) -> tensor[*, f32] = mmap_tensor(m, 0i64, 1i64, {dtype})"
            ),
            &["mmap_tensor's dtype argument must be an active data element dtype"],
        );
    }
}

/// The dtype argument is required.
#[test]
fn the_dtype_argument_is_required() {
    rejects_with(
        "def bad(m: MappedFile) -> tensor[*, f32] = mmap_tensor(m, 0i64, 1i64)",
        &["mmap_tensor expects (mapped, offset, count, T)"],
    );
}

/// The operands are exact: the handle, then `i64` offset and count.
#[test]
fn the_operands_are_exact() {
    rejects_with(
        "def bad(m: MappedFile) -> tensor[*, f32] = mmap_tensor(m, 0i32, 1i64, f32)",
        &["mmap_tensor expects an i64 byte offset", "i32"],
    );
    rejects_with(
        "def bad(m: MappedFile) -> tensor[*, f32] = mmap_tensor(m, 0i64, 1i32, f32)",
        &["mmap_tensor expects an i64 element count", "i32"],
    );
    rejects_with(
        "def bad(m: string) -> tensor[*, f32] = mmap_tensor(m, 0i64, 1i64, f32)",
        &["mmap_tensor expects a MappedFile handle", "string"],
    );
    for builtin in ["mmap_text", "mmap_sha256"] {
        accepts(&format!(
            "def ok(m: MappedFile) -> string = {builtin}(m, 0i64, 4i64)"
        ));
        rejects_with(
            &format!("def bad(m: MappedFile) -> string = {builtin}(m, 0i32, 4i64)"),
            &["expected i64, got i32"],
        );
        rejects_with(
            &format!("def bad(m: MappedFile) -> List[i64] = {builtin}(m, 0i64, 4i64)"),
            &["string"],
        );
    }
}

/// The reads of an open mapping carry no effect, so a definition declared
/// pure may make them (chelis-effects owns the `IO` side of the contract).
#[test]
fn mapped_reads_are_pure() {
    accepts(
        "def ok(m: MappedFile) -> (tensor[*, i32], string, string) ! {} = (mmap_tensor(m, 0i64, 1i64, i32), mmap_text(m, 0i64, 1i64), mmap_sha256(m, 0i64, 1i64))",
    );
}

/// A dtype binder bounded by a family or an explicit set is a dtype
/// argument.
#[test]
fn a_bounded_dtype_binder_is_a_dtype_argument() {
    for bound in ["Float", "Int", "Numeric", "{f32, i16}"] {
        accepts(&format!(
            "def ok[p: {bound}](m: MappedFile, n: i64) -> tensor[*, p] = mmap_tensor(m, 0i64, n, p)"
        ));
    }
}

/// An unbounded binder could stand for `string`, so it is not a dtype
/// argument, and instantiating its definition at `string` cannot reach a run.
#[test]
fn an_unbounded_binder_is_not_a_dtype_argument() {
    rejects_with(
        "def bad[t](m: MappedFile, n: i64) -> tensor[*, t] = mmap_tensor(m, 0i64, n, t)",
        &[
            "mmap_tensor's dtype argument must be an active data element dtype",
            "not the unbounded binder `t`",
        ],
    );
    rejects_with(
        "def load[t](m: MappedFile, n: i64) -> tensor[*, t] = mmap_tensor(m, 0i64, n, t)\n\
         def use_it(m: MappedFile) -> tensor[*, string] = load(m, 1i64)",
        &["mmap_tensor's dtype argument must be an active data element dtype"],
    );
}
