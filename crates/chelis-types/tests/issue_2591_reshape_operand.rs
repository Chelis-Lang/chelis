//! chelis#2591: `reshape`'s operand rule, [05-OP-49]. The operand is a
//! tensor, or a scalar of an active data element dtype, which `reshape`
//! reads as that dtype's rank-0 tensor (as `eval` does). Any other operand is
//! `reshape`'s own error, decided by one rule whichever route reaches it:
//! directly, through a lambda, or through a recursive group's omitted result.

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

/// The diagnostics of a program, the same on both checker ingresses
/// (chelis#1107).
fn diagnostics(source: &str) -> Vec<String> {
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
    let found = diagnostics(source);
    assert!(
        found.is_empty(),
        "must type-check:\n{source}\ngot {found:#?}"
    );
}

/// A rejection carrying a diagnostic that contains every fragment.
fn rejects_with(source: &str, fragments: &[&str]) {
    let found = diagnostics(source);
    assert!(
        found
            .iter()
            .any(|message| fragments.iter().all(|fragment| message.contains(fragment))),
        "expected a diagnostic containing all of {fragments:?} for:\n{source}\ngot {found:#?}"
    );
}

/// REGRESSION TEST (fails on `main` and on `c2ec7adca`, which checked
/// `reshape("s", [3i64])` at score 1 and then failed in `eval`). [05-OP-49]: `reshape`'s operand is a
/// tensor, or a scalar of an active data element dtype read as its rank-0
/// tensor. Any other operand is `reshape`'s own error on every route.
#[test]
fn reshape_rejects_a_non_tensor_operand_on_every_route() {
    for program in [
        "def main() -> i32 = {\n  r = reshape(\"s\", [3i64])\n  0i32\n}\n",
        "def main() -> i32 = {\n  g = fn (y) -> reshape(y, [3i64])\n  r = g(\"s\")\n  0i32\n}\n",
        "def f(n: i32) = if eq(n, 0) then \"s\" else {\n  k = f(n - 1)\n  r = reshape(k, [3i64])\n  \"s\"\n}\n",
        "def f(n: i32) = if eq(n, 0) then \"s\" else {\n  k = g(n - 1)\n  \"s\"\n}\n\n\
         def g(n: i32) -> i32 = if eq(n, 0) then 0i32 else {\n  r = reshape(f(n - 1), [3i64])\n  _ = debug(r)\n  0i32\n}\n",
        "def main() -> i32 = {\n  r = reshape([1i32, 2i32, 3i32], [3i64])\n  0i32\n}\n",
    ] {
        rejects_with(program, &["reshape expects tensor input, got"]);
    }
    for program in [
        "def main() -> tensor[3, f32] = reshape(to_tensor([1.0f32, 2.0f32, 3.0f32]), [3i64])\n",
        "def main() -> tensor[3, f32] = {\n  g = fn (y) -> reshape(y, [3i64])\n  g(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n}\n",
        "def f(n: i32) = if eq(n, 0) then to_tensor([1.0f32, 2.0f32, 3.0f32]) else {\n  k = f(n - 1)\n  reshape(k, [3i64])\n}\n",
        // A data scalar is its rank-0 tensor.
        "def main() -> tensor[1, f32] = reshape(1.5f32, [1i64])\n",
        "def main() -> tensor[1, bool] = reshape(true, [1i64])\n",
    ] {
        accepts(program);
    }
}
