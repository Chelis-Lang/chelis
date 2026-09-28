//! Pin the type-checker behavior for bare unary-builtin pipe stages.
//!
//! Item 2b (PR #35) made `x |> copy` and `x |> realize` parse by
//! synthesizing `fn (v) -> (copy v)` / `fn (v) -> (realize v)` lambdas
//! over a fresh `__chelis_pipe` parameter. `chelis fmt` round-trips
//! cleanly. The 0.7.6 red team (PR #51 Finding 4) lifted that the
//! checker still rejects `x |> copy` with
//! `copy requires tensor input, got ?<freshvar>` even when the upstream
//! pipe value is statically typed. The control case `x |> realize`
//! passes because `infer_realize` just forwards the inner type without
//! a tensor-shape gate.
//!
//! Spec basis: `spec/01-nomenclature.md` §3.6 (first-argument
//! insertion makes `x |> copy` ≡ `copy(x)`); spec
//! `spec/design/implicit_linearity.md` §3 (explicit `copy` lowers to
//! `RiscOp::Copy` and is well-typed on any tensor input).

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_typed_program;

fn typecheck_surf(source: &str) -> Result<(), Vec<chelis_types::errors::CheckError>> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    check_typed_program(&deep)
        .map(|_| ())
        .map_err(|result| result.errors)
}

/// Control case: `x |> realize` with a statically-typed tensor input
/// type-checks today and must continue to type-check after the fix.
/// `infer_realize` is permissive (it forwards the inner type without a
/// shape gate), so the synthesized lambda's fresh-var body type is
/// resolved later when pipe-stage unification binds the parameter to
/// the upstream pipe value's type.
#[test]
fn pipe_realize_with_statically_typed_tensor_typechecks() {
    let result = typecheck_surf("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize");
    assert!(
        result.is_ok(),
        "x |> realize on tensor[3, f32] should type-check; errors={:?}",
        result.err()
    );
}

/// Finding 4: `x |> copy` on a statically-typed tensor input must
/// type-check after the fix. The synthesized lambda body
/// `(copy (var __chelis_pipe))` infers `__chelis_pipe` as a fresh type
/// variable; `infer_copy`'s tensor-only gate then rejects before
/// pipe-stage unification binds the parameter to the upstream tensor
/// type. The fix unifies the lambda parameter with the upstream pipe
/// value's type before the body is checked, so by the time `copy`
/// resolves its inner type the parameter is already a `Tensor`.
#[test]
fn pipe_copy_with_statically_typed_tensor_typechecks() {
    let result = typecheck_surf("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy");
    assert!(
        result.is_ok(),
        "x |> copy on tensor[3, f32] should type-check; errors={:?}",
        result.err()
    );
}

/// TypeCheck-PipeCast-F1 (docs/archive/reports/gap_synthesis.md §5): `x |> cast(f32)` is
/// the same shape of bug — the parser synthesizes
/// `fn (v) -> (cast v f32)` and `infer_cast` rejects the fresh-var
/// body before pipe-stage unification runs. The pipe-stage parameter
/// pre-unification fix closes both the §5 entry and Finding 4. Kept
/// ignored until the fix lands so a single fixture pins both.
#[test]
fn pipe_cast_with_statically_typed_tensor_typechecks() {
    let result = typecheck_surf("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> cast(f32)");
    assert!(
        result.is_ok(),
        "x |> cast(f32) on tensor[3, f32] should type-check; errors={:?}",
        result.err()
    );
}
