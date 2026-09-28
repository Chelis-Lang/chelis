//! Bare unary-builtin pipe stages pass their input type to the
//! synthesized function before checking its body. `x |> copy`,
//! `x |> realize`, and `x |> cast(f32)` type-check for compatible
//! tensor inputs under `spec/01-nomenclature.md` §3.6.

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

/// `x |> realize` type-checks with a tensor input.
#[test]
fn pipe_realize_with_statically_typed_tensor_typechecks() {
    let result = typecheck_surf("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize");
    assert!(
        result.is_ok(),
        "x |> realize on tensor[3, f32] should type-check; errors={:?}",
        result.err()
    );
}

/// `x |> copy` type-checks with a tensor input.
#[test]
fn pipe_copy_with_statically_typed_tensor_typechecks() {
    let result = typecheck_surf("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy");
    assert!(
        result.is_ok(),
        "x |> copy on tensor[3, f32] should type-check; errors={:?}",
        result.err()
    );
}

/// A piped cast sees the input type before its synthesized function
/// body is checked.
#[test]
fn pipe_cast_with_statically_typed_tensor_typechecks() {
    let result = typecheck_surf("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> cast(f32)");
    assert!(
        result.is_ok(),
        "x |> cast(f32) on tensor[3, f32] should type-check; errors={:?}",
        result.err()
    );
}
