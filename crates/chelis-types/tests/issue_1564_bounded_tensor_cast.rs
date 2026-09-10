//! [05-OP-63] preserves shape; [04-DTYPE-2] constrains each instantiation.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{check_ir_program, check_typed_program};

fn check(source: &str, expected: Option<&str>) {
    let program = desugar_program(&parse_str(source).expect("Surf"));
    let messages = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => vec![],
        Err(result) => result
            .errors
            .iter()
            .map(|e| format!("{:?}: {}", e.kind, e.message))
            .collect::<Vec<_>>(),
    };
    let ir = messages(check_ir_program(&program));
    let typed = messages(check_typed_program(&program));
    assert_eq!(ir, typed, "ingress parity: {source}");
    match expected {
        None => assert!(ir.is_empty(), "{source}: {ir:?}"),
        Some(needle) => assert!(
            ir.iter().any(|m| m.contains(needle)),
            "{source}: expected {needle}, got {ir:?}"
        ),
    }
}

#[test]
fn bounded_tensor_targets_preserve_shape_and_instantiate_independently() {
    for (family, first, second) in [
        ("Float", "f32", "f64"),
        ("Float", "f16", "bf16"),
        ("Int", "int16", "int64"),
        ("Int", "int8", "int32"),
        ("Numeric", "f32", "int64"),
    ] {
        check(
            &format!(
                "def recast[p: {family}](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\na = recast(cast(to_tensor([1, 2]), {first}))\nb = recast(cast(to_tensor([3, 4]), {second}))\n"
            ),
            None,
        );
        check(
            &format!(
                "def convert[p: {family}](x: tensor[2, int32], witness: p) -> tensor[2, p] = cast(x, p)\na = convert(to_tensor([1, 2]), cast(0, {first}))\nb = convert(to_tensor([3, 4]), cast(0, {second}))\n"
            ),
            None,
        );
    }
}

#[test]
fn invalid_target_and_family_are_still_type_errors() {
    check(
        "def f[p](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\n",
        Some("04-DTYPE-1"),
    );
    check(
        "def f[p: Float](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\nout = f(to_tensor([1, 2]))\n",
        Some("Float"),
    );
    check(
        "def f[p: Int](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\nout = f(to_tensor([1.0f32, 2.0f32]))\n",
        Some("Int"),
    );
    check(
        "def f[p: Numeric](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\nout = f(to_tensor([true, false]))\n",
        Some("Numeric"),
    );
    check(
        "out = cast(to_tensor([1, 2]), float64)\n",
        Some("not a recognized primitive"),
    );
    check(
        "def f[p: Float](x: tensor[2, p]) -> tensor[3, p] = cast(x, p)\n",
        Some("DimensionMismatch"),
    );
}

#[test]
fn truncating_bounded_tensor_cast_requires_float_to_integer() {
    check(
        "def f[p: Float, q: Int](x: tensor[2, p], witness: q) -> tensor[2, q] = cast_trunc(x, q)\nout = f(to_tensor([1.5f32, -2.5f32]), 0i64)\n",
        None,
    );
    check(
        "def f[p: Int](x: tensor[2, f32], witness: p) -> tensor[2, p] = cast_trunc(x, p)\nout = f(to_tensor([1.5f32, -2.5f32]), 0i64)\n",
        None,
    );
    check(
        "def f[p: Float](x: tensor[2, p]) -> tensor[2, p] = cast_trunc(x, p)\n",
        Some("05-OP-6"),
    );
    check(
        "def f[p: Int](x: tensor[2, p]) -> tensor[2, p] = cast_trunc(x, p)\n",
        Some("05-OP-6"),
    );
}

#[test]
fn result_constraints_obey_bounded_target_family() {
    let declaration = "def convert[p: Float](x: tensor[2, int32]) -> tensor[2, p] = cast(x, p)\n";
    check(
        &format!(
            "{declaration}a: tensor[2, f32] = convert(to_tensor([1, 2]))\nb: tensor[2, f64] = convert(to_tensor([1, 2]))\n"
        ),
        None,
    );
    check(
        &format!("{declaration}a: tensor[2, int64] = convert(to_tensor([1, 2]))\n"),
        Some("TypeMismatch"),
    );
}
