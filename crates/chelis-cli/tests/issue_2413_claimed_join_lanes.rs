//! Standing PR canary for claimed joins. The complete 100-cell matrix lives in
//! `issue_2413_claimed_join_lanes_full`. A direct change selects that target
//! in required change-owned CI; otherwise affected-package final expansion
//! and nightly run it. The standing cases keep the reviewer-found wrong values
//! and their success/trap controls visible on every pull request.

#[path = "common/mod.rs"]
mod common;

include!("support/claimed_join.rs");

/// A false claim in an untaken `else` arm must not size the taken arm.
/// Matching and taken mismatching controls prevent an always-refuse or
/// always-ignore implementation from satisfying the canary.
#[test]
fn shared_origin_untaken_zero_claims_keep_the_taken_value() {
    assert_eq!(cells().len(), 100, "full claimed-join matrix dimensions");
    let mut cells = Vec::new();
    for claim in [Claim::Literal(0), Claim::Runtime(0)] {
        for source in [Source::Result, Source::Ascription] {
            cells.push(cell(false, claim, false, source, true));
        }
    }
    cells.push(cell(
        true,
        Claim::Literal(COMPUTED),
        true,
        Source::Result,
        true,
    ));
    cells.push(cell(true, Claim::Literal(0), true, Source::Result, true));
    assert_eq!(cells.len(), 6);
    let failures = disagreements(&cells);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A declared extent on the join itself, by a local ascription or by the
/// result type, is a claim on an active node: the taken arm's three elements
/// violate the declared 0, so every lane traps with the typed extent message,
/// whatever the untaken arm claims, or refuses the join: these arms' extents
/// (a runtime `shrink` against a claimed 0) are not proven equal.
///
/// Evidentiary status: REGRESSION TEST: at 44c9b23e7 the host interpreter
/// returned `0.0` and an empty tensor, the join's extent read from the
/// untaken arm's zeros.
#[test]
fn a_declared_join_extent_is_checked_against_the_taken_arm() {
    let x = x32();
    let elements = x
        .iter()
        .map(|value| format!("{value:?}f32"))
        .collect::<Vec<_>>();
    let literal = format!("to_tensor([{}])", elements.join(", "));
    let helper = "def zero(y: tensor[*, f32]) -> tensor[0, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 31i64)]])\n";
    let join = format!("if lt(0.0f32, s) then {OTHER} else zero(x)");
    let cells = [
        (
            "ascription on the join",
            format!(
                "{helper}def selected(x: tensor[32, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  r: tensor[0, f32] = {join}\n  sum(r, 0i32)\n}}\ndef main() -> tensor[f32] = selected({literal})\n"
            ),
        ),
        (
            "result type of the join",
            format!(
                "{helper}def selected(x: tensor[32, f32]) -> tensor[0, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  {join}\n}}\ndef main() -> tensor[0, f32] = selected({literal})\n"
            ),
        ),
    ]
    .map(|(name, source)| {
        let mut bindings = UnordMap::new();
        bindings.insert(
            "x".to_owned(),
            TensorValue::from_storage(
                vec![32],
                finalize_tensor(
                    "input",
                    Prim::F32,
                    RawTensor::Float(x.iter().map(|&v| f64::from(v)).collect()),
                )
                .unwrap(),
            ),
        );
        Cell {
            name: name.to_owned(),
            source,
            bindings,
            expected: Expected::Trap {
                claimed: 0,
                actual: 3,
            },
            proven: false,
            refused_in: &[],
        }
    });
    let failures = disagreements(&cells);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The round-2b reviewer's joins whose arms' extents are not proven equal:
/// a nested `if` whose inner `then` arm claims 0 (the taken inner `else`
/// holds `x[0..3]`), and a `vmap` row function whose arms hold 5 and 3
/// elements, the `else` arm unclaimed or claimed 3 by a callee's result
/// type. Every lane returns the taken arm's value or refuses the join.
///
/// Evidentiary status: REGRESSION TEST: at dfaefd9a2 the nested join
/// returned an empty tensor in the host interpreter and the DAG evaluator,
/// and each `vmap` join `[0, 0, 0]` in the DAG evaluator, the condition
/// sized as the larger arm.
#[test]
fn the_reviewers_unproven_joins_are_the_taken_arm_or_refused() {
    let x = x32();
    let elements = x
        .iter()
        .map(|value| format!("{value:?}f32"))
        .collect::<Vec<_>>();
    let literal = format!("to_tensor([{}])", elements.join(", "));
    let rows = [
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
        [9.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let matrix = format!(
        "to_tensor([{}])",
        rows.iter()
            .map(|row| {
                let row = row
                    .iter()
                    .map(|value| format!("{value:?}f32"))
                    .collect::<Vec<_>>();
                format!("[{}]", row.join(", "))
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Each row's sum decides its arm: the first five elements where it is
    // positive, else the first three.
    let row_sums = rows
        .iter()
        .map(|row| {
            let kept = if row.iter().sum::<f32>() > 0.0 { 5 } else { 3 };
            row[..kept].iter().sum::<f32>()
        })
        .collect::<Vec<_>>();
    let vmap_join = |else_arm: &str, helper: &str| {
        format!(
            "{helper}def rowf(r: tensor[8, f32]) -> tensor[f32] = {{\n  s = tensor_to_scalar(sum(copy(r), 0i32))\n  sum(if lt(0.0f32, s) then shrink(copy(r), [[0i64, sub(shape(&r, 0i32), 3i64)]]) else {else_arm}, 0i32)\n}}\ndef selected(xs: tensor[3, 8, f32]) -> tensor[3, f32] = vmap(rowf)(xs)\ndef main() -> tensor[3, f32] = selected({matrix})\n"
        )
    };
    let cells = [
        (
            "nested join, inner then arm claimed 0",
            format!(
                "def zero(y: tensor[*, f32]) -> tensor[0, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 31i64)]])\ndef selected(x: tensor[32, f32]) -> tensor[*, f32] = {{\n  s = tensor_to_scalar(sum(copy(x), 0i32))\n  if lt(0.0f32, s) then (if lt(s, 0.0f32) then zero(copy(x)) else {OTHER}) else zero(x)\n}}\ndef main() -> tensor[*, f32] = selected({literal})\n"
            ),
            ("x", vec![32], x.clone()),
            x[0..3].to_vec(),
        ),
        (
            "vmap row join, arms of 5 and 3",
            vmap_join(
                "shrink(copy(r), [[0i64, sub(shape(&r, 0i32), 5i64)]])",
                "",
            ),
            ("xs", vec![3, 8], rows.concat()),
            row_sums.clone(),
        ),
        (
            "vmap row join, else arm claimed 3",
            vmap_join(
                "three(r)",
                "def three(y: tensor[*, f32]) -> tensor[3, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 5i64)]])\n",
            ),
            ("xs", vec![3, 8], rows.concat()),
            row_sums,
        ),
    ]
    .map(|(name, source, (input, shape, values), expected)| {
        let mut bindings = UnordMap::new();
        bindings.insert(
            input.to_owned(),
            TensorValue::from_storage(
                shape,
                finalize_tensor(
                    "input",
                    Prim::F32,
                    RawTensor::Float(values.iter().map(|&v| f64::from(v)).collect()),
                )
                .unwrap(),
            ),
        );
        Cell {
            name: name.to_owned(),
            source,
            bindings,
            expected: Expected::Value(expected),
            proven: false,
            refused_in: &[],
        }
    });
    let failures = disagreements(&cells);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
