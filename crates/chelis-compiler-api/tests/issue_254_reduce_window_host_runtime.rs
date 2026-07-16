//! Issue #254 surface acceptance — the host runtime dispatches the
//! four `reduce_window_*` builtins to a windowed reduction that
//! matches `chelis_ir::eval::reduce_window` byte-for-byte.
//!
//! Spec source of truth: `spec/05-risc-primitives.md` §2.3.1.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};

/// Evaluate selecting only `out`, mirroring `chelis eval --file`'s
/// behavior in `crates/chelis-cli/src/main.rs::try_eval`. Without
/// this filter the eval pipeline tries to forward-evaluate every
/// top-level binding including `make_x` (a function-bodied tensor
/// whose tensor input is a formal parameter with no bound value).
fn eval_surf(source: &str) -> chelis_compiler_api::schema::EvalResult {
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["out".to_string()],
    )
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"))
}

fn root_tensor<'a>(
    result: &'a chelis_compiler_api::schema::EvalResult,
    name: &str,
) -> &'a chelis_compiler_api::schema::TensorValue {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root {name} in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value,
        other => panic!("expected tensor for {name}, got {other:?}"),
    }
}

// -------- Positive coverage --------

/// Surface reproducer from the issue: rank-4 input + [2,2] window +
/// [1,1] strides + max → output spatial dims `input - window + 1`.
/// We use a 1x1x4x4 stand-in so the expected values are tractable.
#[test]
fn issue254_reduce_window_max_strided_overlap_matches_pool2d() {
    let src = r"
def make_x() -> tensor[1, 1, 4, 4, f32] = to_tensor([[[
    [cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)],
    [cast(5.0, f32), cast(6.0, f32), cast(7.0, f32), cast(8.0, f32)],
    [cast(9.0, f32), cast(10.0, f32), cast(11.0, f32), cast(12.0, f32)],
    [cast(13.0, f32), cast(14.0, f32), cast(15.0, f32), cast(16.0, f32)]
]]])
def run(x: tensor[1, 1, 4, 4, f32]) -> tensor[1, 1, 3, 3, f32] =
    reduce_window_max(&x, [2, 2], [1, 1])
out = run(make_x())
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 1, 3, 3], "Valid pooling output shape");
    assert_eq!(
        out.data,
        vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
    );
}

#[test]
fn issue254_reduce_window_min_runs_and_matches_ir_eval() {
    let src = r"
def make_x() -> tensor[1, 1, 3, 3, f32] = to_tensor([[[
    [cast(1.0, f32), cast(5.0, f32), cast(3.0, f32)],
    [cast(4.0, f32), cast(2.0, f32), cast(6.0, f32)],
    [cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)]
]]])
def run(x: tensor[1, 1, 3, 3, f32]) -> tensor[1, 1, 2, 2, f32] =
    reduce_window_min(&x, [2, 2], [1, 1])
out = run(make_x())
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // 2x2 mins of [[1,5,3],[4,2,6],[7,8,9]]:
    // [1,5,4,2]→1, [5,3,2,6]→2, [4,2,7,8]→2, [2,6,8,9]→2.
    assert_eq!(out.data, vec![1.0, 2.0, 2.0, 2.0]);
}

#[test]
fn issue254_reduce_window_sum_runs_and_matches_ir_eval() {
    let src = r"
def make_x() -> tensor[1, 1, 3, 3, f32] = to_tensor([[[
    [cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)],
    [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)],
    [cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)]
]]])
def run(x: tensor[1, 1, 3, 3, f32]) -> tensor[1, 1, 2, 2, f32] =
    reduce_window_sum(&x, [2, 2], [1, 1])
out = run(make_x())
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_eq!(out.data, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn issue254_reduce_window_mean_runs_and_matches_ir_eval() {
    let src = r"
def make_x() -> tensor[1, 1, 3, 3, f32] = to_tensor([[[
    [cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)],
    [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)],
    [cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)]
]]])
def run(x: tensor[1, 1, 3, 3, f32]) -> tensor[1, 1, 2, 2, f32] =
    reduce_window_mean(&x, [2, 2], [1, 1])
out = run(make_x())
";
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_eq!(out.data, vec![3.0, 4.0, 6.0, 7.0]);
}

// -------- Negative coverage --------

#[test]
fn issue254_reduce_window_rejects_window_larger_than_input_dim() {
    let src = r"
def make_x() -> tensor[1, 1, 2, 2, f32] = to_tensor([[[
    [cast(1.0, f32), cast(2.0, f32)],
    [cast(3.0, f32), cast(4.0, f32)]
]]])
def run(x: tensor[1, 1, 2, 2, f32]) -> tensor[1, 1, 1, 1, f32] =
    reduce_window_max(&x, [3, 3], [1, 1])
out = run(make_x())
";
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "reduce_window_max with window > input dim must fail, got {outcome:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_zero_stride() {
    let src = r"
def make_x() -> tensor[1, 1, 3, 3, f32] = to_tensor([[[
    [cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)],
    [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)],
    [cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)]
]]])
def run(x: tensor[1, 1, 3, 3, f32]) -> tensor[1, 1, 2, 2, f32] =
    reduce_window_max(&x, [2, 2], [1, 0])
out = run(make_x())
";
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "reduce_window_max with zero stride must fail, got {outcome:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_string_input() {
    let src = r#"out = reduce_window_max("not a tensor", [2, 2], [1, 1])"#;
    let outcome = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: src.to_string(),
        bindings: BTreeMap::new(),
    });
    assert!(
        outcome.is_err(),
        "reduce_window_max on string must fail, got {outcome:?}"
    );
}
