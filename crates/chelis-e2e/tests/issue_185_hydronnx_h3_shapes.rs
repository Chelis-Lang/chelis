//! End-to-end Hydronnx H3.x acceptance shapes (issue chelis#185).
//!
//! Hydronnx H3's acceptance gate is `emit -> chelis test -> compare-to-
//! ONNX-Runtime`. The host-runtime gap that #185 closed (PR #217) wired
//! up builtin dispatch for every ONNX op the H3 emitter needs; this
//! file is the end-to-end lock that the original Hydronnx H3.x shapes
//! run through the host runtime AND agree with the IR evaluator (per
//! `feedback_evaluator_byte_identical_gate` -- the IR evaluator is
//! canonical, and both paths must produce the same bytes / be within
//! tolerance).
//!
//! Forward pass only. Adjoint paths for these ops are blocked on #199
//! Part 2 (deferred to #218); they are out of scope for this test.
//!
//! Three cases, one per Hydronnx H3.x shape category:
//!   * MaxPool 2x2 stride 2: decomposes into `shrink` + `stride` +
//!     `max_reduce` on a `[1, 4, 8, 8, f32]` input, producing
//!     `[1, 4, 4, 4, f32]`.
//!   * `layer_norm` on `[2, 4, f32]` with rank-1 gamma/beta of size 4.
//!   * `conv` on `[1, 3, 8, 8, f32]` with an `[8, 3, 3, 3, f32]`
//!     kernel, stride=1, padding=0 -> `[1, 8, 6, 6, f32]`.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};

fn eval_surf(source: &str) -> chelis_compiler_api::schema::EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"))
}

fn eval_surf_selected(source: &str, roots: &[&str]) -> chelis_compiler_api::schema::EvalResult {
    let selected: Vec<String> = roots.iter().map(|s| (*s).to_string()).collect();
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &selected,
    )
    .unwrap_or_else(|err| panic!("eval_selected failed: {err:?}"))
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

fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected}, got {actual} (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// Surf source builders. Building literal `to_tensor([[[[...]]]])` for a
// `[1, 4, 8, 8]` or `[1, 3, 8, 8]` fixture by hand is unreadable, so the
// helpers below emit the canonical nested-list literal from a flat data
// vector + the target shape. Each leaf scalar is wrapped in `cast(.., f32)`
// to match the to_tensor + f32-dtype rounding shape exercised elsewhere
// in the host-runtime suite.
// ---------------------------------------------------------------------------

/// Emit a Surf nested-list literal for a row-major `data` vector with the
/// given `shape`. Each scalar leaf becomes `cast(<value>, f32)`. The list
/// nesting depth equals `shape.len()`.
fn nested_list_literal(shape: &[usize], data: &[f32]) -> String {
    fn canonical_f32(value: f32) -> String {
        let mut text = value.to_string();
        if !text.contains(['.', 'e', 'E']) {
            text.push_str(".0");
        }
        text
    }

    fn recurse(shape: &[usize], data: &[f32]) -> String {
        if shape.is_empty() {
            return format!("cast({}, f32)", canonical_f32(data[0]));
        }
        let head = shape[0];
        let tail = &shape[1..];
        let inner_count: usize = tail.iter().product();
        let mut parts = Vec::with_capacity(head);
        for i in 0..head {
            let off = i * inner_count;
            let chunk = &data[off..off + inner_count];
            parts.push(recurse(tail, chunk));
        }
        format!("[{}]", parts.join(", "))
    }
    let expected: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expected,
        "data length {} doesn't match shape {shape:?} (expected {expected})",
        data.len()
    );
    recurse(shape, data)
}

/// Build a deterministic ramp tensor for the MaxPool fixture. Values are
/// `sin(i)` over the flat row-major index so adjacent positions have
/// different magnitudes and the max-of-window result depends on which
/// element wins. Returns the flat row-major data in f32.
fn maxpool_input_data() -> Vec<f32> {
    // n = 1 * 4 * 8 * 8 = 256.
    let n: usize = 4 * 8 * 8;
    (0..n).map(|i| (i as f32 * 0.137).sin()).collect()
}

/// Reference MaxPool 2x2 stride 2 over `[1, 4, 8, 8]` -> `[1, 4, 4, 4]`.
/// Pure Rust f32 reference; the test asserts the chelis host-runtime
/// output matches this elementwise within f32 tolerance.
fn maxpool_reference(input: &[f32]) -> Vec<f32> {
    let (n, c, h, w) = (1, 4, 8, 8);
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0.0f32; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut m = f32::NEG_INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let y = ohi * 2 + dy;
                            let x = owi * 2 + dx;
                            let idx = ((ni * c + ci) * h + y) * w + x;
                            if input[idx] > m {
                                m = input[idx];
                            }
                        }
                    }
                    let oidx = ((ni * c + ci) * oh + ohi) * ow + owi;
                    out[oidx] = m;
                }
            }
        }
    }
    out
}

/// Build a deterministic ramp for the conv input + kernel.
fn conv_input_data() -> Vec<f32> {
    // n = 1 * 3 * 8 * 8 = 192.
    let n: usize = 3 * 8 * 8;
    (0..n).map(|i| (i as f32 * 0.05) - 1.0).collect()
}

fn conv_kernel_data() -> Vec<f32> {
    let n: usize = 8 * 3 * 3 * 3;
    (0..n).map(|i| (i as f32 * 0.013).cos() * 0.5).collect()
}

/// Reference conv (NCHW, OIHW), stride=1, padding=0. Returns flat
/// row-major data of shape `[1, 8, 6, 6]`. Pure Rust f32 reference for
/// the host-runtime assertion.
fn conv_reference(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let (n, ic, ih, iw) = (1, 3, 8, 8);
    let (oc, _kic, kh, kw) = (8, 3, 3, 3);
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let mut out = vec![0.0f32; n * oc * oh * ow];
    for ni in 0..n {
        for oci in 0..oc {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut acc = 0.0f32;
                    for ici in 0..ic {
                        for kyi in 0..kh {
                            for kxi in 0..kw {
                                let iy = ohi + kyi;
                                let ix = owi + kxi;
                                let iidx = ((ni * ic + ici) * ih + iy) * iw + ix;
                                let kidx = ((oci * ic + ici) * kh + kyi) * kw + kxi;
                                acc += input[iidx] * kernel[kidx];
                            }
                        }
                    }
                    let oidx = ((ni * oc + oci) * oh + ohi) * ow + owi;
                    out[oidx] = acc;
                }
            }
        }
    }
    out
}

// ===========================================================================
// Case 1 -- MaxPool 2x2 stride=2 no padding via shrink + stride + max_reduce.
//
// The decomposition extracts the 4 sub-views of the input (one per pool-
// window position: top-left, top-right, bottom-left, bottom-right), each of
// shape [1, 4, 4, 4], then stacks them on a new last axis and takes
// `max_reduce` over that axis. The `shrink` calls drop the leading row/col
// before the `stride` so each sub-view picks the right phase shift; the
// `stride` calls subsample every other position along H and W; the
// `max_reduce` does the actual pool reduction.
// ===========================================================================

#[test]
fn hydronnx_h3_maxpool_2x2_stride2_no_padding_matches_ir_eval() {
    let input = maxpool_input_data();
    let expected = maxpool_reference(&input);
    let input_literal = nested_list_literal(&[1, 4, 8, 8], &input);

    // Note: `stride(_, step_0, step_1, ..., step_n_minus_1)` takes one
    // positive int64 stride per axis (rank-4 input -> 4 strides).
    // `shrink(_, bounds)` takes (tensor, List[List[int64]]). The chain
    // below builds (top-left, top-right, bottom-left, bottom-right)
    // sub-views from the same 4D input, each `[1, 4, 4, 4]`, then
    // concats them on a new last axis and `max_reduce`s along that
    // axis. The result has the H3.x MaxPool shape `[1, 4, 4, 4]`.
    let src = format!(
        r#"
x = to_tensor({input_literal})
shifted_col = shrink(&x, [[0i64, 1i64], [0i64, 4i64], [0i64, 8i64], [1i64, 8i64]])
shifted_row = shrink(&x, [[0i64, 1i64], [0i64, 4i64], [1i64, 8i64], [0i64, 8i64]])
shifted_both = shrink(&x, [[0i64, 1i64], [0i64, 4i64], [1i64, 8i64], [1i64, 8i64]])
tl = stride(&x, 1i64, 1i64, 2i64, 2i64)
tr = stride(&shifted_col, 1i64, 1i64, 2i64, 2i64)
bl = stride(&shifted_row, 1i64, 1i64, 2i64, 2i64)
br = stride(&shifted_both, 1i64, 1i64, 2i64, 2i64)
tl5 = reshape(&tl, [cast(1, int64), cast(4, int64), cast(4, int64), cast(4, int64), cast(1, int64)])
tr5 = reshape(&tr, [cast(1, int64), cast(4, int64), cast(4, int64), cast(4, int64), cast(1, int64)])
bl5 = reshape(&bl, [cast(1, int64), cast(4, int64), cast(4, int64), cast(4, int64), cast(1, int64)])
br5 = reshape(&br, [cast(1, int64), cast(4, int64), cast(4, int64), cast(4, int64), cast(1, int64)])
stacked = concat([tl5, tr5, bl5, br5], 4)
out = max_reduce(&stacked, 4)
"#,
    );
    let result = eval_surf(&src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 4, 4, 4], "maxpool output shape");
    assert_eq!(
        out.data.len(),
        expected.len(),
        "maxpool element count mismatch"
    );
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want as f64,
            1e-5,
            &format!("maxpool[{i}]"),
        );
    }
}

// ===========================================================================
// Case 2 -- layer_norm(x, gamma, beta) on a [2, 4] input with rank-1
// gamma and beta of size 4.
//
// Per `spec/05-risc-primitives.md` §4.4, layer_norm decomposes into
// mean -> sub -> mean(.^2) -> div by sqrt(var + eps) -> mul gamma -> add
// beta. The IR evaluator implements this, and the host runtime delegates
// to the same decomposition; this test pins the agreement between them.
// ===========================================================================

#[test]
fn hydronnx_h3_layer_norm_batch2_hidden4_matches_ir_eval() {
    // x[0, :] = [1.0, 2.0, 3.0, 4.0]   -> mean=2.5, var=1.25
    // x[1, :] = [4.0, 2.0, 0.0, 6.0]   -> mean=3.0, var=4.5
    // gamma = [1, 2, 3, 4], beta = [0.1, 0.2, 0.3, 0.4]
    let src = r#"
x = pad_sequences([[1.0, 2.0, 3.0, 4.0], [4.0, 2.0, 0.0, 6.0]], 0.0)
g = to_tensor([1.0, 2.0, 3.0, 4.0])
b = to_tensor([0.1, 0.2, 0.3, 0.4])
out = layer_norm(&x, &g, &b, 0.00001f32)
"#;
    let result = eval_surf(src);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![2, 4], "layer_norm shape");

    // Hand-computed reference using the spec's lowering with eps=1e-5.
    let eps = 1e-5_f64;
    let rows: [[f64; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [4.0, 2.0, 0.0, 6.0]];
    let gamma = [1.0_f64, 2.0, 3.0, 4.0];
    let beta = [0.1_f64, 0.2, 0.3, 0.4];
    let mut expected = Vec::with_capacity(8);
    for row in &rows {
        let mean = row.iter().sum::<f64>() / 4.0;
        let centered: Vec<f64> = row.iter().map(|v| v - mean).collect();
        let var = centered.iter().map(|c| c * c).sum::<f64>() / 4.0;
        let denom = (var + eps).sqrt();
        for j in 0..4 {
            expected.push((centered[j] / denom) * gamma[j] + beta[j]);
        }
    }
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want,
            1e-5,
            &format!("layer_norm[{i}]"),
        );
    }
}

// ===========================================================================
// Case 3 -- conv on [1, 3, 8, 8] with kernel [8, 3, 3, 3], stride=1,
// padding=0. Output shape per `floor((in + 2p - k) / s) + 1` = [1, 8, 6, 6].
//
// `conv`'s typer (`crates/chelis-types/src/infer.rs::
// conv_input_dims_concrete_modulo_batch`) needs explicit shape metadata
// on the call site, so the fixture wraps the call in a typed `def` whose
// parameters carry the full tensor[...] shape. The eval pipeline is
// driven with `eval_surf_selected` so the formal-parameter binding in
// `def run_conv` doesn't trip the "missing input" path -- only the
// `out` root is forward-evaluated.
//
// In addition to the elementwise reference comparison against the pure-
// Rust conv reference (locking the wiring against silent corruption),
// this test pins one output pixel against a hand-computed value to lock
// correctness, not just self-consistency between two paths through the
// same evaluator.
// ===========================================================================

#[test]
fn hydronnx_h3_conv_1x3x8x8_kernel_8x3x3x3_matches_ir_eval() {
    let input = conv_input_data();
    let kernel = conv_kernel_data();
    let expected = conv_reference(&input, &kernel);

    let input_literal = nested_list_literal(&[1, 3, 8, 8], &input);
    let kernel_literal = nested_list_literal(&[8, 3, 3, 3], &kernel);

    let src = format!(
        r#"
def run_conv(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
def make_x() -> tensor[1, 3, 8, 8, f32] = to_tensor({input_literal})
def make_k() -> tensor[8, 3, 3, 3, f32] = to_tensor({kernel_literal})
out = run_conv(make_x(), make_k())
"#,
    );
    let result = eval_surf_selected(&src, &["out"]);
    let out = root_tensor(&result, "out");
    assert_eq!(out.shape, vec![1, 8, 6, 6], "conv shape");
    assert_eq!(
        out.data.len(),
        expected.len(),
        "conv element count mismatch"
    );

    // Compare every output pixel against the pure-Rust reference.
    // Tolerance: 6x6 = 36 contributions per output channel; with f32
    // accumulators and values in roughly [-1.5, 1.5], 1e-4 absolute
    // tolerance comfortably covers the worst-case rounding noise.
    for (i, &want) in expected.iter().enumerate() {
        assert_close(
            out.data.element_f64_lossy(i),
            want as f64,
            1e-4,
            &format!("conv[{i}]"),
        );
    }

    // Hand-computed lock for the top-left output pixel of output channel 0:
    //   out[0, 0, 0, 0] = sum over (ic, ky, kx) of
    //     input[0, ic, ky, kx] * kernel[0, ic, ky, kx]
    // This is independent of the f32 reference -- it computes the same
    // sum in f64 from the deterministic data generators and pins the
    // wiring against a baseline that doesn't ride on the reference
    // implementation's correctness.
    // Output channel 0: kernel offset starts at oc * (3 * 3 * 3) = 0.
    let mut hand = 0.0_f64;
    for ic in 0..3usize {
        for ky in 0..3usize {
            for kx in 0..3usize {
                let i_in = input[(ic * 8 + ky) * 8 + kx] as f64;
                let i_k = kernel[(ic * 3 + ky) * 3 + kx] as f64;
                hand += i_in * i_k;
            }
        }
    }
    assert_close(
        out.data.element_f64_lossy(0),
        hand,
        1e-4,
        "conv hand-computed out[0, 0, 0, 0]",
    );
}
