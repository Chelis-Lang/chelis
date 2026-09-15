//! Issue Chelis-Lang/chelis#319 — host-eval `grad` through an IMPORTED,
//! precision-polymorphic `[s, d, p]` attention verb whose body is
//! `permute` + `matmul` + `softmax` + `matmul`.
//!
//! Regression shape (the issue's reproducer): a precision-polymorphic
//! verb `sdpa` is declared in a library module with a SEPARATE `sig`
//! (the canonical School `scaled_dot_product_attention` form, not inline
//! parameter annotations). A concrete-`f32` caller in another module
//! differentiates a scalar loss that routes through the imported `sdpa`.
//! Before the fix the host evaluator failed to lower the backward with
//!
//!   `host runtime could not lower `grad(...)` for evaluation:
//!    lower_matmul expects rank >= 2 tensors`
//!
//! while an INLINED-f32 reimplementation of the identical body grads
//! fine. Two compounding root causes, both specific to the separate-`sig`
//! form:
//!
//!   1. The type checker stamps a separate-`sig` def's parameter types
//!      onto its `fn` params but inferred the BODY against bare param
//!      tvars (the bare `fn` literal carries no inline annotations), so
//!      shape-sensitive body ops (`matmul`, `permute`) annotated as bare
//!      type variables. IR lowering then read a rank-0 `default_type()`
//!      off the `(t-var …)` annotation and `tier2::lower_matmul`
//!      panicked. Fixed in `chelis-types` (`annotate_fn_children` seeds
//!      `fn_env` with the declared sig param types before annotating the
//!      body).
//!   2. The checker renames the sig's precision variable `p` to a fresh
//!      internal name when stamping the resolved body types, so the
//!      call-site precision substitution (keyed on `p`) missed it and a
//!      shape-preserving op (`permute`) tripped the §5.8.1
//!      monomorphization tripwire. Fixed in `chelis-ir` (`permute` takes
//!      precision from its operand; the inline call site binds every body
//!      precision variable to the concrete call-site precision when the
//!      call is precision-monomorphic).
//!
//! Spec authority: spec/04-type-system.md §5.7 (grad), §5.8 / §5.8.1
//! (precision monomorphization); spec/05-risc-primitives.md (matmul /
//! permute lowering); the §B2 row of spec/design/grad_surface_audit.md.
//!
//! These tests drive the `chelis_compiler_api` host pipeline the issue's
//! reproducer used (`eval` for the single-module shape, and the full
//! `compile_reef_context` + `eval_in_context` reef pipeline for the true
//! cross-module-boundary shape), so they reproduce the failure directly.

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context, eval_in_context};
use tempfile::TempDir;

// ─── helpers ─────────────────────────────────────────────────────────────

fn try_eval(source: &str) -> Result<chelis_compiler_api::schema::EvalResult, String> {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .map_err(|err| format!("{err:?}"))
}

fn out_tensor(result: &chelis_compiler_api::schema::EvalResult) -> TensorValue {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some("out"))
        .unwrap_or_else(|| panic!("missing `out` root in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value.clone(),
        other => panic!("expected tensor for `out`, got {other:?}"),
    }
}

/// The exact School `scaled_dot_product_attention` body: a `permute` +
/// two `matmul`s around a `softmax`. `scale` is the `[s, s]` score-scale
/// tensor.
const SDPA_BODY: &str = "{\n  \
  kt = permute(k, 1, 0)\n  \
  scores = matmul(q, kt)\n  \
  weights = softmax(mul(scores, scale), -1)\n  \
  matmul(weights, v)\n}";

/// Concrete-`f32` test inputs and the scalar-loss + `grad(loss, wrt=q)`
/// driver, parameterized over the callee name (so the imported and inline
/// forms share one driver and must produce identical gradients).
fn grad_driver(callee: &str) -> String {
    format!(
        "def loss(q: tensor[2, 3, f32], k: tensor[2, 3, f32], v: tensor[2, 3, f32], scale: tensor[2, 2, f32]) -> f32 =\n\
         \x20 tensor_to_scalar(sum(sum({callee}(q, k, v, scale), cast(0, int32)), cast(0, int32)))\n\
         out = grad(loss, wrt=q)(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), to_tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), to_tensor([[1.0, 0.0], [0.0, 1.0]]))\n",
    )
}

fn assert_close(actual: &TensorValue, expected: &TensorValue, tol: f64, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label}: shape mismatch");
    assert_eq!(
        actual.data.len(),
        expected.data.len(),
        "{label}: length mismatch"
    );
    for (i, (a, e)) in actual
        .data
        .to_f64_lossy_vec()
        .iter()
        .zip(expected.data.to_f64_lossy_vec().iter())
        .enumerate()
    {
        assert!(
            (a - e).abs() <= tol,
            "{label}: element {i} mismatch actual={a} expected={e}",
        );
    }
}

// ─── reef cross-module harness (mirrors compiled_context_compose_adversarial) ─

fn write_pkg(
    root: &Path,
    pkg_name: &str,
    module_prefix: &str,
    files: &[(&str, &str)],
    deps: &[(&str, &str)],
) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let mut manifest = format!(
        "[package]\nname = \"{pkg_name}\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"{module_prefix}\"\n"
    );
    if !deps.is_empty() {
        manifest.push_str("\n[dependencies]\n");
        for (name, path) in deps {
            manifest.push_str(&format!("{name} = {{ path = \"{path}\" }}\n"));
        }
    }
    fs::write(root.join("reef.toml"), manifest).expect("write reef.toml");
    for (rel, content) in files {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("mkdir parent");
        }
        fs::write(&p, content).expect("write source");
    }
}

fn app_reef_lock(pkg_name: &str) -> String {
    format!(
        "[package]\nname = \"{pkg_name}\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    )
}

fn build_pkg(library_src: &str, main_src: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    write_pkg(
        &root,
        "myapp",
        "App",
        &[("src/main.ch", main_src)],
        &[("mylib", "./mylib")],
    );
    let mylib = root.join("mylib");
    write_pkg(
        &mylib,
        "mylib",
        "Mylib",
        &[("src/attn.ch", library_src)],
        &[],
    );
    fs::write(root.join("reef.lock"), app_reef_lock("myapp")).expect("write reef.lock");
    (dir, root)
}

// =====================================================================
// Negative parity baseline: an INLINED-f32 reimplementation of the
// identical body already lowers and grads in the host evaluator. The
// issue isolates the trigger to the cross-module precision-poly verb by
// pointing at this control. Both forms MUST agree, so this provides the
// reference gradient the positive cases are checked against.
// =====================================================================

fn inline_f32_grad_result() -> TensorValue {
    let inline = format!(
        "def sdpa(q: tensor[s, d, f32], k: tensor[s, d, f32], v: tensor[s, d, f32], scale: tensor[s, s, f32]) -> tensor[s, d, f32] = {SDPA_BODY}\n{}",
        grad_driver("sdpa"),
    );
    let result = try_eval(&inline).unwrap_or_else(|err| {
        panic!("issue #319 control: inline-f32 sdpa grad must lower in host eval: {err}")
    });
    out_tensor(&result)
}

#[test]
fn issue_319_control_inline_f32_sdpa_grad_lowers() {
    let grad = inline_f32_grad_result();
    assert_eq!(grad.shape, vec![2, 3], "issue #319 control: d/dq shape");
    assert!(
        grad.data.to_f64_lossy_vec().iter().all(|v| v.is_finite()),
        "issue #319 control: gradient must be finite, got {:?}",
        grad.data,
    );
}

// =====================================================================
// Positive (single module, separate `sig`): the precision-polymorphic
// `[s, d, p]` verb written in the canonical separate-`sig` form grads in
// host eval WITHOUT the `lower_matmul expects rank >= 2` rank error or
// the §5.8.1 monomorphization tripwire — and produces the SAME gradient
// as the inline-f32 control.
// =====================================================================

#[test]
fn issue_319_separate_sig_precision_poly_sdpa_grad_matches_inline() {
    let separate_sig = format!(
        "sig sdpa[p: Float]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]\n\
         def sdpa(q, k, v, scale) = {SDPA_BODY}\n{}",
        grad_driver("sdpa"),
    );
    let result = try_eval(&separate_sig).unwrap_or_else(|err| {
        panic!(
            "issue #319: separate-sig precision-poly sdpa grad must lower in host eval \
             (no rank error, no monomorphization tripwire); got {err}"
        )
    });
    let grad = out_tensor(&result);
    assert_close(
        &grad,
        &inline_f32_grad_result(),
        1e-6,
        "issue #319 separate-sig vs inline-f32 grad parity",
    );
}

// =====================================================================
// Positive (true cross-module boundary): the verb lives in an IMPORTED
// reef library module and the differentiating caller lives in new code.
// This is the exact shape the issue reproduced — grad through an
// imported precision-poly verb. It must lower and match the inline-f32
// control.
// =====================================================================

#[test]
fn issue_319_imported_precision_poly_sdpa_grad_lowers_and_matches_inline() {
    let library = "module Mylib.Attn\nexport (sdpa)\n\n\
                   sig sdpa[p: Float]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]\n\
                   def sdpa(q, k, v, scale) = {\n  \
                     kt = permute(k, 1, 0)\n  \
                     scores = matmul(q, kt)\n  \
                     weights = softmax(mul(scores, scale), -1)\n  \
                     matmul(weights, v)\n}\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    let snippet = format!(
        "module App.Eval\nimport Mylib.Attn (sdpa)\n\n{}",
        grad_driver("sdpa"),
    );

    let ctx = compile_reef_context(Path::new("/tmp/issue319"), &root).expect("compile reef ctx");
    let result = eval_in_context(&ctx, &snippet).unwrap_or_else(|err| {
        panic!(
            "issue #319: grad through an IMPORTED precision-poly sdpa verb must lower \
             across the module boundary; got {err:?}"
        )
    });
    let grad = out_tensor(&result);
    assert_eq!(grad.shape, vec![2, 3], "issue #319 imported: d/dq shape");
    assert_close(
        &grad,
        &inline_f32_grad_result(),
        1e-6,
        "issue #319 imported vs inline-f32 grad parity",
    );
}

// =====================================================================
// Negative parity: the rank-guard must still fire for a genuinely
// ill-formed matmul (a rank-1 operand). The #319 fix recovers the
// declared body shapes for a well-formed precision-poly verb; it must
// not paper over a real rank violation. A `def` that matmuls a rank-1
// vector must still be rejected (clean error), never silently lowered.
// =====================================================================

#[test]
fn issue_319_negative_rank1_matmul_operand_still_rejected() {
    // `bad` matmuls a genuinely rank-1 operand `vec` — ill-formed
    // regardless of the #319 fix. The pipeline must reject it (a clean
    // error), and in particular must NOT lower it as if it were rank-2.
    let src = "def bad(vec: tensor[3, f32], m: tensor[3, 3, f32]) -> tensor[3, f32] = matmul(vec, m)\n\
               out = bad(to_tensor([1.0, 2.0, 3.0]), to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))\n";
    let outcome = try_eval(src);
    let message = match outcome {
        Ok(ok) => panic!(
            "issue #319 negative parity: a rank-1 matmul operand must be rejected, \
             not silently lowered; got Ok({ok:?})"
        ),
        Err(message) => message,
    };
    // Fail with the RIGHT reason: the diagnostic must name the rank-≥2
    // matmul violation (the guard the #319 fix must not paper over), not
    // some unrelated downstream symptom.
    assert!(
        message.contains("rank") && message.to_lowercase().contains("matmul"),
        "issue #319 negative parity: rejection must name the matmul rank-≥2 \
         violation; got {message}",
    );
}

// =====================================================================
// Completeness: additional precision-polymorphic separate-`sig` body
// shapes the single-matmul-pair `sdpa` test does not reach. Each is a
// separate-`sig` verb grad'd via the concrete-f32 driver, asserted to
// produce the SAME gradient as the byte-identical inline-f32 body. This
// exercises the masked-add / extra-transpose precision paths the
// School.Nn.Attention.* verbs use.
// =====================================================================

/// Grad through a precision-poly separate-`sig` verb whose body is
/// `body`, asserted equal to the inline-f32 reimplementation of the same
/// body. `sig`/`params`/inputs are supplied so each shape can vary its
/// arity. The inline form writes concrete-`f32` parameter annotations;
/// the separate-`sig` form declares precision-var `[..., p]` types in a
/// standalone `sig`.
fn assert_separate_sig_grad_matches_inline(
    label: &str,
    inline_params: &str,
    sig: &str,
    bare_params: &str,
    body: &str,
    loss_params: &str,
    call: &str,
) {
    let inline_src = format!(
        "def verb({inline_params}) = {body}\n\
         def loss({loss_params}) -> f32 =\n  \
           tensor_to_scalar(sum(sum(verb({call_args}), cast(0, int32)), cast(0, int32)))\n\
         out = grad(loss, wrt=q)({call})\n",
        call_args = bare_params,
    );
    let sep_src = format!(
        "{sig}\ndef verb({bare_params}) = {body}\n\
         def loss({loss_params}) -> f32 =\n  \
           tensor_to_scalar(sum(sum(verb({call_args}), cast(0, int32)), cast(0, int32)))\n\
         out = grad(loss, wrt=q)({call})\n",
        call_args = bare_params,
    );
    let inline = out_tensor(&try_eval(&inline_src).unwrap_or_else(|err| {
        panic!(
            "issue #319 [{label}] inline-f32 control must grad in host eval: {err}\n{inline_src}"
        )
    }));
    let sep = out_tensor(&try_eval(&sep_src).unwrap_or_else(|err| {
        panic!(
            "issue #319 [{label}] separate-sig precision-poly verb must grad in host eval \
             (no rank error, no monomorphization tripwire): {err}\n{sep_src}"
        )
    }));
    assert_close(&sep, &inline, 1e-6, &format!("issue #319 [{label}] parity"));
}

#[test]
fn issue_319_masked_causal_body_grad_matches_inline() {
    // softmax(add(mul(scores, scale), mask), -1): the masked/causal
    // attention path — an extra `add` of an `[s, s]` mask before softmax.
    assert_separate_sig_grad_matches_inline(
        "masked",
        "q: tensor[s, d, f32], k: tensor[s, d, f32], v: tensor[s, d, f32], scale: tensor[s, s, f32], mask: tensor[s, s, f32]",
        "sig verb[p: Float]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, s, p] -> tensor[s, d, p]",
        "q, k, v, scale, mask",
        "{\n  kt = permute(k, 1, 0)\n  scores = matmul(q, kt)\n  weights = softmax(add(mul(scores, scale), mask), -1)\n  matmul(weights, v)\n}",
        "q: tensor[2, 3, f32], k: tensor[2, 3, f32], v: tensor[2, 3, f32], scale: tensor[2, 2, f32], mask: tensor[2, 2, f32]",
        "to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), to_tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), to_tensor([[1.0, 0.0], [0.0, 1.0]]), to_tensor([[0.0, 0.0], [0.0, 0.0]])",
    );
}

#[test]
fn issue_319_output_transpose_body_grad_matches_inline() {
    // Multi-head-style output transpose: an extra `permute` on the
    // attention output exercises a second transpose-adjoint precision path
    // in the backward.
    assert_separate_sig_grad_matches_inline(
        "out-transpose",
        "q: tensor[s, d, f32], k: tensor[s, d, f32], v: tensor[s, d, f32], scale: tensor[s, s, f32]",
        "sig verb[p: Float]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[d, s, p]",
        "q, k, v, scale",
        "{\n  kt = permute(k, 1, 0)\n  scores = matmul(q, kt)\n  weights = softmax(mul(scores, scale), -1)\n  o = matmul(weights, v)\n  permute(o, 1, 0)\n}",
        "q: tensor[2, 3, f32], k: tensor[2, 3, f32], v: tensor[2, 3, f32], scale: tensor[2, 2, f32]",
        "to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), to_tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), to_tensor([[1.0, 0.0], [0.0, 1.0]])",
    );
}

// =====================================================================
// Completeness (non-grad precision path): the `reshape` and `expand`
// precision paths the sdpa grad bodies never touch. These verbs are
// precision-polymorphic separate-`sig` and must lower + evaluate with
// the precision var concretized from the call site — they would trip the
// §5.8.1 monomorphization tripwire if the `reshape`/`expand` lowering
// mis-handled the renamed precision var. (Grad through `reshape`/`expand`
// has its own, #319-unrelated AD limitations — dim pinning and an
// expand-adjoint dimension-count check — so these pin the precision-
// lowering path via a forward eval rather than a gradient.)
// =====================================================================

#[test]
fn issue_319_reshape_precision_poly_verb_lowers() {
    let src = "sig flat2d: tensor[s, d, p] -> tensor[six, p]\n\
               def flat2d(x) = reshape(permute(x, 1, 0), [cast(6, int64)])\n\
               out = flat2d(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let result = try_eval(src).unwrap_or_else(|err| {
        panic!(
            "issue #319: precision-poly separate-sig `reshape` verb must lower \
             (no monomorphization tripwire): {err}"
        )
    });
    let out = out_tensor(&result);
    assert_eq!(out.shape, vec![6], "issue #319 reshape: flattened shape");
    // permute([[1, 2, 3],[4, 5, 6]]) = [[1i64, 4i64],[2i64, 5i64],[3i64, 6i64]], flattened row-major.
    assert_close(
        &out,
        &TensorValue {
            shape: vec![6],
            data: wire_values::storage_f64(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        },
        1e-6,
        "issue #319 reshape values",
    );
}

#[test]
fn issue_319_expand_precision_poly_verb_lowers() {
    let src = "sig broadcast: tensor[s, p] -> tensor[s, c, p]\n\
               def broadcast(b) = insert(b, cast(1, int32), cast(2, int64))\n\
               out = broadcast(to_tensor([1.0, 2.0]))\n";
    let result = try_eval(src).unwrap_or_else(|err| {
        panic!(
            "issue #319: precision-poly separate-sig `expand` verb must lower \
             (no monomorphization tripwire): {err}"
        )
    });
    let out = out_tensor(&result);
    assert_eq!(out.shape, vec![2, 2], "issue #319 expand: broadcast shape");
    assert_close(
        &out,
        &TensorValue {
            shape: vec![2, 2],
            data: wire_values::storage_f64(vec![1.0, 1.0, 2.0, 2.0]),
        },
        1e-6,
        "issue #319 expand values",
    );
}

// =====================================================================
// Soundness (red-team caveat 1): the precision-var binding must NOT
// promote across precisions. It binds body precision vars to the
// call-site precision ONLY when the call is fully precision-monomorphic.
// =====================================================================

/// chelis#1486 / [04-INF-6] INVERSION of chelis#319's two-precision-var row.
///
/// The original asserted that `sig f: tensor[s,d,p] -> tensor[s,d,w] ->
/// tensor[s,d,w]`, whose body's `add` unifies `p` and `w`, GRADS when both
/// actuals are f32. That was a statement about a program the checker accepted.
/// Under [04-INF-6] it is no longer one: the header promises the body works for
/// any `p` and any `w` INDEPENDENTLY, and the body does not type-check at
/// `p = f32, w = f64`. The declaration is rejected before any call site is
/// reached, so "both pinned to f32" never arises.
///
/// Regression test, red before the rigidity check. Both Surf and serialized
/// Deep must reject this exact chelis#319 program before gradient evaluation.
#[test]
fn issue_319_two_precision_vars_unified_by_the_body_are_rejected() {
    let twovar = "sig f[p: Numeric, w: Numeric]: tensor[s, d, p] -> tensor[s, d, w] -> tensor[s, d, w]\n\
                  def f(q, b) = {\n  qt = permute(q, 1, 0)\n  qb = permute(qt, 1, 0)\n  add(qb, b)\n}\n\
                  def loss(q: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> f32 =\n  \
                    tensor_to_scalar(sum(sum(f(q, b), cast(0, int32)), cast(0, int32)))\n\
                  out = grad(loss, wrt=q)(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))\n";
    let deep =
        chelis_compiler_api::compiler::desugar(chelis_compiler_api::schema::DesugarRequest {
            source: twovar.to_owned(),
        })
        .expect("desugar")
        .deep_text;
    let results = [
        (SourceKind::Surf, twovar.to_owned()),
        (SourceKind::Deep, deep),
    ]
    .into_iter()
    .map(|(source_kind, source)| {
        eval(EvalRequest {
            source_kind,
            source,
            bindings: BTreeMap::new(),
        })
    })
    .collect::<Vec<_>>();
    for result in results {
        let error = result.expect_err("[04-INF-6]: two authored binders must stay independent");
        assert_eq!(error.stage, "check", "{error:?}");
        assert!(
            error.errors.iter().any(|diagnostic| {
                diagnostic.kind() == chelis_vocab::DiagnosticKind::TypeMismatch
                    && diagnostic
                        .message
                        .contains("distinct declared type parameters")
                    && diagnostic.message.contains("`p`")
                    && diagnostic.message.contains("`w`")
                    && diagnostic.message.contains("04-INF-6")
            }),
            "expected the collapse diagnostic naming both binders: {error:?}"
        );
    }
}

/// The honest-header twin, so chelis#319's guarantee survives on the shape the
/// language now admits.
///
/// #319 protected the property that a precision-polymorphic verb still grads at
/// a fully monomorphic call site. The two-binder spelling above cannot carry
/// that any more, so this row carries it with ONE binder, which is what the
/// body's `add` actually requires. Disposition lock: green before and after,
/// because the property it states was never the thing [04-INF-6] changed.
#[test]
fn issue_319_one_precision_var_still_grads_at_a_monomorphic_call_site() {
    let onevar = "sig f[p: Numeric]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p]\n\
                  def f(q, b) = {\n  qt = permute(q, 1, 0)\n  qb = permute(qt, 1, 0)\n  add(qb, b)\n}\n\
                  def loss(q: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> f32 =\n  \
                    tensor_to_scalar(sum(sum(f(q, b), cast(0, int32)), cast(0, int32)))\n\
                  out = grad(loss, wrt=q)(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))\n";
    let result = try_eval(onevar).unwrap_or_else(|err| {
        panic!("issue #319: a one-precision-binder verb must still grad at a monomorphic call site: {err}")
    });
    let grad = out_tensor(&result);
    // loss = sum(add(permute_roundtrip(q), b)); d/dq = ones.
    assert_close(
        &grad,
        &TensorValue {
            shape: vec![2, 3],
            data: wire_values::storage_f64(vec![1.0; 6]),
        },
        1e-6,
        "issue #319 monomorphic grad = ones",
    );
}

#[test]
fn issue_319_heterogeneous_precision_call_is_rejected_not_promoted() {
    // A verb sharing one precision var `p` across params, called with
    // genuinely DIFFERENT actual precisions (f32 + f64). The
    // no-implicit-precision-promotion invariant requires this be REJECTED
    // (the sig forces both to `p`), never silently forced to one
    // precision. The rejection must name the precision mismatch.
    let hetero = "sig f[p: Numeric]: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p]\n\
                  def f(a, b) = {\n  at = permute(a, 1, 0)\n  ar = permute(at, 1, 0)\n  add(ar, b)\n}\n\
                  def use_f(a: tensor[2, 3, f32], b: tensor[2, 3, f64]) -> f32 =\n  \
                    tensor_to_scalar(sum(sum(f(a, b), cast(0, int32)), cast(0, int32)))\n\
                  out = use_f(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), cast(to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), f64))\n";
    let outcome = try_eval(hetero);
    let message = match outcome {
        Ok(ok) => panic!(
            "issue #319 soundness: a heterogeneous-precision call (f32 vs f64) sharing one \
             sig precision var must be rejected, not silently promoted; got Ok({ok:?})"
        ),
        Err(message) => message,
    };
    assert!(
        message.to_lowercase().contains("precision"),
        "issue #319 soundness: rejection must name the precision mismatch; got {message}",
    );
}

#[test]
fn issue_319_distinct_precisions_not_force_merged() {
    // A genuine two-precision verb (`p` f32, `w` f64) whose body keeps the
    // two precisions distinct (no cross-precision op). The call is NOT
    // precision-monomorphic, so the fix binds NOTHING and the f64 chain is
    // preserved (or the tripwire fires) — but f64 is never silently
    // promoted to f32. Here the f64 input threads through a permute chain
    // and is dropped; the f32 result must be exact.
    let distinct = "sig f: tensor[s, d, p] -> tensor[s, d, w] -> tensor[s, d, p]\n\
                    def f(a, b) = {\n  bt = permute(b, 1, 0)\n  bp = permute(bt, 1, 0)\n  a\n}\n\
                    def use_f(a: tensor[2, 3, f32], b: tensor[2, 3, f64]) -> f32 =\n  \
                      tensor_to_scalar(sum(sum(f(a, b), cast(0, int32)), cast(0, int32)))\n\
                    out = use_f(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), cast(to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), f64))\n";
    // The verb type-checks (no cross-precision op). It must NOT silently
    // promote f64→f32; either it evaluates correctly (f64 preserved in the
    // dropped chain) or it surfaces a clean diagnostic — never a wrong-
    // precision Ok. Here the f32 result `sum(a)` = 21.0 is exact.
    // A clean PRECISION diagnostic is acceptable; only a silent
    // wrong-precision Ok is a soundness failure. On the Ok path assert
    // exactness; on the Err path assert the message is a precision
    // diagnostic (issue #319 review) so an UNRELATED failure cannot make
    // this test pass vacuously.
    match try_eval(distinct) {
        Ok(result) => {
            let root = result
                .roots
                .iter()
                .find(|r| r.name.as_deref() == Some("out"))
                .expect("out root");
            match &root.value {
                ExecutionValue::Scalar { value } => {
                    let chelis_types::ElementRef::F32(value) = value.get().element_ref() else {
                        panic!("expected exact f32 scalar, got {value:?}");
                    };
                    assert!(
                        (value - 21.0).abs() < f32::EPSILON,
                        "issue #319: expected 21.0, got {value}"
                    );
                }
                ExecutionValue::Tensor { value } => {
                    let s: f64 = value.data.to_f64_lossy_vec().iter().sum();
                    assert!(
                        (s - 21.0).abs() < 1e-9,
                        "issue #319 distinct-precision: result must be 21.0, got {s}",
                    );
                }
                other => panic!("unexpected out value {other:?}"),
            }
        }
        Err(message) => assert!(
            message.to_lowercase().contains("precision"),
            "issue #319 distinct-precision: an Err outcome must be a clean precision \
             diagnostic, not an unrelated failure; got {message}",
        ),
    }
}
