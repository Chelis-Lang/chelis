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

/// Concrete-`f32` test inputs and the scalar-loss + `grad(loss, wrt=(q))`
/// driver, parameterized over the callee name (so the imported and inline
/// forms share one driver and must produce identical gradients).
fn grad_driver(callee: &str) -> String {
    format!(
        "def loss(q: tensor[2, 3, f32], k: tensor[2, 3, f32], v: tensor[2, 3, f32], scale: tensor[2, 2, f32]) -> f32 =\n\
         \x20 tensor_to_scalar(sum(sum({callee}(q, k, v, scale), cast(0, int32)), cast(0, int32)))\n\
         out = grad(loss, wrt=(q))(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), to_tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), to_tensor([[1.0, 0.0], [0.0, 1.0]]))\n",
    )
}

fn assert_close(actual: &TensorValue, expected: &TensorValue, tol: f64, label: &str) {
    assert_eq!(actual.shape, expected.shape, "{label}: shape mismatch");
    assert_eq!(
        actual.data.len(),
        expected.data.len(),
        "{label}: length mismatch"
    );
    for (i, (a, e)) in actual.data.iter().zip(expected.data.iter()).enumerate() {
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
        grad.data.iter().all(|v| v.is_finite()),
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
        "sig sdpa: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]\n\
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
                   sig sdpa: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]\n\
                   def sdpa(q, k, v, scale) = {\n  \
                     kt = permute(k, 1, 0)\n  \
                     scores = matmul(q, kt)\n  \
                     weights = softmax(mul(scores, scale), -1)\n  \
                     matmul(weights, v)\n}\n";
    let main = "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n";
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
    assert!(
        outcome.is_err(),
        "issue #319 negative parity: a rank-1 matmul operand must be rejected, \
         not silently lowered; got Ok({outcome:?})",
    );
}
