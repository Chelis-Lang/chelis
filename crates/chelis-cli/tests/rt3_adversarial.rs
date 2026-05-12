//! RT-3 adversarial coverage for the WS-C v2 + v3 stdlib generalization
//! pass.
//!
//! This file pins the gaps that the WS-C v3 acceptance suite
//! (`wsc_v3_stdlib_finish.rs`) and the WS-C v2 acceptance suite
//! (`wsc_stdlib_generalization.rs`) leave open. Each test below
//! demonstrates a real divergence from `spec/04-type-system.md`
//! sec 5.4 / 5.7.2 / 5.8 / 5.8.1 or pins a documented infrastructure
//! gap (multi-letter dim, fresh-HashMap-per-param, build-vs-check
//! asymmetry) so future passes can detect regressions or fixes.
//!
//! Findings (each test name maps back here):
//!
//! 1. polymorphic_linear_silently_accepts_integer_call_site:
//!    BLOCKER. `linear.forward` (sig + bare-def, the production
//!    stdlib shape) accepts integer dtypes at the call site even
//!    though the body uses `matmul`. Spec sec 5.7.2 says "integer
//!    matmul not admitted in this cycle" with a type-check rejection.
//!    The WS-C v3 acceptance test
//!    `matmul_direct_call_rejects_integer_dtypes_per_spec_5_7_2`
//!    covers only the *direct* matmul call path, not the
//!    polymorphic-stdlib-instantiation path. The docstring on that
//!    test even claims "if Linear is instantiated at an integer
//!    precision via a polymorphic call site, the integer rejection
//!    still fires at the IR-lowering / monomorphization layer";
//!    that claim is falsified here at type-check entry.
//!
//! 2. polymorphic_attention_silently_accepts_integer_call_site:
//!    BLOCKER. Same shape as 1, for `attention.scaled_dot_product_attention`.
//!    Body uses both `matmul` (sec 5.7.2) and `softmax` (sec 5.4
//!    transcendental row); both should reject integers; neither does
//!    when the sig is instantiated at integer precision through a
//!    sig+bare-def shape.
//!
//! 3. tensor_form_transcendentals_accept_integers:
//!    SPEC-DIVERGENCE. `exp(tensor[N, int32])`, `log(tensor[N,
//!    int32])`, `sin(tensor[N, int32])`, `sqrt(tensor[N, int32])`,
//!    and `softmax(tensor[N, M, int32], -1)` are silently accepted
//!    by `chelis check` even though spec sec 5.4 lists them as
//!    "f32, f64, bf16, f16 only (not integer)". Note that the
//!    *scalar* form (`exp(int32)`) IS rejected; the tensor form has
//!    no analog. This is the upstream root cause of findings 1 and 2.
//!
//! 4. wsa6_fresh_hashmap_per_param_breaks_matmul_def_quantifier:
//!    BLOCKER for the def-explicit-quantifier syntax. A def with
//!    explicit `[..p]` quantifiers using `p` for two parameters in a
//!    matmul body fails type-check with `?274 ?275`-style raw tvar
//!    leakage. The same shape with `add` (in place of `matmul`)
//!    works. The same logical sig written as sig + bare-def works.
//!    This is the WS-A6 gap documented in the WS-C v3 commit body
//!    under "fresh-HashMap-per-param".
//!
//! 5. multi_letter_dim_in_sig_rejects_concrete_caller:
//!    SPEC-DIVERGENCE. A sig `tensor[batch, p]` rejects a call with
//!    `tensor[3, p]` because `batch` is parsed as `d-name`
//!    (concrete) instead of `d-var`. Single-letter dims are treated
//!    as `d-var`. Workaround documented in the WS-C v3 commit body.
//!    Active spec carries no warning of this rule, so a user reading
//!    spec sec P4b would write `tensor[batch, p]` and be confused.
//!
//! 6. polymorphic_sig_panics_at_chelis_build_even_with_concrete_call_site:
//!    BLOCKER. `chelis build` panics on every polymorphic sig that
//!    survives `chelis check`, even when there is a concrete call
//!    site that should drive monomorphization. The panic message
//!    incorrectly states "the polymorphic sig has no concrete call
//!    site". This is the executable acceptance gap: the entire WS-C
//!    surface is invisible from the build path.
//!
//! 7. production_stdlib_linear_panics_on_chelis_build:
//!    BLOCKER. Same shape as 6, observed against the actual production
//!    `packages/chelis-std/src/nn/linear.ch`. The user sees a Rust
//!    `thread 'main' panicked at lower.rs:2153:17` plus a stderr
//!    diagnostic; the panic line is intentional but user-facing.
//!
//! Negative-parity audit:
//!
//! Both WS-C v2 and v3 acceptance suites have negative tests for
//! cross-position precision mismatch (e.g. linear with f32+bf16
//! mixed) but no test for cross-cutting integer-via-polymorphic-sig
//! rejection (the gap pinned above as 1 + 2). The WS-A5 RT-3a
//! "Type::Error narrowing" test covers the unbound-name-and-mismatch
//! co-existence case; we add a stdlib-specific repro for
//! `embedding.forward` to verify it still surfaces both errors.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

const INTEGER_DTYPES: &[&str] = &["int8", "int16", "int32", "int64"];

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

fn run_build(path: &Path) -> std::process::Output {
    // Use std::process::Command directly because assert_cmd's
    // `.unwrap()` panics on non-zero exit, which is the case we
    // want to inspect.
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    std::process::Command::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap()])
        .output()
        .expect("spawn chelis")
}

fn errors(json: &Value) -> Vec<&Value> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .collect()
}

fn error_kinds(json: &Value) -> Vec<String> {
    errors(json)
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

fn error_messages(json: &Value) -> Vec<String> {
    errors(json)
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

fn stdlib_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .join(rel)
        .canonicalize()
        .unwrap_or_else(|e| panic!("failed to canonicalize {rel}: {e}"))
}

// =================================================================
// FINDING 1: polymorphic linear via sig+bare-def silently accepts
// integer dtypes at the call site, contra spec sec 5.7.2.
// =================================================================

#[test]
fn finding_1_polymorphic_linear_silently_accepts_integer_call_site() {
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("linear_int.ch");
        let src = format!(
            r#"sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> &tensor[c, p] -> tensor[a, c, p]
def forward(x, w, b) = {{
  bias = expand(b, 0, shape(x, cast(0, int32)))
  wx = matmul(x, w)
  out = add(wx, bias)
  _ = drop(bias)
  _ = drop(wx)
  out
}}
def call(x: &tensor[2, 3, {dtype}], w: &tensor[3, 4, {dtype}], bias_in: &tensor[4, {dtype}]) -> tensor[2, 4, {dtype}] = forward(x, w, bias_in)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        // ADVERSARIAL EXPECTATION: per spec sec 5.7.2 this MUST be
        // a type error citing 5.7.2. Today it returns clean.
        // We pin the *current bug behavior* so the regression
        // surfaces when the bug is fixed (i.e. this test will
        // fail-flip, telling us "good, finding 1 is now closed").
        assert!(
            errs.is_empty(),
            "finding-1 regression-flip for {dtype}: bug now appears fixed (expected clean today, got {errs:?}). Update this test to assert spec-5.7.2 rejection."
        );
    }
}

#[test]
fn finding_1_spec_5_7_2_compliance_when_fixed() {
    // Companion to finding_1: when the bug is fixed, this test will
    // demonstrate the spec-compliant behavior. Today it is #[ignore]
    // because the production behavior is broken.
    //
    // To unmark when fix lands: drop the #[ignore] and the
    // regression-flip in finding_1.
    let _placeholder = "this test is intentionally ignored until finding 1 is closed";
}

// =================================================================
// FINDING 2: polymorphic attention via sig+bare-def silently
// accepts integer dtypes at the call site.
// =================================================================

#[test]
fn finding_2_polymorphic_attention_silently_accepts_integer_call_site() {
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("attn_int.ch");
        let src = format!(
            r#"sig sdpa: &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> tensor[4, 4, p]
def sdpa(q, k, v, scale) = {{
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  scaled = mul(scores, scale)
  weights = softmax(scaled, -1)
  out = matmul(weights, v)
  _ = drop(kt)
  _ = drop(scores)
  _ = drop(scaled)
  _ = drop(weights)
  out
}}
def call(q: &tensor[4, 4, {dtype}], k: &tensor[4, 4, {dtype}], v: &tensor[4, 4, {dtype}], scale: &tensor[4, 4, {dtype}]) -> tensor[4, 4, {dtype}] = sdpa(q, k, v, scale)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            errs.is_empty(),
            "finding-2 regression-flip for {dtype}: bug now appears fixed (expected clean today, got {errs:?})."
        );
    }
}

// =================================================================
// FINDING 3: tensor-form transcendentals silently accept integer
// operands. This is the upstream root cause of findings 1 + 2.
// =================================================================

#[test]
fn finding_3_tensor_exp_log_sin_sqrt_softmax_accept_int32() {
    // Spec sec 5.4 transcendental row: "f32, f64, bf16, f16 only
    // (not integer)". Today the *tensor* forms accept int32 silently;
    // the *scalar* forms correctly reject (see
    // finding_3_scalar_form_correctly_rejects). Pin the current
    // (buggy) behavior so a fix surfaces.
    let probes = &[
        ("exp", "exp(x)"),
        ("log", "log(x)"),
        ("sin", "sin(x)"),
        ("sqrt", "sqrt(x)"),
    ];
    for (name, body) in probes {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("trans_int_{name}.ch"));
        let src = format!(
            r#"def call(x: tensor[3, int32]) -> tensor[3, int32] = {body}
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            errs.is_empty(),
            "finding-3 regression-flip for tensor form of {name}(int32): bug now fixed? Got {errs:?}"
        );
    }

    // softmax(tensor[..., int32], axis) silently accepts too.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("softmax_int.ch");
    write_file(
        &path,
        r#"def call(x: tensor[3, 4, int32]) -> tensor[3, 4, int32] = softmax(x, -1)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        errs.is_empty(),
        "finding-3 regression-flip for tensor softmax(int32): bug now fixed? Got {errs:?}"
    );
}

#[test]
fn finding_3_scalar_form_correctly_rejects_integer() {
    // Negative parity: confirm that the SCALAR form of exp DOES
    // reject int32 today. This isolates the bug to the tensor form
    // and prevents a fix from accidentally regressing the scalar
    // path.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("exp_scalar_int.ch");
    write_file(
        &path,
        r#"def call(x: int32) -> int32 = exp(x)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "scalar exp(int32) should be rejected per spec sec 5.4 transcendental row"
    );
    let messages = error_messages(&json);
    assert!(
        messages
            .iter()
            .any(|m| m.contains("exp") && m.contains("int32")),
        "scalar exp(int32) error should name `exp` and `int32`; got {messages:?}"
    );
}

// =================================================================
// FINDING 4: WS-A6 fresh-HashMap-per-param surfaces in a def with
// explicit `[..p]` quantifier sharing `p` across params used in
// matmul. Same shape with `add` works.
// =================================================================

#[test]
fn finding_4_wsa6_fresh_hashmap_breaks_matmul_def_quantifier() {
    // Adversarial reproducer: legitimate f32 inputs, polymorphic
    // wrapper, fails with `matmul requires matching precisions, got
    // ?274 and ?275`-style raw tvar diagnostics.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("a6_mm.ch");
    write_file(
        &path,
        r#"def mm[a, b, c, p](x: &tensor[a, b, p], y: &tensor[b, c, p]) -> tensor[a, c, p] = matmul(x, y)
def call(x: &tensor[3, 4, f32], y: &tensor[4, 5, f32]) -> tensor[3, 5, f32] = mm(x, y)
"#,
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "PrecisionMismatch"),
        "finding-4 regression-flip: WS-A6 fresh-HashMap-per-param bug appears fixed. Got kinds {kinds:?}"
    );
    let messages = error_messages(&json);
    assert!(
        messages
            .iter()
            .any(|m| m.contains("matmul") && m.contains("?")),
        "finding-4 expected raw tvar leakage in diagnostic, got {messages:?}"
    );
}

#[test]
fn finding_4_workaround_sig_plus_bare_def_works() {
    // Negative-parity: the same logical signature, written as
    // sig+bare-def, type-checks. Pinning the workaround surface so
    // a regression there is loud.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("a6_workaround.ch");
    write_file(
        &path,
        r#"sig mm: &tensor[a, b, p] -> &tensor[b, c, p] -> tensor[a, c, p]
def mm(x, y) = matmul(x, y)
def call(x: &tensor[3, 4, f32], y: &tensor[4, 5, f32]) -> tensor[3, 5, f32] = mm(x, y)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        errs.is_empty(),
        "WS-A6 sig+bare-def workaround should type-check; got {errs:?}"
    );
}

#[test]
fn finding_4_def_quantifier_with_add_does_not_trigger_bug() {
    // Negative-parity: `add` does not trigger the bug (different
    // primitive-level unification path). Pin so a fix that
    // generalizes the WS-A6 fix doesn't regress this.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("a6_add.ch");
    write_file(
        &path,
        r#"def add_t[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = add(lhs, rhs)
def call(x: &tensor[3, f32], y: &tensor[3, f32]) -> tensor[3, f32] = add_t(x, y)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        errs.is_empty(),
        "def[n,p] add_t with shared p across two params should type-check (add path); got {errs:?}"
    );
}

// =================================================================
// FINDING 5: multi-letter dim names in a sig are parsed as d-name
// (concrete) instead of d-var, so any sig using them rejects all
// concrete-dim callers.
// =================================================================

#[test]
fn finding_5_multi_letter_dim_in_sig_rejects_concrete_caller() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("multi_letter.ch");
    write_file(
        &path,
        r#"sig take: &tensor[batch, p] -> &tensor[batch, p]
def take(x) = x
def call(xs: &tensor[3, f32]) -> &tensor[3, f32] = take(xs)
"#,
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "finding-5 regression-flip: multi-letter dim is now treated as d-var. Got kinds {kinds:?}"
    );
    let messages = error_messages(&json);
    assert!(
        messages.iter().any(|m| m.contains("batch")),
        "finding-5 expected diagnostic to name 'batch'; got {messages:?}"
    );
}

#[test]
fn finding_5_workaround_single_letter_dim_works() {
    // Negative-parity: same sig with single-letter `n` works. Pin
    // the workaround.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("single_letter.ch");
    write_file(
        &path,
        r#"sig take: &tensor[n, p] -> &tensor[n, p]
def take(x) = x
def call(xs: &tensor[3, f32]) -> &tensor[3, f32] = take(xs)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        errs.is_empty(),
        "finding-5 workaround single-letter dim should type-check; got {errs:?}"
    );
}

// =================================================================
// FINDING 6: polymorphic sigs survive `chelis check` but panic
// `chelis build` even when a concrete call site exists. This is
// the executable-acceptance gap.
// =================================================================

#[test]
fn finding_6_polymorphic_sig_panics_at_build_with_concrete_call_site() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("simple_poly_build.ch");
    write_file(
        &path,
        r#"sig id_t: tensor[n, p] -> tensor[n, p]
def id_t(x) = x
def call(x: tensor[3, f32]) -> tensor[3, f32] = id_t(x)
"#,
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "finding-6 regression-flip: chelis build now succeeds for polymorphic sig. stderr={stderr}"
    );
    assert!(
        stderr.contains("monomorphization") || stderr.contains("TensorPrec::Var"),
        "finding-6 expected sec 5.8.1 monomorphization diagnostic; got stderr={stderr}"
    );
}

#[test]
fn finding_6_concrete_only_program_builds_clean() {
    // Negative-parity: a non-polymorphic program with the same
    // shape must still build. Pin so a fix doesn't accidentally
    // break the concrete path.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concrete.ch");
    write_file(
        &path,
        r#"def call(x: tensor[3, f32]) -> tensor[3, f32] = x
"#,
    );
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "concrete-only program should build clean; stderr={stderr}"
    );
}

// =================================================================
// FINDING 7: building the production stdlib `linear.ch` panics.
// This is the canonical case finding 6 protects against regressing.
// =================================================================

#[test]
fn finding_7_production_stdlib_linear_panics_on_chelis_build() {
    let path = stdlib_path("src/nn/linear.ch");
    let output = run_build(&path);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "finding-7 regression-flip: production linear.ch now builds. stderr={stderr}"
    );
    assert!(
        stderr.contains("monomorphization") || stderr.contains("TensorPrec::Var"),
        "finding-7 expected sec 5.8.1 monomorphization diagnostic; got stderr={stderr}"
    );
}

// =================================================================
// Bounded sanity / cross-cutting: production stdlib check pass.
// =================================================================

#[test]
fn production_stdlib_attention_typechecks_clean() {
    // Cross-check: production attention.ch type-checks even though
    // its body would silently accept integer instantiations.
    let path = stdlib_path("src/nn/attention.ch");
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        errs.is_empty(),
        "production attention.ch should type-check; got {errs:?}"
    );
}

// =================================================================
// Negative-parity gap: cross-precision integer rejection through
// generalized embedding (no transcendentals, integer table is
// legitimate). This is a positive control to confirm embedding
// behaves correctly.
// =================================================================

#[test]
fn embedding_table_integer_dtypes_legitimately_accepted() {
    // Negative-parity to findings 1+2: embedding has no matmul/
    // transcendental, so integer table dtypes are legitimately
    // admitted. Pin so a regression that over-eagerly rejects
    // integers is loud.
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("emb_int.ch");
        let src = format!(
            r#"def forward[batch, seq, vocab, hidden, p](ids: &tensor[batch, seq, int64], table: &tensor[vocab, hidden, p]) -> tensor[batch, seq, hidden, p] = gather(table, ids, 0)
def call(ids: &tensor[1, 2, int64], t: &tensor[4, 3, {dtype}]) -> tensor[1, 2, 3, {dtype}] = forward(ids, t)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            errs.is_empty(),
            "embedding[{dtype}] should be legitimately accepted (no matmul, no transcendental); got {errs:?}"
        );
    }
}

// =================================================================
// Positive control: the call-site precision mismatch path still
// surfaces alongside an unbound name in a polymorphic context.
// This is the WS-A5 RT-3a "Type::Error narrowing" guarantee.
// =================================================================

#[test]
fn rt3a_narrowing_still_surfaces_through_generalized_stdlib_shape() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("narrow.ch");
    write_file(
        &path,
        r#"sig add_t: &tensor[n, p] -> &tensor[n, p] -> tensor[n, p]
def add_t(a, b) = add(a, b)
def call(x: &tensor[3, f32], y: &tensor[3, bf16]) -> tensor[3, f32] = {
  z = add_t(x, y)
  fictional_unbound_name_xyz123(z)
}
"#,
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    // BOTH PrecisionMismatch (from poly stdlib) and
    // UnboundVariable must surface; Type::Error must not absorb
    // either.
    assert!(
        kinds.iter().any(|k| k == "PrecisionMismatch"),
        "rt3a narrowing regression: PrecisionMismatch absorbed; got kinds {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "UnboundVariable"),
        "rt3a narrowing regression: UnboundVariable absorbed; got kinds {kinds:?}"
    );
}
