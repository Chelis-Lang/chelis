//! RT-3 adversarial coverage for the WS-C v2 + v3 stdlib generalization
//! pass, post-WS-A8 monomorphization fix.
//!
//! Originally this file pinned the gaps that the WS-C v3 acceptance
//! suite (`stdlib_precision_generalization_followups.rs`) and the WS-C v2 acceptance suite
//! (`stdlib_precision_generalization.rs`) left open. The 5 BLOCKER findings
//! (1, 2, 3, 6, 7) have been closed by WS-A8: monomorphization is now
//! implemented, and §5.4 / §5.7.2 enforcement runs at every
//! polymorphic-call-site instantiation in addition to direct primitive
//! calls. The tests below now assert the POST-FIX behavior; each
//! BLOCKER test has been inverted to verify the spec-compliant
//! rejection / build success per the new contract.
//!
//! The 2 bounded-gap findings (4: WS-A6 fresh-UnordMap-per-param;
//! 5: multi-letter dim names treated as concrete) remain open and
//! are marked `#[ignore = "WS-A9 follow-up"]` so a future agent can
//! pick them up without the regression-flip noise.
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
//!    under "fresh-UnordMap-per-param".
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
use std::path::Path;
use tempfile::tempdir;

const INTEGER_DTYPES: &[&str] = &["int8", "int16", "int32", "int64"];

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. The adversarial fixtures here use
// both clean and error-expecting cases; the helper captures stdout
// regardless of exit status so both shapes work.
fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn run_build(path: &Path) -> std::process::Output {
    // Use std::process::Command directly because assert_cmd's
    // `.unwrap()` panics on non-zero exit, which is the case we
    // want to inspect.
    //
    // Output is pinned to the source file's parent dir (always a
    // tempdir for these tests) via `-o`. Without this, `chelis build`
    // emits its C/header bundle into the cargo test runner's CWD
    // (`crates/chelis-cli/`), littering the working tree with files
    // like `concrete.c` / `simple_poly_build.c` and forcing
    // .gitignore allowlists.
    let out_dir = path.parent().expect("source path has parent");
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    std::process::Command::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
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

// =================================================================
// FINDING 1: polymorphic linear via sig+bare-def silently accepts
// integer dtypes at the call site, contra spec sec 5.7.2.
// =================================================================

#[test]
fn polymorphic_linear_rejects_integer_call_site() {
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("linear_int.ch");
        let src = format!(
            r#"sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> &tensor[c, p] -> tensor[a, c, p]
def forward(x, w, b) = {{
  bias = insert(b, 0, shape(x, cast(0, int32)))
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
        // WS-A8: cross-row enforcement now fires at the polymorphic
        // call site. spec/04-type-system.md §5.7.2 forbids integer
        // matmul; the body's `matmul(x, w)` reaches an integer
        // operand precision when `forward` is instantiated at any
        // INTEGER_DTYPE, and the validator rejects.
        assert!(
            !errs.is_empty(),
            "WS-A8: polymorphic forward+matmul instantiated at \
             integer dtype `{dtype}` must be rejected per \
             spec/04-type-system.md \u{00a7}5.7.2; got clean. errs={errs:?}"
        );
        let kinds = error_kinds(&json);
        assert!(
            kinds.iter().any(|k| k == "PrecisionMismatch"),
            "WS-A8: rejection for `{dtype}` must surface a \
             PrecisionMismatch; got kinds {kinds:?}"
        );
        let messages = error_messages(&json);
        assert!(
            messages.iter().any(|m| m.contains("5.7.2")),
            "WS-A8: rejection for `{dtype}` must cite spec section \
             5.7.2; got messages {messages:?}"
        );
        assert!(
            messages.iter().any(|m| m.contains("matmul")),
            "WS-A8: rejection for `{dtype}` must name the matmul op; \
             got messages {messages:?}"
        );
    }
}

#[test]
fn polymorphic_linear_accepts_float_call_site() {
    // Negative-parity to the integer-rejection test above: a float
    // instantiation of the same polymorphic linear sig must
    // continue to type-check cleanly.
    for dtype in &["f32", "f64", "bf16", "f16"] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("linear_float.ch");
        let src = format!(
            r#"sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> &tensor[c, p] -> tensor[a, c, p]
def forward(x, w, b) = {{
  bias = insert(b, 0, shape(x, cast(0, int32)))
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
        assert!(
            errs.is_empty(),
            "WS-A8: float instantiation `{dtype}` of polymorphic \
             linear must type-check cleanly; got {errs:?}"
        );
    }
}

// =================================================================
// FINDING 2: polymorphic attention via sig+bare-def silently
// accepts integer dtypes at the call site.
// =================================================================

#[test]
fn polymorphic_attention_rejects_integer_call_site() {
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
        // WS-A8: the body uses both `matmul` (§5.7.2) and `softmax`
        // (§5.4 transcendental row); both must reject integer
        // instantiations at the call site.
        assert!(
            !errs.is_empty(),
            "WS-A8: polymorphic attention sdpa instantiated at integer \
             dtype `{dtype}` must be rejected per spec/04-type-system.md \
             \u{00a7}5.7.2 and \u{00a7}5.4; got clean. errs={errs:?}"
        );
        let messages = error_messages(&json);
        assert!(
            messages
                .iter()
                .any(|m| m.contains("5.7.2") || m.contains("5.4")),
            "WS-A8: rejection for `{dtype}` must cite \u{00a7}5.7.2 or \
             \u{00a7}5.4; got messages {messages:?}"
        );
    }
}

// =================================================================
// `div` is FLOAT-ONLY since chelis#178: applying it to integer
// operands is a type error citing spec/05-risc-primitives.md §2.1
// and pointing at `floor_div` / `trunc_div`. `recip` is likewise
// float-only per §2.2 (an integer reciprocal has no useful IEEE-754
// interpretation). These tests pin both sides: polymorphic `div`
// AND `recip` wrappers must reject integer instantiations; the new
// integer-division ops `floor_div` / `trunc_div` accept them.
// =================================================================

#[test]
fn polymorphic_div_wrapper_rejects_integer_call_site() {
    // chelis#178: integer `div` is no longer admitted. A polymorphic
    // `div` wrapper instantiated at an integer dtype must reject at
    // the call site with a §2.1 citation pointing at the integer
    // division ops.
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("div_wrapper_int.ch");
        let src = format!(
            r#"sig my_div: tensor[4, p] -> tensor[4, p] -> tensor[4, p]
def my_div(a, b) = div(a, b)
def call(x: tensor[4, {dtype}], y: tensor[4, {dtype}]) -> tensor[4, {dtype}] = my_div(x, y)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            !errs.is_empty(),
            "chelis#178: polymorphic `div` wrapper at integer dtype \
             `{dtype}` must reject (div is float-only per \
             spec/05-risc-primitives.md \u{00a7}2.1); got clean. errs={errs:?}"
        );
        let messages = error_messages(&json);
        assert!(
            messages.iter().any(|m| {
                m.contains("div") && (m.contains("floor_div") || m.contains("trunc_div"))
            }),
            "chelis#178: rejection of `div({dtype})` wrapper must name \
             `div` and point at `floor_div` / `trunc_div`; got {messages:?}"
        );
    }
}

#[test]
fn polymorphic_floor_div_wrapper_accepts_integer_call_site() {
    // chelis#178: `floor_div` is the integer-admissible replacement.
    // A polymorphic wrapper instantiated at an integer dtype must
    // type-check clean.
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("floor_div_wrapper_int.ch");
        let src = format!(
            r#"sig my_fd: tensor[4, p] -> tensor[4, p] -> tensor[4, p]
def my_fd(a, b) = floor_div(a, b)
def call(x: tensor[4, {dtype}], y: tensor[4, {dtype}]) -> tensor[4, {dtype}] = my_fd(x, y)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            errs.is_empty(),
            "chelis#178: polymorphic `floor_div` wrapper at integer dtype \
             `{dtype}` must type-check clean per \
             spec/05-risc-primitives.md \u{00a7}2.1; got {errs:?}"
        );
    }
}

#[test]
fn polymorphic_trunc_div_wrapper_accepts_integer_call_site() {
    // chelis#178: `trunc_div` is integer-only and admissible at every
    // integer dtype.
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("trunc_div_wrapper_int.ch");
        let src = format!(
            r#"sig my_td: tensor[4, p] -> tensor[4, p] -> tensor[4, p]
def my_td(a, b) = trunc_div(a, b)
def call(x: tensor[4, {dtype}], y: tensor[4, {dtype}]) -> tensor[4, {dtype}] = my_td(x, y)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            errs.is_empty(),
            "chelis#178: polymorphic `trunc_div` wrapper at integer dtype \
             `{dtype}` must type-check clean per \
             spec/05-risc-primitives.md \u{00a7}2.1; got {errs:?}"
        );
    }
}

#[test]
fn polymorphic_recip_wrapper_rejects_integer_call_site() {
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("recip_wrapper_int.ch");
        let src = format!(
            r#"sig my_recip: tensor[4, p] -> tensor[4, p]
def my_recip(x) = recip(x)
def call(y: tensor[4, {dtype}]) -> tensor[4, {dtype}] = my_recip(y)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            !errs.is_empty(),
            "PR #176 red-team: polymorphic `recip` wrapper at integer \
             dtype `{dtype}` must reject per spec/04-type-system.md \
             \u{00a7}5.4; got clean. errs={errs:?}"
        );
        let messages = error_messages(&json);
        assert!(
            messages
                .iter()
                .any(|m| m.contains("5.4") && m.contains("recip")),
            "PR #176 red-team: rejection of recip(`{dtype}`) wrapper \
             must cite \u{00a7}5.4 and name `recip`; got {messages:?}"
        );
    }
}

// =================================================================
// FINDING 3: tensor-form transcendentals silently accept integer
// operands. This is the upstream root cause of findings 1 + 2.
// =================================================================

#[test]
fn tensor_exp_log_sin_sqrt_softmax_now_reject_int32() {
    // WS-A8: spec/04-type-system.md \u{00a7}5.4 transcendental row
    // is now enforced for tensor forms (not just scalar forms). All
    // five ops must reject `tensor[..., int32]` operands.
    let probes = &[
        ("exp", "exp(x)"),
        ("log", "log(x)"),
        ("sin", "sin(x)"),
        ("sqrt", "sqrt(x)"),
        // `recip` is float-only per spec/05-risc-primitives.md §2.2
        // because integer reciprocal has no useful IEEE-754 meaning.
        // `div` is float-only too (chelis#178), but its integer
        // rejection cites §2.1 (not §5.4) and points at `floor_div` /
        // `trunc_div`, so it is locked by `tensor_div_rejects_int32`
        // below rather than asserted against the §5.4 citation here.
        ("recip", "recip(x)"),
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
            !errs.is_empty(),
            "WS-A8: tensor form of `{name}(int32)` must be rejected per \
             spec/04-type-system.md \u{00a7}5.4; got clean. errs={errs:?}"
        );
        let messages = error_messages(&json);
        assert!(
            messages.iter().any(|m| m.contains("5.4")),
            "WS-A8: rejection of `{name}(int32)` must cite spec section \
             5.4; got messages {messages:?}"
        );
    }

    // softmax(tensor[..., int32], axis) is also a transcendental row op
    // and must reject under the same rule.
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
        !errs.is_empty(),
        "WS-A8: tensor softmax(int32) must be rejected per \
         spec/04-type-system.md \u{00a7}5.4; got clean. errs={errs:?}"
    );
    let messages = error_messages(&json);
    assert!(
        messages.iter().any(|m| m.contains("5.4")),
        "WS-A8: rejection of softmax(int32) must cite spec section 5.4; \
         got messages {messages:?}"
    );
}

#[test]
fn tensor_div_rejects_int32() {
    // chelis#178: the direct tensor form `div(int32, int32)` is a type
    // error. The diagnostic cites spec/05-risc-primitives.md §2.1 and
    // names the migration ops `floor_div` / `trunc_div`.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("div_int.ch");
    write_file(
        &path,
        r#"def call(x: tensor[3, int32], y: tensor[3, int32]) -> tensor[3, int32] = div(x, y)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "chelis#178: tensor `div(int32, int32)` must be rejected; got clean. errs={errs:?}"
    );
    let messages = error_messages(&json);
    assert!(
        messages
            .iter()
            .any(|m| m.contains("2.1") && m.contains("floor_div") && m.contains("trunc_div")),
        "chelis#178: rejection of div(int32) must cite \u{00a7}2.1 and name \
         `floor_div` and `trunc_div`; got messages {messages:?}"
    );
}

#[test]
fn tensor_floor_div_trunc_div_accept_int32() {
    // chelis#178 positive parity: the integer division replacement ops
    // type-check clean on integer tensors.
    for (name, body) in &[
        ("floor_div", "floor_div(x, y)"),
        ("trunc_div", "trunc_div(x, y)"),
    ] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("{name}_int.ch"));
        let src = format!(
            r#"def call(x: tensor[3, int32], y: tensor[3, int32]) -> tensor[3, int32] = {body}
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        let errs = errors(&json);
        assert!(
            errs.is_empty(),
            "chelis#178: tensor `{name}(int32, int32)` must type-check clean; got {errs:?}"
        );
    }
}

#[test]
fn tensor_trunc_div_rejects_f32() {
    // chelis#178: `trunc_div` is integer-only. Applying it to a float
    // tensor is a type error.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("trunc_div_f32.ch");
    write_file(
        &path,
        r#"def call(x: tensor[3, f32], y: tensor[3, f32]) -> tensor[3, f32] = trunc_div(x, y)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "chelis#178: `trunc_div(f32, f32)` must be rejected (integer-only); got clean. errs={errs:?}"
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
// FINDING 4: WS-A6 fresh-UnordMap-per-param surfaces in a def with
// explicit `[..p]` quantifier sharing `p` across params used in
// matmul. Same shape with `add` works.
// =================================================================

#[test]
#[ignore = "WS-A9 follow-up: WS-A6 fresh-UnordMap-per-param bug; out of WS-A8 scope"]
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
        "finding-4 regression-flip: WS-A6 fresh-UnordMap-per-param bug appears fixed. Got kinds {kinds:?}"
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
// FINDING 5: regression-lock — multi-letter dim sig accepts concrete
// caller (issue Chelis-Lang/chelis#219, Option A).
// =================================================================
//
// Before #219 Option A, multi-letter dim names in a sig parsed as
// `d-name` (concrete) and `unify_dim` had an asymmetry where
// `Var <-> Lit` was accepted but `Name <-> Lit` was rejected. The
// test below was originally written as a diagnostic pin
// (`finding_5_multi_letter_dim_in_sig_rejects_concrete_caller`,
// `#[ignore]`-flagged) to flip when the fix landed. With Option A,
// the call site now type-checks cleanly; this test pins the
// post-fix contract so a regression in the unify_dim arm is loud.

#[test]
fn finding_5_multi_letter_dim_in_sig_accepts_concrete_caller() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("multi_letter.ch");
    // Fixture renamed from `take` for chelis#353: `take` is a builtin
    // name and bare shadowing defs are now rejected at declaration time.
    write_file(
        &path,
        r#"sig grab: &tensor[batch, p] -> &tensor[batch, p]
def grab(x) = x
def call(xs: &tensor[3, f32]) -> &tensor[3, f32] = grab(xs)
"#,
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        errs.is_empty(),
        "finding-5 post-#219 Option A: multi-letter dim sig should accept concrete \
         caller; got errors {errs:?}"
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
        r#"sig grab: &tensor[n, p] -> &tensor[n, p]
def grab(x) = x
def call(xs: &tensor[3, f32]) -> &tensor[3, f32] = grab(xs)
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
fn polymorphic_sig_with_concrete_call_site_now_builds() {
    // WS-A8: monomorphization is implemented. A polymorphic sig with
    // a concrete call site builds successfully; the call-site
    // instantiation supplies the precision through the LowerCtx
    // prec_substitutions plumbing.
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
        output.status.success(),
        "WS-A8: chelis build must succeed for a polymorphic sig + \
         concrete call site. stderr={stderr}"
    );
    assert!(
        !stderr.contains("monomorphization"),
        "WS-A8: stderr must not surface the monomorphization tripwire \
         for a properly-typed polymorphic instantiation. stderr={stderr}"
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

// FINDING 7 (building the production stdlib `linear.ch` panics) is the
// canonical case finding 6 protects against regressing. The production
// `linear.ch` / `attention.ch` build coverage now lives in
// `monomorphization_build.rs` (which also asserts the
// monomorphization tripwire never surfaces); the production-file
// `chelis check` coverage lives in `production_stdlib_typechecks.rs`.
// Both were deduplicated out of this file so the production stdlib is
// checked/built exactly once across the suite.

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
