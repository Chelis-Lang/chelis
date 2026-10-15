//! chelis#258 Tier-2 rank polymorphism acceptance corpus.
//!
//! A single `def` generic over tensor *rank* via the `..r` spread, in the
//! identity position (`&tensor[..r, p] -> tensor[..r, p]`), replaces the
//! per-rank verb-name family (`relu_forward` / `_2d` / `_3d` / `_4d`).
//!
//! Soundness boundary (`spec/design/rank_polymorphism.md` §Soundness Boundary,
//! spec §4.2): against an opaque rank there are no named axes left to catch a
//! transposition/reshape. Spec/04 §4.5.3 admits name-trackable builtin classes
//! and checked helper compositions; an opaque row never becomes a rank count.
//!
//! Positive/negative parity per CLAUDE.md: every "checks clean" test has a
//! paired "rejected with the right reason" test.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn check_json(src: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

/// Run `chelis fmt <path>` (canonical formatter to stdout) on `src` and return
/// the formatted text. Style gate disabled so an ad-hoc `..r` def is formatted
/// without first having to satisfy every lint rule.
fn fmt_stdout(src: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap()])
        .output()
        .expect("run chelis fmt");
    assert!(
        output.status.success(),
        "chelis fmt must succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 fmt stdout")
}

fn assert_clean(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        errors.is_empty(),
        "{label}: expected no check errors, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "{label}: expected score 1.0, got {score} ({json})"
    );
}

fn assert_body_discipline_rejected(json: &Value, op: &str, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected a Body-Discipline rejection, got a clean check ({json})"
    );
    let has = errors.iter().any(|e| {
        e["message"].as_str().is_some_and(|m| {
            m.contains("rank-polymorphic def")
                && m.contains(&format!("builtin `{op}`"))
                && m.contains("not name-trackable")
        })
    });
    assert!(
        has,
        "{label}: expected a Body-Discipline error naming `{op}`, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(
        score < 1.0,
        "{label}: a rejected body must score < 1.0 ({json})"
    );
}

fn assert_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected rejection, got a clean check ({json})"
    );
}

/// Assert a rejection whose message contains `needle` — pins the *reason*, not
/// just that some error fired, so a wrong-reason regression is caught.
fn assert_rejected_with(json: &Value, needle: &str, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    let has = errors
        .iter()
        .any(|e| e["message"].as_str().is_some_and(|m| m.contains(needle)));
    assert!(
        has,
        "{label}: expected a rejection mentioning {needle:?}, got {errors:?}"
    );
}

/// A Body-Discipline rejection citing the rank-polymorphic def (used for the
/// non-`app` bypass routes — transforms, computed callees, user-fn calls).
fn assert_rank_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    let has = errors.iter().any(|e| {
        e["message"]
            .as_str()
            .is_some_and(|m| m.contains("rank-polymorphic def"))
    });
    assert!(
        has,
        "{label}: expected a rank-polymorphic Body-Discipline rejection, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(
        score < 1.0,
        "{label}: a rejected body must score < 1.0 ({json})"
    );
}

// ── Positives: one rank-poly def, callable across ranks ─────────────────

/// The headline: a single identity-position `..r` def checks clean and is
/// callable at rank 1 AND rank 2 from concrete-rank callers (chelis#258 — the
/// per-rank verb proliferation collapses to one def).
#[test]
fn identity_rank_poly_def_callable_at_ranks_1_and_2() {
    let json = check_json(
        "def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n\
         def use_rank1[n](x: &tensor[n, f32]) -> tensor[n, f32] = relu_forward(x)\n\
         def use_rank2[a, b](x: &tensor[a, b, f32]) -> tensor[a, b, f32] = relu_forward(x)\n",
    );
    assert_clean(&json, "identity rank-poly callable at rank 1 and rank 2");
}

/// chelis#258 option (b) lock: the LEGACY same-name rank-distinct overload
/// pattern — two `def relu_forward`s whose sigs differ only in rank — is
/// rejected with a `DuplicateDefinition` diagnostic, not silently accepted and
/// then mis-dispatched. The issue offered two resolutions: (a) a single
/// rank-polymorphic `..r` def (the positive above) and (b) reject the duplicate
/// `def` with a clear "Chelis does not dispatch same-name defs by rank"
/// diagnostic. Both ship; this pins (b) so the legacy overload form cannot
/// silently start type-checking again (which would re-open the mis-dispatch the
/// issue describes).
#[test]
fn same_name_rank_distinct_def_overloads_rejected_as_duplicate() {
    let json = check_json(
        "module ReproOverload\n\
         sig relu_forward[a]: &tensor[a, f32] -> tensor[a, f32]\n\
         def relu_forward(x) = relu(x)\n\
         sig relu_forward[a, b]: &tensor[a, b, f32] -> tensor[a, b, f32]\n\
         def relu_forward(x) = relu(x)\n",
    );
    assert_rejected_with(
        &json,
        "duplicate definition: `relu_forward`",
        "same-name rank-distinct def overloads must reject as DuplicateDefinition",
    );
    let kinds: Vec<&str> = json["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .filter_map(|e| e["kind"].as_str())
        .collect();
    assert!(
        kinds.contains(&"DuplicateDefinition"),
        "the rejection must be a DuplicateDefinition (option b), got {kinds:?}"
    );
}

/// Also callable at rank 3 and rank 4 — the full activation-family range.
#[test]
fn identity_rank_poly_def_callable_at_ranks_3_and_4() {
    let json = check_json(
        "def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n\
         def r3[a, b, c](x: &tensor[a, b, c, f32]) -> tensor[a, b, c, f32] = relu_forward(x)\n\
         def r4[a, b, c, d](x: &tensor[a, b, c, d, f32]) -> tensor[a, b, c, d, f32] = relu_forward(x)\n",
    );
    assert_clean(&json, "identity rank-poly callable at rank 3 and rank 4");
}

/// Multi-arg `..r`: one rank var SHARED across two params
/// (`add2(x: &tensor[..r], y: &tensor[..r]) -> tensor[..r]`) checks clean and is
/// callable at concrete ranks 1, 2, 3, AND 4 with matching shapes per call. This
/// is the §"identity `R` shared across two args" acceptance case — the rank var
/// flows through both args and the result, and an elementwise binop (`add`) is
/// the only admissible body builtin.
#[test]
fn multi_arg_rank_poly_def_callable_at_ranks_1_through_4() {
    let json = check_json(
        "def add2[r](x: &tensor[..r, f32], y: &tensor[..r, f32]) -> tensor[..r, f32] = add(x, y)\n\
         def use1[n](x: &tensor[n, f32], y: &tensor[n, f32]) -> tensor[n, f32] = add2(x, y)\n\
         def use2[a, b](x: &tensor[a, b, f32], y: &tensor[a, b, f32]) -> tensor[a, b, f32] = add2(x, y)\n\
         def use3[a, b, c](x: &tensor[a, b, c, f32], y: &tensor[a, b, c, f32]) -> tensor[a, b, c, f32] = add2(x, y)\n\
         def use4[a, b, c, d](x: &tensor[a, b, c, d, f32], y: &tensor[a, b, c, d, f32]) -> tensor[a, b, c, d, f32] = add2(x, y)\n",
    );
    assert_clean(&json, "multi-arg shared ..r callable at ranks 1-4");
}

/// Negative parity (rank mismatch): the shared `..r` forces both args to the
/// SAME rank. Calling `add2` with a rank-1 `x` and a rank-2 `y` must be rejected
/// at type-check — the rank var binds to one shape vector, so the two args
/// cannot have different ranks (no implicit broadcast, §4.2).
#[test]
fn multi_arg_rank_poly_def_rank_mismatch_rejected() {
    let json = check_json(
        "def add2[r](x: &tensor[..r, f32], y: &tensor[..r, f32]) -> tensor[..r, f32] = add(x, y)\n\
         def bad[n, a, b](x: &tensor[n, f32], y: &tensor[a, b, f32]) -> tensor[n, f32] = add2(x, y)\n",
    );
    assert_rejected(&json, "multi-arg ..r with rank-1 vs rank-2 args");
}

/// Negative parity (dim mismatch at equal rank): even when both args have the
/// same rank, the shared `..r` forces the SAME shape — differing concrete dims
/// (`tensor[2,3]` vs `tensor[3,2]`) must be rejected. The rank var carries the
/// ordered named-dim vector, not a bare count, so two same-rank-but-different-
/// shape args do not unify (§4.2 — no implicit broadcast across the rank var).
#[test]
fn multi_arg_rank_poly_def_dim_mismatch_rejected() {
    let json = check_json(
        "def add2[r](x: &tensor[..r, f32], y: &tensor[..r, f32]) -> tensor[..r, f32] = add(x, y)\n\
         def bad(x: &tensor[two, three, f32], y: &tensor[three, two, f32]) -> tensor[two, three, f32] = add2(x, y)\n",
    );
    assert_rejected(&json, "multi-arg ..r with [two,three] vs [three,two] args");
}

/// A body composing several shape-identity builtins (silu = x * sigmoid(x))
/// stays clean — composition of elementwise ops preserves the shape.
#[test]
fn composed_identity_builtins_in_rank_poly_body_clean() {
    let json = check_json(
        "def my_silu[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = mul(x, sigmoid(x))\n",
    );
    assert_clean(&json, "composed identity builtins (mul + sigmoid)");
}

// ── Negatives: Body Discipline rejects shape-rewriting ops ──────────────

/// The §4.2 trap: `permute` shares `&tv -> tv` with `relu` but transposes.
/// Under an opaque `..r` no named axis catches it, so Body Discipline must
/// reject it at the definition site.
#[test]
fn permute_in_rank_poly_body_rejected() {
    let json =
        check_json("def evil[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = permute(x, 1, 0)\n");
    assert_body_discipline_rejected(&json, "permute", "permute in ..r body");
}

/// Tier-3 (§4.5.3) supersedes Tier-2 here: a named-axis reduction is now admitted
/// in a `..r` body (see `rank_poly_tier3`), but a *positional* axis on a sole
/// spread `..r` — which has no named anchor to locate — is rejected by the
/// reduction arm, since a positional index is meaningless at symbolic rank.
#[test]
fn positional_reduce_on_sole_spread_rejected() {
    let json =
        check_json("def bad[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = sum(x, cast(0, i32))\n");
    assert_rejected_with(
        &json,
        "positional axes require a concrete-rank operand",
        "positional sum on a sole ..r spread",
    );
}

/// `reshape` (the other op that shares `&tv -> tv` with `relu`) is rejected.
#[test]
fn reshape_in_rank_poly_body_rejected() {
    let json = check_json(
        "def bad[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = reshape(x, [2i64, 3i64])\n",
    );
    assert_body_discipline_rejected(&json, "reshape", "reshape in ..r body");
}

// ── Negatives: the non-`app` bypass routes (transforms / computed callees) ──
// A shape-rewriting op must not sneak into a `..r` body by routing through a
// transform node or a non-builtin/computed callee — the §4.2 hole a pure
// "reject shape-rewriting `app` builtins" walker would leave open.

/// `vmap` applies a *referenced* user function (which here transposes) across
/// the opaque rank — rejected outright. Without this the body checks clean and
/// the build later panics on the surviving rank var.
#[test]
fn vmap_transform_in_rank_poly_body_rejected() {
    let json = check_json(
        "def inner[a, b](x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)\n\
         def evil[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = x |> vmap(inner)\n",
    );
    assert_rank_rejected(&json, "vmap transform in ..r body");
}

/// `grad(loss)(x)` routes through a transform node and a computed callee — rejected.
#[test]
fn grad_transform_in_rank_poly_body_rejected() {
    let json = check_json(
        "def loss[a](x: &tensor[a, f32]) -> tensor[f32] = sum(x, cast(0, i32))\n\
         def evil[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = grad(loss)(x)\n",
    );
    assert_rank_rejected(&json, "grad transform in ..r body");
}

/// A direct call to a user-defined function (not proven rank-safe) is rejected.
#[test]
fn user_fn_call_in_rank_poly_body_rejected() {
    let json = check_json(
        "def helper[a](x: &tensor[a, f32]) -> tensor[a, f32] = relu(x)\n\
         def evil[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = helper(x)\n",
    );
    assert_rank_rejected(&json, "user-fn call in ..r body");
}

// ── Tier-3 supersedes the old parse boundary ────────────────────────────

/// `..r` adjacent to a concrete dim is now valid Tier-3 syntax: a rank-poly
/// identity over `tensor[..r, 3]` checks clean. The Tier-2/Tier-3 boundary
/// moved from parse time to unification, where an *undetermined* split between
/// two adjacent spreads is rejected (see `rank_poly_tier3`).
#[test]
fn rank_var_adjacent_to_concrete_dim_now_checks_clean() {
    let json = check_json("def f[r](x: &tensor[..r, 3, f32]) -> tensor[..r, 3, f32] = relu(x)\n");
    assert_clean(&json, "tensor[..r, 3] adjacency now valid (Tier-3)");
}

/// Control: the same activation written WITHOUT `..r` (concrete rank) still
/// checks clean — the feature does not regress ordinary tensor defs.
#[test]
fn concrete_rank_activation_still_clean() {
    let json =
        check_json("def relu2d[a, b](x: &tensor[a, b, f32]) -> tensor[a, b, f32] = relu(x)\n");
    assert_clean(&json, "concrete-rank activation control");
}

/// Name-collision bypass (red-team #1): a user `def` that SHADOWS an Identity
/// builtin name (`relu`) and transposes must still be rejected inside a `..r`
/// body — the discipline check resolves the callee against the user defs, not
/// just the builtin-name table.
#[test]
fn shadowing_builtin_name_in_rank_poly_body_rejected() {
    let json = check_json(
        "def relu[a, b](x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)\n\
         def evil[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n",
    );
    assert_rank_rejected(&json, "user `relu` shadowing the builtin in a ..r body");
}

// ── Backend lowering: call-site rank monomorphization ───────────────────
// Follow-up to PR #286: a program that calls a `..r` def through a
// concrete-rank caller now `build`s. The lowering pipeline substitutes the
// caller's concrete shape into the callee's `(d-rank)` slots (the rank
// analogue of the precision `prec_subst` path), so every reachable tensor
// type at lowering is `Dim::Rank`-free and the inlined relu produces concrete
// numerics. See spec/design/rank_polymorphism.md.

/// Run `chelis build --target c`, then compile + run the emitted binary and
/// return its stdout. Mirrors the host-toolchain link harness used by
/// `issue_218_numerical_correctness.rs`.
fn build_compile_run(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&src, source).expect("write source");

    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "rank-poly build must succeed; stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let c_source = format!("{name}.c");
    let needs_blas = fs::read_to_string(out_dir.join(&c_source))
        .map(|t| t.contains("cblas_sgemm(") || t.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let bin = out_dir.join(name);
    let mut cc = StdCommand::new(&toolchain.compiler);
    cc.current_dir(&out_dir)
        .arg("-O2")
        .args(&toolchain.compile_flags)
        .arg(&c_source)
        .arg("libchelis_runtime.a")
        .args(&toolchain.link_flags)
        .args(["-o", bin.to_str().unwrap()]);
    let link = cc.status().expect("host compiler runs");
    assert!(link.success(), "link of rank-poly C must succeed: {link}");

    let run = StdCommand::new(&bin).output().expect("binary runs");
    assert!(
        run.status.success(),
        "rank-poly binary must run: {}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

/// Run `chelis eval --file` and return its stdout (the evaluator oracle the
/// backend must agree with, per the backend-numerics discipline).
fn eval_stdout(dir: &Path, source: &str, name: &str) -> String {
    let src = dir.join(format!("{name}.ch"));
    fs::write(&src, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            src.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "rank-poly eval must succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

/// Parse `name = tensor(shape=[..], data=[..])` lines from printed output into
/// `(name, shape, data)` triples — the shared shape/numeric oracle for
/// backend-vs-evaluator agreement.
fn parse_printed_tensors(stdout: &str) -> Vec<(String, Vec<usize>, Vec<f64>)> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let Some((name, rest)) = line.split_once(" = tensor(") else {
            continue;
        };
        let shape = rest
            .split_once("shape=[")
            .and_then(|(_, s)| s.split_once(']'))
            .map(|(s, _)| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let data = rest
            .split_once("data=[")
            .and_then(|(_, s)| s.split_once(']'))
            .map(|(s, _)| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<f64>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        out.push((name.trim().to_string(), shape, data));
    }
    out
}

/// Headline backend acceptance: ONE rank-poly `relu_forward` def, called at
/// concrete ranks 1, 2, 3, AND 4, builds and runs — and the output equals the
/// input with negatives zeroed (the identity/elementwise tier means
/// out[i] == relu(in[i]) at every rank). The compiled-binary output is also
/// asserted byte-for-byte against the evaluator oracle (backend-numerics
/// eval-vs-backend agreement).
#[test]
fn rank_poly_identity_builds_and_runs_at_ranks_1_through_4() {
    let source = "def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n\
         def r1[n](x: &tensor[n, f32]) -> tensor[n, f32] = relu_forward(x)\n\
         def r2[a, b](x: &tensor[a, b, f32]) -> tensor[a, b, f32] = relu_forward(x)\n\
         def r3[a, b, c](x: &tensor[a, b, c, f32]) -> tensor[a, b, c, f32] = relu_forward(x)\n\
         def r4[a, b, c, d](x: &tensor[a, b, c, d, f32]) -> tensor[a, b, c, d, f32] = relu_forward(x)\n\
         out1 = r1(to_tensor([-1.0, 2.0, -3.0, 4.0], f32))\n\
         out2 = r2(to_tensor([[-1.0, 2.0], [3.0, -4.0]], f32))\n\
         out3 = r3(to_tensor([[[-1.0, 2.0]], [[3.0, -4.0]]], f32))\n\
         out4 = r4(to_tensor([[[[-5.0, 6.0]]]], f32))\n";

    let backend = build_compile_run(source, "rank_poly_identity");
    let backend_tensors = parse_printed_tensors(&backend);

    // Expected: relu zeroes the negatives, preserves shape, at every rank.
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out1", &[4], &[0.0, 2.0, 0.0, 4.0]),
        ("out2", &[2, 2], &[0.0, 2.0, 3.0, 0.0]),
        ("out3", &[2, 1, 2], &[0.0, 2.0, 3.0, 0.0]),
        ("out4", &[1, 1, 1, 2], &[0.0, 6.0]),
    ];
    for (name, shape, data) in expected {
        let got = backend_tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }

    // Backend must agree with the evaluator oracle, value-for-value.
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "rank_poly_identity");
    let eval_tensors = parse_printed_tensors(&eval);
    for (name, shape, data) in &backend_tensors {
        let e = eval_tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("evaluator output missing `{name}`: {eval}"));
        assert_eq!(shape, &e.1, "{name}: eval-vs-backend shape disagreement");
        assert_eq!(data.len(), e.2.len(), "{name}: eval-vs-backend length");
        for (i, (b, ev)) in data.iter().zip(e.2.iter()).enumerate() {
            assert!(
                (b - ev).abs() < 1e-6,
                "{name}[{i}]: eval-vs-backend disagreement: backend {b} vs eval {ev}"
            );
        }
    }
}

/// Multi-arg backend acceptance: ONE rank-poly `add2` def, with a single `..r`
/// SHARED across both args and the result, called at concrete ranks 1, 2, 3, AND
/// 4 builds and runs — and the output equals the elementwise sum at every rank
/// (call-site rank monomorphization resolves the shared `(d-rank)` slot to the
/// caller's concrete shape in all positions). The compiled-binary output is also
/// asserted value-for-value against the evaluator oracle (eval-vs-backend
/// agreement). The §"identity `R` shared across two args" backend acceptance.
#[test]
fn multi_arg_rank_poly_builds_and_runs_at_ranks_1_through_4() {
    let source = "def add2[r](x: &tensor[..r, f32], y: &tensor[..r, f32]) -> tensor[..r, f32] = add(x, y)\n\
         def r1[n](x: &tensor[n, f32], y: &tensor[n, f32]) -> tensor[n, f32] = add2(x, y)\n\
         def r2[a, b](x: &tensor[a, b, f32], y: &tensor[a, b, f32]) -> tensor[a, b, f32] = add2(x, y)\n\
         def r3[a, b, c](x: &tensor[a, b, c, f32], y: &tensor[a, b, c, f32]) -> tensor[a, b, c, f32] = add2(x, y)\n\
         def r4[a, b, c, d](x: &tensor[a, b, c, d, f32], y: &tensor[a, b, c, d, f32]) -> tensor[a, b, c, d, f32] = add2(x, y)\n\
         out1 = r1(to_tensor([1.0, 2.0, 3.0, 4.0], f32), to_tensor([10.0, 20.0, 30.0, 40.0], f32))\n\
         out2 = r2(to_tensor([[1.0, 2.0], [3.0, 4.0]], f32), to_tensor([[10.0, 20.0], [30.0, 40.0]], f32))\n\
         out3 = r3(to_tensor([[[1.0, 2.0]], [[3.0, 4.0]]], f32), to_tensor([[[10.0, 20.0]], [[30.0, 40.0]]], f32))\n\
         out4 = r4(to_tensor([[[[5.0, 6.0]]]], f32), to_tensor([[[[50.0, 60.0]]]], f32))\n";

    let backend = build_compile_run(source, "rank_poly_add2");
    let backend_tensors = parse_printed_tensors(&backend);

    // Expected: elementwise sum, shape preserved, at every rank.
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out1", &[4], &[11.0, 22.0, 33.0, 44.0]),
        ("out2", &[2, 2], &[11.0, 22.0, 33.0, 44.0]),
        ("out3", &[2, 1, 2], &[11.0, 22.0, 33.0, 44.0]),
        ("out4", &[1, 1, 1, 2], &[55.0, 66.0]),
    ];
    for (name, shape, data) in expected {
        let got = backend_tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        assert_eq!(got.2.len(), data.len(), "{name}: backend len ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }

    // Backend must agree with the evaluator oracle, value-for-value.
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "rank_poly_add2");
    let eval_tensors = parse_printed_tensors(&eval);
    for (name, shape, data) in &backend_tensors {
        let e = eval_tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("evaluator output missing `{name}`: {eval}"));
        assert_eq!(shape, &e.1, "{name}: eval-vs-backend shape disagreement");
        assert_eq!(data.len(), e.2.len(), "{name}: eval-vs-backend length");
        for (i, (b, ev)) in data.iter().zip(e.2.iter()).enumerate() {
            assert!(
                (b - ev).abs() < 1e-6,
                "{name}[{i}]: eval-vs-backend disagreement: backend {b} vs eval {ev}"
            );
        }
    }
}

/// A composed shape-identity body (`silu = x * sigmoid(x)`) through the
/// rank-poly def also builds and runs at a concrete rank — composition of
/// elementwise ops stays rank-monomorphizable.
#[test]
fn rank_poly_composed_identity_builds_and_runs() {
    let source = "def my_silu[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = mul(x, sigmoid(x))\n\
         def use2d[a, b](x: &tensor[a, b, f32]) -> tensor[a, b, f32] = my_silu(x)\n\
         out = use2d(to_tensor([[0.0, 1.0], [-1.0, 2.0]], f32))\n";
    let backend = build_compile_run(source, "rank_poly_silu");
    let tensors = parse_printed_tensors(&backend);
    let (_, shape, data) = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("missing `out`: {backend}"));
    assert_eq!(shape, &[2, 2], "silu output shape ({backend})");
    // silu(x) = x * sigmoid(x); silu(0)=0, silu(1)=0.731..., silu(-1)=-0.268...,
    // silu(2)=1.761...
    let expected = [0.0, 0.7310586, -0.2689414, 1.7615942];
    for (i, (g, e)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "silu[{i}]: backend {g} != expected {e} ({backend})"
        );
    }
}

/// Positive `grad` × rank-poly composition — the companion to
/// `grad_transform_in_rank_poly_body_rejected`. `grad` INSIDE a `..r` body
/// is rejected by Body Discipline, but differentiating a CONCRETE function
/// whose body inlines a `..r` rank-poly callee is valid and must build, run,
/// and yield the correct gradient. When `grad(loss)` is lowered, the grad
/// sub-context monomorphizes the inlined `sq`'s `(d-rank)` slot to the
/// concrete caller shape — the rank analogue of issue #289's
/// precision-through-grad-sub-context path (this exercises the
/// `subctx.rank_substitutions` seeding added when #286 rebased onto #289).
/// `d/dx Σ(x²) = 2x`, so the gradient over `[1, 2, 3]` is `[2, 4, 6]`,
/// asserted on the compiled binary and against the evaluator oracle.
#[test]
fn grad_over_rank_poly_callee_builds_runs_and_matches_oracle() {
    let source = "def sq[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = mul(x, x)\n\
         def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(sq(&x), cast(0, i32)))\n\
         def dloss(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)\n\
         out = dloss(to_tensor([1.0, 2.0, 3.0], f32))\n";

    let backend = build_compile_run(source, "grad_rank_poly_callee");
    let backend_tensors = parse_printed_tensors(&backend);
    let (_, shape, data) = backend_tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(shape, &[3], "grad output shape ({backend})");
    // d/dx of sum(x^2) is 2x; over [1, 2, 3] that is [2, 4, 6].
    let expected = [2.0, 4.0, 6.0];
    assert_eq!(data.len(), expected.len(), "grad output len ({backend})");
    for (i, (g, e)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "grad[{i}]: backend {g} != expected {e} (d/dx sum(x^2) = 2x) ({backend})"
        );
    }

    // Backend must agree with the evaluator oracle, value-for-value.
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "grad_rank_poly_callee");
    // `chelis eval` prints a SINGLE root as a bare `tensor(...)` with no
    // `name =` prefix (unlike the compiled binary, which names every root),
    // so normalize it for the shared `name = tensor(...)` parser.
    let eval_named = if eval.contains(" = tensor(") {
        eval.clone()
    } else {
        format!("out = {}", eval.trim())
    };
    let eval_tensors = parse_printed_tensors(&eval_named);
    let (_, eshape, edata) = eval_tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("evaluator output missing `out`: {eval}"));
    assert_eq!(shape, eshape, "grad: eval-vs-backend shape disagreement");
    assert_eq!(data.len(), edata.len(), "grad: eval-vs-backend length");
    for (i, (b, ev)) in data.iter().zip(edata.iter()).enumerate() {
        assert!(
            (b - ev).abs() < 1e-6,
            "grad[{i}]: eval-vs-backend disagreement: backend {b} vs eval {ev}"
        );
    }
}

/// Negative parity: a body that violates §4.2 shape-identity (`permute`) is
/// STILL rejected at type-check — call-site rank monomorphization does not
/// loosen the Body-Discipline gate. The negative paired with the positive
/// build above: shape-rewriting under `..r` never reaches the backend because
/// it never type-checks.
#[test]
fn rank_poly_shape_rewriting_body_still_rejected_at_typecheck() {
    let json =
        check_json("def evil[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = permute(x, 1, 0)\n");
    assert_body_discipline_rejected(
        &json,
        "permute",
        "shape-rewriting body must still be rejected pre-build",
    );
}

// ── Formatter round-trip: `..r` survives `chelis fmt` ───────────────────
// The canonical surf formatter (`chelis_surf::format`) must emit and re-parse a
// rank-poly sig/def identically — the formatter-round-trip invariant from
// CLAUDE.md, extended to the `..r` surface. `TypeExpr::RankSpread(name)` prints
// as `..{name}`, and a single round-trip plus an idempotence pass lock it.

/// A rank-poly def survives `chelis fmt`: the formatted output still carries the
/// `..r` spread in every tensor position (param + result), formatting is
/// idempotent (a second pass is byte-identical to the first), and the formatted
/// text still type-checks clean. This locks the `(d-rank)` ↔ `..r` round-trip.
#[test]
fn rank_poly_def_survives_fmt_round_trip() {
    let src = "def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n";
    let formatted = fmt_stdout(src);

    // The spread marker survives in both the param and the result type.
    assert_eq!(
        formatted.matches("..r").count(),
        2,
        "formatted rank-poly def must keep `..r` in both param and result: {formatted:?}"
    );
    assert!(
        formatted.contains("tensor[..r, f32]"),
        "formatted def must keep the `tensor[..r, f32]` shape: {formatted:?}"
    );

    // Idempotence: re-formatting the formatted text is byte-identical (the
    // round-trip has reached the fixed point — `..r` re-parses to the same AST).
    let reformatted = fmt_stdout(&formatted);
    assert_eq!(
        formatted, reformatted,
        "chelis fmt must be idempotent on a rank-poly def (round-trip stable)"
    );

    // The formatted text is still a well-typed rank-poly def, not just lexically
    // intact — re-parsing preserved the `..r` semantics, not only the bytes.
    let json = check_json(&formatted);
    assert_clean(&json, "formatted rank-poly def still type-checks");
}

/// Multi-arg parity: a `..r` shared across two params also survives `chelis fmt`
/// (three `..r` occurrences: two params + the result), idempotently.
#[test]
fn multi_arg_rank_poly_def_survives_fmt_round_trip() {
    let src =
        "def add2[r](x: &tensor[..r, f32], y: &tensor[..r, f32]) -> tensor[..r, f32] = add(x, y)\n";
    let formatted = fmt_stdout(src);
    assert_eq!(
        formatted.matches("..r").count(),
        3,
        "formatted multi-arg rank-poly def must keep `..r` in both params and the result: {formatted:?}"
    );
    let reformatted = fmt_stdout(&formatted);
    assert_eq!(
        formatted, reformatted,
        "chelis fmt must be idempotent on a multi-arg rank-poly def"
    );
    let json = check_json(&formatted);
    assert_clean(&json, "formatted multi-arg rank-poly def still type-checks");
}

// chelis#3668: body-derived evidence composes across rank-generic declarations.

#[test]
fn checked_helper_composition_3668_formats_checks_evaluates_and_executes_c() {
    for (bound, precision) in [("Float", "f32"), ("{f32, f64}", "f64")] {
        let source = format!(
            "def negate[r, p: {bound}](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n\
             def middle[s, q: {bound}](x: &tensor[..s, q]) -> tensor[..s, q] = negate(x)\n\
             def outer[t, v: {bound}](x: &tensor[..t, v]) -> tensor[..t, v] = middle(x)\n\
             out0 = outer(scalar_to_tensor(1.0{precision}))\n\
             observed_rank = rank(out0)\n\
             observed_value = tensor_to_scalar(out0)\n\
             out1 = outer(to_tensor([1.0, 2.0, 3.0], {precision}))\n\
             out2 = outer(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], {precision}))\n\
             out3 = outer(to_tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], {precision}))\n"
        );
        let formatted = fmt_stdout(&source);
        assert_eq!(formatted, fmt_stdout(&formatted), "canonical idempotence");
        assert_clean(&check_json(&formatted), "checked helper composition");
        let dir = tempdir().expect("tempdir");
        let eval_text = eval_stdout(dir.path(), &formatted, "helper_eval");
        let native_text = build_compile_run(&formatted, "helper_native");
        for (label, text) in [("Eval", &eval_text), ("native C", &native_text)] {
            let scalar = |name: &str| -> f64 {
                let prefix = format!("{name} = ");
                text.lines()
                    .find_map(|line| line.strip_prefix(&prefix))
                    .unwrap_or_else(|| panic!("{label}: missing {name}: {text}"))
                    .parse()
                    .expect("exact finite scalar observation")
            };
            assert_eq!(scalar("observed_rank"), 0.0, "{label}: rank-zero result");
            assert_eq!(scalar("observed_value"), -1.0, "{label}: rank-zero value");
        }
        let eval = parse_printed_tensors(&eval_text);
        let native = parse_printed_tensors(&native_text);
        let expected: &[(&str, &[usize], &[f64])] = &[
            ("out1", &[3], &[-1.0, -2.0, -3.0]),
            ("out2", &[2, 3], &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
            ("out3", &[2, 1, 3], &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
        ];
        for (name, shape, data) in expected {
            for (label, output) in [("Eval", &eval), ("native C", &native)] {
                let (_, got_shape, got_data) = output
                    .iter()
                    .find(|(got_name, _, _)| got_name == name)
                    .unwrap_or_else(|| panic!("{label}: missing {name}: {output:?}"));
                assert_eq!(
                    got_shape, shape,
                    "{label}/{precision}/{name}: ordered shape"
                );
                assert_eq!(
                    got_data, data,
                    "{label}/{precision}/{name}: exact finite values"
                );
            }
        }
    }
}

#[test]
fn checked_named_helper_composition_3668_keeps_axis_sensitive_values() {
    let source = "def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
        def wrapped[before, after](x: &tensor[..before, seq, ..after, f32]) -> tensor[..before, ..after, f32] = reduce_seq(x)\n\
        def concrete(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = wrapped(x)\n\
        out = concrete(to_tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]], f32))\n";
    let formatted = fmt_stdout(source);
    assert_clean(&check_json(&formatted), "unitary helper anchors");
    let dir = tempdir().expect("tempdir");
    for output in [
        eval_stdout(dir.path(), &formatted, "anchor_eval"),
        build_compile_run(&formatted, "anchor_native"),
    ] {
        let tensors = parse_printed_tensors(&output);
        let (_, shape, data) = tensors
            .iter()
            .find(|(name, _, _)| name == "out")
            .unwrap_or_else(|| panic!("missing output: {output}"));
        assert_eq!(shape, &[2, 4]);
        assert_eq!(data, &[15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0]);
    }
}

#[test]
fn checked_helpers_3668_still_reject_ambiguous_anchor_and_dtype_calls() {
    for (declarations, caller, needle) in [
        (
            "def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
             def wrapped[before, after](x: &tensor[..before, seq, ..after, f32]) -> tensor[..before, ..after, f32] = reduce_seq(x)\n",
            "def bad(x: &tensor[seq, seq, f32]) -> tensor[seq, f32] = wrapped(x)\n",
            "seq",
        ),
        (
            "def negate[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n\
             def wrapped[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = negate(x)\n",
            "out = wrapped(to_tensor([1i32, 2i32]))\n",
            "Float",
        ),
    ] {
        let json = check_json(&format!("{declarations}{caller}"));
        assert_rejected_with(&json, needle, "helper summaries preserve call obligations");
    }
}

fn helper_package_3668() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("rank-helper-composition");
    fs::create_dir_all(root.join("src")).expect("source directory");
    fs::write(root.join("reef.toml"), format!(
        "schema = \"3\"\n[package]\nname = \"rank-helper-composition\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"RankCompose\"\nresolver = \"2\"\n",
        chelis_compiler_api::COMPILER_VERSION
    )).expect("manifest");
    fs::write(root.join("src/helper.ch"),
        "module RankCompose.Helper\nexport (negate)\ndef negate[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n"
    ).expect("helper");
    fs::write(root.join("src/middle.ch"),
        "module RankCompose.Middle\nimport RankCompose.Helper\nexport (middle)\ndef middle[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = RankCompose.Helper.negate(x)\n"
    ).expect("middle");
    fs::write(root.join("src/main.ch"),
        "module RankCompose.Main\nimport RankCompose.Middle (middle)\nexport (outer)\ndef outer[t, v: Float](x: &tensor[..t, v]) -> tensor[..t, v] = middle(x)\nout = outer(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f32))\n"
    ).expect("main");
    (dir, root)
}

fn helper_package_command_3668(root: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("package command runs")
}

#[test]
fn imported_checked_helpers_3668_use_declaring_identity_across_two_hops() {
    let (_dir, root) = helper_package_3668();
    for name in ["helper", "middle", "main"] {
        let source = format!("src/{name}.ch");
        let formatted = helper_package_command_3668(&root, &["fmt", "--inplace", &source]);
        assert!(
            formatted.status.success(),
            "package formatting: {formatted:?}"
        );
    }
    let check = helper_package_command_3668(&root, &["check", "src/main.ch"]);
    assert!(
        check.status.success(),
        "imported checked helpers: {check:?}"
    );
    assert_clean(
        &serde_json::from_slice(&check.stdout).expect("check JSON"),
        "imported helpers",
    );
    let eval = helper_package_command_3668(&root, &["eval", "--file", "src/main.ch"]);
    assert!(eval.status.success(), "imported helper Eval: {eval:?}");
    let build = helper_package_command_3668(
        &root,
        &[
            "build",
            "--emit-c",
            "src/main.ch",
            "--target",
            "c",
            "--output",
            "native",
        ],
    );
    assert!(
        build.status.success(),
        "imported helper C generation: {build:?}"
    );
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let output = root.join("native");
    let binary = output.join("rank_helpers");
    let link = StdCommand::new(&toolchain.compiler)
        .current_dir(&output)
        .arg("-O2")
        .args(&toolchain.compile_flags)
        .args(["main.c", "libchelis_runtime.a"])
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("native link");
    assert!(link.status.success(), "imported helper link: {link:?}");
    let native = StdCommand::new(&binary)
        .output()
        .expect("execute imported helper binary");
    assert!(
        native.status.success(),
        "imported helper execution: {native:?}"
    );
    for bytes in [&eval.stdout, &native.stdout] {
        let text = std::str::from_utf8(bytes).expect("UTF-8 output");
        let tensors = parse_printed_tensors(text);
        assert!(
            tensors.iter().any(|(_, shape, data)| shape == &[2, 3]
                && data == &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
            "exact imported result: {text}"
        );
    }
}

#[test]
fn imported_helper_name_3668_does_not_authorize_a_shadowing_parameter() {
    let (_dir, root) = helper_package_3668();
    fs::write(root.join("src/main.ch"),
        "module RankCompose.Main\nimport RankCompose.Middle (middle)\nexport (outer)\ndef outer[r, p: Float](middle: &tensor[..r, p] -> tensor[..r, p], x: &tensor[..r, p]) -> tensor[..r, p] = middle(x)\n"
    ).expect("shadow fixture");
    let output = helper_package_command_3668(&root, &["check", "src/main.ch"]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "shadow must reject: {output:?}"
    );
    let json = serde_json::from_slice(&output.stdout).expect("check JSON");
    assert_rank_rejected(&json, "imported name shadow has no body evidence");
}

#[test]
fn recursive_checked_helpers_3668_execute_both_finite_branches_at_two_dtypes() {
    for (bound, precision) in [("Float", "f32"), ("{f32, f64}", "f64")] {
        let source = format!(
            "def left[r, p: {bound}](x: &tensor[..r, p], stop: bool) -> tensor[..r, p] = if stop then neg(x) else right(x, true)\n\
             def right[s, q: {bound}](x: &tensor[..s, q], stop: bool) -> tensor[..s, q] = if stop then neg(x) else left(x, true)\n\
             direct = left(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], {precision}), true)\n\
             recursive = left(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], {precision}), false)\n"
        );
        let formatted = fmt_stdout(&source);
        assert_eq!(formatted, fmt_stdout(&formatted));
        assert_clean(
            &check_json(&formatted),
            "jointly checked recursive helper component",
        );
        let dir = tempdir().expect("tempdir");
        for output in [
            eval_stdout(dir.path(), &formatted, "recursive_helpers_eval"),
            build_compile_run(&formatted, "recursive_helpers_native"),
        ] {
            let tensors = parse_printed_tensors(&output);
            for expected_name in ["direct", "recursive"] {
                let (_, shape, data) = tensors
                    .iter()
                    .find(|(name, _, _)| name == expected_name)
                    .unwrap_or_else(|| panic!("missing {expected_name}: {output}"));
                assert_eq!(shape, &[2, 3]);
                assert_eq!(data, &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
            }
        }
    }
}

#[test]
fn recursive_and_acyclic_row_evidence_3668_do_not_discharge_effect_checks() {
    let recursive = "def left[r, p: Float](x: &tensor[..r, p], stop: bool) -> tensor[..r, p] ! {IO} = if stop then neg(x) else right(x, true)\n\
        def right[s, q: Float](x: &tensor[..s, q], stop: bool) -> tensor[..s, q] ! {IO} = { shown = print(\"recursive body\")\n if stop then neg(x) else left(x, true) }\n";
    let positive = format!(
        "{recursive}def outer[t, v: Float](x: &tensor[..t, v]) -> tensor[..t, v] ! {{IO}} = left(x, false)\n"
    );
    assert_clean(
        &check_json(&positive),
        "joint row evidence with inherited IO",
    );
    let negative = format!(
        "{recursive}def outer[t, v: Float](x: &tensor[..t, v]) -> tensor[..t, v] ! {{}} = left(x, false)\n"
    );
    assert_rejected_with(&check_json(&negative), "IO", "SCC effects still enforced");
    let negative = "def noisy[r](x: &tensor[..r, f32]) -> tensor[..r, f32] ! {IO} = { shown = print(\"called\")\n neg(x) }\n\
        def outer[s](x: &tensor[..s, f32]) -> tensor[..s, f32] ! {} = noisy(x)\n";
    assert_rejected_with(
        &check_json(negative),
        "IO",
        "acyclic effects still enforced",
    );
}
