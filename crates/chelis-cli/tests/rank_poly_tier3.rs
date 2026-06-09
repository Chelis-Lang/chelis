//! chelis#258 Tier-3 rank polymorphism acceptance corpus.
//!
//! Name-preserving rank *arithmetic*: a single `def` reduces a named axis in a
//! rank-polymorphic way and the type/shape checker computes the resulting shape
//! symbolically, carrying the surviving named axes through. The headline form
//! is the named-axis reduction:
//!
//! ```text
//! def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32]
//!   = sum(x, seq)
//! ```
//!
//! A tensor shape is now `Rank? (Name Rank?)*` (spreads interleaved with named
//! anchors). Unification locates each named anchor uniquely in the operand and
//! binds the spreads between — unitary because each split is forced by a name.
//!
//! Soundness boundary (`spec/04-type-system.md` §4.5.3, §4.2): a spread preserves
//! the real named dims it covers (not a count), the reduced axis is a retained
//! name, order is preserved, and the Body-Discipline check admits only
//! name-trackable ops (elementwise + named reductions) — a positional
//! `permute`/`reshape` at symbolic rank is rejected.
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

/// Assert a rejection whose message contains `needle` — pins the *reason*, not
/// just that some error fired, so a wrong-reason regression is caught.
fn assert_rejected_with(json: &Value, needle: &str, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected rejection, got a clean check ({json})"
    );
    let has = errors
        .iter()
        .any(|e| e["message"].as_str().is_some_and(|m| m.contains(needle)));
    assert!(
        has,
        "{label}: expected a rejection mentioning {needle:?}, got {errors:?}"
    );
}

// ── Positives ───────────────────────────────────────────────────────────

/// The headline: ONE rank-poly named-reduce def, callable at ranks 2, 3, and 4
/// from concrete-rank callers, with every surviving named axis carried through.
/// This collapses the `batchnorm1d`/`_2d`/`_3d` verb-name family into one def
/// (chelis#258) for the reduction case.
#[test]
fn named_reduce_callable_at_ranks_2_3_4() {
    let json = check_json(
        "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[seq, hidden, f32]) -> tensor[hidden, f32] = reduce_seq(x)\n\
         def use3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = reduce_seq(x)\n\
         def use4(x: &tensor[batch, depth, seq, hidden, f32]) -> tensor[batch, depth, hidden, f32] = reduce_seq(x)\n",
    );
    assert_clean(&json, "named reduce callable at ranks 2-4");
}

/// A leading anchor with a trailing spread: reduce the first (named) axis.
#[test]
fn reduce_leading_named_axis() {
    let json = check_json(
        "def drop_batch(x: &tensor[batch, ..rest, f32]) -> tensor[..rest, f32] = sum(x, batch)\n\
         def use(x: &tensor[batch, seq, hidden, f32]) -> tensor[seq, hidden, f32] = drop_batch(x)\n",
    );
    assert_clean(&json, "reduce leading named axis");
}

/// A trailing anchor with a leading spread: reduce the last (named) axis.
#[test]
fn reduce_trailing_named_axis() {
    let json = check_json(
        "def drop_last(x: &tensor[..pre, hidden, f32]) -> tensor[..pre, f32] = sum(x, hidden)\n\
         def use(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, seq, f32] = drop_last(x)\n",
    );
    assert_clean(&json, "reduce trailing named axis");
}

/// Multi-axis (the user's "reduce over 2 axes") via composition: drop two named
/// axes by composing single-axis named reductions. The intermediate type with
/// adjacent spreads is sound — it is only ever the operand of the next reduce
/// (located by name), never split against a ground.
#[test]
fn multi_axis_reduce_via_composition() {
    let json = check_json(
        "def reduce_two(x: &tensor[..a, seq, ..b, head, ..c, f32]) -> tensor[..a, ..b, ..c, f32] = sum(sum(x, head), seq)\n\
         def use(x: &tensor[batch, seq, kv, head, feat, f32]) -> tensor[batch, kv, feat, f32] = reduce_two(x)\n",
    );
    assert_clean(&json, "multi-axis reduce over seq and head via composition");
}

/// `mean` is a named-axis reduction too (not only `sum`) — both lower through
/// the tensor-DAG backend and build end-to-end.
#[test]
fn mean_is_name_tracked() {
    let json = check_json(
        "def m(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = mean(x, seq)\n",
    );
    assert_clean(&json, "mean name-tracked in a ..r body");
}

/// `max_reduce`/`min_reduce`/`prod_reduce` are NOT yet admitted in a `..r` body:
/// they route through the host lane in a rank-poly inline and don't compile
/// (chelis#340), so they are rejected at check time to keep check↔backend in
/// sync (a check-clean program must build). They remain usable at concrete rank.
#[test]
fn max_reduce_in_rank_poly_body_rejected() {
    let json = check_json(
        "def m(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = max_reduce(x, seq)\n",
    );
    assert_rejected_with(&json, "name-trackable", "max_reduce in a ..r body");
}

/// Concrete-rank control: a named reduction on a fully-concrete shape (no
/// spread) drops the named axis and keeps the rest — the feature does not
/// require a spread, and does not regress ordinary defs.
#[test]
fn concrete_rank_named_reduce_clean() {
    let json = check_json(
        "def f(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = sum(x, seq)\n",
    );
    assert_clean(&json, "concrete-rank named reduce");
}

/// An elementwise op composed with a named reduction in the same `..r` body.
#[test]
fn elementwise_then_reduce_in_rank_poly_body() {
    let json = check_json(
        "def f(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(relu(x), seq)\n",
    );
    assert_clean(&json, "relu then named reduce in a ..r body");
}

// ── Negatives ───────────────────────────────────────────────────────────

/// The reduced axis must actually be dropped: declaring a return that keeps the
/// reduced axis is rejected (the body's symbolic output does not unify).
#[test]
fn declared_return_keeps_reduced_axis_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, seq, ..post, f32] = sum(x, seq)\n",
    );
    assert_rejected_with(
        &json,
        "doesn't match declared signature",
        "return keeps the reduced `seq` axis",
    );
}

/// Reducing an axis the operand does not have is a hard error.
#[test]
fn reduce_nonexistent_axis_rejected() {
    let json = check_json(
        "def f(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = sum(x, nope)\n",
    );
    assert_rejected_with(
        &json,
        "neither a compile-time constant nor a named axis",
        "reduce a non-existent axis `nope`",
    );
}

/// Name↔Lit hard-reject: a fully-literal caller carries no name to locate the
/// anchor, so the named reduction cannot resolve — rejected, never a guess.
#[test]
fn fully_literal_operand_cannot_locate_anchor() {
    let json = check_json(
        "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use(x: &tensor[2, 768, 64, f32]) -> tensor[2, 64, f32] = reduce_seq(x)\n",
    );
    assert_rejected_with(&json, "seq", "fully-literal operand has no named seq axis");
}

/// An ambiguous anchor (the name appears more than once in the operand) is
/// rejected rather than silently reducing the first occurrence.
#[test]
fn ambiguous_anchor_rejected() {
    let json = check_json(
        "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use(x: &tensor[seq, mid, seq, f32]) -> tensor[mid, f32] = reduce_seq(x)\n",
    );
    assert_rejected_with(&json, "ambiguous", "anchor `seq` appears twice");
}

/// Two adjacent spreads split against a concrete operand is the undetermined
/// (non-unitary) case — rejected at the call site.
#[test]
fn adjacent_spreads_split_rejected() {
    let json = check_json(
        "def bad(x: &tensor[..a, ..b, f32]) -> tensor[..a, ..b, f32] = relu(x)\n\
         def use(x: &tensor[m, n, f32]) -> tensor[m, n, f32] = bad(x)\n",
    );
    assert_rejected_with(&json, "two adjacent rank spreads", "adjacent-spread split");
}

/// A positional shape-rewriting op (`permute`) in a `..r` body is rejected:
/// not name-trackable at symbolic rank (§4.2 / §4.5.3).
#[test]
fn permute_in_rank_poly_body_rejected() {
    let json = check_json(
        "def evil(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = permute(x, 1, 0)\n",
    );
    assert_rejected_with(&json, "name-trackable", "permute in a ..r body");
}

/// `reshape` (fuses axes, destroys names) in a `..r` body is rejected.
#[test]
fn reshape_in_rank_poly_body_rejected() {
    let json = check_json(
        "def evil(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = reshape(x, [2, 3])\n",
    );
    assert_rejected_with(&json, "shape-rewriting", "reshape in a ..r body");
}

/// A duplicate spread name in one tensor shape is a parse error (it would bind
/// the same run twice).
#[test]
fn duplicate_spread_name_rejected() {
    let json =
        check_json("def f(x: &tensor[..r, seq, ..r, f32]) -> tensor[..r, f32] = sum(x, seq)\n");
    assert_rejected_with(
        &json,
        "distinct rank-spread name",
        "duplicate spread name `..r`",
    );
}

// ── Formatter round-trip ────────────────────────────────────────────────

/// The anchored multi-spread sig survives `chelis fmt`: `..pre`/`..post` are
/// preserved in every tensor position, fmt is idempotent, and the formatted
/// text re-checks clean (the CLAUDE.md formatter round-trip invariant).
#[test]
fn anchored_spread_survives_fmt_round_trip() {
    let src = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n";
    let once = fmt_stdout(src);
    assert!(
        once.contains("..pre") && once.contains("..post") && once.contains("seq"),
        "fmt must preserve the spreads and the anchor, got:\n{once}"
    );
    let twice = fmt_stdout(&once);
    assert_eq!(
        once, twice,
        "chelis fmt must be idempotent on the anchored sig"
    );
    assert_clean(&check_json(&once), "formatted anchored sig re-checks clean");
}

// ── Backend build + run (call-site rank monomorphization) ───────────────
// A program that calls a rank-poly named-reduce def through a concrete-rank
// caller builds, compiles, and runs: call-site monomorphization substitutes
// the caller's concrete shape for each spread, and the named axis `seq` is
// resolved to a positional index at lowering (where the operand's named dims
// are available). The output value is verified directly against the hand-
// computed reduction AND against the `chelis eval` oracle: since chelis#338,
// the host runtime routes a def call that requires named-axis resolution
// through the same `lower_subexpr_program` + forward-DAG-eval lane the C
// backend uses (see `apply_named_axis_def_call` in
// crates/chelis-compiler-api/src/runtime.rs), so eval-vs-backend agreement
// is restored for the Tier-3 surface.

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
        "rank-poly reduce build must succeed; stderr: {}",
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
        .args(["-L.", "-lchelis_runtime"])
        .args(&toolchain.link_flags)
        .args(["-o", bin.to_str().unwrap()]);
    let link = cc.status().expect("host compiler runs");
    assert!(
        link.success(),
        "link of rank-poly reduce C must succeed: {link}"
    );

    let run = StdCommand::new(&bin).output().expect("binary runs");
    assert!(
        run.status.success(),
        "rank-poly reduce binary must run: {}\nstderr: {}",
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
        "rank-poly named-reduce eval must succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

/// Run `chelis eval --file` expecting failure; return stderr for message pins.
fn eval_stderr_expecting_failure(dir: &Path, source: &str, name: &str) -> String {
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
        !output.status.success(),
        "eval was expected to fail; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8(output.stderr).expect("utf-8 stderr")
}

/// Assert every backend-printed tensor has an eval twin within 1e-6 (the
/// Tier-2 eval-vs-backend agreement oracle, unblocked for Tier-3 by #338).
fn assert_eval_agrees_with_backend(source: &str, name: &str, backend: &str) {
    let backend_tensors = parse_printed_tensors(backend);
    assert!(
        !backend_tensors.is_empty(),
        "{name}: backend printed no tensors: {backend}"
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, name);
    let eval_tensors = parse_printed_tensors(&eval);
    for (tensor_name, shape, data) in &backend_tensors {
        let e = eval_tensors
            .iter()
            .find(|(n, _, _)| n == tensor_name)
            .unwrap_or_else(|| panic!("evaluator output missing `{tensor_name}`: {eval}"));
        assert_eq!(
            shape, &e.1,
            "{tensor_name}: eval-vs-backend shape disagreement"
        );
        assert_eq!(
            data.len(),
            e.2.len(),
            "{tensor_name}: eval-vs-backend length"
        );
        for (i, (b, ev)) in data.iter().zip(e.2.iter()).enumerate() {
            assert!(
                (b - ev).abs() < 1e-6,
                "{tensor_name}[{i}]: eval-vs-backend disagreement: backend {b} vs eval {ev}"
            );
        }
    }
}

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

/// ONE rank-poly named-reduce def, called at rank 2 and rank 3, builds and
/// runs and the backend reduces over the correct (named) `seq` axis at each
/// rank — the call-site monomorphization + named-axis lowering proof.
///
/// CRITICAL: the operands are deliberately NON-SQUARE (the reduced axis size
/// differs from every surviving axis size). A square operand masks an
/// axis-mislabel bug in call-site monomorphization (the surviving axis was
/// renamed to the reduced axis, which only aborts when the sizes differ —
/// chelis#258 red-team finding). Every distinct size here is load-bearing.
#[test]
fn named_reduce_builds_and_runs_nonsquare_at_ranks_2_3_4() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def avg_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = mean(x, seq)\n\
         def r2(x: &tensor[seq, hidden, f32]) -> tensor[hidden, f32] = reduce_seq(x)\n\
         def r3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = reduce_seq(x)\n\
         def r4(x: &tensor[batch, depth, seq, hidden, f32]) -> tensor[batch, depth, hidden, f32] = reduce_seq(x)\n\
         def m3(x: &tensor[batch, seq, hidden, f32]) -> tensor[batch, hidden, f32] = avg_seq(x)\n\
         out2 = r2(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         out3 = r3(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n\
         out4 = r4(to_tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]))\n\
         outm = m3(to_tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]))\n";
    let backend = build_compile_run(source, "rank_poly_reduce_nonsquare");
    let tensors = parse_printed_tensors(&backend);

    // out2: seq(=2) reduced from [seq=2, hidden=3] → [hidden=3] = col sums = [5, 7, 9].
    // out3: seq(=2) reduced from [batch=2, seq=2, hidden=3]:
    //   b0 [[1,2,3],[4,5,6]] → [5,7,9];  b1 [[7,8,9],[10,11,12]] → [17,19,21].
    // out4: seq(=2) reduced from [batch=1, depth=1, seq=2, hidden=3] → [1,1,3] = [5,7,9].
    // outm: mean over seq(=2) of out3's input → [2.5,3.5,4.5 ; 8.5,9.5,10.5].
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out2", &[3], &[5.0, 7.0, 9.0]),
        ("out3", &[2, 3], &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]),
        ("out4", &[1, 1, 3], &[5.0, 7.0, 9.0]),
        ("outm", &[2, 3], &[2.5, 3.5, 4.5, 8.5, 9.5, 10.5]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
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

    // Backend must agree with the evaluator oracle, value-for-value (#338).
    assert_eval_agrees_with_backend(source, "rank_poly_reduce_nonsquare", &backend);
}

/// The exact chelis#338 repro: a rank-poly named reduce called through a
/// concrete-rank caller evaluates under `chelis eval` and yields the same
/// numerics the C backend produces ([[1,2],[3,4]] summed over `seq` ->
/// [3, 7]). Before #338 this errored with "unknown runtime name `seq`".
#[test]
fn eval_resolves_named_axis_issue_repro() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = reduce_seq(x)\n\
         out = use2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n";
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "issue_338_repro");
    // A single-root program prints the bare value (no `out = ` prefix);
    // pin the exact line: shape AND the issue's expected numerics.
    assert_eq!(
        eval.trim(),
        "tensor(shape=[2], data=[3.0, 7.0])",
        "issue #338 repro: eval must yield the backend's numerics"
    );
}

/// Named-axis reductions at CONCRETE rank (no spread anywhere) share the same
/// eval gap and the same fix: the def call routes through the lowering lane,
/// which resolves `seq` against the declared (named) param dims. `max_reduce`,
/// `min_reduce`, and `prod_reduce` are included because at concrete rank they
/// are checkable and buildable (the chelis#340 Body-Discipline rejection
/// applies only inside `..r` bodies). `argmax_reduce`/`argmin_reduce` are
/// deliberately absent: the C backend mis-prints their int64 output (an
/// unrelated pre-existing backend bug; eval is correct), so the agreement
/// oracle cannot include them yet.
/// Operand is non-square (batch=2, seq=3) per the #258 red-team finding.
#[test]
fn concrete_rank_named_reduce_eval_matches_backend() {
    let source = "def sum_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         def max_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = max_reduce(x, seq)\n\
         def min_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = min_reduce(x, seq)\n\
         def prod_seq(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = prod_reduce(x, seq)\n\
         outs = sum_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outx = max_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outn = min_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n\
         outp = prod_seq(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";
    let backend = build_compile_run(source, "concrete_named_reduce");
    let tensors = parse_printed_tensors(&backend);
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("outs", &[2], &[6.0, 15.0]),
        ("outx", &[2], &[3.0, 6.0]),
        ("outn", &[2], &[1.0, 4.0]),
        ("outp", &[2], &[6.0, 120.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    assert_eval_agrees_with_backend(source, "concrete_named_reduce", &backend);
}

/// Eval-vs-backend parity on every host/tensor-lane boundary shape the
/// chelis#338 routing has to handle, in one program (one build + one eval):
///
///   - `lp`:   a def with an unmarshalable `List` param alongside the tensor
///             (the def-call boundary declines; the body's reduction routes
///             at the reduction site with the frame param's declared type)
///   - `tl`:   a named-axis reduction directly at a top-level root (operand
///             typed from the type-env entry of an earlier root)
///   - `blk`:  a block body whose reduction operand is a local `let` binding
///             (typed from the checker's annotation on the bound expr)
///   - `al`:   a local closure alias of a concrete wrapper (`g = use2`;
///             rank-poly callee routed with the argument's static type)
///   - `pp`:   a pipe whose first stage reduces (`x |> sum(seq)`) inside a def
///   - `tp`:   a root-level pipe chaining a bare Identity stage before the
///             reduction (`y |> relu |> sum(seq)`; the piped type threads
///             through shape-preserving stages)
///   - `outt`: a routed def declared to return a scalar (`-> f32`)
///   - `gr`:   `grad` over a scalar-output def whose body reduces by name
#[test]
fn named_axis_eval_parity_corners() {
    let source = "def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
         def use2(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = reduce_seq(x)\n\
         def lp(x: &tensor[batch, seq, f32], ys: List[f32]) -> tensor[batch, f32] = sum(x, seq)\n\
         def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         def blk(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = {\n\
           y = relu(x)\n\
           sum(y, seq)\n\
         }\n\
         def al(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = {\n\
           g = use2\n\
           g(x)\n\
         }\n\
         def pp(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = x |> sum(seq)\n\
         def total(x: &tensor[seq, f32]) -> f32 = tensor_to_scalar(sum(x, seq))\n\
         y = id2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_lp = lp(to_tensor([[1.0, 2.0], [3.0, 4.0]]), [1.0, 2.0])\n\
         out_tl = sum(y, seq)\n\
         out_blk = blk(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_al = al(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_pp = pp(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out_tp = y |> relu |> sum(seq)\n\
         outt = total(to_tensor([1.0, 2.0, 3.0]))\n\
         gr = grad(total)(to_tensor([1.0, 2.0, 3.0]))\n";
    let backend = build_compile_run(source, "named_axis_parity_corners");
    let tensors = parse_printed_tensors(&backend);
    let expected: &[(&str, &[usize], &[f64])] = &[
        ("out_lp", &[2], &[3.0, 7.0]),
        ("out_tl", &[2], &[3.0, 7.0]),
        ("out_blk", &[2], &[3.0, 7.0]),
        ("out_al", &[2], &[3.0, 7.0]),
        ("out_pp", &[2], &[3.0, 7.0]),
        ("out_tp", &[2], &[3.0, 7.0]),
        ("gr", &[3], &[1.0, 1.0, 1.0]),
    ];
    for (name, shape, data) in expected {
        let got = tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("backend output missing `{name}`: {backend}"));
        assert_eq!(&got.1, shape, "{name}: backend shape mismatch ({backend})");
        for (i, (g, e)) in got.2.iter().zip(data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "{name}[{i}]: backend {g} != expected {e} ({backend})"
            );
        }
    }
    // The scalar-return root prints as a plain scalar line, outside the
    // tensor parser: pin it on both lanes directly.
    assert!(
        backend.contains("outt = 6"),
        "backend must print the scalar return: {backend}"
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, "named_axis_parity_corners");
    assert!(
        eval.contains("outt = 6"),
        "eval must print the scalar return: {eval}"
    );
    assert_eval_agrees_with_backend(source, "named_axis_parity_corners", &backend);
}

/// Known residual gap, pinned: a *shape-rewriting* pipe stage (`permute`)
/// between the typed head and the named reduction drops the threaded type
/// (only Identity-class stages and def stages propagate it), so eval declines
/// with the targeted chelis#338 diagnostic while the backend builds and runs.
/// If this test starts failing because eval learned to handle it, delete the
/// decline assertion and fold the case into the parity corners above.
#[test]
fn pipe_rewriting_stage_then_named_reduce_is_a_pinned_gap() {
    let source = "def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         y = id2(to_tensor([[1.0, 2.0], [3.0, 4.0]]))\n\
         out = y |> permute(1, 0) |> sum(seq)\n";
    // Backend lane: green (permute carries the names through lowering).
    let backend = build_compile_run(source, "pipe_rewriting_gap");
    let tensors = parse_printed_tensors(&backend);
    let out = tensors
        .iter()
        .find(|(n, _, _)| n == "out")
        .unwrap_or_else(|| panic!("backend output missing `out`: {backend}"));
    assert_eq!(out.1, vec![2], "backend shape ({backend})");
    // permute flips to [seq, batch]; reducing `seq` (now axis 0) still
    // sums the same elements per batch: [1+2, 3+4].
    for (i, e) in [3.0, 7.0].iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "out[{i}]: backend {} != {e}",
            out.2[i]
        );
    }
    // Eval lane: targeted decline, never the bare `unknown runtime name`.
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "pipe_rewriting_gap");
    assert!(
        stderr.contains("chelis#338") && stderr.contains("statically known tensor type"),
        "expected the targeted named-axis decline diagnostic, got: {stderr}"
    );
    assert!(
        !stderr.contains("unknown runtime name"),
        "the pre-#338 error must not resurface: {stderr}"
    );
}
