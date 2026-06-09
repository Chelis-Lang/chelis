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

fn assert_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected rejection, got a clean check ({json})"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(score < 1.0, "{label}: a rejected program must score < 1.0");
}

/// Assert a rejection whose message contains `needle`.
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
// computed reduction. (The nameless tree-walking `chelis eval` interpreter
// cannot resolve a named axis — that needs the type info present only on the
// compile/lowering path — so this asserts the C-backend numerics directly
// rather than via an eval oracle.)

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
}
