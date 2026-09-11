//! RED-TEAM adversarial probes for chelis#731 Phase 1 (PR #793).
//!
//! Authored by the fresh-context red team (qualified-pass round) and adopted
//! into the PR suite. Each test asserts a CONTRACT, so a failure is a confirmed
//! finding, not a broken test. The findings these locked (par-bound catch-all,
//! negative `.dp` seed, handle-effect extra child) are fixed checker-side in
//! this PR; the two lowering-side halves (`extract_f64_value`'s over-broad
//! catch-all and `extract_usize_value`'s negative-seed fallback) are filed as a
//! separate `.dp`-reachable #703-class issue routed to chelis#730's census.
//!
//! Probe map:
//! * `par_bound_*`: the `is_static_numeric_bound` catch-all arm used to mirror
//!   `extract_f64_value`'s "any list with a numeric atom at element 2", but the
//!   EVAL lane interprets the expression. `(par {} 2.0 3.0)` evaluates to 3.0
//!   (last child, spec/03 §2.3) while the fold reads element 2 = 2.0, so the
//!   checker must reject it (Phase 1 narrowed the arm to `lit`-tagged lists).
//! * `double_neg_bound_parity`: nested-wrapper control; both accept-set
//!   definitions fold neg(neg(3.0)) to 3.0, so it stays accepted.
//! * `boundary_seed_cross_function_parity`: chelis#771 seed width through the
//!   handler-scope threading path (seed cannot be baked at the op site), at
//!   i32::MAX + 1.
//! * `negative_dp_seed_parity`: a `.dp`-only reachable negative int64 seed;
//!   the checker now rejects it (Phase 1 F2) rather than letting both lanes
//!   silently fall back to seed 0.
//! * `handle_effect_extra_child`: spec/03 gives `handle-effect` the shape
//!   `(handle-effect {effect: name} arg body)` — exactly two children. A third
//!   child is structurally malformed ([04-TOT-3]); Phase 1 rejects `!= 2`.
//! * `baked_seed_deterministic`: PR #793 moved the cli.rs expectation to a
//!   baked `CHELIS_EFFECTIVE_UNIFORM_SEED(7ULL)`; the generated C and its
//!   runtime output must be run-to-run deterministic and eval-bit-identical.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::path::Path;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{
    gcc_available as c_toolchain_available, link_generated, parse_tensor_data, write_file,
};

const PAR_BOUND_DP: &str = r#"(def {} template (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0.0))
(def {} sampled
  (handle-effect {effect: random}
    (lit {type: (t-prim {} int64)} 42)
    (app {} (var {} uniform_like)
      (copy {} (var {} template))
      (par {} 2.0 3.0)
      (lit {type: (t-prim {} f32)} 5.0))))
"#;

const DOUBLE_NEG_DP: &str = r#"(def {} template (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0.0))
(def {} sampled
  (handle-effect {effect: random}
    (lit {type: (t-prim {} int64)} 42)
    (app {} (var {} uniform_like)
      (copy {} (var {} template))
      (app {} (var {} neg) (app {} (var {} neg) (lit {type: (t-prim {} f32)} 3.0)))
      (lit {type: (t-prim {} f32)} 5.0))))
"#;

const NEGATIVE_SEED_DP: &str = r#"(def {} template (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0.0))
(def {} sampled
  (handle-effect {effect: random}
    (lit {type: (t-prim {} int64)} -1)
    (app {} (var {} uniform_like)
      (copy {} (var {} template))
      (lit {type: (t-prim {} f32)} 0.0)
      (lit {type: (t-prim {} f32)} 1.0))))
"#;

const EXTRA_CHILD_DP: &str = r#"(def {} out
  (handle-effect {effect: random}
    (lit {type: (t-prim {} int64)} 42)
    (lit {type: (t-prim {} f32)} 2.5)
    (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} int64)} 2))))
"#;

const BOUNDARY_CROSS_FN_CH: &str = "template = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])\n\
def sample(t: tensor[4, f32]) -> tensor[4, f32] ! { Random } =\n\
  uniform_like(copy(t), 0.0, 1.0)\n\
sampled = with seed(2147483648i64) { sample(copy(template)) }\n";

const BAKED_SEED_CH: &str = "template = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])\n\
sampled = with seed(7i64) { uniform_like(copy(template), cast(0.0, f32), cast(1.0, f32)) }\n";

/// `chelis check` JSON score for a source file.
fn check_score(source: &str, ext: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, source);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("check must emit JSON");
    parsed["score"].as_f64().expect("numeric score")
}

/// `chelis eval --file` stdout, asserting success.
fn eval_stdout(source: &str, ext: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, source);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "chelis eval failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Build to C into `out_dir` (asserting success) and return the generated
/// `<name>.c` source.
fn build_c(source: &str, ext: &str, name: &str, out_dir: &Path) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}{ext}"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    std::fs::read_to_string(out_dir.join(format!("{name}.c"))).expect("generated C source")
}

/// Build, link, run; return the printed `sampled` tensor data.
fn c_sampled(source: &str, ext: &str, name: &str) -> Vec<f64> {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join(format!("{name}-out"));
    build_c(source, ext, name, &out_dir);
    let status = link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link of generated C failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary exited non-zero: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    parse_tensor_data(&String::from_utf8_lossy(&run.stdout), "sampled")
}

fn f32_bits(v: f64) -> u32 {
    (v as f32).to_bits()
}

fn assert_f32_bit_equal(label: &str, eval: &[f64], c: &[f64]) {
    assert_eq!(eval.len(), c.len(), "{label}: length mismatch");
    for (i, (e, cc)) in eval.iter().zip(c).enumerate() {
        assert_eq!(
            f32_bits(*e),
            f32_bits(*cc),
            "{label}: element {i} differs in f32 bits (eval {e:?} vs C {cc:?})\n \
             eval={eval:?}\n c   ={c:?}"
        );
    }
}

/// Extract the two `chelis_f32_from_bits(0x...u)` constants baked at the
/// uniform-sample call site of a generated `.dp` C kernel.
fn baked_bound_bits(c_src: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut rest = c_src;
    while let Some(idx) = rest.find("chelis_f32_from_bits(0x") {
        let hex = &rest[idx + "chelis_f32_from_bits(0x".len()..];
        let end = hex.find('u').expect("bits literal ends with u");
        out.push(u32::from_str_radix(&hex[..end], 16).expect("hex bits"));
        rest = &hex[end..];
    }
    out
}

/// The P0-shape probe: `(par {} 2.0 3.0)` as a `uniform_like` low bound.
/// spec/03-deep-syntax.md §2.3: `par` evaluates its children in order and
/// returns the LAST child's value, so the bound's value is 3.0. Contract
/// assertion: either the checker rejects the non-literal bound (Phase 1 F1
/// narrowed the accept-set to `lit`-tagged lists), or every lane uses 3.0. A
/// baked low of 2.0f (the fold of element 2, the FIRST child) is a
/// checker-blessed silent value substitution (the #703 class).
#[test]
fn redteam_par_bound_folds_par_value_or_rejects() {
    let score = check_score(PAR_BOUND_DP, ".dp");
    if score < 1.0 {
        // The checker rejected the pathological bound: consistent, no finding.
        return;
    }
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("parbound-out");
    let c_src = build_c(PAR_BOUND_DP, ".dp", "parbound", &out_dir);
    let bits = baked_bound_bits(&c_src);
    assert!(
        bits.contains(&3.0f32.to_bits()),
        "checker accepted the par bound, so the baked low must be par's value \
         3.0 (spec/03 §2.3: last child); baked f32 bit patterns: \
         {bits:08x?} (2.0f = {:08x})",
        2.0f32.to_bits()
    );
    // Eval lane must also honor par semantics: all samples in [3,5).
    let eval = parse_tensor_data(&eval_stdout(PAR_BOUND_DP, ".dp"), "sampled");
    for (i, v) in eval.iter().enumerate() {
        assert!(
            (3.0..5.0).contains(v),
            "eval elem[{i}] = {v} must lie in [3,5) per par semantics"
        );
    }
}

/// Nested-wrapper control: neg(neg(3.0)) folds to 3.0 in both accept sets;
/// the program must check clean, the baked C low must be 3.0f, and the eval
/// samples must lie in [3,5).
#[test]
fn redteam_double_neg_bound_parity() {
    let score = check_score(DOUBLE_NEG_DP, ".dp");
    assert!(
        (score - 1.0).abs() < 1e-9,
        "double-neg bound must be accepted (both fold sets resolve it), got {score}"
    );
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("dblneg-out");
    let c_src = build_c(DOUBLE_NEG_DP, ".dp", "dblneg", &out_dir);
    let bits = baked_bound_bits(&c_src);
    assert!(
        bits.contains(&3.0f32.to_bits()),
        "baked low must be neg(neg(3.0)) = 3.0f; baked bits: {bits:08x?}"
    );
    let eval = parse_tensor_data(&eval_stdout(DOUBLE_NEG_DP, ".dp"), "sampled");
    for (i, v) in eval.iter().enumerate() {
        assert!(
            (3.0..5.0).contains(v),
            "eval elem[{i}] = {v} must lie in [3,5)"
        );
    }
}

/// chelis#771 boundary seed through the handler-scope threading path: the
/// cross-function form cannot bake the seed at the random-op site, so the
/// runtime `chelis_rng_current` scope must carry the full-width value.
#[test]
fn redteam_boundary_seed_cross_function_parity() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let eval = parse_tensor_data(&eval_stdout(BOUNDARY_CROSS_FN_CH, ".ch"), "sampled");
    let c1 = c_sampled(BOUNDARY_CROSS_FN_CH, ".ch", "bigseed");
    let c2 = c_sampled(BOUNDARY_CROSS_FN_CH, ".ch", "bigseed2");
    assert_f32_bit_equal("boundary cross-fn eval-vs-C", &eval, &c1);
    assert_f32_bit_equal("boundary cross-fn run-to-run", &c1, &c2);
}

/// A `.dp`-only reachable negative int64 seed: the checker's suffix rule used to
/// accept `(lit {type: int64} -1)` (score 1) while the DAG lowering's
/// `extract_usize_value` rejected negatives and silently fell back, so both
/// lanes ran the DEFAULT stream: seed -1 and seed 0 produced identical output
/// ("distinct seeds yield distinct streams" [05-RNG-1] fails). chelis#794
/// replaced that fold with `extract_u64_value`, which honors [05-RNG-1]'s
/// two's-complement reinterpretation. chelis#1803 admits that signed literal
/// through the checker; rejection must no longer silently skip this oracle.
#[test]
fn redteam_negative_dp_seed_not_silently_dropped() {
    let score = check_score(NEGATIVE_SEED_DP, ".dp");
    assert_eq!(score, 1.0, "signed int64 literal must be admitted");
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("negseed-out");
    let c_src = build_c(NEGATIVE_SEED_DP, ".dp", "negseed", &out_dir);
    assert!(
        !c_src.contains("CHELIS_EFFECTIVE_UNIFORM_SEED(0ULL"),
        "checker-accepted seed -1 must not silently bake the seed-0/default \
         stream into C; generated:\n{}",
        c_src
            .lines()
            .filter(|l| l.contains("EFFECTIVE_UNIFORM_SEED"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    let eval_neg = parse_tensor_data(&eval_stdout(NEGATIVE_SEED_DP, ".dp"), "sampled");
    let zero_src = NEGATIVE_SEED_DP.replace("} -1)", "} 0)");
    let eval_zero = parse_tensor_data(&eval_stdout(&zero_src, ".dp"), "sampled");
    assert_ne!(
        eval_neg, eval_zero,
        "[05-RNG-1] distinct seeds must yield distinct streams: eval(seed -1) \
         equals eval(seed 0), so the negative seed was silently dropped"
    );
}

/// spec/03: `(handle-effect {effect: name} arg body)` — exactly two children.
/// [04-TOT-3]: a structurally malformed form SHALL be rejected. The third
/// child here is an ill-typed subtree (`add(f32, int64)`) that
/// `infer_handle_effect` never visits; Phase 1 rejects `handle-effect` with
/// `!= 2` children, so this must score below 1.0.
#[test]
fn redteam_handle_effect_extra_child_is_rejected() {
    let score = check_score(EXTRA_CHILD_DP, ".dp");
    assert!(
        score < 1.0,
        "[04-TOT-3] / fitness honesty: a handle-effect with a third (ill-typed!) \
         child must not score 1.0, got {score}"
    );
}

/// The direct suffixed seed must produce identical generated C
/// across two builds and the runtime output bit-identical across two runs and
/// against eval.
#[test]
fn redteam_baked_seed_deterministic_and_cross_lane() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let dir = tempdir().expect("tempdir");
    let out_a = dir.path().join("a");
    let out_b = dir.path().join("b");
    let src_a = build_c(BAKED_SEED_CH, ".ch", "seeded", &out_a);
    let src_b = build_c(BAKED_SEED_CH, ".ch", "seeded", &out_b);
    assert_eq!(src_a, src_b, "generated C must be build-to-build identical");
    // The handler transports seed 7 through the invocation-local RNG state
    // (chelis#1799); a former macro spelling is not the execution contract.
    let eval = parse_tensor_data(&eval_stdout(BAKED_SEED_CH, ".ch"), "sampled");
    let c1 = c_sampled(BAKED_SEED_CH, ".ch", "seeded1");
    let c2 = c_sampled(BAKED_SEED_CH, ".ch", "seeded2");
    assert_f32_bit_equal("baked seed eval-vs-C", &eval, &c1);
    assert_f32_bit_equal("baked seed run-to-run", &c1, &c2);
}
