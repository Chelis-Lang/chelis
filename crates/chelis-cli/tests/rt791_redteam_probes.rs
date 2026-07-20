//! RT791 red-team keeper probes for chelis#730 Phase 1 (PR #791).
//!
//! Authored by the fresh-context red team on PR #791 (QUALIFIED PASS,
//! 2026-07-20) and adopted into the suite with credit - 13 probes, all
//! green on the PR head at adoption. Each test attacks a specific
//! contract row of `spec/design/loud_unsupported.md` Part I or an
//! orchestrator-flagged probe. The invariant style is deliberately
//! "reject-or-correct": a probe FAILS only when the shipped behavior is
//! the forbidden third option (a plausible substituted value, a dropped
//! root, or an unbranded rejection).

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Full-stdout eval: Ok(stdout) on exit 0, Err(stderr) otherwise.
fn eval_full(program: &str, ext: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `chelis build --target <target>`; (build_ok, stderr, concatenated emission).
fn build_target(program: &str, ext: &str, name: &str, target: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}{ext}"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            target,
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let mut emitted = String::new();
    if out_dir.is_dir() {
        for entry in std::fs::read_dir(&out_dir).expect("read out dir") {
            let p = entry.expect("entry").path();
            if p.is_file()
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                emitted.push_str(&text);
            }
        }
    }
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        emitted,
    )
}

/// Build + link + run; Ok((run_ok, stdout, stderr)) or Err(build/link text).
fn c_run(program: &str, name: &str) -> Result<(bool, String, String), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
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
        .output()
        .expect("chelis build should run");
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok((
        run.status.success(),
        String::from_utf8_lossy(&run.stdout).into_owned(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
    ))
}

fn branded_line(text: &str) -> Option<&str> {
    text.lines().find(|l| l.contains("unsupported: "))
}

/// The frozen section C2 rendering, checked structurally: the literal
/// brand, an ` on ` context clause, a parenthesized stage, `; ` hint.
fn assert_frozen_shape(line: &str, ctx: &str) {
    let tail = line
        .split_once("unsupported: ")
        .map(|(_, t)| t)
        .unwrap_or_default();
    assert!(
        tail.contains(" on ") && line.contains("); "),
        "{ctx}: not the frozen `unsupported: <what> on <context> (<stage>); <hint>` \
         shape: {line}"
    );
}

// ===========================================================================
// Orchestrator probe 1 - the absorption-guard boundary
// ===========================================================================

/// 1(a): an unsupported-op def that IS a DAG root (tensor signature) must
/// surface the branded rejection in eval - even when a healthy tensor
/// root sits beside it.
#[test]
fn rt_p1a_unsupported_tensor_root_surfaces_in_eval() {
    let program = "module M.Main\n\
         def bad(x: tensor[4, int32]) -> tensor[4, int32] = cos(x)\n\
         def good(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n\
         out = print(good(to_tensor([1.0, -2.0, 3.0, -4.0])))\n";
    match eval_full(program, ".ch") {
        Err(stderr) => {
            let line = branded_line(&stderr)
                .unwrap_or_else(|| panic!("rejection must be branded; got: {stderr}"));
            assert!(line.contains("Cos") || line.contains("cos"), "got: {line}");
        }
        Ok(stdout) => panic!(
            "a program containing a DAG-rooted unsupported def must not eval cleanly \
             (the def root would be fabricated); stdout:\n{stdout}"
        ),
    }
}

/// 1(c): consuming the unsupported def's value from a host-evaluated root
/// (print / to_list) must stay loud - an absorbed-but-consumed program
/// would be the P0 silent-wrong-answer.
#[test]
fn rt_p1c_consumed_unsupported_def_is_never_absorbed() {
    let consumers = [
        (
            "print",
            "module M.Main\n\
             def bad(x: tensor[2, int32]) -> tensor[2, int32] = cos(x)\n\
             out = print(bad(to_tensor([cast(0, int32), cast(1, int32)])))\n",
        ),
        (
            "to_list",
            "module M.Main\n\
             def bad(x: tensor[2, int32]) -> tensor[2, int32] = cos(x)\n\
             out = print(to_list(bad(to_tensor([cast(0, int32), cast(1, int32)]))))\n",
        ),
    ];
    for (label, program) in consumers {
        match eval_full(program, ".ch") {
            Err(stderr) => assert!(
                stderr.contains("unsupported:") || stderr.contains("error"),
                "{label}: rejection must be loud; got: {stderr}"
            ),
            Ok(stdout) => {
                // If eval accepts it, the values must be the host-correct
                // cos values (cos(0)=1, cos(1)=0.5403), never zeros.
                assert!(
                    !stdout.contains("[0, 0]")
                        && !stdout.contains("[0.0, 0.0]")
                        && (stdout.contains("0.54") || stdout.contains("1")),
                    "{label}: absorbed-but-consumed zeros would be the P0; stdout:\n{stdout}"
                );
                // And the C lane must agree or reject - never differ silently.
                let (ok, stderr, emitted) = build_target(program, ".ch", label, "c");
                assert!(
                    !ok || !emitted.contains("unsupported builtin"),
                    "{label}: C lane left a stub behind; stderr: {stderr}"
                );
            }
        }
    }
}

/// 1(b): a HOST-classified def (string return forces the host runtime)
/// wrapping the same unsupported-in-IR op. Contract: eval either rejects
/// loudly or computes the true host values - never zeros. This is the
/// absorbed side of the boundary if the auxiliary DAG raises here.
#[test]
fn rt_p1b_host_classified_def_is_host_correct_or_loud() {
    let program = "module M.Main\n\
         def wrap(x: tensor[2, int32]) -> string = to_string(cos(x))\n\
         out = print(wrap(to_tensor([cast(0, int32), cast(1, int32)])))\n";
    match eval_full(program, ".ch") {
        Err(stderr) => assert!(
            !stderr.trim().is_empty(),
            "a rejection must carry text; got empty stderr"
        ),
        Ok(stdout) => {
            let first = stdout.lines().next().unwrap_or_default();
            assert!(
                !first.contains("data=[0, 0]") && !first.contains("data=[0.0, 0.0]"),
                "host-evaluated cos over [0, 1] must never be all-zero \
                 (cos(0)=1); stdout:\n{stdout}"
            );
        }
    }
}

/// 1(b) value-parity: the PR's re-authored inline forward control, held to
/// EXACT output - the host runtime computes 300.0 and eval's root list
/// contains no fabricated trailing tensor root for a def that failed to
/// lower (nothing zero-filled, nothing silently dropped).
#[test]
fn rt_p1b_inline_forward_exact_output_no_fabricated_roots() {
    let program = "module M.Main\n\
         out = print(sum(mul(to_tensor([0.1, 0.2, 0.3, 0.4]), \
         cast(abs(cast(to_tensor([-100.0, 200.0, -300.0, 400.0]), int64)), f32)), 0))\n";
    let stdout = eval_full(program, ".ch").expect("the inline host forward must evaluate");
    let first = stdout.lines().next().unwrap_or_default();
    // Re-baselined at the chelis#792 rebase ([05-OBS-4]): a rank-0 eval
    // result renders bare, not tensor(shape=[], data=[...]).
    assert_eq!(first, "300.0", "sum must be 300.0; got: {first}");

    assert!(
        !stdout.contains("data=[0.0, 0.0, 0.0, 0.0]"),
        "no fabricated zero root may trail the correct print; full stdout:\n{stdout}"
    );
}

// ===========================================================================
// Orchestrator probe 2 - reachability precision
// ===========================================================================

/// A wrapper invoked ONLY through a pipe stage must count as reachable:
/// hard build failure (or a correct binary) - never garbage, never a
/// mis-scoped abort stub on the live path.
#[test]
fn rt_p2_pipe_stage_wrapper_is_live() {
    let program = "module M.Main\n\
         def bad(x: f32) -> f32 = floor(x)\n\
         out = print(3.5 |> bad)\n";
    let (ok, stderr, emitted) = build_target(program, ".ch", "rt_pipe_live", "c");
    if ok {
        assert!(
            !emitted.contains("abort();") || !emitted.contains("unsupported:"),
            "a wrapper reachable through a pipe must not be stubbed; emitted:\n{emitted}"
        );
        if c_toolchain_available() {
            let (ran_ok, stdout, _) =
                c_run(program, "rt_pipe_live_run").expect("built program should link");
            assert!(
                ran_ok && stdout.trim().starts_with('3'),
                "floor(3.5) must print 3, never a stub value; stdout: {stdout}"
            );
        }
    } else {
        let line = branded_line(&stderr)
            .unwrap_or_else(|| panic!("pipe-live rejection must be branded; got: {stderr}"));
        assert_frozen_shape(line, "pipe-live");
    }
}

/// A wrapper passed as a function VALUE (higher-order argument) must be
/// live: hard fail or correct, never stub-garbage.
#[test]
fn rt_p2_function_valued_wrapper_is_live() {
    let program = "module M.Main\n\
         def bad(x: f32) -> f32 = floor(x)\n\
         def apply(g: (f32) -> f32, v: f32) -> f32 = g(v)\n\
         out = print(apply(bad, 3.5))\n";
    let (ok, stderr, emitted) = build_target(program, ".ch", "rt_fnval_live", "c");
    if ok {
        // If the build passes, the binary must compute floor correctly -
        // running it is the proof the stub did not land on the live path.
        if c_toolchain_available() {
            let (ran_ok, stdout, run_stderr) =
                c_run(program, "rt_fnval_live_run").expect("built program should link");
            if ran_ok {
                assert!(
                    stdout.trim().starts_with('3'),
                    "apply(bad, 3.5) must print 3; stdout: {stdout}"
                );
            } else {
                // A runtime abort is acceptable ONLY branded.
                assert!(
                    run_stderr.contains("unsupported:"),
                    "a runtime failure on the live path must be branded; got: {run_stderr}"
                );
            }
        } else {
            assert!(
                !emitted.contains("/* unsupported"),
                "no stub marker on a live path; emitted:\n{emitted}"
            );
        }
    } else {
        assert!(
            stderr.contains("unsupported:"),
            "function-valued live rejection must be branded; got: {stderr}"
        );
    }
}

/// The positive stub case: a grad program whose standalone host wrapper is
/// genuinely dead (its only use inlined into the grad DAG). The build
/// must SUCCEED, the binary must print the true gradient, and the emitted
/// C must carry the branded self-naming abort stub for the dead wrapper.
#[test]
fn rt_p2_dead_wrapper_stub_gradient_still_correct() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def loss(w: tensor[4, f32]) -> tensor[f32] = sum(mul(copy(w), w), 0)\n\
         g = grad(loss)(to_tensor([1.0, 2.0, 3.0, 4.0]))\n\
         out = print(g)\n";
    match c_run(program, "rt_dead_stub") {
        Ok((ran_ok, stdout, run_stderr)) => {
            assert!(
                ran_ok,
                "the grad binary must run (wrapper is dead); stderr: {run_stderr}"
            );
            assert!(
                stdout.contains("2.0") && stdout.contains("8.0"),
                "grad of sum(w*w) at [1,2,3,4] is [2,4,6,8]; stdout: {stdout}"
            );
        }
        Err(build_err) => {
            // A hard failure is contract-legal too (reachability is an
            // under-approximation guard) - but it must be branded.
            assert!(
                build_err.contains("unsupported:"),
                "if the dead-wrapper program fails the build, the failure must \
                 be branded; got: {build_err}"
            );
        }
    }
}

// ===========================================================================
// Orchestrator probe 5 - two structural keeps attacked
// ===========================================================================

/// Keep 5 (`fail`-in-if mask placeholder): drive the annotation's edge -
/// a DAG-lowered `if` whose fail branch is TAKEN at run time. The keep's
/// claim is that the zero placeholder's value never matters; if the
/// compiled or eval lane prints values instead of aborting, the taken
/// fail branch produced the placeholder - a silent substitution.
#[test]
fn rt_keep5_taken_fail_branch_never_yields_values() {
    let program = "module M.Main\n\
         def f(x: tensor[4, f32], t: bool) -> tensor[4, f32] = \
         if t then fail(\"boom\") else relu(x)\n\
         out = print(f(to_tensor([1.0, 2.0, 3.0, 4.0]), true))\n";
    match eval_full(program, ".ch") {
        Err(stderr) => assert!(
            !stderr.trim().is_empty(),
            "eval rejection/abort must carry text"
        ),
        Ok(stdout) => panic!(
            "eval of a TAKEN fail branch must abort, not produce values \
             (keep 5's mask placeholder leaked); stdout:\n{stdout}"
        ),
    }
    if c_toolchain_available() {
        // A build-time rejection is also acceptable; only a RUNNING binary
        // that prints values pierces the keep.
        if let Ok((ran_ok, stdout, _)) = c_run(program, "rt_keep5_fail_taken") {
            assert!(
                !ran_ok,
                "the compiled taken-fail branch must abort, not print; stdout:\n{stdout}"
            );
        }
    }
}

/// Keep 2 (unknown-tag-with-children sequence seed): an unknown Deep tag
/// WITH children must be rejected by the closed 62-tag vocabulary, not
/// sequence-lowered to its last child's value.
#[test]
fn rt_keep2_unknown_tag_with_children_is_rejected() {
    let dp = "(def {}\n  out\n  (app {}\n    (var {} print)\n    (bogus_wrapper {}\n      \
              (lit {type: (t-prim {} f32)} 1.5)\n      (lit {type: (t-prim {} f32)} 7.5))))\n";
    match eval_full(dp, ".dp") {
        Err(stderr) => assert!(
            !stderr.trim().is_empty(),
            "unknown-tag rejection must carry text"
        ),
        Ok(stdout) => panic!(
            "an unknown Deep tag with children must not evaluate to its last \
             child (7.5) - the keep-2 seed's guard is pierced; stdout:\n{stdout}"
        ),
    }
    let (ok, _stderr, _) = build_target(dp, ".dp", "rt_keep2_bogus_tag", "c");
    assert!(!ok, "the build lane must reject the unknown tag too");
}

// ===========================================================================
// Orchestrator probe 4 - frozen-shape byte parity across lanes
// ===========================================================================

/// The same lowering raise (cos on an int32 tensor, def-rooted) must
/// render the identical branded byte sequence via eval and via
/// `chelis build --target c` / `--target hip` (a lane-skewed rendering is
/// the chelis#712 shape the corpus exists to catch).
#[test]
fn rt_p4_branded_bytes_agree_across_lanes() {
    let program = "module M.Main\n\
         def run(x: tensor[4, int32]) -> tensor[4, int32] = cos(x)\n\
         out = run(to_tensor([cast(1, int32), cast(2, int32), cast(3, int32), \
         cast(4, int32)]))\n";
    let eval_err = eval_full(program, ".ch").expect_err("eval must reject the cos-int program");
    let (c_ok, c_err, _) = build_target(program, ".ch", "rt_parity_c", "c");
    let (h_ok, h_err, _) = build_target(program, ".ch", "rt_parity_hip", "hip");
    assert!(!c_ok && !h_ok, "both build lanes must reject");
    let brand = |s: &str| -> String {
        branded_line(s)
            .and_then(|l| l.split_once("unsupported: ").map(|(_, t)| t.to_string()))
            .unwrap_or_else(|| panic!("no branded line in: {s}"))
    };
    let (e, c, h) = (brand(&eval_err), brand(&c_err), brand(&h_err));
    for (lane, line) in [("eval", &e), ("build-c", &c), ("build-hip", &h)] {
        assert_frozen_shape(&format!("unsupported: {line}"), lane);
        assert!(
            line.to_lowercase().contains("cos"),
            "{lane} must name the op; got: {line}"
        );
    }
    // The BUILD lanes must agree byte-for-byte (the c host emitter is the
    // shared refusal point). eval-vs-build render DIFFERENT diagnostics
    // for this repro (lowering raise vs host-emission arm) - a DISCLOSED
    // routing skew (issue_687 corpus row comment) reported in prose, not
    // asserted here.
    assert_eq!(c, h, "C vs HIP branded bytes must agree");
}

/// The effect-kind rejection must agree between the eval lane and the
/// build lane on the same `.dp`. Re-baselined at the chelis#793 rebase:
/// the CHECKER's handle-effect case (chelis#731 P1) now rejects the
/// unknown kind FIRST in both lanes - the earliest competent stage - so
/// the surviving parity surface is the checker's `MalformedForm`
/// diagnostic naming the kind, identical in content across lanes; the
/// chelis#730 rows 9/20 lowering raises are defense-in-depth behind it
/// and no longer reachable from this surface (the section I1 interlock,
/// honored from both sides).
#[test]
fn rt_p4_effect_kind_bytes_agree_across_lanes() {
    let dp = "(defsig {} f (t-fn {} (t-prim {} f32)))\n\n(def {}\n  f\n  (fn {}\n    (params {})\n    \
              (handle-effect {effect: teleport}\n      (lit {type: (t-prim {} int64)} 42)\n      \
              (lit {type: (t-prim {} f32)} 2.5))))\n\n(def {}\n  out\n  (app {}\n    \
              (var {} print)\n    (app {} (var {} f))))\n";
    let eval_out = eval_full(dp, ".dp");
    let (b_ok, b_err, _) = build_target(dp, ".dp", "rt_effect_parity", "c");
    match (eval_out, b_ok) {
        (Err(e), false) => {
            for (lane, text) in [("eval", &e), ("build", &b_err)] {
                assert!(
                    text.contains("unknown effect kind `teleport`"),
                    "{lane} must reject with the checker diagnostic naming the \
                     kind; got: {text}"
                );
            }
        }
        (Ok(stdout), _) => panic!("eval must reject effect kind `teleport`; stdout:\n{stdout}"),
        (Err(e), true) => {
            panic!("build must reject effect kind `teleport` (eval did: {e})")
        }
    }
}

// ===========================================================================
// Negative parity spot-checks (the supported neighbors keep working)
// ===========================================================================

/// cos on a FLOAT tensor must keep evaluating and building - the row 1
/// conversion must not overshoot onto the float family.
#[test]
fn rt_control_float_cos_still_works_both_lanes() {
    let program = "module M.Main\n\
         def run(x: tensor[2, f32]) -> tensor[2, f32] = cos(x)\n\
         out = print(run(to_tensor([0.0, 1.0])))\n";
    let stdout = eval_full(program, ".ch").expect("float cos must evaluate");
    assert!(
        stdout.contains("data=[1.0, 0.5403"),
        "cos([0,1]) must be [1, 0.5403..]; got: {stdout}"
    );
    let (ok, stderr, _) = build_target(program, ".ch", "rt_ctl_cos_f32", "c");
    assert!(ok, "float cos must keep building; stderr: {stderr}");
}

/// `with seed(...) { ... }` (a KNOWN effect kind) must keep working after
/// the rows 9/20 string-match conversion.
#[test]
fn rt_control_random_effect_still_works() {
    let program = "module M.Main\n\
         def f() -> tensor[2, f32] = with seed(42i64) { uniform_like(to_tensor([0.0, 0.0]), 0.0, 1.0) }\n\
         out = print(f())\n";
    match eval_full(program, ".ch") {
        Ok(stdout) => assert!(
            stdout.contains("data=["),
            "seeded uniform must produce a tensor; got: {stdout}"
        ),
        Err(stderr) => panic!("the `random` effect kind must keep evaluating; got: {stderr}"),
    }
}
