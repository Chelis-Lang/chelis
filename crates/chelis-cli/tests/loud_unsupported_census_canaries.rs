//! chelis#730 Phase 0 - census canaries and liveness evidence locks for
//! the section C5 rows of `spec/design/loud_unsupported.md` that had no
//! committed named executable.
//!
//! Two kinds of test live here, per the census verification contract:
//!
//! - **Evidence locks** (green today): assert the CURRENT substituting
//!   behavior of a live row, the `metal_int64_abs_is_rejected_not_
//!   pre_planted_zero` pattern (its Phase 1 rejection form). Each is replaced by the row's conversion PR at
//!   Phase 1 - the lock failing later IS the signal that the row moved.
//! - **Canaries** (green today): drive the guard that keeps a dead row
//!   dead (section C1.4's prove-half). If a canary ever observes the
//!   placeholder instead of the guard, the row was live and the census
//!   was wrong - file, append, never silently fix (B2.5).
//!
//! Rows covered here: 4, 5 (also 18's chelis#698 half), 13, 14, 16.
//! Every other row is backed by a pre-existing named test; see the census
//! table's status column.
//!
//! The B2.5 protocol fired once during P0 itself: row 13's re-execution
//! found the bogus-cast guard is eval-lane only and the BUILD lane
//! substitutes silently - filed as chelis#744 and carried here as the
//! `#[ignore]`d red-by-design `dp_bogus_cast_target_must_not_build_silently`
//! next to the green eval-guard canary.

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

fn eval_first_line(program: &str, ext: &str) -> Result<String, String> {
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
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// `chelis build` to `target`; (ok, stderr, concatenated emitted files).
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

/// Build to C, link, run; `Ok((run_exit_ok, stdout, stderr))` or the build
/// failure's stderr.
fn c_run_outcome(program: &str, ext: &str, name: &str) -> Result<(bool, String, String), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}{ext}"));
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

/// Canonical Deep of a Surf program, via `chelis deep`.
fn deep_of(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .output()
        .expect("chelis deep should run");
    assert!(out.status.success(), "chelis deep must succeed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Replace the first balanced `(<head> ...)` node whose text contains
/// `needle` with `replacement`. Paren-balanced so metadata content never
/// breaks the surgery.
fn replace_first_balanced_node(dp: &str, head: &str, needle: &str, replacement: &str) -> String {
    let mut search_from = 0;
    while let Some(off) = dp[search_from..].find(head) {
        let start = search_from + off;
        let mut depth = 0usize;
        let mut end = None;
        for (i, ch) in dp[start..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.expect("balanced parens in canonical Deep");
        if dp[start..end].contains(needle) {
            return format!("{}{}{}", &dp[..start], replacement, &dp[end..]);
        }
        search_from = end;
    }
    panic!("no `{head}` node containing `{needle}` found in:\n{dp}");
}

/// `chelis check` score for a program written with the given extension.
fn check_score_and_output(program: &str, ext: &str) -> (f64, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("check must emit JSON: {e}; got: {text}"));
    (parsed["score"].as_f64().expect("numeric score"), text)
}

// ===========================================================================
// Census row 5 (chelis#689) + row 18's chelis#698 half - LIVE evidence lock.
// ===========================================================================

/// **Rejection lock (replaced the Phase 0 evidence lock at Phase 1):**
/// `chelis build --target hip` on an int64 `neg` is REJECTED with the
/// branded section C2 diagnostic - `elem_kind`'s former `_ =>
/// ElemKind::F32` wildcard is deleted (census row 5, chelis#689;
/// runtime-confirmed corrupt on gfx1151). The permissive CLI gate still
/// admits int64 (row 18's chelis#698 half, Phase 3 territory); the
/// EMITTER channel is what refuses now - the enforcement-ladder rung the
/// plan demands. Emission-only; no hipcc needed.
#[test]
fn hip_int64_neg_is_rejected_with_the_branded_diagnostic() {
    let (ok, stderr, emitted) = build_target(
        "def f(x: tensor[4, int64]) -> tensor[4, int64] = neg(x)\n",
        ".ch",
        "hip_i64_neg",
        "hip",
    );
    assert!(
        !ok,
        "census row 5 (chelis#689): the HIP emitter must reject an int64 \
         elementwise kernel, never emit the F32 fallback"
    );
    assert!(
        stderr.contains("unsupported:") && stderr.contains("int64"),
        "the rejection must be the branded section C2 diagnostic naming the \
         dtype; got: {stderr}"
    );
    assert!(
        !emitted.contains("kernel_neg_f32"),
        "no F32 fallback kernel may be left behind on a rejected build"
    );
}

// ===========================================================================
// Census row 4 (chelis#714 symptom) - LIVE evidence lock.
// ===========================================================================

/// **Rejection lock (replaced the Phase 0 evidence lock at Phase 1):**
/// the compiled f16 scalar `floor` program is REJECTED at build with the
/// branded diagnostic - the row 4 `<value>` print arms and the row 2
/// builtin stub are both errors now, so the chelis#714 Unknown chain
/// terminates loudly instead of printing a placeholder.
#[test]
fn c_f16_floor_is_rejected_not_value_placeholder() {
    let err = c_run_outcome(
        "def run() -> f16 = floor(cast(1.5, f16))\nout = run()\n",
        ".ch",
        "f16_floor_value",
    )
    .expect_err("census row 4: the f16 floor program must fail the build loudly");
    assert!(
        err.contains("unsupported:"),
        "the rejection must carry the branded section C2 diagnostic; got: {err}"
    );
}

// ===========================================================================
// Census row 13 (lower_cast's unwrap_or(Prim::F32)) - split at P0.
// The row was censused dead-by-probe; Phase 0 re-execution found the guard
// is EVAL-LANE ONLY and the build lane substitutes silently (chelis#744).
// ===========================================================================

/// The two cast-target spellings `lower_cast` accepts, each with a bogus
/// dtype. Hand-written Deep (the #710 fixture shape); `.dp` is the
/// documented first-class ingestion route.
const BOGUS_CAST_TPRIM: &str = "(def {}\n  out\n  (app {}\n    (var {} print)\n    (cast {}\n      \
     (lit {type: (t-prim {} f32)} 1.5)\n      (t-prim {} bogus_dtype))))\n";
const BOGUS_CAST_BARE: &str = "(def {}\n  out\n  (app {}\n    (var {} print)\n    (cast {}\n      \
     (lit {type: (t-prim {} f32)} 1.5)\n      bogus_dtype)))\n";

/// **Canary (green): a bogus cast target is rejected loudly, never computed
/// with an F32 fallback.** Both spellings are rejected before eval can matter
/// (the audit-backlog item 6 refutation, re-executed at P0). As of chelis#756
/// (chelis#731 Phase 2) the CHECK lane now guards this ahead of the eval-lane
/// guard: the deep-type converter no longer reduces an unknown/malformed cast
/// target to a silent `Type::Error`, so `infer_cast` reports "cast target
/// `<name>` is not a recognized primitive type" at check, and `eval_first_line`
/// (which runs check before eval) surfaces that rejection. The census intent
/// -- loud rejection, no silent F32 fallback -- is preserved and strengthened.
/// The build lane is the live half - see
/// `dp_bogus_cast_target_must_not_build_silently`.
///
/// lower.rs:9833's sibling default (input precision when a DAG node lookup
/// fails) has no user-facing driver at all (an internal desync is required),
/// so it stays dead-by-inspection; section C1.4 raise-or-prove retires it
/// at Phase 1 regardless.
#[test]
fn canary_dp_cast_bogus_dtype_is_guarded() {
    // chelis#756: both spellings now name the bogus target and are rejected at
    // check ("not a recognized primitive type"), ahead of the eval guard.
    for dp in [BOGUS_CAST_TPRIM, BOGUS_CAST_BARE] {
        let eval_err = eval_first_line(dp, ".dp")
            .expect_err("census row 13: a bogus cast target must be rejected, not computed");
        assert!(
            eval_err.contains("not a recognized primitive type")
                || eval_err.contains("cast missing target type"),
            "the guard must reject a bogus cast target loudly; got: {eval_err}"
        );
    }
}

/// Un-ignored by chelis#730 Phase 1 (census row 13, chelis#744):
/// `lower_cast` now raises a fatal branded error on a bogus target dtype
/// in both spellings, so the build lane rejects exactly like the eval
/// guard.
#[test]
fn dp_bogus_cast_target_must_not_build_silently() {
    for (i, dp) in [BOGUS_CAST_TPRIM, BOGUS_CAST_BARE].iter().enumerate() {
        let (ok, stderr, _emitted) = build_target(dp, ".dp", &format!("bogus_cast_{i}"), "c");
        assert!(
            !ok,
            "census row 13 / chelis#744: `chelis build` must reject a bogus \
             cast target, not silently lower it as f32 (fixture {i})"
        );
        assert!(
            stderr.contains("error:"),
            "the rejection must be a clean diagnostic; got: {stderr}"
        );
    }
}

// ===========================================================================
// Census row 14 (named_axis.rs pack_dag_roots precision default) - DEAD canary.
// ===========================================================================

/// **Canary:** `vmap` over an int64 tensor keeps integer precision through
/// `pack_dag_roots` (the transforms.rs caller of the row 14 site). The
/// F32 default at named_axis.rs:430 fires only when a root id is missing
/// from the DAG - an internal desync with no user-facing driver - so this
/// canary locks the reachable surface: root packing threads the real
/// precision and the values print as integers.
#[test]
fn canary_vmap_int64_roots_keep_integer_precision() {
    let got = eval_first_line(
        "module M.Main\n\
         def f(xs: tensor[3, 1, int64]) -> tensor[3, int64] = \
         vmap(fn (r: tensor[1, int64]) -> \
         add(tensor_to_scalar(sum(r, 0)), cast(1, int64)))(xs)\n\
         out = print(to_list(f(cast(to_tensor([[1.0], [2.0], [3.0]]), int64))))\n",
        ".ch",
    )
    .expect("census row 14: vmap over int64 must evaluate");
    assert_eq!(
        got, "[2, 3, 4]",
        "census row 14: int64 roots must pack with integer precision (no \
         F32/float rendering); a float-shaped result here means the \
         precision default fired on a reachable path - update the census \
         per B2.5, do not fix here"
    );
}

// ===========================================================================
// Census row 16 (~25 guarded Const 0.0 sites in lower.rs) - DEAD canaries.
// ===========================================================================

/// **Canary (guard 1 of 3):** an unknown Deep tag is rejected by the parser
/// before lowering ever runs - the 62-tag closed vocabulary that keeps the
/// unknown-tag `Const 0.0` arm dead (audit-backlog item 1 refutation,
/// re-executed).
#[test]
fn canary_unknown_deep_tag_is_rejected() {
    let dp = deep_of("module M.Main\nout = print(1.5)\n").replacen("(app", "(totally-bogus-tag", 1);
    assert!(
        dp.contains("(totally-bogus-tag"),
        "surgery must plant the bogus tag; got:\n{dp}"
    );
    let (score, output) = check_score_and_output(&dp, ".dp");
    assert!(
        score < 1.0,
        "census row 16: an unknown Deep tag must be rejected loudly; a clean \
         score means the parser guard moved - update the census. Output: {output}"
    );
}

/// **Canary (guard 2 of 3):** a bare keyword atom in expression position is
/// a clean parse error, not the `Const 0.0` placeholder (audit-backlog item 4
/// refutation, re-executed).
///
/// chelis#908 moved the first rejection to the earliest competent stage:
/// `:keyword` is metadata-key syntax and is not representable as a parsed Deep
/// expression. Both CLI rungs are asserted: the check score must fall below
/// 1.0 (chelis#710 form 4), and eval must refuse to produce a value with the
/// same parse diagnostic.
#[test]
fn canary_bare_keyword_atom_fails_cleanly() {
    let dp = deep_of("module M.Main\nout = print(1.5)\n");
    let dp = replace_first_balanced_node(&dp, "(lit", "1.5", ":oops");

    let (score, output) = check_score_and_output(&dp, ".dp");
    assert!(
        score < 1.0,
        "chelis#908 / census row 16: a bare keyword token must not score 1.0 at \
         check; a clean score means the parser guard moved. Output: {output}"
    );

    let err = eval_first_line(&dp, ".dp")
        .expect_err("census row 16: a bare keyword token must not evaluate to a value");
    assert!(
        err.contains("expected expression")
            && err.contains("bare :keyword is valid only as a metadata map key"),
        "the rejection must be the clean bare-keyword parse diagnostic; got: {err}"
    );
}

/// **Canary (guard 3 of 3):** `fail` with a runtime-computed message aborts
/// loudly with that message in both lanes; the zero placeholder behind it
/// never fires (audit-backlog item 8 refutation, re-executed).
#[test]
fn canary_dynamic_fail_aborts_loudly() {
    let program = "module M.Main\n\
         def boom(msg: string) -> tensor[2, f32] = fail(msg)\n\
         def run() -> tensor[2, f32] = boom(string_concat(\"dynamic \", \"message\"))\n\
         out = print(run())\n";
    let err =
        eval_first_line(program, ".ch").expect_err("census row 16: fail(dynamic) must abort eval");
    assert!(
        err.contains("dynamic message"),
        "eval must carry the dynamic message; got: {err}"
    );
    if c_toolchain_available() {
        let (ran_ok, stdout, stderr) = c_run_outcome(program, ".ch", "dyn_fail")
            .expect("fail(dynamic) must still BUILD; the abort is a runtime abort");
        assert!(
            !ran_ok,
            "the compiled binary must exit nonzero, not print a value; stdout: {stdout}"
        );
        assert!(
            stderr.contains("dynamic message"),
            "the compiled abort must carry the dynamic message; got: {stderr}"
        );
    }
}
