//! RED-TEAM adversarial probes for chelis#771 (PR #777).
//!
//! Reuses the EXACT `build_and_run` two-lane harness the shipped oracle
//! (`issue_771_seed_width_parity.rs`) uses, but parameterizes the seed
//! *expression* to characterize the peel's edge cases.
//!
//! ## What the red team established
//!
//! The `with seed(...)` random handler is gated in the SHARED front-end
//! (`chelis-effects` `validate_handler_expr`): a `random` handler whose seed
//! is not an int LITERAL (`int_literal`: a bare `Atom::Int` or a single
//! `(lit … Atom::Int)`) is REJECTED with
//! `with seed(...) currently requires an int literal seed` in BOTH lanes
//! (check/eval AND build all run the gate). So every "computed" seed
//! (negation `-1`, `cast(..)`, `add(..)`, a float) fails LOUDLY and
//! IDENTICALLY across lanes — never a silent divergence. The eval
//! `eval_expr` fallback the PR keeps is therefore dead code on every checked
//! path (the gate's accept-set is a subset of the peel's, and the peel
//! returns the same value on it). These probes pin exactly that.
//!
//! Run with `--no-capture` to see the LANE-OUTCOME lines.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, parse_tensor_data, write_file};

/// 8-element f32 [0,1) uniform seeded by an arbitrary seed EXPRESSION string.
fn seeded_uniform_expr(seed_expr: &str) -> String {
    let zeros = ["cast(0.0, f32)"; 8].join(", ");
    format!(
        "template = to_tensor([{zeros}])\n\
         sampled = with seed({seed_expr}) {{ uniform_like(copy(template), 0.0, 1.0) }}\n"
    )
}

#[derive(Debug)]
enum Lane {
    Ok(Vec<f64>),
    Err(String),
}

fn eval_lane(seed_expr: &str) -> Lane {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("probe.ch");
    write_file(&path, &seeded_uniform_expr(seed_expr));
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run");
    if out.status.success() {
        Lane::Ok(parse_tensor_data(
            &String::from_utf8_lossy(&out.stdout),
            "sampled",
        ))
    } else {
        Lane::Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// C lane via the shared build_and_run, tolerant of build/link failure
/// (build_and_run panics on failure, so we catch it and record the message).
fn c_lane(seed_expr: &str) -> Lane {
    let src = seeded_uniform_expr(seed_expr);
    match std::panic::catch_unwind(|| build_and_run(&src, "probe")) {
        Ok(stdout) => Lane::Ok(parse_tensor_data(&stdout, "sampled")),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            Lane::Err(msg)
        }
    }
}

fn f32_bits(x: f64) -> u32 {
    (x as f32).to_bits()
}

fn bit_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| f32_bits(*x) == f32_bits(*y))
}

fn outcome(tag: &str, seed_expr: &str) -> (Lane, Lane) {
    let e = eval_lane(seed_expr);
    let c = c_lane(seed_expr);
    let verdict = match (&e, &c) {
        (Lane::Ok(a), Lane::Ok(b)) if bit_equal(a, b) => "AGREE (f32-bit-equal)".to_string(),
        (Lane::Ok(a), Lane::Ok(b)) => {
            format!("DIVERGE eval[0]={:?} C[0]={:?}", a.first(), b.first())
        }
        (Lane::Err(_), Lane::Err(_)) => "BOTH-ERR (loud, agreeing)".to_string(),
        (Lane::Ok(_), Lane::Err(_)) => "EVAL-OK / C-ERR".to_string(),
        (Lane::Err(_), Lane::Ok(_)) => "EVAL-ERR / C-OK".to_string(),
    };
    println!("[RT771 {tag}] seed({seed_expr}) => {verdict}");
    if let Lane::Err(m) = &e {
        println!("    eval-err: {}", m.lines().next().unwrap_or(""));
    }
    (e, c)
}

/// A "computed" seed must be REJECTED loudly and IDENTICALLY in both lanes —
/// the shared effects gate makes this the only reachable outcome, so there is
/// no silent divergence to hide behind the recorded residual.
fn assert_both_reject(tag: &str, seed_expr: &str) {
    let (e, c) = outcome(tag, seed_expr);
    match (e, c) {
        (Lane::Err(em), Lane::Err(_)) => assert!(
            em.contains("requires an int literal seed"),
            "{seed_expr}: eval rejected but not via the seed-literal gate: {em}"
        ),
        (e, c) => panic!("{seed_expr}: expected BOTH-ERR, got eval={e:?} c={c:?}"),
    }
}

/// A literal seed must AGREE bit-for-bit across lanes post-fix.
fn assert_both_agree(tag: &str, seed_expr: &str) {
    let (e, c) = outcome(tag, seed_expr);
    match (e, c) {
        (Lane::Ok(a), Lane::Ok(b)) => assert!(
            bit_equal(&a, &b),
            "{seed_expr}: lanes must be f32-bit-equal\n eval={a:?}\n c   ={b:?}"
        ),
        (e, c) => panic!("{seed_expr}: expected BOTH-OK agreeing, got eval={e:?} c={c:?}"),
    }
}

#[test]
fn rt_2a_negative_seed_rejected_both_lanes() {
    // `-1` and `-2147483649` are `neg(...)` apps, not int literals: rejected.
    assert_both_reject("2a", "-1");
    assert_both_reject("2a", "-2147483649");
}

#[test]
fn rt_2c_float_seed_rejected_both_lanes() {
    // 1.5 is Atom::Float: peel returns None, gate rejects before eval fallback.
    assert_both_reject("2c", "1.5");
}

#[test]
fn rt_2d_cast_wrapped_seed_rejected_both_lanes() {
    // The recorded residual calls this "computed"; the gate rejects it, so the
    // narrowing it warns about is NOT reachable through the checked pipeline.
    assert_both_reject("2d", "cast(4294967295, int64)");
}

#[test]
fn rt_2e_add_computed_seed_rejected_both_lanes() {
    assert_both_reject("2e", "add(2147483647, 1)");
}

#[test]
fn rt_4_literal_boundaries_agree() {
    // int64-suffixed literals across the int32/int64 boundary pass the gate and
    // must be lane-identical. chelis#731 Phase 1 makes the seed's int64 suffix a
    // checker requirement (an unsuffixed literal is now a type error), so the
    // boundary seeds carry the `i64` suffix; the suffix is meta-only and does
    // not change the seed's raw-atom value the peel reads, so lane parity holds.
    for s in [
        "2147483646i64",
        "2147483647i64",
        "2147483648i64",
        "2147483649i64",
        "4294967296i64",
    ] {
        assert_both_agree("4", s);
    }
}

/// 2b: an int64-*suffixed* seed literal `Ni64` desugars to
/// `(lit {type: (t-prim {} int64)} N)`. The peel is meta-agnostic (it reads
/// the raw atom whether the meta says int32 or int64), so a suffixed seed
/// reads full width and agrees with C — the design-intended §C1.5 form.
#[test]
fn rt_2b_suffixed_int64_seed_agrees() {
    assert_both_agree("2b", "4294967295i64");
}

/// Oracle robustness (protocol item 3): `parse_tensor_data` must FAIL LOUDLY,
/// never silently return an empty/garbage stream that could launder a
/// divergence into a false pass.
#[test]
#[should_panic(expected = "does not contain")]
fn rt_3_oracle_panics_on_missing_line() {
    // No `sampled = tensor(` line at all.
    let _ = parse_tensor_data("some other output\nnope = 1\n", "sampled");
}

#[test]
#[should_panic(expected = "numeric")]
fn rt_3_oracle_panics_on_unparseable_data() {
    // Line present but a data token is not a number.
    let _ = parse_tensor_data(
        "sampled = tensor(shape=[2], data=[0.5, NOTNUM])\n",
        "sampled",
    );
}
