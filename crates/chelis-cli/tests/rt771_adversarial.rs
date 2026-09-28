//! RED-TEAM adversarial probes for chelis#771 (PR #777).
//!
//! Reuses the EXACT `build_and_run` two-lane harness the shipped oracle
//! (`issue_771_seed_width_parity.rs`) uses, but parameterizes the seed
//! *expression* to characterize the peel's edge cases.
//!
//! ## What the probes pin
//!
//! The red team of PR #777 pinned the retired `with seed(...)` handler's
//! literal-seed gate. Under explicit keys (chelis#2413) a seed reaches the
//! draw through `key_from_seed(seed: i64)`, which takes any `i64` value,
//! literal or computed ([05-OP-69]). A seed expression whose dtype is not
//! `i64` (an unsuffixed negative literal, a float, i32 arithmetic) fails
//! LOUDLY and IDENTICALLY in both lanes at the checker, never as a silent
//! divergence; an `i64` seed, literal or computed, reaches the key at full
//! width in both lanes, and the lanes agree with `common::key_ref`.
//!
//! Run with `--no-capture` to see the LANE-OUTCOME lines.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, parse_tensor_data, write_file};

/// 8-element f32 [0,1) uniform keyed by `key_from_seed` of an arbitrary seed
/// EXPRESSION string.
fn seeded_uniform_expr(seed_expr: &str) -> String {
    let zeros = ["cast(0.0, f32)"; 8].join(", ");
    format!(
        "template = to_tensor([{zeros}])\n\
         sampled = uniform_like(key_from_seed({seed_expr}), copy(template), 0.0, 1.0)\n"
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

/// A seed whose dtype is not `i64` must be REJECTED loudly and IDENTICALLY in
/// both lanes, by the checker's diagnostic containing `diagnostic`.
fn assert_both_reject(tag: &str, seed_expr: &str, diagnostic: &str) {
    let (e, c) = outcome(tag, seed_expr);
    match (e, c) {
        (Lane::Err(em), Lane::Err(_)) => assert!(
            em.contains(diagnostic),
            "{seed_expr}: eval rejected but not with `{diagnostic}`: {em}"
        ),
        (e, c) => panic!("{seed_expr}: expected BOTH-ERR, got eval={e:?} c={c:?}"),
    }
}

/// An `i64` seed must AGREE bit-for-bit across lanes, and with the
/// `common::key_ref` draw of `key_from_seed(seed)`.
fn assert_both_agree(tag: &str, seed_expr: &str, seed: i64) {
    let (e, c) = outcome(tag, seed_expr);
    match (e, c) {
        (Lane::Ok(a), Lane::Ok(b)) => {
            assert!(
                bit_equal(&a, &b),
                "{seed_expr}: lanes must be f32-bit-equal\n eval={a:?}\n c   ={b:?}"
            );
            let reference =
                common::key_ref::uniform_f32(common::key_ref::key_from_seed(seed), 8, 0.0, 1.0)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
            assert!(
                bit_equal(&a, &reference),
                "{seed_expr}: eval must draw key_from_seed({seed})\n eval={a:?}\n ref ={reference:?}"
            );
        }
        (e, c) => panic!("{seed_expr}: expected BOTH-OK agreeing, got eval={e:?} c={c:?}"),
    }
}

#[test]
fn rt_2a_negative_seed_rejected_both_lanes() {
    // Unsuffixed negative seeds remain rejected: their dtype is not i64.
    assert_both_reject(
        "2a",
        "-1",
        "key operation expects i64 or a tensor of i64, got i32",
    );
    assert_both_reject("2a", "-2147483649", "out of range for default i32");
}

#[test]
fn rt_2c_float_seed_rejected_both_lanes() {
    // 1.5 is an f32 literal, not an i64 seed.
    assert_both_reject(
        "2c",
        "1.5",
        "key operation expects i64 or a tensor of i64, got f32",
    );
}

#[test]
fn rt_2d_cast_wrapped_seed_reaches_the_key_at_full_width_in_both_lanes() {
    // The recorded residual warned that a cast-wrapped seed could narrow to
    // i32 (`-1`); the key must be that of 4294967295 in both lanes.
    assert_both_agree("2d", "cast(4294967295, i64)", 4_294_967_295);
}

#[test]
fn rt_2e_i32_computed_seed_rejected_and_i64_computed_seed_agrees() {
    assert_both_reject(
        "2e",
        "add(2147483647, 1)",
        "key operation expects i64 or a tensor of i64, got i32",
    );
    assert_both_agree("2e", "add(2147483647i64, 1i64)", 2_147_483_648);
}

#[test]
fn rt_4_literal_boundaries_agree() {
    // i64-suffixed literals across the i32/i64 boundary are lane-identical
    // and draw their own key; an unsuffixed literal is an i32, a type error.
    for (s, seed) in [
        ("2147483646i64", 2_147_483_646),
        ("2147483647i64", 2_147_483_647),
        ("2147483648i64", 2_147_483_648),
        ("2147483649i64", 2_147_483_649),
        ("4294967296i64", 4_294_967_296),
    ] {
        assert_both_agree("4", s, seed);
    }
}

/// 2b: an i64-*suffixed* seed literal `Ni64` desugars to
/// `(lit {type: (t-prim {} i64)} N)` and reaches the key at full width in
/// both lanes.
#[test]
fn rt_2b_suffixed_int64_seed_agrees() {
    assert_both_agree("2b", "4294967295i64", 4_294_967_295);
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
