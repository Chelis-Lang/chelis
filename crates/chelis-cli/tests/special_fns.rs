//! Special-function builtin acceptance (chelis#902).
//!
//! End-to-end contract for `erf` / `erfc` / `norm_cdf` / `norm_ppf` in
//! the eval lane: reference values (mpmath-computed, baked as f64
//! literals), the parity identity `Φ(x) + Φ(−x) = 1` that hand-rolled
//! polynomial CDFs measurably violate, quantile/CDF round-trip, tensor
//! elementwise support, loud domain errors, and the build-lane
//! eval-only rejection for both scalar and tensor uses.
//!
//! Data is task-neutral (plain mathematical constants) — nothing here
//! is copied from any benchmark.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn fmt_in_place(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

/// The full scalar + tensor surface in one pure-chelis program. Every
/// bound is checked in-language; the program writes a single marker
/// string so the test does not depend on print-channel float formatting.
///
/// Reference values: mpmath 1.3 at 50 significant digits, rounded to
/// nearest f64 (erf(1) also matches Abramowitz & Stegun Table 7.1 at
/// its printed precision).
///
/// Tensor lanes: an f64-tagged tensor keeps full f64 accuracy (~1e-16
/// here); an f32-tagged tensor rounds each element's f64 result through
/// f32 (~1e-8 here), per the special-function precision rule. The f64
/// row goes through `cast(..., f64)` because `to_tensor` on today's
/// main tags every float list f32 (orthogonal issue, fixed by
/// chelis#891's `list_to_tensor_data` dtype keying).
fn solve_source(output_path: &Path) -> String {
    let output = output_path.to_str().expect("utf8 path");
    format!(
        r#"erf_one = erf(1.0f64)
erf_ok = (abs(erf_one - 0.8427007929497149f64) < 0.000000000001f64)
erfc_two = erfc(2.0f64)
erfc_ok = (abs(erfc_two - 0.004677734981047266f64) < 0.000000000000001f64)
phi_neg_one = norm_cdf(0.0f64 - 1.0f64)
phi_ok = (abs(phi_neg_one - 0.15865525393145705f64) < 0.000000000001f64)
z_hi = norm_ppf(0.975f64)
z_ok = (abs(z_hi - 1.9599639845400543f64) < 0.000000000001f64)
parity_gap = abs(norm_cdf(1.7f64) + norm_cdf(0.0f64 - 1.7f64) - 1.0f64)
parity_ok = (parity_gap < 0.0000000000000003f64)
round_trip = norm_cdf(norm_ppf(0.001f64))
round_trip_ok = (abs(round_trip - 0.001f64) < 0.000000000000001f64)
xs = cast(to_tensor([0.5f64, 1.5f64]), f64)
erf_sum = tensor_to_scalar(sum(erf(xs), 0))
tensor_ok = (abs(erf_sum - 1.4866050242883574f64) < 0.000000000001f64)
xs32 = to_tensor([0.5f64, 1.5f64])
erf_sum32 = tensor_to_scalar(sum(erf(xs32), 0))
tensor32_ok = (abs(erf_sum32 - 1.4866050242883574f64) < 0.000001f64)
scalar_ok = and(erf_ok, and(erfc_ok, and(phi_ok, z_ok)))
identity_ok = and(parity_ok, round_trip_ok)
tensors_ok = and(tensor_ok, tensor32_ok)
all_ok = and(scalar_ok, and(identity_ok, tensors_ok))
status = if all_ok then "SPECIAL_FNS_OK" else "SPECIAL_FNS_MISMATCH"
done = write_file("{output}", status)
"#
    )
}

#[test]
fn special_fns_eval_end_to_end_scalar_and_tensor() {
    let dir = tempdir().expect("tempdir");
    let output_path = dir.path().join("output.txt");
    let solve_path = dir.path().join("solve.ch");
    write_file(&solve_path, &solve_source(&output_path));

    // Canonicalize through the real formatter, then run through the real
    // style gate (no CHELIS_STYLE_GATE_DISABLE): the gate passing is part
    // of the acceptance, exactly as an agent would run it.
    fmt_in_place(&solve_path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .success();
    let marker = fs::read_to_string(&output_path).expect("output written");
    assert_eq!(
        marker, "SPECIAL_FNS_OK",
        "an in-language accuracy bound failed (the program checks each \
         builtin against mpmath-derived references, the parity identity, \
         the quantile round-trip, and the tensor lane)"
    );
}

/// `norm_ppf` outside [0, 1] aborts the eval with a diagnostic naming
/// the builtin, the domain, and the offending value — the output file
/// is never written. (A silent NaN here is the "passes 34/35 checks"
/// failure class from the chelis#902 evidence.)
#[test]
fn norm_ppf_out_of_domain_fails_eval_loudly() {
    let dir = tempdir().expect("tempdir");
    let output_path = dir.path().join("output.txt");
    let solve_path = dir.path().join("solve.ch");
    let source = format!(
        r#"bad = norm_ppf(1.5f64)
done = write_file("{}", to_string(bad))
"#,
        output_path.to_str().unwrap(),
    );
    write_file(&solve_path, &source);
    fmt_in_place(&solve_path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("norm_ppf domain error"))
        .stderr(predicate::str::contains("[0, 1]"))
        .stderr(predicate::str::contains("1.5"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}

/// Same loud failure through the tensor lane, with the failing element's
/// index in the diagnostic.
#[test]
fn norm_ppf_tensor_element_out_of_domain_names_the_index() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    write_file(
        &solve_path,
        "ps = to_tensor([0.5f64, 2.5f64])\nzs = norm_ppf(ps)\nshown = print(to_string(tensor_to_scalar(sum(zs, 0))))\n",
    );
    fmt_in_place(&solve_path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("norm_ppf"))
        .stderr(predicate::str::contains("element 1"))
        .stderr(predicate::str::contains("2.5"));
}

/// The special-function builtins are eval-only: `chelis build` rejects
/// each of them whole-program with the eval-only diagnostic (the same
/// gate as `process_run`), instead of emitting a silently-wrong compiled
/// value. Scalar use.
#[test]
fn special_fns_are_rejected_by_build_scalar() {
    for builtin in ["erf", "erfc", "norm_cdf", "norm_ppf"] {
        let dir = tempdir().expect("tempdir");
        let solve_path = dir.path().join("solve.ch");
        let out_dir = dir.path().join("out");
        write_file(&solve_path, &format!("value = {builtin}(0.25f64)\n"));
        fmt_in_place(&solve_path);

        Command::cargo_bin("chelis")
            .expect("binary")
            .args([
                "build",
                solve_path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out_dir.to_str().unwrap(),
            ])
            .assert()
            .failure()
            .stderr(predicate::str::contains("eval/test-only builtin"))
            .stderr(predicate::str::contains(builtin));
    }
}

/// Tensor-typed use must also fail the build loudly (never a silent
/// fallthrough into C codegen), through whichever lane the entry takes.
#[test]
fn special_fns_are_rejected_by_build_tensor() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &solve_path,
        "xs = to_tensor([0.25f64, 0.75f64])\nys = erf(xs)\nshown = print(to_string(tensor_to_scalar(sum(ys, 0))))\n",
    );
    fmt_in_place(&solve_path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            solve_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("eval/test-only builtin"))
        .stderr(predicate::str::contains("erf"));
}

/// `grad`/`vmap` over a special function is rejected BY NAME in the eval
/// lane (the tensor_scan host-only gate), not with the cryptic
/// "lowering produced no roots" fallthrough. AD support is out of scope
/// for chelis#902; the derivatives are closed-form and the diagnostic
/// says so.
#[test]
fn grad_over_special_fn_is_rejected_by_name() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    write_file(
        &solve_path,
        "def phi_price(x: f64) -> f64 = norm_cdf(x)\ng = grad(phi_price)\nvalue = g(0.3f64)\nshown = print(to_string(value))\n",
    );
    fmt_in_place(&solve_path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cannot differentiate through host-runtime-only builtin `norm_cdf`",
        ))
        .stderr(predicate::str::contains("closed-form derivative"));
}

/// A `grad`-using def that reaches a special function still gets the
/// clean eval-only diagnostic from `chelis build` — the host-lane
/// forward-mode dual rules (chelis-ir `host.rs`) let lowering proceed
/// far enough for the whole-program eval-only gate to fire, instead of
/// dying earlier with an unnamed lowering error.
#[test]
fn build_of_grad_over_special_fn_hits_the_eval_only_gate() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &solve_path,
        "def phi_price(x: f64) -> f64 = norm_cdf(x)\ndef phi_delta(x: f64) -> f64 = grad(phi_price)(x)\nvalue = phi_delta(0.3f64)\n",
    );
    fmt_in_place(&solve_path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            solve_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("eval/test-only builtin"))
        .stderr(predicate::str::contains("norm_cdf"));
}

/// Type-contract negatives: a non-float argument is a check-time error
/// naming the builtin and the expected slot (never a runtime surprise),
/// and the arity is pinned at one. `chelis check` reports errors as JSON
/// on stdout and exits non-zero iff the errors array is non-empty
/// (issue #207).
#[test]
fn special_fns_reject_non_float_and_bad_arity_at_check() {
    let cases: &[(&str, &str)] = &[
        // String argument.
        ("value = erf(\"hello\")\n", "erf expects"),
        // Integer argument (unsuffixed int literal binds at int32).
        ("value = norm_cdf(1)\n", "norm_cdf expects"),
        // Arity: HM unification against the unop env scheme rejects the
        // extra argument before the concrete-contract arm runs, so the
        // diagnostic is the generic arity mismatch.
        (
            "value = erfc(0.25f64, 0.5f64)\n",
            "arity mismatch: expected 1 args, got 2",
        ),
    ];
    for (source, needle) in cases {
        let dir = tempdir().expect("tempdir");
        let solve_path = dir.path().join("solve.ch");
        write_file(&solve_path, source);

        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["check", solve_path.to_str().unwrap()])
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .assert()
            .failure()
            .stdout(predicate::str::contains(*needle));
    }
}
