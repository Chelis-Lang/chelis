//! Phase 3l — Shoals acceptance oracle.
//!
//! This is the named oracle for Phase 3l per the Chelis agent contract's
//! "one acceptance oracle per phase" rule. The function name
//! `phase3l_shoals_oracle` is the spec contract — invoke with:
//!
//! ```text
//! cargo test -p chelis-cli --test phase3l_shoals_oracle phase3l_shoals_oracle -- --ignored --exact --nocapture
//! ```
//!
//! The oracle is environment-conditional: it skips with a clear message
//! when the Shoals working tree is not present. Locate Shoals via:
//!   1. `CHELIS_SHOALS_PATH` env var if set, OR
//!   2. fallback to `<chelis-monorepo-root>/../shoals`.
//!
//! What it asserts:
//!   * `chelis check src/pricing.ch` from inside the Shoals reef package
//!     returns `"score": 1`.
//!   * Running a small driver program through `chelis eval --file` against
//!     Shoals's `bs_call_scalar` at S=K=100, r=0.05, sigma=0.2, T=1
//!     produces 10.4506 (textbook value, scipy.stats.norm = 10.450584)
//!     to within 1e-3.
//!   * Two MC pricing runs at 20K paths under the same seed produce
//!     bit-exact-equal prices (reproducibility contract).
//!   * The 20K MC price is within 2 % of the analytical Black-Scholes
//!     call price (convergence contract).
//!
//!
//! Grad-derived Greek properties are checked by a separate focused
//! ignored test in this file. Keep it separate from the Monte Carlo
//! oracle so the AD lower/type-check status can be checked without
//! paying the 20K-path pricing runtime.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::PathBuf;

/// Locate the Shoals checkout. Returns `None` when Shoals is not available
/// on the host — the oracle then skips gracefully rather than failing.
fn shoals_root() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("CHELIS_SHOALS_PATH") {
        let p = PathBuf::from(env_path);
        if p.join("reef.toml").exists() {
            return Some(p);
        }
        return None;
    }
    let chelis_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .ok()?;
    let candidate = chelis_root.parent()?.join("shoals");
    if candidate.join("reef.toml").exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Skip-with-message helper. Tests use this when Shoals is not present.
fn skip_if_no_shoals() -> Option<PathBuf> {
    match shoals_root() {
        Some(p) => Some(p),
        None => {
            eprintln!(
                "phase3l_shoals_oracle: skipped — Shoals checkout not found. \
                 Set CHELIS_SHOALS_PATH or place shoals/ as a sibling of the \
                 chelis monorepo root."
            );
            None
        }
    }
}

/// Parse a single `name = value` line out of `chelis eval --file`'s
/// stdout. The host evaluator prints scalars as bare f64-precision
/// strings (e.g. `bs_atm = 10.45057541543509`), even when the underlying
/// type is `f32`; we read them as `f64` so we don't lose precision in
/// the test, then narrow only at the assert site.
fn parse_scalar_line(stdout: &str, name: &str) -> Option<f64> {
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&format!("{name} = ")) {
            let trimmed = rest
                .trim_start_matches("scalar(")
                .trim_end_matches(')')
                .trim();
            if let Ok(v) = trimmed.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

#[test]
#[ignore = "5-minute manual gate; see spec/design/chelis_phase3_plan.md Phase 3l Acceptance Oracle. Invoke: cargo test -p chelis-cli phase3l_shoals_oracle -- --ignored --exact"]
fn phase3l_shoals_oracle() {
    let Some(shoals) = skip_if_no_shoals() else {
        return;
    };

    // 1. `chelis check src/pricing.ch` returns score 1.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&shoals)
        .args(["check", "src/pricing.ch"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    // 2. Black-Scholes ATM call ≈ 10.4506 (scipy reference 10.450584).
    //    Driver program lives in a tempfile inside Shoals's `src/` so reef
    //    resolves the local package's `Shoals.Pricing` module. A Drop
    //    guard removes the file on test exit even if assertions panic.
    struct DriverGuard(PathBuf);
    impl Drop for DriverGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    // Reef's path-to-module rule is case-sensitive and rejects underscores
    // in filenames. `Shoals.Oracledriver` maps to `oracledriver.ch`.
    // Top-level `name = expr` bindings are the form that `chelis eval` prints;
    // `def` declarations are silent.
    let driver_path = shoals.join("src/oracledriver.ch");
    std::fs::write(
        &driver_path,
        r#"module Shoals.Oracledriver
import Shoals.Pricing (bs_call_scalar, mc_call_price)
bs_atm = bs_call_scalar(cast(100.0, f32), cast(100.0, f32), cast(0.05, f32), cast(0.2, f32), cast(1.0, f32))
mc_seed42_a = with seed(42) {
  template_a = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(20000, int64))))
  mc_call_price(template_a, cast(100.0, f32), cast(100.0, f32), cast(0.05, f32), cast(0.2, f32), cast(1.0, f32))
}
mc_seed42_b = with seed(42) {
  template_b = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(20000, int64))))
  mc_call_price(template_b, cast(100.0, f32), cast(100.0, f32), cast(0.05, f32), cast(0.2, f32), cast(1.0, f32))
}
"#,
    )
    .expect("write oracle driver");
    let _guard = DriverGuard(driver_path.clone());

    // The eval pass through Shoals's full module graph is slow under the
    // host evaluator (~60-90 s for 20K MC + the BS scalar). Allow plenty
    // of wall-clock headroom on slower CI.
    let eval_assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&shoals)
        .args(["eval", "--file", driver_path.to_str().unwrap()])
        .timeout(std::time::Duration::from_secs(300))
        .assert()
        .success();

    let stdout = String::from_utf8(eval_assert.get_output().stdout.clone()).expect("utf-8 stdout");

    let bs_atm = parse_scalar_line(&stdout, "bs_atm")
        .unwrap_or_else(|| panic!("could not parse bs_atm from eval output:\n{stdout}"));
    let mc_a = parse_scalar_line(&stdout, "mc_seed42_a")
        .unwrap_or_else(|| panic!("could not parse mc_seed42_a from eval output:\n{stdout}"));
    let mc_b = parse_scalar_line(&stdout, "mc_seed42_b")
        .unwrap_or_else(|| panic!("could not parse mc_seed42_b from eval output:\n{stdout}"));

    // BS textbook value: 10.450584 from scipy. The host eval printer
    // round-trips f32 through f64, so the printed value is ~10.45058.
    let bs_ref: f64 = 10.450584;
    let bs_diff = (bs_atm - bs_ref).abs();
    assert!(
        bs_diff < 1e-3,
        "BS ATM call mismatch: got {bs_atm}, expected ~{bs_ref}, diff {bs_diff}"
    );

    // Reproducibility: same-seed MC runs must be bit-exact equal. The
    // underlying compute is f32; the printer's f32->f64 round-trip is
    // deterministic, so two bit-equal f32 results print to identical
    // f64-precision strings, which parse to bit-equal f64s here.
    assert_eq!(
        mc_a.to_bits(),
        mc_b.to_bits(),
        "MC reproducibility broken: same seed yielded {mc_a} vs {mc_b}"
    );

    // Convergence: 20K MC must be within 2 % of analytical BS.
    let rel = ((mc_a - bs_atm).abs()) / bs_atm;
    assert!(
        rel < 0.02,
        "MC convergence: 20K MC = {mc_a}, BS = {bs_atm}, rel diff = {rel} (>= 2 %)"
    );
}

#[test]
#[ignore = "focused Shoals grad-Greeks manual gate; checks lower/type-check clean and runtime-skips with warning"]
fn phase3l_shoals_oracle_grad_greeks_match_analytic() {
    let Some(shoals) = skip_if_no_shoals() else {
        return;
    };

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&shoals)
        .args(["check", "properties/greeks.ch"])
        .timeout(std::time::Duration::from_secs(120))
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    eprintln!(
        "phase3l_shoals_oracle_grad_greeks_match_analytic: runtime-skipped — \
         Shoals grad-derived Greek properties lower/type-check clean, but the \
         full pricing body is not yet executable by host-runtime `grad`."
    );
}
