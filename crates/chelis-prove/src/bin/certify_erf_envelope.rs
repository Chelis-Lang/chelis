//! WI-13 envelope Arb certifier / cross-check.
//!
//! Two modes, both backed by the WI-14 Arb oracle (whole-box ball arithmetic):
//!
//! - `stamp <draft.json> <out.json>`: reads a draft envelope, Arb-certifies each
//!   box's sup-norm error, stamps `eps` + provenance, and writes the result.
//!   Used to fill the saturation-tail `eps` of the committed envelope.
//!
//! - `validate <envelope.json>`: the independent every-build CROSS-CHECK. Reads
//!   the committed envelope and asserts, for EVERY box (Gappa-proved central and
//!   Arb-enclosure tails alike), that the committed `eps` is `>=` the freshly
//!   Arb-certified sup-norm error -- i.e. the committed bound still soundly
//!   bounds `|approx - erf|`. This is the belt-and-suspenders guard the Option-B
//!   design keeps from Option A: the Gappa proof is the central arm's proof
//!   term, and Arb re-validates the actual numeric bound of every box on every
//!   CI run. Exits non-zero if any committed `eps` is below the Arb bound.
//!
//! This binary requires the `arb` feature (it links FLINT/Arb). It runs offline
//! and in CI only -- it is not part of any deployed build.

use chelis_prove::{DEFAULT_PREC, ErfArm, ErfEnvelope, certify_sup_norm_over_box};
use std::process::ExitCode;

/// Arb working precision and per-box subdivision count for certification.
/// Recorded into the committed provenance so CI re-validation reproduces the
/// exact bound. Subdivision beats the ball-arithmetic dependency problem (see
/// `arb_oracle::certify_sup_norm_over_box`); the central box needs a fine split,
/// the saturation tails are flat so they need very little.
const CERTIFY_PREC: i64 = DEFAULT_PREC;
// Subdivision count per box. The central box's difference-centered mean-value
// bound converges to the true degree-21 fit error (~6.5e-7) by ~5e5 sub-boxes;
// the flat saturation tails are tight at any count. ~9s offline at this count.
const CERTIFY_SUBDIVISIONS: usize = 524288;

/// Read an optional subdivision override from `CERTIFY_SUBDIVISIONS` (used to
/// probe convergence during generation); falls back to the committed constant.
fn subdivisions() -> usize {
    std::env::var("CERTIFY_SUBDIVISIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(CERTIFY_SUBDIVISIONS)
}

fn arm_coeffs(b: &chelis_prove::ErfEnvelopeBox) -> Vec<f64> {
    match &b.arm {
        ErfArm::Saturation { value } => vec![*value],
        ErfArm::Central { coeffs } => coeffs.clone(),
    }
}

fn read_env(path: &str) -> Result<ErfEnvelope, ExitCode> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        eprintln!("error: cannot read {path}: {e}");
        ExitCode::from(2)
    })?;
    serde_json::from_str(&src).map_err(|e| {
        eprintln!("error: {path} is not a valid envelope: {e}");
        ExitCode::from(2)
    })
}

/// `stamp`: Arb-certify every box's eps and write the result.
fn stamp(draft_path: &str, out_path: &str) -> ExitCode {
    let mut env = match read_env(draft_path) {
        Ok(e) => e,
        Err(c) => return c,
    };
    let subdivisions = subdivisions();
    for b in &mut env.boxes {
        let coeffs = arm_coeffs(b);
        let eps = certify_sup_norm_over_box(b.lo, b.hi, &coeffs, subdivisions, CERTIFY_PREC);
        eprintln!("certified box [{}, {}]: eps = {:.6e}", b.lo, b.hi, eps);
        b.eps = eps;
    }
    env.provenance.certify_prec = CERTIFY_PREC;
    env.provenance.certify_subdivisions = subdivisions;
    env.provenance.method =
        "Arb whole-box ball arithmetic (sup |approx - erf| over each subdivided box)".to_string();
    if !env.is_well_formed() {
        eprintln!("error: certified envelope is not well formed (gap, eps<0, or bad saturation)");
        return ExitCode::from(1);
    }
    let out = match serde_json::to_string_pretty(&env) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot serialize envelope: {e}");
            return ExitCode::from(1);
        }
    };
    if let Err(e) = std::fs::write(out_path, format!("{out}\n")) {
        eprintln!("error: cannot write {out_path}: {e}");
        return ExitCode::from(1);
    }
    eprintln!("wrote envelope to {out_path}");
    ExitCode::SUCCESS
}

/// `validate`: the independent every-build cross-check. Assert every committed
/// box's eps is `>=` the freshly Arb-certified sup-norm error.
fn validate(env_path: &str) -> ExitCode {
    let env = match read_env(env_path) {
        Ok(e) => e,
        Err(c) => return c,
    };
    let subdivisions = env.provenance.certify_subdivisions.max(1);
    let mut ok = true;
    for b in &env.boxes {
        let coeffs = arm_coeffs(b);
        let arb_eps = certify_sup_norm_over_box(b.lo, b.hi, &coeffs, subdivisions, CERTIFY_PREC);
        let sound = b.eps >= arb_eps;
        eprintln!(
            "box [{}, {}] ({:?}): committed eps {:.6e} {} Arb eps {:.6e}",
            b.lo,
            b.hi,
            b.proof_kind,
            b.eps,
            if sound { ">=" } else { "<  !! UNSOUND" },
            arb_eps
        );
        if !sound {
            ok = false;
        }
    }
    if !ok {
        eprintln!(
            "error: a committed eps is below the Arb-certified bound -- the envelope is unsound."
        );
        return ExitCode::from(1);
    }
    eprintln!("Arb cross-check passed: every committed eps bounds the Arb sup-norm error.");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("stamp") if args.len() == 4 => stamp(&args[2], &args[3]),
        Some("validate") if args.len() == 3 => validate(&args[2]),
        _ => {
            eprintln!(
                "usage:\n  certify_erf_envelope stamp <draft.json> <out.json>\n  \
                 certify_erf_envelope validate <envelope.json>"
            );
            ExitCode::from(2)
        }
    }
}
