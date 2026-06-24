//! WI-13 envelope certifier: the Arb trust-anchor half of the generator.
//!
//! Reads a coefficients-only DRAFT envelope (produced by
//! `scripts/generate_erf_envelope.py`, `eps` fields null), certifies each box's
//! sup-norm error rigorously with the WI-14 Arb oracle by whole-box ball
//! arithmetic, stamps the certified `eps` and provenance, and writes the
//! committed envelope JSON.
//!
//! This binary requires the `arb` feature (it links FLINT/Arb). It runs offline
//! and in the regen pipeline only — it is not part of any deployed build.
//!
//! Usage:
//!   cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
//!     -- <draft.json> <out.json>
//!
//! The certified `eps` is `sup_{x in box} |arm.approx(x) - erf(x)|` bounded by
//! [`chelis_prove::certify_sup_norm_over_box`] (subdivided ball arithmetic), so
//! a poor proposed polynomial only yields a larger, still-sound `eps`.

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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: certify_erf_envelope <draft.json> <out.json>");
        return ExitCode::from(2);
    }
    let draft_path = &args[1];
    let out_path = &args[2];

    let draft_src = match std::fs::read_to_string(draft_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read draft {draft_path}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut env: ErfEnvelope = match serde_json::from_str(&draft_src) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: draft is not a valid envelope: {e}");
            return ExitCode::from(2);
        }
    };

    let subdivisions = subdivisions();
    // Certify each box's eps with Arb (whole-box, subdivided).
    for b in &mut env.boxes {
        let coeffs: Vec<f64> = match &b.arm {
            ErfArm::Saturation { value } => vec![*value],
            ErfArm::Central { coeffs } => coeffs.clone(),
        };
        let eps = certify_sup_norm_over_box(b.lo, b.hi, &coeffs, subdivisions, CERTIFY_PREC);
        eprintln!(
            "certified box [{}, {}] ({}): eps = {:.6e}",
            b.lo,
            b.hi,
            match &b.arm {
                ErfArm::Saturation { .. } => "saturation",
                ErfArm::Central { .. } => "central",
            },
            eps
        );
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
    eprintln!("wrote committed envelope to {out_path}");
    ExitCode::SUCCESS
}
