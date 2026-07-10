//! chelis#434 special-function envelope Arb certifier (generalizes
//! `certify_erf_envelope` to `{erf, exp, log, sqrt}`).
//!
//! Two modes, both backed by the WI-14 Arb oracle
//! ([`chelis_prove::certify_sup_norm_over_box_general`], the function-general
//! naive whole-box ball certifier):
//!
//! - `stamp <draft.json> <out.json>`: reads a draft [`SpecialFnEnvelope`]
//!   (function name + boxes + polynomial coeffs, `eps` unset), Arb-certifies each
//!   box's sup-norm error against the box's function, stamps `eps` +
//!   `proof_kind = arb_enclosure`, and writes the committed envelope. The
//!   certified `eps` is a rigorous sound bound of `|approx - f|`, so the emitted
//!   envelope is sound regardless of the proposer's fit quality (a poor fit only
//!   enlarges `eps`).
//! - `validate <envelope.json>`: independent cross-check — re-certifies every box
//!   and asserts the committed `eps` is `>=` the fresh Arb bound (else the
//!   envelope is unsound). The `arb` CI-lane guard.
//!
//! Requires the `arb` feature (links FLINT/Arb). Offline / CI only.

use chelis_prove::{
    DEFAULT_PREC, EnvelopeArm, ProofKind, SpecialFn, SpecialFnEnvelope, SpecialFnProvenance,
    certify_sup_norm_over_box_general, certify_sup_norm_over_box_meanvalue,
};
use std::process::ExitCode;

const CERTIFY_PREC: i64 = DEFAULT_PREC;
/// Subdivision count. The tight mean-value certifier converges fast, so this is
/// modest; the naive cross-check reuses it. Overridable via `CERTIFY_SUBDIVISIONS`.
const CERTIFY_SUBDIVISIONS: usize = 65_536;

/// Certify one box: the TIGHT mean-value `eps`, cross-checked to never exceed the
/// sound naive whole-box bound (both bound the true error; mean-value is tighter,
/// so `mv <= naive` must hold — a violation means a certifier bug, so refuse).
/// Returns the mean-value eps, or `None` if either bound is non-finite (out of
/// domain / derivative singularity) or the cross-check fails.
fn certify_box(f: SpecialFn, lo: f64, hi: f64, coeffs: &[f64], subdivisions: usize) -> Option<f64> {
    let mv = certify_sup_norm_over_box_meanvalue(f, lo, hi, coeffs, subdivisions, CERTIFY_PREC);
    let naive = certify_sup_norm_over_box_general(f, lo, hi, coeffs, subdivisions, CERTIFY_PREC);
    if !mv.is_finite() || !naive.is_finite() {
        eprintln!("  box [{lo}, {hi}]: non-finite bound (mv={mv:.3e} naive={naive:.3e})");
        return None;
    }
    if mv > naive {
        eprintln!("  box [{lo}, {hi}]: CROSS-CHECK FAIL mean-value {mv:.3e} > naive {naive:.3e}");
        return None;
    }
    Some(mv)
}

fn subdivisions() -> usize {
    std::env::var("CERTIFY_SUBDIVISIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(CERTIFY_SUBDIVISIONS)
}

fn arm_coeffs(arm: &EnvelopeArm) -> Vec<f64> {
    match arm {
        EnvelopeArm::Saturation { value } => vec![*value],
        EnvelopeArm::Central { coeffs } => coeffs.clone(),
    }
}

fn read_env(path: &str) -> Result<SpecialFnEnvelope, ExitCode> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        eprintln!("error: cannot read {path}: {e}");
        ExitCode::from(2)
    })?;
    serde_json::from_str(&src).map_err(|e| {
        eprintln!("error: {path} is not a valid special-fn envelope: {e}");
        ExitCode::from(2)
    })
}

fn special_fn(env: &SpecialFnEnvelope) -> Result<SpecialFn, ExitCode> {
    SpecialFn::from_name(&env.fn_name).ok_or_else(|| {
        eprintln!(
            "error: '{}' is not an Arb-certifiable special function",
            env.fn_name
        );
        ExitCode::from(2)
    })
}

fn stamp(draft_path: &str, out_path: &str) -> ExitCode {
    let mut env = match read_env(draft_path) {
        Ok(e) => e,
        Err(c) => return c,
    };
    let f = match special_fn(&env) {
        Ok(f) => f,
        Err(c) => return c,
    };
    let subdivisions = subdivisions();
    for b in &mut env.boxes {
        let coeffs = arm_coeffs(&b.arm);
        let Some(eps) = certify_box(f, b.lo, b.hi, &coeffs, subdivisions) else {
            eprintln!(
                "error: box [{}, {}] failed certification (non-finite / cross-check) -- refusing",
                b.lo, b.hi
            );
            return ExitCode::from(1);
        };
        eprintln!(
            "certified {} box [{}, {}]: eps = {:.6e} (mean-value, <= naive)",
            env.fn_name, b.lo, b.hi, eps
        );
        // Commit eps STRICTLY above the raw Arb bound by a tiny relative margin
        // (1 ppb, far above any f64 round-trip / re-cert jitter) so `validate` is
        // robust rather than knife-edge. Still sound — a larger eps only widens
        // the already-sound band.
        b.eps = eps * (1.0 + 1e-9);
        b.proof_kind = ProofKind::ArbEnclosure;
    }
    env.provenance = SpecialFnProvenance {
        certify_prec: CERTIFY_PREC,
        certify_subdivisions: subdivisions,
        method: format!(
            "Arb mean-value whole-box (sup |approx - {}|), cross-checked <= naive whole-box",
            env.fn_name
        ),
    };
    if !env.is_well_formed() {
        eprintln!("error: certified envelope is not well formed (gap / eps<0 / non-finite)");
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
    eprintln!("wrote {} envelope to {out_path}", env.fn_name);
    ExitCode::SUCCESS
}

fn validate(env_path: &str) -> ExitCode {
    let env = match read_env(env_path) {
        Ok(e) => e,
        Err(c) => return c,
    };
    let f = match special_fn(&env) {
        Ok(f) => f,
        Err(c) => return c,
    };
    // Re-certify at the SAME subdivision count the stamp recorded, so the Arb
    // bound is reproduced exactly (the naive whole-box eps depends on the count;
    // a coarser re-cert would be a looser sound bound that could exceed the
    // committed eps and false-alarm). Env override still wins for probing.
    let subdivisions = std::env::var("CERTIFY_SUBDIVISIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(env.provenance.certify_subdivisions.max(1));
    let mut ok = true;
    for b in &env.boxes {
        let coeffs = arm_coeffs(&b.arm);
        let Some(arb_eps) = certify_box(f, b.lo, b.hi, &coeffs, subdivisions) else {
            eprintln!(
                "{} box [{}, {}]: re-certification failed",
                env.fn_name, b.lo, b.hi
            );
            ok = false;
            continue;
        };
        let sound = b.eps >= arb_eps;
        eprintln!(
            "{} box [{}, {}]: committed eps {:.6e} {} Arb eps {:.6e}",
            env.fn_name,
            b.lo,
            b.hi,
            b.eps,
            if sound { ">=" } else { "<  !! UNSOUND" },
            arb_eps
        );
        if !sound {
            ok = false;
        }
    }
    if !ok {
        eprintln!("error: a committed eps is below the Arb-certified bound -- unsound.");
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
                "usage:\n  certify_special_fn_envelope stamp <draft.json> <out.json>\n  \
                 certify_special_fn_envelope validate <envelope.json>"
            );
            ExitCode::from(2)
        }
    }
}
