//! The Clarabel float SDP proposer (behind the BLAS-linked `sdp` backend).
//!
//! This is the ONLY float / BLAS-dependent step. It PROPOSES a candidate SoS
//! certificate by solving the Markov-Lukacs PSD feasibility SDP with Clarabel,
//! reading back the float Gram matrices, rounding to rationals, and running the
//! Peyrl-Parrilo exact projection ([`super::linalg::project_onto_affine`]) so
//! the candidate's polynomial identity holds EXACTLY. It does NOT decide
//! soundness: the engine ([`super::engine`]) re-verifies the candidate exactly
//! before trusting it, so a numerically-off or boundary-degenerate proposal is
//! rejected downstream rather than laundered into a false `Exact`.
//!
//! # SDP variable layout
//!
//! The SDP decision variables are the layout's flat UPPER-TRIANGLE Gram entries
//! `u` (unscaled), exactly the variables [`super::encode::SosLayout`] indexes.
//! Two constraint groups:
//!
//! - coefficient-matching equalities `A u = p` (a `ZeroConeT`): the SoS identity
//!   as a linear system, built by [`super::encode::SosLayout::coeff_matching_system`];
//! - per-block PSD: a `PSDTriangleConeT(s_b)` on the block's svec. Clarabel's PSD
//!   triangle cone consumes the symmetric matrix as its upper triangle,
//!   column-major, with off-diagonal entries scaled by sqrt(2). The cone slack
//!   `s = b - A_psd u` must equal the block's svec, so `A_psd` maps each flat
//!   upper-triangle var to its svec coordinate with the sqrt(2) factor on
//!   off-diagonals and the column-major reordering.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT::PSDTriangleConeT,
    SupportedConeT::ZeroConeT,
};
use num_bigint::{BigInt, Sign};
use num_rational::BigRational;

use super::encode::SosLayout;
use super::engine::SosProposer;
use super::exact::SosCertificate;
use super::extract::PolyOnInterval;
use super::linalg::project_onto_affine;

/// The Clarabel SDP proposer. Solves the Markov-Lukacs PSD feasibility problem
/// in floating point, then rounds + projects to an exact rational candidate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClarabelProposer;

impl ClarabelProposer {
    pub const fn new() -> Self {
        Self
    }
}

/// Denominator the float-Gram rounding rounds each entry to (round `x` to
/// `round(x * D) / D`). A moderate power of two keeps the rationals small while
/// preserving enough of the float solution for the exact projection's free
/// coordinates to land near a genuinely PSD representative.
const ROUND_DENOM: i64 = 1 << 20;

/// Round a finite `f64` to a rational with denominator [`ROUND_DENOM`]:
/// `round(x * D) / D`. A non-finite value yields `None` (the proposal is
/// abandoned; the engine then reports honest Unknown).
fn round_f64_to_rational(x: f64) -> Option<BigRational> {
    if !x.is_finite() {
        return None;
    }
    let scaled = (x * ROUND_DENOM as f64).round();
    if !scaled.is_finite() {
        return None;
    }
    // scaled is an integer-valued f64; convert exactly.
    let num = bigint_from_f64_round(scaled)?;
    Some(BigRational::new(num, BigInt::from(ROUND_DENOM)))
}

/// Convert an integer-valued `f64` to a `BigInt` exactly.
fn bigint_from_f64_round(x: f64) -> Option<BigInt> {
    if !x.is_finite() {
        return None;
    }
    let r = BigRational::from_float(x)?;
    // x is integer-valued, so the rational is an integer (denominator 1).
    let (sign, val) = if r.numer().sign() == Sign::Minus {
        (Sign::Minus, -r.numer().clone())
    } else {
        (Sign::Plus, r.numer().clone())
    };
    // r = numer/denom; for an integer-valued f64 denom is 1.
    let int = val / r.denom();
    Some(match sign {
        Sign::Minus => -int,
        _ => int,
    })
}

/// The svec length of an `s x s` symmetric matrix (upper/lower triangle).
fn svec_len(s: usize) -> usize {
    s * (s + 1) / 2
}

impl SosProposer for ClarabelProposer {
    fn propose(&self, goal: &PolyOnInterval) -> Option<SosCertificate> {
        let degree = goal.poly.degree()?;
        // A degree-0 (constant) goal: handled by the layout (single side-1
        // block). Larger degrees use the even/odd Markov-Lukacs structure.
        let layout = SosLayout::for_goal(degree, &goal.lo, &goal.hi);
        let n_vars = layout.n_vars;
        if n_vars == 0 {
            return None;
        }

        // --- Build the Clarabel SDP in f64 ---
        // Decision variables: the n_vars flat upper-triangle Gram entries.
        // Objective: feasibility (P = 0, q = 0).
        let p_obj = CscMatrix::<f64>::zeros((n_vars, n_vars));
        let q = vec![0.0_f64; n_vars];

        // Coefficient-matching equalities A_eq u = p (ZeroCone). The exact
        // system has degree+1 rows; lower its rational entries to f64 for the
        // float solve (the exact version is re-imposed by the projection).
        let aug = layout.coeff_matching_system(&goal.poly);
        let n_eq = aug.rows();

        // Assemble the full constraint matrix A (n_eq + sum svec_len rows) and b.
        // Row blocks: [equalities; per-block PSD svec].
        let psd_sizes: Vec<usize> = layout.blocks.iter().map(|blk| blk.size).collect();
        let n_psd_rows: usize = psd_sizes.iter().map(|&s| svec_len(s)).sum();
        let total_rows = n_eq + n_psd_rows;

        // Dense build then convert to CSC (problems here are tiny).
        let mut a_dense = vec![vec![0.0_f64; n_vars]; total_rows];
        let mut b = vec![0.0_f64; total_rows];

        // Equalities: A_eq from the augmented matrix's variable columns; b = p.
        for (r, (a_row, b_r)) in a_dense.iter_mut().zip(b.iter_mut()).take(n_eq).enumerate() {
            for (c, cell) in a_row.iter_mut().enumerate().take(n_vars) {
                *cell = rational_to_f64(aug.at(r, c));
            }
            *b_r = rational_to_f64(aug.at(r, n_vars));
        }

        // PSD blocks: cone slack s = b - A u must be the block's svec (in
        // Clarabel's sqrt(2)-scaled, column-major upper-triangle order). With
        // b = 0 for these rows, s = -A u; set A so that -A u = svec(u-block),
        // i.e. A row for svec coordinate maps to the negative of the var with
        // the sqrt(2) factor.
        let sqrt2 = std::f64::consts::SQRT_2;
        let mut row = n_eq;
        for (bidx, &s) in psd_sizes.iter().enumerate() {
            // Column-major upper triangle: for col j, rows i = 0..=j.
            for j in 0..s {
                for i in 0..=j {
                    let var = layout.var_index(bidx, i, j);
                    let scale = if i == j { 1.0 } else { sqrt2 };
                    // s_row = -A[row][var]*u = +scale*u  => A[row][var] = -scale.
                    // b[row] stays 0 (the slack equals the svec directly).
                    a_dense[row][var] = -scale;
                    row += 1;
                }
            }
        }
        debug_assert_eq!(row, total_rows);

        let a_csc = dense_to_csc(&a_dense, total_rows, n_vars);

        // Cones: one ZeroCone for the equalities, then one PSDTriangle per block.
        let mut cones = vec![ZeroConeT(n_eq)];
        for &s in &psd_sizes {
            cones.push(PSDTriangleConeT(s));
        }

        // --- Solve (float) ---
        let settings = DefaultSettings {
            verbose: false,
            ..DefaultSettings::default()
        };
        // `new` returns a Result in clarabel 0.11; a construction error (a
        // malformed problem) is not a proof failure -- treat it as "no
        // proposal" so the engine reports honest Unknown.
        let mut solver = DefaultSolver::new(&p_obj, &q, &a_csc, &b, &cones, settings).ok()?;
        solver.solve();
        // Only a solved (optimal/feasible) status yields a usable proposal.
        // Anything else (infeasible, numerical failure) -> no proposal; the
        // engine reports honest Unknown.
        if !matches!(solver.solution.status, SolverStatus::Solved) {
            return None;
        }

        // --- Round + Peyrl-Parrilo exact projection ---
        let x_float = &solver.solution.x;
        if x_float.len() != n_vars {
            return None;
        }
        let mut x0 = Vec::with_capacity(n_vars);
        for &xf in x_float.iter() {
            x0.push(round_f64_to_rational(xf)?);
        }
        // Project onto the EXACT coefficient-matching affine subspace: the
        // result reproduces the goal polynomial exactly (the identity holds);
        // only PSD-ness can still fail, which the engine's exact verifier
        // catches.
        let x_exact = project_onto_affine(&aug, n_vars, &x0).ok()?;
        Some(layout.certificate_from_vars(&x_exact, &goal.poly, &goal.lo, &goal.hi))
    }
}

/// Lower an exact rational to the nearest `f64` (for building the float SDP).
/// Only used to feed Clarabel; soundness does not depend on this conversion.
fn rational_to_f64(r: &BigRational) -> f64 {
    let num = bigint_to_f64(r.numer());
    let den = bigint_to_f64(r.denom());
    if den == 0.0 { 0.0 } else { num / den }
}

fn bigint_to_f64(n: &BigInt) -> f64 {
    // BigInt -> f64 via its decimal string; adequate for the tiny coefficients
    // here and never on the trust path.
    n.to_string().parse::<f64>().unwrap_or(0.0)
}

/// Convert a dense row-major matrix to Clarabel's CSC form.
fn dense_to_csc(dense: &[Vec<f64>], rows: usize, cols: usize) -> CscMatrix<f64> {
    let mut colptr = Vec::with_capacity(cols + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for c in 0..cols {
        for (r, drow) in dense.iter().enumerate().take(rows) {
            let v = drow[c];
            if v != 0.0 {
                rowval.push(r);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(rows, cols, colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::super::poly::RationalPoly;
    use super::*;
    use crate::clarabel_sos::exact::CertError;

    fn ri(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    fn goal(poly: RationalPoly, lo: i64, hi: i64, strict: bool) -> PolyOnInterval {
        PolyOnInterval {
            var: "x".to_string(),
            poly,
            lo: ri(lo),
            hi: ri(hi),
            strict,
        }
    }

    #[test]
    fn round_f64_is_exact_for_dyadic_values() {
        // 0.5 = 2^-1 rounds to exactly 1/2.
        assert_eq!(
            round_f64_to_rational(0.5).unwrap(),
            BigRational::new(BigInt::from(1), BigInt::from(2))
        );
        // 0.0 -> 0.
        assert_eq!(round_f64_to_rational(0.0).unwrap(), ri(0));
        // negative.
        assert_eq!(round_f64_to_rational(-2.0).unwrap(), ri(-2));
    }

    #[test]
    fn round_f64_rejects_non_finite() {
        assert!(round_f64_to_rational(f64::INFINITY).is_none());
        assert!(round_f64_to_rational(f64::NAN).is_none());
    }

    // --- the actual Clarabel SDP solve (needs the BLAS-linked sdp backend) ---

    #[test]
    fn proposes_a_verifying_certificate_for_x_squared() {
        // x^2 >= 0 on [-1, 1]: globally nonnegative, the SDP is feasible, the
        // proposed certificate must VERIFY exactly.
        let p = ClarabelProposer::new();
        let g = goal(RationalPoly::from_int_coeffs(&[0, 0, 1]), -1, 1, false);
        let cert = p.propose(&g).expect("a certificate is proposed for x^2");
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn proposes_a_verifying_certificate_for_interval_positive_poly() {
        // x - x^2 = x(1-x) >= 0 on [0, 1].
        let p = ClarabelProposer::new();
        let g = goal(RationalPoly::from_int_coeffs(&[0, 1, -1]), 0, 1, false);
        let cert = p.propose(&g).expect("a certificate is proposed");
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn proposes_a_verifying_certificate_for_shifted_square() {
        // (x - 3)^2 = 9 - 6x + x^2 >= 0 on [0, 5].
        let p = ClarabelProposer::new();
        let g = goal(RationalPoly::from_int_coeffs(&[9, -6, 1]), 0, 5, false);
        let cert = p.propose(&g).expect("a certificate is proposed");
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn proposes_for_a_strictly_positive_quartic() {
        // x^4 + 1 >= 0 on [-2, 2] (degree 4, even).
        let p = ClarabelProposer::new();
        let g = goal(
            RationalPoly::from_int_coeffs(&[1, 0, 0, 0, 1]),
            -2,
            2,
            false,
        );
        let cert = p
            .propose(&g)
            .expect("a certificate is proposed for x^4 + 1");
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn negative_polynomial_yields_no_certificate_or_a_failing_one() {
        // p = x^2 - 1 is NEGATIVE on (-1, 1), so it is NOT nonnegative on
        // [-1, 1]: there is no SoS certificate. The proposer must either return
        // None (SDP infeasible) or a candidate that FAILS exact verification --
        // never a verifying certificate (that would be a false proof). Pin the
        // soundness-critical property: no VERIFYING cert comes back.
        let p = ClarabelProposer::new();
        let g = goal(RationalPoly::from_int_coeffs(&[-1, 0, 1]), -1, 1, false);
        match p.propose(&g) {
            None => {}
            Some(cert) => {
                // If a candidate is returned it MUST NOT verify (no false cert).
                assert!(
                    matches!(
                        cert.verify(),
                        Err(CertError::NotPsd { .. }) | Err(CertError::PolynomialMismatch)
                    ),
                    "a negative polynomial must not yield a verifying SoS certificate"
                );
            }
        }
    }
}
