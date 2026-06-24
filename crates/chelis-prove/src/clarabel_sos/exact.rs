//! The exact-rational SoS certificate verifier: the trust anchor.
//!
//! A sum-of-squares certificate for `p(x) >= 0 on [a, b]` is a set of symmetric
//! rational Gram matrices `Q_i` together with the fixed Markov-Lukacs interval
//! multipliers, such that
//!
//! ```text
//! p(x) == sum_i  mult_i(x) * ( z_i(x)^T Q_i z_i(x) )      (as polynomials)
//! ```
//!
//! where `z_i(x) = [1, x, ..., x^{d_i}]` is the monomial vector and each `Q_i`
//! is positive semidefinite. This module decides BOTH conjuncts EXACTLY over the
//! rationals:
//!
//! - [`RatSymMatrix::quadratic_form_poly`] expands `z^T Q z` into a
//!   [`RationalPoly`] exactly, so the polynomial identity is a `==` on trimmed
//!   rational coefficient vectors.
//! - [`RatSymMatrix::is_psd`] decides PSD-ness exactly via a rational
//!   symmetric LDL^T elimination: a negative pivot proves indefinite, and a zero
//!   pivot is PSD-compatible only if its eliminated column is entirely zero.
//!
//! Both are pure `BigRational` arithmetic, so the verifier's verdict does not
//! depend on the float SDP proposer that produced the candidate `Q_i`. A
//! candidate that does not satisfy both conjuncts is rejected here, which is
//! what makes "no false `Exact` certificate" hold regardless of the proposer.

use num_rational::BigRational;
use num_traits::{Signed, Zero};

use super::poly::RationalPoly;

/// A dense symmetric matrix with exact `BigRational` entries.
///
/// Stored as a full `n x n` row-major grid (symmetry is an invariant the
/// constructors enforce / assume, not a packed storage saving). `n` is small
/// (the Gram-matrix side length = SoS half-degree + 1), so density is fine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatSymMatrix {
    n: usize,
    /// Row-major `n*n` entries. `entry(i, j) == entry(j, i)` is an invariant.
    data: Vec<BigRational>,
}

impl RatSymMatrix {
    /// Build from a full row-major `n x n` entry vector. Panics if the length is
    /// not `n*n`. Does NOT symmetrize: callers pass a symmetric grid (the SoS
    /// Gram matrices are symmetric by construction); [`Self::is_symmetric`]
    /// lets a caller assert it.
    pub fn from_rows(n: usize, data: Vec<BigRational>) -> Self {
        assert_eq!(data.len(), n * n, "RatSymMatrix expects n*n entries");
        Self { n, data }
    }

    /// The side length.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Entry `(i, j)`.
    pub fn entry(&self, i: usize, j: usize) -> &BigRational {
        &self.data[i * self.n + j]
    }

    /// Whether the stored grid is exactly symmetric.
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.entry(i, j) != self.entry(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Expand the quadratic form `z(x)^T Q z(x)` where
    /// `z(x) = [1, x, x^2, ..., x^{n-1}]`, as an exact [`RationalPoly`].
    ///
    /// `z^T Q z = sum_{i,j} Q[i][j] * x^{i+j}`, so entry `(i, j)` contributes to
    /// the `x^{i+j}` coefficient. This is the exact polynomial a single SoS block
    /// `z^T Q z` equals; the certificate check sums these (times the interval
    /// multipliers) and compares to the goal polynomial with `==`.
    pub fn quadratic_form_poly(&self) -> RationalPoly {
        if self.n == 0 {
            return RationalPoly::zero();
        }
        // Degree of x^{i+j} tops out at 2*(n-1).
        let mut coeffs = vec![BigRational::zero(); 2 * (self.n - 1) + 1];
        for i in 0..self.n {
            for j in 0..self.n {
                let q = self.entry(i, j);
                if !q.is_zero() {
                    coeffs[i + j] += q;
                }
            }
        }
        RationalPoly::from_coeffs(coeffs)
    }

    /// Decide whether the matrix is positive semidefinite, EXACTLY, over the
    /// rationals.
    ///
    /// Method: a symmetric `LDL^T`-style Gaussian elimination with no pivoting,
    /// over exact rationals. Processing column `k`:
    ///
    /// - If the pivot `m[k][k] < 0`: the matrix is INDEFINITE (a negative
    ///   pivot exposes a direction with negative curvature), so NOT PSD.
    /// - If the pivot `m[k][k] == 0`: PSD requires the rest of column `k` (in
    ///   the current Schur complement) to be entirely zero. A non-zero
    ///   off-diagonal over a zero pivot is the `[[0, c],[c, 0]]` indefinite
    ///   pattern, so NOT PSD. If the column is all zero, the variable is in the
    ///   null space; skip the elimination for this `k` and continue.
    /// - If the pivot `m[k][k] > 0`: eliminate column `k` from the lower-right
    ///   block via the exact Schur complement
    ///   `m[i][j] -= m[i][k] * m[k][j] / m[k][k]`, then continue.
    ///
    /// If every pivot is handled without a NOT-PSD verdict, the matrix is PSD.
    /// All arithmetic is exact `BigRational`; there is no tolerance, so a matrix
    /// with a tiny rational negative eigenvalue is correctly rejected (this is
    /// exactly the near-boundary case naive float rounding gets wrong).
    pub fn is_psd(&self) -> bool {
        // Work on a mutable copy of the full matrix; eliminate in place.
        let n = self.n;
        let mut m = self.data.clone();
        let at = |m: &[BigRational], i: usize, j: usize| m[i * n + j].clone();

        for k in 0..n {
            let pivot = at(&m, k, k);
            if pivot.is_negative() {
                return false;
            }
            if pivot.is_zero() {
                // A zero pivot is PSD-compatible only if its whole column (below
                // the diagonal in the current Schur complement) is zero.
                for i in (k + 1)..n {
                    if !at(&m, i, k).is_zero() {
                        return false;
                    }
                }
                // Null-space direction: nothing to eliminate, move on.
                continue;
            }
            // pivot > 0: Schur-complement elimination of the lower-right block.
            for i in (k + 1)..n {
                let m_ik = at(&m, i, k);
                if m_ik.is_zero() {
                    continue;
                }
                for j in (k + 1)..n {
                    let delta = &m_ik * at(&m, k, j) / &pivot;
                    m[i * n + j] -= delta;
                }
            }
        }
        true
    }
}

/// One SoS block in a Markov-Lukacs certificate: a Gram matrix `gram` whose
/// quadratic form `z^T gram z` is multiplied by the fixed interval multiplier
/// `multiplier` (a polynomial: `1`, `(x - a)(b - x)`, `(x - a)`, or `(b - x)`).
#[derive(Debug, Clone)]
pub struct SosBlock {
    /// The PSD Gram matrix for this block's sum-of-squares part.
    pub gram: RatSymMatrix,
    /// The fixed interval multiplier polynomial this block's SoS is scaled by.
    pub multiplier: RationalPoly,
}

/// A full SoS certificate: the goal polynomial `p` it claims to certify
/// nonnegative, the interval `[a, b]` (for the evidence record / re-derivation),
/// and the SoS blocks.
#[derive(Debug, Clone)]
pub struct SosCertificate {
    /// The polynomial claimed nonnegative on `[lo, hi]`.
    pub poly: RationalPoly,
    /// The interval lower bound (exact rational).
    pub lo: BigRational,
    /// The interval upper bound (exact rational).
    pub hi: BigRational,
    /// The SoS blocks. `sum_i multiplier_i * (z^T gram_i z)` must equal `poly`.
    pub blocks: Vec<SosBlock>,
}

/// Why an SoS certificate failed exact verification. A failed certificate is
/// NEVER `Exact`: the engine maps any of these to an honest non-proof outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertError {
    /// A Gram matrix is not exactly symmetric (a malformed certificate).
    NotSymmetric { block: usize },
    /// A Gram matrix is not positive semidefinite over the rationals. This is
    /// the near-boundary case exact arithmetic catches that float rounding
    /// would wrongly accept.
    NotPsd { block: usize },
    /// The blocks' combined quadratic form does not equal the goal polynomial
    /// EXACTLY. The certificate proves a DIFFERENT polynomial nonnegative, so it
    /// is no evidence for `poly`.
    PolynomialMismatch,
}

impl SosCertificate {
    /// Verify the certificate EXACTLY. `Ok(())` iff every Gram matrix is
    /// symmetric and PSD over the rationals AND the blocks' combined quadratic
    /// form equals `poly` exactly. Any failure returns the specific
    /// [`CertError`]; the engine maps any error to a non-proof discharge (never
    /// `Exact`).
    ///
    /// This is the ONLY gate to a `CertificateBearing@Exact` discharge.
    pub fn verify(&self) -> Result<(), CertError> {
        // 1. Every Gram matrix must be symmetric and exactly PSD.
        for (idx, block) in self.blocks.iter().enumerate() {
            if !block.gram.is_symmetric() {
                return Err(CertError::NotSymmetric { block: idx });
            }
            if !block.gram.is_psd() {
                return Err(CertError::NotPsd { block: idx });
            }
        }
        // 2. The combined quadratic form must equal the goal polynomial exactly.
        let mut combined = RationalPoly::zero();
        for block in &self.blocks {
            let sos = block.gram.quadratic_form_poly();
            combined = combined.add(&block.multiplier.mul(&sos));
        }
        if combined == self.poly {
            Ok(())
        } else {
            Err(CertError::PolynomialMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn ri(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    fn sym(n: usize, rows: &[i64]) -> RatSymMatrix {
        RatSymMatrix::from_rows(n, rows.iter().map(|&x| ri(x)).collect())
    }

    // --- quadratic_form_poly ---

    #[test]
    fn quadratic_form_of_identity_2x2_is_one_plus_x_squared() {
        // z = [1, x]; z^T I z = 1 + x^2.
        let q = sym(2, &[1, 0, 0, 1]);
        assert_eq!(
            q.quadratic_form_poly(),
            RationalPoly::from_int_coeffs(&[1, 0, 1])
        );
    }

    #[test]
    fn quadratic_form_picks_up_cross_terms() {
        // z = [1, x]; Q = [[1, 1],[1, 1]]; z^T Q z = 1 + 2x + x^2 = (1 + x)^2.
        let q = sym(2, &[1, 1, 1, 1]);
        assert_eq!(
            q.quadratic_form_poly(),
            RationalPoly::from_int_coeffs(&[1, 2, 1])
        );
    }

    #[test]
    fn quadratic_form_1x1_is_constant() {
        let q = sym(1, &[5]);
        assert_eq!(q.quadratic_form_poly(), RationalPoly::from_int_coeffs(&[5]));
    }

    // --- is_psd: positive cases ---

    #[test]
    fn identity_is_psd() {
        assert!(sym(3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]).is_psd());
    }

    #[test]
    fn rank_one_outer_product_is_psd() {
        // v v^T with v = [1, 2]: [[1, 2],[2, 4]] (one zero eigenvalue) is PSD.
        assert!(sym(2, &[1, 2, 2, 4]).is_psd());
    }

    #[test]
    fn positive_definite_2x2_is_psd() {
        // [[2, 1],[1, 2]] eigenvalues 1, 3 > 0.
        assert!(sym(2, &[2, 1, 1, 2]).is_psd());
    }

    #[test]
    fn zero_matrix_is_psd() {
        assert!(sym(2, &[0, 0, 0, 0]).is_psd());
    }

    #[test]
    fn psd_with_leading_zero_block_then_positive() {
        // [[0, 0],[0, 3]]: zero pivot with a zero column, then positive. PSD.
        assert!(sym(2, &[0, 0, 0, 3]).is_psd());
    }

    // --- is_psd: negative cases (soundness-critical) ---

    #[test]
    fn negative_diagonal_is_not_psd() {
        assert!(!sym(2, &[-1, 0, 0, 1]).is_psd());
    }

    #[test]
    fn indefinite_2x2_is_not_psd() {
        // [[1, 2],[2, 1]] eigenvalues 3, -1: indefinite. The Schur complement
        // pivot 1 - 2*2/1 = -3 < 0 exposes it.
        assert!(!sym(2, &[1, 2, 2, 1]).is_psd());
    }

    #[test]
    fn zero_pivot_with_nonzero_offdiagonal_is_not_psd() {
        // [[0, 1],[1, 0]] eigenvalues 1, -1: the zero-pivot column is non-zero,
        // so NOT PSD. This is the pattern naive handling gets wrong.
        assert!(!sym(2, &[0, 1, 1, 0]).is_psd());
    }

    #[test]
    fn near_boundary_tiny_negative_eigenvalue_is_not_psd() {
        // [[1, 1],[1, 999/1000]]: det = 999/1000 - 1 = -1/1000 < 0, so one
        // eigenvalue is a tiny negative. EXACT arithmetic rejects it; this is
        // exactly the near-boundary case a float SDP "solution" would wrongly
        // accept and naive rounding cannot reconstruct.
        let q = RatSymMatrix::from_rows(2, vec![ri(1), ri(1), ri(1), r(999, 1000)]);
        assert!(!q.is_psd());
    }

    #[test]
    fn just_psd_boundary_is_accepted() {
        // [[1, 1],[1, 1]]: det = 0, eigenvalues 0 and 2. Exactly PSD (the
        // boundary). Pairs with the tiny-negative test above: 1/1000 either side
        // of the boundary flips the verdict, and exact arithmetic gets both.
        assert!(sym(2, &[1, 1, 1, 1]).is_psd());
    }

    #[test]
    fn three_by_three_indefinite_is_not_psd() {
        // Diagonal-dominant-looking but with a large off-diagonal that makes the
        // third Schur pivot negative.
        assert!(!sym(3, &[1, 0, 0, 0, 1, 2, 0, 2, 1]).is_psd());
    }

    #[test]
    fn is_symmetric_detects_asymmetry() {
        let asym = RatSymMatrix::from_rows(2, vec![ri(1), ri(2), ri(3), ri(4)]);
        assert!(!asym.is_symmetric());
        assert!(sym(2, &[1, 2, 2, 4]).is_symmetric());
    }

    // --- SosCertificate::verify: positive ---

    #[test]
    fn verify_accepts_a_genuine_sos_certificate() {
        // p(x) = 1 + x^2 is globally nonnegative; on any interval the block
        // sigma0 = z^T I z = 1 + x^2 with multiplier 1 certifies it. (We are
        // verifying the certificate identity + PSD, independent of the interval
        // here.)
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 0, 1]),
            lo: ri(-5),
            hi: ri(5),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 0, 0, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn verify_accepts_a_perfect_square_certificate() {
        // p(x) = (1 + x)^2 = 1 + 2x + x^2; Gram [[1,1],[1,1]] (PSD, rank 1).
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 2, 1]),
            lo: ri(-10),
            hi: ri(10),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 1, 1, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn verify_accepts_interval_multiplier_certificate() {
        // On [0, 1]: p(x) = x - x^2 = x(1 - x) = (x - 0)(1 - x), nonnegative on
        // [0,1]. Markov-Lukacs odd-degree form: (x - a) s + (b - x) t. Here a=0,
        // b=1. Take s = (1-x)? No -- s, t must be SoS. Simpler valid cert:
        // p = (x-0)*sigma_a + (1-x)*sigma_b is not unique. Use the even-degree
        // form p = sigma0 + (x-0)(1-x)*sigma1 with sigma0 = 0 (Gram 0) and
        // sigma1 = 1 (Gram [1], the constant SoS 1): then
        // (x)(1-x)*1 = x - x^2 = p. Multiplier = (x-0)(1-x) = x - x^2.
        let mult = RationalPoly::from_int_coeffs(&[0, 1, -1]); // x - x^2
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[0, 1, -1]),
            lo: ri(0),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(1, &[1]), // sigma1 = 1
                multiplier: mult,
            }],
        };
        assert_eq!(cert.verify(), Ok(()));
    }

    // --- SosCertificate::verify: negatives (no false Exact) ---

    #[test]
    fn verify_rejects_non_psd_gram() {
        // The quadratic form of an indefinite Gram may still equal p, but the
        // Gram is not PSD, so it is NOT a sum of squares -> rejected.
        // [[1,2],[2,1]] quadratic form = 1 + 4x + x^2; claim that as poly.
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 4, 1]),
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 2, 2, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        assert_eq!(cert.verify(), Err(CertError::NotPsd { block: 0 }));
    }

    #[test]
    fn verify_rejects_polynomial_mismatch() {
        // PSD Gram (identity, form 1 + x^2) but the claimed poly is 1 + x^2 + x
        // -- the certificate proves a different polynomial nonnegative, so it is
        // no evidence for the claimed one.
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 1, 1]),
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(2, &[1, 0, 0, 1]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        assert_eq!(cert.verify(), Err(CertError::PolynomialMismatch));
    }

    #[test]
    fn verify_rejects_asymmetric_gram() {
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[1, 0, 1]),
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: RatSymMatrix::from_rows(2, vec![ri(1), ri(2), ri(3), ri(1)]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        assert_eq!(cert.verify(), Err(CertError::NotSymmetric { block: 0 }));
    }

    #[test]
    fn verify_rejects_tiny_negative_eigenvalue_gram_as_not_psd() {
        // The certificate's polynomial identity may hold while the Gram has a
        // tiny rational negative eigenvalue (the boundary-degeneracy failure
        // mode). EXACT PSD must reject it -> no false Exact. Gram
        // [[1,1],[1,999/1000]] has quadratic form 1 + 2x + (999/1000) x^2.
        let cert = SosCertificate {
            poly: RationalPoly::from_coeffs(vec![ri(1), ri(2), r(999, 1000)]),
            lo: ri(-1),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: RatSymMatrix::from_rows(2, vec![ri(1), ri(1), ri(1), r(999, 1000)]),
                multiplier: RationalPoly::from_int_coeffs(&[1]),
            }],
        };
        assert_eq!(cert.verify(), Err(CertError::NotPsd { block: 0 }));
    }
}
