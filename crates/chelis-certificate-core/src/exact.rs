//! Exact certificate arithmetic without proof authority.

use alloc::{vec, vec::Vec};

use num_rational::BigRational;
use num_traits::{Signed, Zero};

use super::poly::RationalPoly;

#[derive(Debug, Clone, PartialEq, Eq)]
/// A dense rational matrix with full row-major storage.
/// The constructor does not enforce symmetry.
pub struct RatSymMatrix {
    n: usize,
    data: Vec<BigRational>,
}

impl RatSymMatrix {
    /// Construct a matrix. The entry count must equal `n * n`.
    /// This function does not change entries or enforce symmetry.
    pub fn from_rows(n: usize, data: Vec<BigRational>) -> Self {
        assert_eq!(data.len(), n * n, "RatSymMatrix expects n*n entries");
        Self { n, data }
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn entry(&self, i: usize, j: usize) -> &BigRational {
        &self.data[i * self.n + j]
    }

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

    pub fn quadratic_form_poly(&self) -> RationalPoly {
        if self.n == 0 {
            return RationalPoly::zero();
        }
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

    /// Decide exact positive semidefiniteness for a symmetric input matrix.
    /// The caller must establish symmetry before this call.
    /// This function does not check symmetry.
    pub fn is_psd(&self) -> bool {
        let n = self.n;
        let mut m = self.data.clone();
        let at = |m: &[BigRational], i: usize, j: usize| m[i * n + j].clone();

        for k in 0..n {
            let pivot = at(&m, k, k);
            if pivot.is_negative() {
                return false;
            }
            if pivot.is_zero() {
                for i in (k + 1)..n {
                    if !at(&m, i, k).is_zero() {
                        return false;
                    }
                }
                continue;
            }
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

#[derive(Debug, Clone)]
/// A Gram matrix and its polynomial multiplier.
/// Arithmetic verification does not check multiplier nonnegativity.
pub struct SosBlock {
    pub gram: RatSymMatrix,
    pub multiplier: RationalPoly,
}

#[derive(Debug, Clone)]
/// A polynomial identity candidate with interval metadata.
/// The interval metadata does not constrain arithmetic verification.
pub struct SosCertificate {
    pub poly: RationalPoly,
    pub lo: BigRational,
    pub hi: BigRational,
    pub blocks: Vec<SosBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// An exact arithmetic verification error.
pub enum CertError {
    NotSymmetric { block: usize },
    NotPsd { block: usize },
    PolynomialMismatch,
}

impl SosCertificate {
    /// Check block symmetry, PSD, and equality with `self.poly`.
    /// Blocks retain their order. Symmetry precedes PSD within each block.
    /// All block checks precede the polynomial identity check.
    /// Success grants no proof authority.
    /// This method does not check the requested goal, interval order,
    /// multiplier nonnegativity, strictness, or provenance.
    pub fn verify(&self) -> Result<(), CertError> {
        for (idx, block) in self.blocks.iter().enumerate() {
            if !block.gram.is_symmetric() {
                return Err(CertError::NotSymmetric { block: idx });
            }
            if !block.gram.is_psd() {
                return Err(CertError::NotPsd { block: idx });
            }
        }
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

    #[test]
    fn quadratic_form_of_identity_2x2_is_one_plus_x_squared() {
        let q = sym(2, &[1, 0, 0, 1]);
        assert_eq!(
            q.quadratic_form_poly(),
            RationalPoly::from_int_coeffs(&[1, 0, 1])
        );
    }

    #[test]
    fn quadratic_form_picks_up_cross_terms() {
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

    #[test]
    fn identity_is_psd() {
        assert!(sym(3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]).is_psd());
    }

    #[test]
    fn rank_one_outer_product_is_psd() {
        assert!(sym(2, &[1, 2, 2, 4]).is_psd());
    }

    #[test]
    fn positive_definite_2x2_is_psd() {
        assert!(sym(2, &[2, 1, 1, 2]).is_psd());
    }

    #[test]
    fn zero_matrix_is_psd() {
        assert!(sym(2, &[0, 0, 0, 0]).is_psd());
    }

    #[test]
    fn psd_with_leading_zero_block_then_positive() {
        assert!(sym(2, &[0, 0, 0, 3]).is_psd());
    }

    #[test]
    fn negative_diagonal_is_not_psd() {
        assert!(!sym(2, &[-1, 0, 0, 1]).is_psd());
    }

    #[test]
    fn indefinite_2x2_is_not_psd() {
        assert!(!sym(2, &[1, 2, 2, 1]).is_psd());
    }

    #[test]
    fn zero_pivot_with_nonzero_offdiagonal_is_not_psd() {
        assert!(!sym(2, &[0, 1, 1, 0]).is_psd());
    }

    #[test]
    fn near_boundary_tiny_negative_eigenvalue_is_not_psd() {
        let q = RatSymMatrix::from_rows(2, vec![ri(1), ri(1), ri(1), r(999, 1000)]);
        assert!(!q.is_psd());
    }

    #[test]
    fn just_psd_boundary_is_accepted() {
        assert!(sym(2, &[1, 1, 1, 1]).is_psd());
    }

    #[test]
    fn three_by_three_indefinite_is_not_psd() {
        assert!(!sym(3, &[1, 0, 0, 0, 1, 2, 0, 2, 1]).is_psd());
    }

    #[test]
    fn is_symmetric_detects_asymmetry() {
        let asym = RatSymMatrix::from_rows(2, vec![ri(1), ri(2), ri(3), ri(4)]);
        assert!(!asym.is_symmetric());
        assert!(sym(2, &[1, 2, 2, 4]).is_symmetric());
    }

    #[test]
    fn verify_accepts_a_genuine_sos_certificate() {
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
        let mult = RationalPoly::from_int_coeffs(&[0, 1, -1]);
        let cert = SosCertificate {
            poly: RationalPoly::from_int_coeffs(&[0, 1, -1]),
            lo: ri(0),
            hi: ri(1),
            blocks: vec![SosBlock {
                gram: sym(1, &[1]),
                multiplier: mult,
            }],
        };
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn verify_rejects_non_psd_gram() {
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
