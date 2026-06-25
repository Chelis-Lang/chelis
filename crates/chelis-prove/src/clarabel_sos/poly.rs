//! Exact univariate polynomials over the rationals.
//!
//! The SoS certificate's trust anchor is exact rational arithmetic: a
//! certificate is valid iff its Gram-matrix quadratic form re-multiplies to the
//! goal polynomial EXACTLY (this module's `==`) and every Gram matrix is exactly
//! PSD over the rationals (see [`super::exact`]). Nothing here uses floats, so
//! the verifier is independent of the Clarabel float SDP proposer's precision.
//!
//! Representation: a dense coefficient vector `coeffs`, little-endian by degree
//! (`coeffs[i]` is the coefficient of `x^i`). The canonical form has no trailing
//! zero coefficients (so equal polynomials have an equal coefficient vector and
//! an equal degree), enforced by [`RationalPoly::trim`].

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

/// A univariate polynomial with exact `BigRational` coefficients, little-endian
/// by degree. Always kept trimmed (no trailing zero coefficient), so equality is
/// coefficient-vector equality and [`RationalPoly::degree`] is exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalPoly {
    /// `coeffs[i]` is the coefficient of `x^i`. Trimmed: the last entry (if any)
    /// is non-zero. The zero polynomial has an empty vector.
    coeffs: Vec<BigRational>,
}

impl RationalPoly {
    /// The zero polynomial.
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// A constant polynomial `c`.
    pub fn constant(c: BigRational) -> Self {
        Self { coeffs: vec![c] }.trimmed()
    }

    /// The monomial `x` (degree 1, coefficient 1).
    pub fn x() -> Self {
        Self {
            coeffs: vec![BigRational::zero(), BigRational::one()],
        }
    }

    /// Build from a little-endian coefficient vector (`coeffs[i]` for `x^i`),
    /// trimming trailing zeros.
    pub fn from_coeffs(coeffs: Vec<BigRational>) -> Self {
        Self { coeffs }.trimmed()
    }

    /// Build from integer coefficients (little-endian), for ergonomic tests and
    /// the interval-multiplier construction.
    pub fn from_int_coeffs(coeffs: &[i64]) -> Self {
        Self::from_coeffs(
            coeffs
                .iter()
                .map(|&c| BigRational::from(BigInt::from(c)))
                .collect(),
        )
    }

    /// Drop trailing zero coefficients in place so the representation is
    /// canonical. The zero polynomial canonicalizes to an empty vector.
    fn trim(&mut self) {
        while self.coeffs.last().is_some_and(Zero::is_zero) {
            self.coeffs.pop();
        }
    }

    fn trimmed(mut self) -> Self {
        self.trim();
        self
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The degree, or `None` for the zero polynomial (which has no degree). A
    /// constant non-zero polynomial has degree 0.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// The coefficient of `x^i` (zero past the top degree).
    pub fn coeff(&self, i: usize) -> BigRational {
        self.coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    }

    /// The trimmed little-endian coefficient slice.
    pub fn coeffs(&self) -> &[BigRational] {
        &self.coeffs
    }

    /// Exact sum.
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) + other.coeff(i));
        }
        Self::from_coeffs(out)
    }

    /// Exact difference `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) - other.coeff(i));
        }
        Self::from_coeffs(out)
    }

    /// Exact negation.
    pub fn neg(&self) -> Self {
        Self::from_coeffs(self.coeffs.iter().map(|c| -c.clone()).collect())
    }

    /// Exact product (schoolbook convolution; degrees here are small).
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![BigRational::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Self::from_coeffs(out)
    }

    /// Scale by an exact rational.
    pub fn scale(&self, k: &BigRational) -> Self {
        if k.is_zero() {
            return Self::zero();
        }
        Self::from_coeffs(self.coeffs.iter().map(|c| c * k).collect())
    }

    /// Evaluate at an exact rational point (Horner).
    pub fn eval(&self, x: &BigRational) -> BigRational {
        let mut acc = BigRational::zero();
        for c in self.coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    fn ri(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    #[test]
    fn zero_has_no_degree_and_is_zero() {
        let p = RationalPoly::zero();
        assert!(p.is_zero());
        assert_eq!(p.degree(), None);
        assert!(p.coeffs().is_empty());
    }

    #[test]
    fn from_coeffs_trims_trailing_zeros() {
        // 1 + 2x + 0x^2 + 0x^3  ==  1 + 2x (degree 1, two coeffs).
        let p = RationalPoly::from_coeffs(vec![ri(1), ri(2), ri(0), ri(0)]);
        assert_eq!(p.degree(), Some(1));
        assert_eq!(p.coeffs(), &[ri(1), ri(2)]);
    }

    #[test]
    fn all_zero_coeffs_canonicalize_to_zero() {
        let p = RationalPoly::from_int_coeffs(&[0, 0, 0]);
        assert!(p.is_zero());
        assert_eq!(p, RationalPoly::zero());
    }

    #[test]
    fn equal_polys_compare_equal_regardless_of_trailing_zeros() {
        let a = RationalPoly::from_int_coeffs(&[3, -1, 4]);
        let b = RationalPoly::from_coeffs(vec![ri(3), ri(-1), ri(4), ri(0)]);
        assert_eq!(a, b);
    }

    #[test]
    fn add_and_sub_are_exact() {
        let a = RationalPoly::from_int_coeffs(&[1, 2, 3]);
        let b = RationalPoly::from_int_coeffs(&[0, -2, 5]);
        assert_eq!(a.add(&b), RationalPoly::from_int_coeffs(&[1, 0, 8]));
        assert_eq!(a.sub(&b), RationalPoly::from_int_coeffs(&[1, 4, -2]));
        // (a - a) trims to the zero polynomial.
        assert!(a.sub(&a).is_zero());
    }

    #[test]
    fn mul_is_exact_convolution() {
        // (1 + x)(1 - x) = 1 - x^2.
        let one_plus_x = RationalPoly::from_int_coeffs(&[1, 1]);
        let one_minus_x = RationalPoly::from_int_coeffs(&[1, -1]);
        assert_eq!(
            one_plus_x.mul(&one_minus_x),
            RationalPoly::from_int_coeffs(&[1, 0, -1])
        );
    }

    #[test]
    fn mul_by_zero_is_zero() {
        let a = RationalPoly::from_int_coeffs(&[1, 2, 3]);
        assert!(a.mul(&RationalPoly::zero()).is_zero());
    }

    #[test]
    fn mul_handles_rational_coeffs_exactly() {
        // (1/2 + 1/3 x)^2 = 1/4 + 1/3 x + 1/9 x^2.
        let p = RationalPoly::from_coeffs(vec![r(1, 2), r(1, 3)]);
        let sq = p.mul(&p);
        assert_eq!(
            sq,
            RationalPoly::from_coeffs(vec![r(1, 4), r(1, 3), r(1, 9)])
        );
    }

    #[test]
    fn eval_is_exact() {
        // p(x) = 1 - x^2, p(1/2) = 3/4.
        let p = RationalPoly::from_int_coeffs(&[1, 0, -1]);
        assert_eq!(p.eval(&r(1, 2)), r(3, 4));
        assert_eq!(p.eval(&ri(0)), ri(1));
        assert_eq!(p.eval(&ri(1)), ri(0));
    }

    #[test]
    fn scale_by_zero_is_zero() {
        let a = RationalPoly::from_int_coeffs(&[1, 2, 3]);
        assert!(a.scale(&ri(0)).is_zero());
        assert_eq!(a.scale(&ri(2)), RationalPoly::from_int_coeffs(&[2, 4, 6]));
    }

    #[test]
    fn x_is_the_degree_one_monomial() {
        let x = RationalPoly::x();
        assert_eq!(x.degree(), Some(1));
        assert_eq!(x.coeff(0), ri(0));
        assert_eq!(x.coeff(1), ri(1));
        // x * x = x^2.
        assert_eq!(x.mul(&x), RationalPoly::from_int_coeffs(&[0, 0, 1]));
    }
}
