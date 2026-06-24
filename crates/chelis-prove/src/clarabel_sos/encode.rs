//! The Markov-Lukacs SoS-to-SDP encoding: structural layout and the exact
//! coefficient-matching linear system.
//!
//! Decision-INDEPENDENT (pure `BigRational`): this is the math the Clarabel
//! float SDP and the Peyrl-Parrilo rational repair BOTH consume, identical
//! regardless of which float backend proposes the Gram matrices.
//!
//! # Markov-Lukacs representation of `p(x) >= 0` on `[a, b]`
//!
//! For a univariate polynomial `p` of degree `n` nonnegative on `[a, b]`:
//!
//! - even `n = 2m`:  `p = sigma0 + (x - a)(b - x) * sigma1`, with `sigma0` SoS of
//!   degree `2m` (Gram side `m + 1`, basis `[1, x, ..., x^m]`) and `sigma1` SoS
//!   of degree `2m - 2` (Gram side `m`, basis `[1, ..., x^{m-1}]`).
//! - odd `n = 2m + 1`:  `p = (x - a) * sigma0 + (b - x) * sigma1`, with `sigma0`,
//!   `sigma1` SoS of degree `2m` (each Gram side `m + 1`).
//!
//! Each block's SoS is `sigma_b(x) = z(x)^T Q_b z(x)` for a PSD Gram `Q_b`; the
//! identity `p == sum_b mult_b * sigma_b` is LINEAR in the Gram entries, so
//! "find the decomposition" is a semidefinite feasibility problem: the PSD cones
//! on each `Q_b`, plus the linear coefficient-matching equalities.
//!
//! # Variable layout
//!
//! The free variables are the UPPER-TRIANGLE Gram entries (including the
//! diagonal) of every block, `s_b (s_b + 1) / 2` per block of side `s_b`. A
//! global flat index orders them block-major, then `(i, j)` with `i <= j` in
//! row-major upper-triangle order. [`SosLayout`] owns this indexing.
//!
//! For the symmetric Gram `Q`, the coefficient of `x^k` in `z^T Q z` is
//! `sum_{i<=j, i+j=k} c_{ij} u_{ij}` where `u_{ij}` is the upper-triangle
//! variable and `c_{ij} = 1` on the diagonal (`i == j`) and `2` off it (the
//! entry appears as both `Q[i][j]` and `Q[j][i]`).

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

use super::exact::{RatSymMatrix, SosBlock, SosCertificate};
use super::linalg::RatMatrix;
use super::poly::RationalPoly;

/// One block of the SoS layout: a symmetric Gram of side `size`, scaled by a
/// fixed interval multiplier polynomial.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// The Gram matrix side length.
    pub size: usize,
    /// The interval multiplier this block's SoS is multiplied by.
    pub multiplier: RationalPoly,
    /// The flat index of this block's first upper-triangle variable.
    pub var_offset: usize,
}

/// The structural layout of a Markov-Lukacs SoS decomposition for a given goal:
/// the blocks (sizes + multipliers) and the global upper-triangle variable
/// indexing.
#[derive(Debug, Clone)]
pub struct SosLayout {
    /// The degree of the goal polynomial this layout targets.
    pub degree: usize,
    /// The blocks, in flat-index order.
    pub blocks: Vec<BlockSpec>,
    /// Total number of upper-triangle variables across all blocks.
    pub n_vars: usize,
}

/// The number of upper-triangle entries (including the diagonal) of an
/// `s x s` symmetric matrix.
fn upper_tri_count(s: usize) -> usize {
    s * (s + 1) / 2
}

/// The flat upper-triangle offset of entry `(i, j)` with `i <= j` within an
/// `s x s` block (row-major upper triangle: row 0 has `s` entries, row 1 has
/// `s - 1`, ...).
fn upper_tri_index(s: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j && j < s);
    // entries before row i: sum_{r<i} (s - r) = i*s - (0 + 1 + ... + (i-1))
    //                                         = i*s - i*(i-1)/2.
    // Compute the triangular number i*(i-1)/2 without the usize underflow at
    // i == 0.
    let tri_below = i * i.saturating_sub(1) / 2;
    let before = i * s - tri_below;
    before + (j - i)
}

impl SosLayout {
    /// Build the Markov-Lukacs layout for a goal polynomial of degree `degree`
    /// nonnegative on `[lo, hi]`. The constant / linear interval multipliers are
    /// built from `lo`, `hi`.
    pub fn for_goal(degree: usize, lo: &BigRational, hi: &BigRational) -> Self {
        let one = RationalPoly::from_int_coeffs(&[1]);
        // (x - a): coeffs [-a, 1]; (b - x): coeffs [b, -1].
        let x_minus_a =
            RationalPoly::from_coeffs(vec![-lo.clone(), BigRational::from(BigInt::from(1))]);
        let b_minus_x =
            RationalPoly::from_coeffs(vec![hi.clone(), -BigRational::from(BigInt::from(1))]);
        // (x - a)(b - x).
        let interval = x_minus_a.mul(&b_minus_x);

        let mut blocks: Vec<(usize, RationalPoly)> = Vec::new();
        if degree.is_multiple_of(2) {
            let m = degree / 2;
            // sigma0: SoS of degree 2m, Gram side m+1, multiplier 1.
            blocks.push((m + 1, one));
            // sigma1: SoS of degree 2m-2, Gram side m, multiplier (x-a)(b-x).
            // (Only present when m >= 1; a degree-0 constant goal has just
            // sigma0 of side 1.)
            if m >= 1 {
                blocks.push((m, interval));
            }
        } else {
            let m = (degree - 1) / 2;
            // sigma0 * (x - a) and sigma1 * (b - x), each SoS degree 2m, side m+1.
            blocks.push((m + 1, x_minus_a));
            blocks.push((m + 1, b_minus_x));
        }

        let mut specs = Vec::with_capacity(blocks.len());
        let mut offset = 0;
        for (size, multiplier) in blocks {
            specs.push(BlockSpec {
                size,
                multiplier,
                var_offset: offset,
            });
            offset += upper_tri_count(size);
        }
        Self {
            degree,
            blocks: specs,
            n_vars: offset,
        }
    }

    /// The flat variable index of upper-triangle entry `(i, j)` (`i <= j`) in
    /// block `b`.
    pub fn var_index(&self, b: usize, i: usize, j: usize) -> usize {
        let block = &self.blocks[b];
        block.var_offset + upper_tri_index(block.size, i, j)
    }

    /// Build the coefficient-matching linear system as an augmented matrix
    /// `[A | p]` with `n_vars + 1` columns and `degree + 1` rows (one per output
    /// monomial `x^0 .. x^degree`).
    ///
    /// Row `k` encodes: `sum over all blocks and upper-triangle entries of
    /// (that entry's contribution to monomial x^k) = p_k`. Entry `(i, j)` of
    /// block `b` (variable `u`) contributes `c_{ij}` to monomial `(i + j) + d`
    /// for each term `mult_b[d] x^d` of the block multiplier, where `c_{ij}` is
    /// `1` on the diagonal and `2` off it.
    pub fn coeff_matching_system(&self, goal_poly: &RationalPoly) -> RatMatrix {
        let n_rows = self.degree + 1;
        let n_cols = self.n_vars + 1;
        let mut aug = RatMatrix::zeros(n_rows, n_cols);

        for (b, block) in self.blocks.iter().enumerate() {
            let mult = &block.multiplier;
            for i in 0..block.size {
                for j in i..block.size {
                    let var = self.var_index(b, i, j);
                    let sym_coeff = if i == j { 1 } else { 2 };
                    let base_deg = i + j;
                    // For each multiplier term mult[d] * x^d, the variable lands
                    // on monomial base_deg + d with coefficient sym_coeff*mult[d].
                    for d in 0..mult.coeffs().len() {
                        let md = mult.coeff(d);
                        if md.is_zero() {
                            continue;
                        }
                        let k = base_deg + d;
                        if k >= n_rows {
                            // Should not happen for a well-formed layout (the SoS
                            // degree never exceeds `degree`), but guard anyway.
                            continue;
                        }
                        let add = BigRational::from(BigInt::from(sym_coeff)) * &md;
                        let cur = aug.at(k, var).clone();
                        aug.set(k, var, cur + add);
                    }
                }
            }
        }
        // Augmented column = the goal polynomial's coefficients.
        for k in 0..n_rows {
            aug.set(k, self.n_vars, goal_poly.coeff(k));
        }
        aug
    }

    /// Reassemble a [`SosCertificate`] from a flat variable vector (the
    /// upper-triangle Gram entries), the goal polynomial, and the interval.
    /// Mirrors each upper-triangle entry into the full symmetric Gram.
    pub fn certificate_from_vars(
        &self,
        vars: &[BigRational],
        goal_poly: &RationalPoly,
        lo: &BigRational,
        hi: &BigRational,
    ) -> SosCertificate {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (b, block) in self.blocks.iter().enumerate() {
            let s = block.size;
            let mut data = vec![BigRational::zero(); s * s];
            for i in 0..s {
                for j in i..s {
                    let v = vars[self.var_index(b, i, j)].clone();
                    data[i * s + j] = v.clone();
                    data[j * s + i] = v;
                }
            }
            blocks.push(SosBlock {
                gram: RatSymMatrix::from_rows(s, data),
                multiplier: block.multiplier.clone(),
            });
        }
        SosCertificate {
            poly: goal_poly.clone(),
            lo: lo.clone(),
            hi: hi.clone(),
            blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::linalg::{project_onto_affine, rref};
    use super::*;

    fn ri(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    // --- structural layout ---

    #[test]
    fn even_degree_two_has_sigma0_side2_and_sigma1_side1() {
        // n = 2 (m = 1): sigma0 side 2 (mult 1), sigma1 side 1 (mult (x-a)(b-x)).
        let layout = SosLayout::for_goal(2, &ri(0), &ri(1));
        assert_eq!(layout.blocks.len(), 2);
        assert_eq!(layout.blocks[0].size, 2);
        assert_eq!(
            layout.blocks[0].multiplier,
            RationalPoly::from_int_coeffs(&[1])
        );
        assert_eq!(layout.blocks[1].size, 1);
        // (x-0)(1-x) = x - x^2.
        assert_eq!(
            layout.blocks[1].multiplier,
            RationalPoly::from_int_coeffs(&[0, 1, -1])
        );
        // vars: side-2 upper tri (3) + side-1 (1) = 4.
        assert_eq!(layout.n_vars, 4);
    }

    #[test]
    fn odd_degree_three_has_two_side2_blocks() {
        // n = 3 (m = 1): (x-a)*sigma0 + (b-x)*sigma1, each side 2.
        let layout = SosLayout::for_goal(3, &ri(0), &ri(2));
        assert_eq!(layout.blocks.len(), 2);
        assert_eq!(layout.blocks[0].size, 2);
        assert_eq!(layout.blocks[1].size, 2);
        // (x - 0) = x ; (2 - x).
        assert_eq!(
            layout.blocks[0].multiplier,
            RationalPoly::from_int_coeffs(&[0, 1])
        );
        assert_eq!(
            layout.blocks[1].multiplier,
            RationalPoly::from_int_coeffs(&[2, -1])
        );
        assert_eq!(layout.n_vars, 6); // 3 + 3
    }

    #[test]
    fn degree_zero_has_single_side1_block() {
        // A constant nonnegative goal: just sigma0 side 1, multiplier 1.
        let layout = SosLayout::for_goal(0, &ri(0), &ri(1));
        assert_eq!(layout.blocks.len(), 1);
        assert_eq!(layout.blocks[0].size, 1);
        assert_eq!(layout.n_vars, 1);
    }

    #[test]
    fn upper_tri_index_is_dense_and_ordered() {
        // side 3: (0,0)=0 (0,1)=1 (0,2)=2 (1,1)=3 (1,2)=4 (2,2)=5.
        assert_eq!(upper_tri_index(3, 0, 0), 0);
        assert_eq!(upper_tri_index(3, 0, 1), 1);
        assert_eq!(upper_tri_index(3, 0, 2), 2);
        assert_eq!(upper_tri_index(3, 1, 1), 3);
        assert_eq!(upper_tri_index(3, 1, 2), 4);
        assert_eq!(upper_tri_index(3, 2, 2), 5);
        assert_eq!(upper_tri_count(3), 6);
    }

    // --- coefficient-matching system ---

    #[test]
    fn coeff_matching_for_x_squared_solves_to_a_valid_certificate() {
        // p = x^2 on [-1, 1], degree 2. The system must have a solution whose
        // reassembled certificate VERIFIES (the round-trip: encode -> solve ->
        // certificate_from_vars -> verify).
        let lo = ri(-1);
        let hi = ri(1);
        let p = RationalPoly::from_int_coeffs(&[0, 0, 1]); // x^2
        let layout = SosLayout::for_goal(2, &lo, &hi);
        let aug = layout.coeff_matching_system(&p);
        // A particular point in the solution set: sigma0 = x^2 (Gram diag(0,1)),
        // sigma1 = 0. Upper-tri vars for side-2 block: [u00, u01, u11] then the
        // side-1 block [w00]. We want u00=0, u01=0, u11=1, w00=0.
        let x0 = vec![ri(0), ri(0), ri(1), ri(0)];
        let x = project_onto_affine(&aug, layout.n_vars, &x0).expect("consistent");
        let cert = layout.certificate_from_vars(&x, &p, &lo, &hi);
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn coeff_matching_for_interval_positive_poly_round_trips() {
        // p = x - x^2 = x(1-x) >= 0 on [0,1], degree 2. A known decomposition:
        // sigma0 = 0, sigma1 = 1 (constant SoS), multiplier (x)(1-x) = x - x^2.
        // So u00=u01=u11=0 (sigma0 zero) and w00=1 (sigma1 = 1).
        let lo = ri(0);
        let hi = ri(1);
        let p = RationalPoly::from_int_coeffs(&[0, 1, -1]); // x - x^2
        let layout = SosLayout::for_goal(2, &lo, &hi);
        let aug = layout.coeff_matching_system(&p);
        let x0 = vec![ri(0), ri(0), ri(0), ri(1)];
        let x = project_onto_affine(&aug, layout.n_vars, &x0).expect("consistent");
        let cert = layout.certificate_from_vars(&x, &p, &lo, &hi);
        assert_eq!(cert.verify(), Ok(()));
    }

    #[test]
    fn coeff_matching_is_consistent_for_a_feasible_goal() {
        // The system for a feasible goal (x^2 + 1 on [-2, 2]) is consistent.
        let lo = ri(-2);
        let hi = ri(2);
        let p = RationalPoly::from_int_coeffs(&[1, 0, 1]); // 1 + x^2
        let layout = SosLayout::for_goal(2, &lo, &hi);
        let aug = layout.coeff_matching_system(&p);
        let r = rref(&aug, layout.n_vars);
        assert!(!r.is_inconsistent());
    }

    #[test]
    fn certificate_from_vars_mirrors_upper_triangle_symmetrically() {
        // side-2 block with u00=1, u01=2, u11=3 -> Gram [[1,2],[2,3]].
        let layout = SosLayout::for_goal(2, &ri(0), &ri(1));
        // vars: [u00, u01, u11, w00].
        let vars = vec![ri(1), ri(2), ri(3), ri(0)];
        let p = RationalPoly::from_int_coeffs(&[0, 0, 1]);
        let cert = layout.certificate_from_vars(&vars, &p, &ri(0), &ri(1));
        let g = &cert.blocks[0].gram;
        assert_eq!(*g.entry(0, 0), ri(1));
        assert_eq!(*g.entry(0, 1), ri(2));
        assert_eq!(*g.entry(1, 0), ri(2)); // mirrored
        assert_eq!(*g.entry(1, 1), ri(3));
        assert!(g.is_symmetric());
    }

    #[test]
    fn round_trip_odd_degree_certificate_verifies() {
        // p = x on [0, 4], degree 1 (odd). Markov-Lukacs odd form:
        // p = (x-0)*sigma0 + (4-x)*sigma1, sigma0/sigma1 side 1 (degree 0 SoS,
        // i.e. nonneg constants). x = (x)(s0) + (4-x)(s1) = (s0-s1) x + 4 s1.
        // Match: 4 s1 = 0 -> s1 = 0; s0 - s1 = 1 -> s0 = 1. Both >= 0: PSD. Good.
        let lo = ri(0);
        let hi = ri(4);
        let p = RationalPoly::from_int_coeffs(&[0, 1]); // x
        let layout = SosLayout::for_goal(1, &lo, &hi);
        // vars: side-1 sigma0 [s0], side-1 sigma1 [s1].
        let aug = layout.coeff_matching_system(&p);
        let x0 = vec![ri(1), ri(0)];
        let x = project_onto_affine(&aug, layout.n_vars, &x0).expect("consistent");
        let cert = layout.certificate_from_vars(&x, &p, &lo, &hi);
        assert_eq!(cert.verify(), Ok(()));
    }
}
