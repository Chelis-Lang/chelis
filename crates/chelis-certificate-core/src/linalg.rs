//! Exact row reduction and affine repair.
//! Repair establishes an affine identity, not PSD or proof admission.

use alloc::{vec, vec::Vec};

use num_rational::BigRational;
use num_traits::Zero;

#[derive(Debug, Clone, PartialEq, Eq)]
/// A dense rational matrix with row-major storage.
pub struct RatMatrix {
    rows: usize,
    cols: usize,
    data: Vec<BigRational>,
}

impl RatMatrix {
    /// Construct a matrix. The entry count must equal `rows * cols`.
    pub fn from_data(rows: usize, cols: usize, data: Vec<BigRational>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "RatMatrix expects rows*cols entries"
        );
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![BigRational::zero(); rows * cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn at(&self, i: usize, j: usize) -> &BigRational {
        &self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: BigRational) {
        self.data[i * self.cols + j] = v;
    }
}

#[derive(Debug, Clone)]
/// A reduced augmented system with deterministic pivot order.
pub struct Rref {
    pub reduced: RatMatrix,
    pub n_vars: usize,
    pub pivots: Vec<(usize, usize)>,
}

impl Rref {
    pub fn is_inconsistent(&self) -> bool {
        let last = self.n_vars;
        for i in 0..self.reduced.rows() {
            let mut all_zero_vars = true;
            for j in 0..self.n_vars {
                if !self.reduced.at(i, j).is_zero() {
                    all_zero_vars = false;
                    break;
                }
            }
            if all_zero_vars && !self.reduced.at(i, last).is_zero() {
                return true;
            }
        }
        false
    }

    /// Return non-pivot variable columns in ascending order.
    pub fn free_columns(&self) -> Vec<usize> {
        let pivot_cols: alloc::collections::BTreeSet<usize> =
            self.pivots.iter().map(|&(_, c)| c).collect();
        (0..self.n_vars)
            .filter(|c| !pivot_cols.contains(c))
            .collect()
    }
}

/// Reduce an augmented system with exact rational arithmetic.
/// The matrix must contain `n_vars + 1` columns.
/// Each pivot uses the first nonzero row at or below the current row.
pub fn rref(augmented: &RatMatrix, n_vars: usize) -> Rref {
    assert_eq!(
        augmented.cols(),
        n_vars + 1,
        "augmented matrix must have n_vars + 1 columns"
    );
    let mut m = augmented.clone();
    let rows = m.rows();
    let total_cols = m.cols();
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut pivot_row = 0;

    for col in 0..n_vars {
        if pivot_row >= rows {
            break;
        }
        let Some(sel) = (pivot_row..rows).find(|&r| !m.at(r, col).is_zero()) else {
            continue;
        };
        if sel != pivot_row {
            for j in 0..total_cols {
                let a = m.at(pivot_row, j).clone();
                let b = m.at(sel, j).clone();
                m.set(pivot_row, j, b);
                m.set(sel, j, a);
            }
        }
        let pivot_val = m.at(pivot_row, col).clone();
        for j in 0..total_cols {
            let v = m.at(pivot_row, j) / &pivot_val;
            m.set(pivot_row, j, v);
        }
        for r in 0..rows {
            if r == pivot_row {
                continue;
            }
            let factor = m.at(r, col).clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..total_cols {
                let v = m.at(r, j) - &factor * m.at(pivot_row, j);
                m.set(r, j, v);
            }
        }
        pivots.push((pivot_row, col));
        pivot_row += 1;
    }

    Rref {
        reduced: m,
        n_vars,
        pivots,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// An exact affine repair error.
pub enum ProjectError {
    Inconsistent,
}

/// Preserve free coordinates and solve pivot coordinates exactly.
/// The candidate must contain `n_vars` entries.
/// The augmented matrix must contain `n_vars + 1` columns.
/// An inconsistent system returns `ProjectError::Inconsistent`.
/// Success does not establish PSD or proof admission.
pub fn project_onto_affine(
    augmented: &RatMatrix,
    n_vars: usize,
    x0: &[BigRational],
) -> Result<Vec<BigRational>, ProjectError> {
    assert_eq!(x0.len(), n_vars, "x0 must have n_vars entries");
    let r = rref(augmented, n_vars);
    if r.is_inconsistent() {
        return Err(ProjectError::Inconsistent);
    }
    let mut x = x0.to_vec();
    let aug_col = n_vars;
    for &(row, pcol) in &r.pivots {
        let mut val = r.reduced.at(row, aug_col).clone();
        for fcol in r.free_columns() {
            let coeff = r.reduced.at(row, fcol);
            if !coeff.is_zero() {
                val -= coeff * &x[fcol];
            }
        }
        x[pcol] = val;
    }
    Ok(x)
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

    fn mat(rows: usize, cols: usize, vals: &[i64]) -> RatMatrix {
        RatMatrix::from_data(rows, cols, vals.iter().map(|&v| ri(v)).collect())
    }

    fn satisfies(augmented: &RatMatrix, n_vars: usize, x: &[BigRational]) -> bool {
        for i in 0..augmented.rows() {
            let mut lhs = BigRational::zero();
            for (j, xj) in x.iter().enumerate().take(n_vars) {
                lhs += augmented.at(i, j) * xj;
            }
            if lhs != *augmented.at(i, n_vars) {
                return false;
            }
        }
        true
    }

    #[test]
    fn rref_solves_a_unique_2x2_system() {
        let aug = mat(2, 3, &[1, 1, 3, 1, -1, 1]);
        let r = rref(&aug, 2);
        assert!(!r.is_inconsistent());
        assert_eq!(r.pivots.len(), 2);
        assert!(r.free_columns().is_empty());
        assert_eq!(*r.reduced.at(0, 2), ri(2));
        assert_eq!(*r.reduced.at(1, 2), ri(1));
    }

    #[test]
    fn rref_detects_inconsistency() {
        let aug = mat(2, 3, &[1, 1, 1, 1, 1, 2]);
        let r = rref(&aug, 2);
        assert!(r.is_inconsistent());
    }

    #[test]
    fn rref_identifies_free_columns_in_underdetermined_system() {
        let aug = mat(1, 4, &[1, 1, 1, 1]);
        let r = rref(&aug, 3);
        assert!(!r.is_inconsistent());
        assert_eq!(r.pivots.len(), 1);
        assert_eq!(r.free_columns(), vec![1, 2]);
    }

    #[test]
    fn project_keeps_free_coords_and_solves_pivots_exactly() {
        let aug = mat(1, 4, &[1, 1, 1, 1]);
        let x0 = vec![ri(9), r(1, 2), r(1, 4)];
        let x = project_onto_affine(&aug, 3, &x0).expect("consistent");
        assert_eq!(x[0], r(1, 4));
        assert_eq!(x[1], r(1, 2));
        assert_eq!(x[2], r(1, 4));
        assert!(satisfies(&aug, 3, &x));
    }

    #[test]
    fn project_onto_unique_solution_ignores_x0() {
        let aug = mat(2, 3, &[1, 1, 3, 1, -1, 1]);
        let x0 = vec![ri(100), ri(-100)];
        let x = project_onto_affine(&aug, 2, &x0).expect("consistent");
        assert_eq!(x[0], ri(2));
        assert_eq!(x[1], ri(1));
        assert!(satisfies(&aug, 2, &x));
    }

    #[test]
    fn project_result_satisfies_system_exactly_with_rational_x0() {
        let aug = mat(1, 3, &[2, 4, 6]);
        let x0 = vec![ri(0), r(7, 3)];
        let x = project_onto_affine(&aug, 2, &x0).expect("consistent");
        assert_eq!(x[1], r(7, 3));
        assert_eq!(x[0], r(-5, 3));
        assert!(satisfies(&aug, 2, &x));
    }

    #[test]
    fn project_rejects_inconsistent_system() {
        let aug = mat(2, 3, &[1, 1, 1, 1, 1, 2]);
        let x0 = vec![ri(0), ri(0)];
        assert_eq!(
            project_onto_affine(&aug, 2, &x0),
            Err(ProjectError::Inconsistent)
        );
    }

    #[test]
    fn project_handles_multiple_pivots_and_frees() {
        let aug = mat(2, 4, &[1, 0, 1, 2, 0, 1, 1, 3]);
        let x0 = vec![ri(0), ri(0), r(1, 2)];
        let x = project_onto_affine(&aug, 3, &x0).expect("consistent");
        assert_eq!(x[0], r(3, 2));
        assert_eq!(x[1], r(5, 2));
        assert_eq!(x[2], r(1, 2));
        assert!(satisfies(&aug, 3, &x));
    }
}
