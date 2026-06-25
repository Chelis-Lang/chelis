//! Exact rational dense linear algebra for the SoS encoding and the
//! Peyrl-Parrilo projection.
//!
//! Two operations, both pure `BigRational` (no float, no tolerance):
//!
//! - [`rref`]: reduced row-echelon form of an augmented system, recording the
//!   pivot columns. The coefficient-matching constraints `A x = b` (the SoS
//!   "the blocks' combined quadratic form equals `p`" identity, as a linear
//!   system in the Gram entries) are solved with this.
//! - [`project_onto_affine`]: the Peyrl-Parrilo exact projection. Given the
//!   coefficient-matching system `A x = b` and a candidate point `x0` (the
//!   float SDP solution, rounded to rationals), produce an EXACT rational point
//!   that satisfies `A x = b` exactly, staying as close to `x0` as the
//!   free-variable assignment allows: keep `x0`'s value on every FREE
//!   (non-pivot) coordinate and solve exactly for the pivot coordinates by
//!   back-substitution. The result satisfies `A x = b` by construction, so the
//!   downstream exact verifier's polynomial-identity check passes; only the PSD
//!   check can then fail (the boundary-degeneracy case).

use num_rational::BigRational;
use num_traits::Zero;

/// A dense matrix of exact rationals, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatMatrix {
    rows: usize,
    cols: usize,
    data: Vec<BigRational>,
}

impl RatMatrix {
    /// Build from a row-major entry vector. Panics if the length is not
    /// `rows * cols`.
    pub fn from_data(rows: usize, cols: usize, data: Vec<BigRational>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "RatMatrix expects rows*cols entries"
        );
        Self { rows, cols, data }
    }

    /// A `rows x cols` zero matrix.
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

/// The result of reducing an augmented system `[A | b]` to reduced row-echelon
/// form.
#[derive(Debug, Clone)]
pub struct Rref {
    /// The reduced augmented matrix `[R | c]` with `n_vars + 1` columns.
    pub reduced: RatMatrix,
    /// The number of variable columns (`A` had `n_vars` columns; the last
    /// column of `reduced` is the augmented `b`/`c`).
    pub n_vars: usize,
    /// For each pivot, `(row, var_column)`. Sorted by row.
    pub pivots: Vec<(usize, usize)>,
}

impl Rref {
    /// Whether the system is INCONSISTENT (a pivot in the augmented column: a
    /// `0 = nonzero` row). An inconsistent coefficient-matching system means no
    /// SoS decomposition of the chosen degree structure reproduces `p`.
    pub fn is_inconsistent(&self) -> bool {
        // A row that is all-zero in the variable columns but non-zero in the
        // augmented column is `0 = c != 0`.
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

    /// The free (non-pivot) variable columns, ascending.
    pub fn free_columns(&self) -> Vec<usize> {
        let pivot_cols: std::collections::BTreeSet<usize> =
            self.pivots.iter().map(|&(_, c)| c).collect();
        (0..self.n_vars)
            .filter(|c| !pivot_cols.contains(c))
            .collect()
    }
}

/// Reduce the augmented system `[A | b]` (an `m x (n+1)` matrix; the last column
/// is `b`) to reduced row-echelon form over the rationals, recording pivots.
/// Exact: no tolerance, exact pivot selection (first non-zero in the column).
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
        // Find a row at or below pivot_row with a non-zero entry in this column.
        let Some(sel) = (pivot_row..rows).find(|&r| !m.at(r, col).is_zero()) else {
            continue;
        };
        // Swap into the pivot row.
        if sel != pivot_row {
            for j in 0..total_cols {
                let a = m.at(pivot_row, j).clone();
                let b = m.at(sel, j).clone();
                m.set(pivot_row, j, b);
                m.set(sel, j, a);
            }
        }
        // Normalize the pivot row so the pivot is 1.
        let pivot_val = m.at(pivot_row, col).clone();
        for j in 0..total_cols {
            let v = m.at(pivot_row, j) / &pivot_val;
            m.set(pivot_row, j, v);
        }
        // Eliminate this column from every other row.
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

/// Why an exact affine projection failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectError {
    /// The coefficient-matching system `A x = b` is inconsistent: no point
    /// satisfies it, so no SoS decomposition of the chosen structure reproduces
    /// `p`.
    Inconsistent,
}

/// The Peyrl-Parrilo exact projection. Given the coefficient-matching system
/// `A x = b` (as the augmented matrix `[A | b]`) and a candidate point `x0`
/// (length `n_vars`, the rounded float solution), return an EXACT rational point
/// `x` that satisfies `A x = b` exactly: `x` keeps `x0`'s value on every FREE
/// (non-pivot) coordinate and solves exactly for each pivot coordinate by
/// back-substitution. `Err(Inconsistent)` if no point satisfies the system.
///
/// The returned point satisfies `A x = b` EXACTLY (the polynomial identity holds
/// exactly), so the downstream exact verifier's identity check passes by
/// construction; only the PSD check can then fail (boundary degeneracy). This is
/// the projection step the SoS rational repair runs before verifying.
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
    // Start from x0 on every coordinate; the free coordinates keep their x0
    // value, and we overwrite each pivot coordinate by its reduced-row equation.
    let mut x = x0.to_vec();
    let aug_col = n_vars;
    // In RREF, each pivot row reads:  x[pivot_col] + sum_{free} R[row][free] * x[free] = c[row]
    // so  x[pivot_col] = c[row] - sum_{free} R[row][free] * x[free].
    // The free x values are x0's (already in `x`); reference them, not the
    // pivot ones (a reduced pivot row has zero in every OTHER pivot column).
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

    /// Check `A x = b` exactly for an augmented `[A | b]`.
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
        // x + y = 3 ; x - y = 1  ->  x = 2, y = 1.
        let aug = mat(2, 3, &[1, 1, 3, 1, -1, 1]);
        let r = rref(&aug, 2);
        assert!(!r.is_inconsistent());
        assert_eq!(r.pivots.len(), 2);
        assert!(r.free_columns().is_empty());
        // Reduced augmented column holds the unique solution.
        assert_eq!(*r.reduced.at(0, 2), ri(2));
        assert_eq!(*r.reduced.at(1, 2), ri(1));
    }

    #[test]
    fn rref_detects_inconsistency() {
        // x + y = 1 ; x + y = 2  -> inconsistent.
        let aug = mat(2, 3, &[1, 1, 1, 1, 1, 2]);
        let r = rref(&aug, 2);
        assert!(r.is_inconsistent());
    }

    #[test]
    fn rref_identifies_free_columns_in_underdetermined_system() {
        // x + y + z = 1 (one equation, three vars) -> one pivot, two free.
        let aug = mat(1, 4, &[1, 1, 1, 1]);
        let r = rref(&aug, 3);
        assert!(!r.is_inconsistent());
        assert_eq!(r.pivots.len(), 1);
        assert_eq!(r.free_columns(), vec![1, 2]);
    }

    #[test]
    fn project_keeps_free_coords_and_solves_pivots_exactly() {
        // x + y + z = 1, free y, z. With x0 = [9, 1/2, 1/4] the projection keeps
        // y = 1/2, z = 1/4 and solves x = 1 - 1/2 - 1/4 = 1/4.
        let aug = mat(1, 4, &[1, 1, 1, 1]);
        let x0 = vec![ri(9), r(1, 2), r(1, 4)];
        let x = project_onto_affine(&aug, 3, &x0).expect("consistent");
        assert_eq!(x[0], r(1, 4));
        assert_eq!(x[1], r(1, 2)); // free, unchanged
        assert_eq!(x[2], r(1, 4)); // free, unchanged
        assert!(satisfies(&aug, 3, &x));
    }

    #[test]
    fn project_onto_unique_solution_ignores_x0() {
        // Unique solution (no free vars): x = 2, y = 1 regardless of x0.
        let aug = mat(2, 3, &[1, 1, 3, 1, -1, 1]);
        let x0 = vec![ri(100), ri(-100)];
        let x = project_onto_affine(&aug, 2, &x0).expect("consistent");
        assert_eq!(x[0], ri(2));
        assert_eq!(x[1], ri(1));
        assert!(satisfies(&aug, 2, &x));
    }

    #[test]
    fn project_result_satisfies_system_exactly_with_rational_x0() {
        // 2x + 4y = 6, free y. x0 = [0, 7/3]: keep y = 7/3, solve
        // x = (6 - 4*7/3)/2 = (6 - 28/3)/2 = (-10/3)/2 = -5/3.
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
        // x + z = 2 ; y + z = 3, free z. x0 = [0,0, 1/2]:
        // keep z = 1/2, x = 2 - 1/2 = 3/2, y = 3 - 1/2 = 5/2.
        let aug = mat(2, 4, &[1, 0, 1, 2, 0, 1, 1, 3]);
        let x0 = vec![ri(0), ri(0), r(1, 2)];
        let x = project_onto_affine(&aug, 3, &x0).expect("consistent");
        assert_eq!(x[0], r(3, 2));
        assert_eq!(x[1], r(5, 2));
        assert_eq!(x[2], r(1, 2));
        assert!(satisfies(&aug, 3, &x));
    }
}
