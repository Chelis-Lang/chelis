//! Checked ordered-axis plans shared by the checker and IR.

#[cfg(kani)]
mod kani_harnesses;
mod verified;
pub use verified::{checked_inverse, is_permutation, normalize_axis, reduction_survivors};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermutationError {
    Arity { rank: usize, got: usize },
    NegativeAxis(i64),
    UnrepresentableAxis(i64),
    OutOfBounds(usize),
    Duplicate(usize),
}

/// A full, checked permutation of the positions `0..rank`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Permutation {
    axes: Vec<usize>,
}

impl Permutation {
    pub fn from_raw(rank: usize, raw: &[i64]) -> Result<Self, PermutationError> {
        if raw.len() != rank {
            return Err(PermutationError::Arity {
                rank,
                got: raw.len(),
            });
        }
        let mut axes = Vec::with_capacity(rank);
        let mut seen = vec![false; rank];
        for &axis in raw {
            if axis < 0 {
                return Err(PermutationError::NegativeAxis(axis));
            }
            let axis =
                usize::try_from(axis).map_err(|_| PermutationError::UnrepresentableAxis(axis))?;
            if axis >= rank {
                return Err(PermutationError::OutOfBounds(axis));
            }
            if seen[axis] {
                return Err(PermutationError::Duplicate(axis));
            }
            seen[axis] = true;
            axes.push(axis);
        }
        Self::from_indices(rank, &axes)
    }

    pub fn from_indices(rank: usize, axes: &[usize]) -> Result<Self, PermutationError> {
        if axes.len() != rank {
            return Err(PermutationError::Arity {
                rank,
                got: axes.len(),
            });
        }
        if is_permutation(axes, rank) {
            return Ok(Self {
                axes: axes.to_vec(),
            });
        }
        let mut seen = vec![false; rank];
        for &axis in axes {
            if axis >= rank {
                return Err(PermutationError::OutOfBounds(axis));
            }
            if seen[axis] {
                return Err(PermutationError::Duplicate(axis));
            }
            seen[axis] = true;
        }
        unreachable!("the verified permutation gate rejected an in-range unique list")
    }

    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    /// The checked constructor makes every indexed write valid and unique.
    pub fn inverse(&self) -> Vec<usize> {
        checked_inverse(&self.axes, self.axes.len()).expect("checked permutation remains valid")
    }
}
