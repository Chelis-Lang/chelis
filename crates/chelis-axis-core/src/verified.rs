//! Executable axis functions and their Verus contracts.

use vstd::prelude::*;

verus! {
    pub open spec fn valid_permutation(axes: Seq<usize>, rank: usize) -> bool {
        axes.len() == rank as int
            && (forall|i: int| 0 <= i < rank as int ==> axes[i] < rank)
            && (forall|i: int, j: int| 0 <= i < j < rank as int ==> axes[i] != axes[j])
    }

    pub open spec fn normalized_axis_model(rank: usize, raw: i64) -> Option<usize> {
        if rank as int > i64::MAX as int {
            None
        } else {
            let position = if raw < 0 { raw as int + rank as int } else { raw as int };
            if 0 <= position && position < rank as int {
                Some(position as usize)
            } else {
                None
            }
        }
    }

    /// The executable admission gate for a full axis permutation. Its proof
    /// establishes that every accepted position is present exactly once.
    pub fn is_permutation(axes: &[usize], rank: usize) -> (valid: bool)
        ensures valid == valid_permutation(axes@, rank),
    {
        if axes.len() != rank {
            return false;
        }
        let mut i = 0;
        while i < rank
            invariant
                i <= rank,
                axes.len() == rank,
                forall|k: int| 0 <= k < i as int ==> axes@[k] < rank,
                forall|a: int, b: int| 0 <= a < b < i as int ==> axes@[a] != axes@[b],
            decreases rank - i,
        {
            if axes[i] >= rank {
                return false;
            }
            let mut j = 0;
            while j < i
                invariant
                    j <= i,
                    i < rank,
                    axes.len() == rank,
                    axes@[i as int] < rank,
                    forall|k: int| 0 <= k < i as int ==> axes@[k] < rank,
                    forall|a: int, b: int| 0 <= a < b < i as int ==> axes@[a] != axes@[b],
                    forall|k: int| 0 <= k < j as int ==> axes@[k] != axes@[i as int],
                decreases i - j,
            {
                if axes[j] == axes[i] {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }
        true
    }

    /// Resolve a signed axis against a rank. A rank outside the signed axis
    /// domain has no representable from-the-end position and is rejected.
    pub fn normalize_axis(rank: usize, raw: i64) -> (resolved: Option<usize>)
        ensures
            resolved == normalized_axis_model(rank, raw),
            match resolved { Some(axis) => axis < rank, None => true },
    {
        if (rank as u128) > (i64::MAX as u128) {
            return None;
        }
        let signed_rank = rank as i64;
        if raw < 0 {
            if raw < -signed_rank {
                return None;
            }
            return Some((raw + signed_rank) as usize);
        }
        if raw >= signed_rank {
            None
        } else {
            Some(raw as usize)
        }
    }

    pub open spec fn survivor_at(axis: usize, output_index: int) -> int {
        if output_index < axis as int { output_index } else { output_index + 1 }
    }

    /// Positions retained by removing one checked reduction axis.
    pub fn reduction_survivors(rank: usize, axis: usize) -> (result: Option<Vec<usize>>)
        ensures
            match result {
                None => axis >= rank,
                Some(positions) => {
                    axis < rank
                        && positions.len() == rank - 1
                        && (forall|k: int| 0 <= k < positions@.len()
                            ==> positions@[k] as int == survivor_at(axis, k))
                },
            },
    {
        if axis >= rank {
            return None;
        }
        let mut positions = Vec::new();
        let mut i = 0;
        while i < rank
            invariant
                axis < rank,
                i <= rank,
                positions@.len() == (if i <= axis { i as int } else { i as int - 1 }),
                forall|k: int| 0 <= k < positions@.len()
                    ==> positions@[k] as int == survivor_at(axis, k),
            decreases rank - i,
        {
            if i != axis {
                positions.push(i);
            }
            i += 1;
        }
        Some(positions)
    }

    /// Construct the inverse only for a full permutation; the output maps
    /// every original position back to its position in `axes`.
    pub fn checked_inverse(axes: &[usize], rank: usize) -> (result: Option<Vec<usize>>)
        ensures
            match result {
                None => !valid_permutation(axes@, rank),
                Some(inverse) => valid_permutation(axes@, rank)
                    && inverse.len() == rank
                    && (forall|i: int| 0 <= i < rank as int
                        ==> inverse@[axes@[i] as int] == i as usize),
            },
    {
        if !is_permutation(axes, rank) {
            return None;
        }
        let mut inverse = Vec::new();
        let mut fill = 0;
        while fill < rank
            invariant fill <= rank, inverse.len() == fill,
            decreases rank - fill,
        {
            inverse.push(0);
            fill += 1;
        }
        let mut i = 0;
        while i < rank
            invariant
                i <= rank,
                valid_permutation(axes@, rank),
                inverse.len() == rank,
                forall|k: int| 0 <= k < i as int
                    ==> inverse@[axes@[k] as int] == k as usize,
            decreases rank - i,
        {
            inverse[axes[i]] = i;
            i += 1;
        }
        Some(inverse)
    }
}
