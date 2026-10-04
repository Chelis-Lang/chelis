//! The text of language failures that every lane reports identically.
//!
//! `chelis eval`'s interpreters and the compiled runtime all render these
//! failures through this module, so the lanes cannot drift. A trap ends in
//! the [04-NUM-9] line `numeric trap: domain in <op> at i64`, whose `<prim>`
//! slot is `i64` because the guard finalizes an extent or an index
//! (spec/04-type-system.md section 4.7); the context line before it names
//! the disagreeing sources and the values observed.

fn domain_trap_line_at_i64(op: &str) -> String {
    format!("numeric trap: domain in {op} at i64")
}

fn render_shape(shape: &[i64]) -> String {
    let extents = shape.iter().map(i64::to_string).collect::<Vec<_>>();
    format!("[{}]", extents.join(", "))
}

/// Two operands of `op` whose shapes must be identical disagree at run time
/// (spec/04-type-system.md section 4.7): a `Domain` trap in `op`, after a
/// context line naming each operand's shape and the first disagreeing axis,
/// or the ranks when those differ.
pub fn operand_shape_disagreement(op: &str, lhs: &[i64], rhs: &[i64]) -> String {
    let (lhs_text, rhs_text) = (render_shape(lhs), render_shape(rhs));
    let disagreeing_axis = lhs.iter().zip(rhs).position(|(left, right)| left != right);
    let context = match disagreeing_axis.filter(|_| lhs.len() == rhs.len()) {
        Some(axis) => format!(
            "{op} operands disagree at axis {axis}: lhs {lhs_text} has {}, rhs {rhs_text} has {}",
            lhs[axis], rhs[axis]
        ),
        None => format!(
            "{op} operands disagree in rank: lhs {lhs_text} has rank {}, rhs {rhs_text} has rank {}",
            lhs.len(),
            rhs.len()
        ),
    };
    format!("{context}\n{}", domain_trap_line_at_i64(op))
}

/// `matmul(lhs, rhs)`'s operands disagree at run time: the shared axis (the
/// last axis of `lhs` against the second-to-last of `rhs`) differs, or a
/// batch axis differs where neither extent is 1. spec/04-type-system.md
/// section 4.7 makes it a `Domain` trap in `matmul`, after a context line
/// naming both operands as written. `None` when the operands agree.
pub fn matmul_operand_disagreement(lhs: &[i64], rhs: &[i64]) -> Option<String> {
    let (lhs_rank, rhs_rank) = (lhs.len(), rhs.len());
    if lhs_rank < 2 || rhs_rank < 2 {
        return None;
    }
    let (lhs_text, rhs_text) = (render_shape(lhs), render_shape(rhs));
    let (lhs_shared, rhs_shared) = (lhs_rank - 1, rhs_rank - 2);
    let context = if lhs[lhs_shared] != rhs[rhs_shared] {
        format!(
            "matmul shared axis disagrees: lhs {lhs_text} has {} at axis {lhs_shared}, \
             rhs {rhs_text} has {} at axis {rhs_shared}",
            lhs[lhs_shared], rhs[rhs_shared]
        )
    } else {
        // Batch axes align from the right, and an extent of 1 broadcasts.
        let batch = (1..=(lhs_rank - 2).min(rhs_rank - 2)).find_map(|offset| {
            let (left_axis, right_axis) = (lhs_rank - 2 - offset, rhs_rank - 2 - offset);
            let (left, right) = (lhs[left_axis], rhs[right_axis]);
            (left != right && left != 1 && right != 1).then_some((left_axis, right_axis))
        })?;
        format!(
            "matmul batch axis disagrees: lhs {lhs_text} has {} at axis {}, \
             rhs {rhs_text} has {} at axis {}",
            lhs[batch.0], batch.0, rhs[batch.1], batch.1
        )
    };
    Some(format!("{context}\n{}", domain_trap_line_at_i64("matmul")))
}

/// The same failure observed on matmul's decomposed product, whose operands
/// are `lhs` expanded with the output's column axis last and `rhs` expanded
/// with the output's row axis before its own last two
/// (`[..., i, j, k]` each). Removing those axes recovers the operands as
/// written, so the evaluators and the compiled runtime report one rendering
/// whichever form they observe the disagreement in. Falls back to the
/// product's own shapes if the decomposed operands are not rank three or more.
pub fn matmul_product_disagreement(expanded_lhs: &[i64], expanded_rhs: &[i64]) -> String {
    let recovered = (expanded_lhs.len() >= 3 && expanded_rhs.len() >= 3).then(|| {
        let lhs = &expanded_lhs[..expanded_lhs.len() - 1];
        let row_axis = expanded_rhs.len() - 3;
        let rhs = expanded_rhs[..row_axis]
            .iter()
            .chain(&expanded_rhs[row_axis + 1..])
            .copied()
            .collect::<Vec<_>>();
        matmul_operand_disagreement(lhs, &rhs)
    });
    recovered
        .flatten()
        .unwrap_or_else(|| operand_shape_disagreement("matmul", expanded_lhs, expanded_rhs))
}

/// A runtime target extent of the movement operation `op` is negative: the
/// non-negativity guard of spec/04-type-system.md section 4.7 traps
/// `Domain` in `op`, after a context line naming the axis and the value.
pub fn negative_target_extent(op: &str, axis: usize, extent: i64) -> String {
    format!(
        "{op} target extent at axis {axis} is negative: {extent}\n{}",
        domain_trap_line_at_i64(op)
    )
}

/// `reshape`'s target and its input disagree on the element count at run
/// time: an extent disagreement under spec/04-type-system.md section 4.7,
/// so a `Domain` trap in `reshape`, after a context line naming both counts.
pub fn reshape_element_count_disagreement(target: u64, input: u64) -> String {
    format!(
        "reshape target has {target} elements but the tensor has {input}\n{}",
        domain_trap_line_at_i64("reshape")
    )
}

/// A sparse primitive's index lies outside the selected axis: a `Domain`
/// trap in the primitive ([05-SPARSE-1], spec/05 section 3.5), after a
/// context line naming the index, the axis, and its extent.
pub fn sparse_index_out_of_bounds(op: &str, index: i64, axis: usize, extent: i64) -> String {
    format!(
        "{op} index {index} out of bounds at axis {axis} of extent {extent}\n{}",
        domain_trap_line_at_i64(op)
    )
}

/// The text before the offending value in [`list_argument_negative`]. A lane
/// that assembles the message at run time from the value's rendering uses
/// this prefix, so it cannot drift from the renderer.
pub fn list_argument_negative_prefix(operation: &str, argument: &str) -> String {
    format!("{operation} requires non-negative {argument}, got ")
}

/// A List selection ([05-OP-54]: `index`, `take`, `skip`) received a
/// negative index or count. The spec requires a loud failure and fixes no
/// wording, and the failure is not a numeric trap.
pub fn list_argument_negative(operation: &str, argument: &str, value: i64) -> String {
    format!(
        "{}{value}",
        list_argument_negative_prefix(operation, argument)
    )
}

/// The text before the index in [`list_index_out_of_bounds`].
pub const LIST_INDEX_PREFIX: &str = "index ";
/// The text between the index and the List length in
/// [`list_index_out_of_bounds`].
pub const LIST_INDEX_LEN_INFIX: &str = " out of bounds for list of len ";

/// `index(xs, i)` with `i` outside the List ([05-OP-54]): negative, or at or
/// past the length. The spec requires a loud failure and fixes no wording,
/// and the failure is not a numeric trap.
pub fn list_index_out_of_bounds(index: i64, len: usize) -> String {
    if index < 0 {
        list_argument_negative("index", "index", index)
    } else {
        format!("{LIST_INDEX_PREFIX}{index}{LIST_INDEX_LEN_INFIX}{len}")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        list_index_out_of_bounds, negative_target_extent, operand_shape_disagreement,
        reshape_element_count_disagreement, sparse_index_out_of_bounds,
    };

    #[test]
    fn a_shape_disagreement_names_the_axis_or_the_ranks() {
        assert_eq!(
            operand_shape_disagreement("add", &[2, 3], &[2, 4]),
            "add operands disagree at axis 1: lhs [2, 3] has 3, rhs [2, 4] has 4\n\
             numeric trap: domain in add at i64"
        );
        assert_eq!(
            operand_shape_disagreement("lt", &[2, 3], &[3]),
            "lt operands disagree in rank: lhs [2, 3] has rank 2, rhs [3] has rank 1\n\
             numeric trap: domain in lt at i64"
        );
        // A longer operand whose prefix agrees disagrees in rank.
        assert!(operand_shape_disagreement("mul", &[2], &[2, 1]).contains("in rank"));
    }

    #[test]
    fn a_matmul_disagreement_names_matmul_and_the_operands_as_written() {
        use super::{matmul_operand_disagreement, matmul_product_disagreement};
        let shared = "matmul shared axis disagrees: lhs [2, 3] has 3 at axis 1, \
                      rhs [2, 2] has 2 at axis 0\n\
                      numeric trap: domain in matmul at i64";
        assert_eq!(
            matmul_operand_disagreement(&[2, 3], &[2, 2]).as_deref(),
            Some(shared)
        );
        // The decomposed product of the same operands renders identically.
        assert_eq!(matmul_product_disagreement(&[2, 3, 2], &[2, 2, 2]), shared);
        assert_eq!(
            matmul_operand_disagreement(&[4, 2, 3], &[5, 3, 2]).as_deref(),
            Some(
                "matmul batch axis disagrees: lhs [4, 2, 3] has 4 at axis 0, \
                 rhs [5, 3, 2] has 5 at axis 0\n\
                 numeric trap: domain in matmul at i64"
            )
        );
        // Agreement, including a broadcast batch extent of 1, is no failure.
        assert_eq!(matmul_operand_disagreement(&[1, 2, 3], &[5, 3, 2]), None);
        assert_eq!(matmul_operand_disagreement(&[2, 3], &[3, 2]), None);
    }

    #[test]
    fn index_failures_name_the_index_and_the_bound() {
        assert_eq!(
            sparse_index_out_of_bounds("gather", 3, 0, 3),
            "gather index 3 out of bounds at axis 0 of extent 3\n\
             numeric trap: domain in gather at i64"
        );
        assert_eq!(
            reshape_element_count_disagreement(6, 3),
            "reshape target has 6 elements but the tensor has 3\n\
             numeric trap: domain in reshape at i64"
        );
        assert_eq!(
            negative_target_extent("reshape", 0, -2),
            "reshape target extent at axis 0 is negative: -2\n\
             numeric trap: domain in reshape at i64"
        );
        assert_eq!(
            list_index_out_of_bounds(5, 2),
            "index 5 out of bounds for list of len 2"
        );
        assert_eq!(
            list_index_out_of_bounds(-1, 2),
            "index requires non-negative index, got -1"
        );
        assert_eq!(
            super::list_argument_negative("take", "count", -3),
            "take requires non-negative count, got -3"
        );
    }
}
