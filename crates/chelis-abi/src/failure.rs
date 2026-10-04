//! The text of language failures that every lane reports identically.
//!
//! `chelis eval`'s interpreters, the checker's numeric kernels and the
//! compiled runtime all render these failures through this module, so the
//! lanes cannot drift. [`NumericTrapLine`] is the one renderer of the
//! [04-NUM-9] line `numeric trap: <kind> in <op> at <prim>`. A failure with
//! context ends in that line: the context lines before it name the
//! disagreeing sources and the values observed. A runtime extent or index
//! guard's `<prim>` slot is `i64`, because the guard finalizes an extent or
//! an index (spec/04-type-system.md section 4.7).

use std::fmt;

/// The closed set of [04-NUM-9] numeric-trap kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericTrapKind {
    /// An integer result outside the declared dtype's range.
    Overflow,
    /// An input or result outside the operation's or dtype's domain.
    Domain,
    /// Integer division or remainder by a zero divisor.
    DivZero,
}

impl NumericTrapKind {
    pub const ALL: [Self; 3] = [Self::Overflow, Self::Domain, Self::DivZero];

    /// The kind's frozen spelling inside a trap line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Overflow => "overflow",
            Self::Domain => "domain",
            Self::DivZero => "division by zero",
        }
    }
}

/// One [04-NUM-9] numeric-trap line: `numeric trap: <kind> in <op> at <dtype>`.
///
/// Every lane renders its trap lines through this one formatter, so the
/// evaluator and the compiled runtime cannot drift apart. The line has no
/// prefix or suffix; any context is a separate line before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericTrapLine<'a> {
    pub kind: NumericTrapKind,
    /// The canonical name of the primitive that raised the trap.
    pub op: &'a str,
    /// The canonical name of the dtype the primitive was finalizing to.
    pub dtype: &'a str,
}

impl NumericTrapLine<'_> {
    /// The frozen prefix every trap line begins with.
    pub const PREFIX: &'static str = "numeric trap: ";
    /// The frozen separator before the raising primitive's name.
    pub const OPERATION_SEPARATOR: &'static str = " in ";
    /// The frozen separator before the dtype's name.
    pub const DTYPE_SEPARATOR: &'static str = " at ";
}

impl fmt::Display for NumericTrapLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}{}{}{}{}",
            Self::PREFIX,
            self.kind.as_str(),
            Self::OPERATION_SEPARATOR,
            self.op,
            Self::DTYPE_SEPARATOR,
            self.dtype
        )
    }
}

/// The [04-NUM-9] line of a `Domain` trap in `op` that finalizes an extent,
/// an index or a count, so its `<prim>` slot is `i64`.
pub fn domain_trap_line_at_i64(op: &str) -> String {
    domain_trap_line(op, "i64")
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
        "{}{extent}\n{}",
        negative_target_extent_prefix(op, axis),
        domain_trap_line_at_i64(op)
    )
}

/// The context line of [`negative_target_extent`] before the value. A lane
/// that assembles the message at run time prints this prefix, the value and
/// then [`domain_trap_line_at_i64`], so it cannot drift from the renderer.
pub fn negative_target_extent_prefix(op: &str, axis: usize) -> String {
    format!("{op} target extent at axis {axis} is negative: ")
}

/// A runtime `pad` or `shrink` bound is negative: the non-negativity guard
/// of spec/04-type-system.md section 4.7 traps `Domain` in the movement
/// operation, after a context line naming the axis and the value.
pub fn negative_movement_bound(op: &str, axis: usize, bound: i64) -> String {
    format!(
        "{}{bound}\n{}",
        negative_movement_bound_prefix(op, axis),
        domain_trap_line_at_i64(op)
    )
}

/// The context line of [`negative_movement_bound`] before the value.
pub fn negative_movement_bound_prefix(op: &str, axis: usize) -> String {
    format!("{op} bound at axis {axis} is negative: ")
}

/// A runtime length or count argument of `op` is negative (`tensor_scan`'s
/// length): the non-negativity guard of spec/04-type-system.md section 4.7
/// traps `Domain` in `op`, after a context line naming the value.
pub fn negative_length(op: &str, length: i64) -> String {
    format!(
        "{}{length}\n{}",
        negative_length_prefix(op),
        domain_trap_line_at_i64(op)
    )
}

/// The context line of [`negative_length`] before the value.
pub fn negative_length_prefix(op: &str) -> String {
    format!("{op} length is negative: ")
}

/// The parts of `concat` disagree on an axis other than the concatenated
/// one at run time: an extent disagreement under spec/04-type-system.md
/// section 4.7, so a `Domain` trap in `concat` after a context line naming
/// the axis and both parts' extents.
pub fn concat_extent_disagreement(axis: usize, first: i64, part: usize, extent: i64) -> String {
    format!(
        "concat parts disagree at axis {axis}: part 0 has {first}, part {part} has {extent}\n{}",
        domain_trap_line_at_i64("concat")
    )
}

/// `concat` received an empty List of parts, which [05-OP-62] excludes from
/// its domain and which no element type can complete.
pub fn concat_without_parts() -> String {
    format!(
        "concat received no tensor parts\n{}",
        domain_trap_line_at_i64("concat")
    )
}

/// An entry of a runtime size list of `op` (`split`'s sizes, a
/// `reduce_window_*` window or stride) is negative: the non-negativity
/// guard of spec/04-type-system.md section 4.7 traps `Domain` in `op`, after
/// a context line naming the entry and the value.
pub fn negative_list_entry(op: &str, index: usize, value: i64) -> String {
    format!(
        "{op} list entry {index} is negative: {value}\n{}",
        domain_trap_line_at_i64(op)
    )
}

/// The `split` sizes do not partition the selected axis ([05-OP-53]): an
/// extent disagreement under spec/04-type-system.md section 4.7.
pub fn split_sizes_disagreement(total: i64, axis: usize, extent: i64) -> String {
    format!(
        "split sizes sum to {total} but axis {axis} has extent {extent}\n{}",
        domain_trap_line_at_i64("split")
    )
}

/// Two children of a nested List given to `to_tensor` disagree in shape at
/// run time ([05-OP-57]): an extent disagreement under
/// spec/04-type-system.md section 4.7, after a context line naming the
/// first child's shape and the disagreeing child's.
pub fn to_tensor_ragged(first: &[i64], child: usize, shape: &[i64]) -> String {
    format!(
        "to_tensor children disagree in shape: child 0 has {}, child {child} has {}\n{}",
        render_shape(first),
        render_shape(shape),
        domain_trap_line_at_i64("to_tensor")
    )
}

/// A runtime extent or count guard of `op` failed for a reason no more
/// specific rendering here names: `context`, then the `Domain` trap line.
/// Every lane that can reach the same guard renders it through the same
/// call.
pub fn domain_guard(op: &str, context: &str) -> String {
    format!("{context}\n{}", domain_trap_line_at_i64(op))
}

/// The [04-NUM-9] line of a `Domain` trap in `op` at the dtype `prim`, the
/// dtype of the quantity its guard finalizes, spelled canonically (`f32`,
/// `i64`, ...).
pub fn domain_trap_line(op: &str, prim: &str) -> String {
    NumericTrapLine {
        kind: NumericTrapKind::Domain,
        op,
        dtype: prim,
    }
    .to_string()
}

/// `char_code` received a string that is not exactly one Unicode scalar
/// value ([05-OP-58]): a `Domain` trap in `char_code` at its `i64` result,
/// after a context line naming how many scalar values the string has.
pub fn char_code_not_one_scalar(count: usize) -> String {
    format!(
        "char_code operand has {count} Unicode scalar values, expected exactly one\n{}",
        domain_trap_line_at_i64("char_code")
    )
}

/// `char_from_code` received a negative value, a surrogate code point, or a
/// value above U+10FFFF ([05-OP-58]): a `Domain` trap in `char_from_code` at
/// the `i64` code it reads.
pub fn char_from_code_invalid(code: i64) -> String {
    format!(
        "char_from_code code {code} is not a Unicode scalar value\n{}",
        domain_trap_line_at_i64("char_from_code")
    )
}

/// `string_slice` received a negative start or length ([05-OP-58]): a
/// `Domain` trap in `string_slice` at the `i64` offset, after a context line
/// naming the argument and its value.
pub fn string_slice_negative(argument: &str, value: i64) -> String {
    format!(
        "string_slice {argument} is negative: {value}\n{}",
        domain_trap_line_at_i64("string_slice")
    )
}

/// A `clamp` bound is NaN at a row-major position ([05-OP-33]): a `Domain`
/// trap in `clamp` at the operand dtype `prim`.
pub fn clamp_bound_nan(position: usize, prim: &str) -> String {
    format!(
        "clamp bound is NaN at row-major position {position}\n{}",
        domain_trap_line("clamp", prim)
    )
}

/// A `clamp` lower bound exceeds its upper bound at a row-major position
/// ([05-OP-33]): a `Domain` trap in `clamp` at the operand dtype `prim`.
pub fn clamp_bounds_inverted(position: usize, prim: &str) -> String {
    format!(
        "clamp lower bound exceeds upper bound at row-major position {position}\n{}",
        domain_trap_line("clamp", prim)
    )
}

/// A `clamp` bound is neither a scalar nor shaped like the operand at run
/// time: an extent disagreement under spec/04-type-system.md section 4.7.
pub fn clamp_bound_shape(bound: &str, shape: &[i64], operand: &[i64]) -> String {
    format!(
        "clamp {bound} bound has shape {} but the operand has {}\n{}",
        render_shape(shape),
        render_shape(operand),
        domain_trap_line_at_i64("clamp")
    )
}

/// A runtime window of `op` (a `reduce_window_*` builtin) is wider than the
/// input axis it slides over ([05-RWIN-1..2]): a `Domain` trap in `op`.
pub fn window_exceeds_extent(op: &str, axis: usize, window: i64, extent: i64) -> String {
    format!(
        "{op} window {window} at axis {axis} exceeds the input extent {extent}\n{}",
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
    use super::{NumericTrapKind, NumericTrapLine};

    /// [04-NUM-9]: the closed trap kinds and the one line every lane renders.
    #[test]
    fn numeric_trap_lines_are_closed_and_frozen() {
        let expected = [
            (
                NumericTrapKind::Overflow,
                "numeric trap: overflow in floor_div at i8",
            ),
            (
                NumericTrapKind::Domain,
                "numeric trap: domain in floor_div at i8",
            ),
            (
                NumericTrapKind::DivZero,
                "numeric trap: division by zero in floor_div at i8",
            ),
        ];
        assert_eq!(
            NumericTrapKind::ALL.to_vec(),
            expected.iter().map(|(kind, _)| *kind).collect::<Vec<_>>()
        );
        for (kind, line) in expected {
            let rendered = NumericTrapLine {
                kind,
                op: "floor_div",
                dtype: "i8",
            }
            .to_string();
            assert_eq!(rendered, line);
        }
    }

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
    fn extent_guards_put_the_context_before_the_trap_line() {
        use super::{
            concat_extent_disagreement, concat_without_parts, domain_guard, negative_length,
            negative_list_entry, negative_movement_bound, split_sizes_disagreement,
            to_tensor_ragged, window_exceeds_extent,
        };
        for (rendered, context, op) in [
            (
                negative_movement_bound("pad", 1, -3),
                "pad bound at axis 1 is negative: -3",
                "pad",
            ),
            (
                negative_length("tensor_scan", -2),
                "tensor_scan length is negative: -2",
                "tensor_scan",
            ),
            (
                negative_list_entry("split", 0, -1),
                "split list entry 0 is negative: -1",
                "split",
            ),
            (
                split_sizes_disagreement(2, 0, 3),
                "split sizes sum to 2 but axis 0 has extent 3",
                "split",
            ),
            (
                concat_extent_disagreement(1, 2, 1, 1),
                "concat parts disagree at axis 1: part 0 has 2, part 1 has 1",
                "concat",
            ),
            (
                concat_without_parts(),
                "concat received no tensor parts",
                "concat",
            ),
            (
                to_tensor_ragged(&[2], 1, &[1]),
                "to_tensor children disagree in shape: child 0 has [2], child 1 has [1]",
                "to_tensor",
            ),
            (
                window_exceeds_extent("reduce_window_max", 0, 4, 3),
                "reduce_window_max window 4 at axis 0 exceeds the input extent 3",
                "reduce_window_max",
            ),
            (
                domain_guard(
                    "reduce_window_sum",
                    "reduce_window_sum strides[0] must be >= 1",
                ),
                "reduce_window_sum strides[0] must be >= 1",
                "reduce_window_sum",
            ),
        ] {
            assert_eq!(
                rendered,
                format!("{context}\nnumeric trap: domain in {op} at i64")
            );
        }
        // A lane that prints the prefix and the value renders the same text.
        assert_eq!(
            format!("{}-3", super::negative_movement_bound_prefix("shrink", 0)),
            negative_movement_bound("shrink", 0, -3)
                .lines()
                .next()
                .unwrap()
        );
        assert_eq!(
            format!("{}-1", super::negative_target_extent_prefix("insert", 2)),
            negative_target_extent("insert", 2, -1)
                .lines()
                .next()
                .unwrap()
        );
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
