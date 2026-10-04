//! The [05-HOST-3] `test_*` assertion family, shared by every execution
//! lane.
//!
//! The evaluator and the `chelis_test_*` C exports both decide closeness and
//! build every failure message here, so an assertion traps `Test` with the
//! same text on both lanes. Each lane supplies its own rendering of values,
//! dtypes and shapes, which already agree for printing.

/// `test_assert`'s failure.
pub fn assert_message(label: &str) -> String {
    format!("assert failed: {label}")
}

/// `test_assert_eq`'s failure, given each operand's rendering.
pub fn assert_eq_message(label: &str, expected: &str, actual: &str) -> String {
    format!("assert_eq ({label}): expected {expected}, got {actual}")
}

/// `test_assert_eq_tensor` on tensors of different shape or dtype.
pub fn eq_tensor_header_message(
    label: &str,
    expected_shape: &[usize],
    expected_dtype: &str,
    actual_shape: &[usize],
    actual_dtype: &str,
) -> String {
    format!(
        "assert_eq_tensor ({label}): expected tensor shape {expected_shape:?} at {expected_dtype}, got {actual_shape:?} at {actual_dtype}"
    )
}

/// `test_assert_eq_tensor` at the first unequal stored element.
pub fn eq_tensor_mismatch_message(label: &str, index: usize) -> String {
    format!("assert_eq_tensor ({label}): first mismatch at row-major index {index}")
}

/// A shape as `[d0, d1, ...]`.
pub fn render_shape(shape: &[usize]) -> String {
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{dimensions}]")
}

pub fn close_common_dtype_message(label: &str) -> String {
    format!(
        "assert_close_tensor ({label}): actual, expected, and tolerance must have one common active float dtype"
    )
}

pub fn close_shape_message(label: &str, expected: &[usize], actual: &[usize]) -> String {
    format!(
        "assert_close_tensor ({label}): shape mismatch, expected {}, got {}",
        render_shape(expected),
        render_shape(actual)
    )
}

pub fn close_tolerance_message(label: &str, tolerance: &str) -> String {
    format!(
        "assert_close_tensor ({label}): invalid tolerance {tolerance} (must be finite and non-negative)"
    )
}

/// `test_assert_close_tensor` at the first element that is not close, given
/// each value's rendering at the tensor dtype.
pub fn close_mismatch_message(
    label: &str,
    index: usize,
    expected: &str,
    actual: &str,
    tolerance: &str,
    either_nan: bool,
) -> String {
    let nan_suffix = if either_nan {
        " (NaN is never close)"
    } else {
        ""
    };
    format!(
        "assert_close_tensor ({label}): at index {index} expected {expected}, got {actual}, tol {tolerance}{nan_suffix}"
    )
}

/// [05-OP-35]'s closeness at f64: NaN is never close, equal values are, and
/// otherwise both are finite and differ by at most the tolerance.
pub fn close_at_f64_width(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return false;
    }
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= tolerance
}

/// The same rule computed at f32.
pub fn close_at_f32_width(actual: f32, expected: f32, tolerance: f32) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return false;
    }
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= tolerance
}

/// The first index whose pair is not close at f64.
pub fn first_f64_mismatch(
    actual: impl Iterator<Item = f64>,
    expected: impl Iterator<Item = f64>,
    tolerance: f64,
) -> Option<usize> {
    actual
        .zip(expected)
        .enumerate()
        .find_map(|(index, (actual, expected))| {
            (!close_at_f64_width(actual, expected, tolerance)).then_some(index)
        })
}

/// The first index whose pair is not close at f32.
pub fn first_f32_mismatch(
    actual: impl Iterator<Item = f32>,
    expected: impl Iterator<Item = f32>,
    tolerance: f32,
) -> Option<usize> {
    actual
        .zip(expected)
        .enumerate()
        .find_map(|(index, (actual, expected))| {
            (!close_at_f32_width(actual, expected, tolerance)).then_some(index)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_is_never_close_and_signed_zeros_are() {
        assert!(!close_at_f64_width(f64::NAN, f64::NAN, 1.0));
        assert!(close_at_f64_width(0.0, -0.0, 0.0));
        assert!(close_at_f32_width(f32::INFINITY, f32::INFINITY, 0.0));
        assert!(!close_at_f32_width(f32::INFINITY, f32::MAX, f32::MAX));
        assert_eq!(
            first_f32_mismatch([1.0, 2.0].into_iter(), [1.0, 2.5].into_iter(), 0.1),
            Some(1)
        );
    }

    #[test]
    fn messages_name_the_label_and_operation() {
        assert_eq!(assert_message("x"), "assert failed: x");
        assert_eq!(
            eq_tensor_header_message("t", &[2], "i8", &[1], "i8"),
            "assert_eq_tensor (t): expected tensor shape [2] at i8, got [1] at i8"
        );
        assert_eq!(render_shape(&[]), "[]");
    }
}
