//! A zip of two owned vectors of a Copy element, mapped and collected, as in
//! `chelis_ir::eval` (`lhs.into_iter().zip(rhs).map(..).collect()`). Native Vec
//! collection folds this TrustedRandomAccessNoCoerce zip over the shorter
//! length; the model's repeated next also takes one surplus left element,
//! which is not observable for a Copy element.

pub fn differences(lhs: Vec<i64>, rhs: Vec<i64>) -> Vec<i64> {
    lhs.into_iter().zip(rhs).map(|(l, r)| l - r).collect()
}

pub fn equal_lengths() -> Vec<i64> {
    differences(vec![5, 7, 9], vec![1, 2, 3])
}

pub fn longer_left() -> Vec<i64> {
    differences(vec![5, 7, 9, 11], vec![1, 2])
}

pub fn longer_right() -> Vec<i64> {
    differences(vec![5], vec![1, 2, 3])
}

pub fn empty_left() -> Vec<i64> {
    differences(vec![], vec![1])
}

/// A right side with a callback runs it once per pair in both schedules.
pub fn mapped_right() -> (Vec<i64>, u32) {
    let mut calls = 0;
    let right = [1i64, 2];
    let values = vec![5i64, 7, 9]
        .into_iter()
        .zip(right.iter().map(|r| {
            calls += 1;
            *r
        }))
        .map(|(l, r)| l * r)
        .collect();
    (values, calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_stops_at_the_shorter_vector() {
        assert_eq!(equal_lengths(), [4, 5, 6]);
        assert_eq!(longer_left(), [4, 5]);
        assert_eq!(longer_right(), [4]);
        assert_eq!(empty_left(), Vec::<i64>::new());
        assert_eq!(mapped_right(), (vec![5, 14], 2));
    }
}
