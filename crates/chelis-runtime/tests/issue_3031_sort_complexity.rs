//! chelis#3031: `chelis_tensor_sort` sorts each axis lane in O(n log n).
//!
//! The runtime sorted each lane by insertion sort, so a sort grew
//! quadratically in the lane length: about seven seconds for 10^5 elements
//! in a compiled program that `chelis eval` sorted in milliseconds. The
//! ordering contract of [05-OP-33] and [05-OP-53] is unchanged: ascending,
//! stable, NaNs after every other value in source order, and signed zeros
//! equal.
//!
//! The ordering cases run in this process. The size case runs in a child
//! process under a deadline, because a quadratic sort does not fail, it only
//! takes minutes: a reversed lane of 400 000 elements needs about 8 * 10^10
//! adjacent swaps by insertion and well under a second by an O(n log n)
//! sort, so the deadline is generous on a loaded machine and still far below
//! the quadratic time.

use chelis_runtime::{
    chelis_alloc, chelis_dtype, chelis_tensor, chelis_tensor_begin_write,
    chelis_tensor_borrow_value, chelis_tensor_end_write, chelis_tensor_read_view,
    chelis_tensor_sort, chelis_tensor_write_view, chelis_tuple_get, CHELIS_DTYPE_F64,
    CHELIS_DTYPE_I64,
};
use std::env;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CHILD_ENV: &str = "CHELIS_ISSUE_3031_SORT_CHILD";
const DEADLINE: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(50);
const LARGE: i64 = 400_000;

unsafe fn tensor_of<T: Copy>(
    shape: &[i64],
    dtype: chelis_dtype,
    values: &[T],
) -> *mut chelis_tensor {
    let tensor = chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype);
    let guard = chelis_tensor_begin_write(tensor);
    let view = chelis_tensor_write_view(guard);
    assert_eq!(view.count as usize, values.len());
    view.data
        .cast::<T>()
        .copy_from(values.as_ptr(), values.len());
    chelis_tensor_end_write(guard);
    tensor
}

unsafe fn read<T: Copy>(tensor: *const chelis_tensor) -> Vec<T> {
    let view = chelis_tensor_read_view(tensor);
    std::slice::from_raw_parts(view.data.cast::<T>(), view.count as usize).to_vec()
}

/// The sorted values and source indices of `input` along `axis`.
unsafe fn sorted<T: Copy>(input: *mut chelis_tensor, axis: i32) -> (Vec<T>, Vec<i64>) {
    let result = chelis_tensor_sort(input, axis);
    let values = chelis_tensor_borrow_value(chelis_tuple_get(result, 0));
    let indices = chelis_tensor_borrow_value(chelis_tuple_get(result, 1));
    (read::<T>(values), read::<i64>(indices))
}

#[test]
fn equal_values_keep_their_source_order() {
    unsafe {
        let input = tensor_of(&[7], CHELIS_DTYPE_I64, &[3_i64, 1, 3, 2, 1, 3, 0]);
        let (values, indices) = sorted::<i64>(input, 0);
        assert_eq!(values, [0, 1, 1, 2, 3, 3, 3]);
        assert_eq!(indices, [6, 1, 4, 3, 0, 2, 5]);
    }
}

#[test]
fn nans_follow_every_value_and_signed_zeros_are_equal() {
    unsafe {
        let input = tensor_of(
            &[7],
            CHELIS_DTYPE_F64,
            &[
                f64::NAN,
                0.0_f64,
                -1.5,
                -0.0,
                f64::NAN,
                f64::NEG_INFINITY,
                0.0,
            ],
        );
        let (values, indices) = sorted::<f64>(input, 0);
        assert_eq!(indices, [5, 2, 1, 3, 6, 0, 4]);
        assert_eq!(values[0], f64::NEG_INFINITY);
        assert_eq!(values[1], -1.5);
        // The zeros keep their source order, signs included.
        assert!(values[2] == 0.0 && values[2].is_sign_positive());
        assert!(values[3] == 0.0 && values[3].is_sign_negative());
        assert!(values[4] == 0.0 && values[4].is_sign_positive());
        assert!(values[5].is_nan() && values[6].is_nan());
    }
}

/// The comparator is a strict weak order over every input: all NaNs are one
/// class after every number, whatever their payload or sign, so they keep
/// their source order and their stored bits.
#[test]
fn nan_payloads_keep_their_bits_and_source_order() {
    unsafe {
        let quiet = f64::from_bits(0x7ff8_0000_0000_0001);
        let negative = f64::from_bits(0xfff8_0000_0000_0002);
        let signalling = f64::from_bits(0x7ff0_0000_0000_0003);
        let input = [
            quiet,
            2.0_f64,
            negative,
            f64::MIN,
            signalling,
            -0.0,
            f64::MAX,
            0.0,
        ];
        let tensor = tensor_of(&[8], CHELIS_DTYPE_F64, &input);
        let (values, indices) = sorted::<f64>(tensor, 0);
        assert_eq!(indices, [3, 5, 7, 1, 6, 0, 2, 4]);
        let bits: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(
            bits,
            [
                f64::MIN.to_bits(),
                (-0.0_f64).to_bits(),
                0.0_f64.to_bits(),
                2.0_f64.to_bits(),
                f64::MAX.to_bits(),
                quiet.to_bits(),
                negative.to_bits(),
                signalling.to_bits(),
            ]
        );
    }
}

#[test]
fn integer_extremes_sort_exactly() {
    unsafe {
        let input = [i64::MAX, i64::MIN, 0, i64::MAX, -1, i64::MIN, 1];
        let tensor = tensor_of(&[7], CHELIS_DTYPE_I64, &input);
        let (values, indices) = sorted::<i64>(tensor, 0);
        assert_eq!(values, [i64::MIN, i64::MIN, -1, 0, 1, i64::MAX, i64::MAX]);
        assert_eq!(indices, [1, 5, 4, 2, 6, 0, 3]);
    }
}

#[test]
fn each_lane_of_an_inner_axis_sorts_on_its_own() {
    unsafe {
        // Shape [2, 3], sorted along axis 0: each column is a lane.
        let input = tensor_of(&[2, 3], CHELIS_DTYPE_I64, &[5_i64, 1, 4, 2, 1, 6]);
        let (values, indices) = sorted::<i64>(input, 0);
        assert_eq!(values, [2, 1, 4, 5, 1, 6]);
        assert_eq!(indices, [1, 0, 0, 0, 1, 1]);
        // Sorted along axis 1: each row is a lane.
        let input = tensor_of(&[2, 3], CHELIS_DTYPE_I64, &[5_i64, 1, 4, 2, 1, 6]);
        let (values, indices) = sorted::<i64>(input, 1);
        assert_eq!(values, [1, 4, 5, 1, 2, 6]);
        assert_eq!(indices, [1, 2, 0, 1, 0, 2]);
    }
}

/// A reversed lane with each value twice: the largest number of inversions,
/// and ties whose source order the result must keep.
fn run_large_case() {
    unsafe {
        let input: Vec<i64> = (0..LARGE).map(|k| (LARGE - 1 - k) / 2).collect();
        let tensor = tensor_of(&[LARGE], CHELIS_DTYPE_I64, &input);
        let (values, indices) = sorted::<i64>(tensor, 0);
        for (position, (value, source)) in values.iter().zip(&indices).enumerate() {
            let position = position as i64;
            assert_eq!(*value, position / 2, "value at {position}");
            // Value v sits at source positions LARGE - 2 - 2v and
            // LARGE - 1 - 2v, the earlier one first.
            let expected_source = LARGE - 2 - 2 * (position / 2) + position % 2;
            assert_eq!(*source, expected_source, "source index at {position}");
        }
    }
}

#[test]
fn a_long_lane_sorts_within_the_deadline() {
    if env::var(CHILD_ENV).is_ok() {
        run_large_case();
        return;
    }
    let started = Instant::now();
    let mut child = Command::new(env::current_exe().expect("current test binary"))
        .args([
            "--exact",
            "a_long_lane_sorts_within_the_deadline",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the sorting child");
    // A quadratic sort does not exit in time, so this cannot be a blocking
    // `wait`.
    let status = loop {
        match child.try_wait().expect("poll child") {
            Some(status) => break Some(status),
            None if started.elapsed() >= DEADLINE => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            None => std::thread::sleep(POLL),
        }
    };
    match status {
        None => panic!(
            "sorting {LARGE} elements did not finish within {DEADLINE:?}; the lane sort is \
             not O(n log n)"
        ),
        Some(status) if !status.success() => {
            let output = child.wait_with_output().expect("child output");
            panic!(
                "the sorting child exited {status}\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Some(_) => {}
    }
}
