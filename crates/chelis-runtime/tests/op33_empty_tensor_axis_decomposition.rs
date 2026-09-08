//! Empty operands must not walk their own axis decomposition.
//!
//! `cumsum`, `sort`, and `trace` split a shape into the product of the extents
//! before the axis (`outer`) and after it (`inner`). `inner` is the canonical
//! stride at that axis, which the allocator's suffix walk already checks.
//! `outer` is not checked anywhere: it is the prefix product, and for a tensor
//! whose element count is zero the prefix extents are unconstrained, because
//! the zero elsewhere is what makes the count representable in the first
//! place.
//!
//! That combination became reachable when the allocator stopped folding the
//! element count left-to-right, so a shape like `[2^32, 2^32, 0]` began
//! allocating (correctly: zero elements) instead of trapping. Two distinct
//! failures followed, and an overflow check alone only fixes one of them:
//!
//! * `outer` overflows `usize` and panics `attempt to multiply with overflow`.
//!   Across `extern "C"` that is a non-unwinding abort (SIGABRT), not the
//!   branded `Overflow:` exit [05-OP-33] owes.
//! * `outer` merely gets large without overflowing (`4e9 * 4e9` fits `u64`),
//!   and the operation spins an empty loop for hours. No arithmetic is wrong,
//!   so this reproduces identically in release.
//!
//! The contract is that an empty operand yields an empty result promptly, so
//! these cases assert both the result and a wall-clock bound. The bound is
//! generous enough to survive a loaded machine and still tiny next to the
//! ~1.6e19 iterations the unguarded loop would attempt.

use chelis_runtime::{
    chelis_alloc, chelis_tensor, chelis_tensor_begin_write, chelis_tensor_borrow_value,
    chelis_tensor_cumsum, chelis_tensor_end_write, chelis_tensor_numel, chelis_tensor_rank,
    chelis_tensor_read_view, chelis_tensor_sort, chelis_tensor_trace, chelis_tensor_write_view,
    chelis_tuple_get, CHELIS_DTYPE_F32,
};
use std::env;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CHILD_ENV: &str = "CHELIS_OP33_EMPTY_AXIS_CHILD";
const DEADLINE: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(50);

/// 2^32: squaring it leaves `usize`, so the prefix product panics.
const OVERFLOW_EXTENT: i64 = 1 << 32;
/// 4e9: squaring it stays inside `u64`, so the prefix product is merely
/// astronomical and the loop never finishes.
const HANG_EXTENT: i64 = 4_000_000_000;

unsafe fn tensor(shape: &[i64]) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), CHELIS_DTYPE_F32)
}

unsafe fn write_f32(tensor: *mut chelis_tensor, values: &[f32]) {
    let guard = chelis_tensor_begin_write(tensor);
    let view = chelis_tensor_write_view(guard);
    assert_eq!(view.count as usize, values.len());
    view.data
        .cast::<f32>()
        .copy_from(values.as_ptr(), values.len());
    chelis_tensor_end_write(guard);
}

unsafe fn read_view<T: Copy>(tensor: *const chelis_tensor) -> Vec<T> {
    let view = chelis_tensor_read_view(tensor);
    std::slice::from_raw_parts(view.data.cast::<T>(), view.count as usize).to_vec()
}

fn run_case(case: &str) {
    unsafe {
        match case {
            "cumsum-overflow-axis" => {
                let input = tensor(&[OVERFLOW_EXTENT, OVERFLOW_EXTENT, 0]);
                let out = chelis_tensor_cumsum(input, 2);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            "cumsum-overflow-negative-axis" => {
                let input = tensor(&[OVERFLOW_EXTENT, OVERFLOW_EXTENT, 0]);
                let out = chelis_tensor_cumsum(input, -1);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            "cumsum-int64-max-extents" => {
                let input = tensor(&[i64::MAX, i64::MAX, 0]);
                let out = chelis_tensor_cumsum(input, 2);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            "cumsum-hang-axis" => {
                let input = tensor(&[HANG_EXTENT, HANG_EXTENT, 0]);
                let out = chelis_tensor_cumsum(input, 2);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            // A middle axis reaches the same prefix product through `outer`
            // alone, with no multiplication to overflow: the spin is the whole
            // defect. `2^32` here merely took ~30s, which a deadline generous
            // enough not to flake would wave through, so the prefix is
            // `i64::MAX` and the unguarded loop cannot finish at all.
            "cumsum-spin-middle-axis" => {
                let input = tensor(&[i64::MAX, 2, 0]);
                let out = chelis_tensor_cumsum(input, 1);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            "sort-overflow-axis" => {
                let input = tensor(&[OVERFLOW_EXTENT, OVERFLOW_EXTENT, 0]);
                let sorted = chelis_tensor_sort(input, 2);
                let values = chelis_tensor_borrow_value(chelis_tuple_get(sorted, 0));
                let indices = chelis_tensor_borrow_value(chelis_tuple_get(sorted, 1));
                assert_eq!(chelis_tensor_numel(values), 0);
                assert_eq!(chelis_tensor_numel(indices), 0);
            }
            "sort-int64-max-extents" => {
                let input = tensor(&[i64::MAX, i64::MAX, 0]);
                let sorted = chelis_tensor_sort(input, 2);
                let values = chelis_tensor_borrow_value(chelis_tuple_get(sorted, 0));
                assert_eq!(chelis_tensor_numel(values), 0);
            }
            "sort-hang-axis" => {
                let input = tensor(&[HANG_EXTENT, HANG_EXTENT, 0]);
                let sorted = chelis_tensor_sort(input, 2);
                let values = chelis_tensor_borrow_value(chelis_tuple_get(sorted, 0));
                assert_eq!(chelis_tensor_numel(values), 0);
            }
            // `diagonal` puts the diagonal in axis1's slot and drops axis2, so
            // the zero has to sit after the reduced axis for the empty result
            // to survive trace's own output allocation.
            "trace-overflow-axis" => {
                let input = tensor(&[OVERFLOW_EXTENT, OVERFLOW_EXTENT, 1, 0, 1]);
                let out = chelis_tensor_trace(input, 2, 4);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            "trace-int64-max-extents" => {
                let input = tensor(&[i64::MAX, i64::MAX, 1, 0, 1]);
                let out = chelis_tensor_trace(input, 2, 4);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            "trace-hang-axis" => {
                let input = tensor(&[HANG_EXTENT, HANG_EXTENT, 1, 0, 1]);
                let out = chelis_tensor_trace(input, 2, 4);
                assert_eq!(chelis_tensor_numel(out), 0);
            }
            // Positive controls: nonempty operands with the same operations
            // must still compute exactly.
            "legal-cumsum" => {
                let input = tensor(&[2, 3]);
                write_f32(input, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
                let out = chelis_tensor_cumsum(input, 1);
                assert_eq!(read_view::<f32>(out), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
            }
            "legal-sort" => {
                let input = tensor(&[4]);
                write_f32(input, &[3.0_f32, 1.0, 4.0, 2.0]);
                let sorted = chelis_tensor_sort(input, 0);
                let values = chelis_tensor_borrow_value(chelis_tuple_get(sorted, 0));
                let indices = chelis_tensor_borrow_value(chelis_tuple_get(sorted, 1));
                assert_eq!(read_view::<f32>(values), &[1.0, 2.0, 3.0, 4.0]);
                assert_eq!(read_view::<i64>(indices), &[1, 3, 0, 2]);
            }
            "legal-trace" => {
                let input = tensor(&[3, 3]);
                write_f32(input, &[1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0]);
                let out = chelis_tensor_trace(input, 0, 1);
                assert_eq!(chelis_tensor_rank(out), 0);
                assert_eq!(read_view::<f32>(out), &[7.0_f32]);
            }
            // A zero-extent operand whose other extents are ordinary: the
            // empty path must be correct, not merely fast.
            "legal-empty-small" => {
                let input = tensor(&[2, 0, 3]);
                let out = chelis_tensor_cumsum(input, 2);
                assert_eq!(chelis_tensor_numel(out), 0);
                assert_eq!(chelis_tensor_rank(out), 3);
                assert!(chelis_tensor_read_view(out).data.is_null());
            }
            other => panic!("unknown empty-axis case: {other}"),
        }
    }
}

const CASES: &[&str] = &[
    "cumsum-overflow-axis",
    "cumsum-overflow-negative-axis",
    "cumsum-int64-max-extents",
    "cumsum-hang-axis",
    "cumsum-spin-middle-axis",
    "sort-overflow-axis",
    "sort-int64-max-extents",
    "sort-hang-axis",
    "trace-overflow-axis",
    "trace-int64-max-extents",
    "trace-hang-axis",
    "legal-cumsum",
    "legal-sort",
    "legal-trace",
    "legal-empty-small",
];

#[test]
fn empty_operands_return_promptly_instead_of_overflowing_or_spinning() {
    if let Ok(case) = env::var(CHILD_ENV) {
        run_case(&case);
        return;
    }
    let test_binary = env::current_exe().expect("current test binary");
    let mut failures = Vec::new();
    for case in CASES {
        let started = Instant::now();
        let mut child = Command::new(&test_binary)
            .args([
                "--exact",
                "empty_operands_return_promptly_instead_of_overflowing_or_spinning",
                "--nocapture",
            ])
            .env(CHILD_ENV, case)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| panic!("spawn child `{case}`: {error}"));
        // A spinning child never exits, so this cannot be a blocking `wait`.
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
            None => failures.push(format!(
                "{case}: did not finish within the deadline; the empty operand walked its \
                 axis decomposition instead of returning"
            )),
            Some(status) if !status.success() => {
                let output = child.wait_with_output().expect("child output");
                failures.push(format!(
                    "{case}: exited {status}\n{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            Some(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} empty-operand cases failed:\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n")
    );
}
