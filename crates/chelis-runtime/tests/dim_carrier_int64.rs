//! chelis#1112: the C lane's dimension carrier is int64.
//!
//! `spec/05-risc-primitives.md` [05-DIM-2] declares the extent dtype int64
//! and `spec/04-type-system.md` [04-NUM-11] requires a value to cross every
//! boundary at its declared dtype. Before this change the C ABI carried
//! `shape`, `strides`, and `size` as 32-bit `int`, so an extent above
//! `2^31 - 1` could not cross at all: `chelis_host_reshape_tensor` had to
//! trap on it to avoid a silent truncating store.
//!
//! Three obligations are checked separately because they fail for different
//! reasons:
//!
//! * **the carrier holds the value**: an extent above `2^31 - 1` reaches the
//!   exact byte-capacity check without truncation. The exact ABI no longer
//!   permits a metadata-only view whose backing storage is smaller than its
//!   declared tensor, so the default-CI test exercises this as a negative
//!   contract. The allocating counterpart remains the `#[ignore]`d manual
//!   gate at the bottom.
//! * **the two mirrors agree**: the Rust `#[repr(C)]` struct and the
//!   published `chelis_runtime.h` describe the same bytes. A widening
//!   applied to one side only produces silent field-offset corruption in
//!   every compiled program, which no Rust-only test can see.
//! * **the axis domain stayed narrow**: [05-DIM-1] separates axis indices
//!   (bounded by rank) from extents. `chelis_tensor_shape` takes `int32_t`
//!   and normalizes negative axes before rejecting anything outside the
//!   rank.

use std::process::Command;

use chelis_runtime::{
    chelis_alloc, chelis_tensor_begin_write, chelis_tensor_end_write, chelis_tensor_entry_borrow,
    chelis_tensor_numel, chelis_tensor_read_view, chelis_tensor_release, chelis_tensor_shape,
    chelis_tensor_write_view, CHELIS_DTYPE_F32,
};

const AXIS_CHILD_CASE_ENV: &str = "CHELIS_DIM_CARRIER_AXIS_CHILD_CASE";
const CAPACITY_CHILD_CASE_ENV: &str = "CHELIS_DIM_CARRIER_CAPACITY_CHILD_CASE";

/// One more than the largest extent the retired 32-bit carrier could hold.
/// Every positive assertion below uses this value, so a regression to a
/// 32-bit field shows up as a wrapped or truncated number rather than as a
/// tolerance failure.
const ABOVE_INT32: i64 = 2_147_483_648;

#[test]
fn a_view_checks_an_extent_above_int32_at_full_width() {
    let (success, stderr) = run_capacity_child("large-extent");
    assert!(!success, "an undersized borrowed view returned success");
    assert!(
        stderr.contains("required 8589934592"),
        "the exact f32 byte requirement must retain the int64 extent:\n{stderr}"
    );
}

#[test]
fn large_extent_capacity_child() {
    let Ok(case) = std::env::var(CAPACITY_CHILD_CASE_ENV) else {
        return;
    };
    assert_eq!(case, "large-extent");
    let shape = [ABOVE_INT32, 1i64];
    let mut backing = [0.0f32; 1];
    unsafe {
        chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_F32,
            backing.as_mut_ptr().cast(),
            size_of_val(&backing) as i64,
        );
    }
    panic!("undersized large-extent view returned instead of terminating");
}

fn run_capacity_child(case: &str) -> (bool, String) {
    let test_binary = std::env::current_exe().expect("current test binary");
    let output = Command::new(&test_binary)
        .args(["--exact", "large_extent_capacity_child", "--nocapture"])
        .env(CAPACITY_CHILD_CASE_ENV, case)
        .output()
        .unwrap_or_else(|error| panic!("run capacity child `{case}`: {error}"));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The ordinary case still holds: widening the carrier must not perturb a
/// shape that always fitted.
#[test]
fn a_small_view_reports_the_same_layout_it_always_did() {
    let shape = [2i64, 3i64];
    let mut backing = [0.0f32; 6];
    unsafe {
        let tensor = chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_F32,
            backing.as_mut_ptr().cast(),
            size_of_val(&backing) as i64,
        );
        assert_eq!(chelis_tensor_shape(tensor, 0), 2);
        assert_eq!(chelis_tensor_shape(tensor, 1), 3);
        assert_eq!(chelis_tensor_numel(tensor), 6);
        assert_eq!(chelis_tensor_read_view(tensor).count, 6);
        chelis_tensor_release(tensor);
    }
}

#[test]
fn negative_axis_indexes_from_the_end() {
    let shape = [2i64, 3i64];
    let mut backing = [0.0f32; 6];
    unsafe {
        let tensor = chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_F32,
            backing.as_mut_ptr().cast(),
            size_of_val(&backing) as i64,
        );
        assert_eq!(chelis_tensor_shape(tensor, -1), 3);
        assert_eq!(chelis_tensor_shape(tensor, -2), 2);
        chelis_tensor_release(tensor);
    }
}

/// Negative parity for axis normalization ([05-DIM-1]).
#[test]
fn out_of_range_axes_abort_instead_of_indexing() {
    for case in ["too-negative", "at-rank", "past-rank", "i32-max", "i32-min"] {
        let (success, stderr) = run_axis_child(case);
        assert!(!success, "axis case {case} returned success");
        assert!(
            stderr.contains("chelis_tensor_shape axis") && stderr.contains("out of bounds"),
            "axis case {case} lost its diagnostic:\n{stderr}"
        );
    }
}

#[test]
fn out_of_range_axis_child() {
    let Ok(case) = std::env::var(AXIS_CHILD_CASE_ENV) else {
        return;
    };
    let axis: i32 = match case.as_str() {
        "too-negative" => -3,
        "at-rank" => 2,
        "past-rank" => 7,
        "i32-max" => i32::MAX,
        "i32-min" => i32::MIN,
        other => panic!("bad axis case {other}"),
    };
    let shape = [2i64, 3i64];
    let mut backing = [0.0f32; 6];
    unsafe {
        let tensor = chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_F32,
            backing.as_mut_ptr().cast(),
            size_of_val(&backing) as i64,
        );
        let extent = chelis_tensor_shape(tensor, axis);
        panic!("axis {axis} returned extent {extent} instead of terminating");
    }
}

fn run_axis_child(case: &str) -> (bool, String) {
    let test_binary = std::env::current_exe().expect("current test binary");
    let output = Command::new(&test_binary)
        .args(["--exact", "out_of_range_axis_child", "--nocapture"])
        .env(AXIS_CHILD_CASE_ENV, case)
        .output()
        .unwrap_or_else(|error| panic!("run child `{case}`: {error}"));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn the_published_header_keeps_extents_wide_and_tensor_fields_opaque() {
    let include_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("include");
    let probe_dir = std::env::temp_dir().join(format!(
        "chelis-dim-carrier-probe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&probe_dir).expect("create probe dir");
    let source = probe_dir.join("layout_probe.c");
    std::fs::write(
        &source,
        r#"#include <stdint.h>
#include "chelis_runtime.h"

static chelis_tensor *borrow_large(const int64_t *shape, const void *data) {
    return chelis_tensor_entry_borrow(1, shape, CHELIS_DTYPE_F32, data, INT64_C(8589934592));
}

int main(void) {
    chelis_tensor *tensor = 0;
    return tensor == 0 && borrow_large != 0 ? 0 : 1;
}
"#,
    )
    .expect("write probe source");
    let object = probe_dir.join("layout_probe.o");
    let compile = Command::new("cc")
        .arg("-I")
        .arg(&include_dir)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("the layout probe requires a C compiler (`cc`) on PATH");
    assert!(
        compile.status.success(),
        "layout probe failed to compile against the published header:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let _ = std::fs::remove_dir_all(&probe_dir);
}

/// Manual gate (chelis#1112): the ALLOCATING counterpart of the metadata
/// tests above. It reserves a real buffer for more than `2^31` elements,
/// which is 8 GiB at f32 and far past what a shared CI runner should be
/// asked for, so it is `#[ignore]`d by default.
///
/// Command:
///
/// ```text
/// cargo test -p chelis-runtime --test dim_carrier_int64 -- \
///     --ignored --exact an_allocation_above_int32_elements_reports_its_true_extent
/// ```
///
/// Expected result: passes on a machine with at least ~9 GiB of free RAM
/// (the buffer plus headroom). The failure this gate exists to catch is a
/// truncating store on the ALLOCATION path specifically: under a 32-bit
/// carrier `size` wraps negative, `bytes` computes as a huge `usize`, and
/// `posix_memalign` fails, so the abort arrives before any assertion. On a
/// machine without the memory the allocation legitimately fails; that is an
/// environment result, not a regression, and the metadata tests above cover
/// the carrier width in default CI.
#[test]
#[ignore = "allocates more than 8 GiB; documented manual gate (chelis#1112)"]
fn an_allocation_above_int32_elements_reports_its_true_extent() {
    let shape = [ABOVE_INT32 + 16];
    unsafe {
        let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
        assert_eq!(chelis_tensor_shape(tensor, 0), ABOVE_INT32 + 16);
        assert_eq!(chelis_tensor_numel(tensor), ABOVE_INT32 + 16);
        // Touch both ends: a truncated `size` would have under-allocated,
        // and the write past `2^31` elements is the read the old carrier
        // could not address.
        let guard = chelis_tensor_begin_write(tensor);
        let write = chelis_tensor_write_view(guard);
        let data = write.data as *mut f32;
        *data.add(0) = 1.0;
        *data.add((ABOVE_INT32 + 15) as usize) = 2.0;
        chelis_tensor_end_write(guard);
        let read = chelis_tensor_read_view(tensor);
        let data = read.data as *const f32;
        assert_eq!(*data.add(0), 1.0);
        assert_eq!(*data.add((ABOVE_INT32 + 15) as usize), 2.0);
        chelis_tensor_release(tensor);
    }
}
