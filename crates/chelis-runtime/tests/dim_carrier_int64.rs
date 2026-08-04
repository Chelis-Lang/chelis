//! chelis#1112: the C lane's dimension carrier is int64.
//!
//! `spec/05-risc-primitives.md` [05-DIM-2] declares the extent dtype int64
//! and `spec/04-type-system.md` [04-NUM-11] requires a value to cross every
//! boundary at its declared dtype. Before this change the C ABI carried
//! `shape`, `strides`, and `size` as 32-bit `int`, so an extent above
//! `2^31 - 1` could not cross at all: `chelis_host_reshape_tensor` had to
//! trap on it to avoid a silent truncating store.
//!
//! Three obligations, checked separately because they fail for different
//! reasons:
//!
//! * **the carrier holds the value**: an extent above `2^31 - 1` survives
//!   allocation, the shape accessor, and the element count without
//!   truncation. Checked on metadata alone (`chelis_alloc_view` over a
//!   borrowed buffer), so the positive case costs no memory and runs in
//!   default CI. The allocating counterpart is the `#[ignore]`d manual gate
//!   at the bottom.
//! * **the two mirrors agree**: the Rust `#[repr(C)]` struct and the
//!   published `chelis_runtime.h` describe the same bytes. A widening
//!   applied to one side only produces silent field-offset corruption in
//!   every compiled program, which no Rust-only test can see.
//! * **the axis domain stayed narrow**: [05-DIM-1] separates axis indices
//!   (bounded by rank) from extents. `chelis_tensor_shape` takes `int32_t`
//!   and must still reject an axis outside `[0, ndim)` rather than
//!   indexing.

use std::process::Command;

use chelis_runtime::{
    chelis_alloc, chelis_alloc_view, chelis_free, chelis_tensor, chelis_tensor_numel,
    chelis_tensor_rank, chelis_tensor_shape, CHELIS_F32,
};

const AXIS_CHILD_CASE_ENV: &str = "CHELIS_DIM_CARRIER_AXIS_CHILD_CASE";

/// One more than the largest extent the retired 32-bit carrier could hold.
/// Every positive assertion below uses this value, so a regression to a
/// 32-bit field shows up as a wrapped or truncated number rather than as a
/// tolerance failure.
const ABOVE_INT32: i64 = 2_147_483_648;

/// Metadata-only: `chelis_alloc_view` borrows the caller's buffer and
/// allocates nothing, so a shape whose extents exceed the old carrier can
/// be constructed and read back without reserving the memory such a tensor
/// would really need. Under the retired `int` carrier `shape[0]` stored
/// `-2147483648` and `numel` followed it negative.
#[test]
fn a_view_carries_an_extent_above_int32_without_truncation() {
    let shape = [ABOVE_INT32, 1i64];
    let mut backing = [0.0f32; 1];
    unsafe {
        let tensor = chelis_alloc_view(2, shape.as_ptr(), CHELIS_F32, backing.as_mut_ptr());
        assert_eq!(chelis_tensor_rank(tensor), 2);
        assert_eq!(
            chelis_tensor_shape(tensor, 0),
            ABOVE_INT32,
            "the leading extent must survive the carrier exactly"
        );
        assert_eq!(chelis_tensor_shape(tensor, 1), 1);
        assert_eq!(
            chelis_tensor_numel(tensor),
            ABOVE_INT32,
            "the element count is the product of the extents at full width"
        );
        assert_eq!(
            (*tensor).strides[0],
            1,
            "the row-major stride of the leading axis of a `[n, 1]` view"
        );
        assert_eq!((*tensor).strides[1], 1, "the innermost stride is always 1");
        chelis_free(tensor);
    }
}

/// The stride domain is the extent domain: an inner extent above the old
/// carrier makes the OUTER stride exceed it too, and a 32-bit stride field
/// would wrap where the extent field did not.
#[test]
fn a_view_carries_a_stride_above_int32_without_truncation() {
    let shape = [2i64, ABOVE_INT32];
    let mut backing = [0.0f32; 1];
    unsafe {
        let tensor = chelis_alloc_view(2, shape.as_ptr(), CHELIS_F32, backing.as_mut_ptr());
        assert_eq!(
            (*tensor).strides[0],
            ABOVE_INT32,
            "the outer stride is the inner extent and must not wrap"
        );
        assert_eq!(chelis_tensor_numel(tensor), 2 * ABOVE_INT32);
        chelis_free(tensor);
    }
}

/// The ordinary case still holds: widening the carrier must not perturb a
/// shape that always fitted.
#[test]
fn a_small_view_reports_the_same_layout_it_always_did() {
    let shape = [2i64, 3i64];
    let mut backing = [0.0f32; 6];
    unsafe {
        let tensor = chelis_alloc_view(2, shape.as_ptr(), CHELIS_F32, backing.as_mut_ptr());
        assert_eq!(chelis_tensor_shape(tensor, 0), 2);
        assert_eq!(chelis_tensor_shape(tensor, 1), 3);
        assert_eq!(chelis_tensor_numel(tensor), 6);
        assert_eq!((*tensor).strides[0], 3);
        assert_eq!((*tensor).strides[1], 1);
        chelis_free(tensor);
    }
}

/// Negative parity for the axis narrowing ([05-DIM-1]). The parameter is
/// `int32_t`, so the out-of-range values a caller can still reach are the
/// i32 extrema and any index at or past `ndim`; each must abort with the
/// bounds diagnostic rather than read past the array.
#[test]
fn out_of_range_axes_abort_instead_of_indexing() {
    for case in ["negative", "at-rank", "past-rank", "i32-max", "i32-min"] {
        let (success, stderr) = run_axis_child(case);
        assert!(!success, "axis case {case} returned success");
        assert!(
            stderr.contains("chelis_tensor_shape axis out of bounds"),
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
        "negative" => -1,
        "at-rank" => 2,
        "past-rank" => 7,
        "i32-max" => i32::MAX,
        "i32-min" => i32::MIN,
        other => panic!("bad axis case {other}"),
    };
    let shape = [2i64, 3i64];
    let mut backing = [0.0f32; 6];
    unsafe {
        let tensor = chelis_alloc_view(2, shape.as_ptr(), CHELIS_F32, backing.as_mut_ptr());
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

/// The Rust `#[repr(C)]` mirror and the published header must describe the
/// same bytes. Nothing generates one from the other, so without this the
/// two can be widened independently and every compiled program reads the
/// wrong field offsets while both crates' own tests stay green.
///
/// The probe is compiled by the same `cc` the census and the C-backend
/// compile-run tests already require, and a missing compiler fails loudly
/// for the same reason it does there: a skip would make the mirror
/// unchecked exactly when it matters.
#[test]
fn the_published_header_and_the_rust_mirror_describe_the_same_bytes() {
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
        r#"#include <stddef.h>
#include <stdio.h>
#include "chelis_runtime.h"

int main(void) {
    printf("%zu %zu %zu %zu %zu %zu %zu %zu\n",
           sizeof(chelis_tensor),
           _Alignof(chelis_tensor),
           offsetof(chelis_tensor, data),
           offsetof(chelis_tensor, shape),
           offsetof(chelis_tensor, strides),
           offsetof(chelis_tensor, ndim),
           offsetof(chelis_tensor, dtype),
           offsetof(chelis_tensor, size));
    return 0;
}
"#,
    )
    .expect("write probe source");
    let binary = probe_dir.join("layout_probe");
    let compile = Command::new("cc")
        .arg("-I")
        .arg(&include_dir)
        .arg(&source)
        .arg("-lm")
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("the layout probe requires a C compiler (`cc`) on PATH");
    assert!(
        compile.status.success(),
        "layout probe failed to compile against the published header:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&binary).output().expect("run layout probe");
    assert!(run.status.success(), "layout probe exited nonzero");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let observed: Vec<usize> = stdout
        .split_whitespace()
        .map(|field| field.parse().expect("probe prints decimal sizes"))
        .collect();
    let _ = std::fs::remove_dir_all(&probe_dir);

    let expected = vec![
        std::mem::size_of::<chelis_tensor>(),
        std::mem::align_of::<chelis_tensor>(),
        std::mem::offset_of!(chelis_tensor, data),
        std::mem::offset_of!(chelis_tensor, shape),
        std::mem::offset_of!(chelis_tensor, strides),
        std::mem::offset_of!(chelis_tensor, ndim),
        std::mem::offset_of!(chelis_tensor, dtype),
        std::mem::offset_of!(chelis_tensor, size),
    ];
    assert_eq!(
        observed, expected,
        "the header and the Rust mirror disagree about `chelis_tensor`: \
         [size, align, data, shape, strides, ndim, dtype, size] header={observed:?} \
         rust={expected:?}"
    );
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
        let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_F32);
        assert_eq!(chelis_tensor_shape(tensor, 0), ABOVE_INT32 + 16);
        assert_eq!(chelis_tensor_numel(tensor), ABOVE_INT32 + 16);
        // Touch both ends: a truncated `size` would have under-allocated,
        // and the write past `2^31` elements is the read the old carrier
        // could not address.
        let data = (*tensor).data as *mut f32;
        *data.add(0) = 1.0;
        *data.add((ABOVE_INT32 + 15) as usize) = 2.0;
        assert_eq!(*data.add(0), 1.0);
        assert_eq!(*data.add((ABOVE_INT32 + 15) as usize), 2.0);
        chelis_free(tensor);
    }
}
