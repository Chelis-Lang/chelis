//! [05-OP-33]'s checked-arithmetic contract, exercised at the *i64* ceiling.
//!
//! The atom requires that "shape products, byte counts, offsets, output
//! extents, and allocation sizes use checked arithmetic" and that "an
//! unrepresentable count, extent, offset, or allocation size traps `Overflow`
//! before allocation or element access", with `split` taking "nonnegative
//! i64 sizes whose checked sum equals the selected extent".
//!
//! Representable means representable in the canonical i64 extent domain
//! ([05-DIM-2]), not "fits this host's `usize`". Those two ceilings differ by
//! the whole `[i64::MAX + 1, u64::MAX]` band on a 64-bit host, and they differ
//! in the other direction on a 32-bit one, so every negative case below sits
//! deliberately *inside* that band rather than above `u64::MAX`. A matrix that
//! only probes the host ceiling reports green while the language ceiling is
//! unguarded.
//!
//! The zero-position permutations are the other half of the contract. A count
//! is a product, not a running prefix: [05-OP-33] says a zero extent means zero
//! elements, so `[i64::MAX, i64::MAX, 0]` and `[i64::MAX, 0, i64::MAX]`
//! describe the same empty tensor and must be accepted alike. A left-to-right
//! checked fold accepted one and rejected the other, and no checked-in control
//! caught it because none permuted the zero. The `[0, i64::MAX, i64::MAX]`
//! case is deliberately still a trap: its canonical axis-0 stride is the exact
//! product of the following extents ([05-OP-31]), which really is
//! unrepresentable, so the size repair must not mask it.
//!
//! The `split-negative-*` cases also close a heap out-of-bounds read: with the
//! sum equality as the only guard, `[5, -1]` against an extent-4 axis summed
//! to 4, allocated a `[5, ..]` part, and walked its copy loop past the end of
//! the source buffer. Reproduced before the fix with
//! `DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib MALLOC_PROTECT_AFTER=1`,
//! which turned it into SIGSEGV while the legal control on the same shape
//! exited zero.

use chelis_runtime::{
    chelis_alloc, chelis_list_from_values, chelis_list_index, chelis_list_len,
    chelis_scalar_from_bits, chelis_string_from_cstr, chelis_tensor, chelis_tensor_begin_write,
    chelis_tensor_borrow_value, chelis_tensor_concat, chelis_tensor_einsum,
    chelis_tensor_end_write, chelis_tensor_entry_borrow, chelis_tensor_numel, chelis_tensor_rank,
    chelis_tensor_read_view, chelis_tensor_shape, chelis_tensor_split, chelis_tensor_write_view,
    chelis_value_box_scalar, chelis_value_take_tensor, CHELIS_DTYPE_F32, CHELIS_DTYPE_I64,
};
use std::env;
use std::ffi::CString;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_OP33_INT64_EXTENT_CHILD";

/// Above `i64::MAX` (9_223_372_036_854_775_807) and below `u64::MAX`: the exact
/// band a `usize` fold accepts and an i64 fold rejects. 4e9 * 4e9 = 1.6e19.
const BAND_EXTENT: i64 = 4_000_000_000;

/// 2^31 * 2^31 = 2^62, a legal i64 count whose f32 byte size (2^64) is not.
const BUFFER_BAND_EXTENT: i64 = 1 << 31;

unsafe fn tensor(shape: &[i64], dtype: u8) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype)
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

unsafe fn read_f32(tensor: *const chelis_tensor) -> Vec<f32> {
    let view = chelis_tensor_read_view(tensor);
    std::slice::from_raw_parts(view.data.cast::<f32>(), view.count as usize).to_vec()
}

/// A structurally valid carrier that *declares* a large extent and the
/// matching capacity without materializing it.
///
/// The entry-borrow boundary validates the declaration and constructs an
/// opaque descriptor. Every case built this way must trap before element access, so
/// the undersized backing buffer is never read; that is the property under
/// test.
struct DeclaredCarrier {
    _data: Box<[f32]>,
    tensor: *mut chelis_tensor,
}

unsafe fn declared_carrier(shape: &[i64]) -> DeclaredCarrier {
    let size = shape
        .iter()
        .copied()
        .try_fold(1_i64, i64::checked_mul)
        .expect("fixture element count fits i64");
    let data = vec![0.0_f32; 8].into_boxed_slice();
    let pointer: *const std::ffi::c_void = if size == 0 {
        std::ptr::null()
    } else {
        data.as_ptr().cast()
    };
    let tensor = chelis_tensor_entry_borrow(
        shape.len() as i32,
        shape.as_ptr(),
        CHELIS_DTYPE_F32,
        pointer,
        size.checked_mul(4).expect("fixture capacity fits i64"),
    );
    DeclaredCarrier {
        _data: data,
        tensor,
    }
}

unsafe fn einsum(equation: &str, lhs: *const chelis_tensor, rhs: *const chelis_tensor) {
    let text = CString::new(equation).expect("equation is C-compatible");
    chelis_tensor_einsum(
        chelis_string_from_cstr(text.as_ptr()),
        lhs,
        rhs,
        CHELIS_DTYPE_F32,
    );
}

unsafe fn int_size_list(sizes: &[i64]) -> *mut chelis_runtime::chelis_list {
    let values = sizes
        .iter()
        .map(|size| {
            chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, *size as u64))
        })
        .collect::<Vec<_>>();
    chelis_list_from_values(values.as_ptr(), values.len() as i64)
}

fn run_case(case: &str) -> ! {
    unsafe {
        match case {
            // Reduction count 4e9 * 4e9 = 1.6e19: inside the band.
            "einsum-reduction-at-i64-ceiling" => {
                let lhs = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
                let rhs = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
                einsum("az,cw->zw", lhs, rhs);
            }
            // Output extents 4e9 x 4e9: the same band on the output leg.
            "einsum-output-at-i64-ceiling" => {
                let lhs = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
                let rhs = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
                einsum("az,cw->ac", lhs, rhs);
            }
            // 5e9 * 5e9 = 2.5e19 clears `u64::MAX` too. Kept so the ceiling
            // move cannot regress the case that already trapped.
            "einsum-reduction-above-host-ceiling" => {
                let lhs = tensor(&[5_000_000_000, 0], CHELIS_DTYPE_F32);
                let rhs = tensor(&[5_000_000_000, 0], CHELIS_DTYPE_F32);
                einsum("az,cw->zw", lhs, rhs);
            }
            // Reduction count 2^62 is a legal i64 extent product; the f32
            // accumulation buffer it names is 2^64 bytes and is not.
            "einsum-reduction-buffer-byte-size" => {
                let lhs = declared_carrier(&[BUFFER_BAND_EXTENT, 1]);
                let rhs = declared_carrier(&[BUFFER_BAND_EXTENT, 1]);
                einsum("az,cw->zw", lhs.tensor, rhs.tensor);
            }
            // i64::MAX + 1 through the size list.
            "split-size-sum-at-i64-ceiling" => {
                let input = tensor(&[i64::MAX, 0], CHELIS_DTYPE_F32);
                chelis_tensor_split(input, 0, int_size_list(&[i64::MAX, 1]));
            }
            // Sum equality holds at 4, but part 0 exceeds the source extent.
            "split-negative-size-compensating" => {
                let input = tensor(&[4, 2], CHELIS_DTYPE_F32);
                chelis_tensor_split(input, 0, int_size_list(&[5, -1]));
            }
            // Same, with the negative size first, so no part is built at all.
            "split-negative-size-leading" => {
                let input = tensor(&[4, 2], CHELIS_DTYPE_F32);
                chelis_tensor_split(input, 0, int_size_list(&[-1, 5]));
            }
            // i64::MAX + 1 as a concatenated output extent.
            "concat-output-extent-at-i64-ceiling" => {
                let first = tensor(&[i64::MAX, 0], CHELIS_DTYPE_F32);
                let second = tensor(&[1, 0], CHELIS_DTYPE_F32);
                let parts = [
                    chelis_value_take_tensor(first),
                    chelis_value_take_tensor(second),
                ];
                chelis_tensor_concat(
                    chelis_list_from_values(parts.as_ptr(), parts.len() as i64),
                    0,
                );
            }
            // A zero extent does not make an unrepresentable canonical stride
            // legal: axis 0's stride is the exact product of the following
            // extents, and i64::MAX * i64::MAX is not an i64.
            "alloc-unrepresentable-canonical-stride" => {
                tensor(&[0, i64::MAX, i64::MAX], CHELIS_DTYPE_F32);
            }
            // No zero extent, so the product really is unrepresentable.
            "alloc-extent-product-at-i64-ceiling" => {
                tensor(&[i64::MAX, 2], CHELIS_DTYPE_F32);
            }
            other => panic!("unknown i64-extent-domain case: {other}"),
        }
    }
    panic!("case `{case}` returned instead of trapping")
}

/// Each negative case names the brand and the exact diagnostic it owes, so a
/// future edit cannot satisfy the matrix by trapping for a different reason.
const NEGATIVE_MATRIX: &[(&str, &str)] = &[
    (
        "einsum-reduction-at-i64-ceiling",
        "Overflow: einsum reduction extent product exceeds i64",
    ),
    (
        "einsum-output-at-i64-ceiling",
        "Overflow: einsum output extent product exceeds i64",
    ),
    (
        "einsum-reduction-above-host-ceiling",
        "Overflow: einsum reduction extent product exceeds i64",
    ),
    (
        "einsum-reduction-buffer-byte-size",
        "Overflow: einsum reduction buffer byte size exceeds i64",
    ),
    (
        "split-size-sum-at-i64-ceiling",
        "Overflow: split size sum exceeds i64",
    ),
    (
        "split-negative-size-compensating",
        "Domain: split expects nonnegative i64 sizes, got -1",
    ),
    (
        "split-negative-size-leading",
        "Domain: split expects nonnegative i64 sizes, got -1",
    ),
    (
        "concat-output-extent-at-i64-ceiling",
        "Overflow: concat output extent exceeds i64",
    ),
    (
        "alloc-unrepresentable-canonical-stride",
        "Overflow: chelis_alloc stride product exceeds i64",
    ),
    (
        "alloc-extent-product-at-i64-ceiling",
        "Overflow: chelis_alloc extent product exceeds i64",
    ),
];

#[test]
fn derived_products_and_sums_trap_at_the_int64_extent_ceiling() {
    if let Ok(case) = env::var(CHILD_ENV) {
        run_case(&case);
    }
    let test_binary = env::current_exe().expect("current test binary");
    let mut failures = Vec::new();
    for (case, expected) in NEGATIVE_MATRIX {
        let output = Command::new(&test_binary)
            .env(CHILD_ENV, case)
            .arg("--exact")
            .arg("derived_products_and_sums_trap_at_the_int64_extent_ceiling")
            .arg("--nocapture")
            .output()
            .unwrap_or_else(|error| panic!("run child `{case}`: {error}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() {
            failures.push(format!("{case}: returned success instead of trapping"));
        } else if !stderr.contains(expected) {
            failures.push(format!(
                "{case}: expected `{expected}`, got status {}:\n{stderr}",
                output.status
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} i64 extent-ceiling cases failed:\n{}",
        failures.len(),
        NEGATIVE_MATRIX.len(),
        failures.join("\n")
    );
}

/// A zero extent means zero elements wherever the zero sits, so every
/// permutation of the same extents must be accepted alike, with identical
/// size, capacity, null data, and canonical strides.
#[test]
fn zero_extent_acceptance_does_not_depend_on_axis_order() {
    unsafe {
        for shape in [
            vec![i64::MAX, 0, i64::MAX],
            vec![i64::MAX, i64::MAX, 0],
            vec![BAND_EXTENT, 0, BAND_EXTENT],
            vec![BAND_EXTENT, BAND_EXTENT, 0],
            vec![0, 0],
            vec![0],
        ] {
            let allocated = tensor(&shape, CHELIS_DTYPE_F32);
            assert_eq!(
                chelis_tensor_numel(allocated),
                0,
                "shape {shape:?} must hold no elements"
            );
            let view = chelis_tensor_read_view(allocated);
            assert!(view.data.is_null(), "shape {shape:?} must carry null data");
            assert_eq!(chelis_tensor_rank(allocated), shape.len() as i32);
            for (axis, extent) in shape.iter().copied().enumerate() {
                assert_eq!(chelis_tensor_shape(allocated, axis as i32), extent);
            }
        }

        // The same permutation invariance through a derived einsum count. The
        // output label sits on the zero axis, so the reduction extents are the
        // two `BAND_EXTENT` axes of each operand plus one zero; their product
        // is zero, but a left-to-right fold reaches `BAND_EXTENT` squared
        // first, which is not an i64. Only the zero's position differs
        // between the two cases.
        for (equation_shape, equation) in [
            ([BAND_EXTENT, 0, BAND_EXTENT], "abc,def->b"),
            ([BAND_EXTENT, BAND_EXTENT, 0], "abc,def->c"),
        ] {
            let lhs = tensor(&equation_shape, CHELIS_DTYPE_F32);
            let rhs = tensor(&equation_shape, CHELIS_DTYPE_F32);
            let text = CString::new(equation).expect("equation is C-compatible");
            let contracted = chelis_tensor_einsum(
                chelis_string_from_cstr(text.as_ptr()),
                lhs,
                rhs,
                CHELIS_DTYPE_F32,
            );
            assert_eq!(
                chelis_tensor_numel(contracted),
                0,
                "einsum `{equation}` over {equation_shape:?} must produce an empty result"
            );
        }
    }
}

#[test]
fn legal_products_sums_and_zero_extents_still_execute_exactly() {
    unsafe {
        // einsum: a legal contraction still produces its exact value.
        let lhs = tensor(&[3], CHELIS_DTYPE_F32);
        let rhs = tensor(&[3], CHELIS_DTYPE_F32);
        write_f32(lhs, &[1.0_f32, 2.0, 3.0]);
        write_f32(rhs, &[4.0_f32, 5.0, 6.0]);
        let text = CString::new("i,i->").expect("equation is C-compatible");
        let contracted = chelis_tensor_einsum(
            chelis_string_from_cstr(text.as_ptr()),
            lhs,
            rhs,
            CHELIS_DTYPE_F32,
        );
        assert_eq!(chelis_tensor_rank(contracted), 0);
        assert_eq!(read_f32(contracted), [32.0_f32]);

        // einsum: a large but representable extent with a zero-size partner
        // is legal and must not be swept up by the ceiling checks.
        let wide = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
        let narrow = tensor(&[2, 0], CHELIS_DTYPE_F32);
        let text = CString::new("az,cw->zw").expect("equation is C-compatible");
        let empty = chelis_tensor_einsum(
            chelis_string_from_cstr(text.as_ptr()),
            wide,
            narrow,
            CHELIS_DTYPE_F32,
        );
        assert_eq!(chelis_tensor_numel(empty), 0);

        // split: nonnegative sizes summing to the extent, including a zero.
        let source = tensor(&[4, 1], CHELIS_DTYPE_F32);
        write_f32(source, &[10.0_f32, 20.0, 30.0, 40.0]);
        let parts = chelis_tensor_split(source, 0, int_size_list(&[0, 3, 1]));
        assert_eq!(chelis_list_len(parts), 3);
        let first = chelis_tensor_borrow_value(chelis_list_index(parts, 0));
        assert_eq!(chelis_tensor_numel(first), 0);
        let middle = chelis_tensor_borrow_value(chelis_list_index(parts, 1));
        assert_eq!(read_f32(middle), [10.0, 20.0, 30.0]);
        let last = chelis_tensor_borrow_value(chelis_list_index(parts, 2));
        assert_eq!(read_f32(last), [40.0]);

        // concat: extents that sum inside i64, including a zero-extent part.
        let empty_part = tensor(&[0, 2], CHELIS_DTYPE_F32);
        let filled_part = tensor(&[2, 2], CHELIS_DTYPE_F32);
        write_f32(filled_part, &[1.0_f32, 2.0, 3.0, 4.0]);
        let values = [
            chelis_value_take_tensor(empty_part),
            chelis_value_take_tensor(filled_part),
        ];
        let joined = chelis_tensor_concat(
            chelis_list_from_values(values.as_ptr(), values.len() as i64),
            0,
        );
        assert_eq!(chelis_tensor_numel(joined), 4);
        assert_eq!(read_f32(joined), [1.0, 2.0, 3.0, 4.0]);
    }
}
