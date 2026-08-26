//! [05-OP-33]'s checked-arithmetic contract, exercised at the *int64* ceiling.
//!
//! The atom requires that "shape products, byte counts, offsets, output
//! extents, and allocation sizes use checked arithmetic" and that "an
//! unrepresentable count, extent, offset, or allocation size traps `Overflow`
//! before allocation or element access", with `split` taking "nonnegative
//! int64 sizes whose checked sum equals the selected extent".
//!
//! Representable means representable in the canonical int64 extent domain
//! ([05-DIM-2]), not "fits this host's `usize`". Those two ceilings differ by
//! the whole `[i64::MAX + 1, u64::MAX]` band on a 64-bit host, and they differ
//! in the other direction on a 32-bit one, so every negative case below sits
//! deliberately *inside* that band rather than above `u64::MAX`. A matrix that
//! only probes the host ceiling reports green while the language ceiling is
//! unguarded.
//!
//! The `split-negative-*` cases also close a heap out-of-bounds read: with the
//! sum equality as the only guard, `[5, -1]` against an extent-4 axis summed
//! to 4, allocated a `[5, ..]` part, and walked its copy loop past the end of
//! the source buffer. Reproduced before the fix with
//! `DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib MALLOC_PROTECT_AFTER=1`,
//! which turned it into SIGSEGV while the legal control on the same shape
//! exited zero.

use chelis_runtime::{
    chelis_alloc, chelis_dims, chelis_list_from_values, chelis_list_index, chelis_list_len,
    chelis_scalar_from_bits, chelis_string_from_cstr, chelis_tensor, chelis_tensor_concat,
    chelis_tensor_einsum, chelis_tensor_split, chelis_value_as_tensor, chelis_value_from_scalar,
    chelis_value_from_tensor, CHELIS_DTYPE_F32, CHELIS_DTYPE_I64,
};
use std::env;
use std::ffi::CString;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_OP33_INT64_EXTENT_CHILD";

/// Above `i64::MAX` (9_223_372_036_854_775_807) and below `u64::MAX`: the exact
/// band a `usize` fold accepts and an int64 fold rejects. 4e9 * 4e9 = 1.6e19.
const BAND_EXTENT: i64 = 4_000_000_000;

/// 2^31 * 2^31 = 2^62, a legal int64 count whose f32 byte size (2^64) is not.
const BUFFER_BAND_EXTENT: i64 = 1 << 31;

unsafe fn tensor(shape: &[i64], dtype: u8) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype)
}

/// A structurally valid carrier that *declares* a large extent and the
/// matching capacity without materializing it.
///
/// `validate_tensor` reads the declared metadata, exactly as it must for a
/// foreign caller, so this is the shape of tensor a C consumer can hand the
/// runtime. Every case built this way must trap before any element access, so
/// the undersized backing buffer is never read; that is the property under
/// test.
struct DeclaredCarrier {
    _shape: Box<[i64]>,
    _strides: Box<[i64]>,
    _data: Box<[f32]>,
    tensor: chelis_tensor,
}

fn declared_carrier(shape: &[i64]) -> DeclaredCarrier {
    let mut strides = vec![0_i64; shape.len()];
    let mut running = 1_i64;
    for axis in (0..shape.len()).rev() {
        strides[axis] = running;
        running = running
            .checked_mul(shape[axis])
            .expect("fixture stride fits int64");
    }
    let size = shape
        .iter()
        .copied()
        .try_fold(1_i64, i64::checked_mul)
        .expect("fixture element count fits int64");
    let data = vec![0.0_f32; 8].into_boxed_slice();
    let shape = shape.to_vec().into_boxed_slice();
    let strides = strides.into_boxed_slice();
    let tensor = chelis_tensor {
        data: if size == 0 {
            std::ptr::null_mut()
        } else {
            data.as_ptr() as *mut u8
        },
        shape: chelis_dims(shape.as_ptr()),
        strides: chelis_dims(strides.as_ptr()),
        size,
        byte_capacity: size.checked_mul(4).expect("fixture capacity fits int64"),
        rank: shape.len() as i32,
        dtype: CHELIS_DTYPE_F32,
        owns_data: 0,
        reserved: [0; 2],
    };
    DeclaredCarrier {
        _shape: shape,
        _strides: strides,
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
            chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, *size as u64))
        })
        .collect::<Vec<_>>();
    chelis_list_from_values(values.as_ptr(), values.len() as i64)
}

fn run_case(case: &str) -> ! {
    unsafe {
        match case {
            // Reduction count 4e9 * 4e9 = 1.6e19: inside the band.
            "einsum-reduction-at-int64-ceiling" => {
                let lhs = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
                let rhs = tensor(&[BAND_EXTENT, 0], CHELIS_DTYPE_F32);
                einsum("az,cw->zw", lhs, rhs);
            }
            // Output extents 4e9 x 4e9: the same band on the output leg.
            "einsum-output-at-int64-ceiling" => {
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
            // Reduction count 2^62 is a legal int64 extent product; the f32
            // accumulation buffer it names is 2^64 bytes and is not.
            "einsum-reduction-buffer-byte-size" => {
                let lhs = declared_carrier(&[BUFFER_BAND_EXTENT, 1]);
                let rhs = declared_carrier(&[BUFFER_BAND_EXTENT, 1]);
                einsum("az,cw->zw", &lhs.tensor, &rhs.tensor);
            }
            // i64::MAX + 1 through the size list.
            "split-size-sum-at-int64-ceiling" => {
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
            "concat-output-extent-at-int64-ceiling" => {
                let first = tensor(&[i64::MAX, 0], CHELIS_DTYPE_F32);
                let second = tensor(&[1, 0], CHELIS_DTYPE_F32);
                let parts = [
                    chelis_value_from_tensor(first),
                    chelis_value_from_tensor(second),
                ];
                chelis_tensor_concat(
                    chelis_list_from_values(parts.as_ptr(), parts.len() as i64),
                    0,
                );
            }
            other => panic!("unknown int64-extent-domain case: {other}"),
        }
    }
    panic!("case `{case}` returned instead of trapping")
}

/// Each negative case names the brand and the exact diagnostic it owes, so a
/// future edit cannot satisfy the matrix by trapping for a different reason.
const NEGATIVE_MATRIX: &[(&str, &str)] = &[
    (
        "einsum-reduction-at-int64-ceiling",
        "Overflow: einsum reduction extent product exceeds int64",
    ),
    (
        "einsum-output-at-int64-ceiling",
        "Overflow: einsum output extent product exceeds int64",
    ),
    (
        "einsum-reduction-above-host-ceiling",
        "Overflow: einsum reduction extent product exceeds int64",
    ),
    (
        "einsum-reduction-buffer-byte-size",
        "Overflow: einsum reduction buffer byte size exceeds int64",
    ),
    (
        "split-size-sum-at-int64-ceiling",
        "Overflow: split size sum exceeds int64",
    ),
    (
        "split-negative-size-compensating",
        "Domain: split expects nonnegative int64 sizes, got -1",
    ),
    (
        "split-negative-size-leading",
        "Domain: split expects nonnegative int64 sizes, got -1",
    ),
    (
        "concat-output-extent-at-int64-ceiling",
        "Overflow: concat output extent exceeds int64",
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
        "{}/{} int64 extent-ceiling cases failed:\n{}",
        failures.len(),
        NEGATIVE_MATRIX.len(),
        failures.join("\n")
    );
}

#[test]
fn legal_products_sums_and_zero_extents_still_execute_exactly() {
    unsafe {
        // einsum: a legal contraction still produces its exact value.
        let lhs = tensor(&[3], CHELIS_DTYPE_F32);
        let rhs = tensor(&[3], CHELIS_DTYPE_F32);
        (*lhs)
            .data
            .cast::<f32>()
            .copy_from([1.0_f32, 2.0, 3.0].as_ptr(), 3);
        (*rhs)
            .data
            .cast::<f32>()
            .copy_from([4.0_f32, 5.0, 6.0].as_ptr(), 3);
        let text = CString::new("i,i->").expect("equation is C-compatible");
        let contracted = chelis_tensor_einsum(
            chelis_string_from_cstr(text.as_ptr()),
            lhs,
            rhs,
            CHELIS_DTYPE_F32,
        );
        assert_eq!((*contracted).rank, 0);
        assert_eq!(*(*contracted).data.cast::<f32>(), 32.0_f32);

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
        assert_eq!((*empty).size, 0);

        // split: nonnegative sizes summing to the extent, including a zero.
        let source = tensor(&[4, 1], CHELIS_DTYPE_F32);
        (*source)
            .data
            .cast::<f32>()
            .copy_from([10.0_f32, 20.0, 30.0, 40.0].as_ptr(), 4);
        let parts = chelis_tensor_split(source, 0, int_size_list(&[0, 3, 1]));
        assert_eq!(chelis_list_len(parts), 3);
        let first = chelis_value_as_tensor(chelis_list_index(parts, 0));
        assert_eq!((*first).size, 0);
        let middle = chelis_value_as_tensor(chelis_list_index(parts, 1));
        assert_eq!(
            std::slice::from_raw_parts((*middle).data.cast::<f32>(), 3),
            &[10.0, 20.0, 30.0]
        );
        let last = chelis_value_as_tensor(chelis_list_index(parts, 2));
        assert_eq!(
            std::slice::from_raw_parts((*last).data.cast::<f32>(), 1),
            &[40.0]
        );

        // concat: extents that sum inside int64, including a zero-extent part.
        let empty_part = tensor(&[0, 2], CHELIS_DTYPE_F32);
        let filled_part = tensor(&[2, 2], CHELIS_DTYPE_F32);
        (*filled_part)
            .data
            .cast::<f32>()
            .copy_from([1.0_f32, 2.0, 3.0, 4.0].as_ptr(), 4);
        let values = [
            chelis_value_from_tensor(empty_part),
            chelis_value_from_tensor(filled_part),
        ];
        let joined = chelis_tensor_concat(
            chelis_list_from_values(values.as_ptr(), values.len() as i64),
            0,
        );
        assert_eq!((*joined).size, 4);
        assert_eq!(
            std::slice::from_raw_parts((*joined).data.cast::<f32>(), 4),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }
}
