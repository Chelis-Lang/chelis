use chelis_vocab::{RuntimeDType, RuntimeDTypeDecodeError};

use chelis_crmath::profile::{rows, Row};

use super::{
    chelis_tensor, finalize_bf16, finalize_f16, read_index_slot, tensor_elem_size,
    RuntimeAccumulationOutput,
};

#[test]
fn runtime_dtype_ffi_decoder_rejects_invalid_ids_before_dispatch() {
    let first_unknown = RuntimeDType::ALL
        .iter()
        .map(|dtype| dtype.id())
        .max()
        .expect("non-empty runtime dtype vocabulary")
        + 1;
    for raw in [first_unknown as u8, u8::MAX] {
        assert_eq!(
            super::decode_runtime_dtype(raw),
            Err(RuntimeDTypeDecodeError::InvalidId { id: i32::from(raw) })
        );
    }
}

#[test]
fn internal_sizing_and_reading_helpers_require_a_decoded_dtype() {
    let _: fn(RuntimeDType) -> usize = tensor_elem_size;
    let _: unsafe fn(*const chelis_tensor, usize, RuntimeDType) -> i64 = read_index_slot;
}

// The f16 and bf16 storage finalization against the floating-point profile's
// obligation rows (`chelis_crmath::profile`, chelis#2957 and chelis#3041). The rows
// are MPFR-derived: each f32 or f64 value rounded once, ties to even, to storage,
// with NaN results canonical ([04-NUM-2], [04-NUM-14]). An f32 row checks the f32
// accumulator's finalization; an f64 row checks the f64 accumulator's, which
// `einsum_loop` uses for f16 and bf16 outputs.

fn storage_bits(row: &Row) -> u64 {
    let operand = row.operands[0];
    let f32_operand = || f32::from_bits(u32::try_from(operand).expect("an f32 operand"));
    let f64_operand = || f64::from_bits(operand);
    let bits = match (row.primitive.name, row.primitive.width) {
        ("to_f16", 32) => finalize_f16(f32_operand()).to_bits(),
        ("to_bf16", 32) => finalize_bf16(f32_operand()).to_bits(),
        ("to_f16", 64) => {
            <half::f16 as RuntimeAccumulationOutput<f64>>::from_accumulator(f64_operand()).to_bits()
        }
        ("to_bf16", 64) => {
            <half::bf16 as RuntimeAccumulationOutput<f64>>::from_accumulator(f64_operand())
                .to_bits()
        }
        _ => unreachable!("{} is not a storage conversion", row.primitive),
    };
    u64::from(bits)
}

fn conversion_rows() -> Vec<&'static Row> {
    rows()
        .iter()
        .filter(|row| matches!(row.primitive.name, "to_f16" | "to_bf16"))
        .collect()
}

#[test]
fn storage_finalization_matches_every_conversion_row() {
    let rows = conversion_rows();
    for (name, width) in [
        ("to_f16", 32),
        ("to_bf16", 32),
        ("to_f16", 64),
        ("to_bf16", 64),
    ] {
        assert!(
            rows.iter()
                .any(|row| row.primitive.name == name && row.primitive.width == width),
            "no {name} f{width} rows"
        );
    }
    let bad: Vec<String> = rows
        .iter()
        .filter(|row| storage_bits(row) != row.expected)
        .map(|row| {
            format!(
                "{row}: got 0x{:04x}, expected 0x{:04x}",
                storage_bits(row),
                row.expected
            )
        })
        .collect();
    assert!(
        bad.is_empty(),
        "{} rows differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// chelis#3041: an f64 within 2^-30 of a bf16 midpoint, above it. A conversion that
/// rounds to f32 first lands exactly on the midpoint and ties to even (0x3f80), and
/// `half::bf16::from_f64` also returns 0x3f80.
#[test]
fn f64_accumulator_rounds_to_bf16_once() {
    let value = 1.0 + 2.0_f64.powi(-8) + 2.0_f64.powi(-30);
    assert_eq!(
        <half::bf16 as RuntimeAccumulationOutput<f64>>::from_accumulator(value).to_bits(),
        0x3f81
    );
    assert_eq!(
        <half::bf16 as RuntimeAccumulationOutput<f64>>::from_accumulator(-value).to_bits(),
        0xbf81
    );
    let value = 1.0 + 3.0 * 2.0_f64.powi(-11) - 2.0_f64.powi(-30);
    assert_eq!(
        <half::f16 as RuntimeAccumulationOutput<f64>>::from_accumulator(value).to_bits(),
        0x3c01
    );
}
