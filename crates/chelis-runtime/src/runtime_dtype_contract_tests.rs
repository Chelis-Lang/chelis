use chelis_vocab::{RuntimeDType, RuntimeDTypeDecodeError};

use chelis_crmath::profile::{rows, storage_midpoint_inputs, storage_reference, Output, Row};

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

fn accumulator_bits(bits: u64, output: Output) -> u16 {
    let value = f64::from_bits(bits);
    match output {
        Output::F16 => <half::f16 as RuntimeAccumulationOutput<f64>>::from_accumulator(value).to_bits(),
        Output::Bf16 => <half::bf16 as RuntimeAccumulationOutput<f64>>::from_accumulator(value).to_bits(),
        _ => unreachable!("not a storage format"),
    }
}

/// The f64 accumulator's finalization at every f16 and bf16 rounding boundary, against
/// the integer reference.
#[test]
fn f64_accumulator_matches_the_reference_at_every_boundary() {
    for output in [Output::F16, Output::Bf16] {
        for bits in storage_midpoint_inputs(output) {
            assert_eq!(
                accumulator_bits(bits, output),
                storage_reference(bits, 64, output),
                "input {bits:x}"
            );
        }
    }
}

/// Manual gate (`docs/manual_gates.md`): every f32 bit pattern through the f32 and
/// the f64 accumulators' finalization, against the integer reference.
#[test]
#[ignore = "manual gate: 2^32 inputs (docs/manual_gates.md)"]
fn every_f32_finalizes_once_to_f16_and_bf16_storage() {
    let threads = std::thread::available_parallelism().map_or(1, |count| (count.get() / 2).max(1));
    let span = (1_u64 << 32) / threads as u64 + 1;
    let total: u64 = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads as u64)
            .map(|worker| {
                scope.spawn(move || {
                    let mut count = 0_u64;
                    for bits in worker * span..((worker + 1) * span).min(1 << 32) {
                        let value = f32::from_bits(u32::try_from(bits).unwrap());
                        let wide = f64::from(value).to_bits();
                        for output in [Output::F16, Output::Bf16] {
                            let expected = storage_reference(bits, 32, output);
                            let narrow = match output {
                                Output::F16 => finalize_f16(value).to_bits(),
                                _ => finalize_bf16(value).to_bits(),
                            };
                            count += u64::from(narrow != expected);
                            count += u64::from(accumulator_bits(wide, output) != expected);
                        }
                    }
                    count
                })
            })
            .collect();
        workers.into_iter().map(|worker| worker.join().unwrap()).sum()
    });
    assert_eq!(total, 0, "{total} misrounded finalizations over 2^32 inputs");
}
