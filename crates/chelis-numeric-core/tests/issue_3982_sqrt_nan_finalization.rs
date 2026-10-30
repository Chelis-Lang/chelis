//! [04-NUM-2]/[05-OP-46]: arithmetic finalizes NaNs; ingress and Relu retain bits.
//! Run this target with `--release`: constant fixtures and debug code miss #3982.

use std::hint::black_box;

use chelis_numeric_core::dtype_semantics::{
    FloatUnOp, ScalarValue, TensorStorage, float_tensor_unop, float_unop, tensor_from_le_payload,
};
use chelis_numeric_core::observation::ElementRef;
use chelis_numeric_core::prim::Prim;

struct FloatCase {
    prim: Prim,
    nans: [u64; 4],
    canonical: u64,
    minus_one: u64,
    minus_inf: u64,
    minus_zero: u64,
    plus_inf: u64,
    four: u64,
    two: u64,
}

const FLOATS: [FloatCase; 4] = [
    FloatCase {
        prim: Prim::F16,
        nans: [0xfc01, 0xfe11, 0x7c01, 0x7e11],
        canonical: 0x7e00,
        minus_one: 0xbc00,
        minus_inf: 0xfc00,
        minus_zero: 0x8000,
        plus_inf: 0x7c00,
        four: 0x4400,
        two: 0x4000,
    },
    FloatCase {
        prim: Prim::Bf16,
        nans: [0xff81, 0xffc1, 0x7f81, 0x7fc1],
        canonical: 0x7fc0,
        minus_one: 0xbf80,
        minus_inf: 0xff80,
        minus_zero: 0x8000,
        plus_inf: 0x7f80,
        four: 0x4080,
        two: 0x4000,
    },
    FloatCase {
        prim: Prim::F32,
        nans: [0xff80_0001, 0xffc0_0011, 0x7f80_0001, 0x7fc0_0011],
        canonical: 0x7fc0_0000,
        minus_one: 0xbf80_0000,
        minus_inf: 0xff80_0000,
        minus_zero: 0x8000_0000,
        plus_inf: 0x7f80_0000,
        four: 0x4080_0000,
        two: 0x4000_0000,
    },
    FloatCase {
        prim: Prim::F64,
        nans: [
            0xfff0_0000_0000_0001,
            0xfff8_0000_0000_0011,
            0x7ff0_0000_0000_0001,
            0x7ff8_0000_0000_0011,
        ],
        canonical: 0x7ff8_0000_0000_0000,
        minus_one: 0xbff0_0000_0000_0000,
        minus_inf: 0xfff0_0000_0000_0000,
        minus_zero: 0x8000_0000_0000_0000,
        plus_inf: 0x7ff0_0000_0000_0000,
        four: 0x4010_0000_0000_0000,
        two: 0x4000_0000_0000_0000,
    },
];

fn stored(prim: Prim, words: &[u64]) -> TensorStorage {
    let width = match prim {
        Prim::F16 | Prim::Bf16 => 2,
        Prim::F32 => 4,
        Prim::F64 => 8,
        _ => panic!("floating storage only"),
    };
    let payload: Vec<u8> = words
        .iter()
        .flat_map(|word| word.to_le_bytes()[..width].to_vec())
        .collect();
    tensor_from_le_payload(prim, black_box(&payload)).expect("exact runtime ingress")
}

fn word(value: ScalarValue) -> u64 {
    match value.element_ref() {
        ElementRef::F16(value) => u64::from(value.to_bits()),
        ElementRef::Bf16(value) => u64::from(value.to_bits()),
        ElementRef::F32(value) => u64::from(value.to_bits()),
        ElementRef::F64(value) => value.to_bits(),
        _ => panic!("floating result only"),
    }
}

fn canonical_sqrt(case: &FloatCase) {
    let mut words = case.nans.to_vec();
    words.extend([case.minus_one, case.minus_inf]);
    // Exercise scalar tails and vectorized tensor loops using runtime operands.
    words = words.repeat(4);
    let input = stored(case.prim, black_box(&words));
    let output = float_tensor_unop(black_box(FloatUnOp::Sqrt), black_box(&input))
        .expect("sqrt admits floating storage");
    assert_eq!(output.prim(), case.prim);
    assert_eq!(output.len(), words.len());
    let mut failures = Vec::new();
    for (index, input_word) in words.iter().copied().enumerate() {
        let actual = word(output.scalar_at(index));
        if actual != case.canonical {
            failures.push(format!("tensor({input_word:#x}) = {actual:#x}"));
        }
        let scalar = float_unop(
            black_box(FloatUnOp::Sqrt),
            black_box(input.scalar_at(index)),
        )
        .expect("scalar sqrt admits floating storage");
        let actual = word(scalar);
        if actual != case.canonical {
            failures.push(format!("scalar({input_word:#x}) = {actual:#x}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} sqrt must finalize to {:#x}: {}",
        case.prim.name(),
        case.canonical,
        failures.join(", ")
    );
}

#[test]
fn f16_runtime_sqrt_finalizes_every_nan_to_its_canonical_image() {
    canonical_sqrt(&FLOATS[0]);
}

#[test]
fn bf16_runtime_sqrt_finalizes_every_nan_to_its_canonical_image() {
    canonical_sqrt(&FLOATS[1]);
}

#[test]
fn f32_runtime_sqrt_finalizes_every_nan_to_its_canonical_image() {
    canonical_sqrt(&FLOATS[2]);
}

#[test]
fn f64_runtime_sqrt_finalizes_every_nan_to_its_canonical_image() {
    canonical_sqrt(&FLOATS[3]);
}

#[test]
fn vector_float_math_finalization_matches_scalar_for_nan_images() {
    for case in &FLOATS {
        let input = stored(case.prim, black_box(&case.nans.repeat(5)));
        for operation in [FloatUnOp::Floor, FloatUnOp::Ceil, FloatUnOp::Round] {
            let output = float_tensor_unop(black_box(operation), black_box(&input)).unwrap();
            for index in 0..input.len() {
                let scalar =
                    float_unop(black_box(operation), black_box(input.scalar_at(index))).unwrap();
                assert_eq!(word(scalar), case.canonical, "scalar {operation:?}");
                assert_eq!(
                    word(output.scalar_at(index)),
                    case.canonical,
                    "{} vector {operation:?}",
                    case.prim.name()
                );
            }
        }
    }
}

#[test]
fn runtime_ingress_and_relu_preserve_nan_sign_and_payload() {
    for case in &FLOATS {
        let input = stored(case.prim, black_box(&case.nans));
        let relu = float_tensor_unop(black_box(FloatUnOp::Relu), black_box(&input)).unwrap();
        for (index, expected) in case.nans.iter().copied().enumerate() {
            assert_eq!(word(input.scalar_at(index)), expected, "exact ingress");
            assert_eq!(word(relu.scalar_at(index)), expected, "Relu selection");
        }
    }
}

#[test]
fn runtime_sqrt_preserves_signed_zero_infinity_and_ordinary_results() {
    for case in &FLOATS {
        let input = stored(
            case.prim,
            black_box(&[0, case.minus_zero, case.plus_inf, case.four]),
        );
        let output = float_tensor_unop(black_box(FloatUnOp::Sqrt), black_box(&input)).unwrap();
        for (index, expected) in [0, case.minus_zero, case.plus_inf, case.two]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                word(output.scalar_at(index)),
                expected,
                "{} element {index}",
                case.prim.name()
            );
        }
    }
}

#[test]
fn runtime_sqrt_rejects_integer_storage() {
    let input = tensor_from_le_payload(Prim::Int32, black_box(&1i32.to_le_bytes())).unwrap();
    assert!(float_tensor_unop(black_box(FloatUnOp::Sqrt), black_box(&input)).is_err());
    assert!(float_unop(black_box(FloatUnOp::Sqrt), black_box(input.scalar_at(0))).is_err());
}
