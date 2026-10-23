//! Shared target-width IEEE conversion.

pub(crate) fn round_shift_even_u64(value: u64, shift: u32) -> u64 {
    match shift {
        0 => value,
        1..=63 => {
            let quotient = value >> shift;
            let remainder = value & ((1_u64 << shift) - 1);
            let halfway = 1_u64 << (shift - 1);
            quotient + u64::from(remainder > halfway || (remainder == halfway && quotient & 1 == 1))
        }
        64 => u64::from(value > (1_u64 << 63)),
        _ => 0,
    }
}

/// Round one binary64 value directly into a binary16-shaped IEEE encoding.
///
/// The source significand remains intact until the one target-width
/// round-to-nearest-ties-to-even step. In particular, this must not delegate
/// through f32 or a converter which truncates the low binary64 significand
/// bits before rounding: doing so loses which side of an f16/bf16 midpoint the
/// original f64 occupied ([04-NUM-14]).
fn f64_to_ieee16_bits(value: f64, exponent_bits: u32, mantissa_bits: u32, bias: i32) -> u16 {
    let source = value.to_bits();
    let sign = ((source >> 48) & 0x8000) as u16;
    let source_exponent = ((source >> 52) & 0x7ff) as u32;
    let source_mantissa = source & 0x000f_ffff_ffff_ffff;
    let target_exponent_max = (1_u32 << exponent_bits) - 1;
    let target_exponent_bits = (target_exponent_max << mantissa_bits) as u16;

    if source_exponent == 0x7ff {
        if source_mantissa == 0 {
            return sign | target_exponent_bits;
        }
        // [04-NUM-2]: every NaN becomes the canonical quiet NaN, whatever its
        // sign and payload.
        return target_exponent_bits | (1_u16 << (mantissa_bits - 1));
    }
    if source_exponent == 0 {
        // Every finite binary64 subnormal is below half of the least f16 or
        // bf16 subnormal. This includes signed zero and preserves its sign.
        return sign;
    }

    let mut exponent = source_exponent as i32 - 1023;
    let significand = (1_u64 << 52) | source_mantissa;
    let minimum_exponent = 1 - bias;
    let maximum_exponent = target_exponent_max as i32 - 1 - bias;
    if exponent > maximum_exponent {
        return sign | target_exponent_bits;
    }

    if exponent >= minimum_exponent {
        let mut rounded = round_shift_even_u64(significand, 52 - mantissa_bits);
        if rounded == (1_u64 << (mantissa_bits + 1)) {
            rounded >>= 1;
            exponent += 1;
            if exponent > maximum_exponent {
                return sign | target_exponent_bits;
            }
        }
        let target_exponent = ((exponent + bias) as u16) << mantissa_bits;
        let target_mantissa = (rounded & ((1_u64 << mantissa_bits) - 1)) as u16;
        return sign | target_exponent | target_mantissa;
    }

    let shift = (52 - mantissa_bits) + (minimum_exponent - exponent) as u32;
    let rounded = round_shift_even_u64(significand, shift);
    sign | rounded as u16
}

/// Round binary64 directly to binary16 bits.
pub fn f64_to_f16_bits_rne(value: f64) -> u16 {
    f64_to_ieee16_bits(value, 5, 10, 15)
}

/// Round binary64 directly to bfloat16 bits.
pub fn f64_to_bf16_bits_rne(value: f64) -> u16 {
    f64_to_ieee16_bits(value, 8, 7, 127)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_narrowing_keeps_the_f64_side_of_reduced_midpoints() {
        assert_eq!(f64_to_f16_bits_rne(52847.99970178839), 0x7a73);
        assert_eq!(
            f64_to_f16_bits_rne(f64::from(52847.99970178839_f64 as f32)),
            0x7a74
        );
        assert_eq!(f64_to_bf16_bits_rne(1.0039062500000002), 0x3f81);
        assert_eq!(
            f64_to_bf16_bits_rne(f64::from(1.0039062500000002_f64 as f32)),
            0x3f80
        );
    }

    #[test]
    fn runtime_narrowing_matches_the_dtype_semantic_reference_at_all_midpoints() {
        fn check(
            last_finite: u16,
            image: impl Fn(u16) -> f64,
            runtime_round: impl Fn(f64) -> u16,
            reference_round: impl Fn(f64) -> u16,
        ) {
            for lower_bits in 0..last_finite {
                let midpoint = (image(lower_bits) + image(lower_bits + 1)) / 2.0;
                for value in [
                    f64::from_bits(midpoint.to_bits() - 1),
                    midpoint,
                    f64::from_bits(midpoint.to_bits() + 1),
                ] {
                    assert_eq!(runtime_round(value), reference_round(value));
                    assert_eq!(runtime_round(-value), reference_round(-value));
                }
            }
        }

        check(
            0x7bff,
            |bits| f64::from(half::f16::from_bits(bits)),
            f64_to_f16_bits_rne,
            |value| crate::dtype_semantics::f16_from_f64_rne(value).to_bits(),
        );
        check(
            0x7f7f,
            |bits| f64::from(half::bf16::from_bits(bits)),
            f64_to_bf16_bits_rne,
            |value| crate::dtype_semantics::bf16_from_f64_rne(value).to_bits(),
        );
    }
}
