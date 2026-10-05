//! [05-OP-1] `round_to`, shared by every execution lane.
//!
//! The evaluator and the `chelis_round_to` C export both call these
//! definitions, so the decimal rounding and its result have one definition.
//! The operand's exact binary value is rounded to the nearest multiple of
//! `10^(-places)`, ties to the even integer coefficient, with exact integer
//! arithmetic for every `places`; the multiple is then finalized once at the
//! operand's own storage width.

use crate::decimal_parse::{ratio_to_ieee_bits, rounded_quotient, Natural, Rounded};

/// An IEEE binary float layout: exponent and stored-mantissa widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatLayout {
    pub exponent_bits: u32,
    pub mantissa_bits: u32,
}

impl FloatLayout {
    pub const F64: Self = Self {
        exponent_bits: 11,
        mantissa_bits: 52,
    };
    pub const F32: Self = Self {
        exponent_bits: 8,
        mantissa_bits: 23,
    };
    pub const F16: Self = Self {
        exponent_bits: 5,
        mantissa_bits: 10,
    };
    pub const BF16: Self = Self {
        exponent_bits: 8,
        mantissa_bits: 7,
    };

    fn bias(self) -> i32 {
        (1i32 << (self.exponent_bits - 1)) - 1
    }
}

/// `round_to` over the operand's stored bits at `layout`. A non-finite
/// operand passes through with its bits unchanged; a zero result keeps the
/// operand's sign; a result past the largest finite value is the correctly
/// signed infinity ([04-NUM-2]).
pub fn round_to_bits(bits: u64, layout: FloatLayout, places: i64) -> u64 {
    let FloatLayout {
        exponent_bits,
        mantissa_bits,
    } = layout;
    let sign_bit = 1u64 << (exponent_bits + mantissa_bits);
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent_field = (bits >> mantissa_bits) & exponent_mask;
    let fraction = bits & ((1u64 << mantissa_bits) - 1);
    let negative = bits & sign_bit != 0;
    if exponent_field == exponent_mask || (exponent_field == 0 && fraction == 0) {
        return bits;
    }
    let bias = layout.bias();
    // value = significand * 2^exponent exactly.
    let (significand, exponent) = if exponent_field == 0 {
        (fraction, 1 - bias - mantissa_bits as i32)
    } else {
        (
            fraction | (1u64 << mantissa_bits),
            exponent_field as i32 - bias - mantissa_bits as i32,
        )
    };
    // Every finite binary value is a multiple of 10^(-places) once
    // `places >= -exponent`, so a finer quantum is the identity.
    if places >= i64::from(-exponent.min(0)) {
        return bits;
    }
    // A quantum more than twice the largest finite magnitude of any layout
    // (below 2^1024 < 10^309) rounds every finite value to zero.
    if places < -310 {
        return bits & sign_bit;
    }
    let power = places.unsigned_abs() as u32;
    let mut decimal = Natural::from_u64(1);
    decimal.mul_pow10(power);
    let magnitude = Natural::from_u64(significand).shl(u64::from(exponent.max(0).unsigned_abs()));
    let binary_down = Natural::from_u64(1).shl(u64::from((-exponent).max(0).unsigned_abs()));
    let (numerator, denominator) = if places >= 0 {
        let mut scaled = magnitude;
        scaled.mul_pow10(power);
        (scaled, binary_down)
    } else {
        let mut scaled = binary_down;
        scaled.mul_pow10(power);
        (magnitude, scaled)
    };
    let mut coefficient = rounded_quotient(numerator, denominator);
    if coefficient.is_zero() {
        return bits & sign_bit;
    }
    let (numerator, denominator) = if places >= 0 {
        (coefficient, decimal)
    } else {
        coefficient.mul_pow10(power);
        (coefficient, Natural::from_u64(1))
    };
    match ratio_to_ieee_bits(
        negative,
        numerator,
        denominator,
        exponent_bits,
        mantissa_bits,
        bias,
    ) {
        Rounded::Bits(rounded) => rounded,
        Rounded::Overflow { .. } => (bits & sign_bit) | (exponent_mask << mantissa_bits),
    }
}

/// [05-OP-1] at f64.
pub fn round_to_f64(x: f64, places: i64) -> f64 {
    f64::from_bits(round_to_bits(x.to_bits(), FloatLayout::F64, places))
}

/// [05-OP-1] at f32, rounded and finalized at f32 alone.
pub fn round_to_f32(x: f32, places: i64) -> f32 {
    f32::from_bits(round_to_bits(u64::from(x.to_bits()), FloatLayout::F32, places) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decimal_parse_reference::{
        ratio_to_ieee_bits as reference_ratio_to_ieee_bits,
        rounded_quotient as reference_rounded_quotient, Rounded as ReferenceRounded,
    };
    use num_bigint::BigUint;

    /// The num-bigint `round_to` this module replaced, kept as the oracle for
    /// the differential test below (chelis#2963).
    fn reference_round_to_bits(bits: u64, layout: FloatLayout, places: i64) -> u64 {
        let FloatLayout {
            exponent_bits,
            mantissa_bits,
        } = layout;
        let sign_bit = 1u64 << (exponent_bits + mantissa_bits);
        let exponent_mask = (1u64 << exponent_bits) - 1;
        let exponent_field = (bits >> mantissa_bits) & exponent_mask;
        let fraction = bits & ((1u64 << mantissa_bits) - 1);
        let negative = bits & sign_bit != 0;
        if exponent_field == exponent_mask || (exponent_field == 0 && fraction == 0) {
            return bits;
        }
        let bias = layout.bias();
        // value = significand * 2^exponent exactly.
        let (significand, exponent) = if exponent_field == 0 {
            (fraction, 1 - bias - mantissa_bits as i32)
        } else {
            (
                fraction | (1u64 << mantissa_bits),
                exponent_field as i32 - bias - mantissa_bits as i32,
            )
        };
        // Every finite binary value is a multiple of 10^(-places) once
        // `places >= -exponent`, so a finer quantum is the identity.
        if places >= i64::from(-exponent.min(0)) {
            return bits;
        }
        // A quantum more than twice the largest finite magnitude of any layout
        // (below 2^1024 < 10^309) rounds every finite value to zero.
        if places < -310 {
            return bits & sign_bit;
        }
        let two = BigUint::from(2u8);
        let ten = BigUint::from(10u8);
        let binary_up = two.pow(exponent.max(0) as u32);
        let binary_down = two.pow((-exponent).max(0) as u32);
        let decimal = ten.pow(places.unsigned_abs() as u32);
        let magnitude = BigUint::from(significand) * binary_up;
        let (numerator, denominator) = if places >= 0 {
            (magnitude * &decimal, binary_down)
        } else {
            (magnitude, binary_down * &decimal)
        };
        let coefficient = reference_rounded_quotient(numerator, denominator);
        if coefficient == BigUint::from(0u8) {
            return bits & sign_bit;
        }
        let (numerator, denominator) = if places >= 0 {
            (coefficient, decimal)
        } else {
            (coefficient * decimal, BigUint::from(1u8))
        };
        match reference_ratio_to_ieee_bits(
            negative,
            numerator,
            denominator,
            exponent_bits,
            mantissa_bits,
            bias,
        ) {
            ReferenceRounded::Bits(rounded) => rounded,
            ReferenceRounded::Overflow { .. } => {
                (bits & sign_bit) | (exponent_mask << mantissa_bits)
            }
        }
    }

    /// Bit-identical agreement with the num-bigint `round_to` over random
    /// operands at every layout, normal, subnormal and non-finite, and over
    /// every `places` regime: identity, rounding, and all-zero.
    #[test]
    fn random_operands_match_the_num_bigint_round_to() {
        let mut state = 0x2963_0005u64;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        for layout in [
            FloatLayout::F64,
            FloatLayout::F32,
            FloatLayout::F16,
            FloatLayout::BF16,
        ] {
            let width = 1 + layout.exponent_bits + layout.mantissa_bits;
            for _ in 0..4_000 {
                let bits = next() >> (64 - width);
                let places = (next() % 1_441) as i64 - 330;
                assert_eq!(
                    round_to_bits(bits, layout, places),
                    reference_round_to_bits(bits, layout, places),
                    "bits {bits} places {places} at {} exponent bits",
                    layout.exponent_bits
                );
            }
        }
    }

    #[test]
    fn ties_go_to_even_on_the_exact_binary_value() {
        assert_eq!(round_to_f64(0.125, 2), 0.12);
        assert_eq!(round_to_f64(0.375, 2), 0.38);
        assert_eq!(round_to_f64(2.675, 2), 2.67);
        assert_eq!(round_to_f32(0.125, 2), 0.12_f32);
        assert_eq!(round_to_f32(2.675, 2), 2.67_f32);
    }

    #[test]
    fn negative_places_round_left_of_the_point() {
        assert_eq!(round_to_f64(1234.5, -2), 1200.0);
        assert_eq!(round_to_f64(1250.0, -2), 1200.0);
        assert_eq!(round_to_f64(1350.0, -2), 1400.0);
        assert_eq!(round_to_f64(-49.0, -2).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(round_to_f64(1.0e300, -400).to_bits(), 0);
    }

    #[test]
    fn large_places_are_the_identity_and_overflow_is_infinite() {
        assert_eq!(round_to_f64(0.1, 1000), 0.1);
        let tiny = f64::MIN_POSITIVE / 3.0;
        assert_eq!(round_to_f64(tiny, 2000).to_bits(), tiny.to_bits());
        assert_eq!(round_to_f64(f64::MAX, -308), f64::INFINITY);
        assert_eq!(round_to_f64(-f64::MAX, -308), f64::NEG_INFINITY);
    }

    #[test]
    fn non_finite_operands_keep_their_bits() {
        let payload_nan = f64::from_bits(0x7ff0_0000_0000_0001);
        assert_eq!(
            round_to_f64(payload_nan, 2).to_bits(),
            payload_nan.to_bits()
        );
        assert_eq!(round_to_f32(f32::NEG_INFINITY, 0), f32::NEG_INFINITY);
        assert_eq!(round_to_f64(-0.001, 1).to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn half_widths_round_once_at_their_own_width() {
        // f16 2.675 is stored as 2.67578125; to two places that is 2.68,
        // whose nearest f16 is 2.6796875.
        assert_eq!(round_to_bits(0x415a, FloatLayout::F16, 2), 0x415c);
        // bf16 -1.0 to -1 places is zero with the operand's sign.
        assert_eq!(round_to_bits(0xbf80, FloatLayout::BF16, -1), 0x8000);
    }
}
