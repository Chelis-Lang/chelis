//! Exact finite-decimal to IEEE binary conversion.
//!
//! Parsing through `f64` before narrowing can round twice. This module keeps
//! the decimal as an exact integer ratio and performs one round-to-nearest,
//! ties-to-even operation at the requested storage width.

use num_bigint::BigUint;

fn rounded_quotient(numerator: BigUint, denominator: BigUint) -> BigUint {
    let quotient = &numerator / &denominator;
    let remainder = numerator % &denominator;
    let twice_remainder = remainder << 1usize;
    let quotient_is_odd = (&quotient & BigUint::from(1u8)) == BigUint::from(1u8);
    if twice_remainder > denominator || (twice_remainder == denominator && quotient_is_odd) {
        quotient + BigUint::from(1u8)
    } else {
        quotient
    }
}

fn to_u64(value: &BigUint) -> u64 {
    let words = value.to_u64_digits();
    assert!(words.len() <= 1, "rounded IEEE significand exceeds u64");
    words.first().copied().unwrap_or(0)
}

fn floor_log2_ratio(numerator: &BigUint, denominator: &BigUint) -> i32 {
    let mut exponent = i32::try_from(numerator.bits()).expect("decimal numerator too large")
        - i32::try_from(denominator.bits()).expect("decimal denominator too large");
    let below_power = if exponent >= 0 {
        numerator < &(denominator << exponent as usize)
    } else {
        &(numerator << (-exponent) as usize) < denominator
    };
    if below_power {
        exponent -= 1;
    }
    exponent
}

fn scaled_round(numerator: &BigUint, denominator: &BigUint, binary_shift: i32) -> BigUint {
    if binary_shift >= 0 {
        rounded_quotient(numerator << binary_shift as usize, denominator.clone())
    } else {
        rounded_quotient(numerator.clone(), denominator << (-binary_shift) as usize)
    }
}

fn finite_decimal_ratio(text: &str) -> Option<(bool, BigUint, BigUint)> {
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (mantissa, exponent_text) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, None), |(mantissa, exponent)| {
            (mantissa, Some(exponent))
        });
    let explicit_exponent = match exponent_text {
        Some(exponent) => match exponent.parse::<i64>() {
            Ok(exponent) => exponent,
            Err(_) if exponent.starts_with('-') => i64::MIN,
            Err(_) => i64::MAX,
        },
        None => 0,
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if (whole.is_empty() && fraction.is_empty())
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let mut digits = format!("{whole}{fraction}");
    let leading = digits.bytes().take_while(|byte| *byte == b'0').count();
    digits.drain(..leading);
    if digits.is_empty() {
        return Some((negative, BigUint::from(0u8), BigUint::from(1u8)));
    }
    let mut decimal_exponent = explicit_exponent.saturating_sub(fraction.len() as i64);
    while digits.ends_with('0') {
        digits.pop();
        decimal_exponent = decimal_exponent.saturating_add(1);
    }
    let adjusted_exponent = decimal_exponent.saturating_add(digits.len() as i64 - 1);
    if adjusted_exponent > 400 {
        return None;
    }
    if adjusted_exponent < -400 {
        return Some((negative, BigUint::from(0u8), BigUint::from(1u8)));
    }
    let coefficient = BigUint::parse_bytes(digits.as_bytes(), 10)?;
    let power = u32::try_from(decimal_exponent.unsigned_abs()).ok()?;
    if decimal_exponent >= 0 {
        Some((
            negative,
            coefficient * BigUint::from(10u8).pow(power),
            BigUint::from(1u8),
        ))
    } else {
        Some((negative, coefficient, BigUint::from(10u8).pow(power)))
    }
}

pub(crate) fn parse_ieee_bits(
    text: &str,
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
) -> Option<u64> {
    let (negative, numerator, denominator) = finite_decimal_ratio(text)?;
    let sign = u64::from(negative) << (exponent_bits + mantissa_bits);
    if numerator == BigUint::from(0u8) {
        return Some(sign);
    }

    let minimum_exponent = 1 - bias;
    let maximum_exponent = ((1u32 << exponent_bits) - 2) as i32 - bias;
    let mut exponent = floor_log2_ratio(&numerator, &denominator);
    if exponent >= minimum_exponent {
        let mut significand = to_u64(&scaled_round(
            &numerator,
            &denominator,
            mantissa_bits as i32 - exponent,
        ));
        if significand == 1u64 << (mantissa_bits + 1) {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > maximum_exponent {
            return None;
        }
        let exponent_field = (exponent + bias) as u64;
        let fraction = significand - (1u64 << mantissa_bits);
        return Some(sign | (exponent_field << mantissa_bits) | fraction);
    }

    let subnormal = to_u64(&scaled_round(
        &numerator,
        &denominator,
        mantissa_bits as i32 - minimum_exponent,
    ));
    if subnormal == 1u64 << mantissa_bits {
        Some(sign | (1u64 << mantissa_bits))
    } else {
        Some(sign | subnormal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint_below_and_above_round_once_at_each_float_width() {
        let cases = [
            (
                11,
                52,
                1023,
                "1.000000000000000111022302462515654042363166809082031249999999999999",
                "1.00000000000000011102230246251565404236316680908203125",
                "1.000000000000000111022302462515654042363166809082031250000000000001",
                0x3ff0_0000_0000_0000,
                0x3ff0_0000_0000_0001,
            ),
            (
                8,
                23,
                127,
                "1.000000059604644775390624999999",
                "1.000000059604644775390625",
                "1.000000059604644775390625000001",
                0x3f80_0000,
                0x3f80_0001,
            ),
            (
                5,
                10,
                15,
                "1.000488281249999999999999999999",
                "1.00048828125",
                "1.000488281250000000000000000001",
                0x3c00,
                0x3c01,
            ),
            (
                8,
                7,
                127,
                "1.003906249999999999999999999999",
                "1.00390625",
                "1.003906250000000000000000000001",
                0x3f80,
                0x3f81,
            ),
        ];
        for (exponent_bits, mantissa_bits, bias, below, midpoint, above, lower, upper) in cases {
            assert_eq!(
                parse_ieee_bits(below, exponent_bits, mantissa_bits, bias),
                Some(lower)
            );
            assert_eq!(
                parse_ieee_bits(midpoint, exponent_bits, mantissa_bits, bias),
                Some(lower),
                "an even lower significand wins an exact tie"
            );
            assert_eq!(
                parse_ieee_bits(above, exponent_bits, mantissa_bits, bias),
                Some(upper)
            );
        }
    }

    #[test]
    fn malformed_overflow_and_signed_underflow_are_not_silent_defaults() {
        assert_eq!(parse_ieee_bits("1.2.3", 8, 23, 127), None);
        assert_eq!(parse_ieee_bits("1e1000", 8, 23, 127), None);
        assert_eq!(parse_ieee_bits("-1e-1000", 8, 23, 127), Some(0x8000_0000));
    }
}
