//! Exact finite-decimal to IEEE binary conversion.
//!
//! Parsing through `f64` before narrowing can round twice. This module keeps
//! the decimal as an exact integer ratio and performs one round-to-nearest,
//! ties-to-even operation at the requested storage width. It uses only integer
//! arithmetic, so the caller's floating-point environment cannot change a
//! result.

use num_bigint::BigUint;

/// Significant digits kept before the rest collapse into one sticky digit. A
/// binary64 rounding midpoint, the widest case here, is `m * 2^e` with `m`
/// odd, `m < 2^54` and `e >= -1075`, so it has at most 768 significant
/// decimal digits. Two values that share their first 800 digits
/// and both continue with a nonzero tail therefore lie on the same side of
/// every midpoint and round identically.
const DECIDING_DIGITS: usize = 800;

/// A validated finite decimal spelling as an exact ratio, or a magnitude
/// beyond every supported width's largest finite value.
enum DecimalValue {
    Ratio {
        negative: bool,
        numerator: BigUint,
        denominator: BigUint,
    },
    Overflow {
        negative: bool,
    },
}

/// One correctly rounded conversion: the stored bits, or overflow past the
/// largest finite value. Each caller's contract decides what overflow means.
pub(crate) enum Rounded {
    Bits(u64),
    Overflow { negative: bool },
}

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

/// `None` only for text outside the finite decimal grammar, which [05-OP-31]
/// and [05-OP-59] state identically: an optional sign, then one or more ASCII
/// digits with an optional point and fraction digits or a point and one or
/// more digits, then an optional `[eE][+-]?[0-9]+` exponent.
fn finite_decimal_ratio(text: &str) -> Option<DecimalValue> {
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
        Some(exponent) => {
            let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            // An exponent beyond i64 saturates: no digit count that fits in
            // memory can bring such a value back into range.
            exponent
                .parse::<i64>()
                .unwrap_or(if exponent.starts_with('-') {
                    i64::MIN
                } else {
                    i64::MAX
                })
        }
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
    let zero = DecimalValue::Ratio {
        negative,
        numerator: BigUint::from(0u8),
        denominator: BigUint::from(1u8),
    };
    if digits.is_empty() {
        return Some(zero);
    }
    let mut decimal_exponent = explicit_exponent.saturating_sub(fraction.len() as i64);
    while digits.ends_with('0') {
        digits.pop();
        decimal_exponent = decimal_exponent.saturating_add(1);
    }
    if digits.len() > DECIDING_DIGITS {
        // The last digit is nonzero, so the dropped tail is nonzero and one
        // sticky `1` after the kept digits stands for it exactly enough.
        let dropped = digits.len() - DECIDING_DIGITS;
        digits.truncate(DECIDING_DIGITS);
        digits.push('1');
        decimal_exponent = decimal_exponent.saturating_add(dropped as i64 - 1);
    }
    let adjusted_exponent = decimal_exponent.saturating_add(digits.len() as i64 - 1);
    if adjusted_exponent > 400 {
        return Some(DecimalValue::Overflow { negative });
    }
    if adjusted_exponent < -400 {
        return Some(zero);
    }
    let coefficient = BigUint::parse_bytes(digits.as_bytes(), 10)?;
    // At most `DECIDING_DIGITS + 1` significant digits, with the leading one
    // within 400 places of the decimal point, bound the scale by 1,201.
    let power = u32::try_from(decimal_exponent.unsigned_abs())
        .expect("the significant-digit cap bounds the decimal scale");
    let ten_to_power = BigUint::from(10u8).pow(power);
    Some(if decimal_exponent >= 0 {
        DecimalValue::Ratio {
            negative,
            numerator: coefficient * ten_to_power,
            denominator: BigUint::from(1u8),
        }
    } else {
        DecimalValue::Ratio {
            negative,
            numerator: coefficient,
            denominator: ten_to_power,
        }
    })
}

/// [05-OP-31]'s scalar-carrier conversion, under which finite overflow is
/// `None`, like malformed text.
pub(crate) fn parse_ieee_bits(
    text: &str,
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
) -> Option<u64> {
    match round_decimal(text, exponent_bits, mantissa_bits, bias)? {
        Rounded::Bits(bits) => Some(bits),
        Rounded::Overflow { .. } => None,
    }
}

/// The correctly rounded image of a finite decimal spelling at one IEEE
/// width, with overflow reported as such. `None` only for text outside the
/// finite decimal grammar.
pub(crate) fn round_decimal(
    text: &str,
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
) -> Option<Rounded> {
    let (negative, numerator, denominator) = match finite_decimal_ratio(text)? {
        DecimalValue::Ratio {
            negative,
            numerator,
            denominator,
        } => (negative, numerator, denominator),
        DecimalValue::Overflow { negative } => return Some(Rounded::Overflow { negative }),
    };
    let sign = u64::from(negative) << (exponent_bits + mantissa_bits);
    if numerator == BigUint::from(0u8) {
        return Some(Rounded::Bits(sign));
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
            return Some(Rounded::Overflow { negative });
        }
        let exponent_field = (exponent + bias) as u64;
        let fraction = significand - (1u64 << mantissa_bits);
        return Some(Rounded::Bits(
            sign | (exponent_field << mantissa_bits) | fraction,
        ));
    }

    let subnormal = to_u64(&scaled_round(
        &numerator,
        &denominator,
        mantissa_bits as i32 - minimum_exponent,
    ));
    Some(Rounded::Bits(if subnormal == 1u64 << mantissa_bits {
        sign | (1u64 << mantissa_bits)
    } else {
        sign | subnormal
    }))
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

    fn rounded_f64(text: &str) -> Option<Result<u64, bool>> {
        round_decimal(text, 11, 52, 1023).map(|rounded| match rounded {
            Rounded::Bits(bits) => Ok(bits),
            Rounded::Overflow { negative } => Err(negative),
        })
    }

    #[test]
    fn overflow_is_reported_with_its_sign_and_malformed_text_is_not_overflow() {
        assert_eq!(rounded_f64("1e1000"), Some(Err(false)));
        assert_eq!(rounded_f64("-1e1000"), Some(Err(true)));
        assert_eq!(rounded_f64("1.7976931348623159e308"), Some(Err(false)));
        assert_eq!(
            rounded_f64("-99999999999999999999e99999999999999999999"),
            Some(Err(true))
        );
        assert_eq!(
            rounded_f64("1.7976931348623157e308"),
            Some(Ok(0x7fef_ffff_ffff_ffff))
        );
        for malformed in [
            "1e", "1e+", "1e-", "1e5e5", "1ex", "e5", ".", "1.2.3", "+-1",
        ] {
            assert_eq!(
                rounded_f64(malformed),
                None,
                "malformed spelling `{malformed}`"
            );
        }
    }

    #[test]
    fn digits_past_the_deciding_prefix_keep_their_rounding_effect() {
        let tie = "1.000000059604644775390625";
        let zeros = "0".repeat(2_000);
        assert_eq!(
            parse_ieee_bits(&format!("{tie}{zeros}"), 8, 23, 127),
            Some(0x3f80_0000)
        );
        assert_eq!(
            parse_ieee_bits(&format!("{tie}{zeros}1"), 8, 23, 127),
            Some(0x3f80_0001)
        );
        let below = format!("1.000000059604644775390624{}", "9".repeat(2_000));
        assert_eq!(parse_ieee_bits(&below, 8, 23, 127), Some(0x3f80_0000));
        assert_eq!(
            rounded_f64(&format!("{}e-1000000", "1".repeat(1_000_000))),
            Some(Ok(0x3fbc_71c7_1c71_c71c)),
            "a million-digit spelling of nearly 1/9 rounds like 1/9"
        );
    }

    #[test]
    fn malformed_overflow_and_signed_underflow_are_not_silent_defaults() {
        assert_eq!(parse_ieee_bits("1.2.3", 8, 23, 127), None);
        assert_eq!(parse_ieee_bits("1e1000", 8, 23, 127), None);
        assert_eq!(parse_ieee_bits("-1e-1000", 8, 23, 127), Some(0x8000_0000));
    }
}
