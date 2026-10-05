//! The num-bigint conversion that `decimal_parse` replaced, kept as a test
//! oracle (chelis#2963). The runtime no longer depends on num-bigint, whose
//! size estimates call host `exp`, `log`, `log2` and `cbrt`; this copy is
//! compiled only into this crate's unit tests, where the differential tests
//! in `decimal_parse` compare the two bit for bit.

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

pub(crate) fn rounded_quotient(numerator: BigUint, denominator: BigUint) -> BigUint {
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
    Some(ratio_to_ieee_bits(
        negative,
        numerator,
        denominator,
        exponent_bits,
        mantissa_bits,
        bias,
    ))
}

/// The exact rational `±numerator / denominator` rounded once, to nearest
/// with ties to even, at the given IEEE binary layout, or overflow past the
/// largest finite value.
pub(crate) fn ratio_to_ieee_bits(
    negative: bool,
    numerator: BigUint,
    denominator: BigUint,
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
) -> Rounded {
    let sign = u64::from(negative) << (exponent_bits + mantissa_bits);
    if numerator == BigUint::from(0u8) {
        return Rounded::Bits(sign);
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
            return Rounded::Overflow { negative };
        }
        let exponent_field = (exponent + bias) as u64;
        let fraction = significand - (1u64 << mantissa_bits);
        return Rounded::Bits(sign | (exponent_field << mantissa_bits) | fraction);
    }

    let subnormal = to_u64(&scaled_round(
        &numerator,
        &denominator,
        mantissa_bits as i32 - minimum_exponent,
    ));
    Rounded::Bits(if subnormal == 1u64 << mantissa_bits {
        sign | (1u64 << mantissa_bits)
    } else {
        sign | subnormal
    })
}
