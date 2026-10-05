//! Exact finite-decimal to IEEE binary conversion.
//!
//! Parsing through `f64` before narrowing can round twice. This module keeps
//! the decimal as an exact integer ratio and performs one round-to-nearest,
//! ties-to-even operation at the requested storage width. It uses only integer
//! arithmetic, so the caller's floating-point environment cannot change a
//! result. The arbitrary-precision naturals are this module's own: a general
//! big-integer crate sizes buffers and seeds root guesses through host `exp`,
//! `log`, `log2` and `cbrt`, and the runtime archive links no host
//! transcendental (chelis#2963).

use std::cmp::Ordering;

/// An arbitrary-precision natural number: little-endian 32-bit limbs with no
/// high zero limb, so zero is the empty vector and equal values have equal
/// limbs.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Natural(Vec<u32>);

impl Natural {
    pub(crate) fn from_u64(value: u64) -> Self {
        let mut natural = Natural(vec![value as u32, (value >> 32) as u32]);
        natural.normalize();
        natural
    }

    fn normalize(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// The bit length: zero for zero, otherwise one more than the index of
    /// the highest set bit.
    fn bits(&self) -> u64 {
        match self.0.last() {
            None => 0,
            Some(top) => 32 * self.0.len() as u64 - u64::from(top.leading_zeros()),
        }
    }

    /// `self * factor + addend`, in place.
    fn mul_add_small(&mut self, factor: u32, addend: u32) {
        let mut carry = u64::from(addend);
        for limb in &mut self.0 {
            let product = u64::from(*limb) * u64::from(factor) + carry;
            *limb = product as u32;
            carry = product >> 32;
        }
        if carry != 0 {
            self.0.push(carry as u32);
        }
        self.normalize();
    }

    /// The natural spelled by ASCII decimal digits, which the caller has
    /// validated.
    fn from_decimal_digits(digits: &[u8]) -> Self {
        let mut natural = Natural(Vec::new());
        for chunk in digits.chunks(9) {
            let value = chunk
                .iter()
                .fold(0u32, |value, digit| value * 10 + u32::from(digit - b'0'));
            natural.mul_add_small(10u32.pow(chunk.len() as u32), value);
        }
        natural
    }

    /// `self * 10^power`, in place.
    pub(crate) fn mul_pow10(&mut self, mut power: u32) {
        while power >= 9 {
            self.mul_add_small(1_000_000_000, 0);
            power -= 9;
        }
        self.mul_add_small(10u32.pow(power), 0);
    }

    pub(crate) fn shl(&self, shift: u64) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let limbs = (shift / 32) as usize;
        let bits = (shift % 32) as u32;
        let mut shifted = vec![0u32; limbs];
        shifted.reserve(self.0.len() + 1);
        if bits == 0 {
            shifted.extend_from_slice(&self.0);
        } else {
            let mut carry = 0u32;
            for limb in &self.0 {
                shifted.push((limb << bits) | carry);
                carry = limb >> (32 - bits);
            }
            shifted.push(carry);
        }
        let mut natural = Natural(shifted);
        natural.normalize();
        natural
    }

    fn add_one(mut self) -> Self {
        self.mul_add_small(1, 1);
        self
    }

    /// `self >> 1`, in place.
    fn shr1(&mut self) {
        let mut carry = 0u32;
        for limb in self.0.iter_mut().rev() {
            let next_carry = *limb << 31;
            *limb = (*limb >> 1) | carry;
            carry = next_carry;
        }
        self.normalize();
    }

    /// `self - other`, in place. The caller guarantees `self >= other`.
    fn sub_assign(&mut self, other: &Natural) {
        let mut borrow = 0u64;
        for (index, limb) in self.0.iter_mut().enumerate() {
            let subtrahend = u64::from(other.0.get(index).copied().unwrap_or(0)) + borrow;
            let minuend = u64::from(*limb);
            if minuend >= subtrahend {
                *limb = (minuend - subtrahend) as u32;
                borrow = 0;
            } else {
                *limb = ((1u64 << 32) + minuend - subtrahend) as u32;
                borrow = 1;
            }
        }
        assert_eq!(borrow, 0, "natural subtraction underflowed");
        self.normalize();
    }
}

impl Ord for Natural {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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
        numerator: Natural,
        denominator: Natural,
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

/// `numerator / denominator` rounded to nearest, ties to even, by schoolbook
/// binary long division: one quotient bit per step, from the highest.
pub(crate) fn rounded_quotient(numerator: Natural, denominator: Natural) -> Natural {
    assert!(
        !denominator.is_zero(),
        "decimal ratio with a zero denominator"
    );
    let mut remainder = numerator;
    let mut quotient = Natural(Vec::new());
    if remainder >= denominator {
        let top = remainder.bits() - denominator.bits();
        quotient.0 = vec![0u32; (top / 32 + 1) as usize];
        let mut divisor = denominator.shl(top);
        for bit in (0..=top).rev() {
            if remainder >= divisor {
                remainder.sub_assign(&divisor);
                quotient.0[(bit / 32) as usize] |= 1u32 << (bit % 32);
            }
            divisor.shr1();
        }
        quotient.normalize();
    }
    let twice_remainder = remainder.shl(1);
    let quotient_is_odd = quotient.0.first().is_some_and(|low| low & 1 == 1);
    match twice_remainder.cmp(&denominator) {
        Ordering::Greater => quotient.add_one(),
        Ordering::Equal if quotient_is_odd => quotient.add_one(),
        _ => quotient,
    }
}

/// The value of a natural that the caller has bounded below `2^64`.
fn to_u64(value: &Natural) -> u64 {
    assert!(value.0.len() <= 2, "rounded IEEE significand exceeds u64");
    value
        .0
        .iter()
        .rev()
        .fold(0u64, |acc, limb| (acc << 32) | u64::from(*limb))
}

fn floor_log2_ratio(numerator: &Natural, denominator: &Natural) -> i32 {
    let mut exponent = i32::try_from(numerator.bits()).expect("decimal numerator too large")
        - i32::try_from(denominator.bits()).expect("decimal denominator too large");
    let below_power = if exponent >= 0 {
        numerator < &denominator.shl(exponent as u64)
    } else {
        &numerator.shl(u64::from(exponent.unsigned_abs())) < denominator
    };
    if below_power {
        exponent -= 1;
    }
    exponent
}

fn scaled_round(numerator: &Natural, denominator: &Natural, binary_shift: i32) -> u64 {
    let shift = u64::from(binary_shift.unsigned_abs());
    to_u64(&if binary_shift >= 0 {
        rounded_quotient(numerator.shl(shift), denominator.clone())
    } else {
        rounded_quotient(numerator.clone(), denominator.shl(shift))
    })
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
        numerator: Natural(Vec::new()),
        denominator: Natural::from_u64(1),
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
    let mut coefficient = Natural::from_decimal_digits(digits.as_bytes());
    // At most `DECIDING_DIGITS + 1` significant digits, with the leading one
    // within 400 places of the decimal point, bound the scale by 1,201.
    let power = u32::try_from(decimal_exponent.unsigned_abs())
        .expect("the significant-digit cap bounds the decimal scale");
    Some(if decimal_exponent >= 0 {
        coefficient.mul_pow10(power);
        DecimalValue::Ratio {
            negative,
            numerator: coefficient,
            denominator: Natural::from_u64(1),
        }
    } else {
        let mut ten_to_power = Natural::from_u64(1);
        ten_to_power.mul_pow10(power);
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
    numerator: Natural,
    denominator: Natural,
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
) -> Rounded {
    let sign = u64::from(negative) << (exponent_bits + mantissa_bits);
    if numerator.is_zero() {
        return Rounded::Bits(sign);
    }

    let minimum_exponent = 1 - bias;
    let maximum_exponent = ((1u32 << exponent_bits) - 2) as i32 - bias;
    let mut exponent = floor_log2_ratio(&numerator, &denominator);
    if exponent >= minimum_exponent {
        let mut significand =
            scaled_round(&numerator, &denominator, mantissa_bits as i32 - exponent);
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

    let subnormal = scaled_round(
        &numerator,
        &denominator,
        mantissa_bits as i32 - minimum_exponent,
    );
    Rounded::Bits(if subnormal == 1u64 << mantissa_bits {
        sign | (1u64 << mantissa_bits)
    } else {
        sign | subnormal
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decimal_parse_reference as reference;
    use num_bigint::BigUint;

    /// The four storage widths as (exponent bits, mantissa bits, bias).
    const WIDTHS: [(u32, u32, i32); 4] = [(11, 52, 1023), (8, 23, 127), (5, 10, 15), (8, 7, 127)];

    /// A fixed-seed SplitMix64 stream, so every run draws the same cases.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }

        fn range(&mut self, low: i64, high: i64) -> i64 {
            low + self.below((high - low + 1) as u64) as i64
        }

        fn digits(&mut self, count: usize) -> String {
            (0..count)
                .map(|_| char::from(b'0' + self.below(10) as u8))
                .collect()
        }
    }

    /// One conversion as comparable data: the bits, overflow with its sign,
    /// or `None` for malformed text.
    fn outcome(text: &str, width: (u32, u32, i32)) -> Option<Result<u64, bool>> {
        round_decimal(text, width.0, width.1, width.2).map(|rounded| match rounded {
            Rounded::Bits(bits) => Ok(bits),
            Rounded::Overflow { negative } => Err(negative),
        })
    }

    fn reference_outcome(text: &str, width: (u32, u32, i32)) -> Option<Result<u64, bool>> {
        reference::round_decimal(text, width.0, width.1, width.2).map(|rounded| match rounded {
            reference::Rounded::Bits(bits) => Ok(bits),
            reference::Rounded::Overflow { negative } => Err(negative),
        })
    }

    /// Bit-identical agreement with the num-bigint conversion at every width,
    /// and with the standard library's correctly rounded parse at binary64
    /// and binary32, where an overflow is that parse's signed infinity.
    fn assert_agrees(text: &str) {
        for width in WIDTHS {
            assert_eq!(
                outcome(text, width),
                reference_outcome(text, width),
                "`{text}` at width {width:?}"
            );
        }
        if let Ok(value) = text.parse::<f64>() {
            let expected = value.to_bits();
            let got = match outcome(text, WIDTHS[0]) {
                Some(Ok(bits)) => bits,
                Some(Err(negative)) => (u64::from(negative) << 63) | 0x7ff0_0000_0000_0000,
                None => panic!("`{text}` parses as f64 but not as a finite decimal"),
            };
            assert_eq!(got, expected, "`{text}` against str::parse::<f64>");
        }
        if let Ok(value) = text.parse::<f32>() {
            let expected = u64::from(value.to_bits());
            let got = match outcome(text, WIDTHS[1]) {
                Some(Ok(bits)) => bits,
                Some(Err(negative)) => (u64::from(negative) << 31) | 0x7f80_0000,
                None => panic!("`{text}` parses as f32 but not as a finite decimal"),
            };
            assert_eq!(got, expected, "`{text}` against str::parse::<f32>");
        }
    }

    #[test]
    fn random_short_spellings_match_the_reference_and_the_standard_parse() {
        let mut stream = Stream(0x2963_0001);
        for _ in 0..20_000 {
            let length = stream.range(1, 40) as usize;
            let digits = stream.digits(length);
            let point = stream.below(digits.len() as u64 + 1) as usize;
            let sign = ["", "-", "+"][stream.below(3) as usize];
            let mut text = format!("{sign}{}.{}", &digits[..point], &digits[point..]);
            if stream.below(4) != 0 {
                let exponent = stream.range(-380, 340);
                let marker = if stream.below(2) == 0 { 'e' } else { 'E' };
                text.push_str(&format!("{marker}{exponent}"));
            }
            assert_agrees(&text);
        }
    }

    /// Every rounding boundary at every width, normal and subnormal: the
    /// exact decimal spelling of a midpoint `m * 2^e` with `m` odd, and the
    /// spellings just below and just above it.
    #[test]
    fn exact_midpoints_and_their_neighbours_match_the_reference() {
        let mut stream = Stream(0x2963_0002);
        for width in WIDTHS {
            let (_, mantissa_bits, bias) = width;
            let lowest = -(bias - 1) - mantissa_bits as i32 - 1;
            let highest = bias - mantissa_bits as i32;
            for _ in 0..1_500 {
                let midpoint = (stream.next() >> (63 - mantissa_bits - 1)) | 1;
                let exponent = stream.range(i64::from(lowest), i64::from(highest)) as i32;
                let (coefficient, scale) = if exponent >= 0 {
                    (BigUint::from(midpoint) << exponent as usize, 0i64)
                } else {
                    let power = exponent.unsigned_abs();
                    (
                        BigUint::from(midpoint) * BigUint::from(5u8).pow(power),
                        i64::from(exponent),
                    )
                };
                let padding = stream.range(1, 30) as u32;
                let widened = &coefficient * BigUint::from(10u8).pow(padding);
                let below = &widened - BigUint::from(1u8);
                let above = &widened + BigUint::from(1u8);
                let widened_scale = scale - i64::from(padding);
                assert_agrees(&format!("{coefficient}e{scale}"));
                assert_agrees(&format!("{below}e{widened_scale}"));
                assert_agrees(&format!("{above}e{widened_scale}"));
            }
        }
    }

    #[test]
    fn long_spellings_around_the_deciding_prefix_match_the_reference() {
        let mut stream = Stream(0x2963_0003);
        for _ in 0..600 {
            let length = stream.range(700, 900) as usize;
            let mut digits = stream.digits(length);
            if stream.below(2) == 0 {
                // A run of zeros or nines makes a midpoint-adjacent tail.
                let run = if stream.below(2) == 0 { '0' } else { '9' };
                let start = stream.range(1, 40) as usize;
                digits.replace_range(
                    start..length - 1,
                    &run.to_string().repeat(length - 1 - start),
                );
            }
            let exponent = stream.range(-1_200, 400);
            assert_agrees(&format!("0.{digits}e{exponent}"));
            assert_agrees(&format!("{digits}e{}", exponent - length as i64));
        }
    }

    #[test]
    fn extreme_exponents_and_zero_spellings_match_the_reference() {
        for text in [
            "0",
            "-0",
            "+0.000",
            "0e999999999999999999999",
            "-0.0e-5",
            "4.9406564584124654e-324",
            "2.4703282292062327e-324",
            "2.4703282292062328e-324",
            "2.2250738585072011e-308",
            "2.2250738585072014e-308",
            "1.7976931348623157e308",
            "1.7976931348623158e308",
            "1.401298464324817e-45",
            "7.006492321624085e-46",
            "3.4028235e38",
            "3.4028236e38",
            "6.103515625e-05",
            "5.960464477539063e-08",
            "65504",
            "65520",
            "65519.99",
            "1e400",
            "1e401",
            "1e-400",
            "1e-401",
            "9.99e400",
            "123456789012345678901234567890",
            "1e-9223372036854775808",
            "1e9223372036854775807",
            "1e99999999999999999999",
            ".5",
            "5.",
            "-.5e-3",
        ] {
            assert_agrees(text);
        }
    }

    #[test]
    fn random_malformed_spellings_match_the_reference() {
        let alphabet = b"0123456789.eE+-x ";
        let mut stream = Stream(0x2963_0004);
        for _ in 0..20_000 {
            let length = stream.below(12) as usize;
            let text: String = (0..length)
                .map(|_| char::from(alphabet[stream.below(alphabet.len() as u64) as usize]))
                .collect();
            for width in WIDTHS {
                assert_eq!(
                    outcome(&text, width),
                    reference_outcome(&text, width),
                    "`{text}`"
                );
            }
        }
    }

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
