//! The explicit text parsers `to_int` and `to_float` ([05-OP-59]).
//!
//! The evaluator calls these functions directly and compiled C reaches them
//! through `chelis_to_int` and `chelis_to_float`, so both lanes share one
//! definition of the trimming, the grammar, and the rounding.
//!
//! [05-OP-59] is a different contract from the scalar-carrier parsing of
//! [05-OP-31] (`chelis_parse_scalar`): it trims every Unicode White_Space
//! character rather than only space and tab, admits the case-insensitive
//! signed spellings `inf`, `infinity`, and `nan`, and rounds a finite float
//! that overflows f64 to a signed infinity rather than refusing it.
//!
//! `str::trim` removes exactly the leading and trailing White_Space
//! characters. The integer grammar is then the standard library's: an
//! optional sign followed by one or more ASCII digits. A finite float
//! spelling is rounded by the exact integer-ratio conversion that the
//! scalar-carrier parse also uses, so the result is correctly rounded at any
//! length and does not depend on the caller's floating-point environment; the
//! two contracts differ only in what overflow means.

use crate::decimal_parse::{self, Rounded};

const SIGN_BIT: u64 = 1 << 63;
const INFINITY_BITS: u64 = 0x7ff0_0000_0000_0000;
const QUIET_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

/// [05-OP-59] `to_int`: the exact i64 of a trimmed signed decimal integer, or
/// `None` for malformed or out-of-range text.
pub fn to_int(text: &str) -> Option<i64> {
    text.trim().parse::<i64>().ok()
}

/// [05-OP-59] `to_float`: the correctly rounded f64 of trimmed float text, or
/// `None` for malformed text. No valid spelling fails, overflow included.
pub fn to_float(text: &str) -> Option<f64> {
    let text = text.trim();
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let sign = if negative { SIGN_BIT } else { 0 };
    if unsigned.eq_ignore_ascii_case("inf") || unsigned.eq_ignore_ascii_case("infinity") {
        return Some(f64::from_bits(sign | INFINITY_BITS));
    }
    if unsigned.eq_ignore_ascii_case("nan") {
        return Some(f64::from_bits(sign | QUIET_NAN_BITS));
    }
    if !finite_spelling(unsigned) {
        return None;
    }
    Some(match decimal_parse::round_decimal(text, 11, 52, 1023)? {
        Rounded::Bits(bits) => f64::from_bits(bits),
        Rounded::Overflow { negative } => {
            f64::from_bits(if negative { SIGN_BIT } else { 0 } | INFINITY_BITS)
        }
    })
}

/// The unsigned finite grammar `([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?`.
fn finite_spelling(unsigned: &str) -> bool {
    let digits = |part: &str| part.bytes().all(|byte| byte.is_ascii_digit());
    let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, Some(exponent)),
        None => (unsigned, None),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mantissa_valid =
        !(whole.is_empty() && fraction.is_empty()) && digits(whole) && digits(fraction);
    let exponent_valid = exponent.is_none_or(|exponent| {
        let exponent = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        !exponent.is_empty() && digits(exponent)
    });
    mantissa_valid && exponent_valid
}

#[cfg(test)]
mod tests {
    use super::to_float;
    use crate::fp_env::{read_control, write_control};

    /// The caller's rounding mode or flush-to-zero setting must not change a
    /// parse: the evaluator calls `to_float` directly, and a foreign host can
    /// call `chelis_to_float` outside compiled code's pinned environment.
    #[test]
    fn parsing_ignores_the_callers_floating_point_environment() {
        let cases = [
            ("0.1", 0x3fb9_9999_9999_999a_u64),
            ("0.3", 0x3fd3_3333_3333_3333),
            ("123.456", 0x405e_dd2f_1a9f_be77),
            ("1e23", 0x44b5_2d02_c7e1_4af6),
            ("2.2250738585072011e-308", 0x000f_ffff_ffff_ffff),
            ("-2.5e300", 0xfe4d_dd4b_aa00_9303),
        ];
        #[cfg(target_arch = "aarch64")]
        let modes = [0b01 << 22, 0b10 << 22, 0b11 << 22, (0b11 << 22) | (1 << 24)];
        #[cfg(target_arch = "x86_64")]
        let modes = [
            0x1f80 | 0x4000,
            0x1f80 | 0x2000,
            0x1f80 | 0x6000,
            0x1f80 | 0x6000 | 0x8040,
        ];
        let caller = read_control();
        let mut observed = Vec::new();
        for mode in modes {
            write_control(mode);
            for (text, _) in cases {
                let shared = to_float(text).map(f64::to_bits);
                let exported = unsafe {
                    let runtime_text =
                        crate::chelis_string_from_utf8(text.as_ptr(), text.len() as i64);
                    let parsed = crate::chelis_to_float(runtime_text);
                    crate::chelis_string_release(runtime_text);
                    let bits = crate::chelis_option_is_some(parsed).then(|| {
                        crate::chelis_value_unbox_scalar(crate::chelis_option_unwrap(parsed)).bits
                    });
                    crate::chelis_option_release(parsed);
                    bits
                };
                observed.push((mode, text, shared, exported));
            }
        }
        write_control(caller);
        for (mode, text, shared, exported) in observed {
            let expected = cases
                .iter()
                .find(|(case, _)| *case == text)
                .map(|(_, bits)| *bits);
            assert_eq!(
                shared, expected,
                "to_float(`{text}`) under control {mode:#x}"
            );
            assert_eq!(
                exported, expected,
                "chelis_to_float(`{text}`) under control {mode:#x}"
            );
        }
    }
}
