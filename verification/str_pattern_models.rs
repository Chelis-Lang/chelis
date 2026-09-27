//! `str` pattern consumers and integer parsing of the production front ends
//! (Aeneas.Std StringIter.lean, Parse.lean): `strip_prefix`, `strip_suffix`,
//! `split_once`, `trim_start_matches`, `trim_end_matches`, `trim`, `replace`,
//! `as_bytes`, `str` indexed by `RangeTo` and `RangeFrom`, the ASCII case of
//! `char`, and `str::parse` of `usize`, `u64`, `i32` and `i64` with the
//! kinds of `ParseIntError`, and of `f64` and `f32`; `FromIterator<char>` for `String`.
use std::num::IntErrorKind;

/// As chelis-types `expr_function::surface_span_range` and chelis-ir
/// `lower::surface_range`: `surf:<start>..<end>` parsed as two `usize`.
pub fn surface_range(span_id: &str) -> Option<(usize, usize)> {
    let range = span_id.strip_prefix("surf:")?;
    let (start, end) = range.split_once("..")?;
    Some((start.parse().ok()?, end.parse().ok()?))
}

/// As chelis-types `errors::option_unwrap_hint`: `Option[..]` stripped.
pub fn option_payload(s: &str) -> Option<&str> {
    let s = s.strip_prefix("Option[")?;
    s.strip_suffix(']')
}

/// As chelis-ir `axis_sources`: a `d<digits>` axis name as a `u64`.
pub fn axis_index(name: &str) -> Option<u64> {
    name.strip_prefix('d')?.parse::<u64>().ok()
}

/// As chelis-deep `lexer`: the digits after the sign and the radix prefix.
pub fn hex_digits(text: &str) -> &str {
    text.trim_start_matches('-').trim_start_matches("0x").trim_start_matches("0X")
}

/// As chelis-types `observation`: leading and trailing zeros of a mantissa.
pub fn significant(digits: &str) -> (&str, &str) {
    let trimmed_leading = digits.trim_start_matches('0');
    let trimmed = trimmed_leading.trim_end_matches('0');
    (trimmed_leading, trimmed)
}

/// `split_once` on a `char`, then an `i32` exponent.
pub fn exponent(text: &str) -> Option<(&str, Result<i32, IntErrorKind>)> {
    let (mantissa, exp) = text.split_once('e')?;
    Some((mantissa, exp.parse::<i32>().map_err(|e| *e.kind())))
}

/// `str::trim` and `str::replace` with `&str` and `char` patterns.
pub fn normalized(text: &str) -> (String, String) {
    let trimmed = text.trim();
    (trimmed.replace("::", "__"), trimmed.replace('-', "_"))
}

/// `str::as_bytes`: the first byte and the length.
pub fn bytes_of(text: &str) -> (Option<u8>, usize) {
    let bytes = text.as_bytes();
    (bytes.first().copied(), bytes.len())
}

/// `str::parse` of `usize`, `u64`, `i64` with the error kind.
pub fn parse_kinds(text: &str) -> (Result<usize, IntErrorKind>, Result<u64, IntErrorKind>, Result<i64, IntErrorKind>) {
    (
        text.parse::<usize>().map_err(|e| *e.kind()),
        text.parse::<u64>().map_err(|e| *e.kind()),
        text.parse::<i64>().map_err(|e| *e.kind()),
    )
}

/// As chelis-types `observation::split_sci`: the mantissa's digits collected
/// into a `String` (`FromIterator<char>` over a filtered `Chars`) and the
/// exponent (`expect`, through `Debug for ParseIntError`).
pub fn split_sci(sci: &str) -> (String, i32) {
    let (mantissa, exp) = sci.split_once('e').expect("scientific form contains an exponent");
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let exp: i32 = exp.parse().expect("exponent parses");
    (digits, exp)
}

/// As chelis-types `errors::to_snake_case`: `char::is_ascii_uppercase` and
/// `char::to_ascii_lowercase`.
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// As chelis-types `observation` (`&digits[..split]`, `&digits[split..]`): `str`
/// indexed by `RangeTo` and `RangeFrom`, which panic off a char boundary.
pub fn split_at_digit(digits: &str, split: usize) -> (String, String) {
    (digits[..split].to_string(), digits[split..].to_string())
}

/// `str::parse` of `f64` and `f32` (`dec2flt`) as bits, and the `Display` of
/// `ParseFloatError` and `ParseIntError` (as chelis-types `observation`'s
/// `unwrap_or_else(|e| panic!("..: {e}"))`).
pub fn float_bits(text: &str) -> (Result<u64, String>, Result<u32, String>) {
    (
        text.parse::<f64>().map(f64::to_bits).map_err(|e| format!("f64: {e}")),
        text.parse::<f32>().map(f32::to_bits).map_err(|e| e.to_string()),
    )
}

/// The `Display` of `ParseIntError`, padded.
pub fn int_error(text: &str) -> Option<String> {
    text.parse::<u64>().err().map(|e| format!("[{e:>40}]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_ranges() {
        assert_eq!(surface_range("surf:3..17"), Some((3, 17)));
        assert_eq!(surface_range("surf:3..x"), None);
        assert_eq!(surface_range("span:3..17"), None);
        assert_eq!(surface_range("surf:3-17"), None);
        assert_eq!(surface_range("surf:..17"), None);
    }

    #[test]
    fn prefixes_and_suffixes() {
        assert_eq!(option_payload("Option[u32]"), Some("u32"));
        assert_eq!(option_payload("Option[u32"), None);
        assert_eq!(option_payload("Vec[u32]"), None);
        assert_eq!(axis_index("d12"), Some(12));
        assert_eq!(axis_index("d"), None);
        assert_eq!(axis_index("x1"), None);
        assert_eq!(axis_index("d18446744073709551616"), None);
    }

    #[test]
    fn trimmed_matches() {
        assert_eq!(hex_digits("-0x0x1f"), "1f");
        assert_eq!(hex_digits("--0X2a"), "2a");
        assert_eq!(hex_digits("0b1"), "0b1");
        assert_eq!(significant("0012500"), ("12500", "125"));
        assert_eq!(significant("000"), ("", ""));
        assert_eq!(significant("é0"), ("é0", "é"));
    }

    #[test]
    fn exponents() {
        assert_eq!(exponent("1.5e-3"), Some(("1.5", Ok(-3))));
        assert_eq!(exponent("2e+4"), Some(("2", Ok(4))));
        assert_eq!(exponent("2e"), Some(("2", Err(IntErrorKind::Empty))));
        assert_eq!(exponent("2e-"), Some(("2", Err(IntErrorKind::InvalidDigit))));
        assert_eq!(exponent("2e99999999999"), Some(("2", Err(IntErrorKind::PosOverflow))));
        assert_eq!(exponent("2e-99999999999"), Some(("2", Err(IntErrorKind::NegOverflow))));
        assert_eq!(exponent("25"), None);
    }

    #[test]
    fn trim_and_replace() {
        assert_eq!(normalized("  a::b-c \n"), ("a__b-c".to_string(), "a::b_c".to_string()));
        assert_eq!(normalized("\u{3000}x\u{a0}"), ("x".to_string(), "x".to_string()));
        assert_eq!(normalized(" \t "), (String::new(), String::new()));
    }

    #[test]
    fn scientific_split() {
        assert_eq!(split_sci("-1.25e-7"), ("125".to_string(), -7));
        assert_eq!(split_sci("3e12"), ("3".to_string(), 12));
    }

    #[test]
    #[should_panic(expected = "exponent parses")]
    fn scientific_split_rejects_a_bad_exponent() {
        split_sci("1e+");
    }

    #[test]
    #[should_panic(expected = "scientific form contains an exponent")]
    fn scientific_split_needs_an_exponent() {
        split_sci("1.5");
    }

    #[test]
    fn snake_case() {
        assert_eq!(to_snake_case("TensorShapeÉ"), "tensor_shapeÉ");
        assert_eq!(to_snake_case("aB"), "a_b");
        assert_eq!(to_snake_case(""), "");
    }

    #[test]
    fn range_indexing() {
        assert_eq!(split_at_digit("12345", 2), ("12".to_string(), "345".to_string()));
        assert_eq!(split_at_digit("12345", 5), ("12345".to_string(), String::new()));
        assert_eq!(split_at_digit("é1", 2), ("é".to_string(), "1".to_string()));
    }

    #[test]
    #[should_panic]
    fn range_indexing_off_a_char_boundary() {
        split_at_digit("é1", 1);
    }

    #[test]
    #[should_panic]
    fn range_indexing_past_the_end() {
        split_at_digit("12", 3);
    }

    #[test]
    fn float_parsing() {
        let ok = |a: u64, b: u32| (Ok(a), Ok(b));
        assert_eq!(float_bits("0.1"), ok(0x3fb999999999999a, 0x3dcccccd));
        assert_eq!(float_bits("-0"), ok(0x8000000000000000, 0x80000000));
        assert_eq!(float_bits("1e308"), ok(0x7fe1ccf385ebc8a0, 0x7f800000));
        assert_eq!(float_bits("5e-324"), ok(1, 0));
        assert_eq!(float_bits("3e-324"), ok(1, 0));
        assert_eq!(float_bits("-Infinity"), ok(0xfff0000000000000, 0xff800000));
        assert_eq!(float_bits("nan"), ok(0x7ff8000000000000, 0x7fc00000));
        assert_eq!(float_bits("+.5e-3"), ok(0x3f40624dd2f1a9fc, 0x3a03126f));
        assert_eq!(float_bits("1."), ok(0x3ff0000000000000, 0x3f800000));
        assert_eq!(float_bits("123456789012345678901234567890"), ok(0x45f8ee90ff6c373e, 0x6fc77488));
        let err = (Err("f64: invalid float literal".to_string()), Err("invalid float literal".to_string()));
        assert_eq!(float_bits("1e"), err);
        assert_eq!(float_bits("."), err);
        assert_eq!(float_bits("1e5x"), err);
        assert_eq!(
            float_bits(""),
            (Err("f64: cannot parse float from empty string".to_string()), Err("cannot parse float from empty string".to_string()))
        );
        assert_eq!(int_error("12"), None);
        assert_eq!(int_error("-1"), Some(format!("[{:>40}]", "invalid digit found in string")));
        assert_eq!(int_error("99999999999999999999"), Some(format!("[{:>40}]", "number too large to fit in target type")));
    }

    #[test]
    fn bytes() {
        assert_eq!(bytes_of("é!"), (Some(0xc3), 3));
        assert_eq!(bytes_of(""), (None, 0));
    }

    #[test]
    fn parse_error_kinds() {
        assert_eq!(parse_kinds("42"), (Ok(42), Ok(42), Ok(42)));
        assert_eq!(parse_kinds("+7"), (Ok(7), Ok(7), Ok(7)));
        assert_eq!(parse_kinds("-7"), (Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::InvalidDigit), Ok(-7)));
        assert_eq!(parse_kinds(""), (Err(IntErrorKind::Empty), Err(IntErrorKind::Empty), Err(IntErrorKind::Empty)));
        assert_eq!(parse_kinds("-"), (Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::InvalidDigit)));
        assert_eq!(
            parse_kinds("18446744073709551616"),
            (Err(IntErrorKind::PosOverflow), Err(IntErrorKind::PosOverflow), Err(IntErrorKind::PosOverflow))
        );
        assert_eq!(
            parse_kinds("9223372036854775808"),
            (Ok(9223372036854775808), Ok(9223372036854775808), Err(IntErrorKind::PosOverflow))
        );
        assert_eq!(
            parse_kinds("-9223372036854775809"),
            (Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::NegOverflow))
        );
        assert_eq!(parse_kinds("1 "), (Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::InvalidDigit), Err(IntErrorKind::InvalidDigit)));
    }
}
