//! [05-OP-59]'s explicit text parsers through their public C entries,
//! `chelis_to_int` and `chelis_to_float` (chelis#2870).
//!
//! Every expected f64 image below was computed by an independent correctly
//! rounded parser, not by the runtime. The cases pin what separates [05-OP-59]
//! from [05-OP-31]'s scalar-carrier parse: trimming of every Unicode
//! White_Space character, the signed case-insensitive `inf`, `infinity`, and
//! `nan` spellings, and IEEE overflow to a signed infinity. NaN is compared by
//! class only, because the atom does not fix a NaN payload or sign.

use chelis_runtime::{
    chelis_option, chelis_option_is_some, chelis_option_release, chelis_option_unwrap,
    chelis_parse_scalar, chelis_string, chelis_string_from_utf8, chelis_string_release,
    chelis_to_float, chelis_to_int, chelis_value_unbox_scalar, CHELIS_DTYPE_F64, CHELIS_DTYPE_I64,
};

const POSITIVE_INFINITY: u64 = 0x7ff0_0000_0000_0000;
const NEGATIVE_INFINITY: u64 = 0xfff0_0000_0000_0000;
const NEGATIVE_ZERO: u64 = 0x8000_0000_0000_0000;

/// 2^1024 - 2^970: exactly halfway between f64::MAX and 2^1024, so ties to
/// even rounds it up to infinity.
const OVERFLOW_MIDPOINT: &str = "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792";
/// One below [`OVERFLOW_MIDPOINT`], which rounds down to f64::MAX.
const BELOW_OVERFLOW_MIDPOINT: &str = "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497791";

#[derive(Clone, Copy, PartialEq)]
enum Float {
    Bits(u64),
    Nan,
    Malformed,
}

impl std::fmt::Debug for Float {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Float::Bits(bits) => write!(formatter, "Bits({bits:#018x})"),
            Float::Nan => formatter.write_str("Nan"),
            Float::Malformed => formatter.write_str("Malformed"),
        }
    }
}

/// Run one parse entry on `text` and return the boxed scalar's dtype and bits.
fn call(entry: impl Fn(chelis_string) -> *mut chelis_option, text: &str) -> Option<(u8, u64)> {
    unsafe {
        let runtime_text = chelis_string_from_utf8(text.as_ptr(), text.len() as i64);
        let parsed = entry(runtime_text);
        chelis_string_release(runtime_text);
        let value = if chelis_option_is_some(parsed) {
            let scalar = chelis_value_unbox_scalar(chelis_option_unwrap(parsed));
            Some((scalar.dtype, scalar.bits))
        } else {
            None
        };
        chelis_option_release(parsed);
        value
    }
}

fn to_float(text: &str) -> Float {
    match call(|text| unsafe { chelis_to_float(text) }, text) {
        Some((dtype, bits)) => {
            assert_eq!(
                dtype, CHELIS_DTYPE_F64,
                "to_float({text:?}) must box an f64"
            );
            if f64::from_bits(bits).is_nan() {
                Float::Nan
            } else {
                Float::Bits(bits)
            }
        }
        None => Float::Malformed,
    }
}

fn to_int(text: &str) -> Option<i64> {
    call(|text| unsafe { chelis_to_int(text) }, text).map(|(dtype, bits)| {
        assert_eq!(dtype, CHELIS_DTYPE_I64, "to_int({text:?}) must box an i64");
        i64::from_ne_bytes(bits.to_ne_bytes())
    })
}

fn assert_floats(cases: &[(&str, Float)]) {
    for &(text, expected) in cases {
        assert_eq!(to_float(text), expected, "to_float({text:?})");
    }
}

#[test]
fn valid_float_spellings_round_once_to_nearest_even() {
    use Float::Bits;
    assert_floats(&[
        ("1.5", Bits(0x3ff8_0000_0000_0000)),
        ("-2.25", Bits(0xc002_0000_0000_0000)),
        ("+0.5", Bits(0x3fe0_0000_0000_0000)),
        (".5", Bits(0x3fe0_0000_0000_0000)),
        ("5.", Bits(0x4014_0000_0000_0000)),
        ("1e3", Bits(0x408f_4000_0000_0000)),
        ("1E-3", Bits(0x3f50_624d_d2f1_a9fc)),
        ("+.5e+2", Bits(0x4049_0000_0000_0000)),
        ("007.50", Bits(0x401e_0000_0000_0000)),
        ("0", Bits(0)),
        ("-0", Bits(NEGATIVE_ZERO)),
        ("-0.0", Bits(NEGATIVE_ZERO)),
        ("0.1", Bits(0x3fb9_9999_9999_999a)),
        (
            "123456789012345678901234567890",
            Bits(0x45f8_ee90_ff6c_373e),
        ),
        (
            "1.000000000000000111022302462515654042363166809082031249999999999999",
            Bits(0x3ff0_0000_0000_0000),
        ),
        (
            "1.00000000000000011102230246251565404236316680908203125",
            Bits(0x3ff0_0000_0000_0000),
        ),
        (
            "1.000000000000000111022302462515654042363166809082031250000000000001",
            Bits(0x3ff0_0000_0000_0001),
        ),
        ("2.2250738585072011e-308", Bits(0x000f_ffff_ffff_ffff)),
        ("4.9e-324", Bits(1)),
        ("1.7976931348623157e308", Bits(0x7fef_ffff_ffff_ffff)),
        ("1.7976931348623158e308", Bits(0x7fef_ffff_ffff_ffff)),
        (BELOW_OVERFLOW_MIDPOINT, Bits(0x7fef_ffff_ffff_ffff)),
    ]);
}

#[test]
fn finite_overflow_rounds_to_a_signed_infinity() {
    use Float::Bits;
    assert_floats(&[
        ("1.7976931348623159e308", Bits(POSITIVE_INFINITY)),
        (OVERFLOW_MIDPOINT, Bits(POSITIVE_INFINITY)),
        ("1e400", Bits(POSITIVE_INFINITY)),
        ("-1e400", Bits(NEGATIVE_INFINITY)),
        ("-1.8e308", Bits(NEGATIVE_INFINITY)),
        ("1e99999999999999999999", Bits(POSITIVE_INFINITY)),
    ]);
}

#[test]
fn underflow_rounds_to_a_signed_zero_or_subnormal() {
    use Float::Bits;
    assert_floats(&[
        // 2^-1075 is the midpoint between zero and the least subnormal.
        ("2.4703282292062328e-324", Bits(1)),
        ("2.4703282292062327e-324", Bits(0)),
        ("1e-400", Bits(0)),
        ("-1e-400", Bits(NEGATIVE_ZERO)),
        ("1e-99999999999999999999", Bits(0)),
        ("-1e-99999999999999999999", Bits(NEGATIVE_ZERO)),
    ]);
}

#[test]
fn surrounding_unicode_white_space_is_trimmed() {
    use Float::Bits;
    assert_floats(&[
        (" 1.5 ", Bits(0x3ff8_0000_0000_0000)),
        ("\t2.5\t", Bits(0x4004_0000_0000_0000)),
        ("\n3.5\n", Bits(0x400c_0000_0000_0000)),
        ("\r\n4.5\r\n", Bits(0x4012_0000_0000_0000)),
        ("\u{b}6.5\u{c}", Bits(0x401a_0000_0000_0000)),
        ("\u{a0}7.5\u{a0}", Bits(0x401e_0000_0000_0000)),
        ("\u{2003}8.5\u{2003}", Bits(0x4021_0000_0000_0000)),
        ("\u{3000}9.5", Bits(0x4023_0000_0000_0000)),
        ("\u{2028}-1.5\u{2029}", Bits(0xbff8_0000_0000_0000)),
        ("\u{85} 2.5", Bits(0x4004_0000_0000_0000)),
        ("\u{a0}-inf\u{a0}", Bits(NEGATIVE_INFINITY)),
        (" nan ", Float::Nan),
    ]);
    assert_eq!(to_int(" 7 "), Some(7));
    assert_eq!(to_int("\n7\n"), Some(7));
    assert_eq!(to_int("\u{a0}7"), Some(7));
    assert_eq!(to_int("\u{2003}-7\u{3000}"), Some(-7));
    assert_eq!(to_int("\t+7\r\n"), Some(7));
}

#[test]
fn exceptional_spellings_are_signed_and_case_insensitive() {
    use Float::{Bits, Nan};
    assert_floats(&[
        ("inf", Bits(POSITIVE_INFINITY)),
        ("INF", Bits(POSITIVE_INFINITY)),
        ("Inf", Bits(POSITIVE_INFINITY)),
        ("+inf", Bits(POSITIVE_INFINITY)),
        ("-inf", Bits(NEGATIVE_INFINITY)),
        ("infinity", Bits(POSITIVE_INFINITY)),
        ("Infinity", Bits(POSITIVE_INFINITY)),
        ("+Infinity", Bits(POSITIVE_INFINITY)),
        ("-INFINITY", Bits(NEGATIVE_INFINITY)),
        ("nan", Nan),
        ("NaN", Nan),
        ("NAN", Nan),
        ("+nan", Nan),
        ("-NaN", Nan),
    ]);
}

#[test]
fn malformed_float_text_is_none() {
    for text in [
        "",
        "  ",
        "abc",
        "1.2.3",
        "1e",
        "e5",
        ".",
        "+",
        "-",
        "+-1",
        "--1",
        "1_000",
        "0x10",
        "1,5",
        "1 5",
        "in",
        "infin",
        "infinit",
        "infinityy",
        "nan(1)",
        "1e+",
        "1.5f",
        "1e5.5",
        "\u{661}",
        "\u{ff11}",
        // Neither the byte-order mark nor the zero-width space is White_Space.
        "\u{feff}1",
        "\u{200b}1",
        "1\u{200b}",
        "1\u{0}",
    ] {
        assert_eq!(to_float(text), Float::Malformed, "to_float({text:?})");
    }
}

#[test]
fn integers_are_exact_and_out_of_range_or_malformed_text_is_none() {
    for (text, expected) in [
        ("7", Some(7)),
        ("+7", Some(7)),
        ("-7", Some(-7)),
        ("007", Some(7)),
        ("9223372036854775807", Some(i64::MAX)),
        ("-9223372036854775808", Some(i64::MIN)),
        ("9223372036854775808", None),
        ("-9223372036854775809", None),
        ("", None),
        (" ", None),
        ("+", None),
        ("-", None),
        ("1.0", None),
        ("1e3", None),
        ("0x1", None),
        ("1_0", None),
        ("7 7", None),
        ("\u{661}", None),
        ("\u{feff}7", None),
    ] {
        assert_eq!(to_int(text), expected, "to_int({text:?})");
    }
}

/// The scalar-carrier parse keeps [05-OP-31]'s narrower contract: these are
/// exactly the inputs on which the two contracts differ.
#[test]
fn scalar_carrier_parse_keeps_its_own_contract() {
    for text in [
        "1e400",
        "\n1.5",
        "\u{a0}1.5",
        "INF",
        "Infinity",
        "+inf",
        "nan",
    ] {
        let carrier = call(
            |text| unsafe { chelis_parse_scalar(text, CHELIS_DTYPE_F64) },
            text,
        );
        assert_eq!(carrier, None, "chelis_parse_scalar({text:?}, F64)");
        assert_ne!(to_float(text), Float::Malformed, "to_float({text:?})");
    }
    let carrier = call(
        |text| unsafe { chelis_parse_scalar(text, CHELIS_DTYPE_I64) },
        "\n7",
    );
    assert_eq!(carrier, None);
    assert_eq!(to_int("\n7"), Some(7));
}
