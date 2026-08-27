//! THE element formatter for the observation channel (chelis#732 Phase 1).
//!
//! `format_element` is the single source of truth for how one stored
//! numeric/bool element becomes text at every exit (`print`, `to_list`
//! rendering, diagnostics that embed values, labeled roots) in every lane.
//! The eval lane calls it directly; the compiled-C print helper is
//! GENERATED from it at chelis#732 Phase 2. No other formatting path for
//! tensor/scalar payloads may exist in any lane
//! (`spec/design/faithful_observation.md` §C1/§C3, `spec/05-risc-primitives.md`
//! §8 atoms [05-OBS-1..5]).
//!
//! ## The number grammar (normative, frozen at chelis#732 Phase 1)
//!
//! Rust `{:?}` (`Debug`) float formatting is the normative grammar
//! (`faithful_observation.md` §C1.3, ratified into spec/05 §8.1):
//!
//! * shortest round-trip digits AT THE VALUE'S OWN WIDTH;
//! * decimal form exactly when the RENDERED magnitude - the value the
//!   chosen shortest digits denote - is zero or satisfies
//!   `1e-4 <= |v| < 1e16` ([`DECIMAL_LOWER_BOUND`] /
//!   [`DECIMAL_UPPER_BOUND`], normative constants captured empirically
//!   from rustc; the unit tests below break loudly if a rustc formatting
//!   change ever shifts them). The rule follows the digits actually
//!   printed, not the stored magnitude: when a width's ulp straddles a
//!   threshold, the shortest rendering can sit on the other side of it
//!   (PR #792 red-team finding F2 has the executed boundary cases);
//! * decimal renderings of integral values keep one fractional digit
//!   (`2048.0`, never `2048`);
//! * e-notation is `<mantissa>e<exp>` with no `+` and no zero padding
//!   (`1e-7`, `9.999999980506448e19`);
//! * specials spell `inf` / `-inf` / `NaN`; `-0.0` keeps its sign.
//!
//! `Display` is NOT this grammar (it never emits e-notation), and neither
//! is the C helpers' two-branch printf format split (the chelis#748
//! shape); the generated C normalizes to this grammar at Phase 2.
//!
//! For `f32`/`f64` the implementation IS `format!("{v:?}")`. For
//! `f16`/`bf16` (no native Rust formatter) the shortest-digit search below
//! finds the shortest decimal whose parse-back - `str -> f64`, then a
//! correctly-rounded narrowing to the half width (safe by the same
//! excess-precision argument as [04-NUM-1]'s single-rounding rule) - yields
//! the stored bits; the decimal/e-notation decision applies the same
//! rendered-magnitude rule to the chosen digits. Verified
//! EXHAUSTIVELY over all 65536 bit patterns per half format in the tests
//! below (faithful_observation.md open question 1).
//!
//! ## Signature note
//!
//! The design docs sketched `format_element(prim, value: ElementRef<'_>)`.
//! The landed [`ElementRef`] is a `Copy` enum carrying the element at its
//! dtype's own width (mirroring the runtime's `ScalarBits`); no borrow is
//! needed for scalar payloads, so the lifetime was dropped. Same name, same
//! role, updated in both owning docs in this change set.

use crate::types::Prim;

/// Decimal-form lower threshold: a rendering whose RENDERED magnitude is
/// below `1e-4` (and not zero) takes e-notation. Normative constant
/// (spec/05 §8.1), captured from Rust `{:?}` behavior and locked by tests.
pub const DECIMAL_LOWER_BOUND: f64 = 1e-4;

/// Decimal-form upper threshold: a rendering whose RENDERED magnitude is
/// `>= 1e16` takes e-notation. Normative constant (spec/05 §8.1), captured
/// from Rust `{:?}` behavior and locked by tests.
pub const DECIMAL_UPPER_BOUND: f64 = 1e16;

/// One stored element at its dtype's own width.
///
/// The variant carries the value exactly as stored; [`format_element`]
/// panics loudly on a `prim`/variant mismatch (the same dtype/bits
/// invariant the runtime's `ScalarPayload` enforces at construction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementRef {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(half::f16),
    Bf16(half::bf16),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl ElementRef {
    /// The `Prim` this element intrinsically carries. Must equal the
    /// `prim` argument handed to [`format_element`].
    pub fn dtype(&self) -> Prim {
        match self {
            ElementRef::I8(_) => Prim::Int8,
            ElementRef::I16(_) => Prim::Int16,
            ElementRef::I32(_) => Prim::Int32,
            ElementRef::I64(_) => Prim::Int64,
            ElementRef::F16(_) => Prim::F16,
            ElementRef::Bf16(_) => Prim::Bf16,
            ElementRef::F32(_) => Prim::F32,
            ElementRef::F64(_) => Prim::F64,
            ElementRef::Bool(_) => Prim::Bool,
        }
    }
}

/// Round one sealed float element to a decimal-place boundary while keeping
/// its storage width. This is [05-OP-1]'s formatting-dependent operation,
/// centralized beside [`format_element`] so runtime consumers cannot grow a
/// second Rust numeric formatter. Integer, bool, and half-width callers are
/// rejected by the checked builtin boundary before reaching this helper.
pub fn round_element_to_decimal_places(value: ElementRef, places: usize) -> ElementRef {
    match value {
        ElementRef::F32(value) => ElementRef::F32(round_via_decimal_text(value, places)),
        ElementRef::F64(value) => ElementRef::F64(round_via_decimal_text(value, places)),
        other => panic!(
            "round_element_to_decimal_places: expected an f32/f64 sealed element, got {}",
            other.dtype().name()
        ),
    }
}

fn round_via_decimal_text<T>(value: T, places: usize) -> T
where
    T: std::fmt::Display + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let text = format!("{value:.places$}");
    text.parse::<T>().unwrap_or_else(|error| {
        panic!("sealed decimal formatter emitted an unparsable token `{text}`: {error}")
    })
}

/// THE printed form of one element ([05-OBS-1..2]). Exhaustive over
/// [`Prim`] with no `_` arm (`spec/design/loud_unsupported.md` §C4.1).
///
/// Panics (internal invariants, named per [05-UNS-3]):
/// * `prim` disagrees with the variant's intrinsic dtype - callers
///   construct [`ElementRef`] from dtype-tagged storage (the runtime's
///   `ScalarBits` / tensor precision tags), so a mismatch is a caller bug,
///   never a data condition;
/// * `prim` is `f8e4m3` (not in the active dtype set, spec/04 §1.1.1;
///   rejected at check time, so no exit can carry one) or `string`
///   (strings render as themselves at their exits and are not numeric/bool
///   element payloads).
pub fn format_element(prim: Prim, value: ElementRef) -> String {
    let mismatch = |prim: Prim, value: &ElementRef| -> String {
        panic!(
            "format_element: dtype/value mismatch: prim={} but the element carries {} \
             (callers build ElementRef from dtype-tagged storage; see \
             spec/05-risc-primitives.md section 8)",
            prim.name(),
            value.dtype().name()
        )
    };
    match prim {
        Prim::Int8 => match value {
            ElementRef::I8(v) => v.to_string(),
            other => mismatch(prim, &other),
        },
        Prim::Int16 => match value {
            ElementRef::I16(v) => v.to_string(),
            other => mismatch(prim, &other),
        },
        Prim::Int32 => match value {
            ElementRef::I32(v) => v.to_string(),
            other => mismatch(prim, &other),
        },
        Prim::Int64 => match value {
            ElementRef::I64(v) => v.to_string(),
            other => mismatch(prim, &other),
        },
        Prim::F16 => match value {
            ElementRef::F16(v) => format_half(
                f64::from(v),
                u64::from(v.to_bits()),
                &|x| u64::from(crate::dtype_semantics::f16_from_f64_rne(x).to_bits()),
                "f16",
            ),
            other => mismatch(prim, &other),
        },
        Prim::Bf16 => match value {
            ElementRef::Bf16(v) => format_half(
                f64::from(v),
                u64::from(v.to_bits()),
                &|x| u64::from(crate::dtype_semantics::bf16_from_f64_rne(x).to_bits()),
                "bf16",
            ),
            other => mismatch(prim, &other),
        },
        Prim::F32 => match value {
            ElementRef::F32(v) => format!("{v:?}"),
            other => mismatch(prim, &other),
        },
        Prim::F64 => match value {
            ElementRef::F64(v) => format!("{v:?}"),
            other => mismatch(prim, &other),
        },
        Prim::Bool => match value {
            ElementRef::Bool(v) => v.to_string(),
            other => mismatch(prim, &other),
        },
        Prim::F8e4m3 => panic!(
            "format_element: f8e4m3 is not in the active dtype set \
             (spec/04-type-system.md section 1.1.1); the checker rejects it, \
             so no exit can carry an f8e4m3 element"
        ),
        Prim::String => panic!(
            "format_element formats numeric/bool element payloads; string \
             values render as themselves at their exits and never arrive here"
        ),
    }
}

/// Shortest-round-trip rendering for a half-precision value, in the
/// normative grammar. `image` is the exact f64 widening of the stored
/// value, `stored_bits` its bit pattern (widened to u64), and `narrow`
/// the parse-back narrowing (f64 -> half bits) for the width.
fn format_half(image: f64, stored_bits: u64, narrow: &dyn Fn(f64) -> u64, width: &str) -> String {
    if image.is_nan() {
        // NaN round-trips at the class level: exit text carries no payload
        // (spec/05 section 8.1; matches the harness's canonical-NaN rule).
        return "NaN".to_string();
    }
    if image.is_infinite() {
        return if image < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if image == 0.0 {
        return if image.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_string();
    }
    let neg = image < 0.0;
    let abs = image.abs();
    let round_trips = |text: &str| -> bool {
        let parsed: f64 = text.parse().expect("candidate text is a valid float");
        narrow(if neg { -parsed } else { parsed }) == stored_bits
    };
    let (digits, sci_exp) = shortest_digits(abs, &round_trips, width);
    assemble(neg, &digits, sci_exp)
}

/// Find the shortest decimal digit string `D` (no trailing zeros) and
/// scientific exponent `E` (value = `D[0].D[1..] x 10^E`) such that the
/// signed rendering parses back to the stored bits. Searches digit counts
/// `p = 1..=17`; at each `p` the three candidates around the correctly
/// rounded `p`-digit decimal of `abs` cover every `p`-digit decimal that
/// could round-trip (the nearest decimal and its two grid neighbors, with
/// decade-boundary refinement handled explicitly), so the first hit is
/// shortest in digits. Among same-length hits the numerically closest to
/// `abs` wins; an exact tie prefers the even mantissa (deterministic,
/// spec/05 section 8.1).
fn shortest_digits(abs: f64, round_trips: &dyn Fn(&str) -> bool, width: &str) -> (String, i32) {
    for p in 1..=17u32 {
        // Correctly rounded p-digit scientific form "d.ddd...e<exp>".
        let sci = format!("{:.*e}", (p - 1) as usize, abs);
        let (mantissa, exp) = split_sci(&sci);
        debug_assert_eq!(mantissa.len() as u32, p, "rustc sci form has p digits");
        let m: u64 = mantissa.parse().expect("mantissa digits parse");
        let p10 = 10u64.pow(p - 1);
        // Grid neighbors of m at p digits. At the lower decade boundary
        // (m = 10^(p-1)) the adjacent p-digit decimal below is
        // (10^p - 1) x 10^(exp-1)'s grid, one decade down and finer.
        let lower = if m == p10 {
            (10u64.pow(p) - 1, exp - 1)
        } else {
            (m - 1, exp)
        };
        // At the upper boundary (m = 10^p - 1) the decimal above is
        // 10^(p-1) one decade up (the carried form).
        let upper = if m == 10u64.pow(p) - 1 {
            (p10, exp + 1)
        } else {
            (m + 1, exp)
        };
        let mut best: Option<(u64, i32, f64)> = None;
        for (cand, cand_exp) in [(m, exp), lower, upper] {
            if cand == 0 {
                continue;
            }
            let text = format!("{cand}e{}", cand_exp - (p as i32) + 1);
            if !round_trips(&text) {
                continue;
            }
            let value: f64 = text.parse().expect("candidate parses");
            let dist = (value - abs).abs();
            let better = match &best {
                None => true,
                Some((bm, _, bd)) => dist < *bd || (dist == *bd && cand % 2 == 0 && bm % 2 == 1),
            };
            if better {
                best = Some((cand, cand_exp, dist));
            }
        }
        if let Some((cand, cand_exp, _)) = best {
            // Strip trailing zeros: the equal shorter value parses
            // identically, so digits stay minimal. The scientific exponent
            // is unchanged by the strip.
            let mut digits = cand.to_string();
            while digits.len() > 1 && digits.ends_with('0') {
                digits.pop();
            }
            return (digits, cand_exp);
        }
    }
    // 17 significant digits uniquely identify every f64, and the stored
    // half value IS an exact f64; unreachable for finite nonzero input.
    panic!("shortest_digits: no 17-digit decimal round-trips a finite {width} value ({abs:e})")
}

/// Split "d.ddd...e<exp>" (Rust `{:.*e}` output) into (digits, exponent).
fn split_sci(sci: &str) -> (String, i32) {
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("scientific form contains an exponent");
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let exp: i32 = exp.parse().expect("exponent parses");
    (digits, exp)
}

/// Assemble digits + scientific exponent into the normative grammar,
/// choosing decimal vs e-notation from the RENDERED magnitude - the value
/// the chosen shortest digits denote, whose decade is exactly `sci_exp` -
/// against the normative thresholds: decimal iff the rendered value `v`
/// satisfies `1e-4 <= |v| < 1e16`, i.e. iff `-4 <= sci_exp <= 15` (zero
/// is handled earlier and renders decimal). This is rustc's actual `{:?}`
/// rule (PR #792 red-team finding F2): the stored magnitude can sit on
/// the other side of a threshold from its own shortest rendering when the
/// width's ulp straddles the boundary - e.g. the bf16 whose image is
/// 9.992e15 renders shortest as `1e16`, which must take e-notation, and
/// the f32 whose image is 9.9999997e-5 renders shortest as `0.0001`,
/// which must take decimal form - and the form follows the digits that
/// are actually printed.
fn assemble(neg: bool, digits: &str, sci_exp: i32) -> String {
    let sign = if neg { "-" } else { "" };
    let decimal_form = (-4..=15).contains(&sci_exp);
    if decimal_form {
        let len = digits.len() as i32;
        let body = if sci_exp >= len - 1 {
            // Integral: pad with zeros, keep one fractional digit.
            let zeros = "0".repeat((sci_exp - (len - 1)) as usize);
            format!("{digits}{zeros}.0")
        } else if sci_exp >= 0 {
            let split = (sci_exp + 1) as usize;
            format!("{}.{}", &digits[..split], &digits[split..])
        } else {
            let zeros = "0".repeat((-sci_exp - 1) as usize);
            format!("0.{zeros}{digits}")
        };
        format!("{sign}{body}")
    } else {
        let mantissa = if digits.len() == 1 {
            digits.to_string()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!("{sign}{mantissa}e{sci_exp}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // The normative-constant locks (faithful_observation.md open
    // question 1): if a rustc formatting change ever moves the `{:?}`
    // grammar, these fail loudly instead of silently shifting the
    // grammar the generated C must match.
    // -----------------------------------------------------------------

    #[test]
    fn f64_debug_grammar_thresholds_are_the_normative_constants() {
        // Decimal side of both boundaries.
        assert_eq!(format!("{:?}", 1e-4f64), "0.0001");
        assert_eq!(format!("{:?}", 9999999999999998.0f64), "9999999999999998.0");
        // E-notation side of both boundaries.
        assert_eq!(format!("{:?}", 1e-5f64), "1e-5");
        assert_eq!(format!("{:?}", 1e16f64), "1e16");
        // Exponent form: no plus sign, no zero padding, lowercase e.
        assert_eq!(format!("{:?}", 1e20f64), "1e20");
        assert_eq!(format!("{:?}", 1.5e-7f64), "1.5e-7");
        // Specials and signed zero.
        assert_eq!(format!("{:?}", f64::NAN), "NaN");
        assert_eq!(format!("{:?}", f64::INFINITY), "inf");
        assert_eq!(format!("{:?}", f64::NEG_INFINITY), "-inf");
        assert_eq!(format!("{:?}", -0.0f64), "-0.0");
        // The audit's canonical values.
        assert_eq!(format!("{:?}", f64::MAX), "1.7976931348623157e308");
        assert_eq!(format!("{:?}", 5e-324f64), "5e-324");
        assert_eq!(
            format!("{:?}", 9.999999980506448e19f64),
            "9.999999980506448e19"
        );
        assert_eq!(
            format!("{:?}", 0.30000000000000004f64),
            "0.30000000000000004"
        );
    }

    /// The f32 grammar follows the same thresholds - decided on the
    /// RENDERED magnitude (rustc compares against the constants at the
    /// value's own width, which coincides with the rendered-magnitude rule
    /// at every representable boundary; PR #792 red-team F2). Note the
    /// first row: 1e-4f32 STORES 9.9999997e-5, strictly below the f64
    /// constant, yet renders decimal because its shortest digits denote
    /// exactly 1e-4.
    #[test]
    fn f32_debug_grammar_matches_the_same_thresholds() {
        assert_eq!(format!("{:?}", 1e-4f32), "0.0001");
        assert_eq!(format!("{:?}", 1e-5f32), "1e-5");
        assert_eq!(format!("{:?}", 1e16f32), "1e16");
        assert_eq!(format!("{:?}", f32::MAX), "3.4028235e38");
        assert_eq!(format!("{:?}", 1e-45f32), "1e-45");
        assert_eq!(format!("{:?}", 0.1f32), "0.1");
        assert_eq!(format!("{:?}", 16777216.0f32), "16777216.0");
        assert_eq!(format!("{:?}", -0.0f32), "-0.0");
        assert_eq!(format!("{:?}", f32::NAN), "NaN");
    }

    // -----------------------------------------------------------------
    // format_element per dtype: positive rows.
    // -----------------------------------------------------------------

    #[test]
    fn sealed_decimal_rounding_preserves_width_and_ties_to_even() {
        assert_eq!(
            round_element_to_decimal_places(ElementRef::F64(0.125), 2),
            ElementRef::F64(0.12)
        );
        assert_eq!(
            round_element_to_decimal_places(ElementRef::F64(0.375), 2),
            ElementRef::F64(0.38)
        );
        assert_eq!(
            round_element_to_decimal_places(ElementRef::F32(2.675), 2),
            ElementRef::F32(2.67)
        );
    }

    #[test]
    fn integers_print_as_integers_at_every_width() {
        assert_eq!(format_element(Prim::Int8, ElementRef::I8(127)), "127");
        assert_eq!(format_element(Prim::Int8, ElementRef::I8(-128)), "-128");
        assert_eq!(
            format_element(Prim::Int16, ElementRef::I16(-32767)),
            "-32767"
        );
        assert_eq!(
            format_element(Prim::Int32, ElementRef::I32(2147483647)),
            "2147483647"
        );
        // int64 prints all digits exactly, never through double.
        assert_eq!(
            format_element(Prim::Int64, ElementRef::I64(9007199254740993)),
            "9007199254740993"
        );
        assert_eq!(
            format_element(Prim::Int64, ElementRef::I64(i64::MAX)),
            "9223372036854775807"
        );
        assert_eq!(
            format_element(Prim::Int64, ElementRef::I64(i64::MIN)),
            "-9223372036854775808"
        );
    }

    #[test]
    fn bool_prints_true_false() {
        assert_eq!(format_element(Prim::Bool, ElementRef::Bool(true)), "true");
        assert_eq!(format_element(Prim::Bool, ElementRef::Bool(false)), "false");
    }

    #[test]
    fn f64_formats_shortest_round_trip_debug_grammar() {
        for (v, want) in [
            (0.1f64, "0.1"),
            (0.30000000000000004, "0.30000000000000004"),
            (f64::MAX, "1.7976931348623157e308"),
            (5e-324, "5e-324"),
            (-0.0, "-0.0"),
            (9007199254740992.0, "9007199254740992.0"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(format_element(Prim::F64, ElementRef::F64(v)), want);
        }
        assert_eq!(format_element(Prim::F64, ElementRef::F64(f64::NAN)), "NaN");
    }

    #[test]
    fn f32_formats_shortest_round_trip_at_f32_width() {
        for (v, want) in [
            (0.1f32, "0.1"),
            (f32::MAX, "3.4028235e38"),
            (1e-45f32, "1e-45"),
            (-0.0f32, "-0.0"),
            (16777218.0f32, "16777218.0"),
            // The audit's byte-comparability example: sqrt(2) at f32.
            (std::f32::consts::SQRT_2, "1.4142135"),
        ] {
            assert_eq!(format_element(Prim::F32, ElementRef::F32(v)), want);
        }
    }

    #[test]
    fn f16_formats_shortest_round_trip_at_f16_width() {
        let f16 = crate::dtype_semantics::f16_from_f64_rne;
        // Exact-representable rows from the harness's frozen f16 table.
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(f16(0.75))),
            "0.75"
        );
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(f16(2048.0))),
            "2048.0"
        );
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(f16(-1.5))),
            "-1.5"
        );
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(f16(-0.0))),
            "-0.0"
        );
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(half::f16::INFINITY)),
            "inf"
        );
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(half::f16::NAN)),
            "NaN"
        );
        // Shortest-at-own-width is genuinely shorter than the f64 image:
        // the f16 min subnormal (2^-24, image 5.960464477539063e-8) needs
        // one digit at f16 width.
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(half::f16::from_bits(1))),
            "6e-8"
        );
        // f16::MAX (65504): "65500" already rounds back to it at f16 width.
        assert_eq!(
            format_element(Prim::F16, ElementRef::F16(half::f16::MAX)),
            "65500.0"
        );
    }

    #[test]
    fn bf16_formats_shortest_round_trip_at_bf16_width() {
        let bf16 = crate::dtype_semantics::bf16_from_f64_rne;
        assert_eq!(
            format_element(Prim::Bf16, ElementRef::Bf16(bf16(0.75))),
            "0.75"
        );
        assert_eq!(
            format_element(Prim::Bf16, ElementRef::Bf16(bf16(256.0))),
            "256.0"
        );
        assert_eq!(
            format_element(Prim::Bf16, ElementRef::Bf16(bf16(-0.0))),
            "-0.0"
        );
        assert_eq!(
            format_element(Prim::Bf16, ElementRef::Bf16(half::bf16::NAN)),
            "NaN"
        );
    }

    // -----------------------------------------------------------------
    // Negative rows: the loud arms.
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "dtype/value mismatch")]
    fn prim_variant_mismatch_panics_loudly() {
        format_element(Prim::F16, ElementRef::F32(1.0));
    }

    #[test]
    #[should_panic(expected = "dtype/value mismatch")]
    fn integer_width_mismatch_panics_loudly() {
        format_element(Prim::Int8, ElementRef::I64(1));
    }

    #[test]
    #[should_panic(expected = "f8e4m3 is not in the active dtype set")]
    fn f8e4m3_is_a_loud_arm_not_a_fallback() {
        format_element(Prim::F8e4m3, ElementRef::F32(1.0));
    }

    #[test]
    #[should_panic(expected = "string")]
    fn string_prim_is_a_loud_arm_not_a_fallback() {
        format_element(Prim::String, ElementRef::Bool(true));
    }

    // -----------------------------------------------------------------
    // The exhaustive half-width verification (required Phase 1
    // deliverable, faithful_observation.md open question 1): every one
    // of the 65536 bit patterns per format round-trips through its own
    // rendering, and the rendering is shortest in significant digits.
    // -----------------------------------------------------------------

    /// Count significant digits in a rendering (grammar-agnostic).
    fn significant_digits(text: &str) -> u32 {
        let mantissa = text.split('e').next().expect("mantissa");
        let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
        let trimmed_leading = digits.trim_start_matches('0');
        // "0.0075" -> "75"; integral decimals keep their ".0" out of the
        // count only when it is grammar padding on an integral value
        // ("65500.0" -> 655 after trailing-zero trim below).
        let trimmed = trimmed_leading.trim_end_matches('0');
        if trimmed.is_empty() {
            1
        } else {
            trimmed.len() as u32
        }
    }

    /// Independent shortest-ness check: no decimal with fewer significant
    /// digits parses back to `stored_bits`. Scans the full 3-candidate
    /// neighborhood at every shorter digit count.
    fn no_shorter_rendering(image: f64, stored_bits: u64, narrow: &dyn Fn(f64) -> u64, p: u32) {
        let neg = image < 0.0;
        let abs = image.abs();
        for shorter in 1..p {
            let sci = format!("{:.*e}", (shorter - 1) as usize, abs);
            let (mantissa, exp) = split_sci(&sci);
            let m: u64 = mantissa.parse().unwrap();
            let p10 = 10u64.pow(shorter - 1);
            let lower = if m == p10 {
                (10u64.pow(shorter) - 1, exp - 1)
            } else {
                (m.saturating_sub(1), exp)
            };
            let upper = if m == 10u64.pow(shorter) - 1 {
                (p10, exp + 1)
            } else {
                (m + 1, exp)
            };
            for (cand, cand_exp) in [(m, exp), lower, upper] {
                if cand == 0 {
                    continue;
                }
                let text = format!("{cand}e{}", cand_exp - (shorter as i32) + 1);
                let parsed: f64 = text.parse().unwrap();
                let signed = if neg { -parsed } else { parsed };
                assert_ne!(
                    narrow(signed),
                    stored_bits,
                    "a {shorter}-digit decimal `{text}` round-trips {image:e} \
                     but format_element used {p} digits"
                );
            }
        }
    }

    /// The decimal/e-notation decision is made on the RENDERED magnitude
    /// (spec/05 section 8.1; PR #792 red-team F2): parse the emitted text
    /// back and require e-notation exactly when the denoted value falls
    /// outside `[1e-4, 1e16)`. Run over every bit pattern, this locks the
    /// straddling-ulp boundary cases (e.g. bf16 image 9.9921e15 rendering
    /// `1e16`).
    fn assert_form_matches_rendered_magnitude(text: &str, bits: u16) {
        let rendered: f64 = text.parse().expect("grammar output parses");
        let expect_e = rendered != 0.0
            && !(DECIMAL_LOWER_BOUND..DECIMAL_UPPER_BOUND).contains(&rendered.abs());
        assert_eq!(
            text.contains('e'),
            expect_e,
            "bits {bits:#06x}: `{text}` denotes {rendered:e}; its form must \
             follow the rendered magnitude"
        );
    }

    #[test]
    fn every_f16_bit_pattern_round_trips_shortest() {
        for bits in 0..=u16::MAX {
            let v = half::f16::from_bits(bits);
            let text = format_element(Prim::F16, ElementRef::F16(v));
            if v.is_nan() {
                assert_eq!(text, "NaN", "f16 bits {bits:#06x}");
                continue;
            }
            let parsed: f64 = text.parse().unwrap_or_else(|e| {
                panic!("f16 bits {bits:#06x} rendered unparseable `{text}`: {e}")
            });
            assert_eq!(
                crate::dtype_semantics::f16_from_f64_rne(parsed).to_bits(),
                bits,
                "f16 bits {bits:#06x} (`{}`) rendered `{text}` which does not \
                 round-trip at f16 width",
                f64::from(v)
            );
            if v.is_finite() && f64::from(v) != 0.0 {
                no_shorter_rendering(
                    f64::from(v),
                    u64::from(bits),
                    &|x| u64::from(crate::dtype_semantics::f16_from_f64_rne(x).to_bits()),
                    significant_digits(&text),
                );
                assert_form_matches_rendered_magnitude(&text, bits);
            }
        }
    }

    #[test]
    fn every_bf16_bit_pattern_round_trips_shortest() {
        for bits in 0..=u16::MAX {
            let v = half::bf16::from_bits(bits);
            let text = format_element(Prim::Bf16, ElementRef::Bf16(v));
            if v.is_nan() {
                assert_eq!(text, "NaN", "bf16 bits {bits:#06x}");
                continue;
            }
            let parsed: f64 = text.parse().unwrap_or_else(|e| {
                panic!("bf16 bits {bits:#06x} rendered unparseable `{text}`: {e}")
            });
            assert_eq!(
                crate::dtype_semantics::bf16_from_f64_rne(parsed).to_bits(),
                bits,
                "bf16 bits {bits:#06x} (`{}`) rendered `{text}` which does not \
                 round-trip at bf16 width",
                f64::from(v)
            );
            if v.is_finite() && f64::from(v) != 0.0 {
                no_shorter_rendering(
                    f64::from(v),
                    u64::from(bits),
                    &|x| u64::from(crate::dtype_semantics::bf16_from_f64_rne(x).to_bits()),
                    significant_digits(&text),
                );
                assert_form_matches_rendered_magnitude(&text, bits);
            }
        }
    }

    /// The half grammar follows the same decimal/e-notation thresholds as
    /// `{:?}`, applied to the exact f64 image (spec/05 section 8.1).
    #[test]
    fn half_grammar_thresholds_match_the_normative_constants() {
        // bf16 spans both boundaries; sample values on each side.
        let bf = crate::dtype_semantics::bf16_from_f64_rne;
        // Below 1e-4: e-notation.
        let tiny = bf(5e-5);
        let tiny_text = format_element(Prim::Bf16, ElementRef::Bf16(tiny));
        assert!(
            tiny_text.contains('e'),
            "bf16 {tiny:?} below 1e-4 must use e-notation, got `{tiny_text}`"
        );
        // Above 1e16: e-notation.
        let big = bf(1e20);
        let big_text = format_element(Prim::Bf16, ElementRef::Bf16(big));
        assert!(
            big_text.contains('e'),
            "bf16 {big:?} above 1e16 must use e-notation, got `{big_text}`"
        );
        // In range: decimal with a fractional digit.
        let mid = bf(256.0);
        assert_eq!(format_element(Prim::Bf16, ElementRef::Bf16(mid)), "256.0");
        // The F2 boundary case: bf16(1e16) STORES an image below 1e16
        // (9.9921e15), but its shortest digits denote exactly 1e16, so the
        // rendered-magnitude rule takes e-notation - never the 17-character
        // decimal the stored-magnitude reading would produce.
        let boundary = bf(1e16);
        assert!(f64::from(boundary) < 1e16, "precondition: image below 1e16");
        assert_eq!(
            format_element(Prim::Bf16, ElementRef::Bf16(boundary)),
            "1e16"
        );
    }
}
