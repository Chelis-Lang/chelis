//! The compiled lane's shortest-round-trip float formatter
//! (chelis#732 Phase 2, `spec/design/faithful_observation.md` section C3.3).
//!
//! `chelis_format_shortest(double v, int dtype, char *buf, size_t cap)` is the one
//! float exit routine for everything the compiled lane renders: the
//! generated C print helper, the emitted scalar/labeled-root prints, and
//! the runtime's own nested-value renderer all call it (directly or through
//! [`format_shortest`]). Its output grammar is the frozen section C1.3 /
//! spec/05 section 8.1 contract, byte-identical to the reference renderer
//! `chelis_types::observation::format_element`:
//!
//! * shortest round-trip digits AT THE VALUE'S OWN STORAGE WIDTH (`dtype`
//!   is the `RuntimeDType` id of a float dtype; `v` carries the exact f64
//!   image of the stored value, which every float width widens to
//!   losslessly). STORAGE width, not arithmetic width: spec/04
//!   [04-NUM-8] gives f16 and bf16 an arithmetic width of f32 while their
//!   storage width stays 16, and [05-OBS-2] renders at storage. The
//!   parameter is spelled `dtype` for exactly that reason - it names a
//!   dtype, and "width" now denotes two different properties of one;
//! * decimal form exactly when the RENDERED magnitude is zero or inside
//!   `[1e-4, 1e16)`, e-notation otherwise, decided on the digits actually
//!   printed (the rendered-magnitude rule, PR #792 red-team F2);
//! * decimal renderings of integral values keep one fractional digit
//!   (`2048.0`, never `2048`);
//! * e-notation is `<mantissa>e<exp>`: lowercase `e`, no `+`, no zero
//!   padding;
//! * specials spell `inf` / `-inf` / `NaN`; `-0.0` keeps its sign.
//!
//! ## Implementation note (the section C3.3 sketch vs what landed)
//!
//! The design sketched a C-side `%.{p}g` / `strtod` escalation loop with a
//! printf-variance normalization pass, because it assumed the routine
//! would be written in C. The runtime library is Rust, so the wide widths
//! use the normative grammar's own definition directly: `{:?}` formatting,
//! the exact code path the reference renderer takes for f32/f64 - grammar
//! identity by construction, and no platform printf variance exists to
//! normalize. The narrow widths (f16/bf16, which Rust cannot format
//! natively) use the SAME ratified escalation the reference implements:
//! digit counts 1..=5, the correctly rounded scientific form plus its two
//! decimal grid neighbors at each count, first round-tripping count wins,
//! same-length ties break to the numerically closest then the even
//! mantissa (spec/05 section 8.1). Byte equality across every caller is
//! LOCKED by `tests/format_shortest_matches_reference.rs` - exhaustive
//! over all 65536 bit patterns per half format, table- and sweep-driven
//! for f32/f64.

use crate::require_runtime_dtype;
use chelis_vocab::RuntimeDType;
use libc::{c_char, c_int};

/// Byte capacity every `chelis_format_shortest` output fits in, NUL
/// included. The longest possible rendering is a negative f64 subnormal in
/// e-notation (`-2.2250738585072014e-308`, 24 bytes) or a negative
/// 17-digit decimal just above the lower threshold
/// (`-0.00012345678901234567`, 22 bytes); 32 leaves headroom and matches
/// the `CHELIS_FORMAT_SHORTEST_BUF` macro in `chelis_runtime.h`.
pub const CHELIS_FORMAT_SHORTEST_BUF: usize = 32;

/// Render `image` (the exact f64 widening of a stored float value) in the
/// frozen grammar at `width`'s own precision. Aborts loudly when `width`
/// is not a float dtype: integer and bool payloads have their own exact
/// exits and must never route through the float formatter.
pub(crate) fn format_shortest(image: f64, width: RuntimeDType) -> String {
    match width {
        // `{:?}` IS the normative grammar for the wide widths (spec/05
        // section 8.1); the reference renderer formats them identically.
        RuntimeDType::F64 => format!("{image:?}"),
        RuntimeDType::F32 => format!("{:?}", image as f32),
        RuntimeDType::F16 => format_half(
            image,
            u64::from(half::f16::from_f64(image).to_bits()),
            &|x| u64::from(half::f16::from_f64(x).to_bits()),
            "f16",
        ),
        RuntimeDType::Bf16 => format_half(
            image,
            u64::from(half::bf16::from_f64(image).to_bits()),
            &|x| u64::from(half::bf16::from_f64(x).to_bits()),
            "bf16",
        ),
        other => runtime_fail!(
            "chelis_format_shortest: dtype {} ({}) is not a float dtype",
            other.id(),
            other.name()
        ),
    }
}

/// Shortest-round-trip rendering for a half-precision value, in the
/// normative grammar. `image` is the exact f64 widening of the stored
/// value, `stored_bits` its bit pattern (widened to u64), and `narrow`
/// the parse-back narrowing (f64 -> half bits) for the width. Mirrors
/// `chelis_types::observation::format_half`.
fn format_half(image: f64, stored_bits: u64, narrow: &dyn Fn(f64) -> u64, width: &str) -> String {
    if image.is_nan() {
        // NaN round-trips at the class level: exit text carries no payload
        // (spec/05 section 8.1).
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

/// Digit counts that always suffice for a half format: 5 covers f16's 11
/// significand bits (ceil(11 log10 2) + 1) and bf16's 8; the exhaustive
/// bit-pattern tests prove the bound over every representable value.
const HALF_MAX_DIGITS: u32 = 5;

/// Find the shortest decimal digit string `D` (no trailing zeros) and
/// scientific exponent `E` (value = `D[0].D[1..] x 10^E`) such that the
/// signed rendering parses back to the stored bits. Searches digit counts
/// `p = 1..=5`; at each `p` the three candidates around the correctly
/// rounded `p`-digit decimal of `abs` cover every `p`-digit decimal that
/// could round-trip (the nearest decimal and its two grid neighbors, with
/// decade-boundary refinement handled explicitly), so the first hit is
/// shortest in digits. Among same-length hits the numerically closest to
/// `abs` wins; an exact tie prefers the even mantissa (deterministic,
/// spec/05 section 8.1). Mirrors
/// `chelis_types::observation::shortest_digits` (whose comparison through
/// parsed f64 values is exact here: a five-digit decimal's f64 image is
/// close enough that distinct candidates never collapse).
fn shortest_digits(abs: f64, round_trips: &dyn Fn(&str) -> bool, width: &str) -> (String, i32) {
    for p in 1..=HALF_MAX_DIGITS {
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
    // Five significant digits uniquely identify every half value, and the
    // image is exact at that width; unreachable for finite nonzero input.
    runtime_fail!(
        "chelis_format_shortest: no {HALF_MAX_DIGITS}-digit decimal round-trips a finite {width} value ({abs:e})"
    )
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

/// Assemble digits + scientific exponent into the frozen grammar, choosing
/// decimal vs e-notation from the RENDERED magnitude - the value the
/// chosen shortest digits denote, whose decade is exactly `sci_exp` -
/// against the normative thresholds: decimal iff the rendered value `v`
/// satisfies `1e-4 <= |v| < 1e16`, i.e. iff `-4 <= sci_exp <= 15` (zero is
/// handled earlier and renders decimal). Mirrors
/// `chelis_types::observation::assemble`.
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

/// C ABI entry point (section C3.3). `value` is the exact f64 image of the
/// stored float, `dtype` its `RuntimeDType` id, `buf` the caller's
/// output buffer, and `cap` that buffer's capacity in bytes. Returns the
/// number of bytes written EXCLUDING the terminating NUL.
///
/// Every contract violation aborts loudly rather than truncating or
/// returning a sentinel (the section C1 response; a silent short write is
/// the chelis#703 substitution shape at a byte level): an unknown id or a
/// non-float width aborts with the raw id, a null `buf` aborts, and a
/// `cap` too small for the rendering plus its NUL aborts naming both
/// numbers. The return is therefore always a valid length and a caller
/// never has to branch on it - it is there so a caller that wants the
/// length does not have to `strlen` the result.
///
/// `cap` exists because the previous signature could not express, let
/// alone check, the buffer contract its own docs stated: a red team
/// passed a four-byte logical buffer and the routine wrote past it
/// (PR #863 R2). `CHELIS_FORMAT_SHORTEST_BUF` remains the documented
/// minimum, and `sizeof` at the call site is the intended spelling.
///
/// # Safety
///
/// `buf` must point to at least `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn chelis_format_shortest(
    value: f64,
    dtype: c_int,
    buf: *mut c_char,
    cap: usize,
) -> c_int {
    let width = require_runtime_dtype(dtype, "chelis_format_shortest dtype");
    if buf.is_null() {
        runtime_fail!("chelis_format_shortest: null output buffer");
    }
    let text = format_shortest(value, width);
    let bytes = text.as_bytes();
    if bytes.len() >= CHELIS_FORMAT_SHORTEST_BUF {
        // OUR contract, not the caller's: the grammar's length bound says
        // this is unreachable. Abort rather than let a longer rendering
        // silently demand more room than the documented minimum.
        runtime_fail!(
            "chelis_format_shortest: rendering `{text}` exceeds the {CHELIS_FORMAT_SHORTEST_BUF}-byte contract"
        );
    }
    // The CALLER's contract. Checked separately so the diagnostic names
    // which side is wrong.
    let needed = bytes.len() + 1;
    if cap < needed {
        runtime_fail!(
            "chelis_format_shortest: rendering `{text}` needs {needed} bytes including the NUL, caller declared {cap} (minimum {CHELIS_FORMAT_SHORTEST_BUF})"
        );
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
        *buf.add(bytes.len()) = 0;
    }
    bytes.len() as c_int
}
