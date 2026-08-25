//! Shared numeric-text helpers for the host I/O lane.
//!
//! JSON itself is implemented by `Std.Io.Json`; these helpers are not a
//! second JSON value representation.  CSV uses the JSON number grammar for
//! explicit numeric accessors, while `round_to` owns decimal rounding.

/// Scan one JSON number token
/// (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`) at the start of
/// `bytes`. `Ok(len)` is the token length; `Err((offset, reason))` pinpoints
/// the first offending byte.
pub(super) fn json_number_token_len(bytes: &[u8]) -> Result<usize, (usize, &'static str)> {
    let mut i = 0;
    if bytes.get(i) == Some(&b'-') {
        i += 1;
    }
    match bytes.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            while matches!(bytes.get(i), Some(b'0'..=b'9')) {
                i += 1;
            }
        }
        _ => return Err((i, "invalid number (expected a digit)")),
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        if !matches!(bytes.get(i), Some(b'0'..=b'9')) {
            return Err((i, "invalid number (expected a digit after `.`)"));
        }
        while matches!(bytes.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !matches!(bytes.get(i), Some(b'0'..=b'9')) {
            return Err((i, "invalid number (expected a digit in exponent)"));
        }
        while matches!(bytes.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    Ok(i)
}

pub(super) fn round_to_f64_impl(x: f64, places: i64) -> Result<f64, String> {
    round_to_places(places)?;
    if !x.is_finite() {
        return Ok(x);
    }
    match chelis_types::observation::round_element_to_decimal_places(
        chelis_types::ElementRef::F64(x),
        places as usize,
    ) {
        chelis_types::ElementRef::F64(value) => Ok(value),
        _ => unreachable!("the sealed f64 formatter preserves its variant"),
    }
}

pub(super) fn round_to_f32_impl(x: f32, places: i64) -> Result<f32, String> {
    round_to_places(places)?;
    if !x.is_finite() {
        return Ok(x);
    }
    match chelis_types::observation::round_element_to_decimal_places(
        chelis_types::ElementRef::F32(x),
        places as usize,
    ) {
        chelis_types::ElementRef::F32(value) => Ok(value),
        _ => unreachable!("the sealed f32 formatter preserves its variant"),
    }
}

fn round_to_places(places: i64) -> Result<(), String> {
    if !(0..=100).contains(&places) {
        let reason = if places < 0 {
            "negative decimal places are not supported"
        } else {
            "beyond 100 fractional digits every finite f64/f32 is already exact"
        };
        return Err(format!(
            "round_to: places must be in 0..=100, got {places} ({reason})"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_scanner_requires_the_complete_json_grammar() {
        assert_eq!(json_number_token_len(b"-12.5e+2"), Ok(8));
        assert!(json_number_token_len(b".5").is_err());
        assert!(json_number_token_len(b"01").is_ok());
    }

    #[test]
    fn round_to_keeps_own_width_and_ties_to_even() {
        assert_eq!(round_to_f64_impl(0.125, 2).unwrap(), 0.12);
        assert_eq!(round_to_f64_impl(0.375, 2).unwrap(), 0.38);
        assert_eq!(round_to_f32_impl(0.125, 2).unwrap(), 0.12_f32);
    }
}
