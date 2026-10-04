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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_scanner_requires_the_complete_json_grammar() {
        assert_eq!(json_number_token_len(b"-12.5e+2"), Ok(8));
        assert!(json_number_token_len(b".5").is_err());
        assert!(json_number_token_len(b"01").is_ok());
    }
}
