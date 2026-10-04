//! [05-OP-1] `round_to`, shared by every execution lane.
//!
//! The evaluator and the `chelis_round_to` C export both call these
//! definitions, so the decimal rounding, its admitted `places` domain, and
//! its failure text have one definition.

/// The admitted `places` domain and its failure text.
fn checked_places(places: i64) -> Result<usize, String> {
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
    Ok(usize::try_from(places).expect("checked nonnegative"))
}

/// Rust's fixed-precision formatting rounds the exact binary value to the
/// nearest `places`-digit decimal with ties to even; parsing that decimal is
/// the one finalization to the operand's own width.
fn round_via_decimal_text<T>(value: T, places: usize) -> T
where
    T: std::fmt::Display + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let text = format!("{value:.places$}");
    text.parse::<T>().unwrap_or_else(|error| {
        panic!("decimal formatter emitted an unparsable token `{text}`: {error}")
    })
}

/// [05-OP-1] at f64. A non-finite operand passes through unchanged.
pub fn round_to_f64(x: f64, places: i64) -> Result<f64, String> {
    let places = checked_places(places)?;
    Ok(if x.is_finite() {
        round_via_decimal_text(x, places)
    } else {
        x
    })
}

/// [05-OP-1] at f32, computed and finalized at f32 alone.
pub fn round_to_f32(x: f32, places: i64) -> Result<f32, String> {
    let places = checked_places(places)?;
    Ok(if x.is_finite() {
        round_via_decimal_text(x, places)
    } else {
        x
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_go_to_even_on_the_exact_binary_value() {
        assert_eq!(round_to_f64(0.125, 2), Ok(0.12));
        assert_eq!(round_to_f64(0.375, 2), Ok(0.38));
        assert_eq!(round_to_f64(2.675, 2), Ok(2.67));
        assert_eq!(round_to_f32(0.125, 2), Ok(0.12_f32));
        assert_eq!(round_to_f32(2.675, 2), Ok(2.67_f32));
    }

    #[test]
    fn non_finite_operands_keep_their_bits() {
        assert!(round_to_f64(f64::NAN, 2).unwrap().is_nan());
        assert_eq!(round_to_f32(f32::NEG_INFINITY, 0), Ok(f32::NEG_INFINITY));
        assert_eq!(
            round_to_f64(-0.001, 1).map(f64::to_bits),
            Ok((-0.0_f64).to_bits())
        );
    }

    #[test]
    fn places_outside_the_domain_fail_loudly() {
        assert_eq!(
            round_to_f64(1.0, -1),
            Err("round_to: places must be in 0..=100, got -1 (negative decimal places are not supported)".to_string())
        );
        assert!(round_to_f32(1.0, 101).is_err());
    }
}
