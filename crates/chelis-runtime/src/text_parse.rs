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
//! characters. After it, the standard library's integer grammar is an optional
//! sign followed by one or more ASCII digits, and its float grammar is, ignoring
//! case, an optional sign followed by `inf`, `infinity`, `nan`, or
//! `([0-9]+(\.[0-9]*)?|\.[0-9]+)(e[-+]?[0-9]+)?`. Both are the atom's grammars.
//! The float conversion is correctly rounded to nearest, ties to even, so
//! overflow gives a signed infinity and underflow gives a signed zero or
//! subnormal.

/// [05-OP-59] `to_int`: the exact i64 of a trimmed signed decimal integer, or
/// `None` for malformed or out-of-range text.
pub fn to_int(text: &str) -> Option<i64> {
    text.trim().parse::<i64>().ok()
}

/// [05-OP-59] `to_float`: the correctly rounded f64 of trimmed float text, or
/// `None` for malformed text. No valid spelling fails, overflow included.
pub fn to_float(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok()
}
