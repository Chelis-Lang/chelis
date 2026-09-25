//! [05-RNG-2]'s key derivations and draw words, and the arms of [05-OP-37]
//! and [05-OP-8] these tests read over them, transcribed from
//! `briefs/switch-design-probes/key_ref.py` (the design pass's transcription
//! of `spec/design/randomness_explicit_keys.md`) and the spec/05 atom text.
//! It shares no code with any lane, so an expected draw comes from the text,
//! never from the implementation. `key_reference_matches_key_ref_py` in
//! `dropout_fixed_stream_api.rs` pins it against values `key_ref.py` prints.
#![allow(dead_code)]

pub fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// Spelled out from the definition rather than through `rotate_left`.
#[allow(clippy::manual_rotate)]
fn rotl64(x: u64, r: u32) -> u64 {
    (x << r) | (x >> (64 - r))
}

/// `key_from_seed(seed)`: the seed's two's-complement bits, unmixed.
pub fn key_from_seed(seed: i64) -> u64 {
    seed as u64
}

pub fn derive(k: u64, j: u64) -> u64 {
    splitmix64(k ^ rotl64(splitmix64(j), 29))
}

/// `split_key(k) = (derive(k, 0), derive(k, 1))`.
pub fn split(k: u64) -> (u64, u64) {
    (derive(k, 0), derive(k, 1))
}

/// `fold_in(k, n) = derive(derive(k, 2), n)`.
pub fn fold_in(k: u64, n: i64) -> u64 {
    derive(derive(k, 2), n as u64)
}

pub fn word(k: u64, i: u64) -> u64 {
    splitmix64(k ^ rotl64(splitmix64(i), 41))
}

/// The high 53 bits of `word(k, i)` over 2^53, exact in f64.
pub fn unit(k: u64, i: u64) -> f64 {
    (word(k, i) >> 11) as f64 / (1u64 << 53) as f64
}

/// [05-OP-37] at rate 0.5 over `input`. Element `i` is dropped (positive
/// zero) when the unit value at the dtype's arithmetic width is below the
/// rate: the exact unit for f64 (`exact`), its f32 rounding for f32, f16 and
/// bf16. A kept element is `input[i] / (1 - 0.5)`, which is exact at every
/// float dtype for the small integers these tests use.
pub fn half_dropout(k: u64, input: &[f64], exact: bool) -> Vec<f64> {
    input
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let unit = unit(k, i as u64);
            let unit = if exact { unit } else { f64::from(unit as f32) };
            if unit < 0.5 { 0.0 } else { x / 0.5 }
        })
        .collect()
}

/// `half_dropout` over `count` ones at the arithmetic width `exact` selects.
pub fn mask_at_width(k: u64, count: usize, exact: bool) -> Vec<f64> {
    half_dropout(k, &vec![1.0; count], exact)
}

/// `half_dropout` over `count` ones at a width narrower than f64.
pub fn mask(k: u64, count: usize) -> Vec<f64> {
    mask_at_width(k, count, false)
}

/// [05-OP-8] over `[0, 1)` at f32: `fma(1 - 0, round_f32(u), 0)`, which is
/// `round_f32(u)`.
pub fn unit_f32(k: u64, count: usize) -> Vec<f32> {
    (0..count as u64).map(|i| unit(k, i) as f32).collect()
}

/// The keys a program names `k1` and `k2` after
/// `(k1, k2) = split_key(key_from_seed(42i64))`.
pub fn two_keys() -> (u64, u64) {
    split(key_from_seed(42))
}

/// The keys a program names `k1`, `k2` and `k3` after
/// `(k1, rest) = split_key(key_from_seed(42i64))` and
/// `(k2, k3) = split_key(rest)`.
pub fn three_keys() -> (u64, u64, u64) {
    let (k1, rest) = split(key_from_seed(42));
    let (k2, k3) = split(rest);
    (k1, k2, k3)
}

/// The keys a program names `k1` to `k4` after
/// `(k1, rest) = split_key(key_from_seed(42i64))`,
/// `(k2, more) = split_key(rest)` and `(k3, k4) = split_key(more)`.
pub fn four_keys() -> (u64, u64, u64, u64) {
    let (k1, rest) = split(key_from_seed(42));
    let (k2, more) = split(rest);
    let (k3, k4) = split(more);
    (k1, k2, k3, k4)
}

/// The program text binding `k1` and `k2` as [`two_keys`] names them.
pub const TWO_KEYS: &str = "(k1, k2) = split_key(key_from_seed(42i64))";
/// The program text binding `k1` to `k3` as [`three_keys`] names them.
pub const THREE_KEYS: &str =
    "(k1, rest) = split_key(key_from_seed(42i64))\n (k2, k3) = split_key(rest)";
/// The program text binding `k1` to `k4` as [`four_keys`] names them.
pub const FOUR_KEYS: &str = "(k1, rest) = split_key(key_from_seed(42i64))\n (k2, more) = split_key(rest)\n (k3, k4) = split_key(more)";
