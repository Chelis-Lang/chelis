//! Private target-width IEEE narrowing for runtime-owned f64 carriers.
//!
//! This is deliberately not exported through the C ABI: numeric values cross
//! that boundary on tagged or declared-width carriers. Runtime list ingress
//! and shortest-round-trip validation still begin with a Rust `f64`, so they
//! need the same one-step [04-NUM-14] conversion as the evaluator.

pub(crate) fn f64_to_f16_bits_rne(value: f64) -> u16 {
    chelis_numeric_core::ieee::f64_to_f16_bits_rne(value)
}

pub(crate) fn f64_to_bf16_bits_rne(value: f64) -> u16 {
    chelis_numeric_core::ieee::f64_to_bf16_bits_rne(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_narrowing_keeps_the_f64_side_of_reduced_midpoints() {
        assert_eq!(f64_to_f16_bits_rne(52847.99970178839), 0x7a73);
        assert_eq!(
            f64_to_f16_bits_rne(f64::from(52847.99970178839_f64 as f32)),
            0x7a74
        );
        assert_eq!(f64_to_bf16_bits_rne(1.0039062500000002), 0x3f81);
        assert_eq!(
            f64_to_bf16_bits_rne(f64::from(1.0039062500000002_f64 as f32)),
            0x3f80
        );
    }

    #[test]
    fn runtime_narrowing_rounds_midpoints_ties_to_even_at_each_storage_width() {
        fn check(last_finite: u16, image: impl Fn(u16) -> f64, runtime_round: impl Fn(f64) -> u16) {
            for lower_bits in 0..last_finite {
                let midpoint = (image(lower_bits) + image(lower_bits + 1)) / 2.0;
                let even_bits = lower_bits + (lower_bits & 1);
                for (value, expected) in [
                    (f64::from_bits(midpoint.to_bits() - 1), lower_bits),
                    (midpoint, even_bits),
                    (f64::from_bits(midpoint.to_bits() + 1), lower_bits + 1),
                ] {
                    assert_eq!(runtime_round(value), expected);
                    assert_eq!(runtime_round(-value), expected | 0x8000);
                }
            }
        }

        check(
            0x7bff,
            |bits| f64::from(half::f16::from_bits(bits)),
            f64_to_f16_bits_rne,
        );
        check(
            0x7f7f,
            |bits| f64::from(half::bf16::from_bits(bits)),
            f64_to_bf16_bits_rne,
        );
    }
}
