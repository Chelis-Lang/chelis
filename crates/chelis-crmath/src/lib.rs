//! Correctly rounded transcendentals for every Chelis lane.
//!
//! [05-OP-46] makes `exp`, `log`, `sin`, `cos`, `tan`, `atan`, and `tanh` correctly
//! rounded: the exact real value rounded once, ties to even, at [04-NUM-8]'s
//! arithmetic width, with every NaN result finalized to [04-NUM-2]'s canonical quiet
//! NaN. This crate is the only place a Rust lane may compute them. The kernels are
//! CORE-MATH's (vendored unmodified under `vendor/core-math`, MIT licence), compiled
//! from the generated `csrc/crmath_amalgamation.c` that the C backend also emits, so
//! the evaluator and built programs run the same kernel text. The amalgamation is the
//! upstream text reduced to the generated-C contract (portable arms only, no
//! floating-point environment access); `scripts/vendor_core_math.py` documents the
//! reduction.
//! `spec/design/correctly_rounded_math.md` owns the design.
//!
//! The API is closed and per dtype. The f16 and bf16 forms widen the operand exactly to
//! f32, run the f32 kernel, and finalize that result to storage once, which is the
//! composition [05-OP-46] defines for those dtypes (not a rounding taken directly to
//! the storage width). `sqrt` is not here: IEEE 754 already makes the hardware square
//! root correctly rounded.
//!
//! The workspace `clippy.toml` disallows `f32::exp` and the other libm-backed float
//! methods everywhere else, so a new caller cannot bypass these kernels.

use half::{bf16, f16};

pub mod c_source;

mod ffi {
    // The shims in `csrc/crmath_ffi.c`. They are pure functions of their operand:
    // no global or thread-local state, no pointer arguments.
    unsafe extern "C" {
        pub safe fn chelis_crmath_ffi_expf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_logf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_sinf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_cosf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_tanf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_atanf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_tanhf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_exp(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_log(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_sin(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_cos(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_tan(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_atan(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_tanh(x: f64) -> f64;
    }
}

macro_rules! correctly_rounded {
    ($(($math:literal, $f32:ident, $f64:ident, $f16:ident, $bf16:ident, $ffi32:ident, $ffi64:ident)),* $(,)?) => {
        $(
            #[doc = concat!("Correctly rounded ", $math, " at f32 ([05-OP-46]).")]
            #[inline]
            #[must_use]
            pub fn $f32(x: f32) -> f32 {
                ffi::$ffi32(x)
            }

            #[doc = concat!("Correctly rounded ", $math, " at f64 ([05-OP-46]).")]
            #[inline]
            #[must_use]
            pub fn $f64(x: f64) -> f64 {
                ffi::$ffi64(x)
            }

            #[doc = concat!(
                "The f16 ", $math, ": correctly rounded at f32, then finalized to f16 once ([05-OP-46])."
            )]
            #[inline]
            #[must_use]
            pub fn $f16(x: f16) -> f16 {
                f16::from_f32($f32(x.to_f32()))
            }

            #[doc = concat!(
                "The bf16 ", $math, ": correctly rounded at f32, then finalized to bf16 once ([05-OP-46])."
            )]
            #[inline]
            #[must_use]
            pub fn $bf16(x: bf16) -> bf16 {
                bf16::from_f32($f32(x.to_f32()))
            }
        )*
    };
}

correctly_rounded! {
    ("natural exponential", exp_f32, exp_f64, exp_f16, exp_bf16, chelis_crmath_ffi_expf, chelis_crmath_ffi_exp),
    ("natural logarithm", log_f32, log_f64, log_f16, log_bf16, chelis_crmath_ffi_logf, chelis_crmath_ffi_log),
    ("sine (radians)", sin_f32, sin_f64, sin_f16, sin_bf16, chelis_crmath_ffi_sinf, chelis_crmath_ffi_sin),
    ("cosine (radians)", cos_f32, cos_f64, cos_f16, cos_bf16, chelis_crmath_ffi_cosf, chelis_crmath_ffi_cos),
    ("tangent (radians)", tan_f32, tan_f64, tan_f16, tan_bf16, chelis_crmath_ffi_tanf, chelis_crmath_ffi_tan),
    ("principal arctangent", atan_f32, atan_f64, atan_f16, atan_bf16, chelis_crmath_ffi_atanf, chelis_crmath_ffi_atan),
    ("hyperbolic tangent", tanh_f32, tanh_f64, tanh_f16, tanh_bf16, chelis_crmath_ffi_tanhf, chelis_crmath_ffi_tanh),
}

#[cfg(test)]
mod tests;
