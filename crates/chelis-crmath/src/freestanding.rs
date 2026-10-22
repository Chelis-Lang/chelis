//! IEEE helpers for CORE-MATH on targets without a C library.
//!
//! These private link symbols do not form a published Chelis C ABI.

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_fma(x: f64, y: f64, z: f64) -> f64 {
    libm::fma(x, y, z)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_fmaf(x: f32, y: f32, z: f32) -> f32 {
    libm::fmaf(x, y, z)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_floor(x: f64) -> f64 {
    libm::floor(x)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_fmax(x: f64, y: f64) -> f64 {
    libm::fmax(x, y)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_rint(x: f64) -> f64 {
    libm::rint(x)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_round(x: f64) -> f64 {
    libm::round(x)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_roundf(x: f32) -> f32 {
    libm::roundf(x)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_ldexp(x: f64, exponent: i32) -> f64 {
    libm::scalbn(x, exponent)
}

#[unsafe(no_mangle)]
extern "C" fn chelis_crmath_soft_nanf() -> f32 {
    f32::from_bits(0x7fc00000)
}
