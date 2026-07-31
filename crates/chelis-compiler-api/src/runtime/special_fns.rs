//! Special-function kernels for the eval-lane math builtins
//! (chelis#902): `erf`, `erfc`, `norm_cdf`, `norm_ppf`.
//!
//! All four compute in `f64`. The error-function pair is a line-for-line
//! transcription of the SunPro/fdlibm rational approximations
//! (origin: FreeBSD `/usr/src/lib/msun/src/s_erf.c`, Copyright (C) 1993
//! by Sun Microsystems, Inc., via the Rust `libm` crate's `erf.rs`
//! port, MIT) — accuracy ~1 ulp, NOT an invented polynomial. The
//! original Sun notice:
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```
//!
//! `norm_cdf` is the erfc half-form `Φ(x) = ½·erfc(−x/√2)` — a single
//! implementation path, so the parity identity `Φ(x) + Φ(−x) = 1` holds
//! to rounding and the deep tail keeps full relative accuracy (the naive
//! `½(1+erf)` form loses everything past `x ≈ −6`).
//!
//! `norm_ppf` is Wichura's Algorithm AS 241 (`PPND16`, *Applied
//! Statistics* 37(3), 1988, pp. 477–484), |Δz| ≈ 1e-16 relative over the
//! full domain — again a published algorithm transcribed, not fitted.

// The AS 241 coefficients below are kept digit-for-digit as published
// (18–20 significant digits); clippy's `excessive_precision` would have
// us truncate them to 17, which trades citation fidelity for nothing —
// the extra digits round to the same f64 or its correctly-rounded
// neighbor, and reviewers can diff against the paper directly.
#![allow(clippy::excessive_precision)]

const ERX: f64 = 8.450_629_115_104_675_3e-1; /* 0x3FEB0AC1, 0x60000000 */
/*
 * Coefficients for approximation to erf on [0, 0.84375]
 */
const EFX8: f64 = 1.027_033_336_764_100_7; /* 0x3FF06EBA, 0x8214DB69 */
const PP0: f64 = 1.283_791_670_955_125_9e-1; /* 0x3FC06EBA, 0x8214DB68 */
const PP1: f64 = -3.250_421_072_470_015e-1; /* 0xBFD4CD7D, 0x691CB913 */
const PP2: f64 = -2.848_174_957_559_851e-2; /* 0xBF9D2A51, 0xDBD7194F */
const PP3: f64 = -5.770_270_296_489_442e-3; /* 0xBF77A291, 0x236668E4 */
const PP4: f64 = -2.376_301_665_665_016_3e-5; /* 0xBEF8EAD6, 0x120016AC */
const QQ1: f64 = 3.979_172_239_591_553_4e-1; /* 0x3FD97779, 0xCDDADC09 */
const QQ2: f64 = 6.502_224_998_876_729e-2; /* 0x3FB0A54C, 0x5536CEBA */
const QQ3: f64 = 5.081_306_281_875_766e-3; /* 0x3F74D022, 0xC4D36B0F */
const QQ4: f64 = 1.324_947_380_043_216_4e-4; /* 0x3F215DC9, 0x221C1A10 */
const QQ5: f64 = -3.960_228_278_775_368e-6; /* 0xBED09C43, 0x42A26120 */
/*
 * Coefficients for approximation to erf on [0.84375, 1.25]
 */
const PA0: f64 = -2.362_118_560_752_659_4e-3; /* 0xBF6359B8, 0xBEF77538 */
const PA1: f64 = 4.148_561_186_837_483e-1; /* 0x3FDA8D00, 0xAD92B34D */
const PA2: f64 = -3.722_078_760_357_013_3e-1; /* 0xBFD7D240, 0xFBB8C3F1 */
const PA3: f64 = 3.183_466_199_011_617_5e-1; /* 0x3FD45FCA, 0x805120E4 */
const PA4: f64 = -1.108_946_942_823_966_8e-1; /* 0xBFBC6398, 0x3D3E28EC */
const PA5: f64 = 3.547_830_432_561_823_6e-2; /* 0x3FA22A36, 0x599795EB */
const PA6: f64 = -2.166_375_594_868_790_8e-3; /* 0xBF61BF38, 0x0A96073F */
const QA1: f64 = 1.064_208_804_008_442_3e-1; /* 0x3FBB3E66, 0x18EEE323 */
const QA2: f64 = 5.403_979_177_021_710_5e-1; /* 0x3FE14AF0, 0x92EB6F33 */
const QA3: f64 = 7.182_865_441_419_627e-2; /* 0x3FB2635C, 0xD99FE9A7 */
const QA4: f64 = 1.261_712_198_087_616_4e-1; /* 0x3FC02660, 0xE763351F */
const QA5: f64 = 1.363_708_391_202_905e-2; /* 0x3F8BEDC2, 0x6B51DD1C */
const QA6: f64 = 1.198_449_984_679_910_7e-2; /* 0x3F888B54, 0x5735151D */
/*
 * Coefficients for approximation to erfc on [1.25, 1/0.35]
 */
const RA0: f64 = -9.864_944_034_847_148e-3; /* 0xBF843412, 0x600D6435 */
const RA1: f64 = -6.938_585_727_071_818e-1; /* 0xBFE63416, 0xE4BA7360 */
const RA2: f64 = -1.055_862_622_532_329_1e1; /* 0xC0251E04, 0x41B0E726 */
const RA3: f64 = -6.237_533_245_032_601e1; /* 0xC04F300A, 0xE4CBA38D */
const RA4: f64 = -1.623_966_694_625_734_7e2; /* 0xC0644CB1, 0x84282266 */
const RA5: f64 = -1.846_050_929_067_110_4e2; /* 0xC067135C, 0xEBCCABB2 */
const RA6: f64 = -8.128_743_550_630_66e1; /* 0xC0545265, 0x57E4D2F2 */
const RA7: f64 = -9.814_329_344_169_145e0; /* 0xC023A0EF, 0xC69AC25C */
const SA1: f64 = 1.965_127_166_743_925_7e1; /* 0x4033A6B9, 0xBD707687 */
const SA2: f64 = 1.376_577_541_435_190_4e2; /* 0x4061350C, 0x526AE721 */
const SA3: f64 = 4.345_658_774_752_292_3e2; /* 0x407B290D, 0xD58A1A71 */
const SA4: f64 = 6.453_872_717_332_679e2; /* 0x40842B19, 0x21EC2868 */
const SA5: f64 = 4.290_081_400_275_678_3e2; /* 0x407AD021, 0x57700314 */
const SA6: f64 = 1.086_350_055_417_794_4e2; /* 0x405B28A3, 0xEE48AE2C */
const SA7: f64 = 6.570_249_770_319_282e0; /* 0x401A47EF, 0x8E484A93 */
const SA8: f64 = -6.042_441_521_485_81e-2; /* 0xBFAEEFF2, 0xEE749A62 */
/*
 * Coefficients for approximation to erfc on [1/0.35, 28]
 */
const RB0: f64 = -9.864_942_924_700_099e-3; /* 0xBF843412, 0x39E86F4A */
const RB1: f64 = -7.992_832_376_805_23e-1; /* 0xBFE993BA, 0x70C285DE */
const RB2: f64 = -1.775_795_491_775_475_2e1; /* 0xC031C209, 0x555F995A */
const RB3: f64 = -1.606_363_848_558_219_2e2; /* 0xC064145D, 0x43C5ED98 */
const RB4: f64 = -6.375_664_433_683_896e2; /* 0xC083EC88, 0x1375F228 */
const RB5: f64 = -1.025_095_131_611_077_2e3; /* 0xC0900461, 0x6A2E5992 */
const RB6: f64 = -4.835_191_916_086_514e2; /* 0xC07E384E, 0x9BDC383F */
const SB1: f64 = 3.033_806_074_348_245_8e1; /* 0x403E568B, 0x261D5190 */
const SB2: f64 = 3.257_925_129_965_739_2e2; /* 0x40745CAE, 0x221B9F0A */
const SB3: f64 = 1.536_729_586_084_437e3; /* 0x409802EB, 0x189D5118 */
const SB4: f64 = 3.199_858_219_508_595_5e3; /* 0x40A8FFB7, 0x688C246A */
const SB5: f64 = 2.553_050_406_433_164_4e3; /* 0x40A3F219, 0xCEDF3BE6 */
const SB6: f64 = 4.745_285_412_069_553_7e2; /* 0x407DA874, 0xE79FE763 */
const SB7: f64 = -2.244_095_244_658_581_8e1; /* 0xC03670E2, 0x42712D62 */

/// High 32 bits of the IEEE-754 representation (fdlibm's `GET_HIGH_WORD`).
#[inline]
fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

/// Replace the low 32 bits of the IEEE-754 representation (fdlibm's
/// `SET_LOW_WORD`); used to split `x` into a short prefix `z` so that
/// `-x*x = -z*z + (z-x)*(z+x)` evaluates `exp(-x*x)` without doubling
/// the rounding error of `x*x`.
#[inline]
fn with_set_low_word(x: f64, low: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xFFFF_FFFF_0000_0000) | u64::from(low))
}

/// erf on [0.84375, 1.25] via the Taylor expansion at x = 1:
/// `erfc(x) = 1 - erx - P1(s)/Q1(s)`, `s = |x| - 1`.
fn erfc1(x: f64) -> f64 {
    let s = x.abs() - 1.0;
    let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
    1.0 - ERX - p / q
}

/// erfc on [1.25, 28) for positive x:
/// `erfc(x) = exp(-x² - 0.5625 + R(1/x²)/S(1/x²)) / x`.
fn erfc2(ix: u32, x: f64) -> f64 {
    if ix < 0x3ff4_0000 {
        /* |x| < 1.25 */
        return erfc1(x);
    }
    let x = x.abs();
    let s = 1.0 / (x * x);
    let (r, big_s) = if ix < 0x4006_db6d {
        /* |x| < 1/0.35 ~ 2.85714 */
        (
            RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
            1.0 + s
                * (SA1
                    + s * (SA2
                        + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
        )
    } else {
        /* |x| >= 1/0.35 */
        (
            RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
        )
    };
    let z = with_set_low_word(x, 0);
    (-z * z - 0.5625).exp() * ((z - x) * (z + x) + r / big_s).exp() / x
}

/// Error function, f64, SunPro/fdlibm algorithm (~1 ulp).
///
/// `erf(NaN) = NaN`, `erf(±inf) = ±1`, odd: `erf(-x) = -erf(x)`.
pub fn erf(x: f64) -> f64 {
    let mut ix = get_high_word(x);
    let sign = (ix >> 31) as usize;
    ix &= 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        /* erf(nan)=nan, erf(±inf)=±1 */
        return 1.0 - 2.0 * (sign as f64) + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        /* |x| < 0.84375 */
        if ix < 0x3e30_0000 {
            /* |x| < 2^-28: avoid underflow */
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        return x + x * y;
    }
    let y = if ix < 0x4018_0000 {
        /* 0.84375 <= |x| < 6 */
        1.0 - erfc2(ix, x)
    } else {
        /* |x| >= 6: erf saturates; 1 - 2^-1022 keeps the inexact result */
        let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000);
        1.0 - x1p_1022
    };
    if sign != 0 { -y } else { y }
}

/// Complementary error function, f64, SunPro/fdlibm algorithm (~1 ulp).
///
/// Computed directly (not as `1 - erf(x)`), so the deep positive tail
/// keeps full relative accuracy down to the underflow bound near
/// x ≈ 27.2. `erfc(NaN) = NaN`, `erfc(-inf) = 2`, `erfc(+inf) = 0`.
pub fn erfc(x: f64) -> f64 {
    let mut ix = get_high_word(x);
    let sign = (ix >> 31) as usize;
    ix &= 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        /* erfc(nan)=nan, erfc(±inf)=0,2 */
        return 2.0 * (sign as f64) + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        /* |x| < 0.84375 */
        if ix < 0x3c70_0000 {
            /* |x| < 2^-56 */
            return 1.0 - x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        if sign != 0 || ix < 0x3fd0_0000 {
            /* x < 1/4 */
            return 1.0 - (x + x * y);
        }
        return 0.5 - (x - 0.5 + x * y);
    }
    if ix < 0x403c_0000 {
        /* 0.84375 <= |x| < 28 */
        return if sign != 0 {
            2.0 - erfc2(ix, x)
        } else {
            erfc2(ix, x)
        };
    }
    /* |x| >= 28 */
    let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000);
    if sign != 0 {
        2.0 - x1p_1022 /* raises inexact */
    } else {
        x1p_1022 * x1p_1022 /* raises underflow: +0 */
    }
}

/// `1/√2`, correctly rounded (0x3FE6A09E667F3BCD).
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Standard normal CDF via the erfc half-form:
/// `Φ(x) = ½·erfc(−x/√2)`.
///
/// One implementation path for both tails, so `Φ(x) + Φ(−x) = 1` holds
/// to rounding (the parity identities of option pricing depend on this)
/// and the lower tail keeps relative accuracy down to
/// `Φ(-37.5) ≈ 4.6e-308` instead of flushing to 0 near x = -8 the way
/// `½(1 + erf(x/√2))` does. `Φ(NaN) = NaN`, `Φ(-inf) = 0`, `Φ(+inf) = 1`.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal quantile Φ⁻¹(p): Wichura's Algorithm AS 241,
/// routine PPND16 (*Applied Statistics* 37(3), 1988, pp. 477-484),
/// relative accuracy about 1e-16 over the full domain.
///
/// Domain: `norm_ppf(0) = -inf`, `norm_ppf(1) = +inf`, NaN propagates,
/// and p outside [0, 1] is a loud error (the caller renders it) — never
/// a silent NaN.
pub fn norm_ppf(p: f64) -> Result<f64, String> {
    if p.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&p) {
        // Faithful observation (B2.4): the rejected payload renders through
        // the generated formatter, not a Debug spelling.
        return Err(format!(
            "norm_ppf domain error: p must be in [0, 1], got {}",
            chelis_types::format_element(
                chelis_types::types::Prim::F64,
                chelis_types::ElementRef::F64(p),
            )
        ));
    }
    if p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }

    /* AS 241 PPND16 coefficients, verbatim from Wichura (1988). */
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        /* central region: |q| <= 0.425, rational in r = 0.180625 - q^2 */
        let r = 0.180_625 - q * q;
        let num = (((((((2.5090809287301226727e3) * r + 3.3430575583588128105e4) * r
            + 6.7265770927008700853e4)
            * r
            + 4.5921953931549871457e4)
            * r
            + 1.3731693765509461125e4)
            * r
            + 1.9715909503065514427e3)
            * r
            + 1.3314166789178437745e2)
            * r
            + 3.3871328727963666080;
        let den = (((((((5.2264952788528545610e3) * r + 2.8729085735721942674e4) * r
            + 3.9307895800092710610e4)
            * r
            + 2.1213794301586595867e4)
            * r
            + 5.3941960214247511077e3)
            * r
            + 6.8718700749205790830e2)
            * r
            + 4.2313330701600911252e1)
            * r
            + 1.0;
        return Ok(q * num / den);
    }

    /* tail regions: r = sqrt(-ln(min(p, 1 - p))) */
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let z = if r <= 5.0 {
        /* 1.6 < r <= 5, i.e. min(p, 1-p) down to ~1.39e-11 */
        let r = r - 1.6;
        let num = (((((((7.74545014278341407640e-4) * r + 2.27238449892691845833e-2) * r
            + 2.41780725177450611770e-1)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.76949722146069140550)
            * r
            + 4.63033784615654529590)
            * r
            + 1.42343711074968357734;
        let den = (((((((1.05075007164441684324e-9) * r + 5.47593808499534494600e-4) * r
            + 1.51986665636164571966e-2)
            * r
            + 1.48103976427480074590e-1)
            * r
            + 6.89767334985100004550e-1)
            * r
            + 1.67638483018380384940)
            * r
            + 2.05319162663775882187)
            * r
            + 1.0;
        num / den
    } else {
        /* r > 5: extreme tail, min(p, 1-p) below ~1.39e-11 */
        let r = r - 5.0;
        let num = (((((((2.01033439929228813265e-7) * r + 2.71155556874348757815e-5) * r
            + 1.24266094738807843860e-3)
            * r
            + 2.65321895265761230930e-2)
            * r
            + 2.96560571828504891230e-1)
            * r
            + 1.78482653991729133580)
            * r
            + 5.46378491116411436990)
            * r
            + 6.65790464350110377720;
        let den = (((((((2.04426310338993978564e-15) * r + 1.42151175831644588870e-7) * r
            + 1.84631831751005468180e-5)
            * r
            + 7.86869131145613259100e-4)
            * r
            + 1.48753612908506148525e-2)
            * r
            + 1.36929880922735805310e-1)
            * r
            + 5.99832206555887937690e-1)
            * r
            + 1.0;
        num / den
    };
    Ok(if q < 0.0 { -z } else { z })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values computed independently with mpmath 1.3 at 50
    /// (erf/erfc/Φ) resp. 400 (Φ⁻¹, deep tails) significant digits and
    /// rounded to nearest f64:
    ///
    /// ```python
    /// import mpmath as mp; mp.mp.dps = 400
    /// mp.erf(x); mp.erfc(x); mp.ncdf(x)
    /// mp.sqrt(2) * mp.erfinv(2 * mp.mpf(repr(p)) - 1)   # Φ⁻¹(p)
    /// ```
    ///
    /// Spot values cross-check against Abramowitz & Stegun Table 7.1
    /// (erf) and 26.1/26.2 (Φ) at their printed precision.
    const ERF_REF: &[(f64, f64)] = &[
        (0.0, 0.0),
        (1e-09, 1.1283791670955127e-09),
        (3.725290298461914e-09, 4.203539964167448e-09),
        (0.1, 0.1124629160182849),
        (0.25, 0.27632639016823696),
        (0.5, 0.5204998778130465),
        (0.75, 0.7111556336535151),
        (0.84375, 0.7672256612323416),
        (0.9, 0.7969082124228322),
        (1.0, 0.8427007929497149),
        (1.25, 0.9229001282564583),
        (1.5, 0.9661051464753108),
        (2.0, 0.9953222650189527),
        (2.5, 0.999593047982555),
        (2.857142857142857, 0.9999466876886117),
        (3.0, 0.9999779095030014),
        (4.0, 0.9999999845827421),
        (5.0, 0.9999999999984626),
        (5.9, 0.9999999999999999),
        (6.0, 1.0),
    ];

    const ERFC_REF: &[(f64, f64)] = &[
        (0.0, 1.0),
        (0.1, 0.887537083981715),
        (0.25, 0.7236736098317631),
        (0.3, 0.6713732405408726),
        (0.5, 0.4795001221869535),
        (0.84375, 0.23277433876765838),
        (1.0, 0.15729920705028513),
        (1.25, 0.07709987174354177),
        (2.0, 0.004677734981047266),
        (2.857142857142857, 5.3312311388322795e-05),
        (3.0, 2.209049699858544e-05),
        (4.0, 1.541725790028002e-08),
        (6.0, 2.1519736712498913e-17),
        (8.0, 1.1224297172982926e-29),
        (10.0, 2.088487583762545e-45),
        (15.0, 7.212994172451207e-100),
        (20.0, 5.395865611607901e-176),
        (26.5, 2.2109076642637343e-307),
    ];

    const NORM_CDF_REF: &[(f64, f64)] = &[
        (-37.0, 5.725571222524577e-300),
        (-30.0, 4.906713927148187e-198),
        (-20.0, 2.7536241186062337e-89),
        (-12.0, 1.776482112077679e-33),
        (-8.0, 6.220960574271784e-16),
        (-6.0, 9.86587645037698e-10),
        (-4.0, 3.1671241833119924e-05),
        (-2.5, 0.006209665325776135),
        (-1.0, 0.15865525393145705),
        (-0.5, 0.3085375387259869),
        (0.0, 0.5),
        (0.5, 0.6914624612740131),
        (1.0, 0.8413447460685429),
        (2.5, 0.9937903346742238),
        (4.0, 0.9999683287581669),
        (6.0, 0.9999999990134123),
        (8.0, 0.9999999999999993),
    ];

    /// Φ⁻¹ references. Upper-tail entries (p > 0.5) are Φ⁻¹ evaluated at
    /// the exact f64 value of `1.0 - p`: the [0, 1] parameterization
    /// itself rounds `1 - p` (inherent to the API — SciPy behaves the
    /// same), so that is the correctly-rounded achievable answer.
    const NORM_PPF_REF: &[(f64, f64)] = &[
        (1e-300, -37.0470962993612),
        (1e-250, -33.79958617269484),
        (1e-100, -21.273453560965326),
        (1e-50, -14.933337534788489),
        (1e-20, -9.262340089798407),
        (1e-12, -7.034483825301132),
        (1.3e-11, -6.667611456510741),
        (1.4e-11, -6.656723091518184),
        (1e-10, -6.361340902404057),
        (1e-09, -5.9978070150076865),
        (1e-06, -4.753424308822899),
        (0.001, -3.0902323061678136),
        (0.01, -2.326347874040841),
        (0.025, -1.9599639845400543),
        (0.05, -1.6448536269514726),
        (0.075, -1.439531470938456),
        (0.1, -1.2815515655446004),
        (0.25, -0.6744897501960817),
        (0.425, -0.1891184262727925),
        (0.5, 0.0),
        (0.75, 0.6744897501960817),
        (0.9, 1.2815515655446006),
        (0.925, 1.4395314709384561),
        (0.95, 1.6448536269514722),
        (0.975, 1.9599639845400538),
        (0.99, 2.3263478740408408),
        (0.999, 3.090232306167813),
        (0.999999, 4.753424308817087),
        (0.9999999999, 6.361340889697422),
        (0.999999999986, 6.6567228461403625),
        (0.9999999999999, 7.3487545403000425),
    ];

    fn rel_err(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            got.abs()
        } else {
            ((got - want) / want).abs()
        }
    }

    #[test]
    fn erf_matches_reference_within_1e15_and_is_odd() {
        for &(x, want) in ERF_REF {
            let err = rel_err(erf(x), want);
            assert!(err <= 1e-15, "erf({x}) rel err {err:e}");
            // fdlibm's erf is odd by construction; pin it bit-for-bit.
            assert_eq!(erf(-x).to_bits(), (-erf(x)).to_bits(), "erf odd at {x}");
        }
    }

    #[test]
    fn erfc_matches_reference_within_1e15_both_sides() {
        for &(x, want) in ERFC_REF {
            let err = rel_err(erfc(x), want);
            assert!(err <= 1e-15, "erfc({x}) rel err {err:e}");
            // Negative side via the reflection erfc(-x) = 2 - erfc(x).
            let neg = -x;
            let err = rel_err(erfc(neg), 2.0 - want);
            assert!(err <= 1e-15, "erfc({neg}) rel err {err:e}");
        }
    }

    #[test]
    fn erf_erfc_complement_identity_holds_to_rounding() {
        let mut x = -6.0;
        while x <= 6.0 {
            let sum = erf(x) + erfc(x);
            assert!((sum - 1.0).abs() <= 2.5e-16, "erf + erfc at {x}: {sum:e}");
            x += 0.05;
        }
    }

    #[test]
    fn norm_cdf_matches_reference_incl_deep_tail() {
        for &(x, want) in NORM_CDF_REF {
            let err = rel_err(norm_cdf(x), want);
            // The half-form's only rounding beyond erfc's ~1 ulp is the
            // argument product x·(1/√2), whose relative error grows as
            // ~x²·ε in the tail (d ln Φ/dx ≈ |x| there).
            let bound = (1.5 * x * x * f64::EPSILON).max(2e-15);
            assert!(err <= bound, "norm_cdf({x}) rel err {err:e} > {bound:e}");
        }
    }

    /// The live-run regression class (chelis#902): put-call-parity-style
    /// identities need Φ(x) + Φ(−x) = 1 to rounding, which independent
    /// per-branch approximations of Φ do not satisfy (measured 1.8e-05
    /// on a hand-rolled polynomial). The single erfc path holds it at
    /// ≤ 2^-52 across the whole range including the deep tails.
    #[test]
    fn norm_cdf_parity_identity_holds_to_rounding() {
        let mut x = -37.0;
        while x <= 37.0 {
            let sum = norm_cdf(x) + norm_cdf(-x);
            let neg = -x;
            assert!((sum - 1.0).abs() <= 2.5e-16, "Φ({x}) + Φ({neg}) = {sum:e}");
            x += 0.1;
        }
    }

    #[test]
    fn norm_ppf_matches_reference_incl_extreme_tails() {
        for &(p, want) in NORM_PPF_REF {
            let got = norm_ppf(p).expect("in-domain p");
            let err = if want == 0.0 {
                (got - want).abs()
            } else {
                rel_err(got, want)
            };
            assert!(err <= 5e-15, "norm_ppf({p:e}) rel err {err:e}");
        }
    }

    /// Round-trip Φ(Φ⁻¹(p)) at 1e-12 relative over (1e-10, 1 − 1e-10),
    /// the acceptance bound from chelis#902 (measured headroom ~100×:
    /// the observed max is ~1.2e-14).
    #[test]
    fn norm_ppf_round_trips_norm_cdf_at_1e12() {
        let mut p = 1e-10;
        while p <= 0.5 {
            let z = norm_ppf(p).expect("in-domain p");
            let err = rel_err(norm_cdf(z), p);
            assert!(err <= 1e-12, "round-trip at p = {p:e}: rel err {err:e}");
            // Upper mirror: compare tail mass against tail mass so the
            // comparison stays relative in the quantity that rounds.
            let pu = 1.0 - p;
            let zu = norm_ppf(pu).expect("in-domain p");
            let err = rel_err(1.0 - norm_cdf(zu), 1.0 - pu);
            assert!(err <= 1e-12, "round-trip at p = {pu}: rel err {err:e}");
            p *= 1.7;
        }
    }

    #[test]
    fn special_fn_edge_cases() {
        // NaN propagates through all four.
        assert!(erf(f64::NAN).is_nan());
        assert!(erfc(f64::NAN).is_nan());
        assert!(norm_cdf(f64::NAN).is_nan());
        assert!(
            norm_ppf(f64::NAN)
                .expect("NaN is not a domain error")
                .is_nan()
        );
        // Infinities saturate.
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert_eq!(norm_cdf(f64::INFINITY), 1.0);
        assert_eq!(norm_cdf(f64::NEG_INFINITY), 0.0);
        // Signed zero and tiny arguments (the fdlibm small-x branches).
        assert_eq!(erf(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(erf(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(erfc(0.0), 1.0);
        assert_eq!(norm_cdf(0.0), 0.5);
        assert!(rel_err(erf(1e-300), 1.1283791670955126e-300) <= 1e-15);
        // Quantile endpoints and domain errors.
        assert_eq!(norm_ppf(0.0).expect("edge"), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0).expect("edge"), f64::INFINITY);
        assert_eq!(norm_ppf(0.5).expect("center"), 0.0);
        for bad in [-0.1, 1.5, f64::INFINITY, f64::NEG_INFINITY] {
            let err = norm_ppf(bad).expect_err("out-of-domain p must fail");
            assert!(
                err.contains("norm_ppf domain error"),
                "diagnostic must name the builtin and the domain: {err}"
            );
        }
    }

    /// erfc keeps full relative accuracy into the deep positive tail
    /// (the whole point of having it as a builtin next to erf).
    #[test]
    fn erfc_deep_tail_keeps_relative_accuracy() {
        // mpmath (dps = 50): erfc(26.5) = 2.2109076642637343e-307.
        let got = erfc(26.5);
        assert!(rel_err(got, 2.210_907_664_263_734_3e-307) <= 1e-13);
        assert!(got > 0.0);
        // And underflows to +0 (not a negative or junk value) past ~27.3.
        assert_eq!(erfc(28.0), 0.0);
        assert_eq!(erfc(1e308), 0.0);
    }
}
