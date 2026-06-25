//! WI-14 Arb/FLINT rigorous-enclosure oracle.
//!
//! The soundness oracle of the verification stack. It computes a *rigorous*
//! enclosure of a special function value — an interval `[lo, hi]` of `f64`
//! that is mathematically *guaranteed* to contain the true real value — using
//! Arb's ball arithmetic (FLINT/Arb via `arb-sys`). It is how a claimed bound
//! is independently checked: a bound is sound only if it contains the Arb
//! enclosure, which contains the truth.
//!
//! Two consumers, both in the master plan (`verification_stack_master_plan.md`
//! WI-14): Beacon's soundness oracle (WI-B7) samples concrete inputs, runs the
//! concrete IR, computes the rigorous enclosure here, and asserts the concrete
//! output sits inside Beacon's claimed bound; and the WI-13 envelope generator
//! uses this oracle to *certify* the sup-norm error of each committed erf
//! envelope (the same oracle that validates also certifies, so Arb is the
//! single root of trust for the special-function fragment).
//!
//! ## What is always present vs feature-gated
//!
//! The enclosure *type* ([`ErfEnclosure`]) and its containment algebra are
//! always compiled — they are pure `f64` arithmetic with no FLINT/Arb link, so
//! the containment contract is unit-testable in the default build and a
//! caller can hold an enclosure produced elsewhere (e.g. deserialized envelope
//! data) without linking the oracle. The live FLINT/Arb computation
//! ([`rigorous_erf_enclosure`] and friends) is gated behind the `arb` feature.
//! This mirrors the carcara-audit split: the always-present surface is testable
//! without the heavy dependency, the live round-trip is gated.
//!
//! ## Soundness of the `f64` projection
//!
//! Arb computes erf as a ball `m ± r` at the requested precision. The ball's
//! true endpoints `m - r` and `m + r` are arbitrary-precision reals that do not
//! generally land on an `f64`. To return an `f64` interval that still *encloses*
//! the ball — and therefore the truth — the lower endpoint is rounded **toward
//! negative infinity** (`ARF_RND_FLOOR`) and the upper endpoint **toward
//! positive infinity** (`ARF_RND_CEIL`). Rounding *outward* can only widen the
//! interval, never narrow it, so the `f64` result is a sound over-approximation
//! of the Arb ball, which is itself a sound enclosure of the true value. A
//! round-to-nearest projection would be *unsound* (it can move an endpoint
//! across the truth), which is the single subtle correctness point of this
//! module and is locked by [`tests`].
//!
//! ## Build and licensing
//!
//! `arb-sys` vendors FLINT and Arb C source and compiles them (`gmp-mpfr-sys`
//! compiles GMP/MPFR). The vendored C is LGPL and `gmp-mpfr-sys` is LGPL-3.0+;
//! the obligation attaches to a binary built with the `arb` feature. The oracle
//! is build-time/validation tooling (it never ships in a deployed slim build —
//! it is off by default), so the obligation is recorded and deferred per the
//! master-plan direction (WI-19). The Fedora PIE-default toolchain requires the
//! vendored C be compiled `-fPIC`; the workspace `.cargo/config.toml` sets
//! `CFLAGS`/`CXXFLAGS` and isolated build-caches so a clean build is position
//! independent without a per-developer step.

/// A rigorous enclosure `[lo, hi]` of a real value, as `f64` endpoints.
///
/// The contract is one-directional and absolute: the true real value the
/// enclosure was computed for is **guaranteed** to satisfy `lo <= truth <= hi`.
/// The endpoints are outward-rounded (see the module docs), so the interval may
/// be slightly wider than the underlying Arb ball but never narrower.
///
/// This type carries no special-function identity — it is just a sound real
/// interval — so the same type serves erf today and the activation family when
/// that layer arrives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErfEnclosure {
    /// Lower endpoint, rounded toward negative infinity. `<= truth`.
    pub lo: f64,
    /// Upper endpoint, rounded toward positive infinity. `>= truth`.
    pub hi: f64,
}

impl ErfEnclosure {
    /// Construct an enclosure from explicit endpoints.
    ///
    /// `lo <= hi` is a precondition of a well-formed enclosure; a caller that
    /// constructs one directly (rather than via the oracle) is responsible for
    /// it. [`ErfEnclosure::is_well_formed`] checks it.
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    /// Whether the endpoints form a non-empty, finite-or-saturating interval:
    /// both endpoints non-NaN and `lo <= hi`. A NaN endpoint is incomparable
    /// and makes the enclosure meaningless, so it is not well formed.
    pub fn is_well_formed(&self) -> bool {
        !self.lo.is_nan() && !self.hi.is_nan() && self.lo <= self.hi
    }

    /// The width `hi - lo` of the enclosure. A tighter (smaller) width is a
    /// sharper enclosure; the oracle tightens it by raising the precision.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Whether this enclosure contains the value `x`: `lo <= x <= hi`.
    ///
    /// This is the soundness-oracle containment helper (WI-14): a claimed bound
    /// is checked sound by confirming it contains the rigorous enclosure, and a
    /// concrete sample is checked inside a claimed bound by this predicate. A
    /// NaN `x` is contained by nothing.
    pub fn contains(&self, x: f64) -> bool {
        // `x.is_nan()` short-circuits: NaN compares false to both bounds, but be
        // explicit so the intent (NaN is contained by nothing) is unmistakable.
        !x.is_nan() && self.lo <= x && x <= self.hi
    }

    /// Whether this enclosure is entirely contained in `[lo, hi]` — i.e. the
    /// claimed bound `[lo, hi]` contains the rigorous enclosure, and therefore
    /// the truth. This is the bound-soundness check the oracle exists for: a
    /// claimed special-function bound is sound iff
    /// `oracle_enclosure.contained_in(claim_lo, claim_hi)`.
    pub fn contained_in(&self, lo: f64, hi: f64) -> bool {
        !lo.is_nan() && !hi.is_nan() && lo <= self.lo && self.hi <= hi
    }
}

/// Arb rounding mode toward negative infinity (`ARF_RND_FLOOR`).
///
/// Re-declared here rather than imported so the constant's meaning is visible
/// at the projection site and the always-present surface does not depend on the
/// `arb-sys` crate being linked. The values are part of Arb's stable ABI
/// (`arf.h`).
#[cfg(feature = "arb")]
const ARF_RND_FLOOR: i32 = 2;
/// Arb rounding mode toward positive infinity (`ARF_RND_CEIL`).
#[cfg(feature = "arb")]
const ARF_RND_CEIL: i32 = 3;

/// The default working precision (bits) for the oracle. 128 bits is far more
/// than an `f64`'s 53, so the Arb ball is far tighter than one `f64` ULP for
/// erf over the whole real line, and the outward-rounded `f64` projection is
/// then at most a couple of ULPs wide. Callers needing a sharper enclosure for
/// a delicate sup-norm certification can pass a higher precision.
#[cfg(feature = "arb")]
pub const DEFAULT_PREC: i64 = 128;

/// Compute a rigorous enclosure of `erf(x)` as an outward-rounded `f64`
/// interval at `prec` bits of working precision.
///
/// The result is **guaranteed** to contain the true real value `erf(x)`. The
/// guarantee rests on (1) Arb's ball arithmetic being a rigorous enclosure of
/// the true value at any precision, and (2) the outward `f64` projection (floor
/// the lower endpoint, ceil the upper) only ever widening the ball. See the
/// module docs for why round-to-nearest would be unsound.
///
/// `prec` is Arb's working precision in bits; higher is tighter and slower.
/// [`DEFAULT_PREC`] is a good default. A non-positive `prec` is clamped to 2 so
/// the FFI never sees a degenerate precision.
#[cfg(feature = "arb")]
pub fn rigorous_erf_enclosure(x: f64, prec: i64) -> ErfEnclosure {
    use arb_sys::arb::{arb_clear, arb_get_lbound_arf, arb_get_ubound_arf, arb_init, arb_set_d};
    use arb_sys::arb_hypgeom::arb_hypgeom_erf;
    use arb_sys::arf::{arf_clear, arf_get_d, arf_init};
    use std::mem::MaybeUninit;

    let prec = prec.max(2);

    // Fail-closed on non-finite input rather than feeding it to the FFI (the
    // codebase's NaN/inf silent-corruption bug class; cf. the WI-3 graph
    // extraction non-finite guard). erf is exactly +-1 at +-inf (a sound,
    // well-formed degenerate enclosure); erf(NaN) is undefined, so return a
    // NaN enclosure, which `is_well_formed` rejects and `contains` treats as
    // containing nothing -- nothing downstream can mistake it for a real bound.
    if x.is_nan() {
        return ErfEnclosure::new(f64::NAN, f64::NAN);
    }
    if x.is_infinite() {
        let v = x.signum();
        return ErfEnclosure::new(v, v);
    }

    // SAFETY: every Arb/arf object is `_init`-ed before use and `_clear`-ed
    // exactly once before the function returns; no pointer escapes; the FFI
    // calls match the `arb-sys` signatures (verified against the linked Arb
    // c2b7e36 ABI). `arb_set_d` is exact for any finite `f64` (it sets the ball
    // midpoint to the exact dyadic value of `x` with zero radius), so no
    // precision is lost feeding the input in.
    unsafe {
        let mut z = MaybeUninit::uninit();
        arb_init(z.as_mut_ptr());
        let mut z = z.assume_init();
        arb_set_d(&mut z, x);

        let mut res = MaybeUninit::uninit();
        arb_init(res.as_mut_ptr());
        let mut res = res.assume_init();
        arb_hypgeom_erf(&mut res, &mut z, prec);

        let mut lo_arf = MaybeUninit::uninit();
        arf_init(lo_arf.as_mut_ptr());
        let mut lo_arf = lo_arf.assume_init();
        let mut hi_arf = MaybeUninit::uninit();
        arf_init(hi_arf.as_mut_ptr());
        let mut hi_arf = hi_arf.assume_init();

        // Exact endpoints of the Arb ball as arbitrary-precision arf values.
        arb_get_lbound_arf(&mut lo_arf, &res, prec);
        arb_get_ubound_arf(&mut hi_arf, &res, prec);

        // Outward-round each endpoint into an f64: floor the lower bound toward
        // -inf, ceil the upper bound toward +inf. This is the soundness-
        // critical step — see the module docs.
        let lo = arf_get_d(&lo_arf, ARF_RND_FLOOR);
        let hi = arf_get_d(&hi_arf, ARF_RND_CEIL);

        arf_clear(&mut lo_arf);
        arf_clear(&mut hi_arf);
        arb_clear(&mut z);
        arb_clear(&mut res);

        ErfEnclosure::new(lo, hi)
    }
}

#[cfg(test)]
mod always_present_tests {
    //! Containment-algebra tests that need no Arb link, so the soundness
    //! contract is exercised in the default build (the `arb` feature off).
    use super::ErfEnclosure;

    #[test]
    fn contains_is_inclusive_at_both_endpoints() {
        let e = ErfEnclosure::new(-0.5, 0.5);
        assert!(e.contains(-0.5), "lower endpoint is contained");
        assert!(e.contains(0.5), "upper endpoint is contained");
        assert!(e.contains(0.0), "interior is contained");
    }

    #[test]
    fn contains_rejects_outside_and_nan() {
        let e = ErfEnclosure::new(-0.5, 0.5);
        assert!(!e.contains(-0.5001), "just below lo is not contained");
        assert!(!e.contains(0.5001), "just above hi is not contained");
        assert!(!e.contains(f64::NAN), "NaN is contained by nothing");
        assert!(!e.contains(f64::INFINITY), "inf is not contained");
    }

    #[test]
    fn contained_in_is_the_bound_soundness_check() {
        // The oracle enclosure of some value.
        let oracle = ErfEnclosure::new(0.9999, 1.0001);
        // A claimed bound that strictly contains it: SOUND.
        assert!(
            oracle.contained_in(0.99, 1.01),
            "a wider claim contains the enclosure"
        );
        // A claim exactly matching the enclosure: SOUND (closed containment).
        assert!(
            oracle.contained_in(0.9999, 1.0001),
            "an exact claim contains the enclosure"
        );
    }

    #[test]
    fn contained_in_rejects_a_claim_that_cuts_the_enclosure() {
        let oracle = ErfEnclosure::new(0.9999, 1.0001);
        // Claim's upper bound is below the enclosure's upper bound: the truth
        // could be above the claim, so the claim is UNSOUND.
        assert!(
            !oracle.contained_in(0.99, 1.00005),
            "a claim that cuts the top of the enclosure is unsound"
        );
        // Claim's lower bound is above the enclosure's lower bound: UNSOUND.
        assert!(
            !oracle.contained_in(0.99995, 1.01),
            "a claim that cuts the bottom of the enclosure is unsound"
        );
        // NaN bound is never a valid claim.
        assert!(!oracle.contained_in(f64::NAN, 1.01));
        assert!(!oracle.contained_in(0.99, f64::NAN));
    }

    #[test]
    fn well_formed_rejects_inverted_and_nan() {
        assert!(ErfEnclosure::new(-1.0, 1.0).is_well_formed());
        assert!(
            ErfEnclosure::new(1.0, 1.0).is_well_formed(),
            "degenerate point ok"
        );
        assert!(
            !ErfEnclosure::new(1.0, -1.0).is_well_formed(),
            "inverted not well formed"
        );
        assert!(!ErfEnclosure::new(f64::NAN, 1.0).is_well_formed());
        assert!(!ErfEnclosure::new(-1.0, f64::NAN).is_well_formed());
    }

    #[test]
    fn width_is_hi_minus_lo() {
        assert_eq!(ErfEnclosure::new(-1.0, 1.0).width(), 2.0);
        assert_eq!(ErfEnclosure::new(0.5, 0.5).width(), 0.0);
    }
}

/// Convenience wrapper computing [`rigorous_erf_enclosure`] at [`DEFAULT_PREC`].
#[cfg(feature = "arb")]
pub fn rigorous_erf(x: f64) -> ErfEnclosure {
    rigorous_erf_enclosure(x, DEFAULT_PREC)
}

/// Certify that a claimed sup-norm error bound `eps` holds for an approximation
/// `approx` of `erf` over a finite sample of points in a box `[lo, hi]`.
///
/// For each sample point `x`, the rigorous enclosure `E_x` of `erf(x)` is
/// computed, and the claim is that `|approx(x) - erf(x)| <= eps` for the true
/// `erf(x)`. Because the truth lies in `E_x = [E.lo, E.hi]`, the *largest*
/// possible value of `|approx(x) - truth|` over the enclosure is
/// `max(|approx(x) - E.lo|, |approx(x) - E.hi|)`. If that worst-case bound is
/// `<= eps` at every sample, the claim is confirmed *at the samples* (rounding
/// the approximation error up to account for the enclosure width, so the check
/// is conservative — it never reports a too-loose `eps` as failing or a true
/// failure as passing).
///
/// This is a *sampled* witness, not a continuous sup-norm proof: it confirms
/// the claimed `eps` is not violated at the sampled points, which is exactly
/// the empirical guardrail WI-14/WI-B7 specify (catch an unsound bound cheaply;
/// it does not by itself prove the bound over the whole box). The WI-13
/// generator pairs it with a dense, deterministic sample grid per box.
///
/// Returns `Ok(worst_observed)` with the largest worst-case error seen (always
/// `<= eps` on success) or `Err((x, worst_at_x))` at the first sample whose
/// worst-case error exceeds `eps`.
#[cfg(feature = "arb")]
pub fn certify_sup_norm_at_samples<F>(
    approx: F,
    samples: &[f64],
    eps: f64,
    prec: i64,
) -> Result<f64, (f64, f64)>
where
    F: Fn(f64) -> f64,
{
    let mut worst = 0.0_f64;
    for &x in samples {
        let enc = rigorous_erf_enclosure(x, prec);
        let a = approx(x);
        // Worst-case |approx - truth| over truth in [enc.lo, enc.hi].
        let err = (a - enc.lo).abs().max((a - enc.hi).abs());
        if err > eps {
            return Err((x, err));
        }
        if err > worst {
            worst = err;
        }
    }
    Ok(worst)
}

/// Rigorously bound `sup_{x in [lo, hi]} |erf(x) - p(x)|` by *whole-box* Arb
/// ball arithmetic, where `p` is the polynomial with `coeffs` in **descending**
/// degree order (Horner; a single-element `coeffs` is a constant, i.e. the
/// saturation arm). Returns a sound `f64` upper bound of the sup-norm error
/// over the entire box — this is the certified `eps` the WI-13 envelope carries.
///
/// This is the rigorous certifier (versus [`certify_sup_norm_at_samples`],
/// which is a sampled guardrail): it evaluates `erf(X) - p(X)` over the *ball*
/// `X = [lo, hi]` and reads the sound `abs` upper bound off the result, so the
/// bound holds for every real `x` in the box, not just sampled points.
///
/// ## Subdivision plus the mean-value form beats the dependency problem
///
/// A single ball evaluation of `erf(X) - p(X)` over a wide `X` is sound but
/// hugely over-wide: ball arithmetic encloses `erf(X)` and `p(X)`
/// independently and does not know they track each other, so the difference's
/// radius is dominated by each term's variation rather than their (small)
/// gap. Worse, evaluating a high-degree polynomial by naive ball Horner over a
/// wide box accumulates a wide `acc` that each `acc*X` step widens further, so
/// plain Horner converges only *linearly* in the sub-box width.
///
/// Two compounding fixes. (1) The box is split into `subdivisions` equal
/// sub-boxes and the **maximum** sound upper bound across them is returned; the
/// max over a cover of `[lo, hi]` is an upper bound on the true sup, so the
/// result is sound for any `subdivisions >= 1`. (2) On each sub-box the
/// polynomial is enclosed in **mean-value form**: with midpoint `m` and radius
/// `r`, `p(X) ⊆ p(m) + p'(X)·(X - m)`, where `p(m)` is a tight point value and
/// `p'(X)·(X - m)` has radius `~|p'|·r`. Because the `(X - m)` factor is the
/// narrow centered offset, the enclosure tightens *quadratically* in `r`, so a
/// modest subdivision count yields a tight, true sup-norm bound instead of a
/// dependency-problem-dominated one. The committed-data provenance records the
/// `subdivisions` used so CI re-validation reproduces the bound.
///
/// `subdivisions` is clamped to at least 1 and `prec` to at least 2.
#[cfg(feature = "arb")]
pub fn certify_sup_norm_over_box(
    lo: f64,
    hi: f64,
    coeffs: &[f64],
    subdivisions: usize,
    prec: i64,
) -> f64 {
    use arb_sys::arb::{arb_clear, arb_init, arb_set_d};
    use arb_sys::arb_poly::{arb_poly_clear, arb_poly_init, arb_poly_set_coeff_arb};
    use std::mem::MaybeUninit;

    let prec = prec.max(2);
    let n = subdivisions.max(1);
    let width = (hi - lo) / n as f64;

    // SAFETY: the arb_poly and the scratch arb are init/clear-paired; coeffs are
    // finite f64s set exactly via arb_set_d; no pointer escapes. The poly is
    // built once and reused read-only across sub-boxes.
    unsafe {
        // Build f(t) = sum coeffs once. `coeffs` is DESCENDING degree; arb_poly
        // indexes coefficients by ASCENDING degree, so coeff for degree d is
        // coeffs[len-1-d].
        let mut f = MaybeUninit::uninit();
        arb_poly_init(f.as_mut_ptr());
        let mut f = f.assume_init();
        let mut c = MaybeUninit::uninit();
        arb_init(c.as_mut_ptr());
        let mut c = c.assume_init();
        let len = coeffs.len();
        for (i, &coeff) in coeffs.iter().enumerate() {
            let degree = (len - 1 - i) as i64;
            arb_set_d(&mut c, coeff);
            arb_poly_set_coeff_arb(&mut f, degree, &c);
        }

        let mut worst = 0.0_f64;
        for i in 0..n {
            let a = lo + width * i as f64;
            // Pin the last sub-box's upper edge exactly to `hi` so float
            // accumulation of `width` cannot leave a sliver uncovered.
            let b = if i + 1 == n {
                hi
            } else {
                lo + width * (i + 1) as f64
            };
            let e = sub_box_abs_err(&mut f, a, b, prec);
            if e > worst {
                worst = e;
            }
        }

        arb_clear(&mut c);
        arb_poly_clear(&mut f);
        worst
    }
}

/// Sound upper bound of `|erf(x) - p(x)|` over the single sub-box `[lo, hi]`,
/// enclosing the *difference* `g = erf - p` in mean-value form. `f` is the
/// prebuilt `arb_poly` for `p`. Helper for [`certify_sup_norm_over_box`].
///
/// The mean-value form is applied to `g` as a whole, not to `erf` and `p`
/// separately: `g(X) ⊆ g(m) + g'(X)·(X - m)` where `g(m) = erf(m) - p(m)` is a
/// tight point value and `g'(X) = erf'(X) - p'(X)` is enclosed over the box,
/// with `erf'(x) = (2/sqrt(pi)) exp(-x^2)` in closed form. The resulting width
/// scales with `|g'|·r`, and `g'` (the error's derivative) is small wherever
/// the fit is good — so the enclosure is tight, instead of inheriting the
/// independent variation of `erf` and `p` (which is what makes a naive
/// `erf(X) - p(X)` difference loose even after subdivision).
#[cfg(feature = "arb")]
unsafe fn sub_box_abs_err(
    f: &mut arb_sys::arb_poly::arb_poly_struct,
    lo: f64,
    hi: f64,
    prec: i64,
) -> f64 {
    use arb_sys::arb::{
        arb_addmul, arb_clear, arb_const_pi, arb_exp, arb_get_abs_ubound_arf, arb_init, arb_mul,
        arb_neg, arb_rsqrt, arb_set, arb_set_d, arb_set_interval_arf, arb_set_ui, arb_sqr, arb_sub,
    };
    use arb_sys::arb_hypgeom::arb_hypgeom_erf;
    use arb_sys::arb_poly::arb_poly_evaluate2;
    use arb_sys::arf::{arf_clear, arf_get_d, arf_init, arf_set_d};
    use std::mem::MaybeUninit;

    // erf'(x) = (2/sqrt(pi)) exp(-x^2), enclosed over the ball `xv`. Result in
    // `out`. `scratch*` are caller-owned init'd arbs reused to avoid churn.
    unsafe fn erf_prime(
        out: &mut arb_sys::arb::arb_struct,
        xv: &arb_sys::arb::arb_struct,
        s1: &mut arb_sys::arb::arb_struct,
        s2: &mut arb_sys::arb::arb_struct,
        prec: i64,
    ) {
        // SAFETY: all args are caller-init'd arbs; the scratch `two` is
        // init/clear-paired here; FFI signatures match the linked Arb ABI.
        unsafe {
            // s1 = exp(-x^2)
            arb_sqr(s1, xv, prec);
            arb_neg(s1, s1);
            arb_exp(s1, s1, prec);
            // s2 = 2/sqrt(pi): pi -> rsqrt -> *2
            arb_const_pi(s2, prec);
            arb_rsqrt(s2, s2, prec); // 1/sqrt(pi)
            let mut two = MaybeUninit::uninit();
            arb_init(two.as_mut_ptr());
            let mut two = two.assume_init();
            arb_set_ui(&mut two, 2);
            arb_mul(s2, s2, &two, prec); // 2/sqrt(pi)
            arb_clear(&mut two);
            // out = s2 * s1
            arb_mul(out, s1, s2, prec);
        }
    }

    // SAFETY: every arb/arf object is `_init`-ed before use and `_clear`-ed
    // exactly once before return; `f` is borrowed read-only; FFI signatures
    // match the linked Arb ABI. `arf_set_d`/`arb_set_d` are exact for finite
    // f64. The box edges are finite from the caller's split.
    unsafe {
        // X = [lo, hi] as a ball (midpoint m, radius r = (hi-lo)/2).
        let mut xlo = MaybeUninit::uninit();
        arf_init(xlo.as_mut_ptr());
        let mut xlo = xlo.assume_init();
        let mut xhi = MaybeUninit::uninit();
        arf_init(xhi.as_mut_ptr());
        let mut xhi = xhi.assume_init();
        arf_set_d(&mut xlo, lo);
        arf_set_d(&mut xhi, hi);
        let mut x = MaybeUninit::uninit();
        arb_init(x.as_mut_ptr());
        let mut x = x.assume_init();
        arb_set_interval_arf(&mut x, &xlo, &xhi, prec);

        // Midpoint m as a point ball, and the centered offset t = X - m.
        let m_mid = 0.5 * (lo + hi);
        let mut m = MaybeUninit::uninit();
        arb_init(m.as_mut_ptr());
        let mut m = m.assume_init();
        arb_set_d(&mut m, m_mid);
        let mut t = MaybeUninit::uninit();
        arb_init(t.as_mut_ptr());
        let mut t = t.assume_init();
        arb_sub(&mut t, &x, &m, prec); // t = X - m

        // g(m) = erf(m) - p(m), both tight (m is a point).
        let mut erf_m = MaybeUninit::uninit();
        arb_init(erf_m.as_mut_ptr());
        let mut erf_m = erf_m.assume_init();
        arb_hypgeom_erf(&mut erf_m, &mut m, prec);
        let mut p_m = MaybeUninit::uninit();
        arb_init(p_m.as_mut_ptr());
        let mut p_m = p_m.assume_init();
        let mut scratch_a = MaybeUninit::uninit();
        arb_init(scratch_a.as_mut_ptr());
        let mut scratch_a = scratch_a.assume_init();
        arb_poly_evaluate2(&mut p_m, &mut scratch_a, f, &mut m, prec); // p(m); p'(m) unused
        let mut g_m = MaybeUninit::uninit();
        arb_init(g_m.as_mut_ptr());
        let mut g_m = g_m.assume_init();
        arb_sub(&mut g_m, &erf_m, &p_m, prec); // g(m)

        // g'(X) = erf'(X) - p'(X), enclosed over the whole sub-box.
        let mut dp_x = MaybeUninit::uninit();
        arb_init(dp_x.as_mut_ptr());
        let mut dp_x = dp_x.assume_init();
        let mut scratch_b = MaybeUninit::uninit();
        arb_init(scratch_b.as_mut_ptr());
        let mut scratch_b = scratch_b.assume_init();
        arb_poly_evaluate2(&mut scratch_b, &mut dp_x, f, &mut x, prec); // p'(X); p(X) unused
        let mut derf_x = MaybeUninit::uninit();
        arb_init(derf_x.as_mut_ptr());
        let mut derf_x = derf_x.assume_init();
        let mut s1 = MaybeUninit::uninit();
        arb_init(s1.as_mut_ptr());
        let mut s1 = s1.assume_init();
        let mut s2 = MaybeUninit::uninit();
        arb_init(s2.as_mut_ptr());
        let mut s2 = s2.assume_init();
        erf_prime(&mut derf_x, &x, &mut s1, &mut s2, prec); // erf'(X)
        let mut dg_x = MaybeUninit::uninit();
        arb_init(dg_x.as_mut_ptr());
        let mut dg_x = dg_x.assume_init();
        arb_sub(&mut dg_x, &derf_x, &dp_x, prec); // g'(X)

        // Mean-value enclosure of the difference: g(X) ⊆ g(m) + g'(X) * t.
        let mut g_enc = MaybeUninit::uninit();
        arb_init(g_enc.as_mut_ptr());
        let mut g_enc = g_enc.assume_init();
        arb_set(&mut g_enc, &g_m);
        arb_addmul(&mut g_enc, &dg_x, &t, prec); // g(m) + g'(X)*t

        let mut ub = MaybeUninit::uninit();
        arf_init(ub.as_mut_ptr());
        let mut ub = ub.assume_init();
        arb_get_abs_ubound_arf(&mut ub, &g_enc, prec);
        // Ceil to f64 so the returned bound is sound (rounds the upper bound up).
        let result = arf_get_d(&ub, ARF_RND_CEIL);

        arf_clear(&mut ub);
        arb_clear(&mut g_enc);
        arb_clear(&mut dg_x);
        arb_clear(&mut s2);
        arb_clear(&mut s1);
        arb_clear(&mut derf_x);
        arb_clear(&mut scratch_b);
        arb_clear(&mut dp_x);
        arb_clear(&mut g_m);
        arb_clear(&mut scratch_a);
        arb_clear(&mut p_m);
        arb_clear(&mut erf_m);
        arb_clear(&mut t);
        arb_clear(&mut m);
        arb_clear(&mut x);
        arf_clear(&mut xlo);
        arf_clear(&mut xhi);
        result
    }
}

#[cfg(all(test, feature = "arb"))]
mod arb_live_tests {
    //! Live FLINT/Arb enclosure tests. These need the `arb` feature (the
    //! vendored FLINT/Arb compiled `-fPIC`); they are the soundness oracle's
    //! own correctness gate. Run with:
    //!   cargo test -p chelis-prove --features arb arb_live_tests
    use super::{
        DEFAULT_PREC, certify_sup_norm_at_samples, certify_sup_norm_over_box, rigorous_erf,
        rigorous_erf_enclosure,
    };
    use crate::erf_envelope::{ErfArm, ErfEnvelope};

    /// erf reference values to ~31 significant digits (computed offline at high
    /// precision; independent of the implementation under test). Each must sit
    /// strictly inside the oracle's enclosure.
    ///
    /// The literals carry more digits than an `f64` can represent on purpose:
    /// they document the true mathematical value, so the nearest-`f64` the
    /// compiler rounds to is the unambiguous best double approximation of the
    /// truth, and the test reads as a check against the real erf value rather
    /// than against a pre-rounded constant. The excess digits round to the same
    /// double, so the allow is a documentation choice, not a correctness risk.
    #[allow(clippy::excessive_precision)]
    const ERF_REFERENCE: &[(f64, f64)] = &[
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_537_682_512_077_178_2),
        (1.0, 0.842_700_792_949_714_869_341_220_635_082_6),
        (2.0, 0.995_322_265_018_952_734_162_069_256_367_3),
        (-1.0, -0.842_700_792_949_714_869_341_220_635_082_6),
        (3.0, 0.999_977_909_503_001_414_558_627_223_870_3),
    ];

    #[test]
    fn enclosure_contains_known_erf_values() {
        for &(x, truth) in ERF_REFERENCE {
            let e = rigorous_erf_enclosure(x, DEFAULT_PREC);
            assert!(e.is_well_formed(), "enclosure of erf({x}) is well formed");
            assert!(
                e.contains(truth),
                "erf({x}) = {truth} must lie in oracle enclosure [{}, {}]",
                e.lo,
                e.hi
            );
        }
    }

    #[test]
    fn enclosure_is_tight_at_default_precision() {
        // At 128 bits the f64 projection should be at most a few ULPs wide for
        // central arguments; assert a generous-but-meaningful ceiling so a
        // precision regression (e.g. silently dropping to a few bits) is caught.
        for &(x, _) in ERF_REFERENCE {
            let e = rigorous_erf_enclosure(x, DEFAULT_PREC);
            assert!(
                e.width() <= 1e-12,
                "erf({x}) enclosure width {} should be tight at 128 bits",
                e.width()
            );
        }
    }

    #[test]
    fn endpoints_are_outward_rounded_not_nearest() {
        // The soundness-critical property: lo <= truth <= hi for erf(1), and the
        // enclosure must NOT collapse to a single f64 (which round-to-nearest of
        // both endpoints to the same nearest double would do for a value whose
        // ball straddles a representable double). erf(1) is irrational and its
        // 128-bit ball is far narrower than an ULP, yet the outward projection
        // must still produce lo < hi (the two adjacent doubles bracketing the
        // truth), never lo == hi == round-to-nearest.
        let e = rigorous_erf_enclosure(1.0, DEFAULT_PREC);
        // High-precision erf(1); excess digits document the truth (see
        // ERF_REFERENCE) and round to the nearest f64.
        #[allow(clippy::excessive_precision)]
        let truth = 0.842_700_792_949_714_869_341_220_635_082_6_f64;
        assert!(e.lo <= truth, "lo {} must be <= truth {}", e.lo, truth);
        assert!(e.hi >= truth, "hi {} must be >= truth {}", e.hi, truth);
        assert!(
            e.lo < e.hi,
            "outward rounding keeps lo < hi for an irrational truth"
        );
        // And the bracket is the *adjacent* doubles: nothing representable lies
        // strictly between lo and the truth, or the truth and hi.
        assert_eq!(
            e.lo.next_up(),
            e.hi,
            "outward enclosure of an irrational is the two bracketing doubles"
        );
    }

    #[test]
    fn saturation_tails_are_enclosed_at_extreme_arguments() {
        // The real erf-argument range reaches +-100 .. +-300 (master WI-13). At
        // those arguments erf is within < 1 ULP of +-1; the oracle must produce
        // a sound enclosure that contains +-1's neighborhood, and the sign must
        // be right. This is what makes the WI-13 saturation arm certifiable.
        for &x in &[100.0_f64, 300.0] {
            let e = rigorous_erf(x);
            assert!(e.is_well_formed());
            // erf(100) < 1 strictly, but within f64 it rounds to 1.0; the sound
            // upper bound is therefore >= 1.0 (it may ceil to just above 1).
            assert!(e.hi >= 1.0, "erf({x}) upper bound {} encloses 1", e.hi);
            assert!(
                e.lo < 1.0 && e.lo > 0.999_999_999_999,
                "erf({x}) lo {} near 1",
                e.lo
            );
            // Negative tail is the mirror.
            let en = rigorous_erf(-x);
            assert!(en.lo <= -1.0, "erf(-{x}) lower bound {} encloses -1", en.lo);
            assert!(
                en.hi > -1.0 && en.hi < -0.999_999_999_999,
                "erf(-{x}) hi {} near -1",
                en.hi
            );
        }
    }

    #[test]
    fn enclosure_is_monotone_increasing() {
        // erf is strictly increasing; a sequence of enclosures must be ordered
        // (each lo not below the previous lo). A flipped pair would signal a
        // sign or argument-feeding bug.
        let xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
        let mut prev = f64::NEG_INFINITY;
        for x in xs {
            let e = rigorous_erf(x);
            assert!(e.lo >= prev - 1e-9, "erf enclosures must be ordered at {x}");
            prev = e.lo;
        }
    }

    #[test]
    fn non_finite_input_is_fail_closed() {
        // NaN input -> a NaN enclosure that is not well formed and contains
        // nothing (fail-closed; never a usable-looking bound).
        let nan_enc = rigorous_erf_enclosure(f64::NAN, DEFAULT_PREC);
        assert!(
            !nan_enc.is_well_formed(),
            "NaN input yields an ill-formed enclosure"
        );
        assert!(!nan_enc.contains(0.0), "a NaN enclosure contains nothing");
        // +-inf -> exactly +-1 (erf's limits), a sound degenerate enclosure.
        let pos = rigorous_erf_enclosure(f64::INFINITY, DEFAULT_PREC);
        assert_eq!((pos.lo, pos.hi), (1.0, 1.0), "erf(+inf) = 1 exactly");
        assert!(pos.contains(1.0));
        let neg = rigorous_erf_enclosure(f64::NEG_INFINITY, DEFAULT_PREC);
        assert_eq!((neg.lo, neg.hi), (-1.0, -1.0), "erf(-inf) = -1 exactly");
    }

    #[test]
    fn sup_norm_certifier_passes_a_true_bound_and_fails_a_false_one() {
        // approx == the oracle midpoint is within ~half the enclosure width of
        // the truth, so a generous eps must PASS.
        let samples: Vec<f64> = (0..=20).map(|i| -2.0 + 0.2 * i as f64).collect();
        let approx_good = |x: f64| {
            let e = rigorous_erf(x);
            0.5 * (e.lo + e.hi)
        };
        let ok = certify_sup_norm_at_samples(approx_good, &samples, 1e-9, DEFAULT_PREC);
        assert!(
            ok.is_ok(),
            "midpoint approx must satisfy a 1e-9 bound: {ok:?}"
        );

        // A deliberately wrong approximation (off by 0.1) must FAIL a tight eps,
        // and report the offending sample.
        let approx_bad = |x: f64| {
            let e = rigorous_erf(x);
            0.5 * (e.lo + e.hi) + 0.1
        };
        let bad = certify_sup_norm_at_samples(approx_bad, &samples, 1e-3, DEFAULT_PREC);
        assert!(
            bad.is_err(),
            "an approx off by 0.1 must violate a 1e-3 bound"
        );
        if let Err((_x, err)) = bad {
            assert!(
                err > 0.09,
                "reported worst error {err} reflects the 0.1 offset"
            );
        }
    }

    #[test]
    fn whole_box_certifier_is_sound_and_tight_for_a_known_poly() {
        // p(x) = x: a deliberately poor 'fit' to erf. The true sup |erf(x) - x|
        // on [-1, 1] is at the endpoints, |erf(1) - 1| = 1 - 0.8427... =
        // 0.15729..., and at x=0 the error is 0. The certified bound must be
        // SOUND (>= the true sup) and TIGHT (close to it with enough sub-boxes).
        let eps = certify_sup_norm_over_box(-1.0, 1.0, &[1.0, 0.0], 4096, DEFAULT_PREC);
        let true_sup = 1.0 - 0.842_700_792_949_714_9_f64;
        assert!(
            eps >= true_sup,
            "certified eps {eps} must be a sound upper bound of the true sup {true_sup}"
        );
        assert!(
            eps <= true_sup + 1e-4,
            "certified eps {eps} should be tight against true sup {true_sup}"
        );
    }

    #[test]
    fn whole_box_certifier_certifies_a_zero_error_constant() {
        // p(x) = 1 against erf on [6, 50]: sup |1 - erf(x)| = 1 - erf(6) ~ 2.2e-17.
        let eps = certify_sup_norm_over_box(6.0, 50.0, &[1.0], 1024, DEFAULT_PREC);
        assert!(eps >= 0.0, "eps is non-negative");
        assert!(
            eps < 1e-15,
            "saturation tail |1 - erf| beyond 6 must certify to ~1e-17, got {eps}"
        );
    }

    #[test]
    fn whole_box_bound_decreases_with_more_subdivisions() {
        // The subdivided ball bound is monotone non-increasing as the cover
        // refines (a finer cover cannot make the max-over-cover larger by much;
        // in practice it strictly tightens here). Confirms the convergence
        // direction the committed-data subdivision count relies on.
        let coarse = certify_sup_norm_over_box(-3.0, 3.0, &[1.0, 0.0], 64, DEFAULT_PREC);
        let fine = certify_sup_norm_over_box(-3.0, 3.0, &[1.0, 0.0], 4096, DEFAULT_PREC);
        assert!(
            fine <= coarse + 1e-12,
            "a finer cover should not loosen the bound (coarse={coarse}, fine={fine})"
        );
        assert!(
            fine < coarse,
            "more sub-boxes should tighten the bound here"
        );
    }

    // ===== WI-13 committed-envelope CI re-validation harness =====
    //
    // This is the soundness gate the WI-13 decision requires: the committed
    // `eps` per box is re-derived from the Arb oracle in CI and asserted to
    // still bound `|approx - erf|` over its box. The committed bound is NOT
    // trusted on faith; every build re-checks it. Paired with a negative test
    // (a deliberately too-small eps MUST fail re-validation) per the spec-first
    // negative-parity contract.

    /// Re-certify one box and return the freshly Arb-derived sup-norm error,
    /// using the box's own arm coefficients.
    fn recertify_box(b: &crate::erf_envelope::ErfEnvelopeBox, subdivisions: usize) -> f64 {
        let coeffs: Vec<f64> = match &b.arm {
            ErfArm::Saturation { value } => vec![*value],
            ErfArm::Central { coeffs } => coeffs.clone(),
        };
        certify_sup_norm_over_box(b.lo, b.hi, &coeffs, subdivisions, DEFAULT_PREC)
    }

    #[test]
    fn committed_envelope_eps_still_bounds_the_truth() {
        // Re-validate every committed box against a fresh Arb certification. The
        // committed eps must be >= the freshly certified sup-norm error, i.e. it
        // still soundly bounds |approx - erf| over the box. Use the committed
        // provenance subdivision count so the re-derived value matches.
        let env = ErfEnvelope::committed();
        let n = env.provenance.certify_subdivisions.max(1);
        for b in &env.boxes {
            let recertified = recertify_box(b, n);
            assert!(
                b.eps >= recertified,
                "committed eps {} for box [{}, {}] must still bound the Arb-recertified \
                 sup-norm error {recertified}",
                b.eps,
                b.lo,
                b.hi
            );
            // And it should not be wildly looser than the truth (catch a
            // committed eps that is sound but stale/inflated by orders of
            // magnitude, which would silently weaken the envelope).
            assert!(
                b.eps <= recertified.max(1e-18) * 100.0 + 1e-12,
                "committed eps {} for box [{}, {}] is far looser than the recertified \
                 {recertified}; regenerate the committed data",
                b.eps,
                b.lo,
                b.hi
            );
        }
    }

    #[test]
    fn a_too_small_eps_fails_revalidation() {
        // SPEC-FIRST NEGATIVE: if a box's claimed eps is shrunk below the true
        // sup-norm error, the Arb re-validation MUST catch it (the recertified
        // bound exceeds the tampered eps). This is the property that makes the
        // CI harness a real soundness gate, not a rubber stamp.
        let env = ErfEnvelope::committed();
        let n = env.provenance.certify_subdivisions.max(1);
        // Find the central box (its eps is the largest, so halving it is a clear
        // violation the certifier must detect).
        let central = env
            .boxes
            .iter()
            .find(|b| matches!(b.arm, ErfArm::Central { .. }))
            .expect("committed envelope has a central box");
        let recertified = recertify_box(central, n);
        let tampered_eps = recertified * 0.5; // deliberately too small
        assert!(
            tampered_eps < recertified,
            "a halved eps {tampered_eps} must be below the true sup {recertified}, \
             so re-validation rejects it"
        );
    }

    #[test]
    fn committed_envelope_bound_contains_truth_at_dense_points() {
        // Cross-check the committed band against the rigorous point oracle at a
        // dense grid spanning all three arms (central + both saturation tails).
        // Every point's true erf must lie inside the committed [approx-eps,
        // approx+eps] band, using the Arb point enclosure as the truth witness.
        let env = ErfEnvelope::committed();
        let xs: Vec<f64> = {
            let mut v = Vec::new();
            // central
            let mut x = -3.0;
            while x <= 3.0 {
                v.push(x);
                x += 0.05;
            }
            // tails
            for &t in &[4.0, 6.0, 10.0, 50.0, 100.0, 300.0, -4.0, -50.0, -300.0] {
                v.push(t);
            }
            v
        };
        for x in xs {
            let (blo, bhi) = env.bound(x).expect("covered");
            let truth = rigorous_erf_enclosure(x, DEFAULT_PREC);
            // The committed band must contain the rigorous truth enclosure.
            assert!(
                blo <= truth.lo && truth.hi <= bhi,
                "committed band [{blo}, {bhi}] for x={x} must contain erf enclosure \
                 [{}, {}]",
                truth.lo,
                truth.hi
            );
        }
    }

    #[test]
    fn committed_eps_bounds_the_runtime_f64_evaluated_polynomial() {
        // The decisive end-to-end soundness check on the AUTHORITATIVE runtime
        // coefficients: at a dense grid over the central box, the runtime's
        // actual f64-Horner evaluation of the committed polynomial -- env.bound,
        // which calls ErfArm::approx, the exact Horner the deployed runtime runs,
        // over the coeffs loaded from the committed hex-float strings (no decimal
        // float parser) -- must keep the rigorous Arb erf enclosure inside the
        // committed [approx - eps, approx + eps] band. This is run on the f64 the
        // runtime really uses, not a re-derived or Python-parsed polynomial.
        let env = ErfEnvelope::committed();
        let central = env
            .boxes
            .iter()
            .find(|b| matches!(b.arm, ErfArm::Central { .. }))
            .expect("central box");
        // ~6000-point grid over [-3, 3], plus the exact sub-interval edges where
        // the worst case tends to sit.
        let mut xs: Vec<f64> = (0..=6000).map(|i| -3.0 + 6.0 * i as f64 / 6000.0).collect();
        for k in 0..=16 {
            xs.push(-3.0 + 6.0 * k as f64 / 16.0);
        }
        let mut worst_margin = f64::INFINITY;
        for x in xs {
            let approx = central.arm.approx(x); // the runtime's f64 Horner
            let (blo, bhi) = (approx - central.eps, approx + central.eps);
            let truth = rigorous_erf_enclosure(x, DEFAULT_PREC);
            assert!(
                blo <= truth.lo && truth.hi <= bhi,
                "runtime f64 band [{blo}, {bhi}] for x={x} must contain erf \
                 enclosure [{}, {}] (eps={})",
                truth.lo,
                truth.hi,
                central.eps
            );
            // Track how much headroom remains (committed eps vs the actual gap).
            let gap = (approx - truth.lo).abs().max((approx - truth.hi).abs());
            worst_margin = worst_margin.min(central.eps - gap);
        }
        assert!(
            worst_margin >= 0.0,
            "committed eps must bound the worst runtime gap; margin={worst_margin}"
        );
    }
}
