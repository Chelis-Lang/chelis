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

#[cfg(all(test, feature = "arb"))]
mod arb_live_tests {
    //! Live FLINT/Arb enclosure tests. These need the `arb` feature (the
    //! vendored FLINT/Arb compiled `-fPIC`); they are the soundness oracle's
    //! own correctness gate. Run with:
    //!   cargo test -p chelis-prove --features arb arb_live_tests
    use super::{DEFAULT_PREC, certify_sup_norm_at_samples, rigorous_erf, rigorous_erf_enclosure};

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
}
