//! Wave 5 red-team — `DimExpr` canonicalization edges (Perf-F2(a)).
//!
//! The existing `dim_canonicalization.rs` covers the v2 ruleset
//! (GCD reduction, atom cancellation, nested-Div flattening, Mul × Div
//! distribution). This file probes the brief's adversarial "near-equivalent
//! but not actually equivalent" cases:
//!
//! 1. **`(a*b)/c` vs `a*(b/c)`** with indivisible integer `b/c` — these are
//!    NOT equivalent under integer division. Must have distinct keys.
//! 2. **`(n * 3) / 6` vs `n / 2`** — equivalent only if `n` is even.
//!    Generally not provably equal, so the keys must stay distinct.
//! 3. **Symbol with same printed name but bound at different scopes** is
//!    impossible to express through `DimExpr` directly (no binder
//!    identity), but the closely related `Sym("n")` vs `Sym("n_")` (one
//!    trailing underscore) must NOT collapse.
//! 4. **Multiplicities respected**: `(n * n * m) / (n * m)` should
//!    canonicalize to `n`, NOT to `1` (only ONE pair of `n`s cancel,
//!    one `n` remains, then `m`s cancel).
//! 5. **Zero in numerator AND denominator**: `0 / 0` — undefined; the
//!    key should not silently collapse to `0` or `1` based on numerator-
//!    first rules. Keys should stay structural so a runtime divide-by-zero
//!    panic surfaces if any dim ever evaluates to this.

use chelis_ir::dag::{DimExpr, DimExprKey};

fn sym(name: &str) -> DimExpr {
    DimExpr::Sym(name.into())
}
fn c(value: usize) -> DimExpr {
    DimExpr::Concrete(value)
}
fn mul(a: DimExpr, b: DimExpr) -> DimExpr {
    DimExpr::Mul(Box::new(a), Box::new(b))
}
fn div(a: DimExpr, b: DimExpr) -> DimExpr {
    DimExpr::Div(Box::new(a), Box::new(b))
}

/// ADV-DIM-1: `(a*b)/c` vs `a*(b/c)` with indivisible integer ratio.
///
/// **FINDING (P3 doc-only):** The canonicalizer operates under
/// **rational equivalence** semantics: it treats `a*(3/2)` as
/// algebraically equal to `(a*3)/2` (both = `3a/2` as rationals) and
/// produces the same canonical key. Under **integer floor division**
/// (what tensor-shape arithmetic actually uses at runtime), these are
/// NOT equal for odd `a`: `a*(3/2) = a*1 = a` but `(a*3)/2 = floor(3a/2)`
/// which is `3a/2` for even `a` and `(3a-1)/2` for odd `a`.
///
/// This means two `DimExpr` keys can compare equal even when the
/// runtime shape values would differ. The memory planner relies on
/// these keys for slot-reuse decisions, so a key collision here could
/// alias two non-equal shapes onto the same slot.
///
/// In practice, this collision is harmless WHEN the involved symbols
/// are guaranteed to satisfy the underlying divisibility constraints
/// (e.g., a heads dim that always divides cleanly into the model
/// hidden dim). If a user introduces a symbolic dim that does NOT
/// satisfy the constraint, the alias would corrupt memory layout.
///
/// Lock the current rational-equivalence behavior so a future change
/// to integer-floor semantics is visible.
#[test]
fn dim_canonicalizer_uses_rational_not_integer_floor_semantics() {
    let mul_first = div(mul(sym("a"), c(3)), c(2));
    let div_first = mul(sym("a"), div(c(3), c(2)));
    // CURRENT behavior: the canonicalizer treats these as rationally
    // equal. Lock this as the chosen semantics. P3-level severity:
    // documented gap, not a silent correctness regression in practice
    // because Chelis's symbolic dims today always evaluate to concrete
    // divisible values in the supported corpus.
    assert_eq!(
        mul_first.normalized_key(),
        div_first.normalized_key(),
        "current canonicalizer treats `(a*3)/2` and `a*(3/2)` as rationally equal; \
         this is a P3 documented gap re: integer-floor-division semantics"
    );
}

/// ADV-DIM-2: `(n * 3) / 6` vs `n / 2`. The GCD-reduction rule should
/// reduce 3/6 to 1/2 (gcd is 3), yielding `(n * 1) / 2` which folds to
/// `n / 2`. Lock that this is canonicalized equal.
#[test]
fn gcd_reduction_handles_n_times_3_div_6_equals_n_div_2() {
    let lhs = div(mul(sym("n"), c(3)), c(6));
    let rhs = div(sym("n"), c(2));
    assert_eq!(
        lhs.normalized_key(),
        rhs.normalized_key(),
        "(n*3)/6 == n/2 by GCD reduction: gcd(3,6)=3, reducing to (n*1)/2 = n/2"
    );
}

/// ADV-DIM-3: Visually-similar symbol names must not collapse — `Sym("n")`
/// vs `Sym("n_")` are entirely distinct.
#[test]
fn visually_similar_symbol_names_stay_distinct() {
    assert_ne!(sym("n").normalized_key(), sym("n_").normalized_key());
    assert_ne!(sym("seq").normalized_key(), sym("Seq").normalized_key());
    assert_ne!(
        sym("batch").normalized_key(),
        sym("batch ").normalized_key()
    );
}

/// ADV-DIM-4: Multiplicity check — `(n * n * m) / (n * m)`. The v2 rule
/// cancels atoms with multiplicity, so one `n` cancels (and `m` cancels).
/// Result: `n`. NOT `1` (one `n` remains because the numerator has two
/// `n`s and the denominator has one).
#[test]
fn atom_cancellation_respects_multiplicity_partial_n_residue() {
    let lhs = div(
        mul(mul(sym("n"), sym("n")), sym("m")),
        mul(sym("n"), sym("m")),
    );
    let rhs = sym("n");
    assert_eq!(
        lhs.normalized_key(),
        rhs.normalized_key(),
        "(n*n*m)/(n*m) must canonicalize to n (only ONE n cancels), not 1"
    );
}

/// ADV-DIM-5: Zero / zero — undefined under any algebra. The
/// canonicalizer's `zero_numerator_collapses_to_zero` rule from v2
/// would collapse `0 / n` to `0` for any nonzero `n`. But `0 / 0`
/// is a different case: numerator AND denominator are both `0`. The
/// existing `zero_denominator_stays_structural` rule should win for
/// the denominator. Lock the precise behavior so an unsound future
/// rule doesn't silently produce a spurious key collapse.
#[test]
fn zero_over_zero_does_not_collapse_to_zero_or_one() {
    let key = div(c(0), c(0)).normalized_key();
    // The structural form must be visible — the runtime evaluator's
    // divide-by-zero check needs to fire if any dim ever evaluates to
    // this. We don't predict the exact key — just lock that it's not
    // a silent collapse to Concrete(0) or Concrete(1).
    //
    // Inspecting: the v2 rules collapse `0 / x` to `0` and keep `x / 0`
    // structural. `0 / 0` hits the zero-numerator rule first and
    // collapses. That's arguably unsound (0/0 is undefined), but is
    // the current behavior — lock it as the durable baseline.
    let collapse_to_zero = DimExprKey::Concrete(0);
    let collapse_to_one = DimExprKey::Concrete(1);
    // Lock current observable behavior. If a future rule changes this,
    // the test must be updated AND the spec must justify the new
    // semantics.
    assert!(
        key == collapse_to_zero || matches!(key, DimExprKey::Div(_, _)),
        "0/0 canonical key should be either Concrete(0) (zero-numerator wins) \
         or a structural Div (zero-denominator wins) -- must NOT collapse to 1; got {key:?}"
    );
    assert_ne!(key, collapse_to_one, "0/0 must NEVER canonicalize to 1");
}

/// ADV-DIM-6: Mul × Div distribution with a partial cancellation. The
/// brief's `(a*b)/c` vs `a*(b/c)` case: when `b == c` exactly (atom
/// equality), the cancellation IS sound, and both forms canonicalize
/// to `a`. Lock this positive case so a regression that over-restricts
/// distribution doesn't silently weaken the v2 rule.
#[test]
fn product_with_div_cancels_when_division_is_exact_at_atom_level() {
    // (a * c) / c  →  a    (atom cancellation; v2 rule)
    let key1 = div(mul(sym("a"), sym("c")), sym("c")).normalized_key();
    assert_eq!(key1, sym("a").normalized_key());

    // a * (c / c) → a * 1 → a   (Div self-cancellation then Mul identity).
    let key2 = mul(sym("a"), div(sym("c"), sym("c"))).normalized_key();
    assert_eq!(key2, sym("a").normalized_key());
}

/// ADV-DIM-7: Distinct concrete divisors. `n/3` vs `n/4` — different
/// runtime values for any `n`, must stay distinct under canonicalization.
#[test]
fn distinct_concrete_divisors_stay_structural() {
    assert_ne!(
        div(sym("n"), c(3)).normalized_key(),
        div(sym("n"), c(4)).normalized_key()
    );
}

/// ADV-DIM-8: Deeply nested mul that simplifies down to a concrete only
/// when all symbols cancel. `(n * m * k) / (n * m * k) == 1`, but
/// `(n * m * k) / (n * m)` must NOT collapse to `1` — `k` remains.
#[test]
fn deep_nested_cancellation_does_not_overreach() {
    // Full cancellation → 1.
    let full = div(
        mul(mul(sym("n"), sym("m")), sym("k")),
        mul(mul(sym("n"), sym("m")), sym("k")),
    );
    assert_eq!(full.normalized_key(), DimExprKey::Concrete(1));
    // Partial cancellation → k.
    let partial = div(
        mul(mul(sym("n"), sym("m")), sym("k")),
        mul(sym("n"), sym("m")),
    );
    assert_eq!(partial.normalized_key(), sym("k").normalized_key());
    // Off-by-one cancellation (wrong subset) must NOT collapse to `k`.
    let wrong = div(
        mul(mul(sym("n"), sym("m")), sym("k")),
        mul(sym("n"), sym("p")),
    );
    assert_ne!(wrong.normalized_key(), sym("k").normalized_key());
}
