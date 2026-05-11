//! Phase Perf-F2(a): broader `DimExpr` canonicalization beyond v1's
//! product / identity rules.
//!
//! v1 (locked in `dag.rs::tests`) covers:
//! - Mul commutativity + associativity (factors sorted and flattened)
//! - Mul concrete folding (`2 * 1 * n == 2 * n`)
//! - Mul zero short-circuit (`0 * x == 0`)
//! - Div exact-eval (`12 / 3 == 4`)
//! - Div identity (`x / 1 == x`)
//! - Symbol identity (`Sym("n") != Sym("m")`, no alpha rename)
//!
//! Phase F2(a) adds:
//! - Concrete-factor GCD reduction in Div (`(n * 4) / 2 == n * 2`)
//! - Atom cancellation in Div (`(n * m) / n == m`, `(a * b * c) / (b * c) == a`)
//! - Div self-cancellation on atoms (`n / n == 1`)
//! - Nested-Div flattening (`(a / b) / c == a / (b * c)`,
//!   `a / (b / c) == (a * c) / b`)
//! - Mul × Div distribution (`(a / b) * b == a`,
//!   `(n / 2) * 2 == n`, `(n / 2) * 4 == n * 2`)
//!
//! The negative-test set covers unsoundness boundaries: indivisible
//! constant factors must not collapse, unrelated symbols must not be
//! alpha-renamed, partial cancellation must not over-apply, and a 0
//! denominator must stay visible to the runtime evaluator.

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

// ---------- Positive: GCD reduction on concrete factors ----------

#[test]
fn gcd_reduces_concrete_factor_in_numerator() {
    // (n * 4) / 2 == n * 2
    let lhs = div(mul(sym("n"), c(4)), c(2));
    let rhs = mul(sym("n"), c(2));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn gcd_reduces_concrete_factor_in_denominator() {
    // 6 / (n * 4) == 3 / (n * 2)
    let lhs = div(c(6), mul(sym("n"), c(4)));
    let rhs = div(c(3), mul(sym("n"), c(2)));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn gcd_reduces_both_sides_simultaneously() {
    // (12 * n) / (8 * m) == (3 * n) / (2 * m)
    let lhs = div(mul(c(12), sym("n")), mul(c(8), sym("m")));
    let rhs = div(mul(c(3), sym("n")), mul(c(2), sym("m")));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

// ---------- Positive: atom cancellation ----------

#[test]
fn shared_symbolic_atom_cancels_one_to_one() {
    // (n * m) / n == m
    let lhs = div(mul(sym("n"), sym("m")), sym("n"));
    let rhs = sym("m");
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn multiple_shared_atoms_cancel_in_a_block() {
    // (a * b * c) / (b * c) == a
    let lhs = div(
        mul(mul(sym("a"), sym("b")), sym("c")),
        mul(sym("b"), sym("c")),
    );
    let rhs = sym("a");
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn shared_atom_leaves_residual_quotient() {
    // (a * b) / (a * c) == b / c
    let lhs = div(mul(sym("a"), sym("b")), mul(sym("a"), sym("c")));
    let rhs = div(sym("b"), sym("c"));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn self_division_collapses_to_one_for_atom() {
    // n / n == 1
    let lhs = div(sym("n"), sym("n"));
    assert_eq!(lhs.normalized_key(), DimExprKey::Concrete(1));
}

#[test]
fn self_division_collapses_to_one_for_product() {
    // (a * b) / (a * b) == 1
    let lhs = div(mul(sym("a"), sym("b")), mul(sym("a"), sym("b")));
    assert_eq!(lhs.normalized_key(), DimExprKey::Concrete(1));
}

// ---------- Positive: combined GCD + atom cancellation ----------

#[test]
fn shared_atom_and_gcd_reduce_together() {
    // (n * 6) / (n * 4) == 3 / 2
    let lhs = div(mul(sym("n"), c(6)), mul(sym("n"), c(4)));
    let rhs = div(c(3), c(2));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn full_cancellation_yields_one() {
    // (n * 4) / (n * 4) == 1
    let lhs = div(mul(sym("n"), c(4)), mul(sym("n"), c(4)));
    assert_eq!(lhs.normalized_key(), DimExprKey::Concrete(1));
}

// ---------- Positive: nested-Div flattening ----------

#[test]
fn nested_div_in_numerator_flattens() {
    // (a / b) / c == a / (b * c)
    let lhs = div(div(sym("a"), sym("b")), sym("c"));
    let rhs = div(sym("a"), mul(sym("b"), sym("c")));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn nested_div_in_denominator_flattens() {
    // a / (b / c) == (a * c) / b
    let lhs = div(sym("a"), div(sym("b"), sym("c")));
    let rhs = div(mul(sym("a"), sym("c")), sym("b"));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn nested_div_both_sides_collapse() {
    // (a / b) / (c / d) == (a * d) / (b * c)
    let lhs = div(div(sym("a"), sym("b")), div(sym("c"), sym("d")));
    let rhs = div(mul(sym("a"), sym("d")), mul(sym("b"), sym("c")));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

// ---------- Positive: Mul × Div distribution and cross-term cancellation ----------

#[test]
fn product_containing_div_cancels_factor_in_numerator_with_denominator() {
    // (n / 2) * 2 == n
    let lhs = mul(div(sym("n"), c(2)), c(2));
    let rhs = sym("n");
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn product_containing_div_partially_cancels_with_constant() {
    // (n / 2) * 4 == n * 2
    let lhs = mul(div(sym("n"), c(2)), c(4));
    let rhs = mul(sym("n"), c(2));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn product_of_two_divs_combines_into_single_quotient() {
    // (a / b) * (c / d) == (a * c) / (b * d)
    let lhs = mul(div(sym("a"), sym("b")), div(sym("c"), sym("d")));
    let rhs = div(mul(sym("a"), sym("c")), mul(sym("b"), sym("d")));
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn product_with_div_cancels_shared_atom_across_factors() {
    // (a / b) * b == a, where `b` enters as an outer factor and as the
    // denominator of an inner Div.
    let lhs = mul(div(sym("a"), sym("b")), sym("b"));
    let rhs = sym("a");
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
}

// ---------- Positive: deeper structural equivalence ----------

#[test]
fn equivalent_shape_for_seq_hidden_swap_and_associativity() {
    // [seq, hidden, 4] and [hidden, 4, seq] both have element count
    // seq * hidden * 4. Memory planner asks for product-of-dims keys.
    let shape_a = mul(mul(sym("seq"), sym("hidden")), c(4));
    let shape_b = mul(mul(sym("hidden"), c(4)), sym("seq"));
    assert_eq!(shape_a.normalized_key(), shape_b.normalized_key());
}

#[test]
fn batch_view_dim_canonicalizes_under_grouped_reshape() {
    // A reshape that splits (batch * heads * dim) into (batch * (heads * dim))
    // and another that splits it into ((batch * heads) * dim) must have the
    // same key — the M2a slot planner should reuse the same backing slot.
    let split_a = mul(sym("batch"), mul(sym("heads"), sym("dim")));
    let split_b = mul(mul(sym("batch"), sym("heads")), sym("dim"));
    assert_eq!(split_a.normalized_key(), split_b.normalized_key());
}

// ---------- Negative: alpha-renaming is still forbidden ----------

#[test]
fn unrelated_symbols_do_not_collapse_after_extended_normalization() {
    // n / m must not equal x / y even though they have the same shape —
    // DimExpr carries no binder identity, so symbolic names are literal.
    let lhs = div(sym("n"), sym("m"));
    let rhs = div(sym("x"), sym("y"));
    assert_ne!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn unrelated_atoms_in_product_quotient_do_not_alpha_rename() {
    // (n * m) / p != (x * y) / z
    let lhs = div(mul(sym("n"), sym("m")), sym("p"));
    let rhs = div(mul(sym("x"), sym("y")), sym("z"));
    assert_ne!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn same_shape_different_symbol_quotient_not_collapsed() {
    // n / m must not equal p / q
    assert_ne!(
        div(sym("n"), sym("m")).normalized_key(),
        div(sym("p"), sym("q")).normalized_key()
    );
}

// ---------- Negative: indivisible constants must stay structural ----------

#[test]
fn indivisible_constant_factor_does_not_reduce() {
    // (n * 3) / 2: 3 is not divisible by 2, and `n` may not be even.
    // No sound reduction exists — the key must stay structural.
    let lhs = div(mul(sym("n"), c(3)), c(2));
    let rhs = mul(sym("n"), c(3)); // what an unsound rule might produce
    assert_ne!(lhs.normalized_key(), rhs.normalized_key());

    // It also must not collapse to `n / 2 * 3` rearranged into something
    // else like `n` or `1`.
    assert_ne!(lhs.normalized_key(), sym("n").normalized_key());
    assert_ne!(lhs.normalized_key(), DimExprKey::Concrete(1));
}

#[test]
fn coprime_constants_do_not_cancel() {
    // (n * 5) / 3: 5 and 3 are coprime; no reduction is sound. The key
    // must keep the symbolic 3 in the denominator.
    let lhs = div(mul(sym("n"), c(5)), c(3));
    let key = lhs.normalized_key();
    let DimExprKey::Div(_, denom) = &key else {
        panic!("expected structural Div, got {key:?}");
    };
    // The denominator must remain `Concrete(3)`, not be folded away or
    // recombined with the numerator's 5.
    assert_eq!(**denom, DimExprKey::Concrete(3));
}

#[test]
fn partial_atom_cancellation_does_not_overreach() {
    // (n * m) / (n * p) == m / p, NOT m / (n * p) and NOT (m / p) * n.
    let lhs = div(mul(sym("n"), sym("m")), mul(sym("n"), sym("p")));
    let expected = div(sym("m"), sym("p"));
    let unsound_a = div(sym("m"), mul(sym("n"), sym("p")));
    let unsound_b = mul(div(sym("m"), sym("p")), sym("n"));
    assert_eq!(lhs.normalized_key(), expected.normalized_key());
    assert_ne!(lhs.normalized_key(), unsound_a.normalized_key());
    assert_ne!(lhs.normalized_key(), unsound_b.normalized_key());
}

#[test]
fn multiplicity_is_respected_in_atom_cancellation() {
    // (n * n * m) / n == n * m (not m, because only ONE copy of n cancels).
    let lhs = div(mul(mul(sym("n"), sym("n")), sym("m")), sym("n"));
    let rhs = mul(sym("n"), sym("m"));
    let unsound = sym("m");
    assert_eq!(lhs.normalized_key(), rhs.normalized_key());
    assert_ne!(lhs.normalized_key(), unsound.normalized_key());
}

// ---------- Negative: zero handling stays defensive ----------

#[test]
fn zero_denominator_stays_structural() {
    // A key with a zero denominator is mathematically invalid; the
    // canonicalizer must not collapse it to something that hides the
    // error from the runtime evaluator. We keep a structural form so the
    // evaluator's divide-by-zero check still fires.
    let lhs = div(sym("n"), c(0));
    let key = lhs.normalized_key();
    let DimExprKey::Div(_, denom) = &key else {
        panic!("expected structural Div with zero denom, got {key:?}");
    };
    assert_eq!(**denom, DimExprKey::Concrete(0));
}

#[test]
fn zero_numerator_collapses_to_zero() {
    // 0 / n == 0 is mathematically sound for any n != 0.
    // (At n = 0 it's undefined, but the runtime evaluator catches that.)
    let lhs = div(c(0), sym("n"));
    assert_eq!(lhs.normalized_key(), DimExprKey::Concrete(0));
}

#[test]
fn zero_factor_in_product_short_circuits() {
    // 0 * (anything) == 0 — preserved from v1.
    let lhs = mul(c(0), mul(sym("n"), sym("m")));
    assert_eq!(lhs.normalized_key(), DimExprKey::Concrete(0));
}

// ---------- Negative: distinct quotients with same shape stay distinct ----------

#[test]
fn distinct_concrete_quotients_remain_distinct() {
    // 5 / 3 != 7 / 3, even though both stay structural.
    let lhs = div(c(5), c(3));
    let rhs = div(c(7), c(3));
    assert_ne!(lhs.normalized_key(), rhs.normalized_key());
}

#[test]
fn one_over_n_distinct_from_n_over_one() {
    // 1 / n != n / 1 (the latter collapses to n; the former stays a
    // structural quotient).
    let one_over_n = div(c(1), sym("n"));
    let n_over_one = div(sym("n"), c(1));
    assert_ne!(one_over_n.normalized_key(), n_over_one.normalized_key());
    assert_eq!(n_over_one.normalized_key(), DimExprKey::Sym("n".into()));
}
