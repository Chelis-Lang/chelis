//! Linearity-AliasedConsume-F1 fixtures: pin the aliased-consume
//! bypass and the mixed alias-plus-destructure shape.
//!
//! Background. After PR #60 (V2-F4), the linearity checker treats
//! `let alias = x` as an aliasing consume (`ConsumeKind::Aliasing`
//! once Linearity-F1 lands). The discrimination keeps later borrow
//! reads of `x` alive because the IR-level `lower_let` aliases both
//! names to the same `Load` node. But `LinearScope.bindings` is keyed
//! by name only (see `crates/chelis-types/src/linearity.rs:39-42`),
//! so a structural consume on `alias` did not propagate to `x`'s
//! scope entry. The V3 final red team filed this as
//! `Linearity-AliasedConsume-F1` in `docs/gap_synthesis.md`.
//!
//! W1 PR #83 forwards alias-consumes to the source name's scope
//! entry so the underlying-value lineage is the unit of tracking.
//! Multi-level chains (`let z = y; let y = x; consume(z)`) walk to
//! the underlying source.
//!
//! Fixture 6 (mixed alias + destructure) compounds the aliased-
//! consume bypass with the tuple-destructure linearity gap
//! (Linearity-F2). The violation fires once both fixes ship. PR #83
//! initially routed the violation through `LinearityInfo::warnings`
//! during a deprecation window; the W2-cascade PR (this commit)
//! closes the window after the corpus survey confirmed zero in-tree
//! warnings, so the surfaced violation is now a hard error.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_linearity, check_typed_program};

fn linearity_errors(source: &str) -> Vec<chelis_types::errors::CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("type check should succeed");
    check_linearity(&checked).expect_err("linearity check must error")
}

/// Fixture 3: aliased-consume bypass (Linearity-AliasedConsume-F1).
/// `y = w` (aliasing) followed by `realize(y)` (structural on the
/// aliased name) followed by `add(w, z)` fails with
/// `UseAfterConsume` on `w`. Before W1.3 the consume on `y` did
/// not propagate to `w`'s scope entry, so the check silently passed.
///
/// After W1.3, `LinearScope.aliases` records `y -> w` from the
/// `let y = w` bind; `consume_var_expr` forwards the structural
/// consume on `y` through `resolve_alias_chain` to `w`; the
/// borrow of `w` in `add(w, z)` then trips `read_or_error` on the
/// underlying source.
#[test]
fn aliased_consume_bypass_errors_after_fix() {
    let errors = linearity_errors(
        r#"
def f(w: tensor[4, f32]) -> tensor[4, f32] =
  {
    y: tensor[4, f32] = w
    z: tensor[4, f32] = realize(y)
    add(w, z)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `w`")
        }),
        "expected UseAfterConsume on `w` consumed via aliased binding `y`; got {errors:?}"
    );
}

/// Fixture 6: mixed alias + destructure. After `(a, b) = pair` and
/// `y = a`, `realize(y)` is a structural consume on `a` via the
/// aliased binding. The follow-up `add(a, r)` is flagged as a
/// hard error.
///
/// Before W1.3 this fixture silently passed for two compounding
/// reasons: (i) destructured `a` had no type metadata, so the
/// consume on `realize(y)` was dispatched through the untyped
/// path; (ii) the aliased-consume bypass was independent of
/// destructure and tracked per-name.  W1.3 wired both
/// `tuple_get_element_type` and `LinearScope.aliases`. The
/// W2-cascade flips the surfaced violation from warning to error.
#[test]
fn destructure_then_alias_consume_errors() {
    let errors = linearity_errors(
        r#"
def f(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (a, b) = pair
    y: tensor[4, f32] = a
    r: tensor[4, f32] = realize(y)
    add(a, r)
  }
"#,
    );
    assert!(
        errors.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::UseAfterConsume) && e.message.contains("variable `a`")
        }),
        "expected UseAfterConsume on `a` via aliased binding `y` after destructure; \
         got {errors:?}"
    );
}
