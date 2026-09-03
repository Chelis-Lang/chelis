//! Slice C c6: composite carriers of a deferred positional `expand` result.
//!
//! `spec/04-type-system.md` §4.7.2: "A carrier of a deferred result either
//! supplies an independent shape equation or does not. An anonymous tuple
//! field, a closure capture, a fresh generic field or parameter, a singleton
//! `List`, the seed element of an inferred `List`, and an undeclared closure
//! return carry the unresolved choice unchanged and add no evidence; none of
//! them may select a form, and none may split one result into two that settle
//! differently. A concrete declared record or ADT field, an
//! already-instantiated generic field, a declared `List` element type, and a
//! later `List` element facing an independently resolved accumulated element
//! shape each supply an independent equation and fix the result. Tuple,
//! record, and ADT projection and pattern binding transfer whichever form the
//! result has selected and add no evidence of their own. A record update fixes
//! an updated field only when that field's resolved schema is independent of
//! the result being carried."
//!
//! Every row uses one producer, `expand(a, 0, 3i64)` over `tensor[2, f32]`,
//! whose two legal forms are `tensor[3, f32]` and `tensor[3, 2, f32]`. A
//! propagating carrier is proved by letting a later consumer select the
//! insertion form, which the freeze default never produces; a carrier that
//! selects is proved by the stamped producer type alone.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

const REPLACEMENT: &str = "(t-tensor {} (d-lit {} 3) (t-prim {} f32))";
const INSERTION: &str = "(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))";

fn check(source: &str) -> Result<String, Vec<CheckError>> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The carrier added no evidence: a later consumer still selected the
/// insertion form, which no freeze default produces for this producer.
fn carried_without_evidence(context: &str, source: &str) {
    let rendered = check(source)
        .unwrap_or_else(|errors| panic!("{context}: expected acceptance:\n{}", summary(&errors)));
    assert!(
        rendered.contains(INSERTION),
        "{context}: the carrier must leave the choice open for a later \
         consumer to select:\n{rendered}"
    );
    assert!(
        !rendered.contains(REPLACEMENT),
        "{context}: nothing in this program should be stamped at the rank-1 \
         freeze default:\n{rendered}"
    );
}

/// The carrier supplied an independent equation and fixed the result.
fn fixed_the_result(context: &str, source: &str) {
    let rendered = check(source)
        .unwrap_or_else(|errors| panic!("{context}: expected acceptance:\n{}", summary(&errors)));
    assert!(
        rendered.contains(INSERTION),
        "{context}: the carrier's own equation must select the insertion \
         form:\n{rendered}"
    );
    assert!(
        !rendered.contains(REPLACEMENT),
        "{context}: the producer must not keep the freeze default:\n{rendered}"
    );
}

fn rejected(context: &str, source: &str) {
    let Err(errors) = check(source) else {
        panic!("{context}: this program must be rejected");
    };
    assert!(
        errors.iter().any(|error| {
            let kind = format!("{:?}", error.kind);
            kind.contains("Dimension") || kind.contains("TypeMismatch")
        }),
        "{context}: the rejection must name the shape disagreement:\n{}",
        summary(&errors)
    );
}

const SINK: &str = "def sink(x: tensor[3, 2, f32]) -> int32 = 0\n";

// ---------------------------------------------------------------------
// Carriers that add no evidence.
// ---------------------------------------------------------------------

#[test]
fn an_anonymous_tuple_field_carries_the_choice_and_projection_transfers_it() {
    carried_without_evidence(
        "anonymous tuple field",
        &format!(
            "{SINK}def f(a: tensor[2, f32]) -> int32 = {{\n  \
             t = (expand(a, 0, 3i64), 1)\n  \
             sink(t.0)\n}}\n"
        ),
    );
}

#[test]
fn a_closure_capture_and_an_undeclared_closure_return_carry_the_choice() {
    carried_without_evidence(
        "closure capture",
        &format!(
            "{SINK}def f(a: tensor[2, f32]) -> int32 = {{\n  \
             e = expand(a, 0, 3i64)\n  \
             g = fn (z: int32) -> e\n  \
             sink(g(0))\n}}\n"
        ),
    );
}

#[test]
fn a_fresh_generic_field_carries_the_choice() {
    carried_without_evidence(
        "fresh generic field",
        &format!(
            "type Box[a] = | Box {{ v: a }}\n{SINK}\
             def f(a: tensor[2, f32]) -> int32 = {{\n  \
             b = Box {{ v: expand(a, 0, 3i64) }}\n  \
             sink(b.v)\n}}\n"
        ),
    );
}

#[test]
fn a_singleton_list_carries_the_choice() {
    carried_without_evidence(
        "singleton list",
        "def sink_list(x: List[tensor[3, 2, f32]]) -> int32 = 0\n\
         def f(a: tensor[2, f32]) -> int32 = sink_list([expand(a, 0, 3i64)])\n",
    );
}

// ---------------------------------------------------------------------
// Carriers that supply an independent equation.
// ---------------------------------------------------------------------

#[test]
fn a_concrete_declared_record_field_fixes_the_result() {
    fixed_the_result(
        "declared record field",
        "type Box = | Box { v: tensor[3, 2, f32] }\n\
         def f(a: tensor[2, f32]) -> Box = Box { v: expand(a, 0, 3i64) }\n",
    );
    rejected(
        "declared record field, contradiction",
        "type Box = | Box { v: tensor[9, 9, f32] }\n\
         def f(a: tensor[2, f32]) -> Box = Box { v: expand(a, 0, 3i64) }\n",
    );
}

#[test]
fn a_declared_adt_field_fixes_the_result() {
    fixed_the_result(
        "declared ADT field",
        "type Wrap = | Wrap(tensor[3, 2, f32])\n\
         def f(a: tensor[2, f32]) -> Wrap = Wrap(expand(a, 0, 3i64))\n",
    );
    rejected(
        "declared ADT field, contradiction",
        "type Wrap = | Wrap(tensor[9, 9, f32])\n\
         def f(a: tensor[2, f32]) -> Wrap = Wrap(expand(a, 0, 3i64))\n",
    );
}

#[test]
fn a_declared_list_element_type_fixes_the_result() {
    fixed_the_result(
        "declared list element type",
        "def f(a: tensor[2, f32]) -> List[tensor[3, 2, f32]] = [expand(a, 0, 3i64)]\n",
    );
    rejected(
        "declared list element type, contradiction",
        "def f(a: tensor[2, f32]) -> List[tensor[9, 9, f32]] = [expand(a, 0, 3i64)]\n",
    );
}

#[test]
fn a_later_concrete_element_fixes_the_earlier_pending_seed() {
    fixed_the_result(
        "later concrete list element",
        "def f(a: tensor[2, f32], b: tensor[3, 2, f32]) = [expand(a, 0, 3i64), b]\n",
    );
}

#[test]
fn a_later_element_facing_a_resolved_accumulated_shape_is_fixed_by_it() {
    fixed_the_result(
        "resolved accumulated element shape",
        "def f(a: tensor[2, f32], b: tensor[3, 2, f32]) = [b, expand(a, 0, 3i64)]\n",
    );
}

#[test]
fn a_record_update_fixes_a_field_whose_schema_is_independent() {
    fixed_the_result(
        "record update",
        "type Box = | Box { v: tensor[3, 2, f32], n: int32 }\n\
         def f(a: tensor[2, f32], b: Box) -> Box = b with { v: expand(a, 0, 3i64) }\n",
    );
}

#[test]
fn pattern_binding_transfers_the_selected_form() {
    fixed_the_result(
        "pattern binding",
        &format!(
            "type Box = | Box {{ v: tensor[3, 2, f32] }}\n{SINK}\
             def f(a: tensor[2, f32]) -> int32 = \
             match Box {{ v: expand(a, 0, 3i64) }} with {{ | Box {{ v: w }} => sink(w) }}\n"
        ),
    );
}

// ---------------------------------------------------------------------
// The two clauses a carrier must never violate.
// ---------------------------------------------------------------------

#[test]
fn a_mismatched_list_element_rejects_rather_than_widening_the_axis() {
    // `Cons`'s tensor join deliberately widens mismatched literal dims to a
    // wildcard (chelis#218, tightened by chelis#272). A wildcard satisfies
    // either candidate form, so if a pending element reached that branch the
    // program would be accepted with the choice silently unmade. Both operand
    // orders must reject instead.
    rejected(
        "pending seed, mismatched later element",
        "def f(a: tensor[2, f32], b: tensor[9, 9, f32]) = [expand(a, 0, 3i64), b]\n",
    );
    rejected(
        "mismatched seed, pending later element",
        "def f(a: tensor[2, f32], b: tensor[9, 9, f32]) = [b, expand(a, 0, 3i64)]\n",
    );
}

#[test]
fn no_carrier_splits_one_result_into_two_forms_that_settle_differently() {
    // One binding, carried into two tuple fields. Writing `expand` twice
    // would be two results, which §4.7.2 permits to settle differently; the
    // clause is about one result reaching two carriers, so the producer is
    // bound once and the binding is what fans out.
    rejected(
        "one result, two contradictory consumers",
        "def want_flat(x: tensor[3, f32]) -> int32 = 0\n\
         def want_deep(x: tensor[3, 2, f32]) -> int32 = 0\n\
         def f(a: tensor[2, f32]) -> int32 = {\n  \
         e = expand(a, 0, 3i64)\n  \
         t = (e, e)\n  \
         add(want_flat(t.0), want_deep(t.1))\n}\n",
    );
    // Controls, so the rejection above cannot be read as "a tuple carrying one
    // binding twice always fails". The same shape with both consumers agreeing
    // is accepted, in both directions, which also shows the tuple really is
    // carrying the choice rather than forcing one.
    carried_without_evidence(
        "one result, two agreeing consumers, insertion",
        "def want_deep(x: tensor[3, 2, f32]) -> int32 = 0\n\
         def f(a: tensor[2, f32]) -> int32 = {\n  \
         e = expand(a, 0, 3i64)\n  \
         t = (e, e)\n  \
         add(want_deep(t.0), want_deep(t.1))\n}\n",
    );
    let rendered = check(
        "def want_flat(x: tensor[3, f32]) -> int32 = 0\n\
         def f(a: tensor[2, f32]) -> int32 = {\n  \
         e = expand(a, 0, 3i64)\n  \
         t = (e, e)\n  \
         add(want_flat(t.0), want_flat(t.1))\n}\n",
    )
    .unwrap_or_else(|errors| {
        panic!(
            "two agreeing consumers at the replacement form:\n{}",
            summary(&errors)
        )
    });
    assert!(
        rendered.contains(REPLACEMENT) && !rendered.contains(INSERTION),
        "both consumers agreeing on the replacement form selects it:\n{rendered}"
    );
}
