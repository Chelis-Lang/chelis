//! chelis#1603: a `lit` whose `type:` stamp contradicts its atom must be
//! rejected wherever the checker admits it, at both checker ingresses.
//!
//! # The measurement the issue asks for
//!
//! #1603 asks whether this is a stamp-validator gap (one fix at ingress) or a
//! per-reader gap, and requires measuring literal positions before choosing.
//! This file IS that measurement: four position classes x five stamps x two
//! checker APIs, with the malformed and legal controls in the same table.
//!
//! Position classes:
//!   1. GENERAL   -- `(def {} v <lit>)`, a plain value position.
//!   2. NESTED    -- `<lit>` as an operand of an `app` inside a `fn` body.
//!   3. AXIS      -- `vmap`'s axis child (`spec/06` §3.1).
//!   4. SELECTOR  -- `tuple-get`'s index child.
//!
//! Classes 1 and 2 are visited by the inference walk (`infer_expr` ->
//! `infer_lit`, and `infer_app` visits every operand), so `infer_lit`'s
//! [04-LIT-1] atom/primitive matrix (shipped for chelis#1131) already decides
//! them. Classes 3 and 4 are form slots: `infer_vmap` and `infer_tuple_get`
//! READ the child through the `infer::slot` seam and never VISIT it, so the
//! boundary that owns the rule never ran on it. That asymmetry is the root
//! cause, and it makes this a coverage gap at two forms rather than a missing
//! rule -- so the fix routes those two children through the existing
//! boundary instead of re-deriving the matrix at each reader.
//!
//! # Spec authority
//!
//! * `spec/04-type-system.md` [04-LIT-1]: a primitive literal's Deep value
//!   atom SHALL agree with its declared primitive family; primitive type
//!   metadata never casts or reinterprets an atom. The sole cross-family form
//!   is an exact Int atom marked `literal_source: integer` bound to a float
//!   primitive. Every consumer SHALL reject an unmarked contradiction.
//! * `spec/04-type-system.md` §10 [04-TOT-3]: a structurally malformed Deep
//!   form reaching the checker SHALL be rejected with a diagnostic.
//! * `spec/03-deep-syntax.md` §6.4: the canonical `lit` forms.
//!
//! # The int64-stamped axis row is RECORDED, not asserted
//!
//! `spec/05-risc-primitives.md` [05-DIM-1]/[05-DIM-3] make the axis/extent
//! dtype split normative, not stylistic ("Two things make the distinction
//! normative rather than stylistic"), and [05-DIM-3] requires `int32` for
//! positional axis-domain arguments to movement, shape, reduction, ordering,
//! gathering, scattering, splitting and concatenation OPERATIONS. That rule
//! is already owned elsewhere in this crate -- the semantic/builtin operand
//! registry (`infer/app.rs`: "[05-DIM-3]: the semantic registry owns axis
//! dtype slots") -- not by the literal reader.
//!
//! `vmap` is a `spec/06-transformations.md` transformation, not a spec/05
//! operation, and spec/06 §3.1/§3.4/§3.6 state only "integer axis" with no
//! dtype. `tuple-get`'s index is not an axis at all. So no current normative
//! rule assigns `int32` to either child, and [04-LIT-1] alone admits an
//! `int64`-stamped Int atom exactly as it admits an `int32`-stamped one.
//! Whether [05-DIM-3]'s open "including ..." list reaches a spec/06 transform
//! axis is a spec question this bounded fix does not get to settle, so the
//! `int64` rows below assert only INGRESS PARITY and print the verdict they
//! observe. They deliberately do not pin accept-or-reject.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

// ---------------------------------------------------------------------------
// Harness: every cell reaches both checker APIs.
// ---------------------------------------------------------------------------

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source)
        .unwrap_or_else(|e| panic!("fixture must stamp: {source}\nstamp error: {e}"))
}

fn messages(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

/// Run one fixture through both checker ingresses and require them to agree.
///
/// Ingress disagreement is the chelis#1107 / [04-TOT-5] class and is a
/// failure in its own right, independent of which verdict is correct, so it
/// is checked for every cell including the recorded-only ones.
fn both_ingresses(source: &str) -> Vec<String> {
    let exprs = stamped(source);
    let typed = match check_typed_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    };
    let ir = match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    };
    assert_eq!(
        typed, ir,
        "check_typed_program and check_ir_program must agree \
         (chelis#1107, [04-TOT-5]) for:\n{source}"
    );
    typed
}

fn assert_accepted(label: &str, source: &str) {
    let errors = both_ingresses(source);
    assert!(
        errors.is_empty(),
        "{label}: legal control must stay accepted at both ingresses; \
         got {errors:?}\n{source}"
    );
}

/// A stamp/atom contradiction must be rejected, and the diagnostic must name
/// the incompatibility rather than only the enclosing form. "found a `lit`
/// form" is not an acceptable message here: a `lit` form IS a legal child of
/// these slots, so a slot-shape rejection would misreport the defect.
fn assert_rejected_for_stamp(label: &str, stamp: &str, source: &str) {
    let errors = both_ingresses(source);
    assert!(
        !errors.is_empty(),
        "{label}: a {stamp}-stamped Int atom must be rejected \
         (chelis#1603, [04-LIT-1], [04-TOT-3]); got a clean check\n{source}"
    );
    assert!(
        errors.iter().any(|m| m.contains("atom cannot carry")
            && m.contains(stamp)
            && m.contains("Deep literals")),
        "{label}: the diagnostic must name the literal and the \
         atom/primitive incompatibility, not just the enclosing form; \
         got {errors:?}\n{source}"
    );
}

/// Record a verdict without pinning it. Used only for the rows whose
/// governing rule is genuinely undecided (see the file header).
fn record_verdict(label: &str, source: &str) {
    let errors = both_ingresses(source);
    let verdict = if errors.is_empty() {
        "ACCEPTED".to_string()
    } else {
        format!("REJECTED {errors:?}")
    };
    println!("chelis#1603 recorded row: {label} => {verdict}");
}

// ---------------------------------------------------------------------------
// The five stamps under measurement.
// ---------------------------------------------------------------------------

/// Bare Int atom: hand-written Deep with no `lit` wrapper at all.
const BARE: &str = "1";
/// The spec/04 §5.3 default integer stamp. Legal.
const INT32: &str = "(lit {type: (t-prim {} int32)} 1)";
/// Integer family, wider carrier. Legal under [04-LIT-1]; see header.
const INT64: &str = "(lit {type: (t-prim {} int64)} 1)";
/// The issue's malformed input: an Int atom under a `bool` stamp.
const BOOL: &str = "(lit {type: (t-prim {} bool)} 1)";
/// [04-LIT-1]'s sole cross-family exception: an exact Int payload marked
/// `literal_source: integer` bound directly to a float primitive. LEGAL as a
/// literal; see the residue note at the end of this file for what it means in
/// an axis slot.
const F32_INT_SPELLED: &str =
    "(lit {type: (t-prim {} f32), literal_source: integer} 1)";

// Zero-valued spellings, for the selector position where the index must be
// in bounds for a two-element tuple.
const BARE_0: &str = "0";
const INT32_0: &str = "(lit {type: (t-prim {} int32)} 0)";
const INT64_0: &str = "(lit {type: (t-prim {} int64)} 0)";
const BOOL_0: &str = "(lit {type: (t-prim {} bool)} 0)";

// ---------------------------------------------------------------------------
// Class 1: GENERAL literal position -- `(def {} v <lit>)`.
// ---------------------------------------------------------------------------

fn general_program(lit: &str) -> String {
    format!("(def {{}} v {lit})")
}

#[test]
fn general_position_legal_stamps_stay_accepted() {
    for (label, lit) in [
        ("general/int32", INT32),
        ("general/int64", INT64),
        ("general/f32-integer-spelled", F32_INT_SPELLED),
    ] {
        assert_accepted(label, &general_program(lit));
    }
}

#[test]
fn general_position_bool_stamped_int_is_rejected() {
    assert_rejected_for_stamp("general/bool", "bool", &general_program(BOOL));
}

// ---------------------------------------------------------------------------
// Class 2: NESTED literal position -- an `app` operand inside a `fn` body.
// ---------------------------------------------------------------------------

fn nested_program(lit: &str) -> String {
    format!("(def {{}} h (fn {{}} (params {{}} x) (app {{}} (var {{}} add) (var {{}} x) {lit})))")
}

#[test]
fn nested_position_int32_stays_accepted() {
    assert_accepted("nested/int32", &nested_program(INT32));
}

#[test]
fn nested_position_bool_stamped_int_is_rejected() {
    assert_rejected_for_stamp("nested/bool", "bool", &nested_program(BOOL));
}

// ---------------------------------------------------------------------------
// Class 3: AXIS position -- `vmap`'s axis child.
// ---------------------------------------------------------------------------

/// Axis `1` is in bounds for the vmap wrapper of a rank-1 argument: the
/// checker inserts the batch axis, so the well-formed baseline needs no
/// higher-rank input.
const VMAP_SIG: &str = "(defsig {} f
    (t-fn {}
        (t-tensor {} (d-name {} features) (t-prim {} f32))
        (t-tensor {} (d-name {} features) (t-prim {} f32))))
(def {} f (fn {} (params {} x) (var {} x)))";

fn vmap_program(axis: &str) -> String {
    format!("{VMAP_SIG}\n(def {{}} g (vmap {{}} (var {{}} f) {axis}))")
}

#[test]
fn vmap_axis_legal_stamps_stay_accepted() {
    assert_accepted("axis/bare", &vmap_program(BARE));
    assert_accepted("axis/int32", &vmap_program(INT32));
}

#[test]
fn vmap_axis_bool_stamped_int_is_rejected() {
    assert_rejected_for_stamp("axis/bool", "bool", &vmap_program(BOOL));
}

#[test]
fn vmap_axis_int64_row_is_recorded_not_pinned() {
    record_verdict("axis/int64", &vmap_program(INT64));
}

#[test]
fn vmap_axis_integer_spelled_float_row_is_recorded_not_pinned() {
    record_verdict("axis/f32-integer-spelled", &vmap_program(F32_INT_SPELLED));
}

// ---------------------------------------------------------------------------
// Class 4: SELECTOR position -- `tuple-get`'s index child.
// ---------------------------------------------------------------------------

const TUPLE_VALUE: &str =
    "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 7) (lit {type: (t-prim {} f32)} 2.0)))";

fn tuple_get_program(index: &str) -> String {
    format!("{TUPLE_VALUE}\n(def {{}} x (tuple-get {{}} (var {{}} t) {index}))")
}

#[test]
fn tuple_get_index_legal_stamps_stay_accepted() {
    assert_accepted("selector/bare", &tuple_get_program(BARE_0));
    assert_accepted("selector/int32", &tuple_get_program(INT32_0));
}

#[test]
fn tuple_get_index_bool_stamped_int_is_rejected() {
    assert_rejected_for_stamp("selector/bool", "bool", &tuple_get_program(BOOL_0));
}

#[test]
fn tuple_get_index_int64_row_is_recorded_not_pinned() {
    record_verdict("selector/int64", &tuple_get_program(INT64_0));
}

// ---------------------------------------------------------------------------
// Residue, recorded here so the next reader does not have to rediscover it.
//
// `F32_INT_SPELLED` is a LEGAL literal under [04-LIT-1]'s marked exception,
// but it is a float-typed value, and it reads through the integer extractor
// into `vmap`'s axis slot. Rejecting it needs an axis-operand DTYPE rule for
// `vmap`, which is the same missing rule the `int64` rows above record; it is
// not a stamp/atom contradiction and so is outside #1603's "Expected". It is
// recorded, not fixed.
// ---------------------------------------------------------------------------
