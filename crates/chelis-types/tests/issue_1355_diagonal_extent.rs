//! Issue #1355: `diagonal` declares the smaller selected extent, not a wildcard.
//!
//! `[05-OP-33]` (`spec/05-risc-primitives.md`) states the extent exactly:
//! `diagonal` "keeps source axis order with the second axis removed, and
//! replaces the retained first axis extent with **the smaller selected
//! extent**". When both selected extents are literal, that minimum is
//! statically known, so the checker must declare it. `infer_diagonal_result_type`
//! previously widened every unequal literal pair to `Dim::Wildcard`, which
//! unifies with any declared extent, so a declared return type the runtime
//! cannot produce still type-checked.
//!
//! Every case here uses NON-SQUARE extents. A square `tensor[n, n]` cannot
//! distinguish the old behaviour from the correct one, which is why the defect
//! survived: every checked-in `diagonal`/`trace` case was square.
//!
//! This suite covers the rank-2 and rank-3 f32 literal-extent forms named in
//! each test, the identical-named-extent form, and the distinct-name and
//! mixed literal/symbolic forms that keep the wildcard; it is not evidence for
//! the full per-dtype [05-OP-33] runtime contract.

use chelis_deep::Expr;
use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

fn parse_deep(source: &str) -> Vec<Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture")
}

/// The single diagnostic a wrong declared `diagonal`/`trace` extent must
/// produce, so a negative cannot pass on an unrelated or extra error.
fn sole_error(source: &str, what: &str) -> CheckErrorKind {
    let deep = parse_deep(source);
    let Err(report) = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) else {
        panic!("{what} must be rejected");
    };
    let [error] = report.errors.as_slice() else {
        panic!(
            "expected exactly one error for {what}, got {:?}",
            report.errors
        );
    };
    error.kind.clone()
}

/// Classify a rejected extent without pinning diagnostic prose.
fn sole_dimension_mismatch(source: &str, what: &str) {
    let kind = sole_error(source, what);
    assert!(
        matches!(kind, CheckErrorKind::DimensionMismatch),
        "{what} must reject as a DimensionMismatch, got {kind:?}"
    );
}

fn accepts(source: &str, what: &str) {
    let deep = parse_deep(source);
    if let Err(report) = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        panic!("{what} must type-check, got {:?}", report.errors);
    }
}

// ---------------------------------------------------------------------------
// Regression tests (red before the fix, green after).
// ---------------------------------------------------------------------------

/// REGRESSION TEST. The exact program from chelis#1355: `tensor[3, 4]` over
/// axes (0, 1) has diagonal extent min(3, 4) = 3, so a declared `tensor[4]`
/// is a type error. Before the fix the result extent was `Dim::Wildcard`,
/// which unified with the declared 4 and scored a clean check.
#[test]
fn wider_declared_extent_is_rejected() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 3) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..65"}
      (var {span: "surf:48..56"} diagonal)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[3, 4] declared as tensor[4]",
    );
}

/// REGRESSION TEST. Mirror of the issue program: with the extents swapped the
/// minimum is the SECOND selected extent, so `tensor[4, 3]` also diagonalises
/// to `tensor[3]` and a declared `tensor[4]` is a type error. This is the case
/// a `min` that silently returned the first (retained) extent would pass.
#[test]
fn mirror_operand_rejects_the_retained_axis_extent() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..65"}
      (var {span: "surf:48..56"} diagonal)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[4, 3] declared as tensor[4]",
    );
}

/// REGRESSION TEST. Rank 3 with a retained axis: `diagonal(x, 1, 2)` on
/// `tensor[2, 3, 5]` removes axis 2 and replaces axis 1 with min(3, 5) = 3,
/// so the result is `tensor[2, 3]` and a declared `tensor[2, 5]` is a type
/// error. Proves the retained non-selected axis is untouched while the
/// selected one is narrowed.
#[test]
fn rank_three_retained_axis_rejects_the_larger_selected_extent() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (d-lit {} 5) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 5) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:54..71"}
      (var {span: "surf:54..62"} diagonal)
      (var {span: "surf:63..64"} x)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1)
      (lit {span: "surf:69..70", type: (t-prim {} i32)} 2))))
"#,
        "diagonal on tensor[2, 3, 5] over axes (1, 2) declared as tensor[2, 5]",
    );
}

/// REGRESSION TEST. `[05-OP-33]`'s smaller selected extent for two occurrences
/// of ONE named extent is that name: `spec/04-type-system.md` §4.1 makes two
/// `d-name` unify only when equal, so both axes of `tensor[hidden, hidden]`
/// denote a single runtime value. Declaring a DIFFERENT rigid name is therefore
/// a type error. Before the fix the result extent was `Dim::Wildcard`, which
/// unified with `width` and scored a clean check.
#[test]
fn identical_named_extents_reject_a_different_declared_name() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} hidden) (t-prim {} f32))
    (t-tensor {} (d-name {} width) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:62..79"}
      (var {span: "surf:62..70"} diagonal)
      (var {span: "surf:71..72"} x)
      (lit {span: "surf:74..75", type: (t-prim {} i32)} 0)
      (lit {span: "surf:77..78", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[hidden, hidden] declared as tensor[width]",
    );
}

// ---------------------------------------------------------------------------
// Positive controls (green before and after; they lock that the narrowing is
// not over-applied).
// ---------------------------------------------------------------------------

/// DISPOSITION LOCK. The correct declaration for the issue program keeps
/// type-checking. Green before the fix too (a wildcard unified with 3), so
/// this proves only that the narrowing did not turn a correct program into a
/// rejection.
#[test]
fn correct_smaller_extent_is_accepted() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 3) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..65"}
      (var {span: "surf:48..56"} diagonal)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[3, 4] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Mirror of the above: `tensor[4, 3]` declared `tensor[3]`.
#[test]
fn correct_mirror_extent_is_accepted() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..65"}
      (var {span: "surf:48..56"} diagonal)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[4, 3] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Rank 3 with the correct declaration.
#[test]
fn correct_rank_three_extent_is_accepted() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (d-lit {} 5) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:54..71"}
      (var {span: "surf:54..62"} diagonal)
      (var {span: "surf:63..64"} x)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1)
      (lit {span: "surf:69..70", type: (t-prim {} i32)} 2))))
"#,
        "diagonal on tensor[2, 3, 5] over axes (1, 2) declared as tensor[2, 3]",
    );
}

/// DISPOSITION LOCK. The square case the old equal-literals arm already
/// handled is subsumed by `min`, so `tensor[4, 4]` still declares `tensor[4]`
/// and still rejects `tensor[3]`. Green in both states.
#[test]
fn square_operand_keeps_its_exact_extent() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..65"}
      (var {span: "surf:48..56"} diagonal)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on the square tensor[4, 4]",
    );
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..65"}
      (var {span: "surf:48..56"} diagonal)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[4, 4] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. A symbolic selected extent keeps the wildcard: the
/// minimum of `n` and 4 is genuinely unknown at check time, so both the
/// symbolic and the literal declaration must keep type-checking. Green in
/// both states; it is the control that the fix narrowed only the
/// both-literal arm.
#[test]
fn symbolic_selected_extent_stays_wildcard_compatible() {
    accepts(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..68"}
      (var {span: "surf:51..59"} diagonal)
      (var {span: "surf:60..61"} x)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, 4] declared as tensor[n]",
    );
    accepts(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..68"}
      (var {span: "surf:51..59"} diagonal)
      (var {span: "surf:60..61"} x)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, 4] declared as tensor[4]",
    );
}

/// DISPOSITION LOCK. The positive twin of the two named-extent rejections:
/// `tensor[hidden, hidden]` declared `tensor[hidden]` is the correct program
/// and keeps type-checking. Green before the fix too (a wildcard unified with
/// `hidden`), so this proves only that the narrowing did not turn the correct
/// program into a rejection.
#[test]
fn identical_named_extents_accept_that_name() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} hidden) (t-prim {} f32))
    (t-tensor {} (d-name {} hidden) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:63..80"}
      (var {span: "surf:63..71"} diagonal)
      (var {span: "surf:72..73"} x)
      (lit {span: "surf:75..76", type: (t-prim {} i32)} 0)
      (lit {span: "surf:78..79", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[hidden, hidden] declared as tensor[hidden]",
    );
}

/// DISPOSITION LOCK, recording a boundary this change does NOT move. The named
/// arm does not make a declared LITERAL extent reject: `unify_dim` accepts
/// `Name` against `Lit` without binding a substitution, by the chelis#219
/// Option A rule that lets `def f(x: tensor[batch, hidden, f32])` be called with
/// concrete-shaped inputs. So `tensor[hidden, hidden]` declared `tensor[3, f32]`
/// is still admitted, before and after. That is dimension-unification policy,
/// not diagonal's extent rule, and #1355 does not touch it.
#[test]
fn identical_named_extents_still_admit_a_declared_literal() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} hidden) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:58..75"}
      (var {span: "surf:58..66"} diagonal)
      (var {span: "surf:67..68"} x)
      (lit {span: "surf:70..71", type: (t-prim {} i32)} 0)
      (lit {span: "surf:73..74", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[hidden, hidden] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Two DISTINCT named extents keep the wildcard: the minimum
/// of `hidden` and `width` is genuinely unknown at check time, so every
/// declaration must keep type-checking, including a literal one. Green in both
/// states; it is the control that the named arm fires only on equal names.
#[test]
fn distinct_named_extents_stay_wildcard_compatible() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} width) (t-prim {} f32))
    (t-tensor {} (d-name {} hidden) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:62..79"}
      (var {span: "surf:62..70"} diagonal)
      (var {span: "surf:71..72"} x)
      (lit {span: "surf:74..75", type: (t-prim {} i32)} 0)
      (lit {span: "surf:77..78", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[hidden, width] declared as tensor[hidden]",
    );
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} width) (t-prim {} f32))
    (t-tensor {} (d-name {} width) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:61..78"}
      (var {span: "surf:61..69"} diagonal)
      (var {span: "surf:70..71"} x)
      (lit {span: "surf:73..74", type: (t-prim {} i32)} 0)
      (lit {span: "surf:76..77", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[hidden, width] declared as tensor[width]",
    );
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} width) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:57..74"}
      (var {span: "surf:57..65"} diagonal)
      (var {span: "surf:66..67"} x)
      (lit {span: "surf:69..70", type: (t-prim {} i32)} 0)
      (lit {span: "surf:72..73", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[hidden, width] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. `trace` on identical named extents removes both axes and
/// is unaffected by the named arm, exactly as it is by the literal one.
#[test]
fn trace_on_identical_named_extents_is_rank_zero() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} hidden) (d-name {} hidden) (t-prim {} f32))
    (t-tensor {} (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:55..69"}
      (var {span: "surf:55..60"} trace)
      (var {span: "surf:61..62"} x)
      (lit {span: "surf:64..65", type: (t-prim {} i32)} 0)
      (lit {span: "surf:67..68", type: (t-prim {} i32)} 1))))
"#,
        "trace on tensor[hidden, hidden]",
    );
}

// ---------------------------------------------------------------------------
// `trace` controls: it routes through `infer_trace_result_type`, which removes
// both selected axes and never consults their extents. Green in both states.
// ---------------------------------------------------------------------------

/// DISPOSITION LOCK. `trace` on a NON-SQUARE operand removes both axes and
/// yields the rank-zero tensor, unaffected by the diagonal extent rule.
#[test]
fn trace_on_non_square_operand_is_rank_zero() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 3) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:45..59"}
      (var {span: "surf:45..50"} trace)
      (var {span: "surf:51..52"} x)
      (lit {span: "surf:54..55", type: (t-prim {} i32)} 0)
      (lit {span: "surf:57..58", type: (t-prim {} i32)} 1))))
"#,
        "trace on tensor[3, 4]",
    );
}

/// DISPOSITION LOCK. The failure twin of the above: `trace` removes BOTH
/// axes, so a declared `tensor[3]` (the diagonal's shape) is still a type
/// error. Green in both states; it proves the trace path did not inherit the
/// diagonal's retained axis.
#[test]
fn trace_does_not_retain_the_diagonal_extent() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 3) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..62"}
      (var {span: "surf:48..53"} trace)
      (var {span: "surf:54..55"} x)
      (lit {span: "surf:57..58", type: (t-prim {} i32)} 0)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 1))))
"#,
        "trace on tensor[3, 4] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Rank 3 `trace` keeps only the unselected axis.
#[test]
fn trace_keeps_only_the_unselected_axis() {
    accepts(
        r#"(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-lit {} 2) (d-lit {} 3) (d-lit {} 5) (t-prim {} f32))
    (t-tensor {} (d-lit {} 2) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..65"}
      (var {span: "surf:51..56"} trace)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 1)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 2))))
"#,
        "trace on tensor[2, 3, 5] over axes (1, 2)",
    );
}

// ---------------------------------------------------------------------------
// chelis#1739: a literal selected axis is an UPPER BOUND on the result extent.
//
// #1355 left the mixed (symbolic, literal) pair at `Dim::Wildcard`, which
// unifies with every declared extent. `[05-OP-33]` takes the SMALLER selected
// extent, so `min(n, 4) <= 4` holds for every runtime `n`: a declared extent
// strictly greater than the literal axis is unreachable and is now a type
// error. At or below the literal stays accepted, because the checker cannot
// decide which side of the minimum wins and the runtime guard owns the rest.
//
// The result dim itself is unchanged (still `Dim::Wildcard`); only the
// declared literal is compared against the bound, so the three #1355
// disposition locks above keep their exact dispositions.
// ---------------------------------------------------------------------------

/// REGRESSION TEST. The exact program from chelis#1739: `tensor[n, 4]` over
/// axes (0, 1) has diagonal extent min(n, 4), which is at most 4 for every
/// runtime `n`, so a declared `tensor[9]` is unreachable. Before the fix the
/// result extent was `Dim::Wildcard` and the program scored a clean 1.0.
#[test]
fn a_declared_literal_wider_than_the_literal_axis_is_rejected() {
    sole_dimension_mismatch(
        r#"(defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
        (t-tensor {} (d-lit {} 9) (t-prim {} f32))))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (app {span: "surf:51..68"}
          (var {span: "surf:51..59"} diagonal)
          (var {span: "surf:60..61"} x)
          (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
          (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
    "#,
        "diagonal on tensor[n, 4] declared as tensor[9]",
    );
}

/// REGRESSION TEST. The mirrored pair: the literal is the FIRST selected axis
/// and the symbolic one is second. `min(4, n) <= 4` is the same bound, so
/// `tensor[4, n]` declared `tensor[9]` is the same type error. This is the
/// case a bound that only read the second selected axis would admit.
#[test]
fn the_mirrored_literal_axis_bounds_the_declared_extent_too() {
    sole_dimension_mismatch(
        r#"(defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-lit {} 4) (d-var {} n) (t-prim {} f32))
        (t-tensor {} (d-lit {} 9) (t-prim {} f32))))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (app {span: "surf:51..68"}
          (var {span: "surf:51..59"} diagonal)
          (var {span: "surf:60..61"} x)
          (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
          (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
    "#,
        "diagonal on tensor[4, n] declared as tensor[9]",
    );
}

/// REGRESSION TEST. A RIGID named extent beside the literal is bounded exactly
/// like a dimension variable. Surf desugars a multi-letter dim to `d-name` and
/// a single-letter one to `d-var`, so the two spellings reach
/// `infer_diagonal_result_type` as different `Dim` variants; both are
/// non-literal and both are bounded by the literal axis.
#[test]
fn a_rigid_named_extent_beside_a_literal_is_bounded_the_same_way() {
    sole_dimension_mismatch(
        r#"(defsig {}
      f
      (t-fn {}
        (t-tensor {} (d-name {} hidden) (d-lit {} 4) (t-prim {} f32))
        (t-tensor {} (d-lit {} 9) (t-prim {} f32))))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (app {span: "surf:53..70"}
          (var {span: "surf:53..61"} diagonal)
          (var {span: "surf:62..63"} x)
          (lit {span: "surf:65..66", type: (t-prim {} i32)} 0)
          (lit {span: "surf:68..69", type: (t-prim {} i32)} 1))))
    "#,
        "diagonal on tensor[hidden, 4] declared as tensor[9]",
    );
}

/// REGRESSION TEST. Rank 3 with a retained axis: `diagonal(x, 1, 2)` on
/// `tensor[2, n, 5]` removes axis 2 and bounds the retained axis 1 by 5, so a
/// declared `tensor[2, 9]` is rejected while the untouched leading 2 is
/// compared as usual. Proves the bound is applied at the RESULT index of the
/// retained axis, not at its source index.
#[test]
fn a_rank_three_retained_axis_is_bounded_at_its_result_index() {
    sole_dimension_mismatch(
        r#"(defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-lit {} 2) (d-var {} n) (d-lit {} 5) (t-prim {} f32))
        (t-tensor {} (d-lit {} 2) (d-lit {} 9) (t-prim {} f32))))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (app {span: "surf:57..74"}
          (var {span: "surf:57..65"} diagonal)
          (var {span: "surf:66..67"} x)
          (lit {span: "surf:69..70", type: (t-prim {} i32)} 1)
          (lit {span: "surf:72..73", type: (t-prim {} i32)} 2))))
    "#,
        "diagonal on tensor[2, n, 5] over axes (1, 2) declared as tensor[2, 9]",
    );
}

/// REGRESSION TEST. The second ingress. A block-scoped ascription
/// (`y: tensor[9, f32] = diagonal(...)`) carries its declared type as `"type"`
/// metadata on the application node itself rather than through the enclosing
/// signature, so a rejection that only consulted the expected result type
/// would admit this spelling. Before the fix it scored a clean 1.0.
#[test]
fn the_block_ascription_ingress_is_bounded_too() {
    sole_dimension_mismatch(
        r#"(defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
        (t-tensor {} (d-lit {} 9) (t-prim {} f32))))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (let {span: "surf:51..98"}
          (bind {}
    y
    (app {span: "surf:75..92",
           surf_binding_type: "explicit",
           type: (t-tensor {} (d-lit {} 9) (t-prim {} f32))
         }
      (var {span: "surf:75..83"} diagonal)
      (var {span: "surf:84..85"} x)
      (lit {span: "surf:87..88", type: (t-prim {} i32)} 0)
      (lit {span: "surf:90..91", type: (t-prim {} i32)} 1)))
          (var {span: "surf:95..96"} y))))
    "#,
        "a block ascription of tensor[9] over tensor[n, 4]",
    );
}

/// DISPOSITION LOCK. At the bound is accepted: `min(n, 4)` reaches 4 whenever
/// `n >= 4`, so the declaration is satisfiable and only the runtime can decide.
/// Green before the fix too (a wildcard unified with 4); it is the control that
/// the bound rejects STRICTLY greater and not greater-or-equal.
#[test]
fn a_declared_literal_at_the_literal_axis_is_accepted() {
    accepts(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..68"}
      (var {span: "surf:51..59"} diagonal)
      (var {span: "surf:60..61"} x)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, 4] declared as tensor[4]",
    );
}

/// DISPOSITION LOCK. Below the bound is accepted for the same reason:
/// `min(n, 4) = 3` whenever `n = 3`. Green before the fix too.
#[test]
fn a_declared_literal_below_the_literal_axis_is_accepted() {
    accepts(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 3) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..68"}
      (var {span: "surf:51..59"} diagonal)
      (var {span: "surf:60..61"} x)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, 4] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK recording a case this change does NOT claim. Two
/// occurrences of ONE dimension VARIABLE (`tensor[n, n]`, which Surf desugars
/// to two `d-var` nodes rather than the `d-name` pair of `tensor[hidden,
/// hidden]`) keep the wildcard, so a declared literal is still admitted. chelis
/// #1517 left that disposition explicitly unclaimed and chelis#1739 does not
/// claim it either; this test records the behaviour, it does not assert that
/// the behaviour is right.
#[test]
fn the_two_dimension_variable_pair_stays_unclaimed() {
    accepts(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-lit {} 9) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..68"}
      (var {span: "surf:51..59"} diagonal)
      (var {span: "surf:60..61"} x)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 0)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, n] declared as tensor[9] (unclaimed disposition)",
    );
}

/// DISPOSITION LOCK. `trace` removes BOTH selected axes, so no result axis
/// carries the diagonal's extent and the literal selected axis bounds nothing.
/// The rank-zero declaration is accepted before and after.
#[test]
fn trace_over_a_symbolic_literal_pair_is_unaffected_by_the_bound() {
    accepts(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:48..62"}
      (var {span: "surf:48..53"} trace)
      (var {span: "surf:54..55"} x)
      (lit {span: "surf:57..58", type: (t-prim {} i32)} 0)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 1))))
"#,
        "trace on tensor[n, 4]",
    );
}

/// DISPOSITION LOCK, the failure twin of the above. `trace` on the same
/// operand declared `tensor[9]` is still rejected for the RANK, naming the
/// inferred rank-zero tensor rather than the diagonal's bound. Green in both
/// states; it proves the bound did not leak into the trace route.
#[test]
fn trace_over_a_symbolic_literal_pair_rejects_for_rank_not_for_the_bound() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 9) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:51..65"}
      (var {span: "surf:51..56"} trace)
      (var {span: "surf:57..58"} x)
      (lit {span: "surf:60..61", type: (t-prim {} i32)} 0)
      (lit {span: "surf:63..64", type: (t-prim {} i32)} 1))))
"#,
        "trace on tensor[n, 4] declared as tensor[9]",
    );
}

/// REGRESSION TEST. A declared type ALIAS carries the same verdict as its
/// expansion at both ingresses. The bound compares a resolved `Type`, not the
/// shape of the annotation's Deep metadata, so `type Row = tensor[9, f32]` is
/// rejected exactly as the spelled-out tensor type is.
#[test]
fn a_type_alias_carries_the_same_bound_verdict() {
    sole_dimension_mismatch(
        r#"(typealias {} Row () (t-tensor {} (d-lit {} 9) (t-prim {} f32)))
    
    (defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
        (t-adt {} Row)))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (app {span: "surf:66..83"}
          (var {span: "surf:66..74"} diagonal)
          (var {span: "surf:75..76"} x)
          (lit {span: "surf:78..79", type: (t-prim {} i32)} 0)
          (lit {span: "surf:81..82", type: (t-prim {} i32)} 1))))
    "#,
        "diagonal on tensor[n, 4] declared as the alias Row",
    );
    sole_dimension_mismatch(
        r#"(typealias {} Row () (t-tensor {} (d-lit {} 9) (t-prim {} f32)))
    
    (defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
        (t-adt {} Row)))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (let {span: "surf:66..102"}
          (bind {}
    y
    (app {span: "surf:79..96",
           surf_binding_type: "explicit",
           type: (t-adt {} Row)
         }
      (var {span: "surf:79..87"} diagonal)
      (var {span: "surf:88..89"} x)
      (lit {span: "surf:91..92", type: (t-prim {} i32)} 0)
      (lit {span: "surf:94..95", type: (t-prim {} i32)} 1)))
          (var {span: "surf:99..100"} y))))
    "#,
        "a block ascription of the alias Row over tensor[n, 4]",
    );
}

/// DISPOSITION LOCK. The positive twin: an alias at the bound is accepted, so
/// the alias route is not simply rejecting every aliased declaration.
#[test]
fn a_type_alias_at_the_bound_is_accepted() {
    accepts(
        r#"(typealias {} Row () (t-tensor {} (d-lit {} 4) (t-prim {} f32)))

(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (t-prim {} f32))
    (t-adt {} Row)))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:66..83"}
      (var {span: "surf:66..74"} diagonal)
      (var {span: "surf:75..76"} x)
      (lit {span: "surf:78..79", type: (t-prim {} i32)} 0)
      (lit {span: "surf:81..82", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, 4] declared as the alias Row = tensor[4, f32]",
    );
}

/// REGRESSION TEST (round 1 P2). A declaration whose RANK disagrees is a
/// signature mismatch, not a bound violation. Reading an axis out of it anyway
/// reported "the result extent is at most 4" for `tensor[n, 4, 5] ->
/// tensor[7]`, naming a repair that would not fix the program, and the error
/// type that path returns then suppressed the accurate diagnostic. The bound
/// now stands down on a rank disagreement and unification reports it.
#[test]
fn a_rank_disagreement_reports_the_signature_mismatch_not_the_bound() {
    sole_dimension_mismatch(
        r#"(defsig {}
  f
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (d-lit {} 4) (d-lit {} 5) (t-prim {} f32))
    (t-tensor {} (d-lit {} 7) (t-prim {} f32))))

(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:54..71"}
      (var {span: "surf:54..62"} diagonal)
      (var {span: "surf:63..64"} x)
      (lit {span: "surf:66..67", type: (t-prim {} i32)} 0)
      (lit {span: "surf:69..70", type: (t-prim {} i32)} 1))))
"#,
        "diagonal on tensor[n, 4, 5] over axes (0, 1) declared as the rank-1 tensor[7]",
    );
}

/// DISPOSITION LOCK, the control for the row above. With the RANK agreeing, the
/// same operand and axes keep the bound: the retained axis is bounded by the
/// literal 4 and a declared `tensor[9, 5]` is rejected as unreachable. Proves
/// the precondition narrowed the bound to rank agreement and did not disable it
/// for rank-2 results.
#[test]
fn a_rank_agreeing_declaration_still_carries_the_bound() {
    sole_dimension_mismatch(
        r#"(defsig {}
      f
      (n)
      (t-fn {}
        (t-tensor {} (d-var {} n) (d-lit {} 4) (d-lit {} 5) (t-prim {} f32))
        (t-tensor {} (d-lit {} 9) (d-lit {} 5) (t-prim {} f32))))
    
    (def {}
      f
      (fn {}
        (params {} (x {type: (t-var {} _)}))
        (app {span: "surf:57..74"}
          (var {span: "surf:57..65"} diagonal)
          (var {span: "surf:66..67"} x)
          (lit {span: "surf:69..70", type: (t-prim {} i32)} 0)
          (lit {span: "surf:72..73", type: (t-prim {} i32)} 1))))
    "#,
        "diagonal on tensor[n, 4, 5] over axes (0, 1) declared as tensor[9, 5]",
    );
}
