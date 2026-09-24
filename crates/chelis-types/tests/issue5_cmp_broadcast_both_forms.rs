//! Issue #5 was a diagnosis file: `gt(tensor, scalar)` and `gt(scalar, tensor)`
//! in one module made inference fail for both, and the repair was a rewrite
//! that gave the scalar the tensor's type "for unification purposes only".
//!
//! chelis#1506 removes that rewrite. `[05-OP-36]` says "Mixed surfaces,
//! numeric dtypes, static structured types, or tensor dimensions are type
//! errors"; `spec/05-risc-primitives.md` section 1.2 and
//! `spec/04-type-system.md` sections 4.2-4.3 call broadcasting a hard rule
//! with no exception; section 4.3 lists the comparison rows as "tensor,
//! tensor" only; and the section 2.1 prose above `[05-OP-40]` says the scalar
//! form of `max_elem`/`min_elem` "is the rank-zero instance of the tensor
//! rule, not scalar/tensor broadcasting". `add` and `max_elem` already
//! rejected the same pair, so the seven comparison identities were the one
//! surface where a mixed pair checked clean.
//!
//! So the three rows that asserted issue #5's acceptance now assert its
//! rejection, and issue #5's own subject survives inverted: the two operand
//! ORDERS must behave identically, which they do by both being refused with
//! the same diagnostic.
//!
//! # Evidentiary status, per assertion
//!
//! REGRESSION for every rejection row: each accepted on the base `9b9e7bd56`
//! at both ingresses and rejects after. Proven by running this file against
//! the tree with the rewrite present and watching the rejections fail.
//!
//! DISPOSITION LOCK for every acceptance row: scalar beside scalar, tensor
//! beside tensor, and the replacement spelling all checked clean before and
//! after. They are what stops the rejection widening into "comparisons no
//! longer take tensors".

use chelis_types::check_ir_program;

fn parse_core(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("valid core fixture")
}

use chelis_types::check_typed_program as check_typed;

const IDENTITIES: [&str; 7] = ["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"];

/// Both ingresses, because `infer_app` runs for each and the rejection sits
/// there rather than in a post-inference validator.
fn diagnostics(source: &str) -> (Vec<String>, Vec<String>) {
    let deep = parse_core(source);
    let render = |errors: &[CheckError]| -> Vec<String> {
        let mut out: Vec<String> = errors
            .iter()
            .map(|e| {
                format!(
                    "[{:?}] {} || {}",
                    e.kind,
                    e.message,
                    e.suggestions.join(" ")
                )
            })
            .collect();
        out.sort();
        out
    };
    let ir = match check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(r) => render(&r.errors),
    };
    let typed = match check_typed(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(r) => render(&r.errors),
    };
    assert_eq!(
        ir, typed,
        "both ingresses must agree for:\n{source}\nir={ir:?}\ntyped={typed:?}"
    );
    (ir, typed)
}

fn assert_rejected_naming_the_replacement(source: &str, label: &str) {
    let (diags, _) = diagnostics(source);
    assert!(
        diags
            .iter()
            .any(|diagnostic| diagnostic.starts_with("[TypeMismatch]")),
        "{label}: mixed scalar/tensor operands must reject: {diags:?}"
    );
}

fn assert_accepted(source: &str, label: &str) {
    let (diags, _) = diagnostics(source);
    assert!(diags.is_empty(), "{label} must check clean: {diags:?}");
}

/// REGRESSION. Issue #5's own two-form module: both orders accepted on the
/// base, and both are now refused. The orders still behave identically, which
/// was issue #5's subject.
#[test]
fn issue5_both_operand_orders_are_refused_alike() {
    assert_rejected_naming_the_replacement(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/issue5_both_operand_orders_are_refused_alike.0.deep"
        ),
        "both forms in one module",
    );
}

/// REGRESSION. Each order alone, which the base also accepted.
#[test]
fn issue5_each_operand_order_alone_is_refused() {
    assert_rejected_naming_the_replacement(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/issue5_each_operand_order_alone_is_refused.0.deep"
        ),
        "scalar first, alone",
    );
    assert_rejected_naming_the_replacement(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/issue5_each_operand_order_alone_is_refused.1.deep"
        ),
        "tensor first, alone",
    );
}

/// REGRESSION, all seven identities, both operand orders. The claim is stated
/// over exactly this enumeration of `builtins::COMPARISON_OPS`, which is the
/// list the rejection itself keys on.
#[test]
fn every_comparison_identity_refuses_a_scalar_beside_a_tensor_in_both_orders() {
    for op in IDENTITIES {
        assert_rejected_naming_the_replacement(
            &format!(
                r#"(def {{}}
  f
  (fn {{}}
    (params {{}})
    (app {{span: "surf:10..60"}}
      (var {{span: "surf:10..15"}} {op})
      (app {{span: "surf:16..51"}}
        (var {{span: "surf:16..25"}} to_tensor)
        (app {{span: "surf:26..50"}}
          (var {{}} Cons)
          (lit {{span: "surf:27..33", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 1.0)
          (app {{}}
            (var {{}} Cons)
            (lit {{span: "surf:35..41", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 2.0)
            (app {{}}
              (var {{}} Cons)
              (lit {{span: "surf:43..49", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 3.0)
              (var {{}} Nil)))))
      (lit {{span: "surf:53..59", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 1.5))))
"#
            ),
            &format!("{op}(tensor, scalar)"),
        );
        assert_rejected_naming_the_replacement(
            &format!(
                r#"(def {{}}
  f
  (fn {{}}
    (params {{}})
    (app {{span: "surf:10..60"}}
      (var {{span: "surf:10..15"}} {op})
      (lit {{span: "surf:16..22", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 1.5)
      (app {{span: "surf:24..59"}}
        (var {{span: "surf:24..33"}} to_tensor)
        (app {{span: "surf:34..58"}}
          (var {{}} Cons)
          (lit {{span: "surf:35..41", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 1.0)
          (app {{}}
            (var {{}} Cons)
            (lit {{span: "surf:43..49", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 2.0)
            (app {{}}
              (var {{}} Cons)
              (lit {{span: "surf:51..57", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 3.0)
              (var {{}} Nil))))))))
"#
            ),
            &format!("{op}(scalar, tensor)"),
        );
    }
}

/// REGRESSION. The bool case, which has no numeric precision to disagree
/// about and so could only ever have been admitted by the rewrite.
#[test]
fn eq_refuses_a_bool_scalar_beside_a_bool_tensor_in_both_orders() {
    assert_rejected_naming_the_replacement(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/eq_refuses_a_bool_scalar_beside_a_bool_tensor_in_both_orders.0.deep"
        ),
        "eq(bool tensor, bool)",
    );
    assert_rejected_naming_the_replacement(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/eq_refuses_a_bool_scalar_beside_a_bool_tensor_in_both_orders.1.deep"
        ),
        "eq(bool, bool tensor)",
    );
}

/// DISPOSITION LOCK. Scalar beside scalar stays a `bool`, and tensor beside
/// tensor of one shape stays a `tensor[D, bool]`, for all seven identities.
/// Negative parity: the rejection must not widen into the tensor rule itself.
#[test]
fn scalar_pairs_and_matching_tensor_pairs_are_still_accepted() {
    for op in IDENTITIES {
        assert_accepted(
            &format!(
                r#"(defsig {{}} f (t-fn {{}} (t-prim {{}} bool)))

(def {{}}
  f
  (fn {{}}
    (params {{}})
    (app {{span: "surf:18..39"}}
      (var {{span: "surf:18..23"}} {op})
      (lit {{span: "surf:24..30", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 1.5)
      (lit {{span: "surf:32..38", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 2.0))))
"#
            ),
            &format!("{op}(scalar, scalar)"),
        );
        assert_accepted(
            &format!(
                r#"(defsig {{}} f (t-fn {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} bool))))

(def {{}}
  f
  (fn {{}}
    (params {{}})
    (app {{span: "surf:29..92"}}
      (var {{span: "surf:29..34"}} {op})
      (app {{span: "surf:35..62"}}
        (var {{span: "surf:35..44"}} to_tensor)
        (app {{span: "surf:45..61"}}
          (var {{}} Cons)
          (lit {{span: "surf:46..52", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 1.0)
          (app {{}}
            (var {{}} Cons)
            (lit {{span: "surf:54..60", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 2.0)
            (var {{}} Nil))))
      (app {{span: "surf:64..91"}}
        (var {{span: "surf:64..73"}} to_tensor)
        (app {{span: "surf:74..90"}}
          (var {{}} Cons)
          (lit {{span: "surf:75..81", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 3.0)
          (app {{}}
            (var {{}} Cons)
            (lit {{span: "surf:83..89", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 4.0)
            (var {{}} Nil)))))))
"#
            ),
            &format!("{op}(tensor, tensor)"),
        );
    }
}

/// DISPOSITION LOCK. The replacement the diagnostic names must itself type
/// check, or the rejection sends users nowhere. Both the symbolic-extent form
/// and the literal-extent form.
#[test]
fn the_replacement_spelling_the_diagnostic_names_type_checks() {
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/the_replacement_spelling_the_diagnostic_names_type_checks.0.deep"
        ),
        "symbolic-extent replacement",
    );
    assert_accepted(
        include_str!(
            "fixtures/core/tests/issue5_cmp_broadcast_both_forms/the_replacement_spelling_the_diagnostic_names_type_checks.1.deep"
        ),
        "literal-extent replacement",
    );
}

// A comparison relates its operands' shapes and its result's dtype under
// [05-OP-36], so it routes its result through unification rather than
// constructing it out of band.
//
// The five rows that proved this by having a consumer SELECT among an
// operand's candidate shapes are gone with the candidate model: under
// `spec/04-type-system.md` section 4.7.2 `expand` and `insert` each have one
// result shape, so no operand is ever pending and no consumer selects
// (chelis#1277 S2b). chelis#1265's own subject went with them; the design doc
// records it as superseded and to be re-read against section 4.7.2.
//
// The row that survives is the one whose subject was never selection: a
// comparison must publish a concrete result rather than an inference
// variable. It is a disposition lock under the single result shape.

use chelis_deep::printer::print_canonical;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check_core(source: &str) -> Result<String, Vec<CheckError>> {
    let deep = parse_core(source);
    match check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

#[test]
fn an_unconsumed_comparison_publishes_no_unresolved_variable() {
    let rendered = check_core(
        include_str!("fixtures/core/tests/issue5_cmp_broadcast_both_forms/an_unconsumed_comparison_publishes_no_unresolved_variable.0.deep"),
    )
    .expect("an unconsumed comparison over two expand results still checks");
    assert!(
        !rendered.contains("(t-var"),
        "a comparison result must settle to a concrete tensor rather than \
         escaping as an inference variable:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} bool))"),
        "the result mirrors the operand's shape at bool:\n{rendered}"
    );
}
