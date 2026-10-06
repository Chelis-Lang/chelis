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

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

use chelis_types::check_typed_program as check_typed;

const IDENTITIES: [&str; 7] = ["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"];

/// Both ingresses, because `infer_app` runs for each and the rejection sits
/// there rather than in a post-inference validator.
fn diagnostics(source: &str) -> (Vec<String>, Vec<String>) {
    let deep = surf_to_deep(source);
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
    let ir = match check_ir_program(&deep) {
        Ok(_) => Vec::new(),
        Err(r) => render(&r.errors),
    };
    let typed = match check_typed(&deep) {
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
        diags.iter().any(|d| d.contains("[05-OP-36]")),
        "{label}: the diagnostic must cite the atom that decides it: {diags:?}"
    );
    assert!(
        diags
            .iter()
            .any(|d| d.contains("expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32))")),
        "{label}: the diagnostic must name the replacement spelling, following \
         the `div`/`floor_div` precedent: {diags:?}"
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
        "def above() -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0], f32), 1.5)\n\
         def below() -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0], f32))\n",
        "both forms in one module",
    );
}

/// REGRESSION. Each order alone, which the base also accepted.
#[test]
fn issue5_each_operand_order_alone_is_refused() {
    assert_rejected_naming_the_replacement(
        "def below() -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0], f32))\n",
        "scalar first, alone",
    );
    assert_rejected_naming_the_replacement(
        "def above() -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0], f32), 1.5)\n",
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
            &format!("def f() = {op}(to_tensor([1.0f32, 2.0f32, 3.0f32]), 1.5f32)\n"),
            &format!("{op}(tensor, scalar)"),
        );
        assert_rejected_naming_the_replacement(
            &format!("def f() = {op}(1.5f32, to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"),
            &format!("{op}(scalar, tensor)"),
        );
    }
}

/// REGRESSION. The bool case, which has no numeric precision to disagree
/// about and so could only ever have been admitted by the rewrite.
#[test]
fn eq_refuses_a_bool_scalar_beside_a_bool_tensor_in_both_orders() {
    assert_rejected_naming_the_replacement(
        "def f() = eq(to_tensor([true, false]), true)\n",
        "eq(bool tensor, bool)",
    );
    assert_rejected_naming_the_replacement(
        "def f() = eq(true, to_tensor([true, false]))\n",
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
            &format!("def f() -> bool = {op}(1.5f32, 2.0f32)\n"),
            &format!("{op}(scalar, scalar)"),
        );
        assert_accepted(
            &format!(
                "def f() -> tensor[2, bool] = \
                 {op}(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32]))\n"
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
        "sig f[n]: tensor[n, f32] -> tensor[n, bool]\n\
         def f(xs) = gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))\n",
        "symbolic-extent replacement",
    );
    assert_accepted(
        "def f(xs: tensor[3, f32]) -> tensor[3, bool] = \
         gt(xs, expand(to_tensor([1.5f32]), 0i32, 3i64))\n",
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

fn check_surf(source: &str) -> Result<String, Vec<CheckError>> {
    let deep = surf_to_deep(source);
    match check_typed_program(&deep) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

#[test]
fn an_unconsumed_comparison_publishes_no_unresolved_variable() {
    let rendered = check_surf(
        "t = eq(expand(to_tensor([0.5f32]), 0, 3i64), expand(to_tensor([0.5f32]), 0, 3i64))\n",
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
