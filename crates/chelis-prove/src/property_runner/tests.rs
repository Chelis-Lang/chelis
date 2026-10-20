//! Shared property-runner tests (U4 review-3 unification).

use super::*;

#[test]
fn prepared_surf_fuzz_library_preserves_source_provenance() {
    let forged = "def pkg__demo__Hidden__identity(x: i64) -> i64 = x\n@property holds forall(x: i64): pkg__demo__Hidden__identity(x) == x\n";
    let declarations = chelis_surf::parser::parse_str(forged).unwrap();
    let baseline =
        chelis_compiler_api::compiler::eval_decls_selected(&declarations, BTreeMap::new(), &[])
            .unwrap_err();
    assert!(
        baseline
            .errors
            .iter()
            .any(|error| error.kind().as_str() == "ReservedLinkerName")
    );
    let outcomes = run_surf(forged, "fuzz-only");
    assert_eq!(outcomes[0].status, PropertyStatus::Error, "{outcomes:#?}");
    assert_eq!(outcomes[0].accepted_samples, 0);

    let authored = forged.replace("pkg__demo__Hidden__identity", "identity");
    assert!(run_surf(&authored, "fuzz-only")[0].is_pass());

    // The Reef caller's explicit provenance must survive library preparation.
    let _linked = chelis_types::install_linked_program_guard();
    let options = PropertyRunOptions {
        tier: "fuzz-only".into(),
        samples: 4,
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(outcomes) =
        run_surf_decls_properties(&declarations, &declarations, &declarations, &options).unwrap();
    assert!(outcomes[0].is_pass(), "{outcomes:#?}");
    assert_eq!(outcomes[0].accepted_samples, 4);
}

#[test]
fn prepared_surf_samples_preserve_monolithic_guards_predicates_and_shrinking() {
    for source in [
        "@property scalar forall(x: f32) where x >= 0.0: x > 5.0\n",
        "@property integer forall(x: i64): x > 5i64\n",
        "@property text forall(x: string): x == \"s0\"\n",
    ] {
        let decls = chelis_surf::parser::parse_str(source).unwrap();
        let property = collect_surf_properties(&decls, &decls, None)
            .unwrap()
            .remove(0);
        let deep = chelis_surf::desugar::desugar_program(&decls).unwrap();
        let roots = SampleRoots::for_program(&deep);
        let cell = SurfEvalProgramCell::new();
        let Ok(program) = surf_eval_program(&decls, &cell) else {
            panic!("{source}: the declaration context builds");
        };
        let lane = SurfLane::new(program, &property);
        let monolithic = |sample: &Sample, precondition: bool| {
            let mut whole = decls.clone();
            whole.push(Decl::LetDef {
                name: roots.select(precondition).to_string(),
                ty: None,
                value: sample_block_expr(&property, sample, precondition),
                span: deep_span(),
            });
            bool_property_root(chelis_compiler_api::compiler::eval_decls_selected(
                &whole,
                BTreeMap::new(),
                &[roots.select(precondition).to_string()],
            ))
        };
        let mut rng = Lcg::new(0);
        let samples = (0..8)
            .map(|attempt| sample_property(&property, &mut rng, attempt, 100).unwrap())
            .collect::<Vec<_>>();
        // One compile holds every sample's probes; each answers as the
        // sample's own whole program does.
        let batch = sample_batch::SampleBatch::new(&lane, samples.clone());
        for (index, sample) in samples.iter().enumerate() {
            for precondition in [true, false] {
                let expected = monolithic(sample, precondition).expect("baseline sample executes");
                assert_eq!(
                    eval_surf_sample(&program.library, &property, &roots, sample, precondition),
                    Ok(expected),
                    "{source}"
                );
                if precondition && property.preconditions.is_empty() {
                    continue;
                }
                assert_eq!(batch.eval(index, precondition), Ok(expected), "{source}");
            }
            let still_fails = |sample: &Sample| {
                (property.preconditions.is_empty() || matches!(monolithic(sample, true), Ok(true)))
                    && matches!(monolithic(sample, false), Ok(false))
            };
            if still_fails(sample) {
                let (expected, expected_steps) =
                    shrink_counterexample(sample.clone(), &property.params, |trials| {
                        trials.into_iter().find(|trial| still_fails(trial))
                    });
                let (actual, actual_steps) =
                    sample_batch::shrink_with_lane(sample.clone(), &property.params, &lane);
                assert_eq!(
                    counterexample_json(&actual),
                    counterexample_json(&expected),
                    "{source}"
                );
                assert_eq!(actual_steps, expected_steps, "{source}");
            }
        }
    }
}

#[test]
fn deep_param_rejects_unknown_form_but_reads_a_structural_param() {
    let span = deep_span();
    let structural = DeepExpr::BareList(
        vec![
            DeepExpr::Atom(DeepAtom::Name("x".into()), span),
            DeepExpr::Map(Metadata::default(), span),
        ],
        span,
    );
    let unknown = DeepExpr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
        head: "x".into(),
        meta: Metadata::default(),
        children: Vec::new(),
        span,
    }));
    let malformed = DeepExpr::BareList(
        vec![
            DeepExpr::Atom(DeepAtom::Name("x".into()), span),
            DeepExpr::Atom(DeepAtom::Int(0), span),
        ],
        span,
    );

    let param = deep_param(&structural).expect("structural inline param is readable");
    assert_eq!(param.name, "x");
    assert!(param.ty.is_none());
    assert!(deep_param(&unknown).is_none());
    let malformed =
        deep_param(&malformed).expect("malformed structural inline param retains its binder");
    assert_eq!(malformed.name, "x");
    assert!(malformed.ty.is_none());
}

fn run_surf(source: &str, tier: &str) -> Vec<PropertyOutcome> {
    let opts = PropertyRunOptions {
        tier: tier.to_string(),
        samples: 32,
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(o) = run_surf_source_properties(source, &opts).expect("run");
    o
}

#[cfg(feature = "smt")]
const GENERAL_BOND_INDUCTION: &str = "module M
def bond_value(n: i32, coupon: f64, discount: f64) -> f64 =
  if (n <= 0) then cast(1.0, f64)
  else coupon + discount * bond_value(n - 1, coupon, discount)
@property bond_value_nonnegative forall(n: i32, coupon: f64, discount: f64)
where n >= 0, coupon >= cast(0.0, f64), discount >= cast(0.0, f64):
  (bond_value(n, coupon, discount) >= cast(0.0, f64))
";

#[cfg(feature = "smt")]
#[test]
fn general_bond_induction_disposes_real_base_and_step_obligations() {
    let outcomes = run_surf(GENERAL_BOND_INDUCTION, "induction-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);
    assert!(outcome.is_pass());
    let evidence = outcome
        .induction_evidence
        .as_ref()
        .expect("a green induction must disclose both obligations");
    assert_eq!(evidence.base.status, "proved");
    assert_eq!(evidence.step.status, "proved");
    assert!(
        outcome
            .assumptions
            .iter()
            .all(|record| !format!("{record:?}").contains("ASSUMED")),
        "caller assertions must never become proof: {outcome:#?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn false_general_bond_induction_is_rejected_by_step_obligation() {
    let source = GENERAL_BOND_INDUCTION.replace(
        "coupon + discount * bond_value(n - 1, coupon, discount)",
        "coupon - cast(1.0, f64) + discount * bond_value(n - 1, coupon, discount)",
    );
    let outcomes = run_surf(&source, "induction-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:#?}");
    assert!(!outcome.is_pass());
    let evidence = outcome
        .induction_evidence
        .as_ref()
        .expect("the failed step must remain visible");
    assert_eq!(evidence.base.status, "proved");
    assert_eq!(evidence.step.status, "disproved");
}

#[cfg(feature = "smt")]
#[test]
fn auto_dispatches_eligible_induction_before_recursive_tier_b_lowering() {
    let outcomes = run_surf(GENERAL_BOND_INDUCTION, "auto");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);
    assert_eq!(outcome.samples, 0);
    assert!(outcome.induction_evidence.is_some());
}

#[cfg(feature = "smt")]
#[test]
fn auto_keeps_failed_induction_terminal_instead_of_fuzz_laundering() {
    let source = GENERAL_BOND_INDUCTION.replace(
        "coupon + discount * bond_value(n - 1, coupon, discount)",
        "coupon - cast(1.0, f64) + discount * bond_value(n - 1, coupon, discount)",
    );
    let outcome = &run_surf(&source, "auto")[0];
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);
    assert_eq!(outcome.samples, 0);
    assert!(!outcome.is_pass());
}

#[cfg(feature = "smt")]
#[test]
fn auto_fails_closed_on_recursive_but_unsupported_induction_shape() {
    let source = GENERAL_BOND_INDUCTION.replace("n - 1", "n + 1");
    let outcome = &run_surf(&source, "auto")[0];
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);
    assert_eq!(outcome.samples, 0);
    assert!(outcome.induction_evidence.is_none());
}

#[cfg(feature = "smt")]
#[test]
fn induction_rejects_non_structural_recursion_instead_of_assuming_it() {
    let source = GENERAL_BOND_INDUCTION.replace("n - 1", "n + 1");
    let outcomes = run_surf(&source, "induction-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);
    assert!(!outcome.is_pass());
    assert!(outcome.induction_evidence.is_none());
}

#[cfg(feature = "smt")]
#[test]
fn induction_never_dispatches_an_unchecked_parser_ast() {
    let source = GENERAL_BOND_INDUCTION.replace(
        "discount * bond_value(n - 1, coupon, discount)",
        "true * bond_value(n - 1, coupon, discount)",
    );
    let outcomes = run_surf(&source, "induction-only");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Error, "{outcomes:#?}");
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("type-checked compiler AST"))
    );
    assert!(outcomes[0].induction_evidence.is_none());
}

#[cfg(feature = "smt")]
#[test]
fn induction_accepts_compiler_inlined_alias_recursion_soundly() {
    let source = "module M
def recur_alias(n: i32, coupon: f64, discount: f64) -> f64 =
  bond_value(n, coupon, discount)
def bond_value(n: i32, coupon: f64, discount: f64) -> f64 =
  if (n <= 0) then cast(1.0, f64)
  else coupon + discount * recur_alias(n - 1, coupon, discount)
@property bond_value_nonnegative forall(n: i32, coupon: f64, discount: f64)
where n >= 0, coupon >= cast(0.0, f64), discount >= cast(0.0, f64):
  (bond_value(n, coupon, discount) >= cast(0.0, f64))
";
    let outcomes = run_surf(source, "induction-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:#?}");
    let evidence = outcome.induction_evidence.as_ref().expect("evidence");
    assert_eq!(evidence.base.status, "proved");
    assert_eq!(evidence.step.status, "proved");
}

#[cfg(feature = "smt")]
#[test]
fn literal_dead_exact_recursion_keeps_its_branch_in_the_solver_goal() {
    let source = GENERAL_BOND_INDUCTION.replace(
        "coupon + discount * bond_value(n - 1, coupon, discount)",
        "if true then coupon else coupon + discount * bond_value(n - 1, coupon, discount)",
    );
    let outcomes = run_surf(&source, "induction-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:#?}");
    let evidence = outcome.induction_evidence.as_ref().expect("evidence");
    assert_eq!(evidence.base.status, "proved");
    assert_eq!(evidence.step.status, "proved");
    assert!(
        format!("{:?}", evidence.step.goal.postcondition).contains("BoolLit(true)"),
        "literal branch must remain in the dispatched goal: {evidence:#?}"
    );
}

const NAMED_PROPERTY: &str = "module M
def double(x: f32) -> f32 = x + x
@property double_is_even forall(x: f32):
  (double(x) == x + x)
";

#[test]
fn u4_discovers_differently_named_property_no_hardcoded_name() {
    // The discovery must find a property whose name is NOT `property`.
    let outcomes = run_surf(NAMED_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property discovered: {outcomes:?}");
    assert_eq!(outcomes[0].name, "double_is_even");
    assert!(
        outcomes[0].is_pass(),
        "a true property passes: {:?}",
        outcomes[0]
    );
}

const MULTIPLE_PROPERTIES: &str = "module M
@property always_true forall(x: f32):
  (x == x)
@property also_true forall(y: f32):
  (y + 0.0 == y)
";

#[test]
fn u4_discovers_multiple_properties() {
    let outcomes = run_surf(MULTIPLE_PROPERTIES, "auto");
    assert_eq!(outcomes.len(), 2, "two properties: {outcomes:?}");
    assert!(outcomes.iter().all(|o| o.is_pass()));
}

const NO_PROPERTY: &str = "module M
def f(x: f32) -> f32 = x + 1.0
";

#[test]
fn u4_no_property_is_empty() {
    let outcomes = run_surf(NO_PROPERTY, "auto");
    assert!(outcomes.is_empty(), "no @property => no outcomes");
}

const FAILING_PROPERTY: &str = "module M
@property false_claim forall(x: f32):
  (x > x)
";

#[test]
fn u4_failing_property_is_not_pass() {
    let outcomes = run_surf(FAILING_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Failed);
    assert!(!outcomes[0].is_pass());
}

#[test]
fn fuzz_counterexample_records_accepted_shrink_steps() {
    let outcomes = run_surf(
        "module M
@property too_strong forall(x: f32):
  (x > 5.0)
",
        "fuzz-only",
    );
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert!(
        outcome.shrink_steps > 0,
        "fuzz failure should record accepted shrink steps: {outcome:?}"
    );
    assert_eq!(
        outcome
            .counterexample
            .as_ref()
            .and_then(|cx| cx["x"].as_f64()),
        Some(0.0),
        "zero is still a failing counterexample for x > 5.0: {outcome:?}"
    );
}

#[cfg(feature = "smt")]
const INLINE_SCALAR_GRAD_PROPERTY: &str = "module M
@property inline_grad_negative forall(d: f32, r: f32, g: f32)
where d > 0.5, r > g, r < 9.5:
  (grad(fn (dd: f32, rr: f32, gg: f32) ->
    (dd / (rr - gg)), wrt=rr)(d, r, g) < 0.0)
";

#[cfg(feature = "smt")]
#[test]
fn checked_record_call_reaches_smt_without_losing_its_fields() {
    let outcomes = run_surf(
        "module M\n\
         type Box = | Box { x: f32, y: f32 }\n\
         def sum_fields(p: Box) -> f32 = {\n\
           z = add(p.x, p.y)\n\
           z\n\
         }\n\
         @property record_identity forall(x: f32, y: f32):\n\
           sum_fields(Box { x, y }) == add(x, y)\n\
         @property wrong_field forall(x: f32, y: f32):\n\
           sum_fields(Box { x, y }) == add(x, x)\n",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 2, "{outcomes:?}");
    assert_eq!(outcomes[0].status, PropertyStatus::Passed, "{outcomes:?}");
    assert_eq!(outcomes[0].proof_tier, PropertyTier::Smt, "{outcomes:?}");
    assert_eq!(outcomes[0].samples, 0, "{outcomes:?}");
    assert_eq!(outcomes[1].status, PropertyStatus::Failed, "{outcomes:?}");
    assert_eq!(outcomes[1].proof_tier, PropertyTier::Smt, "{outcomes:?}");
    assert_eq!(outcomes[1].samples, 0, "{outcomes:?}");
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_inline_lambda_reaches_smt() {
    let outcomes = run_surf(INLINE_SCALAR_GRAD_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
    assert_eq!(outcome.samples, 0, "{outcome:?}");
    assert!(
        !outcome.assumptions.is_empty()
            && outcome.assumptions.iter().all(|assumption| {
                assumption
                    .non_vacuity
                    .as_ref()
                    .is_some_and(|record| record.status == NonVacuityStatus::Established)
            }),
        "the gradient proof must retain established non-vacuity: {outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_matches_closed_form_derivative() {
    let outcomes = run_surf(
        "module M
@property quotient_grad_formula forall(d: f32, r: f32, g: f32)
where r > g:
  (grad(fn (dd: f32, rr: f32, gg: f32) ->
    (dd / (rr - gg)), wrt=rr)(d, r, g)
    == (0.0 - d) / ((r - g) * (r - g)))
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_reversed_claim_is_disproved_by_smt() {
    let source = INLINE_SCALAR_GRAD_PROPERTY.replace("< 0.0)", "> 0.0)");
    let outcomes = run_surf(&source, "smt-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(outcome.samples, 0, "{outcome:?}");
    assert!(outcome.counterexample.is_some(), "{outcome:?}");
}

#[cfg(feature = "smt")]
const NAMED_SCALAR_GRAD_PROPERTY: &str = "module M
def gap(rr: f32, gg: f32) -> f32 = rr - gg
def quotient_value(dd: f32, rr: f32, gg: f32) -> f32 = {
  denominator = gap(rr, gg)
  dd / denominator
}
@property named_grad_negative forall(d: f32, r: f32, g: f32)
where d > 0.5, r > g, r < 9.5:
  (grad(quotient_value, wrt=rr)(d, r, g) < 0.0)
";

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_named_function_handles_helpers_and_blocks() {
    let outcomes = run_surf(NAMED_SCALAR_GRAD_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_cast_fails_closed_with_specific_reason() {
    let outcomes = run_surf(
        "module M
def cast_value(x: f32) -> f32 = cast(x * x, f32)
@property cast_grad forall(x: f32):
  (grad(cast_value, wrt=x)(x) >= 0.0)
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert_eq!(
        outcome.reason.as_deref(),
        Some("scalar grad SMT lowering does not support casts in differentiated bodies"),
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_conditional_fails_closed_with_specific_reason() {
    let outcomes = run_surf(
        "module M
def conditional_value(x: f32) -> f32 =
  if x > 0.0 then x * x else 0.0 - x
@property conditional_grad forall(x: f32):
  (grad(conditional_value, wrt=x)(x) >= 0.0)
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.reason.as_deref(),
        Some("scalar grad SMT lowering does not support conditionals"),
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_implicit_wrt_fails_closed_with_specific_reason() {
    let outcomes = run_surf(
        "module M
@property implicit_grad forall(x: f32):
  (grad(fn (xx: f32) -> xx * xx)(x) >= 0.0)
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert_eq!(
        outcomes[0].reason.as_deref(),
        Some("scalar grad SMT lowering requires exactly one explicit `wrt` parameter")
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_nested_transform_fails_closed_with_specific_reason() {
    let outcomes = run_surf(
        "module M
@property nested_grad forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert_eq!(
        outcomes[0].reason.as_deref(),
        Some("scalar grad SMT lowering does not support nested gradients")
    );
}

#[cfg(feature = "smt")]
#[test]
fn valid_nested_grad_properties_reach_their_own_smt_boundary() {
    let outcomes = run_surf(
        "module M
@property nested_grad_left forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
@property nested_grad_right forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx + xx, wrt=xx), wrt=xx)(x) >= 0.0)
",
        "smt-only",
    );

    assert_eq!(outcomes.len(), 2, "{outcomes:#?}");
    for outcome in outcomes {
        assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
        assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
        assert_eq!(
            outcome.reason.as_deref(),
            Some("scalar grad SMT lowering does not support nested gradients"),
            "{outcome:?}"
        );
    }
}

#[cfg(feature = "smt")]
#[test]
fn only_selected_nested_grad_ignores_valid_unsupported_sibling() {
    let source = "module M
@property nested_grad_left forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
@property nested_grad_right forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx + xx, wrt=xx), wrt=xx)(x) >= 0.0)
";
    let options = PropertyRunOptions {
        tier: "smt-only".to_string(),
        only: Some("nested_grad_left".to_string()),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(outcomes) =
        run_surf_source_properties(source, &options).expect("run filtered nested gradients");

    assert_eq!(outcomes.len(), 1, "{outcomes:#?}");
    let outcome = &outcomes[0];
    assert_eq!(
        outcome.name, "nested_grad_left",
        "filter chose the wrong property"
    );
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(
        outcome.reason.as_deref(),
        Some("scalar grad SMT lowering does not support nested gradients"),
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn aliased_nested_grad_reason_matches_generated_deep_without_mislabeling_ordinary_calls() {
    const ALIAS_REASON: &str = "scalar grad SMT lowering cannot resolve function `inner`";
    const GENERIC_REASON: &str = "property does not lower to Tier B (smt-only)";

    let surf = "module M
def square(xx: f32) -> f32 = xx * xx
inner = grad(square, wrt=xx)
@property alias_nested forall(x: f32):
  (grad(inner, wrt=xx)(x) >= 0.0)
@property ordinary_alias_call forall(x: f32):
  (inner(x) >= 0.0)
";
    let declarations = chelis_surf::parser::parse_str(surf).expect("alias fixture parses");
    let deep = chelis_deep::printer::print_canonical(
        &chelis_surf::desugar::desugar_program(&declarations).expect("alias fixture desugars"),
    );
    let all = [
        ("alias_nested", ALIAS_REASON),
        ("ordinary_alias_call", GENERIC_REASON),
    ];
    let alias_only = [("alias_nested", ALIAS_REASON)];
    let ordinary_only = [("ordinary_alias_call", GENERIC_REASON)];
    let cases = [
        ("surf-all", surf, false, None, all.as_slice()),
        (
            "surf-alias-only",
            surf,
            false,
            Some("alias_nested"),
            alias_only.as_slice(),
        ),
        (
            "surf-ordinary-only",
            surf,
            false,
            Some("ordinary_alias_call"),
            ordinary_only.as_slice(),
        ),
        ("deep-all", deep.as_str(), true, None, all.as_slice()),
        (
            "deep-alias-only",
            deep.as_str(),
            true,
            Some("alias_nested"),
            alias_only.as_slice(),
        ),
        (
            "deep-ordinary-only",
            deep.as_str(),
            true,
            Some("ordinary_alias_call"),
            ordinary_only.as_slice(),
        ),
    ];

    for (case, source, is_deep, only, expected) in cases {
        let options = PropertyRunOptions {
            tier: "smt-only".to_string(),
            only: only.map(str::to_string),
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let result = if is_deep {
            run_deep_source_properties(source, &options)
        } else {
            run_surf_source_properties(source, &options)
        };
        let PropertyRunResult::Ran(outcomes) = result.expect("run alias fixture");

        assert_eq!(outcomes.len(), expected.len(), "{case}: {outcomes:#?}");
        for (outcome, (expected_name, expected_reason)) in outcomes.iter().zip(expected) {
            assert_eq!(outcome.name, *expected_name, "{case}: {outcome:#?}");
            assert_eq!(
                outcome.status,
                PropertyStatus::Unsupported,
                "{case}: {outcome:#?}"
            );
            assert_eq!(
                outcome.proof_tier,
                PropertyTier::Smt,
                "{case}: {outcome:#?}"
            );
            assert_eq!(
                outcome.reason.as_deref(),
                Some(*expected_reason),
                "{case}: {outcome:#?}"
            );
        }
    }
}

#[cfg(feature = "smt")]
#[test]
fn selected_nested_grad_with_unknown_outer_wrt_is_an_error() {
    let outcomes = run_surf(
        "module M
@property invalid_nested_grad forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=missing)(x) >= 0.0)
",
        "smt-only",
    );

    assert_eq!(outcomes.len(), 1, "{outcomes:#?}");
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Error, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::None, "{outcome:?}");
    assert!(
        outcome.reason.as_deref().is_some_and(|reason| {
            reason.contains("unknown `grad` parameter") && reason.contains("missing")
        }),
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn unrelated_invalid_grad_declaration_fails_closed_before_scalar_smt_property() {
    let outcomes = run_surf(
        "module M
def pair(x: f32) -> (f32, f32) = (x, x)
bad = grad(pair, wrt=missing)
@property reflexive forall(x: f32):
  (x == x)
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Error, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::None, "{outcome:?}");
    assert!(!outcome.is_pass(), "{outcome:?}");
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("grad") && reason.contains("missing"),
        "the unrelated desugar failure must remain visible: {outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn colliding_property_spans_preserve_invalid_nested_sibling_during_smt_routing() {
    fn collide_property_spans(decls: &mut [Decl]) {
        for decl in decls {
            match decl {
                Decl::Module { decls, .. } => collide_property_spans(decls),
                Decl::Property { span, .. } => *span = chelis_deep::Span::new(0, 0),
                _ => {}
            }
        }
    }

    let parsed = chelis_surf::parser::parse_str(
        "module Inner
def pair(x: f32) -> (f32, f32) = (x, x)
@property invalid_sibling forall(x: f32):
  (grad(pair, wrt=missing)(x) == x)
@property reflexive forall(x: f32):
  (x == x)
",
    )
    .expect("parse collision fixture");
    let mut module_decls = vec![Decl::Module {
        name: "Outer".to_string(),
        decls: parsed,
        span: chelis_deep::Span::new(0, 0),
    }];
    collide_property_spans(&mut module_decls);
    let flat = flatten_module_decls(&module_decls);
    let options = PropertyRunOptions {
        tier: "smt-only".to_string(),
        only: Some("reflexive".to_string()),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };

    let PropertyRunResult::Ran(outcomes) =
        run_surf_decls_properties(&flat, &flat, &module_decls, &options)
            .expect("run collision fixture");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Error, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::None, "{outcome:?}");
    assert!(!outcome.is_pass(), "{outcome:?}");
    let reason = outcome.reason.as_deref().unwrap_or_default();
    assert!(
        reason.contains("grad") && reason.contains("missing"),
        "the colliding invalid sibling must remain visible: {outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn scalar_grad_helper_depth_overflow_fails_closed_with_specific_reason() {
    let outcomes = run_surf(
        "module M
def h4(x: f32) -> f32 = x * x
def h3(x: f32) -> f32 = h4(x)
def h2(x: f32) -> f32 = h3(x)
def h1(x: f32) -> f32 = h2(x)
@property deep_grad forall(x: f32):
  (grad(h1, wrt=x)(x) >= 0.0)
",
        "smt-only",
    );
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert_eq!(
        outcomes[0].reason.as_deref(),
        Some("scalar grad SMT lowering exceeded the helper inlining depth")
    );
}

#[cfg(feature = "smt")]
const UNSUPPORTED_SCALAR_GRAD_PROPERTY: &str = "module M
@property exp_grad_positive forall(x: f32)
where x > 0.5, x < 9.5:
  (grad(fn (xx: f32) -> exp(xx), wrt=xx)(x) > 0.0)
";

#[cfg(feature = "smt")]
#[test]
fn unsupported_scalar_grad_operation_has_specific_prompt_reason() {
    let outcomes = run_surf(UNSUPPORTED_SCALAR_GRAD_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{outcome:?}");
    assert_eq!(outcome.samples, 0, "{outcome:?}");
    assert_eq!(
        outcome.reason.as_deref(),
        Some("scalar grad SMT lowering does not support call `exp`"),
        "{outcome:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn unsupported_scalar_grad_operation_still_falls_to_fuzz_under_auto() {
    let outcomes = run_surf(UNSUPPORTED_SCALAR_GRAD_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Fuzz, "{outcome:?}");
    assert_eq!(
        outcome.composite_verdict,
        CompositeVerdict::FuzzValidatedEmpirical,
        "{outcome:?}"
    );
}

const UNKNOWN_CONTRACT_PROPERTY: &str = r#"module M
@property unknown_contract forall(x: f32):
  x == x
  with contract = "std.normal_cdf.not_a_contract"
"#;

#[test]
fn contract_unknown_id_is_unsupported() {
    let outcomes = run_surf(UNKNOWN_CONTRACT_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("unknown contract"),
        "reason should name unknown contract: {:?}",
        outcomes[0]
    );
}

const NO_CALL_CONTRACT_PROPERTY: &str = r#"module M
@property no_contract_call forall(x: f32):
  x == x
  with contract = "std.normal_cdf.reflection"
"#;

#[test]
fn contract_without_bound_call_is_unsupported() {
    let outcomes = run_surf(NO_CALL_CONTRACT_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("did not bind"),
        "reason should report no binding: {:?}",
        outcomes[0]
    );
}

const MISMATCHED_CONTRACT_PROPERTY: &str = r#"module M
def other_cdf(x: f32) -> f32 = x
@property wrong_binding forall(x: f32):
  other_cdf(0.0 - x) == 1.0 - other_cdf(x)
  with contract = "std.normal_cdf.reflection"
"#;

#[test]
fn contract_bound_to_wrong_call_is_unsupported() {
    let outcomes = run_surf(MISMATCHED_CONTRACT_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert!(
        outcomes[0]
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("did not bind"),
        "reason should report binding mismatch: {:?}",
        outcomes[0]
    );
}

#[cfg(feature = "smt")]
const REFLECTION_CONTRACT_PROPERTY: &str = r#"module M
def pkg__chelis__std__Std__Contracts__normal_cdf(x: f32) -> f32 = x
@property reflected forall(x: f32):
  pkg__chelis__std__Std__Contracts__normal_cdf(-x) == 1.0 - pkg__chelis__std__Std__Contracts__normal_cdf(x)
  with contract = "std.normal_cdf.reflection"
"#;

#[cfg(feature = "smt")]
#[test]
fn local_linker_shaped_normal_cdf_does_not_receive_std_contract() {
    let outcomes = run_surf(REFLECTION_CONTRACT_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(
        outcome.status,
        PropertyStatus::Unsupported,
        "a local spoofed linker-shaped normal_cdf must not bind std contract assumptions: {outcome:?}"
    );
    assert!(
        outcome
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("did not bind"),
        "reason should report missing trusted std binding: {outcome:?}"
    );
}

const INJECTION_PROPERTY: &str = "module M.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
def prob_value(p: Probability) -> f32 = p.value
@property bounded forall(p: Probability):
  (prob_value(p) <= 1.0)
";

#[test]
fn u4_property_with_opaque_invariant_binder_uses_injection() {
    // A property whose binder is an invariant-carrying opaque type is
    // verified ONLY over invariant-satisfying values (assumption injection);
    // `prob_value(p) <= 1.0` holds for every in-domain Probability, so it
    // passes, and the outcome is marked injected.
    let outcomes = run_surf(INJECTION_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert_eq!(outcomes[0].name, "bounded");
    assert!(
        outcomes[0].is_pass(),
        "injected property passes: {:?}",
        outcomes[0]
    );
    assert!(outcomes[0].injected, "verified through the injection path");
}

#[test]
fn wi8_injected_binder_assumption_carries_prover_stamped_discharge_tier() {
    // WI-8: the binder-matched (injected) assumption carries a prover-stamped
    // discharge tier recording which engine discharged it and with what
    // guarantee, keyed to the binder's source identity. c-earchin emits only
    // source identity; the tier is the prover's stamp.
    let outcomes = run_surf(INJECTION_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert!(outcomes[0].injected);

    let binder_assumption = outcomes[0]
        .assumptions
        .iter()
        .find(|a| a.name == "invariant:Probability:binder:p")
        .unwrap_or_else(|| panic!("injected binder assumption present: {:?}", outcomes[0]));

    let tier = binder_assumption
        .discharge_tier
        .as_ref()
        .expect("injected assumption carries a prover-stamped discharge tier");
    assert_eq!(
        tier.engine, "fuzz-sampler",
        "discharged by the fuzz sampler"
    );
    assert_eq!(tier.guarantee, "fuzz", "with the fuzz guarantee kind");
    // The tier is keyed to the binder's source identity (the same id the
    // assumption carries), so the artifact can join tier -> source.
    assert_eq!(
        tier.source.as_deref(),
        Some("invariant:Probability:binder:p"),
        "the tier is keyed to the binder source identity"
    );
}

const INJECTION_FALSE_PROPERTY: &str = "module M.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
def prob_value(p: Probability) -> f32 = p.value
@property too_strong forall(p: Probability):
  (prob_value(p) <= 0.5)
";

#[test]
fn u4_injection_does_not_hide_a_false_property() {
    // Injection restricts to invariant-valid binders, but a property false
    // ON those binders still fails (a Probability with value 0.9 is valid yet
    // violates `<= 0.5`).
    let outcomes = run_surf(INJECTION_FALSE_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Failed);
    assert!(!outcomes[0].is_pass());
}

#[test]
fn u4_statistically_validated_zero_samples_is_not_pass() {
    // A Passed outcome with zero fuzz samples (the vacuous/timeout sentinel)
    // is NOT a genuine pass.
    let zero = PropertyOutcome::new(
        "p",
        PropertyStatus::Passed,
        PropertyTier::Fuzz,
        0,
        0,
        None,
        None,
        false,
        Vec::new(),
    );
    assert!(!zero.is_pass(), "Passed with 0 fuzz samples is not a pass");
    // A Passed SMT proof carries 0 samples but IS a pass. chelis#422: a green
    // SMT base MUST carry its discharge (covered-or-rejected) -- here the cvc5
    // over-reals discharge (SoundApproximate + RealArith), which projects to
    // `proven_modulo_real_arithmetic`.
    let smt = PropertyOutcome::with_base_discharge(
        "p",
        PropertyStatus::Passed,
        PropertyTier::Smt,
        0,
        0,
        None,
        None,
        false,
        Vec::new(),
        Some((
            crate::discharge::Soundness::SoundApproximate,
            crate::discharge::QualifierSet::from_iter_kinds([
                crate::discharge::Qualifier::RealArith,
            ]),
        )),
    );
    assert!(
        smt.is_pass(),
        "an SMT-proved property is a pass at 0 samples"
    );
    assert_eq!(
        smt.composite_verdict,
        CompositeVerdict::ProvenModuloRealArithmetic,
        "an over-reals SMT proof discloses the machine-arith gap"
    );
}

// --- F7 / chelis#507: the deep property path honors the --tier contract ---

fn run_deep(source: &str, tier: &str) -> Vec<PropertyOutcome> {
    let opts = PropertyRunOptions {
        tier: tier.to_string(),
        samples: 16,
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(o) = run_deep_source_properties(source, &opts).expect("run");
    o
}

/// A canonical Deep user `@property` (the form `chelis deep` emits): the
/// always-true `x >= x`.
const DEEP_TRUE_PROPERTY: &str = r#"(module {}
  m
  (defsig {} always_true (t-fn {} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}),
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_source_kind: "user"
       }
    always_true
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {}
        (var {} gte)
        (var {} x)
        (var {} x)))))
"#;

#[test]
fn stamped_deep_property_discovery_never_silently_returns_zero() {
    let exprs = chelis_deep::parser::parse_and_stamp_file(DEEP_TRUE_PROPERTY)
        .expect("canonical Deep property stamps");
    assert!(
        matches!(exprs.first(), Some(DeepExpr::Node(..))),
        "the file ingress must exercise the stamped Node representation: {exprs:#?}"
    );

    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "fuzz-only");
    assert_eq!(
        outcomes.len(),
        1,
        "a stamped user property must produce one outcome, never a silent empty success"
    );
}

#[test]
fn f7_deep_fuzz_only_runs_the_fuzz_loop() {
    // `--tier fuzz-only` on a deep property runs the fuzz loop and passes a
    // true property with samples > 0.
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "fuzz-only");
    assert_eq!(outcomes.len(), 1, "one deep property: {outcomes:?}");
    assert_eq!(outcomes[0].name, "always_true");
    assert!(
        outcomes[0].is_pass(),
        "deep fuzz-only passes: {:?}",
        outcomes[0]
    );
    assert!(outcomes[0].samples > 0, "fuzz-only collected samples");
    assert_eq!(outcomes[0].proof_tier, PropertyTier::Fuzz);
}

#[test]
fn issue_978_deep_induction_only_is_terminal_without_sampling() {
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "induction-only");
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.status, PropertyStatus::Unsupported, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);
    assert_eq!(outcome.samples, 0);
    assert_eq!(outcome.attempted_samples, 0);
    assert_eq!(outcome.accepted_samples, 0);
    assert!(outcome.sampling_method.is_none());
    assert!(!outcome.is_pass());
}

#[test]
#[cfg(feature = "smt")]
fn f7_deep_smt_only_uses_tier_b_not_fuzz() {
    // chelis#507: a Deep property body reaches the same Tier-B SMT lane as
    // the equivalent Surf property. `smt-only` must not report the legacy
    // "no Tier B path" unsupported verdict and must not run fuzz samples.
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "smt-only");
    assert_eq!(outcomes.len(), 1);
    assert_eq!(
        outcomes[0].status,
        PropertyStatus::Passed,
        "deep smt-only should prove through Tier B: {:?}",
        outcomes[0]
    );
    assert_eq!(outcomes[0].samples, 0, "smt-only ran no fuzz samples");
    assert_eq!(outcomes[0].proof_tier, PropertyTier::Smt);
    assert!(outcomes[0].is_pass());
}

#[test]
#[cfg(feature = "smt")]
fn f7_deep_auto_prefers_tier_b_when_deep_goal_lowers() {
    // Under `auto`, an SMT-amenable Deep property should take the proof lane,
    // not trail into fuzz.
    let outcomes = run_deep(DEEP_TRUE_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1);
    assert!(
        outcomes[0].is_pass(),
        "deep auto passes via SMT: {:?}",
        outcomes[0]
    );
    assert_eq!(outcomes[0].samples, 0);
    assert_eq!(outcomes[0].proof_tier, PropertyTier::Smt);
}

#[cfg(feature = "smt")]
const DEEP_BOOL_CONNECTIVE_PROPERTIES: &str = r#"(module {}
  m
  (defsig {} conj (t-fn {} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}),
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_source_kind: "user"
       }
    conj
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {}
        (var {} and)
        (app {} (var {} gte) (var {} x) (var {} x))
        (app {} (var {} lte) (var {} x) (var {} x)))))
  (defsig {} disj (t-fn {} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}),
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_source_kind: "user"
       }
    disj
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {}
        (var {} or)
        (app {} (var {} gte) (var {} x) (var {} x))
        (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0))))))
"#;

#[test]
#[cfg(feature = "smt")]
fn f7_deep_boolean_connectives_lower_to_tier_b() {
    let outcomes = run_deep(DEEP_BOOL_CONNECTIVE_PROPERTIES, "smt-only");
    assert_eq!(outcomes.len(), 2, "two deep properties: {outcomes:?}");
    for outcome in outcomes {
        assert_eq!(
            outcome.status,
            PropertyStatus::Passed,
            "boolean connective property proves: {outcome:?}"
        );
        assert_eq!(outcome.proof_tier, PropertyTier::Smt);
        assert_eq!(outcome.samples, 0);
    }
}

// --- F6: the deep discoverer classifies source kind like the CLI ---

/// A `chelis_role: "property"` def with NO `property_source_kind` is a
/// malformed property: the shared discoverer must ERROR (matching the CLI
/// discoverer), not silently skip it (which previously made tide report
/// total:0 / ok:true while the CLI errored).
#[test]
fn f6_deep_chelis_role_property_without_source_kind_is_error() {
    let source = r#"(module {}
  m
  (def {chelis_role: "property",
         property_quantifiers: (params {} (x {type: (t-prim {} f32)}))}
    nameless_kind
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let opts = PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME);
    let result = run_deep_source_properties(source, &opts);
    assert!(
        result.is_err(),
        "a chelis_role property with no property_source_kind must error, not be skipped: {result:?}"
    );
    assert!(
        result.unwrap_err().contains("property_source_kind"),
        "the error names the missing metadata"
    );
}

/// A `chelis_role: "property"` def with an INVALID `property_source_kind`
/// is likewise an error (matching the CLI), not silently skipped.
#[test]
fn f6_deep_invalid_source_kind_is_error() {
    let source = r#"(module {}
  m
  (def {chelis_role: "property",
         property_source_kind: "bogus",
         property_quantifiers: (params {} (x {type: (t-prim {} f32)}))}
    bad_kind
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let opts = PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME);
    let result = run_deep_source_properties(source, &opts);
    assert!(
        result.is_err(),
        "an invalid property_source_kind must error: {result:?}"
    );
}

/// A `bridge:c-earchin` property is SKIPPED by the shared runner (the CLI
/// bridge path owns it) -- not an error, not run. The shared runner returns
/// an empty user-property set for a bridge-only module.
#[test]
fn f6_deep_bridge_source_kind_is_skipped_not_error() {
    let source = r#"(module {}
  m
  (def {chelis_role: "property",
         property_source_kind: "bridge:c-earchin",
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_preconditions: (tuple {})}
    bridged
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} gte) (var {} x) (var {} x)))))
"#;
    let opts = PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME);
    let PropertyRunResult::Ran(outcomes) =
        run_deep_source_properties(source, &opts).expect("bridge module is not an error");
    assert!(
        outcomes.is_empty(),
        "the shared runner skips bridge:c-earchin properties: {outcomes:?}"
    );
}

// --- WI-8 tier-coverage invariant (WS-5 Part B): every AssumptionRecord
// produced on a green discharge path carries a prover-stamped discharge_tier ---

/// Assert the WI-8 coverage invariant on one outcome: every assumption it
/// carries has a prover-stamped discharge tier. Returns the number of
/// assumptions checked so the caller can assert the path actually produced
/// some (a vacuous zero-assumption pass must not silently satisfy the
/// invariant).
fn assert_assumptions_are_tiered(outcome: &PropertyOutcome) -> usize {
    for assumption in &outcome.assumptions {
        assert!(
            assumption.discharge_tier.is_some(),
            "green-path assumption `{}` on property `{}` is missing a prover-stamped \
             discharge_tier (WI-8): {assumption:?}",
            assumption.name,
            outcome.name,
        );
    }
    outcome.assumptions.len()
}

/// A green fuzz property whose `where` precondition drives the
/// `fuzz_precondition_assumptions` discharge site: the pass carries one
/// `preconditions:*` assumption that must be tiered.
///
/// The body must hold at every `f32` the guard admits, because the subject
/// here is the assumption's tier stamp and not the property's content. The
/// original body was `x + 1.0 > x`, which `f32` first falsifies at `x = 2^24`,
/// where the ulp becomes `2.0` and adding exactly half of it ties to even.
/// Through `[2^24, 2^25)` it alternates with the mantissa parity --
/// `16777216` absorbs, `16777218` rounds up -- and from `2^25` upward the ulp
/// exceeds `2.0` and every representable value absorbs. Tier C passed it
/// only because the sampler never proposed a magnitude that large (chelis#3636);
/// `x * 2.0 > x` is total over the guard, overflowing to `inf` at `f32::MAX`
/// rather than absorbing. The absorption itself is now covered by
/// `issue_3636_f32_absorption_at_large_magnitudes_is_refuted`.
const GREEN_FUZZ_PRECONDITION_PROPERTY: &str = "module M
@property guarded forall(x: f32) where x > 0.0:
  (x * 2.0 > x)
";

#[test]
fn wi8_green_fuzz_precondition_assumptions_carry_discharge_tier() {
    let outcomes = run_surf(GREEN_FUZZ_PRECONDITION_PROPERTY, "fuzz-only");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert!(
        outcomes[0].is_pass(),
        "the guarded property passes via fuzz: {:?}",
        outcomes[0]
    );
    let checked = assert_assumptions_are_tiered(&outcomes[0]);
    assert_eq!(
        checked, 1,
        "the green fuzz precondition path emits exactly one tiered assumption: {:?}",
        outcomes[0]
    );
}

#[test]
fn wi8_green_injection_binder_assumptions_carry_discharge_tier() {
    // The injection green path (a binder-matched invariant) emits a binder
    // assumption that must also be tiered. INJECTION_PROPERTY is defined above.
    let outcomes = run_surf(INJECTION_PROPERTY, "auto");
    assert_eq!(outcomes.len(), 1, "one property: {outcomes:?}");
    assert!(outcomes[0].is_pass(), "injected property passes");
    assert!(outcomes[0].injected, "verified through the injection path");
    let checked = assert_assumptions_are_tiered(&outcomes[0]);
    assert!(
        checked >= 1,
        "the injection green path emits at least one tiered assumption: {:?}",
        outcomes[0]
    );
}

#[test]
fn wi8_smt_precondition_discharge_site_stamps_discharge_tier() {
    // The SMT green discharge site (`property_assumption_records`) is only
    // reached at runtime when Tier B returns Proved, which the default (non-smt)
    // build cannot do (solve_property is a stub). Lock the site directly with a
    // unit call: a discharge with a non-empty precondition set must yield an
    // AssumptionRecord carrying a prover-stamped discharge_tier keyed to the
    // precondition source identity. This is race-free (no process-global env
    // override) and does not require the cvc5 manual gate.
    let smt_prop = crate::tier_b::SmtProperty {
        variables: vec![("x".to_string(), crate::solver::SmtSort::Real)],
        preconditions: vec![crate::solver::SmtExpr::BoolLit(true)],
        postcondition: crate::solver::SmtExpr::BoolLit(true),
    };
    let discharge = AssumptionDischarge::new(
        DischargeMethod::Smt,
        serde_json::json!({"solver": "cvc5", "result": "proved"}),
    );
    let non_vacuity = NonVacuityRecord::established(serde_json::json!({"result": "sat"}));
    let records = property_assumption_records("guarded", &smt_prop, discharge, non_vacuity);
    assert_eq!(
        records.len(),
        1,
        "a non-empty precondition set produces one assumption record"
    );
    let tier = records[0]
        .discharge_tier
        .as_ref()
        .expect("the SMT discharge site stamps a prover-side discharge tier (WI-8)");
    assert_eq!(tier.engine, "cvc5", "discharged by the SMT engine");
    assert_eq!(tier.guarantee, "smt", "with the smt guarantee kind");
    assert_eq!(
        tier.source.as_deref(),
        Some("preconditions:guarded"),
        "keyed to the precondition source identity"
    );

    // Negative companion: an empty precondition set produces no assumption,
    // so there is nothing to tier (the site is vacuous, not silently untiered).
    let empty = crate::tier_b::SmtProperty {
        variables: Vec::new(),
        preconditions: Vec::new(),
        postcondition: crate::solver::SmtExpr::BoolLit(true),
    };
    let none = property_assumption_records(
        "p",
        &empty,
        AssumptionDischarge::new(DischargeMethod::Smt, serde_json::json!({})),
        NonVacuityRecord::established(serde_json::json!({})),
    );
    assert!(
        none.is_empty(),
        "no preconditions => no assumption record to tier"
    );
}

#[test]
fn producer_property_marker_has_no_discovery_authority() {
    let source = r#"(def {c_earchin_role: "property_witness", property_source_kind: "user"} ordinary (fn {} (params {}) (lit {type: (t-prim {} bool)} false)))"#;
    let PropertyRunResult::Ran(outcomes) = run_deep_source_properties(
        source,
        &PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME),
    )
    .unwrap();
    assert!(outcomes.is_empty(), "{outcomes:?}");
}

/// The Beacon route reads a pipe as the scalar bridge only when its last stage
/// is the bare name `tensor_to_scalar`. A pipe that ends anywhere else, or in
/// the zero-argument call `tensor_to_scalar()`, is not that bridge.
#[test]
fn beacon_pipe_goal_must_end_in_bare_tensor_to_scalar() {
    let reason_for = |goal: &str| {
        let source = format!(
            "def neuron(x: tensor[f64]) -> tensor[f64] = relu(x)\n\
             @property bounded_neuron forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:\n  \
             (({goal}) <= 2.0f64)\n"
        );
        let outcomes = run_surf(&source, "beacon-only");
        assert_eq!(outcomes.len(), 1, "{outcomes:#?}");
        assert_eq!(outcomes[0].proof_tier, PropertyTier::Beacon);
        outcomes[0].reason.clone().unwrap_or_default()
    };
    let not_bridged = "Beacon output must use tensor_to_scalar on its scalar graph output";
    for goal in [
        "x |> neuron |> tensor_to_scalar |> abs",
        "x |> neuron |> tensor_to_scalar()",
    ] {
        assert_eq!(reason_for(goal), not_bridged, "{goal}");
    }
    assert_ne!(
        reason_for("x |> neuron |> tensor_to_scalar"),
        not_bridged,
        "the canonical pipe goal reaches past the bridge matcher"
    );
}

// --- chelis#2965: contract verdicts resolve at the consumer's float widths ---

fn widths_of(source: &str) -> Vec<Prim> {
    let decls = flatten_module_decls(&chelis_surf::parser::parse_str(source).expect("parses"));
    let properties = collect_surf_properties(&decls, &decls, None).expect("properties");
    property_float_widths(&decls, &properties[0])
}

#[test]
fn chelis_2965_f32_binders_resolve_contracts_at_f32_only() {
    let widths = widths_of(
        "module M\n@property p forall(x: f32):\n  (x <= x)\n  with contract = \"std.normal_cdf.range\"\n",
    );
    assert_eq!(widths, vec![Prim::F32]);
}

#[test]
fn chelis_2965_every_spelled_width_in_reachable_functions_is_a_consumer_width() {
    // An f64 binder narrowed to f32 inside a reached helper evaluates the
    // contract-bound call at f32 as well, so the f32 verdict must apply.
    let widths = widths_of(
        "module M\ndef narrow(x: f64) -> f32 = cast(x, f32)\n@property p forall(x: f64):\n  (narrow(x) <= narrow(x))\n  with contract = \"std.normal_cdf.range\"\n",
    );
    assert!(
        widths.contains(&Prim::F64) && widths.contains(&Prim::F32),
        "{widths:?}"
    );
    // An unreached helper does not widen the set.
    let widths = widths_of(
        "module M\ndef unused(x: f64) -> f16 = cast(x, f16)\n@property p forall(x: f64):\n  (x <= x)\n  with contract = \"std.normal_cdf.range\"\n",
    );
    assert_eq!(widths, vec![Prim::F64]);
}

#[test]
fn chelis_2965_spelled_widths_cover_suffixes_and_the_unsuffixed_literal_default() {
    assert_eq!(spelled_float_widths("x: f64"), vec![Prim::F64]);
    assert_eq!(
        spelled_float_widths("1.5bf16 + 2f16"),
        vec![Prim::Bf16, Prim::F16]
    );
    assert_eq!(spelled_float_widths("add(x, 1.0)"), vec![Prim::F32]);
    assert_eq!(spelled_float_widths("1e-3"), vec![Prim::F32]);
    assert!(spelled_float_widths("xf32 + count_f64 + 3").is_empty());
}

// --- chelis#2957: a property resolves only the contracts it names ---

fn resolved_rows_of(source: &str) -> Vec<String> {
    let decls = flatten_module_decls(&chelis_surf::parser::parse_str(source).expect("parses"));
    let properties = collect_surf_properties(&decls, &decls, None).expect("properties");
    crate::contracts::RESOLVED_ROWS.with(|rows| rows.borrow_mut().clear());
    contract_assumptions(&decls, &properties[0], &chelis_std_bundle::EMBEDDED_RUNTIME)
        .expect("contracts resolve");
    crate::contracts::RESOLVED_ROWS.with(|rows| rows.borrow().clone())
}

#[test]
fn chelis_2957_a_property_without_a_float_contract_resolves_no_float_discharge() {
    assert!(resolved_rows_of("module M\n@property p forall(x: f32):\n  (x <= x)\n").is_empty());
    assert_eq!(
        resolved_rows_of(
            "module M\n@property p forall(x: f64):\n  (x <= x)\n  with contract = \"std.quantile.range\"\n"
        ),
        vec!["std.quantile.range@untyped".to_string()]
    );
    // A float contract resolves at the property's widths only, with the
    // range contract reflection implies.
    assert_eq!(
        resolved_rows_of(
            "module M\n@property p forall(x: f32):\n  (x <= x)\n  with contract = \"std.normal_cdf.reflection\"\n"
        ),
        vec![
            "std.normal_cdf.range@f32".to_string(),
            "std.normal_cdf.reflection@f32".to_string()
        ]
    );
}

/// Trusted linked contract implementations, shaped as the linker emits them,
/// so the contract abstraction binds calls without a Reef build.
const TRUSTED_CONTRACT_IMPLEMENTATIONS: &str = "module Linked
def pkg__chelis__std__Std__Contracts__normal_cdf(x: f64) -> f64 = x
def pkg__nautilus__Nautilus__Stats__quantile_vec[n](v: &tensor[n, f32], q: f32) -> f32 = q
";

/// The solver variables the Tier B lowering of `source`'s only property
/// declares (binders, then contract-abstraction symbols), and the names the
/// abstraction minted.
fn tier_b_contract_variables(source: &str) -> (Vec<String>, Vec<String>) {
    let decls = flatten_module_decls(&chelis_surf::parser::parse_str(source).expect("parses"));
    let trusted = flatten_module_decls(
        &chelis_surf::parser::parse_str(TRUSTED_CONTRACT_IMPLEMENTATIONS).expect("parses"),
    );
    let properties = collect_surf_properties(&decls, &decls, None).expect("properties");
    let [property] = properties.as_slice() else {
        panic!("exactly one property: {properties:?}");
    };
    let semantic = SemanticSurfProgram::new(&decls).expect("semantic program");
    let lowered =
        lower_surf_tier_b_goal(&semantic, &trusted, property, &expanded_contracts(property))
            .expect("the fixture's contract symbols are fresh");
    assert!(
        lowered.postcondition.is_some() && lowered.preconditions.is_some(),
        "the fixture lowers to Tier B"
    );
    let minted: Vec<String> = lowered
        .abstraction
        .variables()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let declared = property
        .params
        .iter()
        .map(|param| param.name.clone())
        .chain(minted.iter().cloned())
        .collect();
    (declared, minted)
}

fn assert_declared_once(declared: &[String]) {
    let unique: BTreeSet<&String> = declared.iter().collect();
    assert_eq!(
        unique.len(),
        declared.len(),
        "every solver variable is declared once: {declared:?}"
    );
}

#[test]
fn chelis_3236_normal_cdf_symbol_avoids_a_binder_spelled_like_it() {
    let (declared, minted) = tier_b_contract_variables(
        "module M
@property forged forall(x: f64, __contract_std_normal_cdf_0: f64):
  ((pkg__chelis__std__Std__Contracts__normal_cdf(x) <= 1.0) && (__contract_std_normal_cdf_0 <= 1.0))
  with contract = \"std.normal_cdf.range\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(minted, vec!["__contract_std_normal_cdf_0_1".to_string()]);
}

#[test]
fn chelis_3236_normal_cdf_symbol_avoids_a_module_name_spelled_like_it() {
    // A free module-level name reaches the goal as an undeclared variable;
    // a minted symbol of the same spelling would silently define it.
    let (declared, minted) = tier_b_contract_variables(
        "module M
__contract_std_normal_cdf_0 = 5.0
@property forged forall(x: f64):
  ((pkg__chelis__std__Std__Contracts__normal_cdf(x) <= 1.0) && (__contract_std_normal_cdf_0 <= 1.0))
  with contract = \"std.normal_cdf.range\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(minted, vec!["__contract_std_normal_cdf_0_1".to_string()]);
}

#[test]
fn chelis_3236_normal_cdf_symbols_keep_their_spelling_without_a_collision() {
    let (declared, minted) = tier_b_contract_variables(
        "module M
@property parity forall(x: f64):
  ((pkg__chelis__std__Std__Contracts__normal_cdf(x) + pkg__chelis__std__Std__Contracts__normal_cdf(-x)) <= 2.0)
  with contract = \"std.normal_cdf.range\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(
        minted,
        vec![
            "__contract_std_normal_cdf_0".to_string(),
            "__contract_std_normal_cdf_1".to_string()
        ]
    );
}

#[test]
fn chelis_3236_quantile_symbols_avoid_binders_spelled_like_them() {
    let (declared, minted) = tier_b_contract_variables(
        "module M
def observations() -> tensor[3, f32] =
  (to_tensor([3.0f32, 1.0f32, 2.0f32]) : tensor[3, f32])
@property forged forall(p: f32, q: f32, __contract_std_quantile_0: f32, __contract_std_quantile_1: f32)
  where 0.0 <= p, p <= q, q <= 1.0:
  ((pkg__nautilus__Nautilus__Stats__quantile_vec(observations(), p) <= pkg__nautilus__Nautilus__Stats__quantile_vec(observations(), q)) && (__contract_std_quantile_0 <= __contract_std_quantile_1))
  with contract = \"std.quantile.monotonicity\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(
        minted,
        vec![
            "__contract_std_quantile_0_1".to_string(),
            "__contract_std_quantile_1_1".to_string()
        ]
    );
}

/// Red-team F1: a module name that occurs ONLY as an abstracted call's
/// operand is removed from the lowered body with the call, but the
/// monotonicity assumptions re-emit it. It must still be reserved.
#[test]
fn chelis_3236_normal_cdf_symbol_avoids_a_name_used_only_as_a_call_argument() {
    let (declared, minted) = tier_b_contract_variables(
        "module M
__contract_std_normal_cdf_0 = 5.0
@property forged forall(x: f64) where x >= 1.0:
  (pkg__chelis__std__Std__Contracts__normal_cdf(__contract_std_normal_cdf_0) <= pkg__chelis__std__Std__Contracts__normal_cdf(x))
  with contract = \"std.normal_cdf.range\"
  with contract = \"std.normal_cdf.monotonicity\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(
        minted,
        vec![
            "__contract_std_normal_cdf_0_1".to_string(),
            "__contract_std_normal_cdf_1".to_string()
        ]
    );

    let (declared, minted) = tier_b_contract_variables(
        "module M
__contract_std_normal_cdf_1 = 5.0
@property forged_one forall(x: f64) where x >= 1.0:
  (pkg__chelis__std__Std__Contracts__normal_cdf(x) >= pkg__chelis__std__Std__Contracts__normal_cdf(__contract_std_normal_cdf_1))
  with contract = \"std.normal_cdf.range\"
  with contract = \"std.normal_cdf.monotonicity\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(
        minted,
        vec![
            "__contract_std_normal_cdf_0".to_string(),
            "__contract_std_normal_cdf_1_1".to_string()
        ]
    );
}

#[test]
fn chelis_3236_quantile_symbol_avoids_a_name_used_only_as_a_q_operand() {
    let (declared, minted) = tier_b_contract_variables(
        "module M
__contract_std_quantile_0 = 0.5
def observations() -> tensor[3, f32] =
  (to_tensor([3.0f32, 1.0f32, 2.0f32]) : tensor[3, f32])
@property forged forall(q: f32)
  where 0.0 <= q, q <= 1.0:
  (pkg__nautilus__Nautilus__Stats__quantile_vec(observations(), __contract_std_quantile_0) <= pkg__nautilus__Nautilus__Stats__quantile_vec(observations(), q))
  with contract = \"std.quantile.monotonicity\"
",
    );
    assert_declared_once(&declared);
    assert_eq!(
        minted,
        vec![
            "__contract_std_quantile_0_1".to_string(),
            "__contract_std_quantile_1".to_string()
        ]
    );
}

/// The image of a goal `<first> <= x` whose abstracted call carried operand
/// `u`, with `first` minted under the given name.
fn contract_goal_image(first: &str, minted: &str) -> TierBGoalImage {
    use crate::solver::{CmpOp, SmtExpr};
    TierBGoalImage {
        postcondition: SmtExpr::Cmp(
            CmpOp::Le,
            Box::new(SmtExpr::Var(first.to_string())),
            Box::new(SmtExpr::Var("x".to_string())),
        ),
        preconditions: vec![],
        contract_preconditions: vec![SmtExpr::Cmp(
            CmpOp::Le,
            Box::new(SmtExpr::Var("u".to_string())),
            Box::new(SmtExpr::Var("x".to_string())),
        )],
        contract_operands: vec![SmtExpr::Var("u".to_string())],
        minted: vec![minted.to_string()],
    }
}

#[test]
fn chelis_3236_post_check_rejects_a_symbol_that_captured_a_user_name() {
    let discovery = contract_goal_image("placeholder", "placeholder");
    // A fresh symbol is accepted.
    check_contract_symbols_fresh(&[], &discovery, &contract_goal_image("m", "m"))
        .expect("a fresh symbol passes");
    // A symbol spelled like the user operand `u` renames that operand too,
    // so the passes no longer agree: rejected, never solved.
    let reason = check_contract_symbols_fresh(&[], &discovery, &contract_goal_image("u", "u"))
        .expect_err("a captured operand fails closed");
    assert!(reason.contains("#3236"), "{reason}");
    // A symbol spelled like a binder is rejected even if the binder is unused.
    let decls = flatten_module_decls(
        &chelis_surf::parser::parse_str("module M\n@property p forall(m: f64):\n  true\n")
            .expect("parses"),
    );
    let properties = collect_surf_properties(&decls, &decls, None).expect("properties");
    check_contract_symbols_fresh(
        &properties[0].params,
        &discovery,
        &contract_goal_image("m", "m"),
    )
    .expect_err("a captured binder fails closed");
}

#[cfg(feature = "smt")]
#[test]
fn chelis_3236_induction_symbols_avoid_a_binder_spelled_like_them() {
    // `coupon` renamed to the induction lane's value-symbol stem: the binder
    // stays its own variable, so the valid proof still holds and the false
    // variant is still refuted by its step obligation.
    let renamed = GENERAL_BOND_INDUCTION.replace("coupon", "__chelis_induction_value");
    let outcome = &run_surf(&renamed, "induction-only")[0];
    assert_eq!(outcome.status, PropertyStatus::Passed, "{outcome:#?}");
    assert_eq!(outcome.proof_tier, PropertyTier::Induction);

    let false_renamed = renamed.replace(
        "__chelis_induction_value + discount * bond_value",
        "__chelis_induction_value - cast(1.0, f64) + discount * bond_value",
    );
    let outcome = &run_surf(&false_renamed, "induction-only")[0];
    assert_eq!(outcome.status, PropertyStatus::Failed, "{outcome:#?}");
    let evidence = outcome
        .induction_evidence
        .as_ref()
        .expect("the failed step must remain visible");
    assert_eq!(evidence.step.status, "disproved");
}

// chelis#3267: every root the fuzz tier declares in the user's program is
// named fresh against it, so a module that defines a probe's plain spelling
// gets exactly the verdict of the same module with that definition renamed.

/// A valid and a false sampled property, beside a definition of `name`.
fn probe_root_property_module(name: &str) -> String {
    format!(
        "module Probe.Fuzz
{name} = true
@property holds forall(x: f32) where x >= 0.0f32, x <= 1.0f32:
  (x <= 2.0f32)
@property too_strong forall(x: f32) where x >= 0.0f32, x <= 1.0f32:
  (x <= 0.5f32)
"
    )
}

/// An opaque binder whose invariant band (0.5 +- 0.0005) starves rejection
/// sampling, so the binder comes from the constructor tier, and whose
/// constant `eps` is resolved by evaluation. `w` holds; `w_false` does not.
fn probe_root_opaque_module(name: &str) -> String {
    format!(
        "module M
export (norm, prob_value)
@opaque
@invariant(p) ((p.value >= (0.5 - eps)) && (p.value <= (0.5 + eps)))
type T =
  | T {{ value: f32 }}
def eps() -> f32 = 0.0005
def norm(x: f32) -> T = T {{ value: 0.5 }}
def prob_value(p: T) -> f32 = p.value
{name} = true
@property w forall(p: T):
  (prob_value(p) <= 0.5005)
@property w_false forall(p: T):
  (prob_value(p) <= 0.4)
"
    )
}

fn assert_same_verdict(defined: &PropertyOutcome, control: &PropertyOutcome) {
    assert_eq!(defined.name, control.name);
    assert_eq!(defined.status, control.status, "{defined:#?}\n{control:#?}");
    assert_eq!(defined.proof_tier, control.proof_tier, "{defined:#?}");
    assert_eq!(defined.samples, control.samples, "{defined:#?}");
    assert_eq!(
        defined.counterexample, control.counterexample,
        "{defined:#?}"
    );
    assert_eq!(defined.reason, control.reason, "{defined:#?}");
}

fn assert_probe_root_verdicts(outcomes: &[PropertyOutcome], control: &[PropertyOutcome]) {
    assert_eq!(outcomes.len(), control.len(), "{outcomes:#?}");
    for (defined, control) in outcomes.iter().zip(control) {
        assert_same_verdict(defined, control);
    }
    let holds = &control[0];
    assert_eq!(holds.status, PropertyStatus::Passed, "{holds:#?}");
    assert_eq!(holds.proof_tier, PropertyTier::Fuzz, "{holds:#?}");
    let refuted = &outcomes[1];
    assert_eq!(refuted.status, PropertyStatus::Failed, "{refuted:#?}");
    assert!(refuted.counterexample.is_some(), "{refuted:#?}");
}

#[test]
fn chelis_3267_sample_roots_avoid_a_module_definition_of_their_spelling() {
    let control = run_surf(&probe_root_property_module("renamed_probe"), "fuzz-only");
    for name in ["__chelis_property_probe", "__chelis_property_pre"] {
        let outcomes = run_surf(&probe_root_property_module(name), "fuzz-only");
        assert_probe_root_verdicts(&outcomes, &control);
    }
}

#[test]
fn chelis_3267_deep_sample_roots_avoid_a_module_definition_of_their_spelling() {
    let run_deep = |surf: &str| {
        let decls = chelis_surf::parser::parse_str(surf).expect("parse Surf fixture");
        let program = chelis_surf::desugar::desugar_program(&decls).expect("desugar fixture");
        let opts = PropertyRunOptions {
            tier: "fuzz-only".to_string(),
            samples: 32,
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let source = chelis_deep::printer::print_canonical(&program);
        let PropertyRunResult::Ran(outcomes) =
            run_deep_source_properties(&source, &opts).expect("run Deep properties");
        outcomes
    };
    let control = run_deep(&probe_root_property_module("renamed_probe"));
    for name in ["__chelis_property_probe", "__chelis_property_pre"] {
        let outcomes = run_deep(&probe_root_property_module(name));
        assert_probe_root_verdicts(&outcomes, &control);
    }
}

#[test]
fn chelis_3267_injection_roots_avoid_a_module_definition_of_their_spelling() {
    for tier in ["fuzz-only", "auto"] {
        let control = run_surf(&probe_root_opaque_module("renamed_probe"), tier);
        assert_eq!(control[0].status, PropertyStatus::Passed, "{control:#?}");
        assert!(control[0].injected, "{control:#?}");
        for name in [
            "__chelis_gen_probe",
            "__chelis_const_probe",
            "__chelis_prop_probe",
        ] {
            let outcomes = run_surf(&probe_root_opaque_module(name), tier);
            assert_eq!(outcomes.len(), control.len(), "{outcomes:#?}");
            for (defined, control) in outcomes.iter().zip(&control) {
                assert_same_verdict(defined, control);
            }
            let refuted = &outcomes[1];
            assert_eq!(refuted.status, PropertyStatus::Failed, "{refuted:#?}");
            assert!(refuted.counterexample.is_some(), "{refuted:#?}");
        }
    }
}

#[test]
fn semantic_declaration_view_keeps_exact_production_lowering_and_authored_properties() {
    use crate::solver::{CmpOp, SmtExpr};
    let source = "def declared(dummy: i32) -> f64 = 1.1\ndef wide(dummy: i32) -> i64 = 3000000000\n@property true_case forall(x: i32): declared(x) == 1.1f64\n@property false_case forall(x: i32): declared(x) != 1.1f64\n@property wide_case forall(x: i32): wide(x) == 3000000000i64\n";
    let declarations = chelis_surf::parser::parse_str(source).unwrap();
    let original = declarations.clone();
    let properties = collect_surf_properties(&declarations, &declarations, None).unwrap();
    let semantic = SemanticSurfProgram::new(&declarations).unwrap();
    for (property, expected) in properties.iter().zip([
        SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::RealLit(1.1)),
            Box::new(SmtExpr::RealLit(1.1)),
        ),
        SmtExpr::Cmp(
            CmpOp::Ne,
            Box::new(SmtExpr::RealLit(1.1)),
            Box::new(SmtExpr::RealLit(1.1)),
        ),
        SmtExpr::Cmp(
            CmpOp::Eq,
            Box::new(SmtExpr::IntLit(3000000000)),
            Box::new(SmtExpr::IntLit(3000000000)),
        ),
    ]) {
        let view = semantic.property(property).unwrap();
        assert_eq!(view.decl_path, property.decl_path);
        assert_eq!(view.name, property.name);
        assert_eq!(view.samples, property.samples);
        assert_eq!(view.contracts, property.contracts);
        let lowered = lower_surf_tier_b_goal(&semantic, &[], property, &[]).unwrap();
        assert_eq!(lowered.postcondition, Some(expected));
    }
    assert_eq!(
        declarations, original,
        "analysis never rewrites authored declarations"
    );
}

#[test]
fn scalar_grad_literals_share_finalized_values_with_ordinary_lowering() {
    use crate::solver::{CmpOp, SmtExpr};
    for (dtype, literal, expected) in [
        ("f32", "1.1f32", f64::from(1.1f32)),
        ("f32", "1e-45f32", f64::from(f32::from_bits(1))),
        ("f32", "1e-46f32", 0.0),
        ("f64", "1.1f64", 1.1),
    ] {
        let source = format!(
            "def coefficient(x: {dtype}) -> {dtype} = x * {literal}\ndef helper(x: {dtype}) -> {dtype} = coefficient(x)\n@property positive forall(x: {dtype}): grad(coefficient, wrt=x)(x) == {literal}\n@property negative forall(x: {dtype}): grad(coefficient, wrt=x)(x) != {literal}\n@property nested forall(x: {dtype}): grad(helper, wrt=x)(x) == {literal}\n@property inline forall(x: {dtype}): grad(fn (xx: {dtype}) -> xx * {literal}, wrt=xx)(x) == {literal}\n"
        );
        let declarations = chelis_surf::parser::parse_str(&source).unwrap();
        let original = declarations.clone();
        let properties = collect_surf_properties(&declarations, &declarations, None).unwrap();
        let semantic = SemanticSurfProgram::new(&declarations).unwrap();
        for (property, comparison) in
            properties
                .iter()
                .zip([CmpOp::Eq, CmpOp::Ne, CmpOp::Eq, CmpOp::Eq])
        {
            let goal = lower_surf_tier_b_goal(&semantic, &[], property, &[]).unwrap();
            assert_eq!(
                goal.postcondition,
                Some(SmtExpr::Cmp(
                    comparison,
                    Box::new(SmtExpr::RealLit(expected)),
                    Box::new(SmtExpr::RealLit(expected)),
                )),
                "{source}"
            );
        }
        assert_eq!(declarations, original);
        let outcomes = run_surf(&source, "fuzz-only");
        assert_eq!(outcomes.len(), 4, "{outcomes:?}");
        for (outcome, expected) in outcomes.iter().zip([
            PropertyStatus::Passed,
            PropertyStatus::Failed,
            PropertyStatus::Passed,
            PropertyStatus::Passed,
        ]) {
            assert_eq!(outcome.status, expected, "{outcome:?}");
        }
    }
}

#[test]
fn declared_f64_guard_interval_executes_through_default_enabled_fuzz_runner() {
    let source = "def declared(dummy: i32) -> f64 = 1.1\n@property exact forall(x: f64) where x >= declared(0), x <= 1.1f64:\n  x == 1.1f64\n@property false_twin forall(x: f64) where x >= declared(0), x <= 1.1f64:\n  x != 1.1f64\n";
    let declarations = chelis_surf::parser::parse_str(source).unwrap();
    let properties = collect_surf_properties(&declarations, &declarations, None).unwrap();
    let outcomes = run_surf(source, "fuzz-only");
    assert_eq!(outcomes[0].status, PropertyStatus::Passed, "{outcomes:?}");
    assert_eq!(outcomes[1].status, PropertyStatus::Failed, "{outcomes:?}");
    assert_eq!(outcomes[0].accepted_samples, 32);
    assert_eq!(outcomes[0].attempted_samples, 32);
    assert_eq!(
        outcomes[0].sampling_method.as_deref(),
        Some("constraint_directed")
    );
    for (property, outcome) in properties.iter().zip(&outcomes) {
        assert_eq!(
            outcome.goal.as_ref().unwrap(),
            &chelis_surf::format::format_proposition(
                &property.params,
                &property.preconditions,
                &property.body
            )
        );
    }
}

#[test]
fn nested_grad_sibling_preserves_selected_numeric_fuzz_guards() {
    let source = "@property nested_grad forall(x: f32):\n  grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0\ndef declared(dummy: i32) -> f64 = 1.3\n@property exact forall(x: f64) where x >= declared(0), x <= 1.3f64:\n  x == 1.3f64\n@property false_twin forall(x: f64) where x >= declared(0), x <= 1.3f64:\n  x != 1.3f64\n";
    for (name, expected) in [
        ("exact", PropertyStatus::Passed),
        ("false_twin", PropertyStatus::Failed),
    ] {
        let options = PropertyRunOptions {
            tier: "fuzz-only".into(),
            only: Some(name.into()),
            samples: 3,
            seed: 83,
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let PropertyRunResult::Ran(outcomes) =
            run_surf_source_properties(source, &options).unwrap();
        assert_eq!(outcomes.len(), 1, "{outcomes:#?}");
        let outcome = &outcomes[0];
        assert_eq!(outcome.name, name);
        assert_eq!(outcome.status, expected, "{outcome:#?}");
        assert_eq!(outcome.proof_tier, PropertyTier::Fuzz, "{outcome:#?}");
        assert!(outcome.accepted_samples > 0, "{outcome:#?}");
        assert_eq!(outcome.rejected_samples, 0, "{outcome:#?}");
        assert_eq!(
            outcome.sampling_method.as_deref(),
            Some("constraint_directed")
        );
    }
}

#[test]
fn induction_plan_lowers_declared_helpers_from_the_cached_semantic_view() {
    use crate::solver::{CmpOp, SmtExpr};
    for (dtype, literal, expected) in [
        ("f64", "1.1", SmtExpr::RealLit(1.1)),
        ("i64", "3000000000", SmtExpr::IntLit(3000000000)),
    ] {
        let source = format!(
            "def declared(dummy: i32) -> {dtype} = {literal}\ndef model(n: i32) -> {dtype} = if n <= 0 then declared(n) else model(n - 1)\n@property positive forall(n: i32) where n >= 0:\n  model(n) == {literal}{dtype}\n@property negative forall(n: i32) where n >= 0:\n  model(n) != {literal}{dtype}\n"
        );
        let declarations = chelis_surf::parser::parse_str(&source).unwrap();
        let semantic = SemanticSurfProgram::new(&declarations).unwrap();
        let properties = collect_surf_properties(&declarations, &declarations, None).unwrap();
        for (property, comparison) in properties.iter().zip([CmpOp::Eq, CmpOp::Ne]) {
            let plan = classify_surf_induction(&declarations, property, &semantic).unwrap();
            assert_eq!(plan.base_expr, expected);
            assert!(
                matches!(plan.proposition, SmtExpr::Cmp(found, _, ref right) if found == comparison && right.as_ref() == &expected)
            );
        }
    }
}

/// chelis#3635: the integer witness bound is the width's REPRESENTABLE range,
/// not `integer_fuzz_bounds`'s `[-1000, 1000]` convenience window.
///
/// The first version of this lane used the fuzz window, so a guard admitting
/// `n > 4000i32` declined its own witness with "not representable at i32" --
/// a false statement about a value `i32` represents perfectly well. The paired
/// negative keeps the real bound honest: a value genuinely outside the width
/// still fails, and says so accurately.
#[test]
fn issue3635_integer_witness_bound_is_the_width_not_the_fuzz_window() {
    fn param(name: &str, type_name: &str) -> Param {
        Param {
            name: name.to_string(),
            ty: Some(TypeExpr::Named(
                type_name.to_string(),
                chelis_deep::Span::new(0, 0),
            )),
            span: chelis_deep::Span::new(0, 0),
        }
    }

    // Inside i32 but far outside the fuzz window: accepted, and carried
    // through at its exact value rather than clamped to 1000.
    let sample = witness_sample(&[param("n", "i32")], &serde_json::json!({"n": 4001}))
        .expect("4001 is representable at i32");
    assert_eq!(sample.values.len(), 1);
    assert_eq!(sample.values[0].json, serde_json::json!(4001));

    // The i32 extremes are admitted, which the fuzz window would have refused.
    for edge in [i64::from(i32::MIN), i64::from(i32::MAX)] {
        let sample = witness_sample(&[param("n", "i32")], &serde_json::json!({"n": edge}))
            .unwrap_or_else(|reason| panic!("{edge} is representable at i32: {reason}"));
        assert_eq!(sample.values[0].json, serde_json::json!(edge));
    }

    // Outside the declared width: declined, with the width's real range named
    // and no claim that a representable value is unrepresentable.
    let reason = witness_sample(&[param("n", "i8")], &serde_json::json!({"n": 4001}))
        .expect_err("4001 does not fit i8");
    assert!(
        reason.contains("representable range"),
        "the reason names the real bound: {reason}"
    );
    assert!(
        reason.contains("[-128, 127]"),
        "the reason quotes i8's actual range: {reason}"
    );
    witness_sample(&[param("n", "i8")], &serde_json::json!({"n": 127})).expect("127 fits i8");
}

// --- chelis#3636: the proposal window stays narrow -----------------------

/// The edge schedule reads a binder's *admissible* range while a
/// post-schedule random draw reads the narrower *proposal window*. That
/// distinction is the change's scope boundary: widening the window is
/// chelis#644, not this. Nothing observable through the CLI can see it, so it
/// is pinned here.
///
/// `x > 0.0f64` leaves the upper side unbounded, so the admissible range runs
/// to `f64::MAX` while the window synthesis anchors 20 wide at the guard. A
/// draw taken past the end of the schedule must land in the window.
#[test]
fn issue_3636_post_schedule_draw_stays_inside_the_proposal_window() {
    use crate::solver::{CmpOp, SmtExpr};

    let precondition = SmtExpr::Cmp(
        CmpOp::Gt,
        Box::new(SmtExpr::Var("x".to_string())),
        Box::new(SmtExpr::RealLit(0.0)),
    );
    let plan = ConstraintSamplingPlan::derive(
        [("x".to_string(), FloatKind::F64)],
        std::slice::from_ref(&precondition),
    )
    .expect("a one-sided f64 guard is sampleable");

    // The window the derivation chose, and the admissible range the schedule
    // reads, must actually differ -- otherwise this test proves nothing.
    let window = plan.domains["x"];
    let range = plan.ranges["x"];
    assert_eq!(window.lower.value, 0.0);
    assert_eq!(window.upper.value, 20.0, "the historical 20-wide window");
    assert_eq!(range.upper.value, f64::MAX, "the admissible range");

    let schedule = float_edge_candidates(
        FloatKind::F64,
        FloatKind::F64.lower_value(range.lower.value, range.lower.strict),
        FloatKind::F64.upper_value(range.upper.value, range.upper.strict),
        &[],
    );
    assert!(
        schedule.iter().any(|value| *value > 20.0),
        "the schedule must leave the window, or the two ranges are not distinct"
    );

    // Every attempt past the schedule is a random draw, and every one of
    // them belongs to the window.
    let mut rng = Lcg::new(7);
    for attempt in schedule.len()..schedule.len() + 64 {
        let drawn = plan
            .sample(&mut rng, attempt, 100)
            .expect("the guard admits samples")["x"];
        assert!(
            drawn > 0.0 && drawn <= 20.0,
            "attempt {attempt} drew {drawn}, outside the proposal window (0, 20]"
        );
    }
}

/// The same binder inside the schedule: those attempts are expected to leave
/// the window, which is what makes the test above a boundary and not a cap.
#[test]
fn issue_3636_scheduled_attempts_may_leave_the_proposal_window() {
    use crate::solver::{CmpOp, SmtExpr};

    let precondition = SmtExpr::Cmp(
        CmpOp::Gt,
        Box::new(SmtExpr::Var("x".to_string())),
        Box::new(SmtExpr::RealLit(0.0)),
    );
    let plan = ConstraintSamplingPlan::derive(
        [("x".to_string(), FloatKind::F64)],
        std::slice::from_ref(&precondition),
    )
    .expect("a one-sided f64 guard is sampleable");

    let mut rng = Lcg::new(7);
    let outside = (0..48)
        .filter_map(|attempt| plan.sample(&mut rng, attempt, 100).ok())
        .filter(|values| values["x"] > 20.0)
        .count();
    assert!(
        outside > 0,
        "the schedule must propose magnitudes the window cannot reach"
    );
}

// --- chelis#3697: the off-diagonal phase's exact contract ----------------
//
// Three properties of the pairing decide whether it is safe, and none is
// observable through a CLI property: that a representative is always a value
// the diagonal already proposes, that the cells leave arity one alone,
// and that the cells run out so a long run regains its uniform draws.

/// Every representative comes from the binder's own candidate list, there are
/// at most `EXTREMAL_CLASSES` of them, and they are distinct by bit pattern.
///
/// The first of those is the safety property: a representative is in the
/// binder's resolved range by construction, so a cell can no more escape a
/// guard — or propose a value the diagonal would not — than the diagonal can.
#[test]
fn issue_3697_representatives_are_a_deduplicated_subset_of_the_schedule() {
    for (lower, upper) in [
        (-f64::MAX, f64::MAX),
        (f64::from_bits(1), f64::MAX),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, -1.0),
        (0.0, 0.0),
    ] {
        let candidates = float_edge_candidates(FloatKind::F64, lower, upper, &[]);
        let reps = extremal_class_representatives(&candidates);
        assert!(
            reps.len() <= EXTREMAL_CLASSES,
            "[{lower}, {upper}] produced {} representatives",
            reps.len()
        );
        assert!(!reps.is_empty(), "[{lower}, {upper}] produced none");
        for value in &reps {
            assert!(
                candidates.iter().any(|c| c.to_bits() == value.to_bits()),
                "[{lower}, {upper}] invented {value}, which is not a candidate"
            );
        }
        let distinct = reps
            .iter()
            .map(|value| value.to_bits())
            .collect::<BTreeSet<_>>();
        assert_eq!(distinct.len(), reps.len(), "[{lower}, {upper}]: {reps:?}");
    }
    // The comparator choice is observable in exactly one place: a range whose
    // only candidates are the two zeros, where `total_cmp` orders `-0.0`
    // below `+0.0` and so decides which zero is digit 0. `f64::max`/`min`
    // leave that unspecified, which is why the selection does not use them,
    // and asserting only subset/dedup above would not notice.
    let zeros = float_edge_candidates(FloatKind::F64, 0.0, 0.0, &[]);
    assert_eq!(
        zeros.len(),
        2,
        "both signed zeros are candidates: {zeros:?}"
    );
    let reps = extremal_class_representatives(&zeros);
    assert_eq!(reps.len(), 2, "{reps:?}");
    assert!(
        reps[0].is_sign_positive() && reps[1].is_sign_negative(),
        "`total_cmp` puts `+0.0` first, so it is digit 0: {reps:?}"
    );
}

/// The two roles, named. An unguarded `f64` binder's extremes are the finite
/// extrema, and `total_cmp` decides the order deterministically where
/// `f64::max`/`min` would not.
#[test]
fn issue_3697_unguarded_binder_takes_the_two_extremes() {
    let candidates = float_edge_candidates(FloatKind::F64, -f64::MAX, f64::MAX, &[]);
    let reps = extremal_class_representatives(&candidates);
    assert_eq!(reps, vec![f64::MAX, -f64::MAX], "{reps:?}");
}

/// Under a guard that admits no negative value the smallest candidate *is*
/// the smallest positive one, which is why chelis#3697's own reproduction —
/// a large `x` over a subnormal `y` — is reachable from two classes.
#[test]
fn issue_3697_a_positive_guard_makes_the_smallest_a_subnormal() {
    let lower = FloatKind::F64.next_up(0.0);
    let candidates = float_edge_candidates(FloatKind::F64, lower, f64::MAX, &[]);
    let reps = extremal_class_representatives(&candidates);
    assert_eq!(reps, vec![f64::MAX, f64::from_bits(1)], "{reps:?}");
    assert!(
        reps.iter().all(|value| *value > 0.0),
        "`x > 0.0` admits nothing else: {reps:?}"
    );
}

/// A single-binder property takes no cell: its attempt ordinal indexes the
/// schedule one-for-one, which is chelis#3636's behaviour unchanged.
#[test]
fn issue_3697_arity_one_takes_no_cell() {
    for attempt in 0..128 {
        assert_eq!(
            schedule_slot(attempt, 1, 0, 44, 100),
            ScheduleSlot::Diagonal(attempt),
            "attempt {attempt}"
        );
    }
}

/// The diagonal completes at its original one-for-one rate before any cell is
/// proposed, and the cells then take the attempts that would have been
/// uniform draws.
///
/// At radix two and arity two there are `2^2 = 4` cells, so they occupy
/// attempts 44 to 47 and attempt 48 is a uniform draw again. Cell 2 pairs the
/// first binder's largest candidate with the second's smallest — digits 0 and
/// 1 — which is chelis#3697's own reproduction, and so attempt 46, the
/// forty-seventh.
#[test]
fn issue_3697_the_diagonal_completes_before_any_cell() {
    for attempt in 0..DIAGONAL_SPAN {
        for position in 0..2 {
            assert_eq!(
                schedule_slot(attempt, 2, position, 44, 100),
                ScheduleSlot::Diagonal(attempt),
                "attempt {attempt}, position {position}"
            );
        }
    }
    assert_eq!(
        schedule_slot(DIAGONAL_SPAN, 2, 0, 44, 100),
        ScheduleSlot::Cell(0)
    );
    // Cell 2: digit 0 of binder 0 (its largest), digit 1 of binder 1 (its
    // smallest) -- the reproduction's pairing.
    assert_eq!(
        schedule_slot(DIAGONAL_SPAN + 2, 2, 0, 44, 100),
        ScheduleSlot::Cell(0)
    );
    assert_eq!(
        schedule_slot(DIAGONAL_SPAN + 2, 2, 1, 44, 100),
        ScheduleSlot::Cell(1)
    );
}

/// **A binder with an order predecessor contributes a derivable *bound* to
/// the shared boundary, not an exact length.** Its range, and so its
/// candidate list, is resolved from a value drawn inside the attempt, which
/// does not exist at derive time — `resolve` would index an absent key. But
/// the bound needs no predecessor: a schedule is the exclusion-free list, at
/// most `DIAGONAL_SPAN` long, plus at most two representable neighbours per
/// `!=` guard.
///
/// A boundary larger than a binder's real length is harmless — the binder
/// runs past the end of its list and draws uniformly, as a short schedule
/// already does — so the bound is used directly rather than refined.
///
/// This is what lets an order-guarded property take cells at all: the cells
/// then begin after *every* binder's diagonal, including a successor's, which
/// is why lifting the exemption leaves the chain's refutation where it was.
#[test]
fn issue_3697_an_order_successor_contributes_its_bound() {
    use crate::solver::{CmpOp, SmtExpr};

    let edge = SmtExpr::Cmp(
        CmpOp::Gt,
        Box::new(SmtExpr::Var("y".to_string())),
        Box::new(SmtExpr::Var("x".to_string())),
    );
    let exclusions = (20..26)
        .map(|value| {
            SmtExpr::Cmp(
                CmpOp::Ne,
                Box::new(SmtExpr::Var("y".to_string())),
                Box::new(SmtExpr::RealLit(f64::from(value))),
            )
        })
        .collect::<Vec<_>>();
    let mut preconditions = vec![edge];
    preconditions.extend(exclusions);
    let plan = ConstraintSamplingPlan::derive(
        [
            ("x".to_string(), FloatKind::F64),
            ("y".to_string(), FloatKind::F64),
        ],
        &preconditions,
    )
    .expect("an order edge with exclusions is sampleable");

    assert!(
        !plan.incoming["y"].is_empty(),
        "the shape under test must give `y` a predecessor: {:?}",
        plan.incoming
    );
    let excluded = plan.excluded["y"].len();
    assert_eq!(
        excluded, 6,
        "six `!=` guards reach the plan: {:?}",
        plan.excluded
    );
    // `y` is the successor, so it contributes `DIAGONAL_SPAN + 2k`; `x` has no
    // predecessor and contributes its exact length, which is the span.
    assert_eq!(
        plan.cell_window_start,
        DIAGONAL_SPAN + 2 * excluded,
        "the boundary must clear the successor's bound, or its diagonal is cut"
    );
    // And the derivation must not panic for a chained plan, which is the
    // reason the bound exists rather than a resolve.
    assert!(plan.cell_window_start > DIAGONAL_SPAN);
}

/// **No exclusion-free schedule outruns the span**, which is what lets the
/// typed-generator lane use `DIAGONAL_SPAN` as its whole cell-window
/// boundary instead of pricing every binder's candidate list.
///
/// That lane passes no `!=` exclusions, and exclusions are the only thing
/// that lengthens a schedule past the span — `DIAGONAL_SPAN` is defined as
/// the unguarded `f64` length. If a dtype ever produced a longer one, the
/// lane's windows would reopen the disjointness this boundary exists to
/// close, silently. Hence the assertion rather than the comment.
///
/// Integer schedules are checked through the same bound: they are far
/// shorter, and `int_edge_candidates` ignores exclusions entirely.
#[test]
fn issue_3697_no_unexcluded_schedule_outruns_the_span() {
    // Exhaustive by construction: a new `FloatKind` variant makes this match
    // fail to compile, rather than silently escaping the bound and reopening
    // the disjointness the typed lane's `DIAGONAL_SPAN` boundary relies on.
    let every_kind = |kind: FloatKind| match kind {
        FloatKind::F32 => FloatKind::F32,
        FloatKind::F64 => FloatKind::F64,
    };
    for kind in [FloatKind::F32, FloatKind::F64].map(every_kind) {
        let full = float_edge_candidates(kind, -kind.max_finite(), kind.max_finite(), &[]);
        assert!(
            full.len() <= DIAGONAL_SPAN,
            "{kind:?} unguarded schedule is {} candidates, past the \
             {DIAGONAL_SPAN}-attempt span the typed lane assumes",
            full.len()
        );
        // And a *narrower* guard can only drop candidates, never add them.
        let narrow = float_edge_candidates(kind, -1.0, 1.0, &[]);
        assert!(narrow.len() <= full.len(), "{kind:?}: {narrow:?}");
    }
    // `f64` measures exactly `DIAGONAL_SPAN`, so the bound has zero slack:
    // any added IEEE landmark or ladder rung fails this test rather than
    // quietly outgrowing the typed lane's boundary.
    assert_eq!(
        float_edge_candidates(FloatKind::F64, -f64::MAX, f64::MAX, &[]).len(),
        DIAGONAL_SPAN,
        "the span is defined as this length; if they diverge, decide which moved"
    );
    for prim in ["i8", "i16", "i32", "i64"] {
        let (lower, upper) =
            crate::opaque::int_sample_bounds(prim).expect("an integer width has bounds");
        let candidates = int_edge_candidates(lower, upper);
        assert!(
            candidates.len() <= DIAGONAL_SPAN,
            "{prim} schedule is {} candidates, past the span",
            candidates.len()
        );
    }
}

/// A boundary past the span keeps every binder on the diagonal until it is
/// reached, so the longest schedule's tail is never displaced past the
/// budget.
///
/// This is the exact defect round 2 measured: each `!=` exclusion adds two
/// candidates, and under the interleaved revision a two-binder property with
/// four exclusions per binder (52 candidates) reached its last candidate only
/// at attempt 103, so it reported `fuzz_validated` at the default budget of
/// 100 where it had refuted at 52. The boundary is the maximum over the
/// binders, so the binder that sets it is the one whose tail would be lost.
#[test]
fn issue_3697_a_long_schedule_keeps_its_diagonal() {
    let long = 52;
    for attempt in 0..long {
        assert_eq!(
            schedule_slot(attempt, 2, 0, long, 100),
            ScheduleSlot::Diagonal(attempt),
            "attempt {attempt} under a boundary of {long}"
        );
    }
    // The cells begin where the longest schedule ends, not where the span
    // does -- and for every binder at once, including the short ones.
    assert_eq!(schedule_slot(long, 2, 0, long, 100), ScheduleSlot::Cell(0));
    assert_eq!(schedule_slot(long, 2, 1, long, 100), ScheduleSlot::Cell(0));
}

/// The cells run out, so a run longer than the grid regains the uniform
/// draws they displaced. Without this the window would never be sampled again
/// at any budget.
#[test]
fn issue_3697_cells_exhaust_and_yield_back_to_the_uniform_draw() {
    let cells = EXTREMAL_CLASSES * EXTREMAL_CLASSES;
    assert!(
        cells < (100 - DIAGONAL_SPAN) / CELL_BUDGET_DIVISOR,
        "two binders fit inside the default budget's share"
    );
    assert!(matches!(
        schedule_slot(DIAGONAL_SPAN + cells - 1, 2, 0, 44, 100),
        ScheduleSlot::Cell(_)
    ));
    assert_eq!(
        schedule_slot(DIAGONAL_SPAN + cells, 2, 0, 44, 100),
        ScheduleSlot::Random
    );
}

/// **The phase can never take more than its share of the attempts the
/// diagonal left, at any arity, any budget, and any schedule length.** This
/// is the assertion a fixed cap could not make, and its absence was a
/// default-budget false pass: at 100 samples only 56 attempts follow a
/// 44-candidate diagonal, `2^6 = 64` exceeded them, and a six-binder
/// property proposed no uniform draw at all.
///
/// **`window_start` is an axis here because holding it at 44 is what hid the
/// second instance of the same defect.** The share is divided out of
/// `budget - window_start`, and a revision that divided
/// `budget - DIAGONAL_SPAN` instead passed this test at every budget and
/// arity while starving the generator completely for a boundary of 72: 28
/// cells over the window `[72, 100)` is every attempt there was. A grid that
/// varies one parameter of a claim and fixes another cannot falsify the
/// claim.
///
/// The axis starts at `DIAGONAL_SPAN` because the boundary is a maximum
/// against it and can never be smaller.
#[test]
fn issue_3697_the_phase_never_outgrows_its_share_of_the_budget() {
    for budget in [46usize, 50, 60, 100, 160, 200, 1000] {
        for window_start in [DIAGONAL_SPAN, 45, 52, 72, 90, 150] {
            let left = budget.saturating_sub(window_start);
            let share = left / CELL_BUDGET_DIVISOR;
            for arity in [2usize, 3, 4, 5, 6, 8, 12, 25, 64, usize::MAX] {
                let grid = EXTREMAL_CLASSES
                    .checked_pow(u32::try_from(arity).unwrap_or(u32::MAX))
                    .unwrap_or(usize::MAX);
                let cells = grid.min(share);
                // Whatever ends the phase -- the grid or the share -- the
                // attempt after it is a uniform draw.
                assert_eq!(
                    schedule_slot(window_start + cells, arity, 0, window_start, budget),
                    ScheduleSlot::Random,
                    "budget {budget}, boundary {window_start}, arity {arity}: the \
                     cells must yield by attempt {}, leaving {} of the {left} \
                     attempts the diagonal left to the uniform draw",
                    window_start + cells,
                    left - cells
                );
                // And the uniform draw keeps at least half of them.
                assert!(
                    left - cells >= left / CELL_BUDGET_DIVISOR,
                    "budget {budget}, boundary {window_start}, arity {arity}: \
                     {cells} cells of {left} attempts leaves the generator too \
                     little"
                );
            }
        }
    }
}

/// The share does not cost anything at the arities the published table
/// measures: `2^n` is smaller than half the default budget's post-diagonal
/// attempts up to four binders, so those runs are bit-identical to an
/// uncapped phase and the table's figures stand.
#[test]
fn issue_3697_the_share_is_slack_up_to_four_binders() {
    let share = (100 - DIAGONAL_SPAN) / CELL_BUDGET_DIVISOR;
    for arity in 2..=4 {
        let cells = EXTREMAL_CLASSES.pow(u32::try_from(arity).expect("small arity"));
        assert!(
            cells <= share,
            "arity {arity}: {cells} cells must fit in the {share} the budget allows"
        );
        assert!(
            matches!(
                schedule_slot(DIAGONAL_SPAN + cells - 1, arity, 0, 44, 100),
                ScheduleSlot::Cell(_)
            ),
            "arity {arity}: the whole grid is still reachable"
        );
        assert_eq!(
            schedule_slot(DIAGONAL_SPAN + cells, arity, 0, 44, 100),
            ScheduleSlot::Random,
            "arity {arity}: and the grid, not the share, is what ends it"
        );
    }
    // From five binders the share binds instead of the grid, which is the
    // trade: a prefix of the grid, and the generator still runs.
    let cells = EXTREMAL_CLASSES.pow(5);
    assert!(cells > share, "{cells} cells exceed the {share} allowed");
    assert_eq!(
        schedule_slot(DIAGONAL_SPAN + share, 5, 0, 44, 100),
        ScheduleSlot::Random
    );
}

/// A budget that leaves no room for a cell gets no cell phase, rather than a
/// division by zero or a phase that outlives the run.
#[test]
fn issue_3697_a_budget_at_the_span_takes_no_cell() {
    for budget in [0usize, 1, DIAGONAL_SPAN, DIAGONAL_SPAN + 1] {
        assert_eq!(
            schedule_slot(DIAGONAL_SPAN, 3, 0, 44, budget),
            ScheduleSlot::Random,
            "budget {budget}"
        );
    }
}

/// An absurd arity must not panic. `EXTREMAL_CLASSES.pow(arity)` overflows
/// well before `usize::MAX` binders, and `EXTREMAL_CLASSES.pow(position)`
/// overflows at the same point, so neither the cap nor the digit may be
/// computed that way.
#[test]
fn issue_3697_an_absurd_arity_does_not_overflow() {
    for arity in [25usize, 64, 1024, usize::MAX] {
        let slot = schedule_slot(DIAGONAL_SPAN + 7, arity, 0, 44, 100);
        assert!(matches!(slot, ScheduleSlot::Cell(_)), "arity {arity}");
        // A high-order digit is simply zero: the odometer has not reached it.
        assert_eq!(
            schedule_slot(DIAGONAL_SPAN + 7, arity, 24, 44, 100),
            ScheduleSlot::Cell(0),
            "arity {arity}"
        );
    }
}

/// The integer schedule's prefix is its class set, and it is a prefix of the
/// same list the diagonal walks rather than a separate construction.
#[test]
fn issue_3697_integer_representatives_are_the_schedules_prefix() {
    let candidates = int_edge_candidates(-1000, 1000);
    assert!(candidates.len() > EXTREMAL_CLASSES, "{candidates:?}");
    for digit in 0..EXTREMAL_CLASSES {
        assert_eq!(
            scheduled_int(ScheduleSlot::Cell(digit), &candidates),
            Some(candidates[digit]),
            "digit {digit} of {candidates:?}"
        );
    }
    // A shorter list wraps rather than indexing past its end.
    let narrow = int_edge_candidates(0, 1);
    assert_eq!(narrow, vec![0, 1], "{narrow:?}");
    assert_eq!(
        scheduled_int(ScheduleSlot::Cell(5), &narrow),
        Some(narrow[5 % narrow.len()])
    );
}

/// The anti-drift guard for chelis#3697's P1: `takes_schedule_slot` and
/// `sample_value` must agree about which binders read the schedule.
///
/// They are two places stating one fact, and when they disagreed the cost was
/// not a crash but a silent loss of the feature: a `bool` binder switched the
/// odometer on while ignoring its own slot, so a property with one
/// *scheduled* binder beside it paid the halved diagonal rate for no pairing
/// benefit, and a `bool` declared first consumed digit 0 and multiplied every
/// useful cell index by the radix -- which pushed a two-float-binder
/// refutation from attempt 62 to attempt 362, past the default budget.
///
/// So the agreement is checked by *observation* rather than by reading the
/// match arms: for every type the sampler supports, draw twice from an
/// identically seeded generator at two different slots and see whether the
/// value actually moved. The types are parsed from Surf rather than
/// constructed, so the spellings under test are the ones a property can
/// actually declare.
#[test]
fn issue_3697_schedule_participation_matches_the_sampler() {
    let spellings = [
        "bool",
        "string",
        "i8",
        "i16",
        "i32",
        "i64",
        "f32",
        "f64",
        "tensor[2, f32]",
        "tensor[2, 2, f32]",
    ];
    for spelling in spellings {
        let source =
            format!("module T.Probe\n@property p forall(v: {spelling}):\n  (1.0f64 == 1.0f64)\n");
        let decls = chelis_surf::parser::parse_str(&source)
            .unwrap_or_else(|error| panic!("`{spelling}` must parse: {error}"));
        let mut found = Vec::new();
        collect_property_decls(&decls, None, &mut Vec::new(), &mut found);
        let ty = found
            .first()
            .and_then(|(_, decl)| match decl {
                Decl::Property { params, .. } => params.first()?.ty.clone(),
                _ => None,
            })
            .unwrap_or_else(|| panic!("`{spelling}` must yield a typed binder"));

        let draw = |attempt| {
            let mut rng = Lcg::new(7);
            sample_value("v", &ty, &mut rng, attempt, Some((0, 2)), 100).map(|value| value.json)
        };
        let first = draw(0).unwrap_or_else(|error| panic!("`{spelling}` must sample: {error:?}"));
        let second = draw(1).unwrap_or_else(|error| panic!("`{spelling}` must sample: {error:?}"));
        let moved = first != second;
        let scheduled = takes_schedule_slot(&ty);
        assert_eq!(
            moved,
            scheduled,
            "`{spelling}`: takes_schedule_slot says {scheduled}, but the drawn value {} with \
             the slot ({first} vs {second}). The two must agree, or a binder that ignores its \
             slot still consumes an odometer digit.",
            if moved { "moved" } else { "did not move" },
        );
    }
}

/// The odometer's arity and digit positions count only the scheduled binders,
/// so an unscheduled one neither switches the odometer on nor takes a digit.
#[test]
fn issue_3697_unscheduled_binders_take_no_digit() {
    let sp = chelis_deep::Span::new(0, 0);
    let f64_ty = TypeExpr::Named("f64".to_string(), sp);
    let bool_ty = TypeExpr::Named("bool".to_string(), sp);

    // One scheduled binder beside a `bool` is still arity one, so it keeps the
    // carve-out and walks its schedule one attempt at a time.
    let (arity, positions) = arity_and_positions(&[Some(&f64_ty), Some(&bool_ty)]);
    assert_eq!(
        arity, 1,
        "a `bool` must not make this a two-binder odometer"
    );
    assert_eq!(positions, vec![Some(0), None]);

    // A `bool` declared first must not occupy digit 0.
    let (arity, positions) = arity_and_positions(&[Some(&bool_ty), Some(&f64_ty), Some(&f64_ty)]);
    assert_eq!(arity, 2, "only the two floats are scheduled");
    assert_eq!(
        positions,
        vec![None, Some(0), Some(1)],
        "the floats take digits 0 and 1, not 1 and 2"
    );

    // A missing type is not scheduled either; it errors in the sampler.
    let (arity, positions) = arity_and_positions(&[None, Some(&f64_ty)]);
    assert_eq!(arity, 1);
    assert_eq!(positions, vec![None, Some(0)]);
}

// ===========================================================================
// chelis#3971: machine-arithmetic confirmation of an over-the-reals DISPROOF
// ===========================================================================
//
// A Tier B disproof is decided over the reals, so its witness is an exact real
// point. Reporting it as a property failure claims a defect at some MACHINE
// input, which an exact rational -- let alone an irrational algebraic number --
// need not supply. These tests pin the three answers the confirmation can give
// and the routing each one earns, without going through a solver: the three
// arms are decided by the concrete evaluation, so a stubbed evaluator is the
// exact instrument, and the shapes it is fed are the shapes cvc5 prints.

const DISPROVED_VERDICT: &str = "disproved_modulo_real_arithmetic";

fn f64_sample(name: &str, value: f64) -> Sample {
    Sample {
        values: vec![directed_float_sample(name, "f64", value)],
    }
}

/// `(guard_holds, body_holds)` as a stubbed concrete evaluation, in the
/// `(sample, precondition)` shape both lanes pass to `resolve_disproof_witness`.
fn stub_eval(
    guard_holds: Result<bool, String>,
    body_holds: Result<bool, String>,
) -> impl FnMut(&Sample, bool) -> Result<bool, String> {
    move |_sample, precondition| {
        if precondition {
            guard_holds.clone()
        } else {
            body_holds.clone()
        }
    }
}

// The positive case: the guards admit the rounded witness and the body is
// false there. That is a definite machine counterexample, so the lane reports
// it with the concrete point rather than the solver's real-valued model.
#[test]
fn issue3971_an_admitted_witness_with_a_false_body_is_confirmed() {
    let resolved =
        resolve_disproof_witness(f64_sample("x", 12345.0), stub_eval(Ok(true), Ok(false)), 1);
    let DisproofWitness::Confirmed(counterexample) = resolved else {
        panic!("an admitted witness with a false body is confirmed: {resolved:?}");
    };
    assert_eq!(counterexample, serde_json::json!({"x": 12345.0}));
}

// Negative parity, first way: the guards admit the rounded witness but the
// body HOLDS there. Rounding moved the point off the exact real the solver
// exhibited, so the exhibited witness is not a machine counterexample.
#[test]
fn issue3971_a_witness_whose_body_holds_is_unconfirmed_not_confirmed() {
    // `std::f64::consts::SQRT_2` is the double nearest sqrt(2): the chelis#3971
    // reproduction's rounded witness, named through the constant so the value
    // cannot drift away from the one the lane actually evaluates.
    let resolved = resolve_disproof_witness(
        f64_sample("x", std::f64::consts::SQRT_2),
        stub_eval(Ok(true), Ok(true)),
        1,
    );
    assert!(
        matches!(
            resolved,
            DisproofWitness::Unconfirmed(reason)
                if reason == crate::machine_arith::WITNESS_BODY_HOLDS_REASON
        ),
        "{resolved:?}"
    );
}

// Negative parity, second way: rounding can move a value OUT of the guard
// region the exact real satisfied. A point the property's own preconditions
// reject is not a counterexample to the guarded property, and the reason must
// say that rather than blame the body.
#[test]
fn issue3971_a_witness_the_guards_reject_is_unconfirmed_and_the_body_is_not_consulted() {
    let mut body_calls = 0usize;
    let resolved = resolve_disproof_witness(
        f64_sample("x", 2.0),
        |_sample, precondition| {
            if precondition {
                Ok(false)
            } else {
                body_calls += 1;
                Ok(false)
            }
        },
        1,
    );
    assert!(
        matches!(
            resolved,
            DisproofWitness::Unconfirmed(reason)
                if reason == crate::machine_arith::WITNESS_REJECTED_BY_GUARDS_REASON
        ),
        "{resolved:?}"
    );
    assert_eq!(
        body_calls, 0,
        "a rejected witness is not a counterexample whatever the body says there"
    );
}

// A property with no preconditions admits every input, so the guard evaluation
// is skipped entirely rather than run against an empty conjunction.
#[test]
fn issue3971_an_unguarded_property_consults_only_its_body() {
    let mut guard_calls = 0usize;
    let resolved = resolve_disproof_witness(
        f64_sample("x", -1.0),
        |_sample, precondition| {
            if precondition {
                guard_calls += 1;
            }
            Ok(!precondition && false)
        },
        0,
    );
    assert!(
        matches!(resolved, DisproofWitness::Confirmed(_)),
        "{resolved:?}"
    );
    assert_eq!(guard_calls, 0);
}

// An evaluation that ERRORS establishes nothing, on either leg. It must not be
// read as either answer: `Confirmed` would invent a failure and `Unconfirmed`
// would withdraw a decision on no evidence.
#[test]
fn issue3971_an_evaluation_error_establishes_nothing_on_either_leg() {
    for (guard, body) in [
        (Err("guard blew up".to_string()), Ok(false)),
        (Ok(true), Err("body blew up".to_string())),
    ] {
        let resolved = resolve_disproof_witness(f64_sample("x", 1.0), stub_eval(guard, body), 1);
        let DisproofWitness::Unchecked(reason) = resolved else {
            panic!("an evaluation error is Unchecked: {resolved:?}");
        };
        assert!(
            reason.contains("could not be evaluated"),
            "the reason names the limitation: {reason}"
        );
    }
}

// The projection half. An irrational binding decides the whole model on its
// own -- no machine number equals it -- and is the stronger of the two
// declines, so it must win over an unreadable sibling in ANY position. The
// paired negative is the all-unreadable model, which establishes nothing.
#[test]
fn issue3971_an_irrational_binding_outranks_an_unreadable_sibling() {
    let variables = vec![
        ("a".to_string(), crate::solver::SmtSort::Real),
        ("b".to_string(), crate::solver::SmtSort::Real),
    ];
    let algebraic = "(_ real_algebraic_number <1*x^2 + (-2), (5/4, 3/2)>)";
    for model in [
        serde_json::json!({"a": algebraic, "b": "(mod 5 2)"}),
        serde_json::json!({"a": "(mod 5 2)", "b": algebraic}),
        serde_json::json!({"b": algebraic}),
    ] {
        let declined = disproof_witness_scalars(&variables, &model)
            .expect_err("a model binding an irrational has no machine point");
        assert!(
            matches!(
                declined,
                DisproofWitness::Unconfirmed(reason)
                    if reason == crate::machine_arith::IRRATIONAL_WITNESS_REASON
            ),
            "{model}: {declined:?}"
        );
    }

    let declined = disproof_witness_scalars(&variables, &serde_json::json!({"a": "(mod 5 2)"}))
        .expect_err("an unreadable model has no machine point either");
    let DisproofWitness::Unchecked(reason) = declined else {
        panic!("an unreadable model establishes nothing: {declined:?}");
    };
    assert!(
        reason.contains('`'),
        "the reason names the binder it could not read: {reason}"
    );

    // A model that is not a binding map at all.
    assert!(matches!(
        disproof_witness_scalars(&variables, &serde_json::json!([1, 2])),
        Err(DisproofWitness::Unchecked(_))
    ));
}

#[cfg(feature = "smt")]
#[test]
fn issue3971_a_fully_readable_model_projects_to_the_candidate_point() {
    let variables = vec![
        ("x".to_string(), crate::solver::SmtSort::Real),
        ("n".to_string(), crate::solver::SmtSort::Int),
        ("p".to_string(), crate::solver::SmtSort::Bool),
    ];
    let scalars = disproof_witness_scalars(
        &variables,
        &serde_json::json!({"x": "(- (/ 1 2))", "n": "7", "p": "true"}),
    )
    .expect("every binding is readable");
    assert_eq!(
        scalars,
        serde_json::json!({"x": -0.5, "n": 7, "p": true}),
        "the projection is the candidate point, at the goal's sorts"
    );
}

// The routing half, which is where the two tiers differ. `auto` has a Tier C
// to route to, so an unconfirmed witness hands the check to it and the Tier B
// record is withdrawn (`None`). `smt-only` is terminal, so the over-the-reals
// disproof stands as the reported verdict with the check recorded beside it.
#[test]
fn issue3971_an_unconfirmed_witness_routes_to_tier_c_under_auto_only() {
    for (tier, routes) in [("auto", true), ("smt-only", false)] {
        let options = PropertyRunOptions {
            tier: tier.to_string(),
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let mut carried = None;
        let resolved = resolve_disproof_outcome(
            hedged_disproof(),
            DisproofWitness::Unconfirmed(crate::machine_arith::IRRATIONAL_WITNESS_REASON),
            "p",
            &options,
            0,
            &mut carried,
        );
        assert_eq!(resolved.is_none(), routes, "{tier}: {resolved:#?}");
        assert_eq!(carried.is_some(), routes, "{tier}");
        let check = match &resolved {
            Some(outcome) => {
                assert_eq!(
                    outcome.composite_verdict,
                    CompositeVerdict::DisprovedModuloRealArithmetic,
                    "a terminal lane keeps the decision it was asked for: {outcome:#?}"
                );
                assert_eq!(outcome.proof_tier, PropertyTier::Smt);
                outcome.machine_arithmetic.clone()
            }
            None => carried.clone(),
        }
        .expect("the check is recorded on whichever record reports");
        assert_eq!(
            check.status,
            crate::machine_arith::MachineArithmeticStatus::Unconfirmed
        );
        assert_eq!(check.real_arithmetic_verdict, DISPROVED_VERDICT);
    }
}

// The confirmed witness is a concrete machine value, so the record is a
// DEFINITE failure that discloses no real-arithmetic caveat: the evaluation,
// not the solver, decided it. Same in both tiers -- the confirmation is one
// evaluation, not a second tier, so it does not widen `smt-only`.
#[test]
fn issue3971_a_confirmed_witness_is_a_definite_failure_in_both_tiers() {
    for tier in ["auto", "smt-only"] {
        let options = PropertyRunOptions {
            tier: tier.to_string(),
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let mut carried = None;
        let outcome = resolve_disproof_outcome(
            hedged_disproof(),
            DisproofWitness::Confirmed(serde_json::json!({"x": 12345.0})),
            "p",
            &options,
            0,
            &mut carried,
        )
        .expect("a confirmed witness is always reported");
        assert_eq!(outcome.status, PropertyStatus::Failed, "{tier}");
        assert_eq!(
            outcome.composite_verdict,
            CompositeVerdict::Failed,
            "{tier}"
        );
        assert!(
            outcome.disclosed_qualifiers().is_empty(),
            "{tier}: the concrete evaluation decided it: {outcome:#?}"
        );
        assert_eq!(outcome.proof_tier, PropertyTier::Smt, "{tier}");
        assert_eq!(
            outcome.counterexample,
            Some(serde_json::json!({"x": 12345.0})),
            "{tier}: the reported point is the one that was evaluated"
        );
        assert_eq!(
            outcome.machine_arithmetic.map(|check| check.status),
            Some(crate::machine_arith::MachineArithmeticStatus::Confirmed),
            "{tier}"
        );
        assert!(carried.is_none(), "{tier}: nothing is routed onward");
    }
}

// An evaluation limitation never displaces a decision -- the disproof side of
// chelis#3634's "a sampler limitation never degrades a proof". The hedged
// record survives verbatim in both tiers, with the reason recorded.
#[test]
fn issue3971_an_unchecked_witness_leaves_the_disproof_exactly_as_it_was() {
    for tier in ["auto", "smt-only"] {
        let options = PropertyRunOptions {
            tier: tier.to_string(),
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let mut carried = None;
        let outcome = resolve_disproof_outcome(
            hedged_disproof(),
            DisproofWitness::Unchecked("model unreadable".to_string()),
            "p",
            &options,
            0,
            &mut carried,
        )
        .expect("an unchecked witness never withdraws the disproof");
        assert_eq!(outcome.status, PropertyStatus::Failed, "{tier}");
        assert_eq!(
            outcome.composite_verdict,
            CompositeVerdict::DisprovedModuloRealArithmetic,
            "{tier}: {outcome:#?}"
        );
        assert_eq!(
            outcome.counterexample,
            Some(serde_json::json!({"x": "(_ real_algebraic_number <x>)"})),
            "{tier}: the solver's own model is still what is reported"
        );
        let check = outcome.machine_arithmetic.expect("the check is recorded");
        assert_eq!(
            check.status,
            crate::machine_arith::MachineArithmeticStatus::Unchecked,
            "{tier}"
        );
        assert_eq!(check.reason.as_deref(), Some("model unreadable"), "{tier}");
        assert!(carried.is_none(), "{tier}");
    }
}

/// The hedged Tier B disproof the three routing tests resolve: `Failed` with a
/// `RealArith` base, which is what composes to
/// `disproved_modulo_real_arithmetic`.
fn hedged_disproof() -> PropertyOutcome {
    PropertyOutcome::with_base_discharge(
        "p",
        PropertyStatus::Failed,
        PropertyTier::Smt,
        0,
        0,
        Some(serde_json::json!({"x": "(_ real_algebraic_number <x>)"})),
        None,
        false,
        Vec::new(),
        Some((
            crate::discharge::Soundness::SoundApproximate,
            crate::discharge::QualifierSet::from_iter_kinds([
                crate::discharge::Qualifier::RealArith,
            ]),
        )),
    )
}

// The gate itself, end to end on the lane predicate: an over-the-reals disproof
// under a reported tier owes the check, and the exempt cases do not.
#[test]
fn issue3971_the_hedged_disproof_discloses_what_the_gate_reads() {
    let hedged = hedged_disproof();
    assert_eq!(
        hedged.composite_verdict,
        CompositeVerdict::DisprovedModuloRealArithmetic
    );
    assert!(crate::machine_arith::disproof_confirmation_required(
        "auto",
        &hedged.disclosed_qualifiers()
    ));
    // A definite failure with no real-arithmetic base has no witness gap.
    let definite = PropertyOutcome::new(
        "p",
        PropertyStatus::Failed,
        PropertyTier::Fuzz,
        100,
        0,
        Some(serde_json::json!({"x": 1.0})),
        None,
        false,
        Vec::new(),
    );
    assert_eq!(definite.composite_verdict, CompositeVerdict::Failed);
    assert!(!crate::machine_arith::disproof_confirmation_required(
        "auto",
        &definite.disclosed_qualifiers()
    ));
}
