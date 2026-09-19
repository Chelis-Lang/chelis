//! chelis#1955 / [03-META-2]: every public semantic checker rejects a
//! `grad` whose written parameter identity contradicts its operative index.

use chelis_deep::Expr;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    CheckedProgram, TypeEnv, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_fitness,
    check_ir_program, check_ir_with_context, check_ir_with_signature_context, check_program,
    check_typed_program, infer_ir_program, infer_program,
};

fn selector_program(index: i64) -> Vec<Expr> {
    let declarations = parse_surf(
        "def pair(x: f32, w: f32) -> f32 = mul(x, w)\n\
         selected = grad(pair, wrt=w)\n",
    )
    .expect("selector fixture Surf must parse");
    let consistent = desugar_program(&declarations).expect("selector fixture must desugar");
    if index == 1 {
        return consistent;
    }
    let canonical = print_canonical(&consistent);
    let selector = "(lit {type: (t-prim {} i32)} 1)";
    assert_eq!(
        canonical.matches(selector).count(),
        1,
        "fixture must contain exactly one operative selector:\n{canonical}"
    );
    chelis_deep::parser::parse_str(&canonical.replacen(
        selector,
        &format!("(lit {{type: (t-prim {{}} i32)}} {index})"),
        1,
    ))
    .expect("mutated selector fixture must parse")
}

fn stamped_selector_program(index: i64) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(&print_canonical(&selector_program(index)))
        .expect("selector fixture must stamp")
}

fn checked_pair_context() -> (TypeEnv, CheckedProgram) {
    let mut pair = selector_program(1);
    pair.pop()
        .expect("selector fixture must end with the selected binding");
    build_compiled_library_context(&pair).expect("pair library must check")
}

fn contextual_selector_program(index: i64) -> Vec<Expr> {
    let mut program = selector_program(index);
    vec![
        program
            .pop()
            .expect("selector fixture must end with the selected binding"),
    ]
}

fn assert_selector_contradiction(errors: &[CheckError]) {
    let contradiction = errors
        .iter()
        .find(|error| error.message.contains("selector metadata"))
        .unwrap_or_else(|| panic!("missing selector contradiction: {errors:#?}"));
    assert!(
        matches!(contradiction.kind, CheckErrorKind::TypeMismatch),
        "{contradiction:?}"
    );
    assert!(
        contradiction.message.contains("parameter `w`")
            && contradiction.message.contains("index 0")
            && contradiction.message.contains("parameter `x`"),
        "{}",
        contradiction.message
    );
}

#[test]
fn check_ir_program_accepts_consistent_grad_selector_identity() {
    check_ir_program(&selector_program(1)).expect("index 1 selects the metadata parameter `w`");
}

#[test]
fn check_ir_program_rejects_contradictory_grad_selector_identity() {
    let result =
        check_ir_program(&selector_program(0)).expect_err("index 0 selects `x`, not metadata `w`");
    assert_selector_contradiction(&result.errors);
}

#[test]
fn check_typed_program_accepts_consistent_grad_selector_identity() {
    check_typed_program(&stamped_selector_program(1))
        .expect("index 1 selects the metadata parameter `w`");
}

#[test]
fn check_typed_program_rejects_contradictory_grad_selector_identity() {
    let result = check_typed_program(&stamped_selector_program(0))
        .expect_err("index 0 selects `x`, not metadata `w`");
    assert_selector_contradiction(&result.errors);
}

#[test]
fn neighboring_public_semantic_entries_reject_the_same_contradiction() {
    let program = selector_program(0);
    for (entry, errors) in [
        ("infer_ir_program", infer_ir_program(&program).errors),
        ("infer_program", infer_program(&program).errors),
        ("check_ir_fitness", check_ir_fitness(&program).errors),
        ("check_program", check_program(&program).errors),
        (
            "build_type_env_from_library",
            build_type_env_from_library(&program)
                .expect_err("contradictory library must reject")
                .errors,
        ),
        (
            "build_compiled_library_context",
            build_compiled_library_context(&program)
                .expect_err("contradictory compiled library must reject")
                .errors,
        ),
        (
            "check_ir_with_context",
            check_ir_with_context(&TypeEnv::empty(), &program)
                .expect_err("contradictory contextual input must reject")
                .errors,
        ),
    ] {
        assert!(!errors.is_empty(), "{entry} accepted the contradiction");
        assert_selector_contradiction(&errors);
    }
}

#[test]
fn check_ir_with_context_accepts_consistent_library_callable_identity() {
    let (context, _) = checked_pair_context();
    check_ir_with_context(&context, &contextual_selector_program(1))
        .expect("context callable index 1 selects metadata parameter `w`");
}

#[test]
fn check_ir_with_context_rejects_contradictory_library_callable_identity() {
    let (context, _) = checked_pair_context();
    let result = check_ir_with_context(&context, &contextual_selector_program(0))
        .expect_err("context callable index 0 selects `x`, not metadata `w`");
    assert_selector_contradiction(&result.errors);
}

#[test]
fn check_ir_with_signature_context_accepts_consistent_library_callable_identity() {
    let (context, library) = checked_pair_context();
    check_ir_with_signature_context(
        &context,
        library.signature_inference(),
        &contextual_selector_program(1),
    )
    .expect("signature context callable index 1 selects metadata parameter `w`");
}

#[test]
fn check_ir_with_signature_context_rejects_contradictory_library_callable_identity() {
    let (context, library) = checked_pair_context();
    let result = check_ir_with_signature_context(
        &context,
        library.signature_inference(),
        &contextual_selector_program(0),
    )
    .expect_err("signature context callable index 0 selects `x`, not metadata `w`");
    assert_selector_contradiction(&result.errors);
}

#[test]
fn compiled_library_with_base_accepts_consistent_base_callable_identity() {
    let (base, _) = checked_pair_context();
    build_compiled_library_context_with_base(&base, &contextual_selector_program(1))
        .expect("base callable index 1 selects metadata parameter `w`");
}

#[test]
fn compiled_library_with_base_rejects_contradictory_base_callable_identity() {
    let (base, _) = checked_pair_context();
    let result = build_compiled_library_context_with_base(&base, &contextual_selector_program(0))
        .expect_err("base callable index 0 selects `x`, not metadata `w`");
    assert_selector_contradiction(&result.errors);
}

#[test]
fn standalone_and_serialized_type_envs_preserve_library_callable_identity() {
    let mut pair = selector_program(1);
    pair.pop()
        .expect("selector fixture must end with the selected binding");
    let live = build_type_env_from_library(&pair).expect("standalone TypeEnv must build");
    let encoded = bincode::serialize(&live).expect("TypeEnv must serialize");
    let decoded: TypeEnv = bincode::deserialize(&encoded).expect("TypeEnv must deserialize");

    for (label, context) in [("live", live), ("decoded", decoded)] {
        check_ir_with_context(&context, &contextual_selector_program(1))
            .unwrap_or_else(|error| panic!("{label} context lost valid origin: {error:#?}"));
        let result = match check_ir_with_context(&context, &contextual_selector_program(0)) {
            Ok(_) => panic!("{label} context accepted contradictory selector"),
            Err(error) => error,
        };
        assert_selector_contradiction(&result.errors);
    }
}
