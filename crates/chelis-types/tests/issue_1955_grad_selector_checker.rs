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

fn stamped_nested_selector_program() -> Vec<Expr> {
    let declarations = parse_surf(
        "def square(x: f32) -> f32 = mul(x, x)\n\
         selected = grad(grad(square, wrt=x), wrt=x)\n",
    )
    .expect("nested selector fixture Surf must parse");
    let deep = desugar_program(&declarations).expect("nested selector fixture must desugar");
    chelis_deep::parse_and_stamp_file(&print_canonical(&deep))
        .expect("nested selector fixture must stamp")
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
    let contradiction = errors.iter().find(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error.expected.as_deref() == Some("parameter `w`")
            && error.got.as_deref() == Some("parameter `x`")
    }).unwrap_or_else(|| panic!("missing selector contradiction: {errors:#?}"));
    assert!(
        contradiction.message.contains("index 0"),
        "the operative selector must identify the rejected index: {contradiction:?}"
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
fn out_of_range_grad_selector_reports_admitted_index_and_actual_index() {
    let report = check_ir_program(&selector_program(2))
        .expect_err("two-parameter callable cannot have a selector at index 2");
    let error = report.errors.iter().find(|error| {
        matches!(error.kind, CheckErrorKind::ArityMismatch)
            && error.message.contains("selector")
    }).unwrap_or_else(|| panic!("missing selector arity error: {:?}", report.errors));
    assert_eq!(error.expected.as_deref(), Some("index in 0..2"));
    assert_eq!(error.got.as_deref(), Some("index 2"));
    check_ir_program(&selector_program(1))
        .expect("the in-range index selecting the named parameter remains valid");
}

#[test]
fn check_typed_program_accepts_consistent_grad_selector_identity() {
    check_typed_program(&stamped_selector_program(1))
        .expect("index 1 selects the metadata parameter `w`");
}

#[test]
fn check_typed_program_accepts_consistent_nested_grad_selector_identity() {
    check_typed_program(&stamped_nested_selector_program())
        .expect("a transformed callable retains the target parameter identity");
}

#[test]
fn check_typed_program_rejects_contradictory_grad_selector_identity() {
    let result = check_typed_program(&stamped_selector_program(0))
        .expect_err("index 0 selects `x`, not metadata `w`");
    assert_selector_contradiction(&result.errors);
    let error = result.errors.iter().find(|error| {
        matches!(error.kind, CheckErrorKind::TypeMismatch)
            && error.message.contains("selector index 0")
    }).expect("contradictory selector error");
    assert_eq!(error.expected.as_deref(), Some("parameter `w`"));
    assert_eq!(error.got.as_deref(), Some("parameter `x`"));
    assert!(error.span_offset.is_some(), "stamped Deep selector location: {error:?}");
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

#[test]
fn json_type_env_roundtrip_binds_exact_library_callable_metadata() {
    let (context, checked) = checked_pair_context();
    let encoded =
        serde_json::to_value(&context).expect("a nonempty selector context must be JSON-safe");
    let decoded: TypeEnv =
        serde_json::from_value(encoded.clone()).expect("the JSON TypeEnv must round-trip");
    assert_eq!(
        serde_json::to_value(&decoded).expect("the decoded TypeEnv must re-encode"),
        encoded,
        "selector context serialization must be deterministic"
    );
    assert!(
        decoded.matches_checked_program(&checked),
        "an unchanged selector snapshot must match its checked program"
    );

    let mut reordered = encoded.clone();
    let params = reordered["inner"]["selector_callables"]["root"]["pair"]["Known"]["params"]
        .as_array_mut()
        .expect("the pair callable must retain ordered formal names");
    params.swap(0, 1);
    let forged: TypeEnv =
        serde_json::from_value(reordered).expect("the structurally valid forged TypeEnv decodes");
    assert!(
        !forged.matches_checked_program(&checked),
        "swapping `(x, w)` to `(w, x)` must break the checked-library pairing"
    );

    let mut reidentified = encoded;
    let ordinal = reidentified["inner"]["selector_callables"]["root"]["pair"]["Known"]["identity"]
        ["ordinal"]
        .as_u64()
        .expect("the pair callable must retain its lexical origin");
    reidentified["inner"]["selector_callables"]["root"]["pair"]["Known"]["identity"]["ordinal"] =
        serde_json::json!(ordinal + 1);
    let forged: TypeEnv = serde_json::from_value(reidentified)
        .expect("the structurally valid reidentified TypeEnv decodes");
    assert!(
        !forged.matches_checked_program(&checked),
        "changing the callable's lexical identity must break the checked-library pairing"
    );
}
