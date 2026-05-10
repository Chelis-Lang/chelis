use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{
    CheckedProgram, build_compiled_library_context, check_ir_with_signature_context,
    check_typed_program,
};

fn checked_surf(source: &str) -> CheckedProgram {
    let decls = parse_str(source).expect("surf parse");
    let deep = desugar_program(&decls);
    check_typed_program(&deep).expect("type check")
}

#[test]
fn unannotated_read_only_tensor_param_gets_display_ref() {
    let checked = checked_surf(
        r#"
def readonly(x, y: tensor[4, f32]) = add(x, y)
"#,
    );

    let func = checked
        .signature_inference()
        .functions
        .get("readonly")
        .expect("readonly metadata");
    assert!(!func.recursive_cycle);
    assert_eq!(func.params.len(), 2);
    assert!(!func.params[0].written);
    assert!(func.params[0].inferred_read_only);
    assert!(matches!(
        func.params[0].display_type,
        chelis_types::types::Type::Ref(_)
    ));
}

#[test]
fn consuming_param_use_stays_owned_for_display() {
    let checked = checked_surf(
        r#"
def consume(x, y: tensor[4, f32]) = {
  z: tensor[4, f32] = add(x, y)
  realize(x)
}
"#,
    );

    let func = checked
        .signature_inference()
        .functions
        .get("consume")
        .expect("consume metadata");
    assert!(!func.params[0].written);
    assert!(!func.params[0].inferred_read_only);
    assert!(!matches!(
        func.params[0].display_type,
        chelis_types::types::Type::Ref(_)
    ));
}

#[test]
fn written_param_annotation_is_authoritative() {
    let checked = checked_surf(
        r#"
def annotated(x: tensor[4, f32]) = relu(x)
"#,
    );

    let func = checked
        .signature_inference()
        .functions
        .get("annotated")
        .expect("annotated metadata");
    assert!(func.params[0].written);
    assert!(!func.params[0].inferred_read_only);
    assert!(!matches!(
        func.params[0].display_type,
        chelis_types::types::Type::Ref(_)
    ));
}

#[test]
fn earlier_inferred_signature_makes_later_call_read_only() {
    let checked = checked_surf(
        r#"
def helper(x, y: tensor[4, f32]) = add(x, y)
def caller(a, b: tensor[4, f32]) = {
  z: tensor[4, f32] = helper(a, b)
  add(a, z)
}
"#,
    );

    let helper = checked
        .signature_inference()
        .functions
        .get("helper")
        .expect("helper metadata");
    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(helper.params[0].inferred_read_only);
    assert!(caller.params[0].inferred_read_only);
}

#[test]
fn later_helper_does_not_retroactively_make_earlier_unconstrained_call_read_only() {
    let checked = checked_surf(
        r#"
def caller(a, b: tensor[4, f32]) = {
  z: tensor[4, f32] = helper(a, b)
  add(a, z)
}
def helper(x, y: tensor[4, f32]) = add(x, y)
"#,
    );

    let helper = checked
        .signature_inference()
        .functions
        .get("helper")
        .expect("helper metadata");
    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(helper.params[0].inferred_read_only);
    assert!(!caller.params[0].inferred_read_only);
}

#[test]
fn recursive_cycle_member_does_not_infer_read_only_param() {
    let checked = checked_surf(
        r#"
def recur(x, y: tensor[4, f32], n: int32) =
  if eq(n, 0) then add(x, y) else recur(x, y, sub(n, 1))
"#,
    );

    let recur = checked
        .signature_inference()
        .functions
        .get("recur")
        .expect("recur metadata");
    assert!(recur.recursive_cycle);
    assert!(!recur.params[0].written);
    assert!(!recur.params[0].inferred_read_only);
}

#[test]
fn mutual_recursive_cycle_members_do_not_infer_read_only_params() {
    let checked = checked_surf(
        r#"
def ping(x, y: tensor[4, f32], n: int32) =
  if eq(n, 0) then add(x, y) else pong(x, y, sub(n, 1))

def pong(x, y: tensor[4, f32], n: int32) =
  if eq(n, 0) then add(x, y) else ping(x, y, sub(n, 1))
"#,
    );

    let ping = checked
        .signature_inference()
        .functions
        .get("ping")
        .expect("ping metadata");
    let pong = checked
        .signature_inference()
        .functions
        .get("pong")
        .expect("pong metadata");
    assert!(ping.recursive_cycle);
    assert!(pong.recursive_cycle);
    assert!(!ping.params[0].inferred_read_only);
    assert!(!pong.params[0].inferred_read_only);
}

#[test]
fn context_signature_metadata_makes_imported_call_read_only() {
    let library_decls = parse_str(
        r#"
def helper(x, y: tensor[4, f32]) = add(x, y)
"#,
    )
    .expect("library surf parse");
    let library_deep = desugar_program(&library_decls);
    let (type_env, library_checked) =
        build_compiled_library_context(&library_deep).expect("library context");

    let app_decls = parse_str(
        r#"
def caller(a, b: tensor[4, f32]) = {
  z: tensor[4, f32] = helper(a, b)
  add(a, z)
}
"#,
    )
    .expect("app surf parse");
    let app_deep = desugar_program(&app_decls);
    let checked = check_ir_with_signature_context(
        &type_env,
        library_checked.signature_inference(),
        &app_deep,
    )
    .expect("app check");

    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(caller.params[0].inferred_read_only);
}
