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
    // The call to `helper(a, b)` here is truly unconstrained: no
    // ascription is placed on `z`, so the result type comes solely
    // from `helper`'s scheme as known at the point `caller` is
    // analyzed (forward reference). Pre-chelis#159 the test had
    // `z: tensor[4, f32]` but the let-binding ascription was silently
    // dropped, masking what the test claimed to assert. Post-fix
    // dropping the ascription restores the intent: `z` stays
    // unconstrained, `a` stays unconstrained, and read-only inference
    // correctly defaults to owned for the unknown-type param.
    let checked = checked_surf(
        r#"
def caller(a, b: tensor[4, f32]) = {
  z = helper(a, b)
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
fn module_wrapped_helper_signature_visible_to_caller_annotation() {
    // Pre-#181, module-wrapped multi-decl programs had NO inference
    // pre-populated during annotation (a silent no-op), so the
    // wrapped vs unwrapped flavors of the same source produced
    // different signature-inference results. PR #183 fixed
    // `annotate_ir_program{,_with_context}` to descend through
    // `(module {} name ...)` wrappers, which means the helper IS
    // visible at caller-annotation time in the wrapped case.
    //
    // This test pins the post-fix behavior: for a module-wrapped
    // version of `later_helper_does_not_retroactively_...`, caller's
    // `a` becomes read-only because helper's read-only-`x` signature
    // is visible by the time caller is annotated. (Bare-decl
    // forward-reference semantics stay as-is in the sibling test
    // above.) Together the two tests document the intentional
    // asymmetry between bare-decl and module-wrapped programs.
    let checked = checked_surf(
        r#"
module Foo
def caller(a, b: tensor[4, f32]) = {
  z = helper(a, b)
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
    assert!(
        caller.params[0].inferred_read_only,
        "module-wrapped caller's `a` SHOULD become read-only: helper's signature \
         is visible at caller-annotation time after the #183 module-descent fix"
    );
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
