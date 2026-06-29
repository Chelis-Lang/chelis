use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{
    AddFunctionRequest, AddFunctionResult, ApiEnvelope, CheckRequest, SourceKind, WireDeepExpr,
};
use chelis_deep::{Atom, Expr};
use schemars::schema_for;
use serde_json::Value;

const BASE_TWO_FUNCTIONS: &str = r#"(module {}
  handoff.demo
  (export {} first second)
  (defsig {}
    first
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    first
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (var {} x)))
  (defsig {}
    second
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    second
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (var {} x))))
"#;

const ADD_IDENTITY: &str = r#"(defsig {}
  added
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
(def {}
  added
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (var {} x)))
"#;

const TENSOR_MODULE: &str = r#"(module {}
  handoff.tensor
  (export {} passthrough)
  (defsig {}
    passthrough
    (t-fn {eff: (effects {})}
      (t-tensor {} (d-lit {} 4) (t-prim {} f32))
      (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
  (def {}
    passthrough
    (fn {}
      (params {}
        (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
      (var {} x))))
"#;

const ADD_EFFECTING: &str = r#"(defsig {}
  noisy
  (t-fn {eff: (effects {})}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
(def {}
  noisy
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))
"#;

const ADD_LINEARITY_VIOLATION: &str = r#"(defsig {}
  alias_twice
  (t-fn {eff: (effects {})}
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))
    (t-tensor {} (d-lit {} 4) (t-prim {} f32))))
(def {}
  alias_twice
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (let {}
      (bind {} y (realize {} (var {} x)))
      (app {} (var {} add) (var {} x) (var {} y)))))
"#;

const ADD_ILL_TYPED: &str = r#"(defsig {}
  bad_type
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
(def {}
  bad_type
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (lit {type: (t-prim {} bool)} true)))
"#;

fn add_function(module: &str, new_decls: &str, insert_after: Option<&str>) -> AddFunctionResult {
    compiler::add_function(AddFunctionRequest {
        module: module.to_string(),
        new_decls: new_decls.to_string(),
        insert_after_function: insert_after.map(str::to_string),
    })
    .expect("add_function succeeds")
}

fn add_function_failure(module: &str, new_decls: &str, insert_after: Option<&str>) -> Value {
    let envelope = compiler::result_envelope(compiler::add_function(AddFunctionRequest {
        module: module.to_string(),
        new_decls: new_decls.to_string(),
        insert_after_function: insert_after.map(str::to_string),
    }));
    serde_json::to_value(envelope).expect("envelope serializes")
}

fn assert_failure(
    module: &str,
    new_decls: &str,
    insert_after: Option<&str>,
    stage: &str,
    kind: &str,
) {
    let failure = add_function_failure(module, new_decls, insert_after);
    assert_eq!(
        failure["ok"], false,
        "add_function should reject: {failure}"
    );
    assert_eq!(failure["stage"], stage);
    assert!(
        failure.get("result").is_none(),
        "failed add_function must not include result: {failure}"
    );
    let errors = failure["errors"].as_array().expect("errors array");
    assert!(!errors.is_empty());
    assert_eq!(errors[0]["kind"], kind);
}

fn canonical_decls(source: &str) -> Vec<String> {
    let exprs = chelis_deep::parser::parse_str_strict(source).expect("deep parses");
    module_decl_exprs(&exprs)
        .iter()
        .map(chelis_deep::printer::print_expr)
        .collect()
}

fn module_decl_exprs(exprs: &[Expr]) -> Vec<Expr> {
    let module = exprs
        .iter()
        .find_map(|expr| match expr {
            Expr::List(list, _) if is_tag(list, "module") => Some(list),
            _ => None,
        })
        .expect("single module");
    module.elements[3..].to_vec()
}

fn is_tag(list: &chelis_deep::List, tag: &str) -> bool {
    matches!(list.elements.first(), Some(Expr::Atom(Atom::Symbol(found), _)) if found == tag)
}

fn assert_insertion_faithful(original: &str, rewritten: &str, inserted: &str, index: usize) {
    let original_decls = canonical_decls(original);
    let rewritten_decls = canonical_decls(rewritten);
    let inserted_decls = canonical_decls(&format!("(module {{}} fixture\n{inserted})"));

    assert_eq!(
        rewritten_decls.len(),
        original_decls.len() + inserted_decls.len()
    );
    assert_eq!(&rewritten_decls[..index], &original_decls[..index]);
    assert_eq!(
        &rewritten_decls[index..index + inserted_decls.len()],
        inserted_decls.as_slice()
    );
    assert_eq!(
        &rewritten_decls[index + inserted_decls.len()..],
        &original_decls[index..]
    );
}

fn assert_full_check_has_no_fitness_errors(module_deep: &str) {
    let check = compiler::check(CheckRequest {
        source_kind: SourceKind::Deep,
        source: module_deep.to_string(),
    })
    .expect("check runs");
    assert!(
        check.errors.is_empty(),
        "rewritten module has no fitness errors: {:?}",
        check.errors
    );
}

#[test]
fn add_function_appends_bundle_after_declaration_list_and_checks() {
    let result = add_function(BASE_TWO_FUNCTIONS, ADD_IDENTITY, None);
    assert!(result.added_def_deep.contains("added"));
    assert!(
        result
            .added_defsig_deep
            .as_deref()
            .is_some_and(|sig| sig.contains("added"))
    );
    assert_full_check_has_no_fitness_errors(&result.module_deep);
    assert_insertion_faithful(
        BASE_TWO_FUNCTIONS,
        &result.module_deep,
        ADD_IDENTITY,
        canonical_decls(BASE_TWO_FUNCTIONS).len(),
    );
}

#[test]
fn add_function_inserts_after_target_declaration_bundle() {
    let result = add_function(BASE_TWO_FUNCTIONS, ADD_IDENTITY, Some("first"));
    assert_full_check_has_no_fitness_errors(&result.module_deep);
    // BASE_TWO_FUNCTIONS decl order: export, first defsig, first def, second
    // defsig, second def. Inserting after first's bundle means index 3.
    assert_insertion_faithful(BASE_TWO_FUNCTIONS, &result.module_deep, ADD_IDENTITY, 3);
}

#[test]
fn add_function_rejects_bad_request_shapes_before_editing() {
    assert_failure(
        BASE_TWO_FUNCTIONS,
        "(def {}",
        None,
        "add-function",
        "deep_parse_error",
    );
    assert_failure(
        BASE_TWO_FUNCTIONS,
        "(export {} added)",
        None,
        "add-function",
        "deep_decl_error",
    );
    assert_failure(
        BASE_TWO_FUNCTIONS,
        r#"(defsig {} sig_name (t-fn {} (t-prim {} f32) (t-prim {} f32)))
           (def {} def_name (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))"#,
        None,
        "add-function",
        "deep_decl_error",
    );
}

#[test]
fn add_function_unknown_insert_target_is_name_resolution_error() {
    assert_failure(
        BASE_TWO_FUNCTIONS,
        ADD_IDENTITY,
        Some("missing"),
        "name-resolution",
        "name_resolution_error",
    );
}

#[test]
fn add_function_semantic_rejections_come_from_whole_module_check() {
    assert_failure(
        BASE_TWO_FUNCTIONS,
        r#"(def {}
             first
             (fn {}
               (params {} (x {type: (t-prim {} f32)}))
               (var {} x)))"#,
        None,
        "check",
        "type_error",
    );
    assert_failure(
        BASE_TWO_FUNCTIONS,
        ADD_ILL_TYPED,
        None,
        "check",
        "type_error",
    );
    assert_failure(
        TENSOR_MODULE,
        ADD_EFFECTING,
        None,
        "effects",
        "effect_error",
    );
    assert_failure(
        TENSOR_MODULE,
        ADD_LINEARITY_VIOLATION,
        None,
        "linearity",
        "linearity_error",
    );
}

#[test]
fn deep_authoring_wire_schemas_are_available() {
    let add_request_schema =
        serde_json::to_value(schema_for!(AddFunctionRequest)).expect("request schema");
    assert_eq!(add_request_schema["properties"]["module"]["type"], "string");
    assert_eq!(
        add_request_schema["properties"]["new_decls"]["type"],
        "string"
    );

    let add_envelope_schema =
        serde_json::to_value(schema_for!(ApiEnvelope<AddFunctionResult>)).expect("envelope schema");
    assert!(
        add_envelope_schema
            .to_string()
            .contains("AddFunctionResult"),
        "authoring result envelope schema names AddFunctionResult: {add_envelope_schema}"
    );

    let deep_ast_schema = serde_json::to_value(schema_for!(WireDeepExpr)).expect("deep ast schema");
    assert!(
        deep_ast_schema.to_string().contains("WireDeepExprKind"),
        "Deep AST schema names the wire kind enum: {deep_ast_schema}"
    );
}
