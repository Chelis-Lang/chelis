use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{
    AddFunctionRequest, AddFunctionResult, AddPropertyRequest, ApiEnvelope, ChangeSignatureRequest,
    CheckRequest, DeepCallGraphRequest, DeepOutlineRequest, RenameRequest, ReplaceFunctionRequest,
    SourceKind, WireDeepExpr,
};
use chelis_deep::Expr;
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

const ADD_PROPERTY_ALWAYS_TRUE: &str = r#"(defsig {}
  always_true
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} bool)))
(def {chelis_role: "property",
       property_preconditions: (tuple {}),
       property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
       property_source_kind: "user"}
  always_true
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (var {} x))))
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
    matches!(list.tag(), Some(found) if found.as_str() == tag)
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

fn target_preimage(module: &str, function_name: &str) -> String {
    let outline = compiler::deep_outline(DeepOutlineRequest {
        module: module.to_string(),
    })
    .expect("outline");
    outline
        .functions
        .into_iter()
        .find(|function| function.name == function_name)
        .expect("function")
        .preimage_sha256
}

#[test]
fn deep_outline_and_call_graph_are_machine_contracts() {
    let outline = compiler::deep_outline(DeepOutlineRequest {
        module: BASE_TWO_FUNCTIONS.to_string(),
    })
    .expect("outline");
    assert_eq!(outline.module_name, "handoff.demo");
    assert_eq!(outline.functions.len(), 2);
    let first = outline
        .functions
        .iter()
        .find(|function| function.name == "first")
        .expect("first");
    assert_eq!(first.qualified_name, "handoff.demo.first");
    assert_eq!(first.params, vec!["x"]);
    assert_eq!(first.preimage_sha256.len(), 64);

    let graph = compiler::deep_call_graph(DeepCallGraphRequest {
        module: BASE_TWO_FUNCTIONS.to_string(),
    })
    .expect("call graph");
    assert!(graph.edges.is_empty(), "identity functions have no calls");
}

#[test]
fn rename_fails_closed_on_stale_preimage_and_cascades_calls() {
    let caller_module = r#"(module {}
  handoff.rename
  (export {} first second)
  (defsig {} first (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {} first (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))
  (defsig {} second (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {} second (fn {} (params {} (x {type: (t-prim {} f32)})) (app {} (var {} first) (var {} x)))))
"#;
    let stale = compiler::result_envelope(compiler::rename(RenameRequest {
        module: caller_module.to_string(),
        function_name: "first".to_string(),
        new_name: "renamed".to_string(),
        preimage_sha256: Some("0".repeat(64)),
    }));
    let stale_json = serde_json::to_value(stale).expect("stale serializes");
    assert_eq!(stale_json["ok"], false);
    assert_eq!(stale_json["stage"], "preimage");
    assert!(stale_json.get("result").is_none());

    let ok = compiler::rename(RenameRequest {
        module: caller_module.to_string(),
        function_name: "first".to_string(),
        new_name: "renamed".to_string(),
        preimage_sha256: Some(target_preimage(caller_module, "first")),
    })
    .expect("rename");
    assert!(ok.module_deep.contains("(export {} renamed second)"));
    assert!(
        ok.module_deep
            .contains("(app {} (var {} renamed) (var {} x))")
    );
    assert_eq!(ok.renamed_references, 1);
    assert_full_check_has_no_fitness_errors(&ok.module_deep);
}

#[test]
fn rename_cascades_property_precondition_metadata() {
    let module = r#"(module {}
  handoff.property_meta
  (defsig {} guard (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} bool)))
  (def {} guard (fn {} (params {} (x {type: (t-prim {} f32)})) (app {} (var {} gte) (var {} x) (var {} x))))
  (defsig {} holds (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
        property_preconditions: (tuple {} (app {} (var {} guard) (var {} x))),
        property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
        property_source_kind: "user"}
    holds
    (fn {} (params {} (x {type: (t-prim {} f32)})) (app {} (var {} gte) (var {} x) (var {} x)))))
"#;

    let ok = compiler::rename(RenameRequest {
        module: module.to_string(),
        function_name: "guard".to_string(),
        new_name: "renamed".to_string(),
        preimage_sha256: Some(target_preimage(module, "guard")),
    })
    .expect("rename cascades through metadata");

    assert!(
        ok.module_deep
            .contains("property_preconditions: (tuple {} (app {} (var {} renamed) (var {} x)))"),
        "property preconditions are semantically active Deep and must be cascaded: {}",
        ok.module_deep
    );
    assert_full_check_has_no_fitness_errors(&ok.module_deep);
}

#[test]
fn replace_function_and_change_signature_validate_whole_module() {
    let new_decls = r#"(defsig {}
  first
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
(def {}
  first
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (lit {type: (t-prim {} bool)} true)))
"#;
    let rejected = compiler::result_envelope(compiler::replace_function(ReplaceFunctionRequest {
        module: BASE_TWO_FUNCTIONS.to_string(),
        function_name: "first".to_string(),
        new_decls: new_decls.to_string(),
        preimage_sha256: Some(target_preimage(BASE_TWO_FUNCTIONS, "first")),
    }));
    let rejected_json = serde_json::to_value(rejected).expect("rejection serializes");
    assert_eq!(rejected_json["ok"], false);
    assert_eq!(rejected_json["stage"], "check");
    assert!(rejected_json.get("result").is_none());

    let sig_module = r#"(module {}
  handoff.sig
  (defsig {} pair (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {} pair (fn {} (params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)})) (app {} (var {} sub) (var {} x) (var {} y))))
  (defsig {} caller (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {} caller (fn {} (params {} (a {type: (t-prim {} f32)}) (b {type: (t-prim {} f32)})) (app {} (var {} pair) (var {} a) (var {} b)))))
"#;
    let changed = compiler::change_signature(ChangeSignatureRequest {
        module: sig_module.to_string(),
        function_name: "pair".to_string(),
        new_defsig: "(defsig {} pair (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))".to_string(),
        new_params: "(params {} (y {type: (t-prim {} f32)}) (x {type: (t-prim {} f32)}))".to_string(),
        argument_order: vec!["y".to_string(), "x".to_string()],
        param_renames: Default::default(),
        preimage_sha256: Some(target_preimage(sig_module, "pair")),
    })
    .expect("change signature");
    assert!(
        changed
            .module_deep
            .contains("(app {} (var {} pair) (var {} b) (var {} a))")
    );
    assert_eq!(changed.rewritten_calls, 1);
    assert_full_check_has_no_fitness_errors(&changed.module_deep);
}

#[test]
fn change_signature_rewrites_property_precondition_metadata_calls() {
    let module = r#"(module {}
  handoff.property_sig
  (defsig {} guard (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} bool)))
  (def {} guard (fn {} (params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)})) (app {} (var {} gte) (var {} x) (var {} y))))
  (defsig {} holds (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
        property_preconditions: (tuple {} (app {} (var {} guard) (var {} x) (var {} y))),
        property_quantifiers: (params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)})),
        property_source_kind: "user"}
    holds
    (fn {} (params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)})) (app {} (var {} gte) (var {} x) (var {} x)))))
"#;

    let changed = compiler::change_signature(ChangeSignatureRequest {
        module: module.to_string(),
        function_name: "guard".to_string(),
        new_defsig: "(defsig {} guard (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} bool)))".to_string(),
        new_params: "(params {} (y {type: (t-prim {} f32)}) (x {type: (t-prim {} f32)}))"
            .to_string(),
        argument_order: vec!["y".to_string(), "x".to_string()],
        param_renames: Default::default(),
        preimage_sha256: Some(target_preimage(module, "guard")),
    })
    .expect("change_signature cascades through metadata");

    assert!(
        changed
            .module_deep
            .contains("(app {} (var {} guard) (var {} y) (var {} x))"),
        "property precondition call arguments must be reordered: {}",
        changed.module_deep
    );
    assert_eq!(changed.rewritten_calls, 1);
    assert_full_check_has_no_fitness_errors(&changed.module_deep);
}

#[test]
fn add_property_requires_property_role_and_validates_whole_module() {
    let added = compiler::add_property(AddPropertyRequest {
        module: BASE_TWO_FUNCTIONS.to_string(),
        new_decls: ADD_PROPERTY_ALWAYS_TRUE.to_string(),
        insert_after_function: Some("second".to_string()),
    })
    .expect("add property");
    assert!(
        added
            .added_property_def_deep
            .contains("chelis_role: \"property\"")
    );
    assert!(added.module_deep.contains("always_true"));
    assert_full_check_has_no_fitness_errors(&added.module_deep);

    let rejected = compiler::result_envelope(compiler::add_property(AddPropertyRequest {
        module: BASE_TWO_FUNCTIONS.to_string(),
        new_decls: ADD_IDENTITY.to_string(),
        insert_after_function: None,
    }));
    let rejected_json = serde_json::to_value(rejected).expect("rejection serializes");
    assert_eq!(rejected_json["ok"], false);
    assert_eq!(rejected_json["stage"], "add-property");
    assert!(rejected_json.get("result").is_none());
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
