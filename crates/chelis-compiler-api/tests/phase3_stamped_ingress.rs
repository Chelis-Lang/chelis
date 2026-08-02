//! Compiler API ingress regressions for chelis#731 Phase 3.
//!
//! Every public Deep text boundary must consume the role-stamped carrier.
//! A bare name in a RuntimeExpr slot is therefore an ingress error, not a
//! legacy tree that reaches a downstream checker or authoring helper.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{
    CheckRequest, DeepCallGraphRequest, DeepOutlineRequest, DeepReferencesRequest, ParseRequest,
    RenameRequest, SourceKind,
};

const VALID_MODULE: &str =
    "(module {} phase3.ingress (def {} f (lit {type: (t-prim {} int64)} 1)))";
const BARE_NAME_BODY_MODULE: &str = "(module {} phase3.ingress (def {} f unwrapped_name))";
const SEQUENTIAL_LET_SHADOW: &str = r#"(module {}
  phase3.let_scope
  (def {} target
    (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))
  (def {} caller
    (fn {} (params {})
      (let {}
        (bind {}
          target (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x))
          result (app {} (var {} target) (lit {} true)))
        (var {} result)))))"#;

const SEQUENTIAL_LET_TOP_LEVEL_CONTROL: &str = r#"(module {}
  phase3.let_scope_control
  (def {} target
    (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))
  (def {} caller
    (fn {} (params {})
      (let {}
        (bind {}
          result (app {} (var {} target) (lit {} true))
          target (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))
        (var {} result)))))"#;

const RECURSIVE_PATTERN_SHADOW: &str = r#"(module {}
  phase3.pattern_scope
  (def {} target
    (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))
  (def {} caller
    (fn {} (params {})
      (match {} (tuple {} (lit {} true) (lit {} false))
        (arm {}
          (pat-as {} whole
            (pat-tuple {} (pat-var {} target) (pat-wild {})))
          (var {} target)
          (var {} target))))))"#;

fn assert_stamp_error(error: compiler::CompilerError, stage: &str) {
    assert_eq!(error.stage, stage);
    assert_eq!(error.errors.len(), 1);
    assert_eq!(error.errors[0].kind, "deep_stamp_error");
    assert!(error.errors[0].message.contains("bare name"));
    assert!(error.errors[0].span.is_some());
}

#[test]
fn generic_parse_and_check_use_stamped_deep_ingress() {
    let parse_error = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("Deep parse must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(parse_error, "parse");

    let check_error = compiler::check(CheckRequest {
        source_kind: SourceKind::Deep,
        source: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("Deep check must share the stamped ingress boundary");
    assert_stamp_error(check_error, "parse");

    compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: VALID_MODULE.to_string(),
    })
    .expect("well-formed Deep still parses");
}

#[test]
fn generic_parse_wire_preserves_unknown_form_head_and_metadata() {
    let parsed = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: r#"(def {} f
  (future-form {sentinel_meta: "keep-me"}
    (lit {} 1)))"#
            .to_string(),
    })
    .expect("unknown forms remain available for downstream diagnostics");

    let wire = serde_json::to_string(&parsed.deep_ast.expect("Deep AST must be present"))
        .expect("wire AST must serialize");
    assert!(
        wire.contains("future-form"),
        "unknown head was lost: {wire}"
    );
    assert!(
        wire.contains("sentinel_meta"),
        "metadata key was lost: {wire}"
    );
    assert!(wire.contains("keep-me"), "metadata value was lost: {wire}");
}

#[test]
fn read_only_authoring_ingresses_use_the_stamp_gate() {
    let outline_error = compiler::deep_outline(DeepOutlineRequest {
        module: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("outline must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(outline_error, "deep-outline");

    let references_error = compiler::deep_references(DeepReferencesRequest {
        module: BARE_NAME_BODY_MODULE.to_string(),
        symbol: "f".to_string(),
    })
    .expect_err("references must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(references_error, "deep-references");

    let graph_error = compiler::deep_call_graph(DeepCallGraphRequest {
        module: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("call graph must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(graph_error, "deep-call-graph");

    compiler::deep_outline(DeepOutlineRequest {
        module: VALID_MODULE.to_string(),
    })
    .expect("well-formed stamped module still outlines");
    compiler::deep_references(DeepReferencesRequest {
        module: VALID_MODULE.to_string(),
        symbol: "f".to_string(),
    })
    .expect("well-formed stamped module still supports references");
    compiler::deep_call_graph(DeepCallGraphRequest {
        module: VALID_MODULE.to_string(),
    })
    .expect("well-formed stamped module still supports call graph");
}

#[test]
fn arm_pattern_binders_shadow_top_level_references_and_call_edges() {
    let module = r#"(module {}
  phase3.arm_scope
  (def {} target
    (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))
  (def {} shadowed
    (fn {} (params {} (x {type: (t-prim {} bool)}))
      (match {} (var {} x)
        (arm {} (pat-var {} target)
          (var {} target)
          (var {} target)))))
  (def {} unshadowed
    (fn {} (params {}) (app {} (var {} target) (lit {} true)))))"#;

    let references = compiler::deep_references(DeepReferencesRequest {
        module: module.to_string(),
        symbol: "target".to_string(),
    })
    .expect("references query");
    assert_eq!(
        references.references.len(),
        1,
        "only the genuinely unshadowed top-level reference remains: {:?}",
        references.references
    );
    assert_eq!(
        references.references[0].caller,
        "phase3.arm_scope.unshadowed"
    );

    let graph = compiler::deep_call_graph(DeepCallGraphRequest {
        module: module.to_string(),
    })
    .expect("call graph query");
    assert_eq!(
        graph.edges.len(),
        1,
        "pattern-local calls must not become top-level call-graph edges: {:?}",
        graph.edges
    );
    assert_eq!(graph.edges[0].caller, "phase3.arm_scope.unshadowed");
    assert_eq!(graph.edges[0].callee, "phase3.arm_scope.target");

    let renamed = compiler::rename(RenameRequest {
        module: module.to_string(),
        function_name: "target".to_string(),
        new_name: "renamed".to_string(),
        preimage_sha256: None,
    })
    .expect("rename preserves pattern-local shadowing");
    assert_eq!(renamed.renamed_references, 1);
    assert!(renamed.module_deep.contains("(def {} renamed"));
    assert!(renamed.module_deep.contains("(arm {} (pat-var {} target)"));
    assert_eq!(renamed.module_deep.matches("(var {} target)").count(), 2);
    assert_eq!(renamed.module_deep.matches("(var {} renamed)").count(), 1);
}

#[test]
fn later_let_rhs_sees_prior_binding_for_references_call_graph_and_rename() {
    let references = compiler::deep_references(DeepReferencesRequest {
        module: SEQUENTIAL_LET_SHADOW.to_string(),
        symbol: "target".to_string(),
    })
    .expect("references query");
    assert!(references.references.is_empty(), "{references:#?}");

    let graph = compiler::deep_call_graph(DeepCallGraphRequest {
        module: SEQUENTIAL_LET_SHADOW.to_string(),
    })
    .expect("call graph query");
    assert!(graph.edges.is_empty(), "{graph:#?}");

    let renamed = compiler::rename(RenameRequest {
        module: SEQUENTIAL_LET_SHADOW.to_string(),
        function_name: "target".to_string(),
        new_name: "renamed".to_string(),
        preimage_sha256: None,
    })
    .expect("rename preserves the prior local binding");
    assert_eq!(renamed.renamed_references, 0, "{renamed:#?}");
    assert!(renamed.module_deep.contains("(def {} renamed"));
    assert!(renamed.module_deep.contains("(app {} (var {} target)"));
}

#[test]
fn earlier_let_rhs_still_sees_top_level_before_shadowing_binding() {
    let references = compiler::deep_references(DeepReferencesRequest {
        module: SEQUENTIAL_LET_TOP_LEVEL_CONTROL.to_string(),
        symbol: "target".to_string(),
    })
    .expect("references query");
    assert_eq!(references.references.len(), 1, "{references:#?}");

    let graph = compiler::deep_call_graph(DeepCallGraphRequest {
        module: SEQUENTIAL_LET_TOP_LEVEL_CONTROL.to_string(),
    })
    .expect("call graph query");
    assert_eq!(graph.edges.len(), 1, "{graph:#?}");

    let renamed = compiler::rename(RenameRequest {
        module: SEQUENTIAL_LET_TOP_LEVEL_CONTROL.to_string(),
        function_name: "target".to_string(),
        new_name: "renamed".to_string(),
        preimage_sha256: None,
    })
    .expect("rename updates the unshadowed earlier RHS");
    assert_eq!(renamed.renamed_references, 1, "{renamed:#?}");
    assert!(renamed.module_deep.contains("(app {} (var {} renamed)"));
    assert!(
        renamed.module_deep.contains("target"),
        "the later local binder must retain its old spelling: {}",
        renamed.module_deep
    );
}

#[test]
fn recursive_pattern_binders_scope_guard_and_body() {
    let references = compiler::deep_references(DeepReferencesRequest {
        module: RECURSIVE_PATTERN_SHADOW.to_string(),
        symbol: "target".to_string(),
    })
    .expect("references query");
    assert!(references.references.is_empty(), "{references:#?}");

    let graph = compiler::deep_call_graph(DeepCallGraphRequest {
        module: RECURSIVE_PATTERN_SHADOW.to_string(),
    })
    .expect("call graph query");
    assert!(graph.edges.is_empty(), "{graph:#?}");
}

#[test]
fn read_only_authoring_functions_do_not_normalize_stamped_nodes_to_lists() {
    let source = include_str!("../../chelis-deep/src/authoring.rs");
    let ranges = [
        ("pub fn outline(", "pub fn references("),
        ("pub fn references(", "pub fn call_graph("),
        ("pub fn call_graph(", "pub fn rename_function("),
    ];
    for (start, end) in ranges {
        let body = source
            .split_once(start)
            .and_then(|(_, rest)| rest.split_once(end).map(|(body, _)| body))
            .unwrap_or_else(|| panic!("missing authoring source range {start}..{end}"));
        assert!(
            !body.contains("normalize_for_mutation"),
            "read-only {start} must traverse the stamped Node carrier directly"
        );
    }
}
