//! Authoring scope regressions for chelis#731 Phase 3.
//!
//! The read-only authoring surfaces (references, call graph) and the rename
//! rewriter must model the scopes a Deep module actually introduces: a `let`
//! binds pair by pair, so a later right-hand side sees an earlier binder while
//! an earlier one still sees the top level, and an arm's pattern binders cover
//! that arm's guard and body. A binder that shadows a top-level name must
//! therefore remove its references, its call-graph edges, and its renames.
//!
//! These cases are carrier-independent: they exercise the public compiler API
//! text boundary as it stands today. The ingress migration that routes these
//! same entry points through `parse_and_stamp_file` is chelis#1088 and is not
//! required by anything here.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{DeepCallGraphRequest, DeepReferencesRequest, RenameRequest};

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
    assert_eq!(renamed.renamed_references.get(), 1);
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
    assert_eq!(renamed.renamed_references.get(), 0, "{renamed:#?}");
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
    assert_eq!(renamed.renamed_references.get(), 1, "{renamed:#?}");
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
        // Both spellings: the mutating helper's current name and the name
        // it carried before the read-only paths stopped calling it. A
        // rename must not turn this tripwire vacuous.
        for helper in ["normalize_for_mutation", "normalize_to_list"] {
            assert!(
                !body.contains(helper),
                "read-only {start} must traverse the node carrier directly, \
                 but it calls {helper}"
            );
        }
    }
}
