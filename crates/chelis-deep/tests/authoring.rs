use chelis_deep::authoring::{
    call_graph, change_signature, outline, references, rename_function, replace_function,
};

const MODULE: &str = r#"(module {}
  demo.auth
  (export {} caller target)
  (defsig {}
    target
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    target
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} add) (var {} x) (lit {type: (t-prim {} f32)} 1.0))))
  (defsig {}
    caller
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    caller
    (fn {}
      (params {} (y {type: (t-prim {} f32)}))
      (app {} (var {} target) (var {} y)))))
"#;

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str_strict(source).expect("deep parses")
}

#[test]
fn outline_lists_functions_with_canonical_node_hash_preimage_text() {
    let exprs = parse(MODULE);
    let summary = outline(&exprs).expect("outline");
    assert_eq!(summary.module_name, "demo.auth");
    assert_eq!(summary.functions.len(), 2);
    let target = summary
        .functions
        .iter()
        .find(|function| function.name == "target")
        .expect("target function");
    assert_eq!(target.qualified_name, "demo.auth.target");
    assert_eq!(target.params, vec!["x"]);
    assert!(target.has_defsig);
    assert!(target.def_deep.contains("target"));
}

#[test]
fn call_graph_uses_structural_calls_and_ignores_shadowed_names() {
    let source = r#"(module {}
  demo.shadow
  (defsig {} target (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {} target (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))
  (defsig {} caller (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    caller
    (fn {}
      (params {} (target {type: (t-prim {} f32)}))
      (var {} target))))
"#;
    let graph = call_graph(&parse(source)).expect("call graph");
    assert!(
        graph.edges.is_empty(),
        "parameter `target` shadows the top-level function: {graph:?}"
    );

    let graph = call_graph(&parse(MODULE)).expect("call graph");
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.edges[0].caller, "demo.auth.caller");
    assert_eq!(graph.edges[0].callee, "demo.auth.target");
    assert_eq!(graph.edges[0].path, "body.0");
}

#[test]
fn rename_updates_defsig_def_exports_and_unshadowed_calls() {
    let report = rename_function(&parse(MODULE), "target", "renamed").expect("rename");
    let rewritten = chelis_deep::printer::print_canonical(&report.module);
    let summary = outline(&report.module).expect("outline");
    assert!(rewritten.contains("(export {} caller renamed)"));
    assert!(
        summary
            .functions
            .iter()
            .any(|function| { function.name == "renamed" && function.has_defsig })
    );
    assert!(rewritten.contains("(app {} (var {} renamed) (var {} y))"));
    assert_eq!(report.renamed_references, 1);
    assert_eq!(report.residual_old_references, 0);
}

#[test]
fn replace_function_preserves_declaration_position() {
    let new_decls = r#"(defsig {}
  target
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
(def {}
  target
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (var {} x)))
"#;
    let report = replace_function(&parse(MODULE), "target", &parse(new_decls)).expect("replace");
    let decls: Vec<String> = chelis_deep::authoring::outline(&report.module)
        .expect("outline")
        .functions
        .into_iter()
        .map(|function| function.name)
        .collect();
    assert_eq!(decls, vec!["target", "caller"]);
    assert!(chelis_deep::printer::print_canonical(&report.module).contains("(var {} x)"));
}

#[test]
fn change_signature_reorders_direct_calls_and_checks_completeness() {
    let source = r#"(module {}
  demo.sig
  (defsig {}
    pair
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {}
    pair
    (fn {}
      (params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)}))
      (app {} (var {} sub) (var {} x) (var {} y))))
  (defsig {}
    caller
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {}
    caller
    (fn {}
      (params {} (a {type: (t-prim {} f32)}) (b {type: (t-prim {} f32)}))
      (app {} (var {} pair) (var {} a) (var {} b)))))
"#;
    let new_defsig = r#"(defsig {}
  pair
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))"#;
    let new_params = r#"(params {} (y {type: (t-prim {} f32)}) (x {type: (t-prim {} f32)}))"#;

    let report = change_signature(
        &parse(source),
        "pair",
        &parse(new_defsig)[0],
        &parse(new_params)[0],
        &["y".to_string(), "x".to_string()],
        &[],
    )
    .expect("change signature");

    let rewritten = chelis_deep::printer::print_canonical(&report.module);
    assert!(
        rewritten.contains("(params {} (y {type: (t-prim {} f32)}) (x {type: (t-prim {} f32)}))")
    );
    assert!(rewritten.contains("(app {} (var {} pair) (var {} b) (var {} a))"));
    assert_eq!(report.rewritten_calls, 1);
    assert_eq!(report.stale_calls, 0);
}

#[test]
fn change_signature_rejects_duplicate_argument_order_before_rewrite() {
    let source = r#"(module {}
  demo.sig
  (defsig {}
    pair
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {}
    pair
    (fn {}
      (params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)}))
      (app {} (var {} sub) (var {} x) (var {} y))))
  (defsig {}
    caller
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {}
    caller
    (fn {}
      (params {} (a {type: (t-prim {} f32)}) (b {type: (t-prim {} f32)}))
      (app {} (var {} pair) (var {} a) (var {} b)))))
"#;
    let new_defsig = r#"(defsig {}
  pair
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))"#;
    let new_params = r#"(params {} (x {type: (t-prim {} f32)}) (y {type: (t-prim {} f32)}))"#;

    let error = change_signature(
        &parse(source),
        "pair",
        &parse(new_defsig)[0],
        &parse(new_params)[0],
        &["x".to_string(), "x".to_string()],
        &[],
    )
    .expect_err("duplicate argument_order must fail before rewriting calls");

    assert!(
        error
            .to_string()
            .contains("argument_order contains duplicate old parameter `x`"),
        "error should explain the malformed cascade request: {error}"
    );
}

/// chelis#1136 red-team M1: an unknown form (a `List` whose head is not in
/// the 62-tag vocabulary - the shape the chelis#1088 stamped ingress
/// deliberately preserves) must still be TRAVERSED by the read-only
/// walkers. The salvaged NodeView traversal skipped it, so `references`
/// dropped to 0 on the List carrier while the mutating walker still
/// rewrote inside it, and `rename_function` failed its cascade
/// accounting. Lenient parse: `parse_str` accepts the unknown head that
/// `parse_str_strict` rejects.
#[test]
fn unknown_form_subtrees_are_traversed_not_dropped() {
    let source = r#"(module {}
  m.unknown_form
  (def {} target
    (fn {} (params {} (x {type: (t-prim {} bool)})) (var {} x)))
  (def {} caller
    (fn {} (params {})
      (future-form {} (app {} (var {} target) (lit {} true))))))"#;
    let exprs = chelis_deep::parser::parse_str(source).expect("lenient deep parses");
    let refs = references(&exprs, "target").expect("references");
    assert_eq!(
        refs.references.len(),
        1,
        "the reference inside the unknown form must be found"
    );
    let report = rename_function(&exprs, "target", "renamed").expect("rename");
    assert_eq!(report.renamed_references, 1);
    assert_eq!(report.residual_old_references, 0);
}
