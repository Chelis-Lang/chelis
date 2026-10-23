use chelis_deep::Expr;
use chelis_types::{CheckedProgram, InferResult, TypeEnv, build_type_env_from_library,
    check_ir_program, check_ir_with_context};
use serde_json::{Value, json};
use std::sync::Arc;

fn parse(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source, &chelis_syntax_host::NativeStack)
        .expect("the parity fixture must parse")
}

fn surf(source: &str) -> Vec<Expr> {
    let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("the parity Surf source must parse");
    let deep = chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .expect("the parity Surf source must desugar");
    chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .expect("the parity Surf source must expand").into_exprs()
}

fn checked_value(checked: CheckedProgram) -> Value {
    let bytes = bincode::serialize(&checked).expect("the checked product must serialize");
    let decoded: CheckedProgram = bincode::deserialize(&bytes).expect("the checked product must decode");
    assert_eq!(bincode::serialize(&decoded).unwrap(), bytes);
    let stats = checked.infer_stats();
    json!({"status": "accepted", "checked_bytes": bytes,
        "tree": checked.exprs(), "types": checked.type_env(),
        "typed_nodes": stats.typed_nodes, "total_nodes": stats.total_nodes})
}

fn result_value(result: Result<CheckedProgram, InferResult>) -> Value {
    match result {
        Ok(checked) => checked_value(checked),
        Err(report) => {
            assert!(!report.errors.is_empty());
            json!({"status": "rejected", "typed_nodes": report.typed_nodes,
                "total_nodes": report.total_nodes,
                "diagnostics": report.errors.iter().map(|error| json!({
                    "kind": error.kind.diagnostic_name(), "message": error.message,
                    "identifier": error.kind.unresolved_identifier(),
                    "suggestions": error.suggestions, "severity_bits": error.severity.to_bits(),
                    "expected": error.expected, "got": error.got,
                    "span_offset": error.span_offset, "span_id": error.span_id,
                })).collect::<Vec<_>>()})
        }
    }
}

fn snapshot() {
    let cases = [
        ("generic", true, r#"
(def {} identity (fn {} (params {} value) (var {} value)))
(def {} integer (app {} (var {} identity) (lit {type: (t-prim {} i32)} 1)))
(def {} boolean (app {} (var {} identity) (lit {type: (t-prim {} bool)} true)))
"#),
        ("tuple", true, r#"
(def {} pair (tuple {} (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} bool)} true)))
"#),
        ("tensor", true, r#"
(defsig {} identity_tensor (t-fn {} (t-tensor {} (d-name {} width) (t-prim {} f32)) (t-tensor {} (d-name {} width) (t-prim {} f32))))
(def {} identity_tensor (fn {} (params {} x) (var {} x)))
"#),
        ("named_rank", true, r#"
(defsig {} identity_rank (r) (t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) (t-tensor {} (d-rank {} r) (t-prim {} f32))))
(def {} identity_rank (fn {} (params {} x) (var {} x)))
"#),
        ("anchored_rank", true, r#"
(defsig {} identity_anchored (pre post) (t-fn {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)) (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32))))
(def {} identity_anchored (fn {} (params {} x) (var {} x)))
"#),
        ("nominal", true, r#"
(deftype {} Box () (variant {} Box (field {} value (t-prim {} i32))))
(def {} boxed (record {} Box (kv {} value (lit {type: (t-prim {} i32)} 1))))
"#),
        ("unbound", false, r#"
(def {span: "parity:unbound"} missing (var {span: "parity:missing"} absent))
"#),
        ("precision", false, r#"
(def {} mismatch (app {span: "parity:precision"} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f64)} 2.0)))
"#),
        ("branch", false, r#"
(def {} mismatch (if {span: "parity:branch"} (lit {type: (t-prim {} bool)} true) (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} bool)} false)))
"#),
        ("bad_nominal", false, r#"
(defsig {span: "parity:nominal"} bad (t-fn {} (t-adt {} Missing) (t-prim {} i32)))
(def {} bad (fn {} (params {} x) (lit {type: (t-prim {} i32)} 1)))
"#),
    ];
    let mut records = serde_json::Map::new();
    for (name, accepted, source) in cases {
        let result = check_ir_program(&parse(source));
        assert_eq!(result.is_ok(), accepted, "the fixture disposition differs: {name}: {:?}",
            result.as_ref().err().map(|report| &report.errors));
        records.insert(name.into(), result_value(result));
    }
    let surf_cases: Vec<(String, bool, String)> = serde_json::from_str(SURF_CORPUS_JSON)
        .expect("the complete current parity corpus must decode");
    for (name, accepted, source) in surf_cases {
        let result = check_ir_program(&surf(&source));
        assert_eq!(result.is_ok(), accepted, "the Surf fixture disposition differs: {name}: {:?}",
            result.as_ref().err().map(|report| &report.errors));
        assert!(records.insert(name, result_value(result)).is_none());
    }
    let library = parse("(def {} library_id (fn {} (params {} value) (var {} value)))");
    let context = build_type_env_from_library(&library).expect("the library must check");
    let bytes = bincode::serialize(&context).expect("the context must serialize");
    let decoded: TypeEnv = bincode::deserialize(&bytes).expect("the context must decode");
    assert_eq!(bincode::serialize(&decoded).unwrap(), bytes);
    let context = Arc::new(context);
    let sources = [
        "(def {} int_use (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 1)))",
        "(def {} bool_use (app {} (var {} library_id) (lit {type: (t-prim {} bool)} true)))",
        "(def {} bad_use (var {span: \"parity:context\"} absent))",
    ];
    let expected = sources.iter().map(|source| {
        let program = parse(source);
        let direct = result_value(check_ir_with_context(&context, &program));
        let cached = result_value(check_ir_with_context(&decoded, &program));
        assert_eq!(direct, cached, "the cache round trip changes the check");
        direct
    }).collect::<Vec<_>>();
    let workers = (0..8).map(|_| {
        let shared = Arc::clone(&context);
        std::thread::spawn(move || sources.iter().map(|source|
            result_value(check_ir_with_context(&shared, &parse(source)))).collect::<Vec<_>>())
    }).collect::<Vec<_>>();
    for worker in workers {
        assert_eq!(worker.join().expect("the shared context check must not panic"), expected);
    }
    assert_eq!(bincode::serialize(context.as_ref()).unwrap(), bytes);
    println!("TYPE_HOST_PARITY {}", json!({"cases": records,
        "context": {"bytes": bytes, "checks": expected, "threads": 8}}));
}

fn main() {
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        snapshot();
        send.send(()).unwrap();
    });
    receive.recv_timeout(std::time::Duration::from_secs(60))
        .expect("the shared context fixture failed or stalled");
}
