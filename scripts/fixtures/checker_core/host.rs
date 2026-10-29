use chelis_deep::Expr;
use checker_data::InferResult;
use chelis_types::{CheckedProgram, TypeEnv, build_compiled_library_context,
    build_compiled_library_context_with_base, check_ir_program, check_typed_program,
    check_ir_with_context, infer_program, infer_ir_program, resolve_declared_surface};
use serde_json::{Value, json};

fn parse(source: &str, format: &str) -> Result<Vec<Expr>, String> {
    if format == "deep" {
        return chelis_deep::parse_and_stamp_file(source, &chelis_syntax_host::NativeStack)
            .map_err(|error| format!("{error:?}"));
    }
    let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .map_err(|error| format!("{error:?}"))?;
    let expressions = chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .map_err(|error| format!("{error:?}"))?;
    chelis_macros::expand_program(&expressions, &chelis_macros::ExpansionOptions::default())
        .map(|expanded| expanded.into_exprs()).map_err(|error| format!("{error:?}"))
}

fn diagnostics(report: &InferResult) -> Value {
    json!({"typed_nodes": report.typed_nodes, "total_nodes": report.total_nodes,
        "errors": report.errors.iter().map(|error| json!({
            "debug": format!("{error:?}"), "kind": error.kind.diagnostic_name(),
            "identifier": error.kind.unresolved_identifier(), "message": error.message,
            "suggestions": error.suggestions, "severity_bits": error.severity.to_bits(),
            "expected": error.expected, "got": error.got,
            "span_offset": error.span_offset, "span_id": error.span_id,
        })).collect::<Vec<_>>()})
}

fn checked(program: &CheckedProgram) -> Value {
    let bytes = bincode::serialize(program).unwrap();
    let restored: CheckedProgram = bincode::deserialize(&bytes).unwrap();
    assert_eq!(bincode::serialize(&restored).unwrap(), bytes);
    json!({"bytes": bytes, "payload": program, "tree": program.annotated_exprs(),
        "type_env": program.type_env(), "signatures": program.signature_inference(),
        "local_ascriptions": program.local_ascriptions(), "linearity": program.linearity(),
        "adt_registry": program.adt_registry(), "library_proof_id": program.library_proof_id(),
        "typed_nodes": program.infer_stats().typed_nodes,
        "total_nodes": program.infer_stats().total_nodes})
}

fn result(value: Result<CheckedProgram, InferResult>) -> Value {
    match value {
        Ok(program) => {
            let effects = match chelis_effects::check_program(&program) {
                Ok(output) => {
                    let linearity = match chelis_types::check_linearity(&output) {
                        Ok(owned) => json!({"status": "accepted", "checked": checked(&owned),
                            "copy_repairs": format!("{:?}", chelis_types::copy_repairs(&owned, None))}),
                        Err(errors) => json!({"status": "rejected", "errors": format!("{errors:?}")}),
                    };
                    json!({"status": "accepted", "checked": checked(&output), "linearity": linearity})
                }
                Err(errors) => json!({"status": "rejected", "errors": format!("{errors:?}")}),
            };
            json!({"status": "accepted", "checked": checked(&program), "effects": effects})
        }
        Err(report) => json!({"status": "rejected", "report": diagnostics(&report)}),
    }
}

fn context(value: &TypeEnv, program: &CheckedProgram) -> Value {
    let bytes = bincode::serialize(value).unwrap();
    let restored: TypeEnv = bincode::deserialize(&bytes).unwrap();
    assert_eq!(bincode::serialize(&restored).unwrap(), bytes);
    json!({"bytes": bytes, "payload": value, "library_def_count": value.library_def_count(),
        "matches_program": value.matches_checked_program(program)})
}

fn snapshot() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).unwrap();
    let mut cases = serde_json::Map::new();
    for entry in corpus["cases"].as_array().unwrap() {
        let value = match parse(entry["source"].as_str().unwrap(), entry["format"].as_str().unwrap()) {
            Err(error) => json!({"status": "admission-rejected", "error": error}),
            Ok(expressions) => json!({"status": "admitted", "source_tree": expressions,
                "inference": diagnostics(&infer_program(&expressions)),
                "ir_inference": diagnostics(&infer_ir_program(&expressions)),
                "ir": result(check_ir_program(&expressions)),
                "typed": result(check_typed_program(&expressions)),
                "declared_surface": format!("{:?}", resolve_declared_surface(&expressions))}),
        };
        assert!(cases.insert(entry["id"].as_str().unwrap().into(), value).is_none());
    }
    let library_exprs = parse(corpus["library"].as_str().unwrap(), "surf").unwrap();
    let (base, library) = build_compiled_library_context(&library_exprs).unwrap();
    let layer_exprs = parse(corpus["layer"].as_str().unwrap(), "surf").unwrap();
    let (layered, layer_program) = build_compiled_library_context_with_base(&base, &layer_exprs).unwrap();
    let base_before = context(&base, &library);
    let layered_before = context(&layered, &layer_program);
    let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&base).unwrap()).unwrap();
    let decoded_layer: TypeEnv = bincode::deserialize(&bincode::serialize(&layered).unwrap()).unwrap();
    let mut extensions = serde_json::Map::new();
    for entry in corpus["extensions"].as_array().unwrap() {
        let expressions = parse(entry["source"].as_str().unwrap(), "surf").unwrap();
        let direct = result(check_ir_with_context(&base, &expressions));
        let restored = result(check_ir_with_context(&decoded, &expressions));
        assert_eq!(direct, restored);
        let second = result(check_ir_with_context(&layered, &expressions));
        let restored_second = result(check_ir_with_context(&decoded_layer, &expressions));
        assert_eq!(second, restored_second);
        let composition = check_ir_with_context(&base, &expressions).ok()
            .and_then(|program| CheckedProgram::compose(&library, &program));
        let foreign = check_ir_program(&expressions).ok()
            .and_then(|program| CheckedProgram::compose(&library, &program));
        assert!(foreign.is_none());
        let combined = library_exprs.iter().cloned().chain(expressions).collect::<Vec<_>>();
        assert!(extensions.insert(entry["id"].as_str().unwrap().into(), json!({
            "contextual": direct, "restored": restored, "layered": second,
            "restored_layered": restored_second, "monolithic": result(check_ir_program(&combined)),
            "composition": composition.as_ref().map(checked), "foreign_composition": foreign.is_some(),
        })).is_none());
    }
    let isolation = std::thread::scope(|scope| {
        let handles = (0..8).map(|_| scope.spawn(|| {
            let expressions = parse("(def {} isolated (app {} (var {} identity) (lit {type: (t-prim {} i64)} 1)))", "deep").unwrap();
            result(check_ir_with_context(&base, &expressions))
        })).collect::<Vec<_>>();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
    });
    assert!(isolation.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(context(&base, &library), base_before);
    assert_eq!(context(&layered, &layer_program), layered_before);
    println!("CHECKER_HOST_SNAPSHOT {}", json!({"cases": cases, "extensions": extensions,
        "library": checked(&library), "context": base_before,
        "layer": checked(&layer_program), "layered_context": layered_before,
        "isolation": isolation}));
}

fn main() {
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(snapshot).unwrap()
        .join().expect("The host snapshot must succeed.");
}
