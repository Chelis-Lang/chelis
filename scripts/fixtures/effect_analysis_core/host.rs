use chelis_deep::Expr;
use chelis_effects::{check_program, check_effects_with_context, def_effect_rows,
    def_effect_rows_in_context, validate_build_target};
use chelis_types::{CheckedProgram, InferResult, check_ir_program, check_ir_with_context,
    build_type_env_from_library};
use serde_json::{Value, json};

fn parse(source: &str, format: &str) -> Result<Vec<Expr>, String> {
    if format == "deep" {
        return chelis_deep::parse_and_stamp_file(source, &chelis_syntax_host::NativeStack)
            .map_err(|error| format!("{error:?}"));
    }
    let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .map_err(|error| format!("{error:?}"))?;
    chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .map_err(|error| format!("{error:?}"))
}

fn checked(program: &CheckedProgram) -> Value {
    let bytes = bincode::serialize(program).unwrap();
    let decoded: CheckedProgram = bincode::deserialize(&bytes).unwrap();
    assert_eq!(bytes, bincode::serialize(&decoded).unwrap());
    json!({"bytes": bytes, "payload": program,
        "tree": program.annotated_exprs(), "type_env": program.type_env(),
        "signatures": program.signature_inference(), "linearity": program.linearity(),
        "typed_nodes": program.infer_stats().typed_nodes,
        "total_nodes": program.infer_stats().total_nodes})
}

fn type_errors(report: &InferResult) -> Value {
    json!({"typed_nodes": report.typed_nodes, "total_nodes": report.total_nodes,
        "errors": report.errors.iter().map(|error| json!({
            "kind": error.kind.diagnostic_name(), "identifier": error.kind.unresolved_identifier(),
            "message": error.message, "suggestions": error.suggestions,
            "severity_bits": error.severity.to_bits(), "expected": error.expected,
            "got": error.got, "span_offset": error.span_offset, "span_id": error.span_id,
        })).collect::<Vec<_>>()})
}

fn errors<T: std::fmt::Debug>(items: &[T]) -> Value {
    json!(items.iter().map(|item| format!("{item:?}")).collect::<Vec<_>>())
}

fn effect_result(program: &CheckedProgram, library: Option<&CheckedProgram>) -> Value {
    let rows = match library {
        Some(library) => def_effect_rows_in_context(Some(library), program),
        None => def_effect_rows(program),
    };
    let row_value = rows.iter().map(|(name, row)| (name.clone(),
        json!(row.iter().map(|effect| format!("{effect:?}")).collect::<Vec<_>>())))
        .collect::<serde_json::Map<_, _>>();
    let result = match library {
        Some(library) => check_effects_with_context(library, program),
        None => check_program(program),
    };
    match result {
        Ok(output) => {
            assert_eq!(output.type_env(), program.type_env());
            assert_eq!(output.signature_inference(), program.signature_inference());
            assert_eq!(output.linearity(), program.linearity());
            let targets = ["c", "hip", "metal"].into_iter().map(|target| {
                let value = match validate_build_target(&output, target) {
                    Ok(()) => json!({"status": "accepted"}),
                    Err(items) => json!({"status": "rejected", "errors": errors(&items)}),
                };
                (target.to_string(), value)
            }).collect::<serde_json::Map<_, _>>();
            json!({"status": "accepted", "input": checked(program), "rows": row_value,
                "output": checked(&output), "targets": targets})
        }
        Err(items) => {
            assert!(!items.is_empty());
            json!({"status": "rejected", "stage": "effects", "input": checked(program),
                "rows": row_value, "errors": items.iter().map(|error| json!({
                    "kind": format!("{:?}", error.kind), "message": error.message,
                    "suggestions": error.suggestions,
                })).collect::<Vec<_>>()})
        }
    }
}

fn case(source: &str, format: &str) -> Value {
    match parse(source, format) {
        Err(message) => json!({"status": "rejected", "stage": "admission", "error": message}),
        Ok(expressions) => match check_ir_program(&expressions) {
            Err(report) => json!({"status": "rejected", "stage": "types", "report": type_errors(&report)}),
            Ok(program) => effect_result(&program, None),
        }
    }
}

fn snapshot() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).unwrap();
    let mut cases = serde_json::Map::new();
    for entry in corpus["cases"].as_array().unwrap() {
        let name = entry[0].as_str().unwrap();
        let result = case(entry[3].as_str().unwrap(), entry[1].as_str().unwrap());
        assert_eq!(result["status"] == "accepted", entry[2].as_bool().unwrap(), "{name}: {result}");
        assert!(cases.insert(name.to_string(), result).is_none());
    }
    let library_source = corpus["library"].as_str().unwrap();
    let library_exprs = parse(library_source, "surf").unwrap();
    let context = build_type_env_from_library(&library_exprs).unwrap();
    let library = check_program(&check_ir_program(&library_exprs).unwrap()).unwrap();
    let library_before = checked(&library);
    let context_before = bincode::serialize(&context).unwrap();
    let mut extensions = serde_json::Map::new();
    for entry in corpus["extensions"].as_array().unwrap() {
        let name = entry[0].as_str().unwrap();
        let source = entry[2].as_str().unwrap();
        let program = check_ir_with_context(&context, &parse(source, "surf").unwrap()).unwrap();
        let result = effect_result(&program, Some(&library));
        assert_eq!(result["status"] == "accepted", entry[1].as_bool().unwrap(), "{name}: {result}");
        assert_eq!(result, effect_result(&program, Some(&library)));
        let extension = parse(source, "surf").unwrap();
        let owned_names = extension.iter().filter_map(definition_name).map(str::to_owned).collect::<Vec<_>>();
        let combined = library_exprs.iter()
            .filter(|expr| definition_name(expr).is_none_or(|name| !owned_names.iter().any(|owned| owned == name)))
            .cloned().chain(extension).collect::<Vec<_>>();
        let monolithic = match check_ir_program(&combined) {
            Ok(program) => effect_result(&program, None),
            Err(report) => json!({"status": "rejected", "stage": "types", "report": type_errors(&report)}),
        };
        assert_eq!(result["status"], monolithic["status"]);
        assert!(extensions.insert(name.to_string(), json!({"contextual": result,
            "monolithic": monolithic})).is_none());
        assert_eq!(checked(&library), library_before);
        assert_eq!(bincode::serialize(&context).unwrap(), context_before);
    }
    println!("EFFECT_HOST_SNAPSHOT {}", json!({"cases": cases, "extensions": extensions,
        "library": library_before, "context": {"bytes": context_before, "payload": context}}));
}

fn definition_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        chelis_deep::ExprCarrier::DecodedNode(chelis_deep::DeepTag::Def | chelis_deep::DeepTag::Defsig, _, children) =>
            match children.first() {
                Some(Expr::Atom(chelis_deep::Atom::Name(name), _)) => Some(name.as_str()),
                _ => None,
            },
        _ => None,
    }
}

fn main() {
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(snapshot).unwrap()
        .join().expect("the host snapshot must succeed");
}
