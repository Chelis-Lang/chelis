use chelis_dag_core::dag::Dag;
use chelis_effects::realizability::{compute_root_manifest, infer_realizability};
use chelis_ir::eval::eval_tensor_roots_exact;
use chelis_ir::host::try_lower_compiled_program_with_manifest;
use chelis_ir_core::lower::LowerDiagnostic;
use chelis_ir::lower::try_lower_program_to_library;
use chelis_ir_core::lowering_trace as trace;
use chelis_ir::ownership::{lower_dag_ownership, lower_host_ownership, verify_ownership};
use chelis_numeric_core::prim::Prim;
use chelis_types::manifest::ManifestedProgram;
use chelis_types::types::Target;
use chelis_types::{CheckedProgram, check_linearity, check_typed_program};
use serde_json::{Value, json};

fn graph(dag: &Dag) -> Value {
    json!({"json": serde_json::to_value(dag).unwrap(), "binary": bincode::serialize(dag).unwrap()})
}

fn diagnostic(error: &LowerDiagnostic) -> Value {
    json!({"message": error.message, "span": error.span, "span_id": error.span_id,
        "fatal": error.fatal, "host_control": error.requires_host_control(),
        "dimension_mismatch": error.is_dimension_mismatch(),
        "unsupported": error.unsupported().map(|value| format!("{value:?}")),
        "display": error.to_string()})
}

fn application(value: &trace::Application) -> Value {
    json!({"formal_types": value.formal_types, "actual_types": value.actual_types,
        "specialized": graph(&value.specialized), "arguments": value.arguments,
        "wrt_actuals": value.wrt_actuals, "remap": value.remap,
        "before_splice": graph(&value.before_splice), "after_splice": graph(&value.after_splice),
        "after_packing": graph(&value.after_packing), "result": format!("{:?}", value.result)})
}

fn observed(value: &trace::LoweringTrace) -> Value {
    let normalization = &value.normalization;
    json!({
        "contexts": value.contexts.iter().map(|context| json!({
            "parent": context.parent.map(|id| id.0), "kind": format!("{:?}", context.kind)
        })).collect::<Vec<_>>(),
        "gradients": value.gradients.iter().map(|gradient| json!({
            "context": gradient.context.0, "forward": graph(&gradient.forward),
            "output": gradient.output, "wrt": gradient.wrt, "backward": graph(&gradient.backward),
            "backward_output": gradient.backward_output, "gradients": gradient.gradients,
            "application": gradient.application.as_ref().map(application)
        })).collect::<Vec<_>>(),
        "boundaries": value.boundaries.iter().map(|boundary| json!({
            "context": boundary.context.0, "kind": format!("{:?}", boundary.kind)
        })).collect::<Vec<_>>(),
        "normalization": {"before_dce": graph(&normalization.before_dce),
            "after_dce": graph(&normalization.after_dce), "after_copies": graph(&normalization.after_copies),
            "after_drops": graph(&normalization.after_drops), "dce_remap": normalization.dce_remap,
            "copy_remap": normalization.copy_remap}
    })
}

fn checked(case: &Value) -> Result<CheckedProgram, Value> {
    let source = case["source"].as_str().unwrap();
    let deep = if case["frontend"] == "deep" {
        chelis_deep::parser::parse_and_stamp(source, &chelis_syntax_host::NativeStack)
            .map_err(|error| json!({"stage": "parse-deep", "error": format!("{error:?}")}))?
    } else {
        let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
            .map_err(|error| json!({"stage": "parse-surf", "error": format!("{error:?}")}))?;
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .map_err(|error| json!({"stage": "desugar", "error": format!("{error:?}")}))?
    };
    let result = check_typed_program(&deep)
        .map_err(|error| json!({"stage": "types", "error": format!("{:?}", error.errors)}))?;
    let result = chelis_effects::check_program(&result)
        .map_err(|error| json!({"stage": "effects", "error": format!("{error:?}")}))?;
    check_linearity(&result)
        .map_err(|error| json!({"stage": "linearity", "error": format!("{error:?}")}))
}

fn evaluate(dag: &Dag) -> Value {
    match eval_tensor_roots_exact(dag, dag.roots(), |_| None) {
        Ok(values) => json!({"status": "ok", "values": dag.roots().iter().map(|id| {
            let value = &values[id];
            json!({"root": id, "shape": value.shape, "precision": value.prim(),
                "storage": serde_json::to_value(value.storage()).unwrap(),
                "storage_binary": bincode::serialize(value.storage()).unwrap()})
        }).collect::<Vec<_>>()}),
        Err(error) => json!({"status": "error", "error": error}),
    }
}

fn snapshot(case: &Value) -> Value {
    let mut result = json!({"id": case["id"]});
    let checked = match checked(case) {
        Ok(checked) => checked,
        Err(error) => { result["check_error"] = error; return result; }
    };
    result["checked"] = json!({"json": serde_json::to_value(&checked).unwrap(),
        "binary": bincode::serialize(&checked).unwrap()});
    result["lowering"] = match try_lower_program_to_library(&checked) {
        Ok(library) => {
            let dag = library.dag();
            let ownership = lower_dag_ownership(dag.clone()).and_then(verify_ownership);
            let ownership = match ownership {
                Ok(verified) => json!({"status": "ok", "render": verified.render()}),
                Err(error) => json!({"status": "error", "error": format!("{error:?}")}),
            };
            json!({"status": "ok", "library_json": serde_json::to_value(&library).unwrap(),
                "library_binary": bincode::serialize(&library).unwrap(), "graph": graph(dag),
                "evaluation": evaluate(dag), "ownership": ownership})
        }
        Err(error) => json!({"status": "error", "diagnostic": diagnostic(&error)}),
    };
    result["trace"] = match chelis_ir::lowering_trace::try_lower_program_to_library_with_trace(&checked) {
        Ok((library, value)) => json!({"status": "ok", "library": graph(library.dag()), "trace": observed(&value)}),
        Err(error) => json!({"status": "error", "diagnostic": diagnostic(&error)}),
    };
    const PRIMS: &[Prim] = &[Prim::F32, Prim::Bool, Prim::Bf16, Prim::F16, Prim::Int32, Prim::Int64];
    let realizability = infer_realizability(&checked, PRIMS);
    let manifest = compute_root_manifest(&checked, &realizability);
    result["manifest"] = json!({"entries": manifest.entries.iter().map(|entry| json!({
        "name": entry.name, "path": format!("{:?}", entry.path), "def_name": entry.def_name,
        "ty": entry.ty, "lane": format!("{:?}", entry.lane), "required_inputs": entry.required_inputs,
        "reasons": format!("{:?}", entry.reasons)
    })).collect::<Vec<_>>()});
    result["compiled"] = match try_lower_compiled_program_with_manifest(&checked, &manifest) {
        Ok(compiled) => {
            let dag = compiled.dag.as_ref().map(graph);
            let host = compiled.host.as_ref().map(|host| format!("{host:?}"));
            let ownership = compiled.host.map(|host| {
                let manifested = ManifestedProgram::new(checked.clone(), manifest.clone(), Target::C);
                match lower_host_ownership(&manifested, host).and_then(verify_ownership) {
                    Ok(verified) => json!({"status": "ok", "render": verified.render()}),
                    Err(error) => json!({"status": "error", "error": format!("{error:?}")}),
                }
            });
            json!({"status": "ok", "dag": dag, "host": host, "ownership": ownership})
        }
        Err(error) => json!({"status": "error", "diagnostic": diagnostic(&error)}),
    };
    result
}

fn context_snapshot(case: &Value) -> Value {
    use chelis_types::{build_type_env_from_library, check_ir_program, check_ir_with_context};
    use chelis_ir::lower::{lower_program, lower_program_to_library, lower_program_with_context};
    let parse = |source: &str| chelis_deep::parser::parse_str(
        source, &chelis_syntax_host::NativeStack).unwrap();
    let library_source = case["library"].as_str().unwrap();
    let snippet_source = case["snippet"].as_str().unwrap();
    let library_exprs = parse(library_source);
    let environment = build_type_env_from_library(&library_exprs).unwrap();
    let finish = |checked: CheckedProgram| check_linearity(
        &chelis_effects::check_program(&checked).unwrap()).unwrap();
    let library_checked = finish(check_ir_program(&library_exprs).unwrap());
    let snippet_checked = finish(check_ir_with_context(&environment, &parse(snippet_source)).unwrap());
    let combined = finish(check_ir_program(&parse(&format!("{library_source}\n{snippet_source}"))).unwrap());
    let library = lower_program_to_library(&library_checked);
    let before = bincode::serialize(&library).unwrap();
    let composed = lower_program_with_context(&library, &snippet_checked);
    assert_eq!(before, bincode::serialize(&library).unwrap());
    let monolithic = lower_program(&combined);
    json!({"id": case["id"], "library_json": serde_json::to_value(&library).unwrap(),
        "library_binary": before, "snippet_json": serde_json::to_value(&snippet_checked).unwrap(),
        "snippet_binary": bincode::serialize(&snippet_checked).unwrap(),
        "composed": graph(&composed), "monolithic": graph(&monolithic),
        "composed_values": evaluate(&composed), "monolithic_values": evaluate(&monolithic)})
}

fn main() {
    let input = std::env::var("CHELIS_IR_CORPUS").unwrap();
    let corpus: Value = serde_json::from_str(&std::fs::read_to_string(input).unwrap()).unwrap();
    let output = corpus["cases"].as_array().unwrap().iter().map(snapshot).collect::<Vec<_>>();
    assert_eq!(output.len(), 52);
    let contexts = corpus["contexts"].as_array().unwrap().iter().map(context_snapshot).collect::<Vec<_>>();
    assert_eq!(contexts.len(), 3);
    println!("IR_SNAPSHOT {}", json!({"version": 1, "cases": output, "contexts": contexts}));
}
