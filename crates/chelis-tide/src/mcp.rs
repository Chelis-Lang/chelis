use std::io::{self, BufRead, Write};

use schemars::{JsonSchema, schema_for};
use serde_json::{Value, json};

use crate::compiler;
use crate::schema::{
    AddFunctionRequest, AddPropertyRequest, ChangeSignatureRequest, CheckRequest, CompileRequest,
    DecompileRequest, DeepCallGraphRequest, DeepOutlineRequest, DeepReferencesRequest,
    DesugarRequest, EvalRequest, GradRequest, RenameRequest, ReplaceFunctionBodyRequest,
    ReplaceFunctionRequest, ValidateRequest,
};

pub fn run_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());

    while let Some(message) = read_message(&mut reader)? {
        if let Some(response) = handle_message(&message) {
            write_message(&mut writer, &response)?;
        }
    }

    writer.flush()
}

pub fn run_stdio_blocking() -> io::Result<()> {
    run_stdio()
}

pub fn handle_message(message: &Value) -> Option<Value> {
    let method = message.get("method")?.as_str()?;
    let id = message.get("id").cloned();
    match method {
        "initialize" => Some(success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "chelis-tide",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "capabilities": {
                    "tools": {}
                }
            }),
        )),
        "tools/list" => Some(success(id, json!({ "tools": tool_list() }))),
        "tools/call" => Some(handle_tool_call(id, message.get("params"))),
        "shutdown" => Some(success(id, json!({}))),
        "exit" => None,
        other => Some(error(id, -32601, format!("unknown method `{other}`"))),
    }
}

fn handle_tool_call(id: Option<Value>, params: Option<&Value>) -> Value {
    let Some(params) = params else {
        return error(id, -32602, "missing params");
    };
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return error(id, -32602, "missing tool name");
    };
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let payload = match name {
        "chelis_check" => deserialize_and_run::<CheckRequest, _, _>(&args, compiler::check),
        "chelis_compile" => deserialize_and_run::<CompileRequest, _, _>(&args, compiler::compile),
        "chelis_desugar" => deserialize_and_run::<DesugarRequest, _, _>(&args, compiler::desugar),
        "chelis_decompile" => {
            deserialize_and_run::<DecompileRequest, _, _>(&args, compiler::decompile)
        }
        "chelis_eval" => deserialize_and_run::<EvalRequest, _, _>(&args, compiler::eval),
        "chelis_grad" => deserialize_and_run::<GradRequest, _, _>(&args, compiler::grad),
        "chelis_validate" => {
            deserialize_and_run::<ValidateRequest, _, _>(&args, compiler::validate)
        }
        "chelis_replace_function_body" => deserialize_and_run::<ReplaceFunctionBodyRequest, _, _>(
            &args,
            compiler::replace_function_body,
        ),
        "chelis_add_function" => {
            deserialize_and_run::<AddFunctionRequest, _, _>(&args, compiler::add_function)
        }
        "chelis_deep_outline" => {
            deserialize_and_run::<DeepOutlineRequest, _, _>(&args, compiler::deep_outline)
        }
        "chelis_deep_references" => {
            deserialize_and_run::<DeepReferencesRequest, _, _>(&args, compiler::deep_references)
        }
        "chelis_deep_call_graph" => {
            deserialize_and_run::<DeepCallGraphRequest, _, _>(&args, compiler::deep_call_graph)
        }
        "chelis_replace_function" => {
            deserialize_and_run::<ReplaceFunctionRequest, _, _>(&args, compiler::replace_function)
        }
        "chelis_add_property" => {
            deserialize_and_run::<AddPropertyRequest, _, _>(&args, compiler::add_property)
        }
        "chelis_rename" => deserialize_and_run::<RenameRequest, _, _>(&args, compiler::rename),
        "chelis_change_signature" => {
            deserialize_and_run::<ChangeSignatureRequest, _, _>(&args, compiler::change_signature)
        }
        "chelis_prove" => handle_prove_tool(&args),
        other => {
            return error(id, -32601, format!("unknown tool `{other}`"));
        }
    };

    success(
        id,
        json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string(&payload).expect("payload should serialize")
            }],
            "structuredContent": payload,
            "isError": payload.get("ok").and_then(Value::as_bool) == Some(false),
        }),
    )
}

fn deserialize_and_run<T, R, F>(args: &Value, f: F) -> Value
where
    T: serde::de::DeserializeOwned,
    R: serde::Serialize,
    F: FnOnce(T) -> Result<R, crate::compiler::CompilerError>,
{
    match serde_json::from_value::<T>(args.clone()) {
        Ok(request) => serialize_payload(crate::compiler::result_envelope(f(request))),
        Err(err) => json!({
            "ok": false,
            "stage": "mcp",
            "errors": [{
                "kind": "invalid_arguments",
                "message": err.to_string(),
                "severity": 1.0,
                "suggestions": [],
            }]
        }),
    }
}

fn tool_list() -> Vec<Value> {
    vec![
        tool::<CheckRequest>(
            "chelis_check",
            "Type-check Chelis source and return fitness diagnostics",
        ),
        tool::<CompileRequest>(
            "chelis_compile",
            "Compile Chelis source to generated C or HIP artifacts",
        ),
        tool::<DesugarRequest>(
            "chelis_desugar",
            "Desugar Surf Chelis source into canonical Deep",
        ),
        tool::<DecompileRequest>("chelis_decompile", "Decompile canonical Deep into Surf"),
        tool::<EvalRequest>(
            "chelis_eval",
            "Evaluate Chelis source with named tensor bindings",
        ),
        tool::<GradRequest>(
            "chelis_grad",
            "Differentiate Chelis source and return a gradient DAG",
        ),
        tool::<ValidateRequest>(
            "chelis_validate",
            "Validate Surf, Deep, or desugared Chelis source",
        ),
        tool::<ReplaceFunctionBodyRequest>(
            "chelis_replace_function_body",
            "Replace one function's body in a Deep module and return the canonical Deep of the changed def and the rewritten module",
        ),
        tool::<AddFunctionRequest>(
            "chelis_add_function",
            "Add one Deep function declaration bundle to a module and return the canonical rewritten module",
        ),
        tool::<DeepOutlineRequest>(
            "chelis_deep_outline",
            "Return the stable Deep authoring outline with canonical preimage hashes",
        ),
        tool::<DeepReferencesRequest>(
            "chelis_deep_references",
            "Return scope-aware Deep references to a top-level symbol",
        ),
        tool::<DeepCallGraphRequest>(
            "chelis_deep_call_graph",
            "Return the scope-aware direct-call graph for a Deep module",
        ),
        tool::<ReplaceFunctionRequest>(
            "chelis_replace_function",
            "Replace one whole Deep function declaration bundle with whole-module validation",
        ),
        tool::<AddPropertyRequest>(
            "chelis_add_property",
            "Add one Deep property declaration bundle with whole-module validation",
        ),
        tool::<RenameRequest>(
            "chelis_rename",
            "Rename one Deep function and structurally cascade unshadowed references",
        ),
        tool::<ChangeSignatureRequest>(
            "chelis_change_signature",
            "Change one Deep function signature and structurally rewrite direct callsites",
        ),
        prove_tool_schema(),
    ]
}

fn tool<T: JsonSchema>(name: &str, description: &str) -> Value {
    let schema = schema_for!(T);
    json!({
        "name": name,
        "description": description,
        "inputSchema": serde_json::to_value(schema).expect("tool schema should serialize")
    })
}

fn success(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        }
    })
}

fn read_message(reader: &mut impl BufRead) -> io::Result<Option<Value>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            let parsed = value.trim().parse::<usize>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid content length: {err}"),
                )
            })?;
            content_length = Some(parsed);
        }
    }

    let Some(content_length) = content_length else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing Content-Length header",
        ));
    };

    let mut buf = vec![0; content_length];
    reader.read_exact(&mut buf)?;
    let value = serde_json::from_slice(&buf)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
    Ok(Some(value))
}

fn write_message(writer: &mut impl Write, value: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(value)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()
}

fn serialize_payload<T: serde::Serialize>(value: T) -> Value {
    serde_json::to_value(value).expect("payload should serialize")
}

fn prove_tool_schema() -> Value {
    json!({
        "name": "chelis_prove",
        "description": "Verify Chelis properties via three-tier dispatch (type check → SMT → fuzz)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_kind": { "type": "string", "enum": ["surf", "deep"] },
                "source": { "type": "string", "description": "Chelis source containing @property declarations" },
                "tier": { "type": "string", "enum": ["auto", "fuzz-only", "smt-only", "type-only"], "default": "auto" },
                "amenability": { "type": "string", "enum": ["linear", "polynomial", "transcendental", "opaque"], "description": "SMT amenability classification" },
                "smt_timeout": { "type": "integer", "description": "SMT timeout in ms (default 5000)", "default": 5000 },
                "samples": { "type": "integer", "description": "Fuzz samples per property (default 100)", "default": 100 },
                "seed": { "type": "integer", "description": "Fuzz seed (default 0)", "default": 0 }
            },
            "required": ["source_kind", "source"]
        }
    })
}

fn handle_prove_tool(args: &Value) -> Value {
    let source = match args.get("source").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None => {
            return json!({
                "ok": false,
                "stage": "mcp",
                "errors": [{"kind": "invalid_arguments", "message": "missing `source`", "severity": 1.0, "suggestions": []}]
            });
        }
    };

    let source_kind = args
        .get("source_kind")
        .and_then(Value::as_str)
        .unwrap_or("surf")
        .to_string();
    let tier = args.get("tier").and_then(Value::as_str).unwrap_or("auto");
    let smt_timeout = args
        .get("smt_timeout")
        .and_then(Value::as_u64)
        .unwrap_or(5000);
    let samples = args.get("samples").and_then(Value::as_u64).unwrap_or(100) as usize;
    let seed = args.get("seed").and_then(Value::as_u64).unwrap_or(0);

    // Validate the tier and source_kind up front (the `amenability` arg is
    // accepted for schema compatibility but no longer drives dispatch -- the
    // shared property runner classifies amenability internally).
    if !matches!(tier, "auto" | "fuzz-only" | "smt-only" | "type-only") {
        return json!({
            "ok": false,
            "stage": "mcp",
            "errors": [{"kind": "invalid_arguments", "message": format!("invalid tier `{tier}`"), "severity": 1.0, "suggestions": []}]
        });
    }
    if !matches!(source_kind.as_str(), "surf" | "deep") {
        return json!({
            "ok": false,
            "stage": "mcp",
            "errors": [{"kind": "invalid_arguments", "message": format!("invalid source_kind `{source_kind}`; must be surf|deep"), "severity": 1.0, "suggestions": []}]
        });
    }
    if let Some(amenability) = args.get("amenability").and_then(Value::as_str)
        && !matches!(
            amenability,
            "linear" | "polynomial" | "transcendental" | "opaque"
        )
    {
        return json!({
            "ok": false,
            "stage": "mcp",
            "errors": [{"kind": "invalid_arguments", "message": format!("invalid amenability `{amenability}`; must be linear|polynomial|transcendental|opaque"), "severity": 1.0, "suggestions": []}]
        });
    }

    let invariant_min_rate = args
        .get("invariant_min_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.01);
    let source_is_deep = source_kind == "deep";

    // Discover and run the user @property declarations through the SHARED
    // property runner the CLI also drives (U4 / D-PARITY): no hardcoded
    // property name, the SAME discovery + engine for `.ch` and `.dp`, so a
    // prove through tide is identical to the CLI on the same module.
    use chelis_prove::property_runner::{
        PropertyRunOptions, PropertyRunResult, PropertyStatus, PropertyTier,
        run_deep_source_properties, run_surf_source_properties,
    };
    let prop_options = PropertyRunOptions {
        seed,
        samples,
        smt_timeout_ms: smt_timeout,
        tier: tier.to_string(),
        only: None,
        invariant_min_rate,
        max_attempts: None,
    };
    let property_run = if source_is_deep {
        run_deep_source_properties(&source, &prop_options)
    } else {
        run_surf_source_properties(&source, &prop_options)
    };
    let mut property_records: Vec<serde_json::Value> = Vec::new();
    let mut prop_failed = 0usize;
    let mut prop_unsupported = 0usize;
    let mut prop_errored = 0usize;
    let mut prop_proved = 0usize;
    let mut property_run_failed = false;
    let mut property_total = 0usize;
    match property_run {
        Ok(PropertyRunResult::Ran(outcomes)) => {
            property_total = outcomes.len();
            for o in &outcomes {
                // A property is a PASS only when it is a genuine pass (Proved,
                // or StatisticallyValidated with samples > 0). Every other
                // status -- Disproved/Failed, Unsupported, Error, AND a
                // Passed-with-zero-samples sentinel -- is a non-pass (U4).
                if o.is_pass() {
                    if o.proof_tier == PropertyTier::Smt {
                        prop_proved += 1;
                    }
                } else {
                    match o.status {
                        PropertyStatus::Failed => prop_failed += 1,
                        PropertyStatus::Unsupported => prop_unsupported += 1,
                        PropertyStatus::Error => prop_errored += 1,
                        // Passed-but-not-a-pass (zero-sample sentinel): count
                        // as unsupported so it lowers ok and is visible.
                        PropertyStatus::Passed => prop_unsupported += 1,
                    }
                }
            }
            property_records = outcomes.iter().map(property_to_json).collect();
        }
        Err(message) => {
            // A genuinely unparseable module is an error.
            property_run_failed = true;
            property_records.push(json!({
                "kind": "error",
                "stage": "parse",
                "reason": message,
            }));
        }
    }

    // Derived producer obligations (RFC D-OBLIG, D-PARITY): run the SAME
    // chelis-prove obligation engine the CLI uses on the module source, so
    // a prove through tide is identical to the CLI on the same module.
    let ob_options = chelis_prove::obligation_engine::ObligationRunOptions {
        seed,
        samples,
        smt_timeout_ms: smt_timeout,
        tier: tier.to_string(),
        only: None,
        invariant_min_rate,
    };
    // Derived obligations. A type-broken module surfaces a check-failure
    // record rather than silently reporting zero obligations (RT3-F2).
    use chelis_prove::obligation_engine::{
        ObligationRunResult, ObligationStatus, run_deep_source_obligations,
        run_surf_source_obligations,
    };
    let mut obligation_records: Vec<serde_json::Value> = Vec::new();
    let mut ob_failed = 0usize;
    let mut ob_unsupported = 0usize;
    let mut ob_errored = 0usize;
    let mut check_failed = false;
    // Dispatch the obligation run on source kind exactly as the property run
    // above does (D-PARITY): a `.dp` is fed to the Deep source entry, a `.ch`
    // to the Surf one. Both go through the SAME `run_module_obligations`, so a
    // Deep module's derived producer obligations and whole-module type-check
    // run identically to the CLI `chelis prove foo.dp` path -- never skipped.
    let ob_result = if source_is_deep {
        run_deep_source_obligations(&source, &ob_options)
    } else {
        run_surf_source_obligations(&source, &ob_options)
    };
    match ob_result {
        Ok(ObligationRunResult::Ran(outcomes)) => {
            for o in &outcomes {
                match o.status {
                    ObligationStatus::Failed => ob_failed += 1,
                    ObligationStatus::Unsupported => ob_unsupported += 1,
                    ObligationStatus::Error => ob_errored += 1,
                    ObligationStatus::Passed => {}
                }
            }
            obligation_records = outcomes.into_iter().map(obligation_to_json).collect();
        }
        Ok(ObligationRunResult::CheckFailed(messages)) => {
            check_failed = true;
            obligation_records.push(json!({
                "kind": "error",
                "stage": "check",
                "reason": "module does not type-check; obligations not verified",
                "diagnostics": messages,
            }));
        }
        Err(message) => {
            check_failed = true;
            obligation_records.push(json!({
                "kind": "error",
                "stage": "parse",
                "reason": message,
            }));
        }
    }
    let obligations_count = obligation_records.len();

    // ok is true ONLY when every property AND obligation is a genuine pass
    // and the module type-checks (U4 / D-PARITY). A failed/unsupported/
    // errored property OR obligation, a zero-sample sentinel property, a
    // parse failure, or a type-check failure makes the response not-ok.
    let ok = prop_failed == 0
        && prop_unsupported == 0
        && prop_errored == 0
        && !property_run_failed
        && ob_failed == 0
        && ob_unsupported == 0
        && ob_errored == 0
        && !check_failed;

    json!({
        "ok": ok,
        "stage": "prove",
        "properties": property_records,
        "obligations": obligation_records,
        "summary": {
            "total": property_total,
            "proved": prop_proved,
            "failed": prop_failed + ob_failed,
            "unsupported": prop_unsupported + ob_unsupported,
            "errors": prop_errored + ob_errored + if check_failed || property_run_failed { 1 } else { 0 },
            "obligations": obligations_count,
        }
    })
}

/// Render a property outcome into a JSON record (the same additive shape the
/// CLI emits), so the cross-surface parity test can compare directly. The
/// status is the `is_pass`-bucketed `display_status` (F8): a zero-sample
/// `Passed` sentinel renders as "unsupported" exactly as the CLI does, so a
/// property's reported status never diverges between the two surfaces.
fn property_to_json(o: &chelis_prove::property_runner::PropertyOutcome) -> Value {
    let mut value = json!({
        "kind": "property",
        "name": o.name,
        "status": o.display_status(),
        // chelis#422 (D2): disclose the verdict on the tide surface too -- the
        // weakest `composite_verdict` token plus the full `qualifiers` set --
        // so a tide prove and a CLI prove agree on the honest verdict.
        "composite_verdict": o.composite_verdict.as_str(),
        "qualifiers": o.disclosed_qualifiers(),
        "proof_tier": o.proof_tier.as_str(),
        "samples": o.samples,
        "seed": o.seed,
        "assumptions": &o.assumptions,
    });
    // chelis#436: the discharged proposition travels with the record on the
    // tide surface too, so a tide prove and a CLI prove agree on the goal.
    if let Some(goal) = &o.goal {
        value["goal"] = json!(goal);
    }
    if let Some(method) = &o.sampling_method {
        value["sampling_method"] = json!(method);
        value["accepted_samples"] = json!(o.samples);
        value["attempted_samples"] = json!(o.attempted_samples);
        value["rejected_samples"] = json!(o.rejected_samples);
    }
    if let Some(cx) = &o.counterexample {
        value["counterexample"] = cx.clone();
        value["shrink_steps"] = json!(o.shrink_steps);
    }
    if let Some(r) = &o.reason {
        value["reason"] = json!(r);
    }
    value
}

/// Render an obligation outcome into the same additive JSON record shape
/// the CLI emits (RFC D-OBLIG), so the cross-surface parity test can
/// compare them directly.
fn obligation_to_json(o: chelis_prove::obligation_engine::ObligationOutcome) -> Value {
    use chelis_prove::obligation_engine::ObligationStatus;
    let status = match o.status {
        ObligationStatus::Passed => "passed",
        ObligationStatus::Failed => "failed",
        ObligationStatus::Unsupported => "unsupported",
        ObligationStatus::Error => "error",
    };
    let mut value = json!({
        "kind": "obligation",
        "obligation_kind": o.meta.obligation_kind,
        "source_type": o.meta.source_type,
        "producer": o.meta.producer,
        "name": o.name,
        "status": status,
        // chelis#422 (D2): disclose the verdict + full qualifier set here too.
        "composite_verdict": o.composite_verdict.as_str(),
        "qualifiers": o.disclosed_qualifiers(),
        "proof_tier": o.proof_tier.as_str(),
        "samples": o.samples,
        "seed": o.seed,
    });
    // chelis#436: the discharged proposition (the invariant predicate) travels
    // with the record on the tide surface too.
    if let Some(goal) = &o.goal {
        value["goal"] = json!(goal);
    }
    if o.proof_tier == chelis_prove::obligation_engine::ObligationTier::Smt {
        value["arith_model"] = json!("real");
    }
    if let Some(cx) = o.counterexample {
        value["counterexample"] = cx;
        value["shrink_steps"] = json!(o.shrink_steps);
    }
    if let Some(r) = o.reason {
        value["reason"] = json!(r);
    }
    value
}
