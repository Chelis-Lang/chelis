use std::io::{self, BufRead, Write};

use schemars::{JsonSchema, schema_for};
use serde_json::{Value, json};

use crate::compiler;
use crate::schema::{
    CheckRequest, CompileRequest, DecompileRequest, DesugarRequest, EvalRequest, GradRequest,
    ValidateRequest,
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

/// Whether the Surf `source` defines a top-level binding named `name`
/// (function, let-binding, or property). The tide prove tool probes a
/// property literally named `property`; CR2-3 uses this to avoid emitting
/// a phantom `unbound variable: property` rejection for obligation-only
/// modules that define no such binding.
fn source_defines_binding(source: &str, name: &str) -> bool {
    fn scan(decls: &[chelis_surf::ast::Decl], name: &str) -> bool {
        use chelis_surf::ast::Decl;
        decls.iter().any(|d| match d {
            Decl::FunDef { name: n, .. }
            | Decl::LetDef { name: n, .. }
            | Decl::Property { name: n, .. } => n == name,
            // A `module M { ... }` wraps its declarations.
            Decl::Module { decls, .. } => scan(decls, name),
            _ => false,
        })
    }
    match chelis_surf::parser::parse_str(source) {
        Ok(decls) => scan(&decls, name),
        // A parse failure is surfaced elsewhere (the obligation engine
        // reports a parse error). Treat it as "no user property" so the
        // phantom dispatch does not add a second, misleading rejection.
        Err(_) => false,
    }
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

    let tier = args.get("tier").and_then(Value::as_str).unwrap_or("auto");
    let smt_timeout = args
        .get("smt_timeout")
        .and_then(Value::as_u64)
        .unwrap_or(5000);
    let samples = args.get("samples").and_then(Value::as_u64).unwrap_or(100) as usize;
    let seed = args.get("seed").and_then(Value::as_u64).unwrap_or(0);

    let amenability = match args
        .get("amenability")
        .and_then(Value::as_str)
        .unwrap_or("polynomial")
    {
        "linear" => chelis_prove::dispatch::SmtAmenability::Linear,
        "polynomial" => chelis_prove::dispatch::SmtAmenability::Polynomial,
        "transcendental" => chelis_prove::dispatch::SmtAmenability::Transcendental,
        "opaque" => chelis_prove::dispatch::SmtAmenability::Opaque,
        other => {
            return json!({
                "ok": false,
                "stage": "mcp",
                "errors": [{"kind": "invalid_arguments", "message": format!("invalid amenability `{other}`; must be linear|polynomial|transcendental|opaque"), "severity": 1.0, "suggestions": []}]
            });
        }
    };

    let tier_mode = match tier {
        "auto" => chelis_prove::dispatch::TierMode::Auto,
        "fuzz-only" => chelis_prove::dispatch::TierMode::FuzzOnly,
        "smt-only" => chelis_prove::dispatch::TierMode::SmtOnly,
        "type-only" => chelis_prove::dispatch::TierMode::TypeOnly,
        other => {
            return json!({
                "ok": false,
                "stage": "mcp",
                "errors": [{"kind": "invalid_arguments", "message": format!("invalid tier `{other}`"), "severity": 1.0, "suggestions": []}]
            });
        }
    };

    let options = chelis_prove::dispatch::DispatchOptions {
        tier_mode,
        smt_timeout_ms: smt_timeout,
        fuzz_samples: samples,
        fuzz_seed: seed,
    };

    // Dispatch the user property through the three-tier pipeline. The tool
    // probes a single property literally named `property` in the source.
    // CR2-3: a module that does NOT define `property` (e.g. an
    // obligation-only opaque module) must NOT be dragged to ok:false by a
    // phantom `unbound variable: property` rejection. Only run the
    // user-property dispatch when the module actually defines that binding;
    // otherwise there is no user property to fold and `ok` is driven solely
    // by the derived obligations and the check status.
    let has_user_property = source_defines_binding(&source, "property");
    let result = if has_user_property {
        Some(chelis_prove::dispatch::dispatch_property(
            &source,
            "property",
            amenability,
            &options,
        ))
    } else {
        None
    };

    // Derived producer obligations (RFC D-OBLIG, D-PARITY): run the SAME
    // chelis-prove obligation engine the CLI uses on the module source, so
    // a prove through tide is identical to the CLI on the same module.
    let ob_options = chelis_prove::obligation_engine::ObligationRunOptions {
        seed,
        samples,
        smt_timeout_ms: smt_timeout,
        tier: tier.to_string(),
        only: None,
        invariant_min_rate: args
            .get("invariant_min_rate")
            .and_then(Value::as_f64)
            .unwrap_or(0.01),
    };
    // Derived obligations. A type-broken module surfaces a check-failure
    // record rather than silently reporting zero obligations (RT3-F2).
    use chelis_prove::obligation_engine::{ObligationRunResult, ObligationStatus};
    let mut obligation_records: Vec<serde_json::Value> = Vec::new();
    // The worst obligation status, folded into the response ok/summary
    // exactly as the CLI folds it into the overall prove exit status
    // (CR-12 / D-PARITY): a failed / unsupported / errored producer
    // obligation must lower the tide response, not be ignored.
    let mut ob_failed = 0usize;
    let mut ob_unsupported = 0usize;
    let mut ob_errored = 0usize;
    let mut check_failed = false;
    match chelis_prove::obligation_engine::run_surf_source_obligations(&source, &ob_options) {
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

    use chelis_prove::ProofStatus;
    // CR2-3: a property is a PASS only when Proved or StatisticallyValidated.
    // Every other status -- Disproved, Rejected, NotAmenable -- is a non-pass
    // and must lower `ok`. The prior fold treated only Disproved as a
    // failure, so a Rejected / NotAmenable property reported ok:true (a
    // non-pass masquerading as a pass). The non-pass status is also bucketed
    // into the summary by kind, mirroring the CLI's Status mapping
    // (Disproved => failed, NotAmenable => unsupported, Rejected => error).
    // When the module defines no `property`, there is no user-property
    // contribution at all.
    let (property_proved, prop_passed, prop_failed, prop_unsupported, prop_errored) = match &result
    {
        None => (false, true, 0, 0, 0),
        Some(r) => match &r.status {
            ProofStatus::Proved => (true, true, 0, 0, 0),
            ProofStatus::StatisticallyValidated { .. } => (false, true, 0, 0, 0),
            ProofStatus::Disproved { .. } => (false, false, 1, 0, 0),
            ProofStatus::NotAmenable { .. } => (false, false, 0, 1, 0),
            ProofStatus::Rejected { .. } => (false, false, 0, 0, 1),
        },
    };
    // ok is false if the user property did not pass, OR any obligation
    // failed / was unsupported / errored, OR the module did not type-check.
    let ok = prop_passed
        && ob_failed == 0
        && ob_unsupported == 0
        && ob_errored == 0
        && !check_failed;

    let property_total = if result.is_some() { 1 } else { 0 };
    let properties: Vec<serde_json::Value> = result
        .as_ref()
        .map(|r| vec![serde_json::to_value(r).unwrap_or(json!(null))])
        .unwrap_or_default();

    json!({
        "ok": ok,
        "stage": "prove",
        "properties": properties,
        "obligations": obligation_records,
        "summary": {
            "total": property_total,
            "proved": if property_proved { 1 } else { 0 },
            "failed": prop_failed + ob_failed,
            "unsupported": prop_unsupported + ob_unsupported,
            "errors": prop_errored + ob_errored + if check_failed { 1 } else { 0 },
            "obligations": obligations_count,
        }
    })
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
        "proof_tier": o.proof_tier.as_str(),
        "samples": o.samples,
        "seed": o.seed,
    });
    if o.proof_tier == chelis_prove::obligation_engine::ObligationTier::Smt {
        value["arith_model"] = json!("real");
    }
    if let Some(cx) = o.counterexample {
        value["counterexample"] = cx;
    }
    if let Some(r) = o.reason {
        value["reason"] = json!(r);
    }
    value
}
