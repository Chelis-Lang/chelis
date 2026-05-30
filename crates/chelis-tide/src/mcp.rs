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
        "chelis_verify_spec" => handle_verify_spec_tool(&args),
        "chelis_explain_failure" => handle_explain_failure_tool(&args),
        "chelis_proof_artifact" => handle_proof_artifact_tool(&args),
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
        json!({
            "name": "chelis_verify_spec",
            "description": "Verify all @property declarations in .ch files within a spec directory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_directory": { "type": "string", "description": "Path to directory containing .ch spec files" }
                },
                "required": ["spec_directory"]
            }
        }),
        json!({
            "name": "chelis_explain_failure",
            "description": "Re-run verification and return detailed result for a specific property",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_directory": { "type": "string", "description": "Path to directory containing .ch spec files" },
                    "property_id": { "type": "string", "description": "Name of the property to explain" }
                },
                "required": ["spec_directory", "property_id"]
            }
        }),
        json!({
            "name": "chelis_proof_artifact",
            "description": "Return proof artifact metadata (solver, logic) for a specific property",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_directory": { "type": "string", "description": "Path to directory containing .ch spec files" },
                    "property_id": { "type": "string", "description": "Name of the property" }
                },
                "required": ["spec_directory", "property_id"]
            }
        }),
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
        None => return mcp_error("missing `source`"),
    };
    let tier = match args.get("tier").and_then(Value::as_str).unwrap_or("auto") {
        "auto" => chelis_prove::TierSelection::Auto,
        "fuzz-only" => chelis_prove::TierSelection::FuzzOnly,
        "smt-only" => chelis_prove::TierSelection::SmtOnly,
        "type-only" => chelis_prove::TierSelection::TypeOnly,
        other => return mcp_error(&format!("invalid tier `{other}`")),
    };
    let smt_timeout = args.get("smt_timeout_ms").and_then(Value::as_u64).unwrap_or(5000);
    let samples = args.get("fuzz_samples").and_then(Value::as_u64).unwrap_or(100) as u32;
    let seed = args.get("fuzz_seed").and_then(Value::as_u64).unwrap_or(0);

    let req = chelis_prove::VerificationRequest {
        source,
        source_kind: chelis_prove::SourceKind::Surf,
        tier,
        smt_timeout_ms: smt_timeout,
        fuzz_samples: samples,
        fuzz_seed: seed,
        inlining_depth_limit: 3,
    };

    match chelis_prove::verify_source(req) {
        Ok(result) => serde_json::to_value(&result).unwrap_or(json!({"ok": false})),
        Err(e) => json!({ "ok": false, "stage": "prove", "errors": [{"message": e.to_string()}] }),
    }
}

fn mcp_error(msg: &str) -> Value {
    json!({ "ok": false, "stage": "mcp", "errors": [{"kind": "invalid_arguments", "message": msg, "severity": 1.0, "suggestions": []}] })
}

fn handle_verify_spec_tool(args: &Value) -> Value {
    let spec_dir = match args.get("spec_directory").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None => return mcp_error("missing `spec_directory`"),
    };
    let path = std::path::Path::new(&spec_dir);
    if !path.is_dir() {
        return mcp_error(&format!("not a directory: {spec_dir}"));
    }
    let mut source = String::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "ch").unwrap_or(false) {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    source.push_str(&content);
                    source.push('\n');
                }
            }
        }
    }
    if source.is_empty() {
        return mcp_error("no .ch files found in directory");
    }
    let req = chelis_prove::VerificationRequest { source, ..Default::default() };
    match chelis_prove::verify_source(req) {
        Ok(result) => serde_json::to_value(&result).unwrap_or(json!({"ok": false})),
        Err(e) => json!({ "ok": false, "stage": "prove", "errors": [{"message": e.to_string()}] }),
    }
}

fn handle_explain_failure_tool(args: &Value) -> Value {
    let spec_dir = match args.get("spec_directory").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None => return mcp_error("missing `spec_directory`"),
    };
    let property_id = match args.get("property_id").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None => return mcp_error("missing `property_id`"),
    };
    let path = std::path::Path::new(&spec_dir);
    let mut source = String::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "ch").unwrap_or(false) {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    source.push_str(&content);
                    source.push('\n');
                }
            }
        }
    }
    let req = chelis_prove::VerificationRequest { source, ..Default::default() };
    match chelis_prove::verify_source(req) {
        Ok(result) => {
            if let Some(prop) = result.properties.iter().find(|p| p.name == property_id) {
                serde_json::to_value(prop).unwrap_or(json!({"ok": false}))
            } else {
                mcp_error(&format!("property `{property_id}` not found"))
            }
        }
        Err(e) => json!({ "ok": false, "errors": [{"message": e.to_string()}] }),
    }
}

fn handle_proof_artifact_tool(args: &Value) -> Value {
    let result = handle_explain_failure_tool(args);
    if let Some(obj) = result.as_object() {
        let mut enriched = obj.clone();
        enriched.insert("solver".to_string(), json!("cvc5-1.3.1"));
        enriched.insert("logic".to_string(), json!("QF_NRA / QF_NRAT"));
        json!(enriched)
    } else {
        result
    }
}
