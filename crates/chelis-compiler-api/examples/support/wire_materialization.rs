//! Observations at existing source admission APIs, under spec/10 §3.3.
//!
//! There is no WireDeepExpr/WireLiteral -> executable AST API. Their JSON
//! codecs below only observe transport. SourceProgram calls the production
//! text APIs. RawSourceAdmission calls the parser/stamper, and SourceHistory
//! copies a preserved raw argument into a live annotation before stamping.
//! None of these probe adapters introduces a production ingress.
use chelis_compiler_api::{compiler, schema};
use chelis_deep::{Expr, ExtensionData, RawExpr};
use schema::{
    CheckRequest, DecompileRequest, DesugarRequest, EvalRequest, ParseRequest, SourceKind,
    ValidateMode, ValidateRequest, WireDeepAtom, WireDeepExpr, WireLiteral, WireSurfExpr,
};
use serde::Serialize;
use serde_json::{Value, json};

// Coordinates have separate C6 cases. Preserve every other serialized field
// and make f64 bits observable: JSON numeric equality cannot distinguish zeros.
fn syntax(value: impl Serialize) -> Result<Value, String> {
    fn observe(value: Value) -> Value {
        match value {
            Value::Number(number) if number.is_f64() => {
                json!({"f64_bits": format!("{:016x}", number.as_f64().expect("float").to_bits())})
            }
            Value::Array(values) => Value::Array(values.into_iter().map(observe).collect()),
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .filter(|(key, _)| key != "span")
                    .map(|(key, value)| (key, observe(value)))
                    .collect(),
            ),
            value => value,
        }
    }
    serde_json::to_value(value)
        .map(observe)
        .map(|value| json!({"syntax": value}))
        .map_err(|error| error.to_string())
}

fn failure(error: compiler::CompilerError) -> String {
    format!(
        "{}: {}",
        error.stage,
        serde_json::to_string(&error.errors).expect("diagnostics serialize")
    )
}

fn source_program(codec: &str, source: &str) -> Result<Value, String> {
    let source_kind = if codec.ends_with("-surf") {
        SourceKind::Surf
    } else {
        SourceKind::Deep
    };
    match codec {
        "parse-deep" | "parse-surf" => {
            let parsed = compiler::parse(ParseRequest {
                source_kind,
                source: source.into(),
            })
            .map_err(failure)?;
            match source_kind {
                SourceKind::Deep => syntax(parsed.deep_ast.expect("Deep parse result")),
                SourceKind::Surf => syntax(parsed.surf_ast.expect("Surf parse result")),
            }
        }
        "desugar-surf" => {
            let result = compiler::desugar(DesugarRequest {
                source: source.into(),
            })
            .map_err(failure)?;
            syntax(result.deep_ast)
        }
        "check-deep" | "check-surf" => {
            let report = compiler::check(CheckRequest {
                source_kind,
                source: source.into(),
            })
            .map_err(failure)?;
            if report.score.get() != 1.0
                || !report.errors.is_empty()
                || !report.unresolved_names.is_empty()
                || report.total_nodes.get() == 0
            {
                return Err(format!(
                    "check: {}",
                    serde_json::to_string(&report).expect("check report serializes")
                ));
            }
            Ok(json!({"admitted": true}))
        }
        "eval-deep" | "eval-surf" => {
            let result = compiler::eval(EvalRequest {
                source_kind,
                source: source.into(),
                bindings: Default::default(),
            })
            .map_err(failure)?;
            let roots: Vec<_> = result
                .roots
                .into_iter()
                .map(|root| json!({"name": root.name, "value": root.value}))
                .collect();
            Ok(json!({"roots": roots}))
        }
        "validate-deep" => {
            compiler::validate(ValidateRequest {
                mode: ValidateMode::Deep,
                source: source.into(),
            })
            .map_err(failure)?;
            Ok(json!({"admitted": true}))
        }
        "decompile-deep" => {
            let result = compiler::decompile(DecompileRequest {
                source: source.into(),
            })
            .map_err(failure)?;
            Ok(json!({"surf": result.surf_text}))
        }
        _ => Err("unknown source program probe codec".into()),
    }
}

fn stamped(expressions: Vec<Expr>) -> Value {
    json!({"forms": expressions.len(), "deep": chelis_deep::printer::print_canonical_flat(&expressions)})
}

fn history(codec: &str, source: &str) -> Result<Value, String> {
    let expressions = chelis_deep::parse_and_stamp_file(source).map_err(|e| e.to_string())?;
    let [Expr::Node(owner, _)] = expressions.as_slice() else {
        return Err("history probe requires one declaration".into());
    };
    let history = owner
        .meta()
        .source()
        .ok_or("history probe requires source metadata")?;
    let [argument] = history.arguments() else {
        return Err("history probe requires one preserved argument".into());
    };
    match codec {
        "transport" => {
            let data = ExtensionData::from_raw(argument).map_err(|e| e.to_string())?;
            Ok(json!({"data": data.syntax()}))
        }
        "live-annotation" => {
            // A producer may reuse history, but the destination has a new role.
            // Use the existing RawExpr -> typed-file boundary, with the raw
            // payload intact so earlier parsing cannot pre-admit its metadata.
            let RawExpr::List(mut elements, span) = expressions[0].to_raw() else {
                return Err("history owner must project to a raw node".into());
            };
            let RawExpr::Map(entries, _) = &mut elements[1] else {
                return Err("history owner must carry raw metadata".into());
            };
            entries.push(("property_seed".into(), argument.clone()));
            let admitted =
                chelis_deep::stamp_to_typed::stamp_deep_file(vec![RawExpr::List(elements, span)])
                    .map_err(|e| e.to_string())?;
            // Stamping preserves unknown forms for diagnostics. As with the
            // production authoring APIs, the checker must still approve the
            // reconstructed program before it can be used as computation.
            source_program(
                "check-deep",
                &chelis_deep::printer::print_canonical_flat(&admitted),
            )
        }
        _ => Err("unknown source history probe codec".into()),
    }
}

pub fn observe(carrier: &str, codec: &str, input: &str) -> Result<Value, String> {
    match (carrier, codec) {
        ("WireLiteral", "json") => {
            syntax(serde_json::from_str::<WireLiteral>(input).map_err(|e| e.to_string())?)
        }
        ("WireDeepAtom", "json") => {
            syntax(serde_json::from_str::<WireDeepAtom>(input).map_err(|e| e.to_string())?)
        }
        ("WireDeepExpr", "json") => {
            syntax(serde_json::from_str::<WireDeepExpr>(input).map_err(|e| e.to_string())?)
        }
        ("WireSurfExpr", "json") => {
            syntax(serde_json::from_str::<WireSurfExpr>(input).map_err(|e| e.to_string())?)
        }
        ("SourceProgram", codec) => source_program(codec, input),
        ("SourceHistory", codec) => history(codec, input),
        ("RawSourceAdmission", "file" | "runtime") => {
            let raw = chelis_deep::parser::parse_raw_str(input).map_err(|e| e.to_string())?;
            let expressions = if codec == "file" {
                chelis_deep::stamp_to_typed::stamp_deep_file(raw)
            } else {
                chelis_deep::stamp_to_typed::stamp_runtime_exprs(raw)
            }
            .map_err(|e| e.to_string())?;
            Ok(stamped(expressions))
        }
        ("ExtensionData", "data" | "runtime") => {
            let data = ExtensionData::parse(input).map_err(|e| e.to_string())?;
            if codec == "data" {
                Ok(json!({"data": data.syntax()}))
            } else {
                chelis_deep::stamp_to_typed::stamp_runtime_exprs(vec![RawExpr::ExtensionData(data)])
                    .map(stamped)
                    .map_err(|e| e.to_string())
            }
        }
        _ => Err("unknown source materialization probe route".into()),
    }
}
