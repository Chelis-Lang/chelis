//! Developer-only observations for C6 schema codec admission.
//! The Python oracle owns expected results and verdicts.
use chelis_compiler_api::compiler::{ExecutionDim, ExecutionTensorSpec};
use chelis_compiler_api::schema::numbers::{
    NonnegativeCount, NonnegativeExtent, SourceFloat, SourceInteger, UnitInterval,
};
use chelis_compiler_api::schema::{
    ArtifactAbiVersion, CheckResult, CompileTarget, CompiledArtifactManifest, DiagnosticSpan,
    EvalResult, EvaluatedRoot, FitnessComponents, GradResult, LowerResult, NumericScalar,
    OrderedInferredParameters, Span, TensorValue, WireApiEnvelope, WireBatchResult,
    WireCheckResult, WireDag, WireDagNode, WireInferredDim, WireInferredParameter,
    WireInferredPrecision, WireInferredSignature, WireInferredType,
};
use chelis_types::{ElementRef, ScalarValue, TensorStorage};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

#[path = "support/wire_materialization.rs"]
mod materialization;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    carrier: String,
    codec: String,
    input: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(text: &str) -> Result<Vec<u8>, String> {
    if !text.is_ascii() || !text.len().is_multiple_of(2) {
        return Err("invalid probe hex".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}
fn decode<T: DeserializeOwned>(request: &Request) -> Result<T, String> {
    match request.codec.as_str() {
        "json" => serde_json::from_str(&request.input).map_err(|e| e.to_string()),
        "binary" => bincode::deserialize(&unhex(&request.input)?).map_err(|e| e.to_string()),
        _ => Err("unknown probe codec".into()),
    }
}
fn element(value: ElementRef) -> Value {
    match value {
        ElementRef::F64(v) => json!(format!("{:016x}", v.to_bits())),
        ElementRef::F32(v) => json!(format!("{:08x}", v.to_bits())),
        ElementRef::F16(v) => json!(format!("{:04x}", v.to_bits())),
        ElementRef::Bf16(v) => json!(format!("{:04x}", v.to_bits())),
        ElementRef::I64(v) => json!(v),
        ElementRef::I32(v) => json!(v),
        ElementRef::I16(v) => json!(v),
        ElementRef::I8(v) => json!(v),
        ElementRef::Bool(v) => json!(v),
    }
}
trait FixedNumber: Serialize + DeserializeOwned + TryFrom<ScalarValue, Error = String> {
    fn scalar(&self) -> ScalarValue;
}
macro_rules! fixed_number {
    ($($name:ty),* $(,)?) => {$(impl FixedNumber for $name {
        fn scalar(&self) -> ScalarValue { <$name>::scalar(*self) }
    })*};
}
fixed_number!(
    UnitInterval,
    SourceFloat,
    SourceInteger,
    NonnegativeCount,
    NonnegativeExtent
);
impl FixedNumber for NumericScalar {
    fn scalar(&self) -> ScalarValue {
        self.get()
    }
}
fn number<T: FixedNumber>(request: &Request) -> Result<Value, String> {
    let value: T = if request.codec == "scalar" {
        let scalar: ScalarValue =
            serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        T::try_from(scalar)?
    } else {
        decode(request)?
    };
    let scalar = value.scalar();
    Ok(
        json!({"dtype": scalar.prim().name(), "elements": [element(scalar.element_ref())],
        "json": serde_json::to_value(&value).expect("encode observed number"),
        "binary": hex(&bincode::serialize(&value).expect("encode observed binary number"))}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorInput {
    shape: Vec<i64>,
    data: TensorStorage,
}
fn tensor(request: &Request) -> Result<Value, String> {
    let value: TensorValue = if request.codec == "construct" {
        let input: TensorInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        let value = TensorValue {
            shape: input.shape,
            data: input.data,
        };
        value.validate()?;
        value
    } else {
        decode(request)?
    };
    let elements: Vec<_> = (0..value.data.len())
        .map(|i| element(value.data.element_ref(i)))
        .collect();
    Ok(
        json!({"dtype": value.data.prim().name(), "elements": elements, "shape": value.shape,
        "json": serde_json::to_value(&value).expect("encode observed tensor"),
        "binary": hex(&bincode::serialize(&value).expect("encode observed binary tensor"))}),
    )
}
#[derive(Deserialize)]
struct ReportInput {
    score: UnitInterval,
    components: FitnessComponents,
    typed_nodes: NonnegativeCount,
    untyped_nodes: NonnegativeCount,
    total_nodes: NonnegativeCount,
    unresolved_names: Vec<String>,
    inferred_signatures: Option<Vec<WireInferredSignature>>,
}
fn report(request: &Request) -> Result<Value, String> {
    let value: WireCheckResult = if request.codec == "construct" {
        let input: ReportInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        let value = CheckResult {
            score: input.score,
            components: input.components,
            typed_nodes: input.typed_nodes,
            untyped_nodes: input.untyped_nodes,
            total_nodes: input.total_nodes,
            unresolved_names: input.unresolved_names,
            errors: vec![],
            inferred_signatures: input.inferred_signatures,
        };
        let encoded = serde_json::to_string(&value).map_err(|e| e.to_string())?;
        serde_json::from_str(&encoded).expect("decode valid produced report")
    } else {
        decode(request)?
    };
    Ok(json!({
        "score": value.score,
        "components": value.components,
        "counts": [value.typed_nodes, value.untyped_nodes, value.total_nodes],
        "unresolved_names": value.unresolved_names,
        "error_count": value.errors.len(),
        "inferred_signatures": value.inferred_signatures,
    }))
}
fn parameters(request: &Request) -> Result<Value, String> {
    let value: OrderedInferredParameters = if request.codec == "construct" {
        let input: Vec<WireInferredParameter> =
            serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        input.try_into()?
    } else {
        decode(request)?
    };
    Ok(serde_json::to_value(value).expect("encode admitted parameter references"))
}
#[derive(Deserialize)]
struct EvalInput {
    schema_version: u32,
    roots: Vec<EvaluatedRoot>,
}
fn eval(request: &Request) -> Result<Value, String> {
    let value: EvalResult = if request.codec == "construct" {
        let input: EvalInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        EvalResult {
            schema_version: input.schema_version,
            roots: input.roots,
            manifest: Default::default(),
            transcript: vec![],
        }
    } else {
        decode(request)?
    };
    serde_json::to_value(value).map_err(|e| e.to_string())
}
fn envelope<T>(value: WireApiEnvelope<T>, observe: impl FnOnce(T) -> Value) -> Value {
    match value {
        WireApiEnvelope::Success(success) => {
            json!({"variant":"success", "ok":success.ok, "value":observe(success.result)})
        }
        WireApiEnvelope::Failure(failure) => {
            json!({"variant":"failure", "ok":failure.ok, "stage":failure.stage, "error_count":failure.errors.len()})
        }
    }
}
fn batch(request: &Request) -> Result<Value, String> {
    let value: WireBatchResult = decode(request)?;
    let (kind, observation) = match value {
        WireBatchResult::Parse(v) => ("parse", envelope(v, |_| Value::Null)),
        WireBatchResult::Desugar(v) => ("desugar", envelope(v, |_| Value::Null)),
        WireBatchResult::Check(v) => ("check", envelope(v, |_| Value::Null)),
        WireBatchResult::Lower(v) => ("lower", envelope(v, |_| Value::Null)),
        WireBatchResult::Compile(v) => ("compile", envelope(v, |_| Value::Null)),
        WireBatchResult::Eval(v) => (
            "eval",
            envelope(v, |v| {
                serde_json::to_value(v).expect("encode admitted eval")
            }),
        ),
        WireBatchResult::Grad(v) => ("grad", envelope(v, |_| Value::Null)),
        WireBatchResult::Validate(v) => ("validate", envelope(v, |_| Value::Null)),
        WireBatchResult::Decompile(v) => ("decompile", envelope(v, |_| Value::Null)),
    };
    Ok(json!({"kind":kind,"envelope":observation}))
}
#[derive(Deserialize)]
struct DagInput {
    schema_version: u32,
    nodes: Vec<WireDagNode>,
    roots: Vec<u64>,
}
fn dag(request: &Request) -> Result<Value, String> {
    let value: WireDag = if request.codec == "admit" {
        WireDag::from_validated_json(&request.input).map_err(|error| error.to_string())?
    } else if request.codec == "construct" {
        let input: DagInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        WireDag {
            schema_version: input.schema_version,
            nodes: input.nodes,
            roots: input.roots,
        }
    } else {
        decode(request)?
    };
    serde_json::to_value(value).map_err(|e| e.to_string())
}
#[derive(Deserialize)]
struct SliceInput {
    span: Span,
    source: String,
}
fn source_span(request: &Request) -> Result<Value, String> {
    if request.codec == "slice" {
        let input: SliceInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        return input
            .span
            .slice(&input.source)
            .map(|value| json!({"slice": value}));
    }
    let value: Span = decode(request)?;
    Ok(serde_json::to_value(value).expect("encode admitted source coordinate"))
}
fn source_location(request: &Request) -> Result<Value, String> {
    let value: Option<DiagnosticSpan> = decode(request)?;
    Ok(
        json!({"json": value, "offset": value.map(DiagnosticSpan::offset),
              "extent": value.and_then(DiagnosticSpan::extent)}),
    )
}
#[derive(Deserialize)]
struct LowerInput {
    dag: WireDag,
    named_roots: std::collections::BTreeMap<String, u64>,
}
#[derive(Deserialize)]
struct GradInput {
    dag: WireDag,
    output_node: u64,
    grad_nodes_by_name: std::collections::BTreeMap<String, u64>,
    forward_nodes_by_name: std::collections::BTreeMap<String, u64>,
}
fn lower_result(request: &Request) -> Result<Value, String> {
    let value: LowerResult = if request.codec == "construct" {
        let input: LowerInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        LowerResult {
            dag: input.dag,
            named_roots: input.named_roots,
        }
    } else {
        decode(request)?
    };
    serde_json::to_value(value).map_err(|e| e.to_string())
}
fn grad_result(request: &Request) -> Result<Value, String> {
    let value: GradResult = if request.codec == "construct" {
        let input: GradInput = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        GradResult {
            dag: input.dag,
            output_node: input.output_node,
            grad_nodes_by_name: input.grad_nodes_by_name,
            forward_nodes_by_name: input.forward_nodes_by_name,
        }
    } else {
        decode(request)?
    };
    serde_json::to_value(value).map_err(|e| e.to_string())
}
fn observed_json<T: Serialize + DeserializeOwned>(request: &Request) -> Result<Value, String> {
    let value: T = decode(request)?;
    Ok(serde_json::to_value(value).expect("encode admitted observed metadata"))
}
fn artifact_version(request: &Request) -> Result<Value, String> {
    let version: ArtifactAbiVersion = if request.codec == "construct" {
        let raw: u32 = serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        ArtifactAbiVersion::try_from(raw)?
    } else {
        decode(request)?
    };
    serde_json::to_value(version).map_err(|e| e.to_string())
}
#[derive(Deserialize)]
struct ArtifactInput {
    version: u32,
    size: i64,
}
fn artifact_manifest(request: &Request) -> Result<Value, String> {
    let manifest: CompiledArtifactManifest = if request.codec == "construct" {
        let input: ArtifactInput =
            serde_json::from_str(&request.input).map_err(|e| e.to_string())?;
        CompiledArtifactManifest {
            abi_version: ArtifactAbiVersion::try_from(input.version)?,
            target: CompileTarget::C,
            host_entry_name: "chelis_main".into(),
            device_entry_name: None,
            inputs: vec![ExecutionTensorSpec {
                name: "x".into(),
                dtype: "float32".into(),
                dims: vec![ExecutionDim {
                    name: Some("n".into()),
                    size: Some(NonnegativeExtent::new(input.size)?),
                }],
            }],
            outputs: vec![],
            symbolic_dims: vec![],
            source_path: "model.chelis".into(),
            source_hash: "digest".into(),
        }
    } else {
        decode(request)?
    };
    serde_json::to_value(manifest).map_err(|e| e.to_string())
}
fn main() {
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(output, "{}", json!({"schema_probe": 1})).expect("write probe header");
    for line in io::stdin().lock().lines() {
        let request: Request =
            serde_json::from_str(&line.expect("read request")).expect("probe request");
        let result = match request.carrier.as_str() {
            "ArtifactAbiVersion" => artifact_version(&request),
            "CompiledArtifactManifest" => artifact_manifest(&request),
            "UnitInterval" => number::<UnitInterval>(&request),
            "SourceFloat" => number::<SourceFloat>(&request),
            "SourceInteger" => number::<SourceInteger>(&request),
            "NonnegativeCount" => number::<NonnegativeCount>(&request),
            "NonnegativeExtent" => number::<NonnegativeExtent>(&request),
            "NumericScalar" => number::<NumericScalar>(&request),
            "TensorValue" => tensor(&request),
            "CheckResult" => report(&request),
            "OrderedInferredParameters" => parameters(&request),
            "EvalResult" => eval(&request),
            "WireDag" => dag(&request),
            "LowerResult" => lower_result(&request),
            "GradResult" => grad_result(&request),
            "Span" => source_span(&request),
            "ExecutionDim" => observed_json::<ExecutionDim>(&request),
            "WireInferredType" => observed_json::<WireInferredType>(&request),
            "WireInferredPrecision" => observed_json::<WireInferredPrecision>(&request),
            "WireInferredDim" => observed_json::<WireInferredDim>(&request),
            "EvaluatedRoot" => observed_json::<EvaluatedRoot>(&request),
            "DiagnosticLocation" => source_location(&request),
            "WireApiEnvelope<EvalResult>" => {
                decode::<WireApiEnvelope<EvalResult>>(&request).map(|v| {
                    envelope(v, |v| {
                        serde_json::to_value(v).expect("encode admitted eval")
                    })
                })
            }
            "WireBatchResult" => batch(&request),
            "WireLiteral" | "WireDeepAtom" | "WireDeepExpr" | "WireSurfExpr" | "SourceProgram"
            | "SourceHistory" | "RawSourceAdmission" | "ExtensionData" => {
                materialization::observe(&request.carrier, &request.codec, &request.input)
            }
            _ => panic!("unknown schema probe carrier"),
        };
        let row = match result {
            Ok(value) => json!({"id": request.id, "observation": value}),
            Err(error) => json!({"id": request.id, "observation": null, "decode_error": error}),
        };
        writeln!(output, "{row}").expect("write observation");
    }
}
