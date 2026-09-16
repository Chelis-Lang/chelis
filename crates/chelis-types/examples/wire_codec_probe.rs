//! Developer-only observation driver for the C6 codec oracle.
//! This is not a published runtime protocol. Python owns cases and verdicts.

use chelis_types::types::Prim;
use chelis_types::{ElementRef, ScalarValue, TensorStorage};
use chelis_vocab::RuntimeDType;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

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
    if !text.len().is_multiple_of(2) || !text.is_ascii() {
        return Err("invalid probe input hex".into());
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

fn observation<T: Serialize>(value: &T, dtype: Prim, elements: Vec<Value>) -> Value {
    // Encoder errors abort this probe; only the actual decode failure above is
    // reported as a rejected input. A broken encoder cannot impersonate rejection.
    json!({
        "dtype": dtype.interchange_name(), "elements": elements,
        "json": serde_json::to_value(value).expect("encode observation JSON"),
        "binary": hex(&bincode::serialize(value).expect("encode observation binary")),
    })
}

fn main() {
    let vocabulary: Vec<_> = RuntimeDType::ALL
        .iter()
        .map(|dtype| {
            let prim = Prim::parse_interchange_name(dtype.name()).expect("runtime primitive");
            let kind = if prim.is_float() {
                "float"
            } else if prim.is_integer() {
                "integer"
            } else {
                assert_eq!(prim, Prim::Bool);
                "bool"
            };
            json!({"name": dtype.name(), "width": dtype.byte_width(), "kind": kind})
        })
        .collect();
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(output, "{}", json!({"vocabulary": vocabulary})).expect("write vocabulary");
    for line in io::stdin().lock().lines() {
        let request: Request =
            serde_json::from_str(&line.expect("read request")).expect("probe request");
        let result = match request.carrier.as_str() {
            "scalar" => decode::<ScalarValue>(&request)
                .map(|v| observation(&v, v.prim(), vec![element(v.element_ref())])),
            "storage" => decode::<TensorStorage>(&request).map(|v| {
                let elements = (0..v.len()).map(|i| element(v.element_ref(i))).collect();
                observation(&v, v.prim(), elements)
            }),
            _ => panic!("unknown probe carrier"),
        };
        let row = match result {
            Ok(value) => json!({"id": request.id, "observation": value}),
            Err(error) => json!({"id": request.id, "observation": null, "decode_error": error}),
        };
        writeln!(output, "{row}").expect("write observation");
    }
}
