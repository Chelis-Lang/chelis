//! chelis#3801: a call's inputs are admitted at exactly their declared types,
//! before evaluation ([05-OBS-9]).
//!
//! Generated oracle over the three things a call receives.
//!
//! - Scalar, boolean and string parameters take call arguments. For each such
//!   parameter type, every foreign value (every other scalar dtype, the other
//!   kinds, containers, records, tensors) is rejected with an error naming the
//!   parameter, and the well-formed value is admitted and returned bit-exactly
//!   by an identity call.
//! - Every other non-tensor parameter type (tuples, lists, dicts, records,
//!   variants, opaque types, generic types) is a call argument the boundary
//!   does not implement: every value, well-formed or not, is rejected with the
//!   typed `unsupported` diagnostic that names the parameter and chelis#3804.
//! - Tensor parameters arrive as bindings. For each tensor parameter type,
//!   every binding of another dtype, another rank, or another literal extent
//!   is rejected before evaluation, never cast or rounded; the well-formed
//!   binding is admitted and returned bit-exactly.

use std::collections::BTreeMap;

#[allow(deprecated)]
use chelis_compiler_api::compiler::{PreparedEval, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_numeric_core::prim::Prim;
use serde_json::json;

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use wire_values::{scalar_f32, scalar_f64, scalar_integer};

const TYPES: &str = r#"module M
@opaque
@invariant(p) (p.v >= 0.0)
type Positive =
  | Positive { v: f32 }

type Inner =
  | Inner { a: i64 }

type Color =
  | Red
  | Green
"#;

/// Scalar, boolean and string parameter types: the call-argument kinds.
const ARGUMENT_TYPES: [&str; 10] = [
    "i64", "i32", "i16", "i8", "f64", "f32", "f16", "bf16", "bool", "string",
];

/// Non-tensor parameter types the boundary does not implement as arguments,
/// each with a value of that type.
fn unimplemented_types() -> Vec<(&'static str, ExecutionValue)> {
    vec![
        (
            "(i64, f32)",
            ExecutionValue::Tuple {
                value: vec![scalar_integer(Prim::Int64, 1), scalar_f32(1.0)],
            },
        ),
        (
            "List[i64]",
            ExecutionValue::List {
                value: vec![scalar_integer(Prim::Int64, 1)],
            },
        ),
        (
            "Dict[string, i64]",
            serde_json::from_value(json!({"type": "dict", "entries": [
                {"key": {"type": "string", "value": "a"},
                 "value": {"type": "scalar", "value": {"dtype": "int64", "value": 1}}}
            ]}))
            .unwrap(),
        ),
        (
            "Inner",
            ExecutionValue::Adt {
                ctor: "Inner".to_string(),
                fields: vec![scalar_integer(Prim::Int64, 1)],
            },
        ),
        (
            "Color",
            ExecutionValue::Adt {
                ctor: "Red".to_string(),
                fields: Vec::new(),
            },
        ),
        (
            "Positive",
            ExecutionValue::Adt {
                ctor: "Positive".to_string(),
                fields: vec![scalar_f32(1.0)],
            },
        ),
        (
            "Option[i64]",
            ExecutionValue::Adt {
                ctor: "Some".to_string(),
                fields: vec![scalar_integer(Prim::Int64, 1)],
            },
        ),
    ]
}

/// Tensor parameter types with their dtype and literal shape.
const TENSOR_TYPES: [(&str, Prim, &[usize]); 4] = [
    ("tensor[3, f32]", Prim::F32, &[3]),
    ("tensor[2, 2, f64]", Prim::F64, &[2, 2]),
    ("tensor[3, i32]", Prim::Int32, &[3]),
    ("tensor[3, i64]", Prim::Int64, &[3]),
];

/// Tensor parameters whose type is not literal: a bounded precision variable
/// and a dimension variable. The call boundary does not implement them.
const GENERIC_TENSORS: &str = r#"
def bounded_precision[p: Float](t: tensor[3, p], k: i64) -> i64 = k

def named_extent[n](t: tensor[n, f32], k: i64) -> i64 = k
"#;

fn half(prim: Prim, bits: &str) -> ExecutionValue {
    serde_json::from_value(json!({"type": "scalar", "value": {"dtype": prim.name(), "bits": bits}}))
        .unwrap()
}

/// The well-formed argument of each argument type: values whose bits a
/// conversion would change.
fn well_formed(spelling: &str) -> ExecutionValue {
    match spelling {
        "i64" => scalar_integer(Prim::Int64, i64::MIN),
        "i32" => scalar_integer(Prim::Int32, i32::MIN.into()),
        "i16" => scalar_integer(Prim::Int16, i16::MIN.into()),
        "i8" => scalar_integer(Prim::Int8, i8::MIN.into()),
        "f64" => serde_json::from_value(
            json!({"type": "scalar", "value": {"dtype": "f64", "bits": "7ff8000000000001"}}),
        )
        .unwrap(),
        "f32" => serde_json::from_value(
            json!({"type": "scalar", "value": {"dtype": "f32", "bits": "80000001"}}),
        )
        .unwrap(),
        "f16" => half(Prim::F16, "7e01"),
        "bf16" => half(Prim::Bf16, "8001"),
        "bool" => ExecutionValue::Bool { value: true },
        "string" => ExecutionValue::String {
            value: "s".to_string(),
        },
        other => panic!("no fixture for {other}"),
    }
}

fn tensor_value(prim: Prim, shape: &[usize]) -> TensorValue {
    let count = shape.iter().product::<usize>();
    let data = match prim {
        Prim::F32 => json!({"dtype": "f32", "bits": (0..count)
            .map(|index| format!("{:08x}", (1.1f32 + index as f32).to_bits()))
            .collect::<Vec<_>>()}),
        Prim::F64 => json!({"dtype": "f64", "bits": (0..count)
            .map(|index| format!("{:016x}", (1.1f64 + index as f64).to_bits()))
            .collect::<Vec<_>>()}),
        Prim::Int32 => json!({"dtype": "int32", "values": (0..count)
            .map(|index| index as i64 + 1).collect::<Vec<_>>()}),
        Prim::Int64 => json!({"dtype": "int64", "values": (0..count)
            .map(|index| index as i64 + 3_000_000_000).collect::<Vec<_>>()}),
        other => panic!("no tensor fixture for {other:?}"),
    };
    serde_json::from_value(json!({"shape": shape, "data": data})).unwrap()
}

/// Values planted in place of a well-formed argument.
fn foreign() -> Vec<ExecutionValue> {
    let mut values = vec![
        scalar_integer(Prim::Int64, 1),
        scalar_integer(Prim::Int32, 1),
        scalar_integer(Prim::Int16, 1),
        scalar_integer(Prim::Int8, 1),
        scalar_f64(1.0),
        scalar_f32(1.0),
        half(Prim::F16, "3c00"),
        half(Prim::Bf16, "3f80"),
        ExecutionValue::Bool { value: false },
        ExecutionValue::String {
            value: "t".to_string(),
        },
        ExecutionValue::Unit,
        ExecutionValue::Tensor {
            value: tensor_value(Prim::F32, &[1]),
        },
    ];
    values.extend(unimplemented_types().into_iter().map(|(_, value)| value));
    values
}

/// Whether `value` is a value of the argument type `spelling`.
fn is_value_of(value: &ExecutionValue, spelling: &str) -> bool {
    let json = serde_json::to_value(value).unwrap();
    match spelling {
        "bool" => json["type"] == "bool",
        "string" => json["type"] == "string",
        _ => {
            json["type"] == "scalar"
                && json["value"]["dtype"]
                    .as_str()
                    .and_then(Prim::parse_interchange_name)
                    == Prim::parse_name(spelling)
        }
    }
}

fn source() -> String {
    let mut source = TYPES.to_string();
    for (index, spelling) in ARGUMENT_TYPES.iter().enumerate() {
        source.push_str(&format!(
            "\ndef keep_{index}(p: {spelling}) -> i64 = 1i64\n"
        ));
        source.push_str(&format!(
            "\ndef same_{index}(p: {spelling}) -> {spelling} = p\n"
        ));
    }
    for (index, (spelling, _)) in unimplemented_types().iter().enumerate() {
        source.push_str(&format!(
            "\ndef whole_{index}(p: {spelling}) -> i64 = 1i64\n"
        ));
    }
    for (index, (spelling, _, _)) in TENSOR_TYPES.iter().enumerate() {
        source.push_str(&format!(
            "\ndef tensor_{index}(t: {spelling}, k: i64) -> {spelling} = t\n"
        ));
    }
    source.push_str(GENERIC_TENSORS);
    source
}

#[allow(deprecated)]
fn prepared() -> PreparedEval {
    prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source(),
        bindings: BTreeMap::new(),
    })
    .expect("oracle source compiles")
}

fn messages(error: chelis_compiler_api::compiler::CompilerError) -> String {
    error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

fn call(
    program: &PreparedEval,
    root: &str,
    arguments: Vec<(&str, ExecutionValue)>,
    bindings: Vec<(&str, TensorValue)>,
) -> Result<chelis_compiler_api::schema::EvalResult, String> {
    program
        .call_root(
            root,
            arguments
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
            bindings
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
        )
        .map_err(messages)
}

#[test]
fn argument_kinds_admit_exactly_their_declared_type() {
    let program = prepared();
    let mut planted = 0usize;
    for (index, spelling) in ARGUMENT_TYPES.iter().enumerate() {
        let value = well_formed(spelling);
        let returned = call(
            &program,
            &format!("same_{index}"),
            vec![("p", value.clone())],
            vec![],
        )
        .unwrap_or_else(|error| panic!("{spelling}: well-formed argument rejected: {error}"));
        let [root] = returned.roots.as_slice() else {
            panic!(
                "{spelling}: the identity call owes one root: {:?}",
                returned.roots
            );
        };
        assert_eq!(
            serde_json::to_value(&root.value).unwrap(),
            serde_json::to_value(&value).unwrap(),
            "{spelling}: the identity call changed its argument"
        );
        for foreign in foreign() {
            if is_value_of(&foreign, spelling) {
                continue;
            }
            planted += 1;
            let error = match call(
                &program,
                &format!("keep_{index}"),
                vec![("p", foreign.clone())],
                vec![],
            ) {
                Ok(result) => panic!(
                    "{spelling}: {} was admitted and evaluated to {:?}",
                    serde_json::to_value(&foreign).unwrap(),
                    result.roots
                ),
                Err(error) => error,
            };
            assert!(
                error.contains("argument `p` of")
                    && error.contains("is not a value of its parameter type"),
                "{spelling}: {error}"
            );
        }
    }
    assert!(
        planted >= 150,
        "the oracle planted only {planted} arguments"
    );
}

#[test]
fn other_argument_types_are_typed_unsupported_rejections() {
    let program = prepared();
    for (index, (spelling, value)) in unimplemented_types().into_iter().enumerate() {
        for planted in std::iter::once(value).chain(foreign()) {
            let error = call(
                &program,
                &format!("whole_{index}"),
                vec![("p", planted)],
                vec![],
            )
            .expect_err("an unimplemented argument type is rejected");
            assert!(
                error.contains("a call argument of type")
                    && error.contains("parameter `p`")
                    && error.contains("3804"),
                "{spelling}: {error}"
            );
        }
    }
}

#[test]
fn tensor_bindings_are_admitted_at_their_exact_type_or_rejected() {
    let program = prepared();
    let candidates: Vec<(Prim, Vec<usize>)> = [Prim::F32, Prim::F64, Prim::Int32, Prim::Int64]
        .into_iter()
        .flat_map(|prim| {
            [vec![3], vec![2, 2], vec![4], vec![1, 3], vec![3, 1]]
                .into_iter()
                .map(move |shape| (prim, shape))
        })
        .collect();
    let mut planted = 0usize;
    for (index, (spelling, prim, shape)) in TENSOR_TYPES.iter().enumerate() {
        let root = format!("tensor_{index}");
        let k = || vec![("k", scalar_integer(Prim::Int64, 0))];
        let binding = tensor_value(*prim, shape);
        let result = call(&program, &root, k(), vec![("t", binding.clone())])
            .unwrap_or_else(|error| panic!("{spelling}: well-formed binding rejected: {error}"));
        let [returned] = result.roots.as_slice() else {
            panic!("{spelling}: {:?}", result.roots);
        };
        assert_eq!(
            serde_json::to_value(&returned.value).unwrap(),
            serde_json::to_value(ExecutionValue::Tensor { value: binding }).unwrap(),
            "{spelling}: the identity call changed its binding"
        );
        for (candidate_prim, candidate_shape) in &candidates {
            if candidate_prim == prim && candidate_shape.as_slice() == *shape {
                continue;
            }
            planted += 1;
            let error = match call(
                &program,
                &root,
                k(),
                vec![("t", tensor_value(*candidate_prim, candidate_shape))],
            ) {
                Ok(result) => panic!(
                    "{spelling}: a {candidate_prim:?} binding of shape {candidate_shape:?} was \
                     admitted and evaluated to {:?}",
                    result.roots
                ),
                Err(error) => error,
            };
            assert!(error.contains("binding `t` of"), "{spelling}: {error}");
        }
    }
    assert!(planted >= 70, "the oracle planted only {planted} bindings");
}

#[test]
fn tensor_parameters_of_a_non_literal_type_are_typed_unsupported_rejections() {
    let program = prepared();
    // Both directions: a binding the bound or extent admits and one it does
    // not are rejected alike, before evaluation, with the typed diagnostic.
    for (root, bindings) in [
        (
            "bounded_precision",
            [
                tensor_value(Prim::F32, &[3]),
                tensor_value(Prim::Int32, &[3]),
            ],
        ),
        (
            "named_extent",
            [tensor_value(Prim::F32, &[3]), tensor_value(Prim::F64, &[3])],
        ),
    ] {
        for binding in bindings {
            let error = call(
                &program,
                root,
                vec![("k", scalar_integer(Prim::Int64, 0))],
                vec![("t", binding)],
            )
            .expect_err("a non-literal tensor parameter type is rejected");
            assert!(
                error.contains("a tensor parameter `t`") && error.contains("3804"),
                "{root}: {error}"
            );
        }
        let error = call(
            &program,
            root,
            vec![("k", scalar_integer(Prim::Int64, 0))],
            vec![],
        )
        .expect_err("an unbound non-literal tensor parameter is rejected too");
        assert!(error.contains("3804"), "{root}: {error}");
    }
}
