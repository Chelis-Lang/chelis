//! chelis#3801: a call argument is admitted against the parameter's checked
//! type at every depth, before evaluation ([05-OBS-9]).
//!
//! Generated oracle. Each parameter type below has one well-formed value. For
//! every position in that value, and for every foreign value that is not a
//! value of the type declared at that position, the planted mutation must be
//! rejected by admission, and the rejection must name the planted position.
//! The well-formed value must be admitted, and its call must produce its owed
//! root (an identity body returns the argument exactly). A value of an
//! invariant-carrying type that breaks the invariant, planted at each depth
//! where that type occurs, must be rejected too.

use std::collections::BTreeMap;

#[allow(deprecated)]
use chelis_compiler_api::compiler::{PreparedEval, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use chelis_types::types::Prim;
use serde_json::json;

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use wire_values::{scalar_f32, scalar_f64, scalar_integer};

/// A parameter type, as far as the oracle needs to describe it.
#[derive(Clone, Debug)]
enum Shape {
    Scalar(Prim),
    Bool,
    Str,
    Tuple(Vec<Shape>),
    List(Box<Shape>),
    /// A record or variant constructor of a named data type, with its fields
    /// in declared order (`None` names a positional field).
    Adt {
        ty: &'static str,
        ctor: &'static str,
        fields: Vec<(Option<&'static str>, Shape)>,
    },
    Tensor(Prim, Vec<usize>),
}

const SOURCE_TYPES: &str = r#"module M
@opaque
@invariant(p) (p.v >= 0.0)
type Positive =
  | Positive { v: f32 }

type Inner =
  | Inner { a: i64 }

type Other =
  | Other { b: f32 }

type Outer =
  | Outer { inner: Inner }

type WithList =
  | WithList { a: i64, xs: List[i64] }

type WithTensor =
  | WithTensor { t: tensor[3, f32] }

type Holder =
  | Holder { p: Positive }

type Color =
  | Red
  | Green
"#;

fn shapes() -> Vec<(&'static str, Shape)> {
    let inner = || Shape::Adt {
        ty: "Inner",
        ctor: "Inner",
        fields: vec![(Some("a"), Shape::Scalar(Prim::Int64))],
    };
    let positive = || Shape::Adt {
        ty: "Positive",
        ctor: "Positive",
        fields: vec![(Some("v"), Shape::Scalar(Prim::F32))],
    };
    vec![
        ("i64", Shape::Scalar(Prim::Int64)),
        ("i32", Shape::Scalar(Prim::Int32)),
        ("i16", Shape::Scalar(Prim::Int16)),
        ("i8", Shape::Scalar(Prim::Int8)),
        ("f64", Shape::Scalar(Prim::F64)),
        ("f32", Shape::Scalar(Prim::F32)),
        ("f16", Shape::Scalar(Prim::F16)),
        ("bf16", Shape::Scalar(Prim::Bf16)),
        ("bool", Shape::Bool),
        ("string", Shape::Str),
        (
            "(i64, f32)",
            Shape::Tuple(vec![Shape::Scalar(Prim::Int64), Shape::Scalar(Prim::F32)]),
        ),
        (
            "List[i64]",
            Shape::List(Box::new(Shape::Scalar(Prim::Int64))),
        ),
        (
            "Outer",
            Shape::Adt {
                ty: "Outer",
                ctor: "Outer",
                fields: vec![(Some("inner"), inner())],
            },
        ),
        (
            "WithList",
            Shape::Adt {
                ty: "WithList",
                ctor: "WithList",
                fields: vec![
                    (Some("a"), Shape::Scalar(Prim::Int64)),
                    (
                        Some("xs"),
                        Shape::List(Box::new(Shape::Scalar(Prim::Int64))),
                    ),
                ],
            },
        ),
        (
            "WithTensor",
            Shape::Adt {
                ty: "WithTensor",
                ctor: "WithTensor",
                fields: vec![(Some("t"), Shape::Tensor(Prim::F32, vec![3]))],
            },
        ),
        (
            "Holder",
            Shape::Adt {
                ty: "Holder",
                ctor: "Holder",
                fields: vec![(Some("p"), positive())],
            },
        ),
        (
            "Color",
            Shape::Adt {
                ty: "Color",
                ctor: "Red",
                fields: Vec::new(),
            },
        ),
        ("List[Inner]", Shape::List(Box::new(inner()))),
        ("List[Positive]", Shape::List(Box::new(positive()))),
        (
            "(Inner, List[bool])",
            Shape::Tuple(vec![inner(), Shape::List(Box::new(Shape::Bool))]),
        ),
        (
            "(tensor[3, f32], i64)",
            Shape::Tuple(vec![
                Shape::Tensor(Prim::F32, vec![3]),
                Shape::Scalar(Prim::Int64),
            ]),
        ),
        (
            "List[tensor[3, f32]]",
            Shape::List(Box::new(Shape::Tensor(Prim::F32, vec![3]))),
        ),
    ]
}

fn tensor(prim: Prim, shape: &[usize]) -> ExecutionValue {
    let count = shape.iter().product::<usize>();
    let bits = match prim {
        Prim::F32 => vec![json!(format!("{:08x}", 1.0f32.to_bits())); count],
        Prim::F64 => vec![json!(format!("{:016x}", 1.0f64.to_bits())); count],
        other => panic!("no tensor fixture for {other:?}"),
    };
    serde_json::from_value(json!({"type": "tensor", "value": {
        "shape": shape, "data": {"dtype": prim.name(), "bits": bits}
    }}))
    .expect("tensor fixture")
}

fn scalar(prim: Prim) -> ExecutionValue {
    match prim {
        Prim::F32 => scalar_f32(1.5),
        Prim::F64 => scalar_f64(1.5),
        Prim::F16 | Prim::Bf16 => serde_json::from_value(
            json!({"type": "scalar", "value": {"dtype": prim.name(), "bits": "3e00"}}),
        )
        .expect("half fixture"),
        integer => scalar_integer(integer, 3),
    }
}

/// The well-formed value of `shape`.
fn well_formed(shape: &Shape) -> ExecutionValue {
    match shape {
        Shape::Scalar(prim) => scalar(*prim),
        Shape::Bool => ExecutionValue::Bool { value: true },
        Shape::Str => ExecutionValue::String {
            value: "s".to_string(),
        },
        Shape::Tuple(items) => ExecutionValue::Tuple {
            value: items.iter().map(well_formed).collect(),
        },
        Shape::List(item) => ExecutionValue::List {
            value: vec![well_formed(item), well_formed(item)],
        },
        Shape::Adt { ctor, fields, .. } => ExecutionValue::Adt {
            ctor: ctor.to_string(),
            fields: fields.iter().map(|(_, shape)| well_formed(shape)).collect(),
        },
        Shape::Tensor(prim, extents) => tensor(*prim, extents),
    }
}

/// Values planted in place of a well-formed position.
fn foreign() -> Vec<ExecutionValue> {
    vec![
        scalar_integer(Prim::Int64, 1),
        scalar_integer(Prim::Int32, 1),
        scalar_integer(Prim::Int8, 1),
        scalar_f32(1.0),
        scalar_f64(1.0),
        serde_json::from_value(
            json!({"type": "scalar", "value": {"dtype": "f16", "bits": "3c00"}}),
        )
        .unwrap(),
        ExecutionValue::Bool { value: false },
        ExecutionValue::String {
            value: "t".to_string(),
        },
        ExecutionValue::Unit,
        ExecutionValue::Tuple {
            value: vec![scalar_integer(Prim::Int64, 1)],
        },
        ExecutionValue::List {
            value: vec![ExecutionValue::String {
                value: "t".to_string(),
            }],
        },
        ExecutionValue::Adt {
            ctor: "Other".to_string(),
            fields: vec![scalar_f32(1.0)],
        },
        ExecutionValue::Adt {
            ctor: "Inner".to_string(),
            fields: vec![scalar_f32(1.0)],
        },
        ExecutionValue::Adt {
            ctor: "Inner".to_string(),
            fields: Vec::new(),
        },
        ExecutionValue::Adt {
            ctor: "Green".to_string(),
            fields: Vec::new(),
        },
        tensor(Prim::F32, &[1]),
        tensor(Prim::F32, &[3]),
        tensor(Prim::F64, &[3]),
        tensor(Prim::F32, &[3, 1]),
    ]
}

/// Whether the planted `value` is a value of `shape`, judged by the oracle
/// independently of the compiler.
fn is_value_of(value: &ExecutionValue, shape: &Shape) -> bool {
    let json = serde_json::to_value(value).unwrap();
    match shape {
        Shape::Scalar(prim) => {
            json["type"] == "scalar" && wire_dtype(&json["value"]["dtype"]) == Some(*prim)
        }
        Shape::Bool => json["type"] == "bool",
        Shape::Str => json["type"] == "string",
        Shape::Tuple(items) => match value {
            ExecutionValue::Tuple { value } => {
                value.len() == items.len()
                    && value
                        .iter()
                        .zip(items)
                        .all(|(value, item)| is_value_of(value, item))
            }
            _ => false,
        },
        Shape::List(item) => match value {
            ExecutionValue::List { value } => value.iter().all(|value| is_value_of(value, item)),
            _ => false,
        },
        Shape::Adt { ty, fields, .. } => match value {
            ExecutionValue::Adt {
                ctor,
                fields: values,
            } => {
                let owner = match ctor.as_str() {
                    "Red" | "Green" => "Color",
                    other => other,
                };
                owner == *ty
                    && values.len() == fields.len()
                    && values
                        .iter()
                        .zip(fields)
                        .all(|(value, (_, field))| is_value_of(value, field))
            }
            _ => false,
        },
        Shape::Tensor(prim, extents) => {
            json["type"] == "tensor"
                && wire_dtype(&json["value"]["data"]["dtype"]) == Some(*prim)
                && json["value"]["shape"] == json!(extents)
        }
    }
}

fn wire_dtype(spelling: &serde_json::Value) -> Option<Prim> {
    Prim::parse_interchange_name(spelling.as_str()?)
}

/// Every position of `shape`'s well-formed value, as the admission names it,
/// with the shape declared there and a function that rebuilds the value with
/// that position replaced.
type Planter = Box<dyn Fn(ExecutionValue) -> ExecutionValue>;

fn positions(shape: &Shape, at: String) -> Vec<(String, Shape, Planter)> {
    let mut out: Vec<(String, Shape, Planter)> =
        vec![(at.clone(), shape.clone(), Box::new(|planted| planted))];
    let children: Vec<(String, Shape, usize)> = match shape {
        Shape::Tuple(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| (format!("{at}.{index}"), item.clone(), index))
            .collect(),
        Shape::List(item) => vec![(format!("{at}[1]"), (**item).clone(), 1)],
        Shape::Adt { fields, .. } => fields
            .iter()
            .enumerate()
            .map(|(index, (name, field))| {
                let label = match name {
                    Some(name) => format!("{at}.{name}"),
                    None => format!("{at}.{index}"),
                };
                (label, field.clone(), index)
            })
            .collect(),
        Shape::Scalar(_) | Shape::Bool | Shape::Str | Shape::Tensor(..) => Vec::new(),
    };
    for (label, child, index) in children {
        let parent = well_formed(shape);
        for (inner_label, inner_shape, inner_plant) in positions(&child, label) {
            let parent = parent.clone();
            let plant: Planter = Box::new(move |planted| {
                let mut value = parent.clone();
                let slot = match &mut value {
                    ExecutionValue::Tuple { value }
                    | ExecutionValue::List { value }
                    | ExecutionValue::Adt { fields: value, .. } => &mut value[index],
                    _ => unreachable!("a position has a container parent"),
                };
                *slot = inner_plant(planted);
                value
            });
            out.push((inner_label, inner_shape, plant));
        }
    }
    out
}

fn source() -> String {
    let mut source = SOURCE_TYPES.to_string();
    for (index, (spelling, _)) in shapes().iter().enumerate() {
        source.push_str(&format!(
            "\ndef keep_{index}(p: {spelling}) -> i64 = 1i64\n"
        ));
        source.push_str(&format!(
            "\ndef same_{index}(p: {spelling}) -> {spelling} = p\n"
        ));
    }
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

fn call(
    program: &PreparedEval,
    root: &str,
    value: ExecutionValue,
) -> Result<chelis_compiler_api::schema::EvalResult, String> {
    let mut arguments = BTreeMap::new();
    arguments.insert("p".to_string(), value);
    program
        .call_root(root, arguments, BTreeMap::new())
        .map_err(|error| {
            error
                .errors
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect::<Vec<_>>()
                .join("; ")
        })
}

#[test]
fn every_planted_wrong_type_is_rejected_at_its_position() {
    let program = prepared();
    let mut planted = 0usize;
    for (index, (spelling, shape)) in shapes().iter().enumerate() {
        let keep = format!("keep_{index}");
        let same = format!("same_{index}");

        // The well-formed value is admitted, and the identity call returns it
        // as its owed root.
        let value = well_formed(shape);
        let kept = call(&program, &keep, value.clone())
            .unwrap_or_else(|error| panic!("{spelling}: well-formed value rejected: {error}"));
        assert_eq!(kept.roots.len(), 1, "{spelling}: {:?}", kept.roots);
        let returned = call(&program, &same, value)
            .unwrap_or_else(|error| panic!("{spelling}: identity call rejected: {error}"));
        assert!(
            !returned.roots.is_empty(),
            "{spelling}: the identity call produced no owed root"
        );

        for (position, declared, plant) in positions(shape, "p".to_string()) {
            for foreign in foreign() {
                if is_value_of(&foreign, &declared) {
                    continue;
                }
                planted += 1;
                let mutated = plant(foreign.clone());
                let error = match call(&program, &keep, mutated) {
                    Ok(result) => panic!(
                        "{spelling}: {:?} planted at `{position}` (declared {declared:?}) was \
                         admitted and evaluated to {:?}",
                        serde_json::to_value(&foreign).unwrap(),
                        result.roots
                    ),
                    Err(error) => error,
                };
                // The rejection names the planted position, or a position
                // inside the planted value.
                let named = ["`", ".", "[", "{"]
                    .iter()
                    .any(|next| error.contains(&format!("`{position}{next}")));
                assert!(
                    named
                        && (error.contains("is not a value of its parameter type")
                            || error.contains("field(s), and the constructor declares")),
                    "{spelling}: the rejection of a value planted at `{position}` does not name \
                     it: {error}"
                );
            }
        }
    }
    assert!(planted > 400, "the oracle planted only {planted} mutations");

    // An invariant-carrying value that breaks its invariant, at each depth.
    let broken = ExecutionValue::Adt {
        ctor: "Positive".to_string(),
        fields: vec![scalar_f32(-1.0)],
    };
    for (index, (spelling, shape)) in shapes().iter().enumerate() {
        for (position, declared, plant) in positions(shape, "p".to_string()) {
            if !matches!(declared, Shape::Adt { ty: "Positive", .. }) {
                continue;
            }
            let error = call(&program, &format!("keep_{index}"), plant(broken.clone()))
                .expect_err("an invariant-breaking value is rejected");
            assert!(
                error.contains("breaks an invariant"),
                "{spelling} at `{position}`: {error}"
            );
        }
    }
}

#[test]
fn identity_calls_return_their_argument_exactly() {
    let program = prepared();
    for (index, (spelling, shape)) in shapes().iter().enumerate() {
        if !matches!(shape, Shape::Scalar(_) | Shape::Bool | Shape::Str) {
            continue;
        }
        let value = well_formed(shape);
        let result = call(&program, &format!("same_{index}"), value.clone())
            .unwrap_or_else(|error| panic!("{spelling}: {error}"));
        let [root] = result.roots.as_slice() else {
            panic!("{spelling}: {:?}", result.roots);
        };
        assert_eq!(
            serde_json::to_value(&root.value).unwrap(),
            serde_json::to_value(&value).unwrap(),
            "{spelling}"
        );
    }
}
