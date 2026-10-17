//! chelis#3801: a selected parameterized root is called with typed argument
//! values ([05-OBS-7], [05-OBS-9]).
//!
//! A call supplies every authored non-tensor parameter by name, as a value of
//! exactly the parameter's checked type; tensors keep the binding form. Each
//! admitted call computes what the same definition applied to the same values
//! as literals computes, runs on the Host lane, and lists no argument among
//! the manifest's required inputs. Every malformed call is rejected before
//! evaluation: a missing, extra, tensor-valued or mistyped argument, a
//! narrowing or a promotion, a value of a generic data type, and a record that
//! breaks its type's invariant.

use std::collections::BTreeMap;

#[allow(deprecated)]
use chelis_compiler_api::compiler::{PreparedEval, eval_selected, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, EvalResult, ExecutionValue, SourceKind};
use chelis_types::types::Lane;
use chelis_types::types::Prim;

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use wire_values::{scalar_f32, scalar_f64, scalar_integer};

const SOURCE: &str = r#"module M
@opaque
@invariant(p) (p.v >= 0.0)
type Positive =
  | Positive { v: f32 }

type Pair =
  | Pair { left: i64, right: f32 }

def mixed(x: i64, s: string, b: bool, y: f32) -> i64 = if (b && (s == "a") && (y > 1.0)) then add(x, 1i64) else x

def pair_sum(p: Pair) -> f32 = (p.right + 1.0)

def positive_value(p: Positive) -> f32 = p.v

def list_length(xs: List[i64]) -> i64 = len(xs)

def maybe(o: Option[i64]) -> bool = true

def scaled(t: tensor[2, f32], k: f32) -> f32 = k

def tensor_echo(t: tensor[2, f32], k: f32) -> tensor[2, f32] = t

def constant() -> i64 = 7i64
"#;

#[allow(deprecated)]
fn prepared(source: &str) -> PreparedEval {
    prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .expect("source compiles")
}

/// The root `literal` of `SOURCE` extended with `literal = <expression>`.
fn literal(expression: &str) -> serde_json::Value {
    let source = format!("{SOURCE}\nliteral = {expression}\n");
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source,
            bindings: BTreeMap::new(),
        },
        &["literal".to_string()],
    )
    .expect("literal application evaluates");
    root_value(&result)
}

fn root_value(result: &EvalResult) -> serde_json::Value {
    let [root] = result.roots.as_slice() else {
        panic!("one root: {:?}", result.roots);
    };
    serde_json::to_value(&root.value).expect("value serializes")
}

fn args(values: Vec<(&str, ExecutionValue)>) -> BTreeMap<String, ExecutionValue> {
    values
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect()
}

fn string(value: &str) -> ExecutionValue {
    ExecutionValue::String {
        value: value.to_string(),
    }
}

fn boolean(value: bool) -> ExecutionValue {
    ExecutionValue::Bool { value }
}

fn rejection(program: &PreparedEval, root: &str, arguments: Vec<(&str, ExecutionValue)>) -> String {
    let error = program
        .call_root(root, args(arguments), BTreeMap::new())
        .expect_err("the call is rejected");
    error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

#[test]
fn admitted_calls_compute_the_literal_application_on_the_host_lane() {
    let program = prepared(SOURCE);
    for (b, s, y, expression) in [
        (true, "a", 1.5f32, "mixed(3i64, \"a\", true, 1.5)"),
        (true, "b", 1.5f32, "mixed(3i64, \"b\", true, 1.5)"),
        (false, "a", 0.5f32, "mixed(3i64, \"a\", false, 0.5)"),
    ] {
        let result = program
            .call_root(
                "mixed",
                args(vec![
                    ("x", scalar_integer(Prim::Int64, 3)),
                    ("s", string(s)),
                    ("b", boolean(b)),
                    ("y", scalar_f32(y)),
                ]),
                BTreeMap::new(),
            )
            .expect("the call evaluates");
        assert_eq!(root_value(&result), literal(expression), "{expression}");
        let entry = result
            .manifest
            .entries
            .iter()
            .find(|entry| entry.name == "mixed")
            .expect("the call is a manifest entry");
        assert_eq!(entry.lane, Lane::Host, "{expression}");
        assert!(
            entry.required_inputs.is_empty(),
            "argument values are call inputs, not required inputs: {:?}",
            entry.required_inputs
        );
    }

    let record = ExecutionValue::Adt {
        ctor: "Pair".to_string(),
        fields: vec![scalar_integer(Prim::Int64, 2), scalar_f32(0.25)],
    };
    let result = program
        .call_root("pair_sum", args(vec![("p", record)]), BTreeMap::new())
        .expect("a record argument evaluates");
    assert_eq!(
        root_value(&result),
        literal("pair_sum(Pair { left: 2i64, right: 0.25 })")
    );

    let list = ExecutionValue::List {
        value: vec![
            scalar_integer(Prim::Int64, 1),
            scalar_integer(Prim::Int64, 2),
        ],
    };
    let result = program
        .call_root("list_length", args(vec![("xs", list)]), BTreeMap::new())
        .expect("a list argument evaluates");
    assert_eq!(root_value(&result), literal("list_length([1i64, 2i64])"));

    // The same compile serves every call: a second call of the same root with
    // other values computes its own result.
    let again = program
        .call_root(
            "mixed",
            args(vec![
                ("x", scalar_integer(Prim::Int64, -9)),
                ("s", string("a")),
                ("b", boolean(true)),
                ("y", scalar_f32(2.0)),
            ]),
            BTreeMap::new(),
        )
        .expect("the call evaluates");
    assert_eq!(
        root_value(&again),
        literal("mixed(-9i64, \"a\", true, 2.0)")
    );
}

#[test]
fn malformed_calls_are_rejected_before_evaluation() {
    let program = prepared(SOURCE);
    let well_formed = || {
        vec![
            ("x", scalar_integer(Prim::Int64, 3)),
            ("s", string("a")),
            ("b", boolean(true)),
            ("y", scalar_f32(1.5)),
        ]
    };

    let mut missing = well_formed();
    missing.retain(|(name, _)| *name != "s");
    assert!(rejection(&program, "mixed", missing).contains("supplies no value for parameter `s`"));

    let mut extra = well_formed();
    extra.push(("z", boolean(false)));
    assert!(rejection(&program, "mixed", extra).contains("`z`, which names no parameter"));

    // No promotion: an i32 is not an i64, and no narrowing: an f64 is not an
    // f32, whatever its value.
    let mut promoted = well_formed();
    promoted[0] = ("x", scalar_integer(Prim::Int32, 3));
    assert!(
        rejection(&program, "mixed", promoted).contains("argument `x` of `mixed` is not a value")
    );
    let mut narrowed = well_formed();
    narrowed[3] = ("y", scalar_f64(1.5));
    assert!(
        rejection(&program, "mixed", narrowed).contains("argument `y` of `mixed` is not a value")
    );
    let mut wrong_kind = well_formed();
    wrong_kind[2] = ("b", string("true"));
    assert!(
        rejection(&program, "mixed", wrong_kind).contains("argument `b` of `mixed` is not a value")
    );

    let mismatched_list = ExecutionValue::List {
        value: vec![scalar_integer(Prim::Int32, 1)],
    };
    assert!(
        rejection(&program, "list_length", vec![("xs", mismatched_list)])
            .contains("argument `xs` of `list_length` is not a value")
    );

    let wrong_record = ExecutionValue::Adt {
        ctor: "Positive".to_string(),
        fields: vec![scalar_f32(1.0)],
    };
    assert!(
        rejection(&program, "pair_sum", vec![("p", wrong_record)])
            .contains("argument `p` of `pair_sum` is not a value")
    );

    // A record value of an invariant-carrying type enters through the codec,
    // which revalidates the invariant.
    let broken = ExecutionValue::Adt {
        ctor: "Positive".to_string(),
        fields: vec![scalar_f32(-1.0)],
    };
    assert!(
        rejection(&program, "positive_value", vec![("p", broken)]).contains("breaks an invariant")
    );
    let valid = ExecutionValue::Adt {
        ctor: "Positive".to_string(),
        fields: vec![scalar_f32(2.0)],
    };
    program
        .call_root("positive_value", args(vec![("p", valid)]), BTreeMap::new())
        .expect("a record that keeps its invariant is admitted");

    let some = ExecutionValue::Adt {
        ctor: "Some".to_string(),
        fields: vec![scalar_integer(Prim::Int64, 1)],
    };
    let generic = rejection(&program, "maybe", vec![("o", some)]);
    assert!(generic.contains("generic data type `Option`"), "{generic}");

    // A tensor parameter arrives as a binding, never as an argument.
    let tensor_argument = rejection(
        &program,
        "scaled",
        vec![("t", scalar_f32(1.0)), ("k", scalar_f32(1.0))],
    );
    assert!(
        tensor_argument.contains("arrives as a binding"),
        "{tensor_argument}"
    );

    // Tensor parameters follow the free-tensor-input rule: a live one is a
    // required input, and a call without its binding is rejected rather than
    // returning no root; a dead one needs no binding.
    let unbound = rejection(&program, "tensor_echo", vec![("k", scalar_f32(1.0))]);
    assert!(unbound.contains("is not owed"), "{unbound}");
    let tensor: ExecutionValue = serde_json::from_value(serde_json::json!({
        "type": "tensor",
        "value": {"shape": [2], "data": {"dtype": "f32", "bits": ["3f800000", "40000000"]}}
    }))
    .unwrap();
    let ExecutionValue::Tensor { value: binding } = &tensor else {
        unreachable!("a tensor fixture");
    };
    let echoed = program
        .call_root(
            "tensor_echo",
            args(vec![("k", scalar_f32(1.0))]),
            BTreeMap::from([("t".to_string(), binding.clone())]),
        )
        .expect("a call with its live tensor bound evaluates");
    assert_eq!(root_value(&echoed), serde_json::to_value(&tensor).unwrap());
    let dead = program
        .call_root(
            "scaled",
            args(vec![("k", scalar_f32(2.5))]),
            BTreeMap::new(),
        )
        .expect("a dead tensor parameter needs no binding");
    assert_eq!(
        root_value(&dead),
        serde_json::to_value(scalar_f32(2.5)).unwrap()
    );

    assert!(rejection(&program, "constant", vec![]).contains("is not a parameterized definition"));
    assert!(rejection(&program, "absent", vec![]).contains("is not a parameterized definition"));
}
