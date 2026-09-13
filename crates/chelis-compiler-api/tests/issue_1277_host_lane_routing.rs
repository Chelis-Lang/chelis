//! chelis#1277 Slice B2h: which evaluator a user `def` application reaches.
//!
//! `eval_compiled` sends every `Lane::Tensor` manifest entry to
//! `chelis_ir::eval` and every other entry to the host interpreter in
//! `runtime/eval.rs`. The two controls here pin the lane each source form of
//! chelis#1376 lands in, with the manifest's own `lane` field as the witness,
//! so the routing change can be measured against them rather than inferred.
//!
//! `chelis eval --file` constructs its request with an empty bindings map
//! (`main.rs::try_eval_result_for_target`), and even an embedding that binds
//! `f`'s inputs cannot reach the DAG evaluator for it: the lowerer never gives
//! a `shape`-reading def a named root. Both facts are locked below so the
//! routing change is measured against them rather than inferred.

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{check, eval, eval_selected};
use chelis_compiler_api::schema::{
    CheckRequest, EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_types::types::Lane;

/// chelis#1376's callee: the declared result claims `m` on axis 1 while the
/// `insert` reads `n` for it.
const CALLEE_1376: &str = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = \
                           insert(x, 1, shape(x, 0))\n";

/// The CLI form: a nullary `main` applying `f` to two literals.
const HOST_APPLIED_1376: &str = "def f(x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = \
                                 insert(x, 1, shape(x, 0))\n\
                                 def main() = f(to_tensor([1.0f32, 2.0f32]), \
                                 to_tensor([3.0f32, 4.0f32, 5.0f32]))\n";

fn f32_tensor(shape: &[usize], data: &[f32]) -> TensorValue {
    TensorValue {
        shape: shape
            .iter()
            .map(|value| i64::try_from(*value).unwrap())
            .collect(),
        data: wire_values::storage_f32(data.to_vec()),
    }
}

fn root_shape(result: &EvalResult, name: &str) -> Vec<i64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root `{name}` in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value.shape.clone(),
        other => panic!("root `{name}` is not a tensor: {other:?}"),
    }
}

fn lane_of(result: &EvalResult, name: &str) -> Lane {
    result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no manifest entry `{name}` in {:?}", result.manifest))
        .lane
}

/// #2013: a missing named DAG root selects Host, not missing actual arguments.
#[test]
fn bound_parameterized_entry_uses_supplied_host_inputs() {
    let checked = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: CALLEE_1376.into(),
    })
    .unwrap();
    assert_eq!(checked.score.get(), 1.0);
    assert!(checked.errors.is_empty(), "{checked:?}");
    assert!(checked.unresolved_names.is_empty(), "{checked:?}");
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: CALLEE_1376.into(),
            bindings: BTreeMap::from([
                ("x".into(), f32_tensor(&[2], &[1.0, 2.0])),
                ("y".into(), f32_tensor(&[2], &[3.0, 4.0])),
            ]),
        },
        &["f".into()],
    )
    .unwrap();
    assert_eq!(lane_of(&result, "f"), Lane::Host);
    assert_eq!(result.roots.len(), 1);
    assert_eq!(
        serde_json::to_value(&result.roots[0].value).unwrap(),
        serde_json::json!({"type":"tensor","value":{"shape":[2,2],"data":{
            "dtype":"f32","bits":["3f800000","3f800000","40000000","40000000"]
        }}})
    );
    assert!(result.transcript.is_empty(), "{result:?}");
}

/// Supplying live actuals must reach the real named-extent guard (§4.7.2).
#[test]
fn bound_parameterized_mismatch_names_the_disagreeing_sources() {
    let mut bindings = BTreeMap::new();
    bindings.insert("x".to_string(), f32_tensor(&[2], &[1.0, 2.0]));
    bindings.insert("y".to_string(), f32_tensor(&[3], &[3.0, 4.0, 5.0]));
    let err = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: CALLEE_1376.to_string(),
            bindings,
        },
        &["f".to_string()],
    )
    .expect_err("a claimed 3 cannot silently return an actual axis of 2");
    let messages = err
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(err.stage, "eval");
    assert!(err.transcript.is_empty(), "{err:?}");
    assert!(
        messages.contains("extent `m`: y axis 0 = 3, x axis 0 = 2"),
        "{messages}"
    );
    assert!(
        messages.ends_with("numeric trap: domain in load at int64"),
        "{messages}"
    );
}

/// A matching call still routes the nullary main through the host lane.
#[test]
fn host_applied_def_main_is_a_host_lane_root() {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: HOST_APPLIED_1376.replace("3.0f32, 4.0f32, 5.0f32", "3.0f32, 4.0f32"),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"));

    assert_eq!(lane_of(&result, "main"), Lane::Host);
    assert_eq!(root_shape(&result, "main"), vec![2, 2]);
}

/// chelis#1376 through the inlined root: the foreign claim `m` on the set
/// axis of `insert(x, 1, shape(x, 0))` is witnessed by `y` and produced from
/// `x`, so `spec/04-type-system.md` §4.7.2 checks it at execution.
///
/// The single-letter binders make `n` and `m` polymorphic dimension
/// variables, which inference instantiates against the literal argument
/// extents. The root's inferred result therefore RESTATED `m` as the literal
/// 3, and that restatement rendered ahead of the named guard, so this row
/// asserted `claimed = 3, x axis 0 = 2` and named one source and a number.
/// chelis#1782 declines a literal claim whose comparison a named claim
/// already makes, and the guard that survives names both.
///
/// EVIDENTIARY STATUS: regression test for the ATTRIBUTION, in two steps.
/// The row was first red at the local extent consumer, with
/// `claimed = 3, node N axis 1 = 2` and `domain in expand`, naming a node
/// identity rather than any source. It is now red for the restated literal
/// measured at `d861a6c6f`, and asserts the two disagreeing sources
/// `spec/04-type-system.md` section 4.7's [04-NUM-9] asks for.
#[test]
fn host_applied_mismatch_names_the_disagreeing_sources() {
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: HOST_APPLIED_1376.to_string(),
        bindings: BTreeMap::new(),
    })
    .expect_err("a claimed 3 cannot silently return an actual axis of 2");
    let messages = error
        .errors
        .iter()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        messages.contains("extent `m`: y axis 0 = 3, x axis 0 = 2"),
        "{messages}"
    );
    assert!(
        messages.ends_with("numeric trap: domain in load at int64"),
        "{messages}"
    );
}

// #1897 / §4.5: anonymous unknown extents are not shared runtime binders.
#[allow(deprecated)] // Exercise repeated calls through the compatibility API too.
fn assert_host_values(source: &str, expected: &[serde_json::Value]) {
    let request = || EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_owned(),
        bindings: BTreeMap::new(),
    };
    let prepared = chelis_compiler_api::compiler::prepare_eval(request()).unwrap();
    let results = [
        eval_selected(request(), &["output".to_owned()]),
        prepared.eval_root(BTreeMap::new(), "output"),
        prepared.eval_root(BTreeMap::new(), "output"),
    ];
    for result in results {
        let result = result.unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        assert!(result.transcript.is_empty(), "{result:?}");
        let values = result
            .roots
            .iter()
            .map(|root| {
                assert_eq!(lane_of(&result, root.name.as_deref().unwrap()), Lane::Host);
                serde_json::to_value(&root.value).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, expected, "{source}");
    }
}

fn int64_values(values: &[i64]) -> Vec<serde_json::Value> {
    values
        .iter()
        .map(|value| serde_json::json!({"type":"scalar", "value":{"dtype":"int64", "value":value}}))
        .collect()
}

#[test]
fn wildcard_axes_remain_independent_in_direct_and_prepared_host_calls() {
    assert_host_values(
        include_str!("../../../examples/wildcard_extents.ch"),
        &int64_values(&[2, 3]),
    );
    for (input, dims) in [
        ("to_tensor([[1.0f32], [2.0f32], [3.0f32]])", [3, 1]),
        ("reshape(to_tensor(empty), [0i64, 3i64])", [0, 3]),
        ("reshape(to_tensor(empty), [4i64, 0i64])", [4, 0]),
    ] {
        let source = format!(
            "def widths(x: tensor[*, *, f32]) = (shape(&x, 0i32), shape(x, 1i32))\nempty: List[f32] = []\noutput = widths({input})\n"
        );
        assert_host_values(&source, &int64_values(&dims));
    }
}

#[test]
fn wildcard_arguments_do_not_hide_real_shared_host_binders() {
    let source = "def widths(x: tensor[extent, *, f32], y: tensor[extent, *, f32]) = (shape(&x, 0i32), shape(x, 1i32), shape(&y, 0i32), shape(y, 1i32))\noutput = widths(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), to_tensor([[5.0f32], [6.0f32]]))\n";
    assert_host_values(source, &int64_values(&[2, 2, 2, 1]));
    let mismatch = source.replace("[[5.0f32], [6.0f32]]", "[[5.0f32]]");
    let error = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: mismatch,
            bindings: BTreeMap::new(),
        },
        &["output".to_owned()],
    )
    .expect_err("the real extent binder must still agree");
    assert_eq!(error.stage, "eval");
    assert_eq!(error.errors.len(), 1);
    // chelis#1788 re-rendered this verdict. The wildcard exclusion chelis#1898
    // added is unchanged and is what this row actually pins: `*` axes stay
    // independent while the real binder `extent` is still caught. What moved is
    // the message, from a private sentence to the frozen [04-NUM-9] pair with
    // the declaring witness first, which is byte-identical to what the C lane
    // prints for the same program.
    assert_eq!(
        error.errors[0].message,
        "extent `extent`: x axis 0 = 2, y axis 0 = 1\nnumeric trap: domain in load at int64"
    );
}

#[test]
fn wildcard_column_concat_preserves_all_values_through_host_helpers() {
    let source = "def join_columns[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([x, y], 1i32)\ndef columns[s](x: tensor[s, s, f32], y: tensor[s, 1, f32]) = join_columns(x, y)\noutput = columns(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), to_tensor([[5.0f32], [6.0f32]]))\n";
    assert_host_values(
        source,
        &[serde_json::json!({"type":"tensor", "value": {
            "shape":[2,3], "data":{"dtype":"f32", "bits":["3f800000", "40000000", "40a00000", "40400000", "40800000", "40c00000"]}
        }})],
    );
}
