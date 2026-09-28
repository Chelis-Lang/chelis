//! chelis#2413 step 3, slice 4: keys as spec/10 section 3.2 execution values.
//!
//! A scalar key is `{"type":"key","bits":h}` and a key tensor's storage is
//! `{"dtype":"key","bits":[h,...]}` inside a tensor execution value, with `h`
//! exactly 16 lowercase hex digits. Eval prints both, and both decode back:
//! as eval bindings, as a key field of a data type, and through the
//! `EvalResult` envelope. A key has no scalar object, and a graph's storage
//! carrier still holds no key. Expected words come from
//! `briefs/switch-design-probes/key_ref.py` and `slice2_ref.py`.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_compiler_api::{RuntimeValue, decode_adt_value};
use chelis_deep::ast::Expr;
use serde_json::{Value, json};

/// `split_keys(key_from_seed(7), 3)`.
const ROWS_KEY7: [&str; 3] = ["25ea33e61c10576f", "707124fbecd5f054", "823936153a565205"];

fn run(source: &str, bindings: BTreeMap<String, TensorValue>) -> EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings,
    })
    .unwrap_or_else(|error| panic!("eval failed: {error:?}"))
}

fn root_json(result: &EvalResult, name: &str) -> Value {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no root `{name}`"));
    serde_json::to_value(&root.value).expect("execution value serializes")
}

/// Evaluate the entry `main` with runtime `bindings` for its parameters.
fn run_main(source: &str, bindings: BTreeMap<String, TensorValue>) -> EvalResult {
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings,
        },
        &["main".to_string()],
    )
    .unwrap_or_else(|error| panic!("eval failed: {error:?}"))
}

fn only_root_json(result: &EvalResult) -> Value {
    let [root] = result.roots.as_slice() else {
        panic!("expected exactly one root, got {}", result.roots.len())
    };
    serde_json::to_value(&root.value).expect("execution value serializes")
}

#[test]
fn eval_prints_scalar_keys_and_key_tensors_in_the_spec_grammar() {
    let result = run(
        "kk = key_from_seed(7i64)\nkm = key_from_seed(-1i64)\nks = split_keys(key_from_seed(7i64), 3i64)\n",
        BTreeMap::new(),
    );
    assert_eq!(
        root_json(&result, "kk"),
        json!({"type": "key", "bits": "0000000000000007"})
    );
    assert_eq!(
        root_json(&result, "km"),
        json!({"type": "key", "bits": "ffffffffffffffff"})
    );
    assert_eq!(
        root_json(&result, "ks"),
        json!({"type": "tensor", "value": {"shape": [3], "data": {"dtype": "key", "bits": ROWS_KEY7}}})
    );
    // The envelope round-trips: print, parse, print again.
    let printed = serde_json::to_string(&result).expect("envelope serializes");
    let parsed: EvalResult = serde_json::from_str(&printed).expect("envelope decodes");
    assert_eq!(serde_json::to_string(&parsed).unwrap(), printed);
}

#[test]
fn printed_keys_feed_back_as_bindings() {
    let printed = run(
        "k = key_from_seed(7i64)\nks = split_keys(key_from_seed(7i64), 3i64)\n",
        BTreeMap::new(),
    );
    let ExecutionValue::Key { bits } = serde_json::from_value(root_json(&printed, "k")).unwrap()
    else {
        panic!("a scalar key decodes as the key variant")
    };
    assert_eq!(bits.key().bits(), 7);
    // Bindings supply tensor parameters only (`eval_selected`), so the
    // printed key tensor is what feeds back.
    let ExecutionValue::Tensor { value: rows } =
        serde_json::from_value(root_json(&printed, "ks")).unwrap()
    else {
        panic!("a key tensor decodes as a tensor")
    };
    let echoed = run_main(
        "def main(ks: tensor[3, key]) -> tensor[3, key] = ks\n",
        BTreeMap::from([("ks".to_string(), rows)]),
    );
    assert_eq!(
        only_root_json(&echoed),
        json!({"type": "tensor", "value": {"shape": [3], "data": {"dtype": "key", "bits": ROWS_KEY7}}})
    );
}

#[test]
fn a_key_field_of_a_data_type_decodes_only_from_a_key() {
    let decls =
        chelis_surf::parser::parse_str("type Boxed = | Boxed { k: key }\n").expect("surf parse");
    let program: Vec<Expr> =
        chelis_surf::desugar::desugar_program(&decls).expect("the fixture desugars");
    let decoded = decode_adt_value(
        &program,
        &serde_json::from_value(json!({
            "type": "adt", "ctor": "Boxed",
            "fields": [{"type": "key", "bits": "0000000000000007"}]
        }))
        .unwrap(),
    )
    .expect("a key field decodes");
    let RuntimeValue::Adt { fields, .. } = decoded else {
        panic!("an ADT value")
    };
    assert!(matches!(fields.as_slice(), [RuntimeValue::Key(key)] if key.bits() == 7));
    let error = decode_adt_value(
        &program,
        &serde_json::from_value(json!({
            "type": "adt", "ctor": "Boxed",
            "fields": [{"type": "scalar", "value": {"dtype": "int64", "value": 7}}]
        }))
        .unwrap(),
    )
    .expect_err("an integer is not a key");
    assert!(error.contains("expects scalar `key`"), "{error}");
}

#[test]
fn key_decoders_are_strict() {
    let digits15 = "000000000000007";
    let digits17 = "00000000000000007";
    let upper = "00000000000000AB";
    for scalar in [
        json!({"type": "key", "bits": digits15}),
        json!({"type": "key", "bits": digits17}),
        json!({"type": "key", "bits": upper}),
        json!({"type": "key", "bits": 7}),
        json!({"type": "key", "bits": "0000000000000007", "extra": 1}),
        json!({"type": "key", "value": "0000000000000007"}),
        // A key is not a number: it has no scalar object.
        json!({"type": "scalar", "value": {"dtype": "key", "bits": "0000000000000007"}}),
    ] {
        assert!(
            serde_json::from_value::<ExecutionValue>(scalar.clone()).is_err(),
            "accepted {scalar}"
        );
    }
    for storage in [
        json!({"dtype": "key", "bits": [digits15]}),
        json!({"dtype": "key", "bits": [digits17]}),
        json!({"dtype": "key", "bits": [upper]}),
        json!({"dtype": "key", "bits": [7]}),
        json!({"dtype": "key", "values": [7]}),
        json!({"dtype": "key", "bits": ["0000000000000007"], "values": [7]}),
    ] {
        let tensor = json!({"shape": [1], "data": storage});
        assert!(
            serde_json::from_value::<TensorValue>(tensor.clone()).is_err(),
            "accepted {tensor}"
        );
    }
    // The positive control for the same decoder.
    serde_json::from_value::<TensorValue>(
        json!({"shape": [1], "data": {"dtype": "key", "bits": ["0000000000000007"]}}),
    )
    .expect("a well-formed key tensor decodes");
    // A graph's storage carrier holds no key literal (spec/10 section 3.2).
    let error = serde_json::from_value::<chelis_types::TensorStorage>(
        json!({"dtype": "key", "bits": ["0000000000000007"]}),
    )
    .expect_err("a graph holds no key literal");
    assert!(error.to_string().contains("no literal carrier"), "{error}");
}
