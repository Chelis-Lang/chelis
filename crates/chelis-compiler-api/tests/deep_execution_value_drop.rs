//! Deep execution results must release their wire tree without native recursion.

use std::process::Command;

use chelis_compiler_api::schema::{
    DictEntryValue, EXECUTION_VALUE_SCHEMA_VERSION, EvalResult, EvaluatedRoot, ExecutionValue,
    RootManifestResult,
};

const CHILD_CASE_ENV: &str = "CHELIS_DEEP_EXECUTION_VALUE_DROP_CHILD";
const DEPTH: usize = 200_000;

fn wrap(case: &str, value: ExecutionValue) -> ExecutionValue {
    match case {
        "list" => ExecutionValue::List { value: vec![value] },
        "tuple" => ExecutionValue::Tuple { value: vec![value] },
        "adt" => ExecutionValue::Adt {
            ctor: "Link".to_owned(),
            fields: vec![value],
        },
        "dict" => ExecutionValue::Dict {
            entries: vec![DictEntryValue {
                key: ExecutionValue::String {
                    value: "k".to_owned(),
                },
                value,
            }],
        },
        other => panic!("unknown container {other}"),
    }
}

#[test]
fn deep_drop_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    let mut value = ExecutionValue::Unit;
    for _ in 0..DEPTH {
        value = wrap(&case, value);
    }
    let result = EvalResult {
        schema_version: EXECUTION_VALUE_SCHEMA_VERSION,
        roots: vec![EvaluatedRoot {
            node_id: 0,
            name: Some("a".to_owned()),
            value,
            display: None,
        }],
        manifest: RootManifestResult::default(),
        transcript: Vec::new(),
    };
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || drop(result))
        .expect("spawn small-stack drop thread")
        .join()
        .expect("release complete");
}

#[test]
fn every_result_container_releases_on_a_small_stack() {
    for case in ["list", "tuple", "adt", "dict"] {
        let output = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "deep_drop_child", "--nocapture"])
            .env(CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run {case} child: {error}"));
        assert!(
            output.status.success(),
            "{case} release aborted: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn shallow_execution_value_keeps_its_wire_shape() {
    for (case, expected) in [
        ("list", r#"{"type":"list","value":[{"type":"unit"}]}"#),
        ("tuple", r#"{"type":"tuple","value":[{"type":"unit"}]}"#),
        (
            "adt",
            r#"{"type":"adt","ctor":"Link","fields":[{"type":"unit"}]}"#,
        ),
        (
            "dict",
            r#"{"type":"dict","entries":[{"key":{"type":"string","value":"k"},"value":{"type":"unit"}}]}"#,
        ),
    ] {
        let value = wrap(case, ExecutionValue::Unit);
        assert_eq!(
            serde_json::to_string(&value).expect("serialize shallow value"),
            expected,
            "{case} wire spelling changed"
        );
    }
}
