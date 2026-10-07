//! Deep execution results must release their wire tree without native recursion.

use std::process::Command;

use chelis_compiler_api::schema::{
    DictEntryValue, EXECUTION_VALUE_SCHEMA_VERSION, EvalResult, EvaluatedRoot, ExecutionValue,
    RootManifestResult,
};

const CHILD_CASE_ENV: &str = "CHELIS_DEEP_EXECUTION_VALUE_DROP_CHILD";
const SERIALIZE_CHILD_CASE_ENV: &str = "CHELIS_DEEP_EXECUTION_VALUE_SERIALIZE_CHILD";
const DEPTH: usize = 200_000;
const SERIALIZE_DEPTH: usize = 5_000;

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

#[test]
fn serialization_write_failure_is_reported() {
    struct RefuseWrite;
    impl std::io::Write for RefuseWrite {
        fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("writer refused value"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let value = wrap("adt", ExecutionValue::Unit);
    let error = serde_json::to_writer(RefuseWrite, &value).expect_err("writer failure propagates");
    assert!(error.is_io(), "unexpected error: {error}");
}

#[test]
fn deep_serialize_child() {
    let Ok(case) = std::env::var(SERIALIZE_CHILD_CASE_ENV) else {
        return;
    };
    let mut value = ExecutionValue::Unit;
    for _ in 0..SERIALIZE_DEPTH {
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
        .spawn(move || {
            let json = serde_json::to_string(&result).expect("serialize deep result");
            assert!(json.contains(&format!(
                "\"schema_version\":{EXECUTION_VALUE_SCHEMA_VERSION}"
            )));
            assert!(json.contains("\"type\":\"unit\""));
            assert!(json.len() > SERIALIZE_DEPTH);
        })
        .expect("spawn small-stack serialization thread")
        .join()
        .expect("serialization complete");
}

#[test]
fn every_result_container_serializes_on_a_small_stack() {
    for case in ["list", "tuple", "adt", "dict"] {
        let output = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", "deep_serialize_child", "--nocapture"])
            .env(SERIALIZE_CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run {case} child: {error}"));
        assert!(
            output.status.success(),
            "{case} serialization aborted: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
