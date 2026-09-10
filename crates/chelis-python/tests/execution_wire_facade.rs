//! spec/10 §3.2 and spec/11: actual Rust codec → facade → registered PyO3 → Rust.
use chelis_compiler_api::schema::TensorValue;
use pyo3::{prelude::*, types::PyModule};
use serde_json::json;
use std::{ffi::CString, path::PathBuf, process::Command};

#[test]
fn actual_facade_and_native_preserve_exact_values_and_reject_malformed_carriers() {
    let mut cases = Vec::new();
    for dtype in ["f16", "bf16"] {
        cases.push(json!({"shape":[65536],"data":{"dtype":dtype,"bits":
            (0..=u16::MAX).map(|word| format!("{word:04x}")).collect::<Vec<_>>()}}));
    }
    for (dtype, bits) in [
        (
            "f32",
            vec![
                "00000000", "80000000", "00000001", "007fffff", "7f7fffff", "7f800000", "ff800000",
                "7fc00042", "ff800042",
            ],
        ),
        (
            "f64",
            vec![
                "0000000000000000",
                "8000000000000000",
                "0000000000000001",
                "000fffffffffffff",
                "7fefffffffffffff",
                "7ff0000000000000",
                "fff0000000000000",
                "7ff8000000000042",
                "fff0000000000042",
            ],
        ),
    ] {
        cases.push(json!({"shape":[bits.len()],"data":{"dtype":dtype,"bits":bits}}));
    }
    for (dtype, low, high) in [
        ("int8", i64::from(i8::MIN), i64::from(i8::MAX)),
        ("int16", i64::from(i16::MIN), i64::from(i16::MAX)),
        ("int32", i64::from(i32::MIN), i64::from(i32::MAX)),
        ("int64", i64::MIN, i64::MAX),
    ] {
        cases.push(json!({"shape":[2],"data":{"dtype":dtype,"values":[low,high]}}));
    }
    cases.push(json!({"shape":[2],"data":{"dtype":"bool","values":[false,true]}}));
    cases.push(json!({"shape":[],"data":{"dtype":"f32","bits":["80000000"]}}));
    cases.push(json!({"shape":[0],"data":{"dtype":"f64","bits":[]}}));

    // The reference is emitted by the real Rust codec, not a Python fixture.
    let tensors: Vec<TensorValue> = cases
        .iter()
        .cloned()
        .map(|value| serde_json::from_value(value).expect("valid reference tensor"))
        .collect();
    let reference = serde_json::to_string(&tensors).expect("Rust codec");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = std::env::var_os("PYO3_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(".venv/bin/python"));
    let output = Command::new(python)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ])
        .output()
        .expect("query managed Python dependencies");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let site = String::from_utf8(output.stdout).expect("Python site path");
    let returned = Python::with_gil(|py| {
        let sys = py.import("sys").expect("sys");
        let path = sys.getattr("path").expect("path");
        path.call_method1("insert", (0, site.trim()))
            .expect("managed dependencies");
        path.call_method1(
            "insert",
            (
                0,
                root.join("bindings/python").to_str().expect("facade path"),
            ),
        )
        .expect("facade source");
        let native = PyModule::new(py, "_native").expect("native module");
        chelis_python::register_module(&native).expect("actual PyO3 registration");
        sys.getattr("modules")
            .expect("modules")
            .set_item("chelis._native", native)
            .expect("publish actual native module");
        let source = CString::new(include_str!(
            "../../../bindings/python/tests/native_execution_wire.py"
        ))
        .expect("probe source");
        let probe = PyModule::from_code(
            py,
            &source,
            c"native_execution_wire.py",
            c"native_execution_wire",
        )
        .expect("import actual facade and probe");
        probe
            .getattr("run")
            .expect("run")
            .call1((reference,))
            .expect("actual boundary assertions")
            .extract::<String>()
            .expect("returned wire")
    });
    let decoded: Vec<TensorValue> = serde_json::from_str(&returned).expect("Rust consumer");
    assert_eq!(decoded.len(), 11, "every selected dtype and shape executed");
    assert_eq!(
        serde_json::to_value(decoded).expect("roundtrip wire"),
        json!(cases)
    );
}
