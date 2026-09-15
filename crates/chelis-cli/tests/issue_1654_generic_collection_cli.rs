//! chelis#1654 through the serialized CLI check surface.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::ExitStatus;
use tempfile::tempdir;

struct CheckResult {
    status: ExitStatus,
    report: serde_json::Value,
}

fn check_path(path: &Path) -> CheckResult {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    CheckResult {
        status: output.status,
        report: serde_json::from_slice(&output.stdout).expect("check JSON"),
    }
}

fn check(source: &str) -> CheckResult {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("collection_contract.ch");
    fs::write(&path, source).expect("write fixture");
    check_path(&path)
}

fn assert_rejected(source: &str, operation: &str) {
    let CheckResult { status, report } = check(source);
    assert!(!status.success(), "{status:?}: {report}");
    assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
    let errors = report["errors"].as_array().unwrap();
    assert!(
        errors.iter().any(|error| {
            error["kind"] == "TypeMismatch"
                && error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(operation))
        }),
        "{report}"
    );
}

fn assert_accepted(source: &str) {
    let CheckResult { status, report } = check(source);
    assert!(status.success(), "{status:?}: {report}");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(report["errors"].as_array().unwrap().is_empty(), "{report}");
}

fn assert_type_mismatch(source: &str) {
    let CheckResult { status, report } = check(source);
    assert!(!status.success(), "{status:?}: {report}");
    assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error["kind"] == "TypeMismatch"),
        "{report}"
    );
}

fn assert_type_or_dimension_rejected(source: &str) {
    let CheckResult { status, report } = check(source);
    assert!(!status.success(), "{status:?}: {report}");
    assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| matches!(
                error["kind"].as_str(),
                Some("TypeMismatch" | "DimensionMismatch")
            )),
        "{report}"
    );
}

#[test]
fn declaration_boundary_rejects_implicit_collection_contracts() {
    for (operation, source) in [
        ("len", "def size(x) = len(x)\n"),
        ("len", "def size[a](x: a) -> int64 = len(x)\n"),
        ("len", "measure = fn (x) -> len(x)\n"),
        ("len", "measure = len\ndef size(x) = measure(x)\n"),
        (
            "len",
            "def invoke(f, x) = f(x)\ndef size(x) = invoke(len, x)\n",
        ),
        ("index", "def at(xs, i) = index(xs, i)\n"),
        ("append", "def push(xs, x) = append(xs, x)\n"),
        ("concat", "def join(lhs, rhs) = concat(lhs, rhs)\n"),
    ] {
        assert_rejected(source, operation);
    }
    let CheckResult { report, .. } = check("def size(x) = len(x)\n");
    assert!(
        report["errors"].as_array().unwrap().iter().any(|error| {
            error["message"]
                .as_str()
                .is_some_and(|message| message.contains("in `size` at declaration boundary"))
        }),
        "{report}"
    );
}

#[test]
fn checked_builtin_values_reject_invalid_indirect_calls() {
    for (operation, source) in [
        ("len", "measure = len\nout = measure(1i64)\n"),
        ("index", "op = index\nout = op(1i64, 0i64)\n"),
        ("index", "op = index\nout = op([1i64], 0i32)\n"),
        ("index", "op = index\nout = op([1i64], 0i16)\n"),
        ("append", "op = append\nout = op([1i64], \"bad\")\n"),
        ("concat", "op = concat\nout = op([1i64], [1.0f32])\n"),
    ] {
        assert_rejected(source, operation);
    }
    assert_type_mismatch("measure = len\nout: string = measure([1i64])\n");
}

#[test]
fn explicit_and_transported_valid_collection_contracts_score_one() {
    for source in [
        "def size[a](xs: List[a]) -> int64 = len(xs)\nout = size([1i64])\n",
        "measure = len\nout = measure([1i64])\n",
        "measure = len\nout: int64 = measure([1i64])\n",
        "op = index\nout: int64 = op([1i64], 0i64)\n",
        "op = append\nout: List[int64] = op([1i64], 2i64)\n",
        "op = concat\nout: List[int64] = op([1i64], [2i64])\n",
    ] {
        assert_accepted(source);
    }
}

#[test]
fn transported_tensor_concat_keeps_axis_and_exact_shape_checks() {
    assert_type_or_dimension_rejected(
        "def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = {\n\
         op = concat\n\
         op([a, b], 1i32)\n\
         }\n",
    );
    assert_rejected(
        "def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
         op = concat\n\
         op([a, b], 9i32)\n\
         }\n",
        "concat",
    );
    for source in [
        "def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
         op = concat\n\
         op([a, b], 1i32)\n\
         }\n",
        "def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
         op = concat\n\
         op([a, b], -1i32)\n\
         }\n",
        "def good(a: tensor[2, 3, f32], b: tensor[4, 3, f32]) -> tensor[6, 3, f32] = {\n\
         op = concat\n\
         op([a, b], 0i32)\n\
         }\n",
    ] {
        assert_accepted(source);
    }
}

#[test]
fn higher_order_and_aggregate_transport_remain_checked() {
    for source in [
        "def identity(f) = f\nmeasure = identity(len)\nout = measure(1i64)\n",
        "def pair() = (len, 1i32)\nmeasure = pair().0\nout = measure(1i64)\n",
        "def invoke(f, x) = f(x)\nout = invoke(len, 1i64)\n",
        "def choose(flag: bool) = if flag then len else len\nmeasure = choose(true)\nout = measure(1i64)\n",
    ] {
        assert_rejected(source, "len");
    }
    for source in [
        "def pair() = (len, 1i32)\nout = pair().1\n",
        "def ignore(f) = 1i32\nout = ignore(len)\n",
        "def ignore(f, x) = x\nout = ignore(len, 1i64)\n",
    ] {
        assert_accepted(source);
    }
}

#[test]
fn imported_checked_values_keep_their_collection_contract() {
    let directory = tempdir().expect("tempdir");
    let dependency = directory.path().join("dep");
    let application = directory.path().join("app");
    fs::create_dir_all(dependency.join("src")).expect("dependency source directory");
    fs::create_dir_all(application.join("src")).expect("application source directory");
    fs::write(
        dependency.join("reef.toml"),
        format!(
            "[package]\nname = \"dep\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Contract\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    )
    .expect("dependency manifest");
    fs::write(
        dependency.join("src/measure.ch"),
        "module Contract.Measure\nexport (measure, join)\nmeasure = len\njoin = concat\n",
    )
    .expect("dependency source");
    fs::write(
        application.join("reef.toml"),
        format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Demo\"\n\n[dependencies]\ndep = {{ path = \"../dep\" }}\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    )
    .expect("application manifest");
    let main = application.join("src/main.ch");
    fs::write(
        &main,
        "module Demo.Main\nimport Contract.Measure (measure)\nout = measure(1i64)\n",
    )
    .expect("invalid application source");
    let CheckResult {
        status: rejected_status,
        report: rejected,
    } = check_path(&main);
    assert!(
        !rejected_status.success(),
        "{rejected_status:?}: {rejected}"
    );
    assert!(rejected["score"].as_f64().unwrap() < 1.0, "{rejected}");
    assert!(
        rejected["errors"].as_array().unwrap().iter().any(|error| {
            error["kind"] == "TypeMismatch"
                && error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("len"))
        }),
        "{rejected}"
    );

    fs::write(
        &main,
        "module Demo.Main\nimport Contract.Measure (measure)\nout: int64 = measure([1i64])\n",
    )
    .expect("valid application source");
    let CheckResult {
        status: accepted_status,
        report: accepted,
    } = check_path(&main);
    assert!(accepted_status.success(), "{accepted_status:?}: {accepted}");
    assert_eq!(accepted["score"].as_f64(), Some(1.0), "{accepted}");
    assert!(
        accepted["errors"].as_array().unwrap().is_empty(),
        "{accepted}"
    );

    fs::write(
        &main,
        "module Demo.Main\n\
         import Contract.Measure (join)\n\
         def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = \
         join([a, b], 1i32)\n",
    )
    .expect("invalid imported concat source");
    let CheckResult {
        status: rejected_status,
        report: rejected,
    } = check_path(&main);
    assert!(
        !rejected_status.success(),
        "{rejected_status:?}: {rejected}"
    );
    assert!(rejected["score"].as_f64().unwrap() < 1.0, "{rejected}");
    assert!(
        rejected["errors"].as_array().unwrap().iter().any(|error| {
            matches!(
                error["kind"].as_str(),
                Some("TypeMismatch" | "DimensionMismatch")
            )
        }),
        "{rejected}"
    );

    fs::write(
        &main,
        "module Demo.Main\n\
         import Contract.Measure (join)\n\
         def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = \
         join([a, b], 1i32)\n",
    )
    .expect("valid imported concat source");
    let CheckResult {
        status: accepted_status,
        report: accepted,
    } = check_path(&main);
    assert!(accepted_status.success(), "{accepted_status:?}: {accepted}");
    assert_eq!(accepted["score"].as_f64(), Some(1.0), "{accepted}");
    assert!(
        accepted["errors"].as_array().unwrap().is_empty(),
        "{accepted}"
    );
}
