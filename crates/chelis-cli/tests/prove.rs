use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::tempdir;

fn write_prop(source: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("prop.ch"), source).expect("write property");
    dir
}

#[test]
fn prove_passes_filtered_samples() {
    let dir = write_prop(
        r#"
@property non_negative forall(x: f32) where x >= 0.0:
  x >= 0.0
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "3",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: non_negative -- 3/3 passed",
        ));
}

#[test]
fn prove_reports_counterexample_exit_one() {
    let dir = write_prop(
        r#"
@property always_non_negative forall(x: f32):
  x >= 0.0
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "10",
            "--seed",
            "0",
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains(
            "property failure: always_non_negative",
        ));
}

#[test]
fn prove_json_schema_has_property_and_summary_records() {
    let dir = write_prop(
        r#"
@property truth forall(x: bool):
  x == x
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "2",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = String::from_utf8(output.stdout).expect("utf8");
    let records = lines
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["kind"], "property");
    assert_eq!(records[0]["name"], "truth");
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(records[1]["kind"], "summary");
    assert_eq!(records[1]["passed"], 1);
}

#[test]
fn property_binder_types_are_required() {
    let dir = write_prop(
        r#"
@property missing_type forall(x):
  true
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("explicit property binder type"));
}

#[test]
fn symbolic_tensor_binder_is_unsupported_exit_two() {
    let dir = write_prop(
        r#"
@property symbolic_tensor forall(x: tensor[n, f32]):
  true
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .stdout(predicate::str::contains(
            "symbolic tensor dimensions are not supported in L2 v1",
        ));
}

#[test]
fn duplicate_property_name_is_parse_error() {
    let dir = write_prop(
        r#"
@property repeated forall(x: f32):
  true
@property repeated forall(y: f32):
  true
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("unique property name"));
}

#[test]
fn cli_samples_and_seed_override_source_options() {
    let dir = write_prop(
        r#"
@property source_options forall(x: f32):
  x == x
  with samples = 5
  with seed = 7
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "2",
            "--seed",
            "42",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["samples"], 2);
    assert_eq!(records[0]["seed"], 42);
}

#[test]
fn f64_tensor_binder_is_supported() {
    let dir = write_prop(
        r#"
@property f64_tensor forall(x: tensor[2, 3, f64]):
  {
    _ = drop(x)
    true
  }
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "1",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: f64_tensor -- 1/1 passed",
        ));
}

#[test]
fn tensor_binder_can_be_used_by_shape_queries() {
    let dir = write_prop(
        r#"
@property tensor_shape forall(x: tensor[3, f32]):
  {
    n = shape(x, 0)
    _ = drop(x)
    n == 3
  }
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "1",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: tensor_shape -- 1/1 passed",
        ));
}

#[test]
fn desugared_deep_property_uses_source_samples_and_seed() {
    let dir = write_prop(
        r#"
@property source_options forall(x: f32):
  x == x
  with samples = 2
  with seed = 7
"#,
    );
    let deep_path = dir.path().join("prop.dp");
    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", dir.path().join("prop.ch").to_str().unwrap()])
        .output()
        .expect("desugar");
    assert!(
        deep.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&deep.stderr)
    );
    std::fs::write(&deep_path, deep.stdout).expect("write deep");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", deep_path.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["samples"], 2);
    assert_eq!(records[0]["seed"], 7);
}

#[test]
fn deep_property_uses_metadata_quantifiers_and_cli_samples() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prop.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {} (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))}
  deep_non_negative
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0))))
"#,
    )
    .expect("write deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--samples",
            "2",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["name"], "deep_non_negative");
    assert_eq!(records[0]["samples"], 2);
}

#[test]
fn deep_property_quantifiers_must_match_fn_params() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mismatch.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {})}
  mismatch
  (fn {}
    (params {} (y {type: (t-prim {} f32)}))
    (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write deep");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "property_quantifiers must match fn parameters",
        ));
}

#[test]
fn deep_property_source_kind_must_be_valid_and_drives_json_source_kind() {
    let dir = tempdir().expect("tempdir");
    let invalid = dir.path().join("invalid.dp");
    std::fs::write(
        &invalid,
        r#"
(def {chelis_role: "property",
      property_source_kind: "bogus",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  invalid_kind
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write invalid deep");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", invalid.to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("property_source_kind"));

    let user = dir.path().join("user.dp");
    std::fs::write(
        &user,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  user_kind
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write user deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", user.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["source"]["kind"], "user");
}

#[test]
fn spans_override_requires_single_explicit_deep_file() {
    let dir = tempfile::Builder::new()
        .prefix("chelis-prove-spans")
        .tempdir()
        .expect("tempdir");
    for name in ["one.dp", "two.dp"] {
        std::fs::write(
            dir.path().join(name),
            r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  trivial
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
        )
        .expect("write deep");
    }
    let spans = dir.path().join("override.spans.json");
    std::fs::write(&spans, "{}").expect("write spans");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
        ])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "--spans can only be used with a single explicit .dp input",
        ));
}

#[test]
fn bridge_failure_resolves_spans_manifest_to_ears_source() {
    let dir = tempdir().expect("tempdir");
    let deep = dir.path().join("options.dp");
    let spans = dir.path().join("options.spans.json");
    std::fs::write(
        &deep,
        r#"
(def {chelis_role: "property",
      property_source_kind: "bridge:c-earchin",
      property_source_id: "FIN-003",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  req_FIN_003
  (fn {} (params {}) (lit {type: (t-prim {} bool)} false)))
"#,
    )
    .expect("write deep");
    std::fs::write(
        &spans,
        r#"
{
  "source": "references/finance_options/options_rules.ears",
  "source_hash": "sha256:test",
  "spans": [
    {
      "deep_node_id": "req_FIN_003",
      "deep_path": "module.def[2]",
      "ears_id": "FIN-003",
      "ears_file": "references/finance_options/options_rules.ears",
      "ears_text": "WHILE the exchange is open, the portfolio delta shall be at most the limit.",
      "ears": {
        "start_byte": 143,
        "end_byte": 224,
        "start_line": 3,
        "start_column": 1,
        "end_line": 3,
        "end_column": 81
      },
      "clauses": []
    }
  ]
}
"#,
    )
    .expect("write spans");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            deep.to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains(
            "references/finance_options/options_rules.ears:3:1 FIN-003",
        ))
        .stdout(predicate::str::contains(
            "WHILE the exchange is open, the portfolio delta shall be at most the limit.",
        ));

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            deep.to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("run prove");
    assert_eq!(output.status.code(), Some(1));
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["source"]["requirement"]["id"], "FIN-003");
    assert_eq!(records[0]["source"]["requirement"]["line"], 3);
    assert_eq!(
        records[0]["source"]["requirement"]["text"],
        "WHILE the exchange is open, the portfolio delta shall be at most the limit."
    );
}

#[test]
fn prove_accepts_dotted_deep_symbols_for_bridge_references() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bridge.dp");
    std::fs::write(
        &path,
        r#"
(module {}
  cearchin.generated.vocabularymiss
  (def {span: "req:MISS-001", c_earchin_role: "property_witness"}
    req_MISS_001
    (fn {} (params {}) (lit {type: (t-prim {} bool)} true))))
"#,
    )
    .expect("write deep");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\":\"req_MISS_001\""));
}
