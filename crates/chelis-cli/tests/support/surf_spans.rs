use assert_cmd::Command;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

fn hash(text: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(text.as_bytes()))
}

fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let source = "# Requirements\r\nBOUND-001  The engine shall be at most 1.0.\r\n";
    let artifact = "def actual(x: f64) -> f64 = x + 2.0f64\n@property prop_BOUND_001 forall(x: f64) where x >= 0.0f64:\n  actual(x) <= 1.0f64\n";
    std::fs::write(dir.path().join("requirements.ears"), source).unwrap();
    std::fs::write(dir.path().join("properties.ch"), artifact).unwrap();
    let start = source.find("BOUND-001").unwrap();
    let end = source.trim_end().len();
    let manifest = json!({
        "schema_version": 1,
        "artifact": {"path":"properties.ch", "sha256":hash(artifact)},
        "sources":[{"path":"requirements.ears", "sha256":hash(source)}],
        "requirements":[{"id":"BOUND-001", "source":"requirements.ears",
            "span":{"start_byte":start,"end_byte":end},
            "clauses":[{"kind":"response","span":{"start_byte":source.find("be at most").unwrap(),"end_byte":end}}]}],
        "properties":[{"name":"prop_BOUND_001","requirement_id":"BOUND-001"}]
    });
    write_manifest(dir.path(), &manifest);
    dir
}

fn write_manifest(dir: &Path, manifest: &Value) {
    std::fs::write(
        dir.join("properties.spans.json"),
        serde_json::to_string_pretty(manifest).unwrap(),
    )
    .unwrap();
}

pub(super) fn verify_legacy_fixture(path: &Path) {
    let mut value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let entry = &value["spans"][0];
    let text = entry["ears_text"].as_str().unwrap().to_owned();
    let line = entry["ears"]["start_line"].as_u64().unwrap() as usize;
    let before = "\n".repeat(line - 1);
    let source = before.clone() + &text;
    let file = path.parent().unwrap().join("options_rules.ears");
    std::fs::write(&file, &source).unwrap();
    value["source"] = json!(file);
    value["source_hash"] = json!(hash(&source));
    value["spans"][0]["ears_file"] = json!(file);
    value["spans"][0]["ears"]["start_byte"] = json!(before.len());
    value["spans"][0]["ears"]["end_byte"] = json!(source.len());
    std::fs::write(path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .arg("prove")
        .arg(dir.join("properties.ch"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn surf_spans_resolve_original_requirement_in_json_and_text() {
    let dir = fixture();
    let spans = dir.path().join("properties.spans.json");
    let out = run(dir.path(), &["--json", "--spans", spans.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let record: Value = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|v| v["kind"] == "property")
        .unwrap();
    assert_eq!(record["source"]["requirement"]["id"], "BOUND-001");
    assert_eq!(record["source"]["requirement"]["line"], 2);
    assert_eq!(record["source"]["requirement"]["column"], 1);
    assert_eq!(
        record["source"]["requirement"]["text"],
        "BOUND-001  The engine shall be at most 1.0."
    );
    assert!(record["counterexample"].is_object());
    assert_eq!(record["status"], "failed");
    let out = run(dir.path(), &[]);
    assert_eq!(out.status.code(), Some(1));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("requirements.ears:2:1 BOUND-001"), "{text}");
}

#[test]
fn surf_spans_reject_tampering_before_any_property_verdict() {
    for case in [
        "source",
        "artifact",
        "unknown",
        "duplicate",
        "range",
        "version",
        "requirement",
        "clause",
        "missing",
        "source_alias",
    ] {
        let dir = fixture();
        let path = dir.path().join("properties.spans.json");
        let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        match case {
            "source" => std::fs::write(dir.path().join("requirements.ears"), "changed").unwrap(),
            "artifact" => {
                let p = dir.path().join("properties.ch");
                std::fs::write(&p, std::fs::read_to_string(&p).unwrap() + "\n").unwrap();
            }
            "unknown" => manifest["properties"][0]["name"] = json!("missing"),
            "duplicate" => {
                let entry = manifest["properties"][0].clone();
                manifest["properties"].as_array_mut().unwrap().push(entry);
            }
            "range" => manifest["requirements"][0]["span"]["end_byte"] = json!(usize::MAX),
            "version" => manifest["schema_version"] = json!(99),
            "requirement" => manifest["properties"][0]["requirement_id"] = json!("missing"),
            "clause" => manifest["requirements"][0]["clauses"][0]["span"]["start_byte"] = json!(0),
            "missing" => manifest["properties"] = json!([]),
            "source_alias" => {
                let mut source = manifest["sources"][0].clone();
                source["path"] = json!("./requirements.ears");
                manifest["sources"].as_array_mut().unwrap().push(source);
            }
            _ => unreachable!(),
        }
        write_manifest(dir.path(), &manifest);
        let out = run(dir.path(), &["--json"]);
        assert_eq!(out.status.code(), Some(3), "case={case}: {out:?}");
        assert!(
            !String::from_utf8_lossy(&out.stdout).contains("\"kind\":\"property\""),
            "case={case}: {out:?}"
        );
    }
}

#[test]
fn surf_spans_filter_does_not_require_removing_unselected_mappings() {
    let dir = fixture();
    let out = run(dir.path(), &["--only", "prop_BOUND_001", "--json"]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("BOUND-001"));
}

#[cfg(feature = "chelis-prove")]
#[test]
fn surf_spans_follow_reef_linked_property_names() {
    let dir = fixture();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    let artifact = "module Demo.Properties\nimport Demo.Model (actual)\n@property prop_BOUND_001 forall(x: f64) where x >= 0.0f64:\n  actual(x) <= 1.0f64\n";
    std::fs::write(dir.path().join("src/properties.ch"), artifact).unwrap();
    std::fs::write(
        dir.path().join("src/model.ch"),
        "module Demo.Model\nexport (actual)\ndef actual(x: f64) -> f64 = x + 2.0f64\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("reef.toml"), format!("[package]\nname = \"spans-fixture\"\nversion = \"0.1.0\"\nmodule_prefix = \"Demo\"\ncompiler = \"={}\"\n", env!("CARGO_PKG_VERSION"))).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("properties.spans.json")).unwrap())
            .unwrap();
    manifest["artifact"] = json!({"path":"src/properties.ch","sha256":hash(artifact)});
    write_manifest(dir.path(), &manifest);
    let out = Command::cargo_bin("chelis")
        .unwrap()
        .arg("prove")
        .arg(dir.path().join("src/properties.ch"))
        .arg("--spans")
        .arg(dir.path().join("properties.spans.json"))
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("\"id\":\"BOUND-001\""), "{text}");
    assert!(!text.contains("\"name\":\"pkg__"), "{text}");
}

#[test]
fn surf_spans_identical_property_passes_conforming_and_fails_mutant() {
    let dir = fixture();
    let path = dir.path().join("properties.ch");
    let mutant = std::fs::read_to_string(&path).unwrap();
    let conforming = mutant.replace("x + 2.0f64", "0.0f64");
    let property = mutant.split_once("@property").unwrap().1;
    assert_eq!(property, conforming.split_once("@property").unwrap().1);
    std::fs::write(&path, &conforming).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("properties.spans.json")).unwrap())
            .unwrap();
    manifest["artifact"]["sha256"] = json!(hash(&conforming));
    write_manifest(dir.path(), &manifest);
    let out = run(dir.path(), &["--json"]);
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("\"id\":\"BOUND-001\""), "{text}");
    assert!(text.contains("\"status\":\"passed\""), "{text}");
    std::fs::write(&path, &mutant).unwrap();
    manifest["artifact"]["sha256"] = json!(hash(&mutant));
    write_manifest(dir.path(), &manifest);
    assert_eq!(run(dir.path(), &["--json"]).status.code(), Some(1));
}

#[test]
fn surf_spans_validate_utf8_boundaries_and_derive_byte_columns() {
    let dir = fixture();
    let source = "前言\né BOUND-001 The engine shall be at most 1.0.\n";
    let path = dir.path().join("properties.spans.json");
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    std::fs::write(dir.path().join("requirements.ears"), source).unwrap();
    manifest["sources"][0]["sha256"] = json!(hash(source));
    let start = source.find("BOUND-001").unwrap();
    let end = source.trim_end().len();
    manifest["requirements"][0]["span"] = json!({"start_byte":start,"end_byte":end});
    manifest["requirements"][0]["clauses"] = json!([]);
    write_manifest(dir.path(), &manifest);
    let out = run(dir.path(), &["--json"]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let record: Value = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|v| v["kind"] == "property")
        .unwrap();
    assert_eq!(record["source"]["requirement"]["line"], 2);
    assert_eq!(record["source"]["requirement"]["column"], 4);
    manifest["requirements"][0]["span"]["start_byte"] = json!(1);
    write_manifest(dir.path(), &manifest);
    let out = run(dir.path(), &["--json"]);
    assert_eq!(out.status.code(), Some(3), "{out:?}");
    assert!(out.stdout.is_empty(), "{out:?}");
}

#[test]
fn surf_spans_directory_preflight_rejects_bad_manifest_before_green_record() {
    let dir = fixture();
    std::fs::write(
        dir.path().join("aaa.ch"),
        "@property green forall(x: f64):\n  x == x\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("requirements.ears"), "stale source").unwrap();
    let out = Command::cargo_bin("chelis")
        .unwrap()
        .arg("prove")
        .arg(dir.path())
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3), "{out:?}");
    assert!(out.stdout.is_empty(), "{out:?}");
}
