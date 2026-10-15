//! Shared Surf/Deep acceptance run by the required prove_deep_obligations SMT target.
use assert_cmd::Command;
use serde_json::Value;

fn prove(source: &str, deep: bool) -> (i32, Vec<Value>) {
    let directory = tempfile::tempdir().unwrap();
    let surf = directory.path().join("numeric.ch");
    std::fs::write(&surf, source).unwrap();
    let formatted = Command::cargo_bin("chelis")
        .unwrap()
        .args(["fmt", "--inplace"])
        .arg(&surf)
        .output()
        .unwrap();
    assert!(formatted.status.success(), "{formatted:?}");
    let input = if deep {
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .arg("deep")
            .arg(&surf)
            .env_remove("CHELIS_STYLE_GATE_DISABLE")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let path = directory.path().join("numeric.dp");
        std::fs::write(&path, output.stdout).unwrap();
        path
    } else {
        surf
    };
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .arg("prove")
        .arg(input)
        .args(["--json", "--tier", "smt-only", "--seed", "37"])
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .output()
        .unwrap();
    let records: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(
        !records.is_empty(),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.code().unwrap(), records)
}
fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .find(|r| r["kind"] == "property" && r["name"] == name)
        .unwrap_or_else(|| panic!("missing {name}: {records:?}"))
}
fn exact_integer(value: &Value) -> i64 {
    if let Some(value) = value.as_i64() {
        return value;
    }
    let text = value.as_str().unwrap();
    if let Some(magnitude) = text.strip_prefix("(- ").and_then(|s| s.strip_suffix(')')) {
        i64::try_from(-magnitude.trim().parse::<i128>().unwrap()).unwrap()
    } else {
        text.parse().unwrap()
    }
}

#[test]
fn typed_integer_domains_cast_minima_and_implementation_mutants_match_on_surf_and_deep() {
    for (dtype, minimum, maximum) in [
        ("i8", i8::MIN as i64, i8::MAX as i64),
        ("i16", i16::MIN as i64, i16::MAX as i64),
        ("i32", i32::MIN as i64, i32::MAX as i64),
        ("i64", i64::MIN, i64::MAX),
    ] {
        let source = format!(
            "def value(x: {dtype}) -> {dtype} = x\ndef minimum(dummy: {dtype}) -> {dtype} = cast({minimum}, {dtype})\n@property lower forall(x: {dtype}):\n  value(x) >= minimum(x)\n@property upper forall(x: {dtype}):\n  value(x) <= cast({maximum}, {dtype})\n@property at_minimum forall(x: {dtype}) where x == cast({minimum}, {dtype}):\n  value(x) == x\n@property strict_minimum forall(x: {dtype}):\n  value(x) > cast({minimum}, {dtype})\n"
        );
        for deep in [false, true] {
            let (code, records) = prove(&source, deep);
            assert_eq!(code, 1, "{records:?}"); // only the strict bound is false
            for name in ["lower", "upper", "at_minimum"] {
                let record = property(&records, name);
                assert_eq!(record["status"], "passed", "{record}");
                assert_eq!(record["proof_tier"], "smt", "{record}");
                assert_eq!(record["qualifiers"], serde_json::json!(["real_arithmetic"]));
            }
            assert_eq!(
                property(&records, "lower")["assumptions"],
                serde_json::json!([])
            );
            let guarded = property(&records, "at_minimum");
            assert_eq!(guarded["assumptions"].as_array().unwrap().len(), 1);
            assert_eq!(
                guarded["assumptions"][0]["name"],
                "preconditions:at_minimum"
            );
            assert_eq!(
                guarded["assumptions"][0]["non_vacuity"]["evidence"]["assumption_count"],
                1
            );
            assert_eq!(
                guarded["assumptions"][0]["non_vacuity"]["evidence"]["intrinsic_domain_count"],
                2
            );
            let strict = property(&records, "strict_minimum");
            assert_eq!(strict["status"], "failed", "{strict}");
            assert_eq!(exact_integer(&strict["counterexample"]["x"]), minimum);
            let mutant = source.replace("= x\n", &format!("= x + cast(1, {dtype})\n"));
            let (_, records) = prove(&mutant, deep);
            let failure = property(&records, "at_minimum");
            assert_eq!(failure["status"], "failed", "{failure}");
            assert_eq!(failure["proof_tier"], "smt");
            assert_eq!(exact_integer(&failure["counterexample"]["x"]), minimum);
        }
    }
}

#[test]
fn nonminimum_casts_float_rounding_sentinel_and_nested_substitution_keep_values() {
    let source = "def inner(dummy: i32) -> i32 = cast(-3, i32)\ndef outer(x: i32) -> i32 = inner(x)\n@property nearby forall(x: i32) where x == cast(-3, i32):\n  outer(x) == x\n@property rounding forall(x: f32):\n  cast(16777217, f32) == 16777216f32\n@property rounding_false forall(x: f32):\n  cast(16777217, f32) != 16777216f32\n@property decimal forall(x: f64):\n  cast(1.1, f64) == 1.1f64\n@property sentinel forall(x: i64):\n  -9223372036854775808i64 < 0i64\n@property sentinel_false forall(x: i64):\n  -9223372036854775808i64 > 0i64\n";
    for deep in [false, true] {
        let (code, records) = prove(source, deep);
        assert_eq!(code, 1, "{records:?}");
        for name in ["nearby", "rounding", "decimal", "sentinel"] {
            assert_eq!(property(&records, name)["status"], "passed", "{records:?}");
            assert_eq!(property(&records, name)["proof_tier"], "smt");
        }
        for name in ["rounding_false", "sentinel_false"] {
            assert_eq!(property(&records, name)["status"], "failed", "{records:?}");
        }
        let mutant = source
            .replace("cast(-3, i32)", "cast(-2, i32)")
            .replace("where x == cast(-2, i32)", "where x == cast(-3, i32)");
        let (_, records) = prove(&mutant, deep);
        assert_eq!(
            property(&records, "nearby")["status"],
            "failed",
            "{records:?}"
        );
    }
}

#[test]
fn typed_domains_reject_vacuous_guards_without_hiding_unbounded_intermediate_arithmetic() {
    let source = "@property impossible forall(x: i64) where x < cast(-9223372036854775808, i64):\n  true\n@property arithmetic forall(x: i8):\n  x + 1i8 > x\n@property computed_argument forall(x: i8):\n  identity(x + 1i8) > x\ndef identity(x: i8) -> i8 = x\n";
    for deep in [false, true] {
        let (code, records) = prove(source, deep);
        assert_ne!(code, 0);
        let invalid = property(&records, "impossible");
        assert_eq!(invalid["status"], "unsupported", "{invalid}");
        assert!(
            invalid["reason"]
                .as_str()
                .unwrap()
                .contains("non_vacuity_invalid")
        );
        for name in ["arithmetic", "computed_argument"] {
            let record = property(&records, name);
            assert_eq!(record["status"], "passed", "{record}");
            assert_eq!(record["composite_verdict"], "proven_modulo_real_arithmetic");
        }
    }
}
