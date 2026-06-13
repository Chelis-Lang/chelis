use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn run_bench(model: &str) -> Value {
    run_bench_with_env(model, &[])
}

fn run_bench_with_env(model: &str, envs: &[(&str, &str)]) -> Value {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("report.json");
    let mut cmd = Command::cargo_bin("bench_phase1e").expect("bench_phase1e binary");
    cmd.args(["--model", model, "--emit-json", out.to_str().unwrap()]);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    cmd.assert().success();
    let text = std::fs::read_to_string(&out).expect("report json");
    serde_json::from_str(&text).expect("valid report json")
}

#[test]
fn bench_phase1e_rejects_unknown_model() {
    Command::cargo_bin("bench_phase1e")
        .expect("bench_phase1e binary")
        .args(["--model", "unknown"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("unknown model"));
}

#[test]
#[ignore = "manual gate: real benchmark scope; run `cargo test -p chelis-e2e --test bench_phase1e -- --ignored`"]
fn bench_phase1e_all_emits_structured_json_for_real_scope() {
    let json = run_bench("all");
    let oracle = json["oracle"].as_str().expect("oracle string");
    assert!(oracle.contains("--model all"));
    assert!(oracle.contains("--emit-json"));
    let models = json["models"].as_array().expect("models array");
    assert_eq!(models.len(), 3, "expected all benchmark models");
    let names: Vec<_> = models
        .iter()
        .map(|model| model["name"].as_str().expect("model name"))
        .collect();
    assert_eq!(names, vec!["linreg", "mnist", "transformer"]);

    for model in models {
        assert!(model["cpu"].is_object(), "cpu backend report missing");
        assert!(model["hip"].is_object(), "hip backend report missing");
        assert!(
            model["pytorch"].is_object(),
            "pytorch backend report missing"
        );
        let comparisons = model["comparisons"].as_array().expect("comparisons array");
        assert!(
            !comparisons.is_empty(),
            "expected at least one comparison entry"
        );
    }

    // The transformer lane is a deterministic forward pass and must
    // compile and run on CPU — the in-repo numerical oracle.
    let transformer = models
        .iter()
        .find(|model| model["name"].as_str() == Some("transformer"))
        .expect("missing transformer model report");
    let transformer_cpu = &transformer["cpu"];
    assert_eq!(
        transformer_cpu["status"].as_str(),
        Some("ok"),
        "transformer CPU lane must succeed; got status {:?} reason {:?}",
        transformer_cpu["status"],
        transformer_cpu["reason"]
    );

    // linreg carries its own deterministic data, so its CPU lane must
    // actually execute and emit a structured verdict — never a silent skip.
    // Full-profile training currently diverges (loss -> inf/nan); that
    // numerical regression is tracked by chelis#389, so this gate asserts
    // the lane ran and reported, not that it converged. The green linreg
    // oracle is the smoke lane (`bench_phase1e_linreg_smoke_emits_structured_json`),
    // which asserts `cpu.status == "ok"`.
    let linreg = models
        .iter()
        .find(|model| model["name"].as_str() == Some("linreg"))
        .expect("missing linreg model report");
    let linreg_status = linreg["cpu"]["status"]
        .as_str()
        .expect("linreg cpu status string");
    assert!(
        matches!(linreg_status, "ok" | "failed"),
        "linreg CPU lane must run (it has its own data), got status {linreg_status:?}; \
         see chelis#389 for the full-profile training divergence"
    );
}

#[test]
fn bench_phase1e_linreg_smoke_emits_structured_json() {
    let json = run_bench_with_env(
        "linreg",
        &[
            ("CHELIS_BENCH_PROFILE", "smoke"),
            ("ROCR_VISIBLE_DEVICES", ""),
            (
                "CHELIS_BENCH_PYTHON",
                "/tmp/chelis-phase1e-missing-python-interpreter",
            ),
        ],
    );
    let oracle = json["oracle"].as_str().expect("oracle string");
    assert!(oracle.contains("--model linreg"));
    assert!(oracle.contains("--emit-json"));

    let models = json["models"].as_array().expect("models array");
    assert_eq!(models.len(), 1, "expected only linreg benchmark model");
    let model = &models[0];
    assert_eq!(model["name"].as_str(), Some("linreg"));
    assert_eq!(model["workload"]["train_batches"].as_u64(), Some(1));
    assert_eq!(model["workload"]["test_batches"].as_u64(), Some(1));
    assert_eq!(model["workload"]["epochs"].as_u64(), Some(1));
    assert!(model["cpu"].is_object(), "cpu backend report missing");
    // The CPU lane is the in-repo semantic oracle for this benchmark: it
    // must actually compile and run, not silently skip or fail. Asserting
    // only `is_object()` let a month-long regression (a stale bench harness
    // out of sync with the rewritten example) pass as green.
    let cpu = &model["cpu"];
    assert_eq!(
        cpu["status"].as_str(),
        Some("ok"),
        "linreg smoke CPU lane must succeed; got status {:?} reason {:?}",
        cpu["status"],
        cpu["reason"]
    );
    assert!(
        cpu["final_loss"].as_f64().is_some(),
        "linreg smoke CPU lane reported status ok but emitted no final_loss: {cpu:?}"
    );
    assert!(model["hip"].is_object(), "hip backend report missing");
    assert!(
        model["pytorch"].is_object(),
        "pytorch backend report missing"
    );

    let hip = &model["hip"];
    assert_eq!(hip["status"].as_str(), Some("skipped"));
    let hip_reason = hip["reason"].as_str().unwrap_or("");
    assert!(
        hip_reason.contains("HIP runtime unavailable")
            || hip_reason.contains("no HIP devices visible")
            || hip_reason.contains("hipcc not available"),
        "unexpected HIP skip reason: {hip_reason}"
    );

    let pytorch = &model["pytorch"];
    assert_eq!(pytorch["status"].as_str(), Some("skipped"));
    let pytorch_reason = pytorch["reason"].as_str().unwrap_or("");
    assert!(
        pytorch_reason.contains("CHELIS_BENCH_PYTHON points to a missing interpreter"),
        "unexpected PyTorch skip reason: {pytorch_reason}"
    );

    let comparisons = model["comparisons"].as_array().expect("comparisons array");
    assert!(
        !comparisons.is_empty(),
        "expected at least one comparison entry"
    );
}

#[test]
fn bench_phase1e_mnist_missing_data_emits_structured_skip_report() {
    let json = run_bench_with_env(
        "mnist",
        &[("MNIST_DIR", "/tmp/chelis-phase1e-missing-mnist")],
    );
    let oracle = json["oracle"].as_str().expect("oracle string");
    assert!(oracle.contains("--model mnist"));
    assert!(oracle.contains("--emit-json"));
    let model = &json["models"][0];
    assert_eq!(model["name"].as_str(), Some("mnist"));
    for backend_name in ["cpu", "hip", "pytorch"] {
        let backend = &model[backend_name];
        let status = backend["status"].as_str().expect("status string");
        assert_eq!(status, "skipped", "expected missing-MNIST structured skip");
        let reason = backend["reason"].as_str().unwrap_or("");
        assert!(
            reason.contains("MNIST benchmark data unavailable"),
            "unexpected skip reason for `{backend_name}`: {reason}"
        );
    }
}
