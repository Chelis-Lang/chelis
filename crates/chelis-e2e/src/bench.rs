use chelis_unord::UnordMap;
use chelis_unord::UnordSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use chelis_backend_c::CodegenResult as CCodegenResult;
use chelis_backend_hip::HipCodegenResult;
use chelis_ir::dag::{Dag, NodeId, RiscOp};
use chelis_ir::eval::TensorValue;
use chelis_ir::{fuse, grad_then_fuse};
use serde::Serialize;

use crate::data::{default_mnist_dir, load_mnist};
use crate::pipeline::compile_surf;
use crate::train::{build_mnist_program, init_params};

const LINREG_TRAIN_BATCHES: usize = 16;
const LINREG_TEST_BATCHES: usize = 4;
const LINREG_BATCH_SIZE: usize = 64;
const LINREG_FEATURES: usize = 64;
const LINREG_EPOCHS: usize = 12;
const LINREG_LR: f32 = 0.02;

const MNIST_TRAIN_BATCHES: usize = 32;
const MNIST_TEST_BATCHES: usize = 8;
const MNIST_BATCH_SIZE: usize = 32;
const MNIST_EPOCHS: usize = 5;
const MNIST_LR: f32 = 0.1;
const MNIST_SEED: u64 = 42;

const TRANSFORMER_SEQ_LEN: usize = 128;
const TRANSFORMER_D_MODEL: usize = 256;
const TRANSFORMER_HEADS: usize = 1;
const TRANSFORMER_HEAD_DIM: usize = 64;
const TRANSFORMER_D_FF: usize = 1024;
const TRANSFORMER_ITERS: usize = 5;
const FORWARD_TOL: f32 = 1e-4;
const PYTORCH_SETUP_HINT: &str = "run `uv venv --python 3.12 py/.venv` and `uv pip install --python py/.venv/bin/python --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --prerelease allow torch torchaudio torchvision`";
const BENCH_PROFILE_ENV: &str = "CHELIS_BENCH_PROFILE";

#[derive(Clone, Copy)]
struct LinregWorkload {
    train_batches: usize,
    test_batches: usize,
    batch_size: usize,
    features: usize,
    epochs: usize,
    lr: f32,
}

impl LinregWorkload {
    fn current() -> Self {
        Self::for_profile(env::var(BENCH_PROFILE_ENV).ok().as_deref())
    }

    fn for_profile(profile: Option<&str>) -> Self {
        match profile {
            // `examples/linreg.ch` fixes the batch dimension at
            // `tensor[64, 64]`, so the compiled program demands
            // batch_size == LINREG_BATCH_SIZE. The smoke profile takes its
            // speedup from a single train/test batch and one epoch, not
            // from shrinking that fixed architecture dimension.
            Some("smoke") => Self {
                train_batches: 1,
                test_batches: 1,
                batch_size: LINREG_BATCH_SIZE,
                features: LINREG_FEATURES,
                epochs: 1,
                lr: LINREG_LR,
            },
            _ => Self {
                train_batches: LINREG_TRAIN_BATCHES,
                test_batches: LINREG_TEST_BATCHES,
                batch_size: LINREG_BATCH_SIZE,
                features: LINREG_FEATURES,
                epochs: LINREG_EPOCHS,
                lr: LINREG_LR,
            },
        }
    }
}

#[derive(Clone, Copy)]
pub enum Model {
    Linreg,
    Mnist,
    Transformer,
}

impl Model {
    pub fn parse(arg: &str) -> Result<Vec<Self>, String> {
        match arg {
            "linreg" => Ok(vec![Self::Linreg]),
            "mnist" => Ok(vec![Self::Mnist]),
            "transformer" => Ok(vec![Self::Transformer]),
            "all" => Ok(vec![Self::Linreg, Self::Mnist, Self::Transformer]),
            other => Err(format!(
                "unknown model `{other}`: expected linreg, mnist, transformer, or all"
            )),
        }
    }

    #[cfg(test)]
    fn name(self) -> &'static str {
        match self {
            Self::Linreg => "linreg",
            Self::Mnist => "mnist",
            Self::Transformer => "transformer",
        }
    }
}

#[derive(Serialize)]
pub struct BenchmarkReport {
    pub oracle: String,
    pub models: Vec<ModelReport>,
}

#[derive(Serialize)]
pub struct ModelReport {
    pub name: String,
    pub workload: WorkloadReport,
    pub cpu: BackendReport,
    pub hip: BackendReport,
    pub pytorch: BackendReport,
    pub comparisons: Vec<ComparisonReport>,
}

#[derive(Serialize)]
pub struct WorkloadReport {
    pub summary: String,
    pub batch_size: Option<usize>,
    pub train_batches: Option<usize>,
    pub test_batches: Option<usize>,
    pub epochs: Option<usize>,
    pub seq_len: Option<usize>,
    pub d_model: Option<usize>,
    pub n_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub d_ff: Option<usize>,
    pub iterations: Option<usize>,
}

#[derive(Serialize, Clone)]
pub struct BackendReport {
    pub status: &'static str,
    pub reason: Option<String>,
    pub compile_ms: Option<f64>,
    pub run_ms: Option<f64>,
    pub peak_device_bytes_formula: Option<String>,
    pub peak_device_bytes_estimate: Option<usize>,
    pub loss_history: Option<Vec<f32>>,
    pub final_loss: Option<f32>,
    pub final_accuracy: Option<f32>,
    pub output_len: Option<usize>,
    pub output_checksum: Option<f64>,
    pub output_sample: Option<Vec<f32>>,
}

#[derive(Serialize)]
pub struct ComparisonReport {
    pub name: String,
    pub status: &'static str,
    pub note: String,
    pub max_abs_diff: Option<f32>,
    pub mean_abs_diff: Option<f32>,
}

struct RunArtifacts {
    report: BackendReport,
    output: Vec<f32>,
}

#[derive(Clone, Copy)]
enum Backend {
    Cpu,
    Hip,
}

impl Backend {
    fn tool(self) -> String {
        match self {
            Self::Cpu => chelis_backend_c::toolchain::c_compiler(),
            Self::Hip => "hipcc".to_string(),
        }
    }
}

struct TrainingPrograms {
    train_c: CCodegenResult,
    train_hip: HipCodegenResult,
    train_labels: UnordMap<String, usize>,
}

struct ForwardPrograms {
    cpu: CCodegenResult,
    hip: HipCodegenResult,
    output_index: usize,
}

pub fn run_phase1e(models: &[Model], emit_json: Option<&Path>) -> Result<BenchmarkReport, String> {
    let mut reports = Vec::new();
    for model in models {
        reports.push(match model {
            Model::Linreg => run_linreg()?,
            Model::Mnist => run_mnist()?,
            Model::Transformer => run_transformer()?,
        });
    }
    Ok(BenchmarkReport {
        oracle: oracle_command(models, emit_json),
        models: reports,
    })
}

fn oracle_command(models: &[Model], emit_json: Option<&Path>) -> String {
    let model = match models {
        [Model::Linreg] => "linreg",
        [Model::Mnist] => "mnist",
        [Model::Transformer] => "transformer",
        [Model::Linreg, Model::Mnist, Model::Transformer] => "all",
        _ => "all",
    };
    let emit_json = emit_json
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "<stdout>".to_string());
    format!(
        "cargo run --release -p chelis-e2e --bin bench_phase1e -- --model {model} --emit-json {emit_json}"
    )
}

fn run_linreg() -> Result<ModelReport, String> {
    let workload = LinregWorkload::current();
    let temp = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
    let data_path = temp.path().join("linreg.bin");
    write_linreg_data(&data_path)?;
    let pytorch = run_pytorch("benchmarks/pytorch/linreg.py", &data_path, None, "linreg");
    let (cpu, hip) = match build_linreg_programs() {
        Ok(programs) => (
            run_training_backend(Backend::Cpu, "linreg_train", &data_path, &programs, false),
            run_training_backend(Backend::Hip, "linreg_train", &data_path, &programs, false),
        ),
        Err(reason) => {
            let cpu = skipped(format!(
                "linreg benchmark training DAG unavailable: {reason}"
            ));
            let hip = if let Some(hip_reason) = hip_prerequisite_skip_reason() {
                skipped(hip_reason)
            } else {
                skipped(format!(
                    "linreg benchmark training DAG unavailable: {reason}"
                ))
            };
            (cpu, hip)
        }
    };

    Ok(ModelReport {
        name: "linreg".to_string(),
        workload: WorkloadReport {
            summary: "Synthetic linear regression training".to_string(),
            batch_size: Some(workload.batch_size),
            train_batches: Some(workload.train_batches),
            test_batches: Some(workload.test_batches),
            epochs: Some(workload.epochs),
            seq_len: None,
            d_model: None,
            n_heads: None,
            head_dim: None,
            d_ff: None,
            iterations: None,
        },
        comparisons: build_training_comparisons(&cpu, &hip, &pytorch, false),
        cpu: cpu.report,
        hip: hip.report,
        pytorch: pytorch.report,
    })
}

fn run_mnist() -> Result<ModelReport, String> {
    let temp = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
    let data_path = temp.path().join("mnist.bin");
    if let Err(err) = write_mnist_data(&data_path) {
        return Ok(skipped_model_report(
            "mnist",
            mnist_workload(),
            format!("MNIST benchmark data unavailable: {err}"),
        ));
    }
    let pytorch = run_pytorch("benchmarks/pytorch/mnist.py", &data_path, None, "mnist");
    let (cpu, hip) = match build_mnist_programs() {
        Ok(programs) => (
            run_training_backend(Backend::Cpu, "mnist_train", &data_path, &programs, true),
            run_training_backend(Backend::Hip, "mnist_train", &data_path, &programs, true),
        ),
        Err(reason) => {
            let cpu = skipped(format!(
                "mnist benchmark training DAG unavailable: {reason}"
            ));
            let hip = if let Some(hip_reason) = hip_prerequisite_skip_reason() {
                skipped(hip_reason)
            } else {
                skipped(format!(
                    "mnist benchmark training DAG unavailable: {reason}"
                ))
            };
            (cpu, hip)
        }
    };

    Ok(ModelReport {
        name: "mnist".to_string(),
        workload: mnist_workload(),
        comparisons: build_training_comparisons(&cpu, &hip, &pytorch, true),
        cpu: cpu.report,
        hip: hip.report,
        pytorch: pytorch.report,
    })
}

fn run_transformer() -> Result<ModelReport, String> {
    let temp = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
    let data_path = temp.path().join("transformer.bin");
    write_transformer_data(&data_path)?;
    let programs = build_transformer_programs()?;

    let cpu = run_forward_backend(Backend::Cpu, "transformer_block", &data_path, &programs);
    let hip = run_forward_backend(Backend::Hip, "transformer_block", &data_path, &programs);
    let pytorch = run_pytorch(
        "benchmarks/pytorch/transformer_block.py",
        &data_path,
        Some("--iterations"),
        "transformer",
    );

    Ok(ModelReport {
        name: "transformer".to_string(),
        workload: WorkloadReport {
            summary: "Sequence-only transformer-block-style forward pass".to_string(),
            batch_size: None,
            train_batches: None,
            test_batches: None,
            epochs: None,
            seq_len: Some(TRANSFORMER_SEQ_LEN),
            d_model: Some(TRANSFORMER_D_MODEL),
            n_heads: Some(TRANSFORMER_HEADS),
            head_dim: Some(TRANSFORMER_HEAD_DIM),
            d_ff: Some(TRANSFORMER_D_FF),
            iterations: Some(TRANSFORMER_ITERS),
        },
        comparisons: build_forward_comparisons(&cpu, &hip, &pytorch),
        cpu: cpu.report,
        hip: hip.report,
        pytorch: pytorch.report,
    })
}

fn build_training_comparisons(
    cpu: &RunArtifacts,
    hip: &RunArtifacts,
    pytorch: &RunArtifacts,
    uses_accuracy: bool,
) -> Vec<ComparisonReport> {
    vec![
        compare_training("cpu_vs_hip", cpu, hip, uses_accuracy),
        compare_training("cpu_vs_pytorch", cpu, pytorch, uses_accuracy),
        compare_training("hip_vs_pytorch", hip, pytorch, uses_accuracy),
    ]
}

fn mnist_workload() -> WorkloadReport {
    WorkloadReport {
        summary: "MNIST MLP training on a fixed subset".to_string(),
        batch_size: Some(MNIST_BATCH_SIZE),
        train_batches: Some(MNIST_TRAIN_BATCHES),
        test_batches: Some(MNIST_TEST_BATCHES),
        epochs: Some(MNIST_EPOCHS),
        seq_len: None,
        d_model: None,
        n_heads: None,
        head_dim: None,
        d_ff: None,
        iterations: None,
    }
}

fn skipped_model_report(name: &str, workload: WorkloadReport, reason: String) -> ModelReport {
    let cpu = skipped(reason.clone());
    let hip = skipped(reason.clone());
    let pytorch = skipped(reason);
    ModelReport {
        name: name.to_string(),
        workload,
        comparisons: build_training_comparisons(&cpu, &hip, &pytorch, name == "mnist"),
        cpu: cpu.report,
        hip: hip.report,
        pytorch: pytorch.report,
    }
}

fn build_forward_comparisons(
    cpu: &RunArtifacts,
    hip: &RunArtifacts,
    pytorch: &RunArtifacts,
) -> Vec<ComparisonReport> {
    vec![
        compare_forward("cpu_vs_hip", cpu, hip),
        compare_forward("cpu_vs_pytorch", cpu, pytorch),
        compare_forward("hip_vs_pytorch", hip, pytorch),
    ]
}

fn compare_training(
    name: &str,
    left: &RunArtifacts,
    right: &RunArtifacts,
    uses_accuracy: bool,
) -> ComparisonReport {
    if left.report.status != "ok" || right.report.status != "ok" {
        return ComparisonReport {
            name: name.to_string(),
            status: "skipped",
            note: "one or both backends did not complete".to_string(),
            max_abs_diff: None,
            mean_abs_diff: None,
        };
    }

    let (max_abs_diff, mean_abs_diff) = diff_metrics(&left.output, &right.output);
    let left_loss = left
        .report
        .loss_history
        .as_ref()
        .map(|v| is_decreasing(v))
        .unwrap_or(false);
    let right_loss = right
        .report
        .loss_history
        .as_ref()
        .map(|v| is_decreasing(v))
        .unwrap_or(false);
    let mut ok = left_loss && right_loss;
    let mut note = "both runs decreased training loss".to_string();

    if uses_accuracy
        && let (Some(a), Some(b)) = (left.report.final_accuracy, right.report.final_accuracy)
    {
        let acc_gap = (a - b).abs();
        ok &= acc_gap <= 0.15;
        note = format!("loss trends decrease; final accuracy gap = {:.4}", acc_gap);
    }

    ComparisonReport {
        name: name.to_string(),
        status: if ok { "ok" } else { "warn" },
        note,
        max_abs_diff: Some(max_abs_diff),
        mean_abs_diff: Some(mean_abs_diff),
    }
}

fn compare_forward(name: &str, left: &RunArtifacts, right: &RunArtifacts) -> ComparisonReport {
    if left.report.status != "ok" || right.report.status != "ok" {
        return ComparisonReport {
            name: name.to_string(),
            status: "skipped",
            note: "one or both backends did not complete".to_string(),
            max_abs_diff: None,
            mean_abs_diff: None,
        };
    }
    let (max_abs_diff, mean_abs_diff) = diff_metrics(&left.output, &right.output);
    ComparisonReport {
        name: name.to_string(),
        status: if max_abs_diff <= FORWARD_TOL {
            "ok"
        } else {
            "warn"
        },
        note: format!("forward tolerance target {:.1e}", FORWARD_TOL),
        max_abs_diff: Some(max_abs_diff),
        mean_abs_diff: Some(mean_abs_diff),
    }
}

fn is_decreasing(losses: &[f32]) -> bool {
    match (losses.first(), losses.last()) {
        (Some(first), Some(last)) => last < first,
        _ => false,
    }
}

fn diff_metrics(left: &[f32], right: &[f32]) -> (f32, f32) {
    if left.len() != right.len() || left.is_empty() {
        return (f32::INFINITY, f32::INFINITY);
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (&a, &b) in left.iter().zip(right.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
    }
    (max_abs, sum_abs / left.len() as f32)
}

fn run_training_backend(
    backend: Backend,
    prefix: &str,
    data_path: &Path,
    programs: &TrainingPrograms,
    uses_accuracy: bool,
) -> RunArtifacts {
    let tool = backend.tool();
    if !tool_available(&tool, &["--version"]) {
        return skipped(format!("{tool} not available"));
    }

    let train = match backend {
        Backend::Cpu => &programs.train_c,
        Backend::Hip => {
            return run_training_backend_hip(prefix, data_path, programs, uses_accuracy);
        }
    };
    let main_source = build_training_main_c(
        prefix,
        data_path,
        train,
        &programs.train_labels,
        uses_accuracy,
    );
    match compile_and_run_c(
        prefix,
        &[("train_model.c", &train.c_source)],
        &main_source,
        train.requirements,
    ) {
        Ok((compile_ms, stdout)) => parse_run_output(stdout, compile_ms, None, None),
        Err(err) => failed(err),
    }
}

fn run_training_backend_hip(
    prefix: &str,
    data_path: &Path,
    programs: &TrainingPrograms,
    uses_accuracy: bool,
) -> RunArtifacts {
    if let Some(reason) = hip_prerequisite_skip_reason() {
        return skipped(reason);
    }
    let train = &programs.train_hip;
    let main_source = build_training_main_c(
        prefix,
        data_path,
        &c_shim(train),
        &programs.train_labels,
        uses_accuracy,
    );
    match compile_and_run_hip(
        prefix,
        &[("train_model.cpp", &train.c_source)],
        &main_source,
        &train.compile_flags,
        &train.link_flags,
    ) {
        Ok((compile_ms, stdout)) => parse_run_output(
            stdout,
            compile_ms,
            Some(train.peak_device_bytes_formula.clone()),
            train.peak_device_bytes_estimate,
        ),
        Err(err) => failed(err),
    }
}

fn run_forward_backend(
    backend: Backend,
    prefix: &str,
    data_path: &Path,
    programs: &ForwardPrograms,
) -> RunArtifacts {
    let tool = backend.tool();
    if !tool_available(&tool, &["--version"]) {
        return skipped(format!("{tool} not available"));
    }

    match backend {
        Backend::Cpu => {
            let main_source =
                build_forward_main_c(prefix, data_path, &programs.cpu, programs.output_index);
            match compile_and_run_c(
                prefix,
                &[("model.c", &programs.cpu.c_source)],
                &main_source,
                programs.cpu.requirements,
            ) {
                Ok((compile_ms, stdout)) => parse_run_output(stdout, compile_ms, None, None),
                Err(err) => failed(err),
            }
        }
        Backend::Hip => {
            if let Some(reason) = hip_prerequisite_skip_reason() {
                return skipped(reason);
            }
            let main_source = build_forward_main_c(
                prefix,
                data_path,
                &c_shim(&programs.hip),
                programs.output_index,
            );
            match compile_and_run_hip(
                prefix,
                &[("model.cpp", &programs.hip.c_source)],
                &main_source,
                &programs.hip.compile_flags,
                &programs.hip.link_flags,
            ) {
                Ok((compile_ms, stdout)) => parse_run_output(
                    stdout,
                    compile_ms,
                    Some(programs.hip.peak_device_bytes_formula.clone()),
                    programs.hip.peak_device_bytes_estimate,
                ),
                Err(err) => failed(err),
            }
        }
    }
}

fn run_pytorch(
    script: &str,
    data_path: &Path,
    extra_flag: Option<&str>,
    name: &str,
) -> RunArtifacts {
    let python = match resolve_pytorch_python() {
        Ok(path) => path,
        Err(reason) => return skipped(reason),
    };
    run_pytorch_with_python(&python, script, data_path, extra_flag, name)
}

fn run_pytorch_with_python(
    python: &Path,
    script: &str,
    data_path: &Path,
    extra_flag: Option<&str>,
    name: &str,
) -> RunArtifacts {
    let mut probe = Command::new(python);
    apply_pytorch_env(&mut probe);
    let torch_probe = probe.args(["-c", "import torch"]).output();
    match torch_probe {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            return skipped(format!(
                "PyTorch import failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Err(err) => return skipped(format!("failed to probe PyTorch: {err}")),
    }

    let mut cmd = Command::new(python);
    apply_pytorch_env(&mut cmd);
    cmd.arg(script).arg("--data-file").arg(data_path);
    if let Some(flag) = extra_flag {
        cmd.arg(flag).arg(TRANSFORMER_ITERS.to_string());
    }
    let start = Instant::now();
    let output = match cmd.output() {
        Ok(output) => output,
        Err(err) => return failed(format!("failed to run python for {name}: {err}")),
    };
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    if !output.status.success() {
        return failed(format!(
            "python benchmark failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_run_output(
        String::from_utf8_lossy(&output.stdout).into_owned(),
        compile_ms,
        None,
        None,
    )
}

fn apply_pytorch_env(cmd: &mut Command) {
    cmd.env_remove("HSA_OVERRIDE_GFX_VERSION");
    cmd.env("PYTORCH_ROCM_ARCH", "gfx1151");
    cmd.env("HSA_XNACK", "1");
    cmd.env("HSA_FORCE_FINE_GRAIN_PCIE", "1");
    cmd.env("GPU_MAX_HEAP_SIZE", "100");
    cmd.env("GPU_MAX_ALLOC_PERCENT", "100");

    let rocm_lib_dirs = [
        workspace_root().join("py/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"),
        workspace_root()
            .join("py/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/lib"),
    ];
    let existing = env::var_os("LD_LIBRARY_PATH");
    let mut paths: Vec<PathBuf> = rocm_lib_dirs.into_iter().filter(|p| p.is_dir()).collect();
    if let Some(existing) = existing {
        paths.extend(env::split_paths(&existing));
    }
    if let Ok(joined) = env::join_paths(paths) {
        cmd.env("LD_LIBRARY_PATH", joined);
    }
}

fn parse_run_output(
    stdout: String,
    compile_ms: f64,
    peak_device_bytes_formula: Option<String>,
    peak_device_bytes_estimate: Option<usize>,
) -> RunArtifacts {
    #[derive(serde::Deserialize)]
    struct RawRun {
        run_ms: f64,
        #[serde(default)]
        loss_history: Vec<f32>,
        #[serde(default)]
        final_loss: Option<f32>,
        #[serde(default)]
        final_accuracy: Option<f32>,
        #[serde(default)]
        output: Vec<f32>,
    }

    match serde_json::from_str::<RawRun>(stdout.trim()) {
        Ok(raw) => {
            let checksum: f64 = raw.output.iter().map(|v| *v as f64).sum();
            RunArtifacts {
                report: BackendReport {
                    status: "ok",
                    reason: None,
                    compile_ms: Some(compile_ms),
                    run_ms: Some(raw.run_ms),
                    peak_device_bytes_formula,
                    peak_device_bytes_estimate,
                    loss_history: if raw.loss_history.is_empty() {
                        None
                    } else {
                        Some(raw.loss_history.clone())
                    },
                    final_loss: raw.final_loss,
                    final_accuracy: raw.final_accuracy,
                    output_len: Some(raw.output.len()),
                    output_checksum: Some(checksum),
                    output_sample: Some(raw.output.iter().take(8).copied().collect()),
                },
                output: raw.output,
            }
        }
        Err(err) => failed(format!(
            "failed to parse benchmark output as JSON: {err}\n{stdout}"
        )),
    }
}

fn resolve_pytorch_python() -> Result<PathBuf, String> {
    resolve_pytorch_python_in(&workspace_root())
}

fn resolve_pytorch_python_in(workspace_root: &Path) -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("CHELIS_BENCH_PYTHON") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "CHELIS_BENCH_PYTHON points to a missing interpreter `{}`",
            path.display()
        ));
    }

    for candidate in benchmark_python_candidates(workspace_root) {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    Err(format!(
        "PyTorch benchmark environment not prepared; {PYTORCH_SETUP_HINT}"
    ))
}

fn benchmark_python_candidates(workspace_root: &Path) -> [PathBuf; 2] {
    [
        workspace_root.join("py/.venv/bin/python"),
        workspace_root.join("py/.venv/Scripts/python.exe"),
    ]
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn skipped(reason: String) -> RunArtifacts {
    RunArtifacts {
        report: BackendReport {
            status: "skipped",
            reason: Some(reason),
            compile_ms: None,
            run_ms: None,
            peak_device_bytes_formula: None,
            peak_device_bytes_estimate: None,
            loss_history: None,
            final_loss: None,
            final_accuracy: None,
            output_len: None,
            output_checksum: None,
            output_sample: None,
        },
        output: Vec::new(),
    }
}

fn failed(reason: String) -> RunArtifacts {
    RunArtifacts {
        report: BackendReport {
            status: "failed",
            reason: Some(reason),
            compile_ms: None,
            run_ms: None,
            peak_device_bytes_formula: None,
            peak_device_bytes_estimate: None,
            loss_history: None,
            final_loss: None,
            final_accuracy: None,
            output_len: None,
            output_checksum: None,
            output_sample: None,
        },
        output: Vec::new(),
    }
}

fn build_linreg_programs() -> Result<TrainingPrograms, String> {
    let src = include_str!("../../../examples/linreg.ch");
    let compiled = compile_surf(src)?;
    let loss = *require_named_root(&compiled.root_nodes, &["loss"])?;
    let pred = *require_named_root(&compiled.root_nodes, &["predict", "pred"])?;
    build_training_programs_from_compiled(compiled.dag, loss, pred, &["w", "b"])
}

fn build_mnist_programs() -> Result<TrainingPrograms, String> {
    let program = build_mnist_program()?;
    build_training_programs_from_compiled(
        program.dag,
        program.loss_node,
        program.logits_node,
        &["w1", "b1", "w2", "b2"],
    )
}

fn build_training_programs_from_compiled(
    mut forward_dag: Dag,
    loss_node: NodeId,
    infer_node: NodeId,
    param_names: &[&str],
) -> Result<TrainingPrograms, String> {
    // The compiled module lowers every `def` into one shared DAG, so a
    // parameter name like `w` can appear as a Load in more than one
    // function (e.g. both `predict` and `loss`). Gradients are taken with
    // respect to the loss function, so resolve each parameter to the Load
    // that is actually reachable from `loss_node`; the global-first match
    // would otherwise bind to `predict`'s loads and produce no gradient.
    let param_nodes = param_names
        .iter()
        .map(|name| {
            find_load_in_cone(&forward_dag, loss_node, name).ok_or_else(|| {
                format!("missing parameter load `{name}` in the loss function's dependency cone")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    add_named_store(&mut forward_dag, "eval_output", infer_node);
    let grad = grad_then_fuse(&forward_dag, loss_node, &param_nodes)
        .ok_or("failed to build gradient DAG".to_string())?;

    let mut train_dag = grad.dag.clone();
    add_named_store(&mut train_dag, "loss", grad.output_node);
    for (&name, &param_node) in param_names.iter().zip(param_nodes.iter()) {
        let grad_node = *grad
            .grad_nodes
            .get(&param_node)
            .ok_or_else(|| format!("missing gradient for `{name}`"))?;
        add_named_store(&mut train_dag, &format!("grad_{name}"), grad_node);
    }
    let train_dag = dag_without_roots(&train_dag);

    let c_selected = chelis_backend_c::prepare_dag_for_codegen(
        train_dag.clone(),
        chelis_backend_c::CodegenOptions::default(),
    );
    let c_verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(c_selected).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let hip_selected = chelis_backend_hip::prepare_dag_for_codegen(train_dag);
    let hip_verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(hip_selected).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let train_c =
        chelis_backend_c::codegen(c_verified, "chelis_train").map_err(|e| e.to_string())?;
    let train_hip =
        chelis_backend_hip::codegen_hip(hip_verified, "chelis_train").map_err(|e| e.to_string())?;

    let train_labels = output_index_map(&train_c.output_labels);
    if !train_labels.contains_key("eval_output") {
        return Err("missing eval_output store in training DAG".to_string());
    }

    Ok(TrainingPrograms {
        train_c,
        train_hip,
        train_labels,
    })
}

fn build_transformer_programs() -> Result<ForwardPrograms, String> {
    let src = include_str!("../../../examples/transformer_block.ch");
    let compiled = compile_surf(src)?;
    let out = *require_named_root(&compiled.root_nodes, &["forward", "out"])?;
    let mut dag = compiled.dag;
    add_named_store(&mut dag, "out", out);
    let fused = fuse::fuse(&dag);
    let fused = dag_without_roots(&fused);
    let c_selected = chelis_backend_c::prepare_dag_for_codegen(
        fused.clone(),
        chelis_backend_c::CodegenOptions::default(),
    );
    let c_verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(c_selected).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let hip_selected = chelis_backend_hip::prepare_dag_for_codegen(fused);
    let hip_verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(hip_selected).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let cpu = chelis_backend_c::codegen(c_verified, "chelis_forward").map_err(|e| e.to_string())?;
    let hip = chelis_backend_hip::codegen_hip(hip_verified, "chelis_forward")
        .map_err(|e| e.to_string())?;
    let output_index = *output_index_map(&cpu.output_labels)
        .get("out")
        .ok_or("missing `out` output label".to_string())?;
    Ok(ForwardPrograms {
        cpu,
        hip,
        output_index,
    })
}

/// Set of nodes reachable from `root` by walking input edges (the
/// dependency cone of `root`).
fn dependency_cone(dag: &Dag, root: NodeId) -> UnordSet<NodeId> {
    let mut seen = UnordSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(node) = dag.get(id) {
            stack.extend(node.inputs.iter().copied());
        }
    }
    seen
}

/// Find a Load named `name` that is reachable from `root`. Restricting
/// the search to the dependency cone of `root` disambiguates parameters
/// shared by several functions in the same lowered DAG (for example a
/// `w` Load present in both `predict` and `loss`).
fn find_load_in_cone(dag: &Dag, root: NodeId, name: &str) -> Option<NodeId> {
    let cone = dependency_cone(dag, root);
    dag.nodes().iter().find_map(|node| match &node.op {
        RiscOp::Load { name: load } if load == name && cone.contains(&node.id) => Some(node.id),
        _ => None,
    })
}

fn add_named_store(dag: &mut Dag, name: &str, input: NodeId) {
    let source = dag
        .get(input)
        .unwrap_or_else(|| panic!("missing node for store `{name}`"));
    let (decl, ty) = (source.owner, source.output_type.clone());
    dag.add_node(
        decl,
        RiscOp::Store { name: name.into() },
        vec![input],
        ty,
        None,
    );
}

/// An id-preserving copy, so every owner stands.
fn dag_without_roots(dag: &Dag) -> Dag {
    let mut out = Dag::new();
    out.inherit_declarations(dag);
    for node in dag.nodes() {
        out.add_node(
            node.owner,
            node.op.clone(),
            node.inputs.clone(),
            node.output_type.clone(),
            None,
        );
    }
    out
}

fn output_index_map(labels: &[String]) -> UnordMap<String, usize> {
    labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, label)| (label, idx))
        .collect()
}

fn require_named_root<'a>(
    roots: &'a UnordMap<String, NodeId>,
    candidates: &[&str],
) -> Result<&'a NodeId, String> {
    for candidate in candidates {
        if let Some(node) = roots.get(*candidate) {
            return Ok(node);
        }
    }
    Err(format!(
        "missing expected root; tried {}",
        candidates.join(", ")
    ))
}

fn c_shim(hip: &HipCodegenResult) -> CCodegenResult {
    CCodegenResult {
        c_source: hip.c_source.clone(),
        h_header: hip.h_header.clone(),
        requirements: chelis_backend_c::toolchain::CodegenRequirements::default(),
        input_labels: hip.input_labels.clone(),
        output_labels: hip.output_labels.clone(),
        symbolic_dims: hip.symbolic_dims.clone(),
    }
}

fn tool_available(tool: &str, args: &[&str]) -> bool {
    Command::new(tool)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn cpu_runtime_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

/// Resolve the workspace `target/` directory from the running binary rather
/// than a `CARGO_MANIFEST_DIR`-relative path, so an external
/// `CARGO_TARGET_DIR` (e.g. a concurrent agent building into
/// `target/agents/<name>`) is honored. `cpu_runtime_library` is only reached
/// from the `bench_phase1e` binary, which lives at `<target>/<profile>/<bin>`
/// (a test binary would instead be at `<target>/<profile>/deps/<bin>`); strip
/// a trailing `deps` component if present, then drop the profile component to
/// reach `<target>`. See chelis#747.
fn target_dir_from_current_exe() -> PathBuf {
    let exe = std::env::current_exe().expect("could not determine current executable");
    let mut profile_dir = exe
        .parent()
        .expect("executable should have a parent directory");
    if profile_dir.file_name().and_then(|name| name.to_str()) == Some("deps") {
        profile_dir = profile_dir
            .parent()
            .expect("`deps` directory should have a parent");
    }
    profile_dir
        .parent()
        .map(PathBuf::from)
        .expect("profile directory should have a parent target directory")
}

fn cpu_runtime_library() -> PathBuf {
    let target_dir = target_dir_from_current_exe();
    for dir in [
        target_dir.join("debug/deps"),
        target_dir.join("release/deps"),
    ] {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
                {
                    return path;
                }
            }
        }
    }
    panic!("could not locate libchelis_runtime.a for e2e benchmarks");
}

fn hip_runtime_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-backend-hip/runtime")
}

fn compile_and_run_c(
    prefix: &str,
    model_sources: &[(&str, &str)],
    main_source: &str,
    requirements: chelis_backend_c::toolchain::CodegenRequirements,
) -> Result<(f64, String), String> {
    let temp = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
    write_runtime_files(temp.path(), false)?;
    for (name, source) in model_sources {
        fs::write(temp.path().join(name), source)
            .map_err(|e| format!("write {name} failed: {e}"))?;
    }
    fs::write(temp.path().join("main.c"), main_source)
        .map_err(|e| format!("write main.c failed: {e}"))?;

    let bin = temp.path().join(prefix);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(requirements);
    let mut cmd = Command::new(&toolchain.compiler);
    cmd.arg("-O3");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(temp.path().join("main.c"));
    for (name, _) in model_sources {
        cmd.arg(temp.path().join(name));
    }
    cmd.arg("-L").arg(temp.path());
    cmd.arg("-lchelis_runtime");
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o").arg(&bin);

    let start = Instant::now();
    let output = cmd
        .output()
        .map_err(|e| format!("{} failed to start: {e}", toolchain.compiler))?;
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}",
            toolchain.compiler,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let run = Command::new(&bin)
        .output()
        .map_err(|e| format!("failed to run compiled benchmark: {e}"))?;
    if !run.status.success() {
        return Err(format!(
            "compiled benchmark failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    Ok((
        compile_ms,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    ))
}

fn compile_and_run_hip(
    prefix: &str,
    model_sources: &[(&str, &str)],
    main_source: &str,
    compile_flags: &[String],
    link_flags: &[String],
) -> Result<(f64, String), String> {
    let temp = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
    write_runtime_files(temp.path(), true)?;
    for (name, source) in model_sources {
        fs::write(temp.path().join(name), source)
            .map_err(|e| format!("write {name} failed: {e}"))?;
    }
    fs::write(temp.path().join("main.cpp"), main_source)
        .map_err(|e| format!("write main.cpp failed: {e}"))?;

    let bin = temp.path().join(prefix);
    let mut cmd = Command::new("hipcc");
    cmd.arg("-O3");
    cmd.args(compile_flags);
    cmd.arg(temp.path().join("main.cpp"));
    for (name, _) in model_sources {
        cmd.arg(temp.path().join(name));
    }
    cmd.arg("-L").arg(temp.path());
    cmd.arg("-lchelis_runtime");
    cmd.arg("-lpthread");
    cmd.arg("-ldl");
    cmd.args(link_flags);
    cmd.arg("-o").arg(&bin);

    let start = Instant::now();
    let output = cmd
        .output()
        .map_err(|e| format!("hipcc failed to start: {e}"))?;
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    if !output.status.success() {
        return Err(format!(
            "hipcc failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let run = Command::new(&bin)
        .output()
        .map_err(|e| format!("failed to run compiled HIP benchmark: {e}"))?;
    if !run.status.success() {
        return Err(format!(
            "compiled HIP benchmark failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    Ok((
        compile_ms,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    ))
}

fn hip_prerequisite_skip_reason() -> Option<String> {
    static HIP_RUNTIME_PROBE: OnceLock<Option<String>> = OnceLock::new();
    HIP_RUNTIME_PROBE
        .get_or_init(probe_hip_runtime_prerequisite)
        .clone()
}

fn probe_hip_runtime_prerequisite() -> Option<String> {
    if !tool_available("hipcc", &["--version"]) {
        return Some("hipcc not available".to_string());
    }

    let temp = match tempfile::tempdir() {
        Ok(temp) => temp,
        Err(err) => {
            return Some(format!(
                "failed to create HIP prerequisite probe dir: {err}"
            ));
        }
    };
    let src = temp.path().join("probe.cpp");
    let bin = temp.path().join("probe");
    let source = r#"#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        fprintf(stderr, "hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
        return 2;
    }
    if (count <= 0) {
        fprintf(stderr, "no HIP devices visible\n");
        return 3;
    }
    return 0;
}
"#;
    if let Err(err) = fs::write(&src, source) {
        return Some(format!("failed to write HIP prerequisite probe: {err}"));
    }

    let compile = match Command::new("hipcc")
        .arg("-O2")
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return Some(format!(
                "hipcc failed to start for prerequisite probe: {err}"
            ));
        }
    };
    if !compile.status.success() {
        return Some(format!(
            "HIP prerequisite probe failed to compile:\n{}",
            String::from_utf8_lossy(&compile.stderr).trim()
        ));
    }

    let run = match Command::new(&bin).output() {
        Ok(output) => output,
        Err(err) => return Some(format!("failed to run HIP prerequisite probe: {err}")),
    };
    if run.status.success() {
        return None;
    }

    let stderr = String::from_utf8_lossy(&run.stderr).trim().to_string();
    if stderr.is_empty() {
        Some("HIP runtime unavailable".to_string())
    } else {
        Some(format!("HIP runtime unavailable: {stderr}"))
    }
}

fn write_runtime_files(dir: &Path, hip: bool) -> Result<(), String> {
    let cpu_runtime = cpu_runtime_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        fs::write(
            dir.join(header),
            fs::read_to_string(cpu_runtime.join(header))
                .map_err(|e| format!("read {header} failed: {e}"))?,
        )
        .map_err(|e| format!("write {header} failed: {e}"))?;
    }
    fs::copy(cpu_runtime_library(), dir.join("libchelis_runtime.a"))
        .map_err(|e| format!("copy libchelis_runtime.a failed: {e}"))?;
    if hip {
        let hip_runtime = hip_runtime_dir();
        fs::write(
            dir.join("chelis_hip_runtime.h"),
            fs::read_to_string(hip_runtime.join("chelis_hip_runtime.h"))
                .map_err(|e| format!("read chelis_hip_runtime.h failed: {e}"))?,
        )
        .map_err(|e| format!("write chelis_hip_runtime.h failed: {e}"))?;
    }
    Ok(())
}

fn write_linreg_data(path: &Path) -> Result<(), String> {
    let workload = LinregWorkload::current();
    let mut buf = Vec::new();
    push_u64(&mut buf, workload.train_batches as u64);
    push_u64(&mut buf, workload.test_batches as u64);
    push_u64(&mut buf, workload.batch_size as u64);
    push_u64(&mut buf, workload.features as u64);
    push_u64(&mut buf, workload.epochs as u64);
    push_f32(&mut buf, workload.lr);

    let true_w: Vec<f32> = (0..workload.features)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.07)
        .collect();
    let true_b = 0.3f32;

    let train_total = workload.train_batches * workload.batch_size;
    let test_total = workload.test_batches * workload.batch_size;
    let x_train = generate_linreg_inputs(train_total, workload.features);
    let y_train = generate_linreg_targets(&x_train, &true_w, true_b, workload.features);
    let x_test = generate_linreg_inputs(train_total + test_total, workload.features);
    let y_test = generate_linreg_targets(
        &x_test[train_total * workload.features..],
        &true_w,
        true_b,
        workload.features,
    );
    push_f32s(&mut buf, &x_train);
    push_f32s(&mut buf, &y_train);
    push_f32s(&mut buf, &x_test[train_total * workload.features..]);
    push_f32s(&mut buf, &y_test);
    push_f32s(&mut buf, &vec![0.0; workload.features]);
    push_f32(&mut buf, 0.0);
    fs::write(path, buf).map_err(|e| format!("write linreg data failed: {e}"))
}

fn generate_linreg_inputs(samples: usize, features: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples * features);
    for sample in 0..samples {
        for feature in 0..features {
            let idx = sample * features + feature;
            let value = ((idx as f32) * 0.013).sin() + ((sample as f32) * 0.031).cos() * 0.25;
            out.push(value);
        }
    }
    out
}

fn generate_linreg_targets(x: &[f32], w: &[f32], b: f32, features: usize) -> Vec<f32> {
    let samples = x.len() / features;
    let mut out = Vec::with_capacity(samples);
    for sample in 0..samples {
        let row = &x[sample * features..(sample + 1) * features];
        let value = row.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>() + b;
        out.push(value);
    }
    out
}

fn write_mnist_data(path: &Path) -> Result<(), String> {
    let mnist_dir = default_mnist_dir();
    let (train_data, test_data) = load_mnist(&mnist_dir)?;
    let train_data: Vec<_> = train_data.into_iter().take(MNIST_TRAIN_BATCHES).collect();
    let test_data: Vec<_> = test_data.into_iter().take(MNIST_TEST_BATCHES).collect();
    if train_data.len() != MNIST_TRAIN_BATCHES || test_data.len() != MNIST_TEST_BATCHES {
        return Err("MNIST benchmark subset is smaller than requested".to_string());
    }

    let mut buf = Vec::new();
    push_u64(&mut buf, MNIST_TRAIN_BATCHES as u64);
    push_u64(&mut buf, MNIST_TEST_BATCHES as u64);
    push_u64(&mut buf, MNIST_BATCH_SIZE as u64);
    push_u64(&mut buf, MNIST_EPOCHS as u64);
    push_f32(&mut buf, MNIST_LR);

    for (x, _) in &train_data {
        push_tensor_f32(&mut buf, x);
    }
    for (_, y) in &train_data {
        push_tensor_f32(&mut buf, y);
    }
    for (x, _) in &test_data {
        push_tensor_f32(&mut buf, x);
    }
    for (_, y) in &test_data {
        push_tensor_f32(&mut buf, y);
    }

    let params = init_params(MNIST_SEED);
    for name in ["w1", "b1", "w2", "b2"] {
        let tensor = params
            .get(name)
            .ok_or_else(|| format!("missing MNIST init param `{name}`"))?;
        push_tensor_f32(&mut buf, tensor);
    }

    fs::write(path, buf).map_err(|e| format!("write mnist data failed: {e}"))
}

fn write_transformer_data(path: &Path) -> Result<(), String> {
    let mut buf = Vec::new();
    push_u64(&mut buf, TRANSFORMER_SEQ_LEN as u64);
    push_u64(&mut buf, TRANSFORMER_D_MODEL as u64);
    push_u64(&mut buf, TRANSFORMER_HEADS as u64);
    push_u64(&mut buf, TRANSFORMER_HEAD_DIM as u64);
    push_u64(&mut buf, TRANSFORMER_D_FF as u64);
    push_u64(&mut buf, TRANSFORMER_ITERS as u64);

    let x = deterministic_array(TRANSFORMER_SEQ_LEN * TRANSFORMER_D_MODEL, 0.011, 0.37);
    push_f32s(&mut buf, &x);
    for offset in 0..TRANSFORMER_HEADS {
        push_f32s(
            &mut buf,
            &deterministic_array(
                TRANSFORMER_D_MODEL * TRANSFORMER_HEAD_DIM,
                0.007 + offset as f32 * 0.001,
                0.11,
            ),
        );
        push_f32s(
            &mut buf,
            &deterministic_array(
                TRANSFORMER_D_MODEL * TRANSFORMER_HEAD_DIM,
                0.009 + offset as f32 * 0.001,
                0.21,
            ),
        );
        push_f32s(
            &mut buf,
            &deterministic_array(
                TRANSFORMER_D_MODEL * TRANSFORMER_HEAD_DIM,
                0.013 + offset as f32 * 0.001,
                0.31,
            ),
        );
        push_f32s(
            &mut buf,
            &deterministic_array(
                TRANSFORMER_HEAD_DIM * TRANSFORMER_D_MODEL,
                0.005 + offset as f32 * 0.001,
                0.41,
            ),
        );
    }
    push_f32s(
        &mut buf,
        &deterministic_array(TRANSFORMER_D_MODEL * TRANSFORMER_D_FF, 0.004, 0.17),
    );
    push_f32s(
        &mut buf,
        &deterministic_array(TRANSFORMER_D_FF * TRANSFORMER_D_MODEL, 0.003, 0.23),
    );
    for offset in 0..4 {
        push_f32s(
            &mut buf,
            &deterministic_array(
                TRANSFORMER_D_MODEL,
                0.01 + offset as f32 * 0.001,
                0.05 * offset as f32,
            ),
        );
    }
    fs::write(path, buf).map_err(|e| format!("write transformer data failed: {e}"))
}

fn deterministic_array(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * scale + bias).sin() * 0.5)
        .collect()
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(buf: &mut Vec<u8>, value: f32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_f32s(buf: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        push_f32(buf, *value);
    }
}

fn push_tensor_f32(buf: &mut Vec<u8>, tensor: &TensorValue) {
    push_f32s(
        buf,
        &tensor
            .to_f64_lossy_vec()
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<_>>(),
    );
}

fn build_training_main_c(
    _prefix: &str,
    data_path: &Path,
    train: &CCodegenResult,
    train_labels: &UnordMap<String, usize>,
    uses_accuracy: bool,
) -> String {
    let train_input_slots = slot_assignments(
        "train_inputs",
        &train.input_labels,
        &[
            ("x", "x_tensor"),
            ("y", "y_tensor"),
            ("labels", "y_tensor"),
            ("w", "w_tensor"),
            ("b", "b_tensor"),
            ("w1", "w1_tensor"),
            ("b1", "b1_tensor"),
            ("w2", "w2_tensor"),
            ("b2", "b2_tensor"),
        ],
    );
    let loss_idx = train_labels["loss"];
    let eval_output_index = train_labels["eval_output"];
    let grad_w_idx = train_labels.get("grad_w").copied();
    let grad_b_idx = train_labels.get("grad_b").copied();
    let grad_w1_idx = train_labels.get("grad_w1").copied();
    let grad_b1_idx = train_labels.get("grad_b1").copied();
    let grad_w2_idx = train_labels.get("grad_w2").copied();
    let grad_b2_idx = train_labels.get("grad_b2").copied();

    let (param_allocs, param_fills, update_code, free_params) = if uses_accuracy {
        (
            r#"
    chelis_tensor *w1_tensor = chelis_alloc(2, w1_shape, CHELIS_DTYPE_F32);
    chelis_tensor *b1_tensor = chelis_alloc(1, b1_shape, CHELIS_DTYPE_F32);
    chelis_tensor *w2_tensor = chelis_alloc(2, w2_shape, CHELIS_DTYPE_F32);
    chelis_tensor *b2_tensor = chelis_alloc(1, b2_shape, CHELIS_DTYPE_F32);
"#,
            r#"
    tensor_copy_in(w1_tensor, w1_init, sizeof(float) * 784 * 128);
    tensor_copy_in(b1_tensor, b1_init, sizeof(float) * 128);
    tensor_copy_in(w2_tensor, w2_init, sizeof(float) * 128 * 10);
    tensor_copy_in(b2_tensor, b2_init, sizeof(float) * 10);
"#,
            format!(
                r#"
            tensor_sgd_update(w1_tensor, train_outputs[{gw1}], 784 * 128, lr);
            tensor_sgd_update(b1_tensor, train_outputs[{gb1}], 128, lr);
            tensor_sgd_update(w2_tensor, train_outputs[{gw2}], 128 * 10, lr);
            tensor_sgd_update(b2_tensor, train_outputs[{gb2}], 10, lr);
"#,
                gw1 = grad_w1_idx.unwrap(),
                gb1 = grad_b1_idx.unwrap(),
                gw2 = grad_w2_idx.unwrap(),
                gb2 = grad_b2_idx.unwrap(),
            ),
            r#"
    chelis_tensor_release(w1_tensor);
    chelis_tensor_release(b1_tensor);
    chelis_tensor_release(w2_tensor);
    chelis_tensor_release(b2_tensor);
"#,
        )
    } else {
        (
            r#"
    chelis_tensor *w_tensor = chelis_alloc(2, w_shape, CHELIS_DTYPE_F32);
    chelis_tensor *b_tensor = chelis_alloc(1, b_shape, CHELIS_DTYPE_F32);
"#,
            r#"
    tensor_copy_in(w_tensor, w_init, sizeof(float) * features);
    tensor_copy_in(b_tensor, b_init, sizeof(float) * 1);
"#,
            format!(
                r#"
            tensor_sgd_update(w_tensor, train_outputs[{gw}], features, lr);
            tensor_sgd_update(b_tensor, train_outputs[{gb}], 1, lr);
"#,
                gw = grad_w_idx.unwrap(),
                gb = grad_b_idx.unwrap(),
            ),
            r#"
    chelis_tensor_release(w_tensor);
    chelis_tensor_release(b_tensor);
"#,
        )
    };

    let metric_calc = if uses_accuracy {
        r#"
    int correct = 0;
    for (uint64_t batch = 0; batch < test_batches; batch++) {
        tensor_copy_in(x_tensor, x_test + batch * batch_size * x_dim, sizeof(float) * batch_size * x_dim);
        tensor_copy_in(y_tensor, y_test + batch * batch_size * y_dim, sizeof(float) * batch_size * y_dim);
        chelis_tensor *infer_outputs[TRAIN_OUTPUT_COUNT] = {0};
        chelis_tensor *infer_inputs[TRAIN_INPUT_COUNT] = {0};
TRAIN_INPUT_ASSIGNMENTS
        chelis_train(infer_inputs, TRAIN_INPUT_COUNT, infer_outputs, TRAIN_OUTPUT_COUNT);
        memcpy(eval_output + batch * batch_size * y_dim, tensor_f32_data(infer_outputs[EVAL_OUTPUT_INDEX]), sizeof(float) * batch_size * y_dim);
        for (int b = 0; b < batch_size; b++) {
            int pred = 0;
            int truth = 0;
            float pred_best = tensor_f32_data(infer_outputs[EVAL_OUTPUT_INDEX])[b * y_dim];
            float truth_best = tensor_f32_data(y_tensor)[b * y_dim];
            for (int cls = 1; cls < y_dim; cls++) {
                float pred_val = tensor_f32_data(infer_outputs[EVAL_OUTPUT_INDEX])[b * y_dim + cls];
                if (pred_val > pred_best) { pred_best = pred_val; pred = cls; }
                float truth_val = tensor_f32_data(y_tensor)[b * y_dim + cls];
                if (truth_val > truth_best) { truth_best = truth_val; truth = cls; }
            }
            if (pred == truth) correct++;
        }
        for (int i = 0; i < TRAIN_OUTPUT_COUNT; i++) if (infer_outputs[i]) chelis_tensor_release(infer_outputs[i]);
    }
    final_accuracy = (float)correct / (float)(test_batches * batch_size);
"#
    } else {
        r#"
    for (uint64_t batch = 0; batch < test_batches; batch++) {
        tensor_copy_in(x_tensor, x_test + batch * batch_size * x_dim, sizeof(float) * batch_size * x_dim);
        tensor_copy_in(y_tensor, y_test + batch * batch_size * y_dim, sizeof(float) * batch_size * y_dim);
        chelis_tensor *infer_outputs[TRAIN_OUTPUT_COUNT] = {0};
        chelis_tensor *infer_inputs[TRAIN_INPUT_COUNT] = {0};
TRAIN_INPUT_ASSIGNMENTS
        chelis_train(infer_inputs, TRAIN_INPUT_COUNT, infer_outputs, TRAIN_OUTPUT_COUNT);
        memcpy(eval_output + batch * batch_size, tensor_f32_data(infer_outputs[EVAL_OUTPUT_INDEX]), sizeof(float) * batch_size);
        for (int i = 0; i < TRAIN_OUTPUT_COUNT; i++) if (infer_outputs[i]) chelis_tensor_release(infer_outputs[i]);
    }
"#
    };

    let final_json = if uses_accuracy {
        r#"
    printf("{\"run_ms\":%.6f,\"loss_history\":[", run_ms);
    for (uint64_t epoch = 0; epoch < epochs; epoch++) {
        if (epoch) printf(",");
        printf("%.8f", loss_history[epoch]);
    }
    printf("],\"final_loss\":%.8f,\"final_accuracy\":%.8f,\"output\":[", loss_history[epochs - 1], final_accuracy);
    for (int i = 0; i < eval_output_len; i++) {
        if (i) printf(",");
        printf("%.8f", eval_output[i]);
    }
    printf("]}");
"#
    } else {
        r#"
    printf("{\"run_ms\":%.6f,\"loss_history\":[", run_ms);
    for (uint64_t epoch = 0; epoch < epochs; epoch++) {
        if (epoch) printf(",");
        printf("%.8f", loss_history[epoch]);
    }
    printf("],\"final_loss\":%.8f,\"final_accuracy\":null,\"output\":[", loss_history[epochs - 1]);
    for (int i = 0; i < eval_output_len; i++) {
        if (i) printf(",");
        printf("%.8f", eval_output[i]);
    }
    printf("]}");
"#
    };

    format!(
        r#"#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chelis_runtime.h"

{train_header}

static uint64_t read_u64(FILE *f) {{
    uint64_t value = 0;
    if (fread(&value, sizeof(uint64_t), 1, f) != 1) {{
        fprintf(stderr, "failed to read u64\n");
        exit(1);
    }}
    return value;
}}

static float read_f32(FILE *f) {{
    float value = 0.0f;
    if (fread(&value, sizeof(float), 1, f) != 1) {{
        fprintf(stderr, "failed to read f32\n");
        exit(1);
    }}
    return value;
}}

static void read_f32s(FILE *f, float *out, size_t n) {{
    if (fread(out, sizeof(float), n, f) != n) {{
        fprintf(stderr, "failed to read %zu floats\n", n);
        exit(1);
    }}
}}

static const float *tensor_f32_data(const chelis_tensor *tensor) {{
    return (const float *)chelis_tensor_read_view(tensor).data;
}}

static void tensor_copy_in(chelis_tensor *tensor, const void *source, size_t bytes) {{
    chelis_tensor_write *guard = chelis_tensor_begin_write(tensor);
    chelis_write_view view = chelis_tensor_write_view(guard);
    memcpy(view.data, source, bytes);
    chelis_tensor_end_write(guard);
}}

static void tensor_sgd_update(
    chelis_tensor *parameter, const chelis_tensor *gradient, int64_t count, float rate
) {{
    const float *gradient_data = tensor_f32_data(gradient);
    chelis_tensor_write *guard = chelis_tensor_begin_write(parameter);
    chelis_write_view view = chelis_tensor_write_view(guard);
    float *parameter_data = (float *)view.data;
    for (int64_t index = 0; index < count; ++index) {{
        parameter_data[index] -= rate * gradient_data[index];
    }}
    chelis_tensor_end_write(guard);
}}

static double elapsed_ms(struct timespec start, struct timespec end) {{
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}}

int main(void) {{
    FILE *f = fopen("{data_path}", "rb");
    if (!f) {{
        fprintf(stderr, "failed to open benchmark data file\n");
        return 1;
    }}

    uint64_t train_batches = read_u64(f);
    uint64_t test_batches = read_u64(f);
    uint64_t batch_size = read_u64(f);
    uint64_t features = {features_read};
    uint64_t epochs = read_u64(f);
    float lr = read_f32(f);
    uint64_t x_dim = {x_dim};
    uint64_t y_dim = {y_dim};
    int64_t x_shape[2] = {{ (int)batch_size, (int)x_dim }};
    int64_t y_shape[2] = {{ (int)batch_size, (int)y_dim }};
{shape_decls}

    size_t train_x_len = (size_t)train_batches * batch_size * x_dim;
    size_t train_y_len = (size_t)train_batches * batch_size * y_dim;
    size_t test_x_len = (size_t)test_batches * batch_size * x_dim;
    size_t test_y_len = (size_t)test_batches * batch_size * y_dim;

    float *x_train = (float*)malloc(sizeof(float) * train_x_len);
    float *y_train = (float*)malloc(sizeof(float) * train_y_len);
    float *x_test = (float*)malloc(sizeof(float) * test_x_len);
    float *y_test = (float*)malloc(sizeof(float) * test_y_len);
    read_f32s(f, x_train, train_x_len);
    read_f32s(f, y_train, train_y_len);
    read_f32s(f, x_test, test_x_len);
    read_f32s(f, y_test, test_y_len);
{init_allocs}
    fclose(f);

    chelis_tensor *x_tensor = chelis_alloc(2, x_shape, CHELIS_DTYPE_F32);
    chelis_tensor *y_tensor = chelis_alloc(2, y_shape, CHELIS_DTYPE_F32);
{param_allocs}
{param_fills}

    float *loss_history = (float*)calloc(epochs, sizeof(float));
    int eval_output_len = (int)(test_batches * batch_size * y_dim);
    float *eval_output = (float*)calloc((size_t)eval_output_len, sizeof(float));
    float final_accuracy = 0.0f;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (uint64_t epoch = 0; epoch < epochs; epoch++) {{
        double epoch_loss = 0.0;
        for (uint64_t batch = 0; batch < train_batches; batch++) {{
            tensor_copy_in(x_tensor, x_train + batch * batch_size * x_dim, sizeof(float) * batch_size * x_dim);
            tensor_copy_in(y_tensor, y_train + batch * batch_size * y_dim, sizeof(float) * batch_size * y_dim);

            chelis_tensor *train_outputs[{train_outs}] = {{0}};
            chelis_tensor *train_inputs[{train_ins}] = {{0}};
{train_input_slots}
            chelis_train(train_inputs, {train_ins}, train_outputs, {train_outs});
            epoch_loss += tensor_f32_data(train_outputs[{loss_idx}])[0];
{update_code}
            for (int i = 0; i < {train_outs}; i++) chelis_tensor_release(train_outputs[i]);
        }}
        loss_history[epoch] = (float)(epoch_loss / (double)train_batches);
    }}
    clock_gettime(CLOCK_MONOTONIC, &end);

    double run_ms = elapsed_ms(start, end);
{metric_calc}
{final_json}

    free(x_train);
    free(y_train);
    free(x_test);
    free(y_test);
{free_inits}
    chelis_tensor_release(x_tensor);
    chelis_tensor_release(y_tensor);
{free_params}
    free(loss_history);
    free(eval_output);
    return 0;
}}
"#,
        data_path = escape_c_string(data_path),
        train_header = train.h_header,
        x_dim = if uses_accuracy {
            "784"
        } else {
            "(int)features"
        },
        y_dim = if uses_accuracy { 10 } else { 1 },
        shape_decls = if uses_accuracy {
            r#"
    int64_t w1_shape[2] = { 784, 128 };
    int64_t b1_shape[1] = { 128 };
    int64_t w2_shape[2] = { 128, 10 };
    int64_t b2_shape[1] = { 10 };
"#
        } else {
            r#"
    int64_t w_shape[2] = { (int)features, 1 };
    int64_t b_shape[1] = { 1 };
"#
        },
        features_read = if uses_accuracy { "0" } else { "read_u64(f)" },
        init_allocs = if uses_accuracy {
            r#"
    float *w1_init = (float*)malloc(sizeof(float) * 784 * 128);
    float *b1_init = (float*)malloc(sizeof(float) * 128);
    float *w2_init = (float*)malloc(sizeof(float) * 128 * 10);
    float *b2_init = (float*)malloc(sizeof(float) * 10);
    read_f32s(f, w1_init, 784 * 128);
    read_f32s(f, b1_init, 128);
    read_f32s(f, w2_init, 128 * 10);
    read_f32s(f, b2_init, 10);
"#
        } else {
            r#"
    float *w_init = (float*)malloc(sizeof(float) * features);
    float *b_init = (float*)malloc(sizeof(float) * 1);
    read_f32s(f, w_init, features);
    read_f32s(f, b_init, 1);
"#
        },
        param_allocs = param_allocs,
        param_fills = param_fills,
        train_outs = train.output_labels.len(),
        train_ins = train.input_labels.len(),
        train_input_slots = train_input_slots,
        loss_idx = loss_idx,
        update_code = update_code,
        metric_calc = metric_calc
            .replace(
                "TRAIN_INPUT_ASSIGNMENTS",
                &slot_assignments(
                    "infer_inputs",
                    &train.input_labels,
                    &[
                        ("x", "x_tensor"),
                        ("y", "y_tensor"),
                        ("labels", "y_tensor"),
                        ("w", "w_tensor"),
                        ("b", "b_tensor"),
                        ("w1", "w1_tensor"),
                        ("b1", "b1_tensor"),
                        ("w2", "w2_tensor"),
                        ("b2", "b2_tensor"),
                    ]
                )
            )
            .replace("TRAIN_INPUT_COUNT", &train.input_labels.len().to_string())
            .replace("TRAIN_OUTPUT_COUNT", &train.output_labels.len().to_string())
            .replace("EVAL_OUTPUT_INDEX", &eval_output_index.to_string()),
        final_json = final_json,
        free_inits = if uses_accuracy {
            r#"
    free(w1_init);
    free(b1_init);
    free(w2_init);
    free(b2_init);
"#
        } else {
            r#"
    free(w_init);
    free(b_init);
"#
        },
        free_params = free_params,
    )
}

fn build_forward_main_c(
    _prefix: &str,
    data_path: &Path,
    model: &CCodegenResult,
    output_index: usize,
) -> String {
    let input_slots = slot_assignments(
        "inputs",
        &model.input_labels,
        &[
            ("x", "x_tensor"),
            ("wq", "wq_tensor"),
            ("wk", "wk_tensor"),
            ("wv", "wv_tensor"),
            ("wo", "wo_tensor"),
            ("ff1", "ff1_tensor"),
            ("ff2", "ff2_tensor"),
            ("gamma1", "gamma1_tensor"),
            ("beta1", "beta1_tensor"),
            ("gamma2", "gamma2_tensor"),
            ("beta2", "beta2_tensor"),
        ],
    );

    format!(
        r#"#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chelis_runtime.h"

{header}

static uint64_t read_u64(FILE *f) {{
    uint64_t value = 0;
    if (fread(&value, sizeof(uint64_t), 1, f) != 1) {{
        fprintf(stderr, "failed to read u64\n");
        exit(1);
    }}
    return value;
}}

static void read_f32s(FILE *f, float *out, size_t n) {{
    if (fread(out, sizeof(float), n, f) != n) {{
        fprintf(stderr, "failed to read %zu floats\n", n);
        exit(1);
    }}
}}

static double elapsed_ms(struct timespec start, struct timespec end) {{
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}}

static chelis_tensor *alloc_and_fill(FILE *f, int32_t rank, const int64_t *shape, size_t count) {{
    chelis_tensor *tensor = chelis_alloc(rank, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(tensor);
    chelis_write_view view = chelis_tensor_write_view(guard);
    read_f32s(f, (float *)view.data, count);
    chelis_tensor_end_write(guard);
    return tensor;
}}

int main(void) {{
    FILE *f = fopen("{data_path}", "rb");
    if (!f) {{
        fprintf(stderr, "failed to open benchmark data file\n");
        return 1;
    }}

    uint64_t seq_len = read_u64(f);
    uint64_t d_model = read_u64(f);
    uint64_t n_heads = read_u64(f);
    uint64_t head_dim = read_u64(f);
    uint64_t d_ff = read_u64(f);
    uint64_t iters = read_u64(f);
    int64_t x_shape[2] = {{ (int64_t)seq_len, (int64_t)d_model }};
    int64_t head_shape[2] = {{ (int64_t)d_model, (int64_t)head_dim }};
    int64_t proj_shape[2] = {{ (int64_t)head_dim, (int64_t)d_model }};
    int64_t ff1_shape[2] = {{ (int64_t)d_model, (int64_t)d_ff }};
    int64_t ff2_shape[2] = {{ (int64_t)d_ff, (int64_t)d_model }};
    int64_t norm_shape[1] = {{ (int64_t)d_model }};

    chelis_tensor *x_tensor = alloc_and_fill(f, 2, x_shape, seq_len * d_model);
    chelis_tensor *wq_tensor = alloc_and_fill(f, 2, head_shape, d_model * head_dim);
    chelis_tensor *wk_tensor = alloc_and_fill(f, 2, head_shape, d_model * head_dim);
    chelis_tensor *wv_tensor = alloc_and_fill(f, 2, head_shape, d_model * head_dim);
    chelis_tensor *wo_tensor = alloc_and_fill(f, 2, proj_shape, head_dim * d_model);
    chelis_tensor *ff1_tensor = alloc_and_fill(f, 2, ff1_shape, d_model * d_ff);
    chelis_tensor *ff2_tensor = alloc_and_fill(f, 2, ff2_shape, d_ff * d_model);
    chelis_tensor *gamma1_tensor = alloc_and_fill(f, 1, norm_shape, d_model);
    chelis_tensor *beta1_tensor = alloc_and_fill(f, 1, norm_shape, d_model);
    chelis_tensor *gamma2_tensor = alloc_and_fill(f, 1, norm_shape, d_model);
    chelis_tensor *beta2_tensor = alloc_and_fill(f, 1, norm_shape, d_model);
    fclose(f);

    chelis_tensor *inputs[{n_inputs}] = {{0}};
{input_slots}

    chelis_tensor *outputs[{n_outputs}] = {{0}};
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (uint64_t iter = 0; iter < iters; iter++) {{
        for (int i = 0; i < {n_outputs}; i++) {{
            if (outputs[i]) {{
                chelis_tensor_release(outputs[i]);
                outputs[i] = NULL;
            }}
        }}
        chelis_forward(inputs, {n_inputs}, outputs, {n_outputs});
    }}
    clock_gettime(CLOCK_MONOTONIC, &end);

    chelis_tensor *out = outputs[{output_index}];
    chelis_read_view out_view = chelis_tensor_read_view(out);
    int out_size = (int)out_view.count;
    printf("{{\"run_ms\":%.6f,\"loss_history\":[],\"final_loss\":null,\"final_accuracy\":null,\"output\":[", elapsed_ms(start, end));
    for (int i = 0; i < out_size; i++) {{
        if (i) printf(",");
        printf("%.8f", ((const float *)out_view.data)[i]);
    }}
    printf("]}}");

    for (int i = 0; i < {n_outputs}; i++) {{
        if (outputs[i]) chelis_tensor_release(outputs[i]);
    }}
    chelis_tensor_release(x_tensor);
    chelis_tensor_release(wq_tensor);
    chelis_tensor_release(wk_tensor);
    chelis_tensor_release(wv_tensor);
    chelis_tensor_release(wo_tensor);
    chelis_tensor_release(ff1_tensor);
    chelis_tensor_release(ff2_tensor);
    chelis_tensor_release(gamma1_tensor);
    chelis_tensor_release(beta1_tensor);
    chelis_tensor_release(gamma2_tensor);
    chelis_tensor_release(beta2_tensor);
    return 0;
}}
"#,
        header = model.h_header,
        data_path = escape_c_string(data_path),
        n_inputs = model.input_labels.len(),
        input_slots = input_slots,
        n_outputs = model.output_labels.len(),
        output_index = output_index,
    )
}

fn slot_assignments(array_name: &str, labels: &[String], mapping: &[(&str, &str)]) -> String {
    let map: UnordMap<&str, &str> = mapping.iter().copied().collect();
    let mut lines = Vec::new();
    for (idx, label) in labels.iter().enumerate() {
        let value = map
            .get(label.as_str())
            .unwrap_or_else(|| panic!("missing slot assignment for `{label}`"));
        lines.push(format!("    {array_name}[{idx}] = {value};"));
    }
    lines.join("\n")
}

fn escape_c_string(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn model_parser_accepts_all_keyword() {
        let parsed = Model::parse("all").expect("parse all");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].name(), "linreg");
        assert_eq!(parsed[1].name(), "mnist");
        assert_eq!(parsed[2].name(), "transformer");
    }

    #[test]
    fn training_comparison_detects_decreasing_loss() {
        let make = |losses: Vec<f32>| RunArtifacts {
            report: BackendReport {
                status: "ok",
                reason: None,
                compile_ms: None,
                run_ms: None,
                peak_device_bytes_formula: None,
                peak_device_bytes_estimate: None,
                loss_history: Some(losses),
                final_loss: None,
                final_accuracy: None,
                output_len: Some(1),
                output_checksum: Some(0.0),
                output_sample: Some(vec![0.0]),
            },
            output: vec![0.0],
        };

        let comparison = compare_training(
            "cpu_vs_pytorch",
            &make(vec![3.0, 2.0]),
            &make(vec![4.0, 1.0]),
            false,
        );
        assert_eq!(comparison.status, "ok");
    }

    #[test]
    fn resolve_pytorch_python_prefers_repo_venv() {
        let dir = tempdir().expect("tempdir");
        let python = dir.path().join("py/.venv/bin/python");
        fs::create_dir_all(python.parent().expect("parent")).expect("create parent");
        fs::write(&python, b"").expect("write fake python");

        let resolved = resolve_pytorch_python_in(dir.path()).expect("resolve repo python");
        assert_eq!(resolved, python);
    }

    #[test]
    fn resolve_pytorch_python_reports_setup_hint_when_missing() {
        let dir = tempdir().expect("tempdir");
        let err = resolve_pytorch_python_in(dir.path()).expect_err("missing repo python");
        assert!(err.contains(PYTORCH_SETUP_HINT));
    }

    #[test]
    fn run_pytorch_with_non_python_binary_becomes_structured_skip() {
        let current = env::current_exe().expect("current exe");
        let result = run_pytorch_with_python(
            &current,
            "benchmarks/pytorch/linreg.py",
            Path::new("unused.bin"),
            None,
            "linreg",
        );
        assert_eq!(result.report.status, "skipped");
        let reason = result.report.reason.expect("skip reason");
        assert!(reason.contains("PyTorch import failed"));
    }

    #[test]
    fn linreg_workload_smoke_profile_is_small() {
        let workload = LinregWorkload::for_profile(Some("smoke"));

        // The smoke profile's speedup comes from a single train/test batch
        // and one epoch. `batch_size` must stay at the example's fixed
        // dimension (`examples/linreg.ch` is `tensor[64, 64]`); shrinking it
        // would make the compiled program reject its own input shape.
        assert_eq!(workload.train_batches, 1);
        assert_eq!(workload.test_batches, 1);
        assert_eq!(workload.batch_size, LINREG_BATCH_SIZE);
        assert_eq!(workload.epochs, 1);
    }
}
