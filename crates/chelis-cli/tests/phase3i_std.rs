use assert_cmd::Command;
use chelis_ir::host::lower_compiled_program;
use chelis_reef::prepare_program_for_file;
use chelis_surf::desugar::desugar_program;
use chelis_types::{check_linearity, check_phase0e_program};
use predicates::prelude::*;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn example_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .canonicalize()
        .expect("path should exist")
}

fn package_std() -> PathBuf {
    example_path("../../packages/chelis-std")
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

fn gcc_link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let needs_blas = fs::read_to_string(out_dir.join(source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(source);
    cmd.args(["-L.", "-lchelis_runtime", "-lpthread", "-ldl"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("gcc should run")
}

fn make_app(dir_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join(dir_name);
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();
    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "=0.1.7"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

#[test]
fn reef_std_time_and_decimal_modules_eval() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-time-decimal");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Time (
  Date,
  Duration,
  add_days,
  date,
  date_gte,
  date_lt,
  date_to_string,
  day_of_year,
  day_of_week_name,
  days_between,
  duration,
  is_leap_year,
  parse_date
)
import Std.Decimal (
  decimal,
  decimal_add,
  decimal_div,
  decimal_eq,
  decimal_from_int,
  decimal_to_string,
  round_half_even,
  try_decimal
)

next_day = date_to_string(add_days(date(cast(2024, int64), cast(2, int64), cast(28, int64)), cast(1, int64)))
weekday = day_of_week_name(date(cast(2024, int64), cast(2, int64), cast(26, int64)))
ordinal = day_of_year(date(cast(2024, int64), cast(12, int64), cast(31, int64)))
leap = is_leap_year(cast(2024, int64))
span = duration(cast(1, int64), cast(2, int64), cast(3, int64), cast(4, int64))
parsed_ok = match parse_date("2024-12-31") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
parsed = match parse_date("2024-02-30") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
cross_year_days = days_between(
  date(cast(2024, int64), cast(12, int64), cast(31, int64)),
  date(cast(2025, int64), cast(1, int64), cast(2, int64))
)
cross_year_lt = date_lt(
  date(cast(2024, int64), cast(12, int64), cast(31, int64)),
  date(cast(2025, int64), cast(1, int64), cast(2, int64))
)
cross_year_gte = date_gte(
  date(cast(2025, int64), cast(1, int64), cast(2, int64)),
  date(cast(2024, int64), cast(12, int64), cast(31, int64))
)
exact = decimal_eq(decimal_add(decimal("0.1"), decimal("0.2")), decimal("0.3"))
banker = decimal_to_string(
  decimal_div(decimal_from_int(cast(5, int64)), decimal_from_int(cast(2, int64)), cast(0, int64), round_half_even())
)
bad_decimal = match try_decimal("x.y") with {
  | Some(_) => "bad"
  | None => "invalid-decimal"
}
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("next_day = 2024-02-29"))
        .stdout(predicate::str::contains("weekday = monday"))
        .stdout(predicate::str::contains("ordinal = 366"))
        .stdout(predicate::str::contains("leap = true"))
        .stdout(predicate::str::contains("span = Duration(1, 2, 3, 4)"))
        .stdout(predicate::str::contains("parsed_ok = 2024-12-31"))
        .stdout(predicate::str::contains("parsed = invalid"))
        .stdout(predicate::str::contains("cross_year_days = 2"))
        .stdout(predicate::str::contains("cross_year_lt = true"))
        .stdout(predicate::str::contains("cross_year_gte = true"))
        .stdout(predicate::str::contains("exact = true"))
        .stdout(predicate::str::contains("banker = 2"))
        .stdout(predicate::str::contains("bad_decimal = invalid-decimal"));
}

#[test]
fn reef_std_schedule_and_optim_modules_eval() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-schedule-optim");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Schedule (
  CosineWarmupConfig,
  LinearWarmupConfig,
  StepDecayConfig,
  cosine_with_warmup,
  linear_warmup,
  step_decay
)
import Std.Optim (AdamWConfig, adamw_init_like, adamw_step, LAMBConfig, lamb_init_like, lamb_step)

params = to_tensor([1.0, 2.0, 3.0])
grads = to_tensor([0.1, 0.2, 0.3])

cos_lr = cosine_with_warmup(cast(5, int64), CosineWarmupConfig {
  warmup_steps: cast(2, int64),
  total_steps: cast(10, int64),
  min_lr: 0.1,
  max_lr: 1.0
})
cos_zero = cosine_with_warmup(cast(0, int64), CosineWarmupConfig {
  warmup_steps: cast(2, int64),
  total_steps: cast(10, int64),
  min_lr: 0.1,
  max_lr: 1.0
})
cos_end = cosine_with_warmup(cast(10, int64), CosineWarmupConfig {
  warmup_steps: cast(2, int64),
  total_steps: cast(10, int64),
  min_lr: 0.1,
  max_lr: 1.0
})
lin_lr = linear_warmup(
  cast(1, int64),
  LinearWarmupConfig {
    warmup_steps: cast(4, int64),
    target_lr: 0.8
  }
)
step_lr = step_decay(
  cast(7, int64),
  StepDecayConfig {
    initial_lr: 1.0,
    decay_factor: 0.5,
    decay_steps: [cast(3, int64), cast(6, int64)]
  }
)

adam_state = adamw_init_like(params)
adam_pair = adamw_step(
  params,
  grads,
  adam_state,
  AdamWConfig {
    lr: 0.1,
    beta1: 0.9,
    beta2: 0.999,
    eps: 0.000001,
    weight_decay: 0.01
  }
)
adam_params = adam_pair.0
decay_only = adamw_step(
  params,
  to_tensor([0.0, 0.0, 0.0]),
  adamw_init_like(params),
  AdamWConfig {
    lr: 0.1,
    beta1: 0.9,
    beta2: 0.999,
    eps: 0.000001,
    weight_decay: 0.01
  }
).0

lamb_state = lamb_init_like(params)
lamb_pair = lamb_step(
  params,
  grads,
  lamb_state,
  LAMBConfig {
    lr: 0.1,
    beta1: 0.9,
    beta2: 0.999,
    eps: 0.000001,
    weight_decay: 0.01
  }
)
lamb_params = lamb_pair.0
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("cos_lr ="))
        .stdout(predicate::str::contains("cos_zero = 0"))
        .stdout(predicate::str::contains("cos_end = 0.1"))
        .stdout(predicate::str::contains("lin_lr = 0.2"))
        .stdout(predicate::str::contains("step_lr = 0.25"))
        .stdout(predicate::str::contains(
            "decay_only = tensor(shape=[3], data=[0.999, 1.998, 2.997])",
        ))
        .stdout(predicate::str::contains("adam_params = tensor(shape=[3]"))
        .stdout(predicate::str::contains(
            "lamb_params = tensor(shape=[3], data=[0.7861008931225082, 1.783982000313614, 2.7818638134364133])",
        ));
}

#[test]
fn reef_std_schedule_and_optim_reject_old_positional_configs() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-schedule-optim-bad");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Schedule (CosineWarmupConfig, cosine_with_warmup)
import Std.Optim (AdamWConfig, adamw_init_like, adamw_step)

params = to_tensor([1.0, 2.0, 3.0])
grads = to_tensor([0.1, 0.2, 0.3])
bad_lr = cosine_with_warmup(cast(5, int64), CosineWarmupConfig(cast(2, int64), cast(10, int64), 0.1, 1.0))
bad_pair = adamw_step(params, grads, adamw_init_like(params), AdamWConfig(0.1, 0.9, 0.999, 0.000001, 0.01))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("record constructor"))
        .stdout(predicate::str::contains("CosineWarmupConfig"))
        .stdout(predicate::str::contains("AdamWConfig"));
}

#[test]
fn reef_std_generate_module_checks_and_evals_real_generation() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-generate");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Generate (GenerateConfig, KVCache, generate, generate_with)

def keep_cache[p](cache: Option[KVCache[p]]) -> KVCache[p] =
  match cache with {
    | Some(value) => value
    | None => KVCache([])
  }

def toy_model[batch, seq](input_ids: tensor[batch, seq, int64], cache: Option[KVCache[tensor[kv, f32]]]) = {
  logits = pad_sequences_to([[0.05, 0.10, 0.90, 0.60]], cast(4, int64), 0.0)
  (logits, keep_cache(cache))
}

context = pad_sequences_to([[cast(1, int64), cast(2, int64)]], cast(2, int64), cast(0, int64))
cfg = GenerateConfig {
  max_tokens: cast(2, int64),
  temperature: 1.0,
  top_k: cast(2, int64),
  top_p: 0.95
}
greedy = generate(toy_model, context, cast(2, int64))
sampled_a = with seed(7) { generate_with(toy_model, context, cfg) }
sampled_b = with seed(7) { generate_with(toy_model, context, cfg) }
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "greedy = tensor(shape=[1, 4], data=[1.0, 2.0, 2.0, 2.0])",
        ))
        .stdout(predicate::str::contains(
            "sampled_a = tensor(shape=[1, 4], data=[1.0, 2.0,",
        ))
        .stdout(predicate::str::contains(
            "sampled_b = tensor(shape=[1, 4], data=[1.0, 2.0,",
        ))
        .stdout(predicate::str::contains(
            "sampled_a = tensor(shape=[1, 4], data=[1.0, 2.0, 2.0, 3.0])",
        ))
        .stdout(predicate::str::contains(
            "sampled_b = tensor(shape=[1, 4], data=[1.0, 2.0, 2.0, 3.0])",
        ));
}

#[test]
fn reef_package_mode_preserves_split_map_and_runtime_reshape_typing() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-package-typing");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

source = pad_sequences_to([[1.0, 2.0], [3.0, 4.0]], cast(2, int64), 0.0)
rows = split(source, cast(0, int32), [cast(1, int64), cast(1, int64)])
flat_rows = map(
  fn (row: tensor[piece, seq, f32]) -> reshape(row, [cast(2, int64)]),
  rows
)
first = index(flat_rows, cast(0, int64))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "first = tensor(shape=[2], data=[1.0, 2.0])",
        ));
}

#[test]
fn reef_std_generate_module_rejects_old_positional_config() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-generate-bad");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Generate (GenerateConfig)

cfg = GenerateConfig(cast(2, int64), 1.0, cast(2, int64), 0.95)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("record constructor"))
        .stdout(predicate::str::contains("GenerateConfig"));
}

#[test]
fn reef_std_generate_with_zero_temp_still_requires_seed() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-generate-zero-temp");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Generate (GenerateConfig, KVCache, generate_with)

def keep_cache[p](cache: Option[KVCache[p]]) -> KVCache[p] =
  match cache with {
    | Some(value) => value
    | None => KVCache([])
  }

def toy_model[batch, seq](input_ids: tensor[batch, seq, int64], cache: Option[KVCache[tensor[kv, f32]]]) = {
  logits = pad_sequences_to([[0.05, 0.10, 0.90, 0.60]], cast(4, int64), 0.0)
  (logits, keep_cache(cache))
}

context = pad_sequences_to([[cast(1, int64), cast(2, int64)]], cast(2, int64), cast(0, int64))
cfg = GenerateConfig {
  max_tokens: cast(1, int64),
  temperature: 0.0,
  top_k: cast(2, int64),
  top_p: 0.95
}
out = generate_with(toy_model, context, cfg)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unhandled effect `Random`"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            app_pkg.join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unhandled effect `Random`"));
}

#[test]
fn reef_std_package_mode_builds_and_runs_time_decimal_schedule_program() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-build");
    let out_dir = app_pkg.join("out");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Time (add_days, date, date_gte, date_lt, date_to_string, days_between, parse_date)
import Std.Decimal (decimal, decimal_add, decimal_to_string)
import Std.Schedule (LinearWarmupConfig, linear_warmup)

tomorrow = date_to_string(add_days(date(cast(2024, int64), cast(2, int64), cast(28, int64)), cast(1, int64)))
parsed_ok = match parse_date("2024-12-31") with {
  | Some(value) => date_to_string(value)
  | None => "invalid"
}
cross_year_days = days_between(
  date(cast(2024, int64), cast(12, int64), cast(31, int64)),
  date(cast(2025, int64), cast(1, int64), cast(2, int64))
)
cross_year_lt = date_lt(
  date(cast(2024, int64), cast(12, int64), cast(31, int64)),
  date(cast(2025, int64), cast(1, int64), cast(2, int64))
)
cross_year_gte = date_gte(
  date(cast(2025, int64), cast(1, int64), cast(2, int64)),
  date(cast(2024, int64), cast(12, int64), cast(31, int64))
)
exact = decimal_to_string(decimal_add(decimal("0.1"), decimal("0.2")))
lr = linear_warmup(cast(1, int64), LinearWarmupConfig { warmup_steps: cast(4, int64), target_lr: 0.8 })
"#,
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = gcc_link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("main"))
        .current_dir(&app_pkg)
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
    let stdout = String::from_utf8(run_output.stdout).expect("compiled stdout should be utf-8");
    assert!(stdout.contains("tomorrow = 2024-02-29"));
    assert!(stdout.contains("parsed_ok = 2024-12-31"));
    assert!(stdout.contains("cross_year_days = 2"));
    assert!(stdout.contains("cross_year_lt = true"));
    assert!(stdout.contains("cross_year_gte = true"));
}

#[test]
fn reef_std_generate_builds_and_runs_compiled_program() {
    let (_dir, reef_home, app_pkg) = make_app("phase3i-generate-build");
    let out_dir = app_pkg.join("out");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Generate (KVCache, generate)

def keep_cache[p](cache: Option[KVCache[p]]) -> KVCache[p] =
  match cache with {
    | Some(value) => value
    | None => KVCache([])
  }

def toy_model[batch, seq](input_ids: tensor[batch, seq, int64], cache: Option[KVCache[tensor[kv, f32]]]) = {
  logits = pad_sequences_to([[0.05, 0.10, 0.90, 0.60]], cast(4, int64), 0.0)
  (logits, keep_cache(cache))
}

context = pad_sequences_to([[cast(1, int64), cast(2, int64)]], cast(2, int64), cast(0, int64))
greedy = generate(toy_model, context, cast(1, int64))
"#,
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = gcc_link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("main"))
        .current_dir(&app_pkg)
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn reef_std_generate_is_lowered_through_host_lane() {
    let dir = tempdir().expect("tempdir");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("phase3i-generate-lowering");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "phase3i-generate-lowering"
version = "0.1.0"
compiler = "=0.1.7"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ path = "{}" }}
"#,
            std_pkg.display()
        ),
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Generate (KVCache, generate)

def keep_cache[p](cache: Option[KVCache[p]]) -> KVCache[p] =
  match cache with {
    | Some(value) => value
    | None => KVCache([])
  }

def toy_model[batch, seq](input_ids: tensor[batch, seq, int64], cache: Option[KVCache[tensor[kv, f32]]]) = {
  logits = pad_sequences_to([[0.05, 0.10, 0.90, 0.60]], cast(4, int64), 0.0)
  (logits, keep_cache(cache))
}

context = pad_sequences_to([[cast(1, int64), cast(2, int64)]], cast(2, int64), cast(0, int64))
greedy = generate(toy_model, context, cast(1, int64))
"#,
    );

    let prepared = prepare_program_for_file(&app_pkg.join("src/main.ch")).expect("prepare program");
    let prepared = prepared.expect("package mode should prepare");
    let deep = desugar_program(&prepared.decls);
    let checked = check_phase0e_program(&deep).expect("typecheck");
    let checked = chelis_effects::check_program(&checked).expect("effect check");
    let checked = check_linearity(&checked).expect("linearity");
    let compiled = lower_compiled_program(&checked);

    let host = compiled
        .host
        .expect("generate program should lower to host");
    assert!(
        host.globals
            .iter()
            .any(|binding| binding.name.ends_with("__greedy")),
        "expected greedy host binding, got {:?}",
        host.globals
            .iter()
            .map(|binding| binding.name.clone())
            .collect::<Vec<_>>()
    );
    let generate = host
        .functions
        .iter()
        .find(|function| function.name.ends_with("__generate"))
        .expect("generate host function");
    assert_eq!(
        generate.params.len(),
        3,
        "generate should keep callable + context + max_tokens params"
    );
    assert!(matches!(
        generate.params.first().map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Fn(_, _))
    ));
    assert!(matches!(
        generate.params.get(1).map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Tensor(_))
    ));
    assert!(matches!(
        generate.params.get(2).map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Int64)
    ));
    let generate_loop = host
        .functions
        .iter()
        .find(|function| function.name.ends_with("__generate_greedy_loop"))
        .expect("generate_greedy_loop host function");
    assert_eq!(
        generate_loop.params.len(),
        4,
        "generate_greedy_loop should keep callable + current + cache + remaining params"
    );
    assert!(matches!(
        generate_loop.params.first().map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Fn(_, _))
    ));
    assert!(matches!(
        generate_loop.params.get(1).map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Tensor(_))
    ));
    assert!(matches!(
        generate_loop.params.get(2).map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Option(_))
    ));
    assert!(matches!(
        generate_loop.params.get(3).map(|param| &param.ty),
        Some(chelis_ir::host::HostType::Int64)
    ));
}
