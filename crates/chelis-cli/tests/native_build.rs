//! Acceptance oracle for spec/08 §7 and [05-OBS-11] native artifacts.
use assert_cmd::Command;
use predicates::prelude::*;
use std::{fs, path::Path, process::Command as Process};
use tempfile::tempdir;

fn build(file: &Path, out: &Path) -> Command {
    let mut cmd = Command::cargo_bin("chelis").unwrap();
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1").args([
        "build",
        file.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ]);
    cmd
}

#[test]
fn cpu_build_produces_executable_and_matches_eval() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("answer.ch");
    let out = dir.path().join("out with spaces");
    fs::write(&file, "answer = add(20i64, 22i64)\n").unwrap();
    build(&file, &out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Built executable"));
    let result = Process::new(out.join("answer")).output().unwrap();
    assert!(result.status.success());
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", file.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(eval.status.success());
    assert_eq!(result.stdout, eval.stdout);
    assert!(out.join("answer.c").exists());
}

#[test]
fn cpu_library_has_no_entry_and_can_be_linked() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("functions.ch");
    let out = dir.path().join("out");
    fs::write(&file, "def twice(x: i64) -> i64 = add(x, x)\n").unwrap();
    build(&file, &out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Built static library"));
    assert!(out.join("libfunctions.a").exists());
    let driver = out.join("driver.c");
    fs::write(&driver, "#include \"chelis_runtime.h\"\n#include \"functions.h\"\nint main(void) { return chelis_fn_7477696365(21) == 42 ? 0 : 1; }\n").unwrap();
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let status = Process::new(toolchain.compiler)
        .arg(&driver)
        .arg(out.join("libfunctions.a"))
        .arg(out.join("libchelis_runtime.a"))
        .args(toolchain.link_flags)
        .arg("-o")
        .arg(out.join("driver"))
        .status()
        .unwrap();
    assert!(status.success());
    assert!(Process::new(out.join("driver")).status().unwrap().success());
}

#[test]
fn emit_c_does_not_require_native_tools() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("answer.ch");
    let out = dir.path().join("out");
    fs::write(&file, "answer = 42i64\n").unwrap();
    build(&file, &out)
        .arg("--emit-c")
        .env("CHELIS_CC", "/missing/compiler")
        .env("CHELIS_AR", "/missing/archiver")
        .assert()
        .success();
    assert!(out.join("answer.c").exists());
    assert!(!out.join("answer").exists());
    fs::write(&file, "def twice(x: i64) -> i64 = add(x, x)\n").unwrap();
    build(&file, &out)
        .arg("--emit-c")
        .env("CHELIS_CC", "/missing/compiler")
        .env("CHELIS_AR", "/missing/archiver")
        .assert()
        .success()
        .stdout(predicate::str::contains("Compile object:"))
        .stdout(predicate::str::contains("Archive:"));
    assert!(!out.join("libanswer.a").exists());
}

#[test]
fn missing_compiler_fails_with_install_guidance() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("answer.ch");
    fs::write(&file, "answer = 42i64\n").unwrap();
    build(&file, &dir.path().join("out"))
        .env("CHELIS_CC", "/missing/compiler")
        .assert()
        .failure()
        .stderr(predicate::str::contains("/missing/compiler"))
        .stderr(predicate::str::contains("install"));
}

#[test]
fn missing_default_compiler_names_tool_and_install_guidance() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("answer.ch");
    fs::write(&file, "answer = 42i64\n").unwrap();
    let compiler = if cfg!(target_os = "macos") {
        "clang"
    } else {
        "gcc"
    };
    build(&file, &dir.path().join("out"))
        .env_remove("CHELIS_CC")
        .env("PATH", dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "tool `{compiler}` was not found"
        )))
        .stderr(predicate::str::contains("install"));
}

#[cfg(unix)]
fn executable_script(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
#[test]
fn compiler_failure_and_missing_product_preserve_previous_artifact() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("answer.ch");
    let out = dir.path().join("out");
    fs::write(&file, "answer = 42i64\n").unwrap();
    build(&file, &out).assert().success();
    let original = fs::read(out.join("answer")).unwrap();
    // The build probes the compiler's identity and predefined macros before
    // compiling, so the broken compiler hands those probes to a real one and
    // breaks only the compile itself. The last case refuses the probe too.
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let real =
        chelis_backend_c::toolchain::verify_compiler(&toolchain.compiler, &toolchain.compile_flags)
            .unwrap()
            .path;
    let probes = format!(
        "case \" $* \" in *' --version '*|*' -dM '*) exec '{}' \"$@\";; esac\n",
        real.display()
    );
    let compiler = dir.path().join("broken compiler");
    for (body, diagnostic) in [
        (
            format!("{probes}echo deliberate-compiler-failure >&2; exit 37"),
            "deliberate-compiler-failure",
        ),
        (format!("{probes}exit 0"), "did not produce"),
        ("exit 37".to_string(), "--version failed"),
    ] {
        executable_script(&compiler, &body);
        build(&file, &out)
            .env("CHELIS_CC", &compiler)
            .assert()
            .failure()
            .stderr(predicate::str::contains(diagnostic))
            .stdout(predicate::str::contains("Built executable").not());
        assert_eq!(fs::read(out.join("answer")).unwrap(), original);
    }
}

#[cfg(unix)]
#[test]
fn archive_failure_preserves_previous_library() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("functions.ch");
    let out = dir.path().join("out");
    fs::write(&file, "def twice(x: i64) -> i64 = add(x, x)\n").unwrap();
    build(&file, &out).assert().success();
    let original = fs::read(out.join("libfunctions.a")).unwrap();
    let archiver = dir.path().join("broken archiver");
    for (body, diagnostic) in [
        (
            "echo deliberate-archive-failure >&2; exit 38",
            "deliberate-archive-failure",
        ),
        ("exit 0", "did not produce"),
    ] {
        executable_script(&archiver, body);
        build(&file, &out)
            .env("CHELIS_AR", &archiver)
            .assert()
            .failure()
            .stderr(predicate::str::contains(diagnostic));
        assert_eq!(fs::read(out.join("libfunctions.a")).unwrap(), original);
    }
}

#[test]
fn deep_build_and_explicit_output_preserve_source_identity() {
    let dir = tempdir().unwrap();
    let surf = dir.path().join("answer.ch");
    let deep = dir.path().join("answer.dp");
    let source = dir.path().join("nested/custom.c");
    fs::write(&surf, "answer = 42i64\n").unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .arg("deep")
        .arg(&surf)
        .output()
        .unwrap();
    assert!(output.status.success());
    fs::write(&deep, output.stdout).unwrap();
    build(&deep, &source).assert().success();
    assert!(source.exists());
    let run = Process::new(source.with_extension("")).output().unwrap();
    assert!(run.status.success());
    assert_eq!(run.stdout, b"answer = 42\n");
}

#[cfg(unix)]
#[test]
fn hip_and_metal_build_commands_include_support_and_ordered_flags() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("tensor.ch");
    fs::write(
        &file,
        "out = add(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    )
    .unwrap();
    // Native tools run with an allowlisted environment, so the recording
    // compiler finds its log beside itself rather than through a variable.
    let compiler = dir.path().join("recording compiler");
    let log = dir.path().join("recording compiler.argv");
    executable_script(
        &compiler,
        r#"
if [ "$1" = "--version" ]; then echo 'clang version test'; exit 0; fi
printf '%s\n' "$@" > "$0.argv"
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then shift; cp /usr/bin/true "$1"; exit 0; fi
    shift
done
exit 1"#,
    );
    for (target, override_var) in [("hip", "CHELIS_HIPCC"), ("metal", "CHELIS_METAL_CXX")] {
        let out = dir.path().join(target);
        build(&file, &out)
            .args(["--target", target])
            .env(override_var, &compiler)
            .assert()
            .success();
        let argv = fs::read_to_string(&log).unwrap();
        assert!(argv.contains("-ffp-contract=off\n"), "{argv}");
        assert!(argv.contains("libchelis_runtime.a\n"), "{argv}");
        if target == "hip" {
            assert!(argv.contains("chelis_device_owner.cpp\n"), "{argv}");
        } else {
            assert!(
                fs::read_to_string(out.join("tensor_metal.mm"))
                    .unwrap()
                    .contains("int main(void)")
            );
        }
        fs::write(
            &file,
            "def combine(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n",
        )
        .unwrap();
        let output = build(&file, &out)
            .args(["--target", target])
            .env(override_var, &compiler)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        if target == "hip" {
            assert!(stdout.contains("-lhiprtc"), "{stdout}");
        } else {
            assert!(stdout.contains("-framework Metal"), "{stdout}");
            assert!(stdout.contains("-framework Foundation"), "{stdout}");
        }
        fs::write(
            &file,
            "out = add(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
        )
        .unwrap();
        build(&file, &out)
            .args(["--target", target])
            .env(override_var, "/missing/gpu-compiler")
            .assert()
            .failure();
        build(&file, &out)
            .args(["--target", target, "--emit-c"])
            .env(override_var, "/missing/gpu-compiler")
            .assert()
            .success();
    }
}

#[test]
fn cpu_observation_corpus_matches_eval_exactly() {
    let programs = [
        "wide = 9007199254740993i64\nflag = true\n",
        "out = (42i64, true, 2.5f64)\n",
        "out = add(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
        "def main() -> i64 = 42i64\n",
        "def arithmetic(a: f64, b: f64, c: f64) -> f64 = add(mul(a, b), c)\nout = arithmetic(134217729.0f64, 134217727.0f64, -18014398509481984.0f64)\n",
    ];
    for (index, program) in programs.iter().enumerate() {
        let dir = tempdir().unwrap();
        let file = dir.path().join(format!("program_{index}.ch"));
        let out = dir.path().join("out");
        fs::write(&file, program).unwrap();
        build(&file, &out).assert().success();
        let actual = Process::new(out.join(format!("program_{index}")))
            .output()
            .unwrap();
        assert!(actual.status.success(), "{actual:?}");
        let expected = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", file.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(expected.status.success());
        chelis_types::agreement::compare_exact_observations(
            &format!("native program {index}"),
            std::str::from_utf8(&expected.stdout).unwrap(),
            std::str::from_utf8(&actual.stdout).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn invalid_program_fails_before_native_compilation() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("bad.ch");
    let out = dir.path().join("out");
    fs::write(&file, "out = add(1i64, 1.0f32)\n").unwrap();
    build(&file, &out)
        .env("CHELIS_CC", "/missing/compiler")
        .assert()
        .failure()
        .stderr(predicate::str::contains("native compile/link").not());
    assert!(!out.exists());
}

#[test]
fn missing_archiver_does_not_fall_back() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("functions.ch");
    let out = dir.path().join("out");
    fs::write(&file, "def twice(x: i64) -> i64 = add(x, x)\n").unwrap();
    build(&file, &out)
        .env("CHELIS_AR", "/missing/archiver")
        .assert()
        .failure()
        .stderr(predicate::str::contains("/missing/archiver"))
        .stderr(predicate::str::contains("install"));
    assert!(!out.join("libfunctions.a").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_builds_native_library_and_host_executable() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("tensor_lib.ch");
    let out = dir.path().join("out");
    fs::write(
        &file,
        "def combine(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n",
    )
    .unwrap();
    build(&file, &out)
        .args(["--target", "metal"])
        .assert()
        .success();
    assert!(out.join("libtensor_lib_metal.a").exists());
    fs::write(&file, "answer = 42i64\n").unwrap();
    build(&file, &out)
        .args(["--target", "metal"])
        .assert()
        .success();
    let run = Process::new(out.join("tensor_lib_metal")).output().unwrap();
    assert!(run.status.success());
    assert_eq!(run.stdout, b"answer = 42\n");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires a real Metal device; run locally with --ignored"]
fn metal_native_library_executes_tensor_kernel() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("tensor_lib.ch");
    let out = dir.path().join("out");
    fs::write(
        &file,
        "def combine(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n",
    )
    .unwrap();
    build(&file, &out)
        .args(["--target", "metal"])
        .assert()
        .success();
    fs::write(
        out.join("driver.mm"),
        r#"
#include "chelis_runtime.h"
#include "tensor_lib_metal.h"
int main(void) {
    int64_t extent = 2;
    chelis_tensor *inputs[2];
    for (int i = 0; i < 2; ++i) {
        inputs[i] = chelis_alloc(1, &extent, CHELIS_DTYPE_F32);
        chelis_tensor_write *guard = chelis_tensor_begin_write(inputs[i]);
        chelis_fill_scalar(guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0x3fc00000));
        chelis_tensor_end_write(guard);
    }
    chelis_tensor *outputs[1] = {0};
    tensor_lib(inputs, 2, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    bool ok = ((const float *)view.data)[0] == 3.0f && ((const float *)view.data)[1] == 3.0f;
    chelis_tensor_release(inputs[0]);
    chelis_tensor_release(inputs[1]);
    chelis_tensor_release(outputs[0]);
    return ok ? 0 : 1;
}
"#,
    )
    .unwrap();
    assert!(
        Process::new("clang++")
            .arg(out.join("driver.mm"))
            .arg(out.join("libtensor_lib_metal.a"))
            .arg(out.join("libchelis_runtime.a"))
            .args([
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "Accelerate"
            ])
            .arg("-o")
            .arg(out.join("driver"))
            .status()
            .unwrap()
            .success()
    );
    assert!(Process::new(out.join("driver")).status().unwrap().success());
}

#[test]
fn native_artifacts_cannot_overwrite_input_or_carried_runtime() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("program");
    let original = b"answer = 42i64\n";
    fs::write(&file, original).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["build", "program"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("overwrite input"));
    assert_eq!(fs::read(&file).unwrap(), original);
    let output = dir.path().join("out/libchelis_runtime.a.c");
    build(&file, &output)
        .assert()
        .failure()
        .stderr(predicate::str::contains("runtime archive"));
    let runtime = fs::read(dir.path().join("out/libchelis_runtime.a")).unwrap();
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!("{:x}", Sha256::digest(runtime)),
        chelis_runtime_bundle::carried_sha256().unwrap()
    );
}

#[cfg(unix)]
#[test]
fn hard_linked_generated_source_or_header_cannot_overwrite_input() {
    for alias in ["custom.c", "custom.h"] {
        let dir = tempdir().unwrap();
        let file = dir.path().join("original.ch");
        let out = dir.path().join("out");
        let original = b"answer = 42i64\n";
        fs::write(&file, original).unwrap();
        fs::create_dir(&out).unwrap();
        fs::hard_link(&file, out.join(alias)).unwrap();
        build(&file, &out.join("custom.c"))
            .assert()
            .failure()
            .stderr(predicate::str::contains("overwrite input"));
        assert_eq!(fs::read(&file).unwrap(), original);
    }
}

/// chelis#2962: the C compiler runs in an allowlisted environment, so variables
/// a driver reads cannot change the build, and a compiler that does not apply
/// the pinned floating-point profile is refused rather than recorded.
#[cfg(unix)]
#[test]
fn c_build_ignores_compiler_environment_and_refuses_profile_changing_wrapper() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempdir().unwrap();
    let file = dir.path().join("growth.ch");
    fs::write(&file, "y = exp(cast(0.3, f32))\n").unwrap();
    let shadow = dir.path().join("shadow");
    fs::create_dir(&shadow).unwrap();
    fs::write(shadow.join("math.h"), "#error shadow math.h was used\n").unwrap();
    let run = |out: &Path| {
        let result = Process::new(out.join("growth")).output().unwrap();
        assert!(result.status.success());
        result.stdout
    };

    let clean = dir.path().join("clean");
    build(&file, &clean)
        .assert()
        .success()
        .stdout(predicate::str::contains("Compiler: "));
    let hostile = dir.path().join("hostile");
    build(&file, &hostile)
        .env("CCC_OVERRIDE_OPTIONS", "+-ffast-math +-O0")
        .env("NIX_CFLAGS_COMPILE", "-ffast-math -O0")
        .env("CPATH", &shadow)
        .env("C_INCLUDE_PATH", &shadow)
        .assert()
        .success();
    assert_eq!(run(&clean), run(&hostile));

    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let real =
        chelis_backend_c::toolchain::verify_compiler(&toolchain.compiler, &toolchain.compile_flags)
            .unwrap()
            .path;
    let wrapper = dir.path().join("fast-cc");
    fs::write(
        &wrapper,
        format!("#!/bin/sh\nexec '{}' \"$@\" -ffast-math\n", real.display()),
    )
    .unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    build(&file, &dir.path().join("fast"))
        .env("CHELIS_CC", &wrapper)
        .assert()
        .failure()
        .stderr(predicate::str::contains("__FAST_MATH__"));
}
