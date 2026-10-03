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
    // The build probes the compiler's identity, predefined macros, and
    // floating-point canary before compiling, so the broken compiler hands
    // those probes to a real one and breaks only the compile itself. The last
    // case refuses the probe too.
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(Default::default());
    let real = chelis_backend_c::toolchain::verify_compiler(
        &toolchain.compiler,
        &toolchain.compile_flags,
        &toolchain.link_flags,
    )
    .unwrap()
    .path;
    let probes = format!(
        "case \" $* \" in *' --version '*|*' -dM '*|*chelis-compiler-canary*) exec '{}' \"$@\";; esac\n",
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
    if [ "$1" = "-o" ]; then
        shift
        printf '#!/bin/sh\nexit 0\n' > "$1"
        chmod 755 "$1"
        exit 0
    fi
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
    let real = chelis_backend_c::toolchain::verify_compiler(
        &toolchain.compiler,
        &toolchain.compile_flags,
        &toolchain.link_flags,
    )
    .unwrap()
    .path;
    // No predefined macro reveals contraction, so the floating-point canary
    // refuses that wrapper. The x86-64 baseline has no fused multiply-add, so
    // there the wrapper also adds the `-mfma` a `-march=native` one would.
    let contract = if cfg!(target_arch = "x86_64") {
        "-mfma -ffp-contract=fast"
    } else {
        "-ffp-contract=fast"
    };
    for (name, extra, diagnostic) in [
        ("fast-cc", "-ffast-math", "__FAST_MATH__"),
        ("contract-cc", contract, "floating-point canary disagrees"),
    ] {
        let wrapper = dir.path().join(name);
        fs::write(
            &wrapper,
            format!("#!/bin/sh\nexec '{}' \"$@\" {extra}\n", real.display()),
        )
        .unwrap();
        fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
        build(&file, &dir.path().join(format!("{name}-out")))
            .env("CHELIS_CC", &wrapper)
            .assert()
            .failure()
            .stderr(predicate::str::contains(diagnostic));
    }
}

/// One float storage width of the NaN finalization oracle: its Surf name, C
/// storage type, dtype macro, canonical quiet NaN, and the input classes
/// (positive payload NaN, negative NaN, signaling NaN, one, minus one, +inf,
/// zero) as stored bits.
struct NanWidth {
    name: &'static str,
    storage: &'static str,
    dtype: &'static str,
    canonical: u64,
    payload: u64,
    negative: u64,
    signaling: u64,
    one: u64,
    minus_one: u64,
    inf: u64,
    zero: u64,
}

const NAN_WIDTHS: [NanWidth; 4] = [
    NanWidth {
        name: "f16",
        storage: "uint16_t",
        dtype: "CHELIS_DTYPE_F16",
        canonical: 0x7e00,
        payload: 0x7e55,
        negative: 0xfe55,
        signaling: 0x7c01,
        one: 0x3c00,
        minus_one: 0xbc00,
        inf: 0x7c00,
        zero: 0,
    },
    NanWidth {
        name: "bf16",
        storage: "uint16_t",
        dtype: "CHELIS_DTYPE_BF16",
        canonical: 0x7fc0,
        payload: 0x7fc5,
        negative: 0xffe5,
        signaling: 0x7f81,
        one: 0x3f80,
        minus_one: 0xbf80,
        inf: 0x7f80,
        zero: 0,
    },
    NanWidth {
        name: "f32",
        storage: "uint32_t",
        dtype: "CHELIS_DTYPE_F32",
        canonical: 0x7fc0_0000,
        payload: 0x7fc1_2345,
        negative: 0xffc5_4321,
        signaling: 0x7f81_2345,
        one: 0x3f80_0000,
        minus_one: 0xbf80_0000,
        inf: 0x7f80_0000,
        zero: 0,
    },
    NanWidth {
        name: "f64",
        storage: "uint64_t",
        dtype: "CHELIS_DTYPE_F64",
        canonical: 0x7ff8_0000_0000_0000,
        payload: 0x7ff8_0000_0000_0055,
        negative: 0xfff8_abcd_1234_5678,
        signaling: 0x7ff0_1234_5678_9abc,
        one: 0x3ff0_0000_0000_0000,
        minus_one: 0xbff0_0000_0000_0000,
        inf: 0x7ff0_0000_0000_0000,
        zero: 0,
    },
];

impl NanWidth {
    fn is_nan(&self, bits: u64) -> bool {
        let exponent = self.inf;
        bits & exponent == exponent && bits & !(exponent | self.sign()) != 0
    }

    fn sign(&self) -> u64 {
        match self.name {
            "f16" | "bf16" => 0x8000,
            "f32" => 0x8000_0000,
            _ => 0x8000_0000_0000_0000,
        }
    }

    fn hex_digits(&self) -> usize {
        match self.name {
            "f16" | "bf16" => 4,
            "f32" => 8,
            _ => 16,
        }
    }

    /// One case per input class: three NaN encodings on either side and the
    /// non-NaN operands from which invalid operations produce a NaN.
    fn unary_cases(&self) -> Vec<Vec<u64>> {
        [
            self.payload,
            self.negative,
            self.signaling,
            self.minus_one,
            self.inf,
        ]
        .map(|x| vec![x])
        .to_vec()
    }

    fn binary_cases(&self) -> Vec<Vec<u64>> {
        vec![
            vec![self.payload, self.one],
            vec![self.negative, self.one],
            vec![self.signaling, self.one],
            vec![self.one, self.signaling],
            vec![self.inf, self.inf],
            vec![self.zero, self.zero],
        ]
    }
}

/// The arity of a float builtin in the host lane's NaN inventory. A builtin
/// added to the inventory fails here until the oracle knows how to call it.
/// Link `out/driver.c` against a `chelis build` static library and the
/// runtime archive it staged, as `chelis build` links an executable: the
/// profile for the requirements every generated unit declares (OpenMP wanted,
/// so gcc links `-fopenmp` for the library's parallel regions) through the
/// shared `toolchain::link_args`, in the tools' cleared environment.
fn link_static_library_driver(out: &Path, library: &str, needs_blas: bool) -> bool {
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let driver = out.join("driver.c");
    let library = out.join(library);
    let runtime = out.join("libchelis_runtime.a");
    let product = out.join("driver");
    chelis_backend_c::toolchain::tool_command(&toolchain.compiler)
        .args(chelis_backend_c::toolchain::link_args(
            &toolchain.compile_flags,
            &[driver.as_os_str(), library.as_os_str(), runtime.as_os_str()],
            &toolchain.link_flags,
            product.as_os_str(),
        ))
        .status()
        .unwrap()
        .success()
}

fn nan_inventory_arity(name: &str) -> usize {
    match name {
        "add" | "sub" | "mul" | "div" | "floor_div" | "min" | "max" | "min_elem" | "max_elem" => 2,
        "neg" | "sqrt" | "exp" | "log" | "sin" | "cos" | "tan" | "atan" | "tanh" | "relu"
        | "sigmoid" | "silu" | "gelu" | "floor" | "ceil" | "round" | "recip" | "abs" => 1,
        other => panic!("classify the arity of new float builtin `{other}` in this oracle"),
    }
}

/// Builtins the host lane can emit but the checker admits from no source
/// program; the oracle requires eval to reject them, so admitting one makes
/// the oracle cover it.
const NAN_INVENTORY_UNREACHABLE: [&str; 2] = ["min", "max"];

fn nan_eval_bits(source: &str, inputs: &[(&str, &NanWidth, Vec<u64>)], out: &NanWidth) -> Vec<u64> {
    let shaped: Vec<_> = inputs
        .iter()
        .map(|(name, width, bits)| (*name, *width, vec![bits.len() as i64], bits.clone()))
        .collect();
    nan_eval_shaped_bits(source, &shaped, out)
}

/// [`nan_eval_bits`] with each input's shape given; the result is flattened
/// in row-major order.
fn nan_eval_shaped_bits(
    source: &str,
    inputs: &[(&str, &NanWidth, Vec<i64>, Vec<u64>)],
    out: &NanWidth,
) -> Vec<u64> {
    let bindings = inputs
        .iter()
        .map(|(name, width, shape, bits)| {
            let bits: Vec<String> = bits
                .iter()
                .map(|bits| format!("{bits:0digits$x}", digits = width.hex_digits()))
                .collect();
            (
                name.to_string(),
                chelis_compiler_api::schema::TensorValue {
                    shape: shape.clone(),
                    data: serde_json::from_value(
                        serde_json::json!({"dtype": width.name, "bits": bits}),
                    )
                    .unwrap(),
                },
            )
        })
        .collect();
    let result = chelis_compiler_api::compiler::eval_selected(
        chelis_compiler_api::schema::EvalRequest {
            source_kind: chelis_compiler_api::schema::SourceKind::Surf,
            source: source.to_string(),
            bindings,
        },
        &["main".to_string()],
    )
    .unwrap_or_else(|error| panic!("evaluate {source}: {error:?}"));
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("main"))
        .expect("main root");
    let value = serde_json::to_value(&root.value).unwrap();
    let data = &value["value"]["data"];
    assert_eq!(data["dtype"], out.name, "{value}");
    data["bits"]
        .as_array()
        .expect("tensor bits")
        .iter()
        .map(|bits| u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap())
        .collect()
}

/// One oracle row: an operation at one input and output width, called as a
/// scalar `def` (host scalar lane) and a tensor `def` (typed-DAG lane).
struct NanRow {
    label: String,
    call: String,
    input: &'static NanWidth,
    output: &'static NanWidth,
    arity: usize,
    cases: Vec<Vec<u64>>,
    bit_preserving: bool,
}

/// chelis#2957 round 2, chelis#2964: [04-NUM-2] in a built static library.
/// Every float builtin of the host lane's NaN inventory
/// (`chelis_backend_c::host_builtin_nan_inventory`) and every float-to-float
/// conversion, at f16, bf16, f32 and f64, through the scalar and the tensor
/// C ABI, agrees bit for bit with eval on payload, negative and signaling
/// NaNs and on the non-NaN operands of invalid operations. Canonicalizing
/// rows produce only the canonical quiet NaN; bit-preserving rows only an
/// input's NaN.
#[test]
fn every_float_result_finalizes_nan_like_eval_through_the_static_library_abi() {
    use chelis_backend_c::fp_env::NanFinalization;
    let mut rows = Vec::new();
    for (name, finalization) in chelis_backend_c::host_builtin_nan_inventory() {
        let Some(finalization) = finalization else {
            continue;
        };
        let arity = nan_inventory_arity(name);
        if NAN_INVENTORY_UNREACHABLE.contains(&name) {
            let params = if arity == 1 { "x" } else { "x, y" };
            let source = format!("def main(x: f32, y: f32) -> f32 = {name}({params})\n");
            let rejected = chelis_compiler_api::compiler::eval_selected(
                chelis_compiler_api::schema::EvalRequest {
                    source_kind: chelis_compiler_api::schema::SourceKind::Surf,
                    source,
                    bindings: Default::default(),
                },
                &["main".to_string()],
            );
            assert!(
                rejected.is_err(),
                "`{name}` is now admitted; remove it from the unreachable list"
            );
            continue;
        }
        for width in &NAN_WIDTHS {
            rows.push(NanRow {
                label: format!("{name}_{}", width.name),
                call: format!("{name}({})", if arity == 1 { "x" } else { "x, y" }),
                input: width,
                output: width,
                arity,
                cases: if arity == 1 {
                    width.unary_cases()
                } else {
                    width.binary_cases()
                },
                bit_preserving: finalization == NanFinalization::BitPreserving,
            });
        }
    }
    for source in &NAN_WIDTHS {
        for target in &NAN_WIDTHS {
            if source.name != target.name {
                rows.push(NanRow {
                    label: format!("cast_{}_{}", source.name, target.name),
                    call: format!("cast(x, {})", target.name),
                    input: source,
                    output: target,
                    arity: 1,
                    cases: source.unary_cases(),
                    bit_preserving: false,
                });
            }
        }
    }

    let mut program = String::new();
    let mut harness = String::from(
        "#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\n\
         #include \"chelis_runtime.h\"\n#include \"nan_oracle.h\"\n\
         static float f32_of(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }\n\
         static double f64_of(uint64_t b) { double x; memcpy(&x, &b, 8); return x; }\n\
         static uint32_t f32_bits(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }\n\
         static uint64_t f64_bits(double x) { uint64_t b; memcpy(&b, &x, 8); return b; }\n\
         int main(void) {\n",
    );
    let mangle = |name: &str| -> String {
        let hex: String = name.bytes().map(|byte| format!("{byte:02x}")).collect();
        format!("chelis_fn_{hex}")
    };
    let scalar_in = |width: &NanWidth, bits: u64| match width.name {
        "f32" => format!("f32_of(UINT32_C({bits:#x}))"),
        "f64" => format!("f64_of(UINT64_C({bits:#x}))"),
        _ => format!("(uint16_t){bits:#x}"),
    };
    let scalar_out = |width: &NanWidth, call: String| match width.name {
        "f32" => format!("(unsigned long long)f32_bits({call})"),
        "f64" => format!("(unsigned long long)f64_bits({call})"),
        _ => format!("(unsigned long long){call}"),
    };
    for row in &rows {
        let (input, output) = (row.input, row.output);
        let n = row.cases.len();
        let params = ["x", "y"][..row.arity]
            .iter()
            .map(|param| format!("{param}: {}", input.name))
            .collect::<Vec<_>>()
            .join(", ");
        let tensor_params = ["x", "y"][..row.arity]
            .iter()
            .map(|param| format!("{param}: tensor[{n}, {}]", input.name))
            .collect::<Vec<_>>()
            .join(", ");
        program.push_str(&format!(
            "def s_{label}({params}) -> {out} = {call}\n\
             def t_{label}({tensor_params}) -> tensor[{n}, {out}] = {call}\n",
            label = row.label,
            out = output.name,
            call = row.call,
        ));
        for (case, operands) in row.cases.iter().enumerate() {
            let args = operands
                .iter()
                .map(|bits| scalar_in(input, *bits))
                .collect::<Vec<_>>()
                .join(", ");
            let call = format!("{}({args})", mangle(&format!("s_{}", row.label)));
            harness.push_str(&format!(
                "    printf(\"s_{} {case} %llx\\n\", {});\n",
                row.label,
                scalar_out(output, call)
            ));
        }
        let mut tensors = Vec::new();
        for operand in 0..row.arity {
            let data = row
                .cases
                .iter()
                .map(|case| format!("{:#x}", case[operand]))
                .collect::<Vec<_>>()
                .join(", ");
            let var = format!("t_{}_{operand}", row.label);
            harness.push_str(&format!(
                "    chelis_tensor *{var};\n    {{ static const {storage} d[{n}] = {{ {data} }}; static const int64_t shape[1] = {{ {n} }};\n      \
                 {var} = chelis_tensor_entry_borrow(1, shape, {dtype}, d, sizeof d); }}\n",
                storage = input.storage,
                dtype = input.dtype,
            ));
            tensors.push(var);
        }
        harness.push_str(&format!(
            "    {{ chelis_tensor *r = {}({}); const {} *o = (const {} *)chelis_tensor_read_view(r).data;\n      \
             for (int i = 0; i < {n}; ++i) printf(\"t_{} %d %llx\\n\", i, (unsigned long long)o[i]); }}\n",
            mangle(&format!("t_{}", row.label)),
            tensors.join(", "),
            output.storage,
            output.storage,
            row.label,
        ));
    }
    harness.push_str("    return 0;\n}\n");

    let dir = tempdir().unwrap();
    let file = dir.path().join("nan_oracle.ch");
    let out = dir.path().join("out");
    fs::write(&file, &program).unwrap();
    build(&file, &out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Built static library"));
    let driver = out.join("driver.c");
    fs::write(&driver, &harness).unwrap();
    assert!(
        link_static_library_driver(&out, "libnan_oracle.a", false),
        "the oracle driver must link"
    );
    let run = Process::new(out.join("driver")).output().unwrap();
    assert!(run.status.success(), "{run:?}");
    let stdout = String::from_utf8(run.stdout).unwrap();
    let c_bits = |lane: &str, label: &str, case: usize| -> u64 {
        let prefix = format!("{lane}_{label} {case} ");
        let line = stdout
            .lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("missing {prefix}"));
        u64::from_str_radix(line, 16).unwrap()
    };

    let mut failures = Vec::new();
    for row in &rows {
        let n = row.cases.len();
        let tensor_params = ["x", "y"][..row.arity]
            .iter()
            .map(|param| format!("{param}: tensor[{n}, {}]", row.input.name))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "def main({tensor_params}) -> tensor[{n}, {}] = {}\n",
            row.output.name, row.call
        );
        let inputs: Vec<_> = (0..row.arity)
            .map(|operand| {
                (
                    ["x", "y"][operand],
                    row.input,
                    row.cases.iter().map(|case| case[operand]).collect(),
                )
            })
            .collect();
        let eval = nan_eval_bits(&source, &inputs, row.output);
        for (case, operands) in row.cases.iter().enumerate() {
            let expected = eval[case];
            if row.output.is_nan(expected) {
                let admitted = if row.bit_preserving {
                    operands.contains(&expected)
                } else {
                    expected == row.output.canonical
                };
                if !admitted {
                    failures.push(format!(
                        "eval {} {operands:x?} gave {expected:#x}",
                        row.label
                    ));
                }
            }
            for lane in ["s", "t"] {
                let got = c_bits(lane, &row.label, case);
                if got != expected {
                    failures.push(format!(
                        "{lane}_{} {operands:x?}: C {got:#x}, eval {expected:#x}",
                        row.label
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// One kernel row of the NaN finalization oracle: a tensor `def` whose result
/// an emitter writes through its own store (a reduction's total, a window, a
/// BLAS call, a draw, a selection or a data movement), with each input's
/// shape and stored bits, the helper `def` its call names, if any, and the
/// [04-NUM-2] finalization `fp_env::risc_nan_finalization` gives the atom the
/// row exercises (`None` for data movement). A gradient row's result is the
/// gradient's adjoint accumulation, which is arithmetic, so it expects the
/// canonical NaN whatever the atom acting inside it.
struct NanKernelRow {
    label: String,
    width: &'static NanWidth,
    params: Vec<(&'static str, Vec<i64>, Vec<u64>)>,
    output: Vec<i64>,
    call: String,
    helper: Option<String>,
    finalization: Option<chelis_backend_c::fp_env::NanFinalization>,
    gradient: bool,
}

/// How the kernel oracle covers one RISC atom.
enum NanAtomCoverage {
    /// Kernel rows below, built by [`nan_atom_rows`].
    Rows,
    /// The builtin oracle above covers it through the host builtin inventory.
    BuiltinInventory,
    /// The builtin oracle above covers it through its cast rows.
    CastRows,
    /// The atom yields no float value: integer, bool, key or shape results.
    NoFloatValue,
}

/// One representative operation for each semantic RISC atom, and how the
/// oracle covers it. The match is exhaustive over the atom universe, so a new
/// atom does not compile until it is classified here; the representative's
/// disposition is checked against the atom, and its NaN finalization is read
/// from `fp_env::risc_nan_finalization`, never restated.
fn nan_atom_coverage(
    atom: chelis_ir::dag::RiscAtomIdentity,
) -> (chelis_ir::dag::RiscOp, NanAtomCoverage) {
    use NanAtomCoverage::{BuiltinInventory, CastRows, NoFloatValue, Rows};
    use chelis_ir::dag::{
        ComparisonKind, DimExpr, ExtremaKind, ExtremaOperand, KeyBranch, LogicalKind,
        ReduceWindowKind, RiscAtomIdentity as Id, RiscOp, RtDim, UniformBound,
    };
    use chelis_types::{BitwiseKind, types::Prim};
    let window = |reducer| RiscOp::ReduceWindow {
        reducer,
        window_shape: vec![2],
        strides: vec![1],
    };
    match atom {
        Id::Add => (RiscOp::Add, BuiltinInventory),
        Id::Sub => (RiscOp::Sub, BuiltinInventory),
        Id::Mul => (RiscOp::Mul, BuiltinInventory),
        Id::Div => (RiscOp::Div, BuiltinInventory),
        Id::FloorDiv => (RiscOp::FloorDiv, BuiltinInventory),
        Id::MaxElem => (RiscOp::MaxElem, BuiltinInventory),
        Id::MinElem => (RiscOp::MinElem, BuiltinInventory),
        Id::Relu => (RiscOp::Relu, BuiltinInventory),
        Id::Neg => (RiscOp::Neg, BuiltinInventory),
        Id::Exp => (RiscOp::Exp, BuiltinInventory),
        Id::Log => (RiscOp::Log, BuiltinInventory),
        Id::Sin => (RiscOp::Sin, BuiltinInventory),
        Id::Sqrt => (RiscOp::Sqrt, BuiltinInventory),
        Id::Cos => (RiscOp::Cos, BuiltinInventory),
        Id::Tan => (RiscOp::Tan, BuiltinInventory),
        Id::Atan => (RiscOp::Atan, BuiltinInventory),
        Id::Tanh => (RiscOp::Tanh, BuiltinInventory),
        Id::Abs => (RiscOp::Abs, BuiltinInventory),
        Id::Floor => (RiscOp::Floor, BuiltinInventory),
        Id::Ceil => (RiscOp::Ceil, BuiltinInventory),
        Id::Round => (RiscOp::Round, BuiltinInventory),
        Id::Recip => (RiscOp::Recip, BuiltinInventory),
        Id::Cast => (
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            CastRows,
        ),
        Id::ReluAdjoint => (RiscOp::ReluAdjoint, Rows),
        Id::ExtremaAdjoint => (
            RiscOp::ExtremaAdjoint {
                kind: ExtremaKind::Max,
                operand: ExtremaOperand::Left,
            },
            Rows,
        ),
        Id::Where => (RiscOp::Where, Rows),
        Id::Dropout => (RiscOp::Dropout, Rows),
        Id::DropoutReplay => (RiscOp::DropoutReplay, Rows),
        Id::UniformBoundAdjoint => (
            RiscOp::UniformBoundAdjoint {
                bound: UniformBound::Low,
            },
            Rows,
        ),
        Id::Sum => (
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            Rows,
        ),
        Id::MaxReduce => (RiscOp::MaxReduce { axis: 0 }, Rows),
        Id::MinReduce => (RiscOp::MinReduce { axis: 0 }, Rows),
        Id::ProdReduce => (RiscOp::ProdReduce { axis: 0 }, Rows),
        Id::ReduceWindowMax => (window(ReduceWindowKind::Max), Rows),
        Id::ReduceWindowMin => (window(ReduceWindowKind::Min), Rows),
        Id::ReduceWindowSum => (window(ReduceWindowKind::Sum), Rows),
        Id::ReduceWindowMean => (window(ReduceWindowKind::Mean), Rows),
        Id::ReduceWindowGrad => (
            RiscOp::ReduceWindowGrad {
                reducer: ReduceWindowKind::Sum,
                window_shape: vec![2],
                strides: vec![1],
            },
            Rows,
        ),
        Id::Reshape => (
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(3)],
            },
            Rows,
        ),
        Id::Permute => (RiscOp::Permute { axes: vec![1, 0] }, Rows),
        Id::Expand => (
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(2),
            },
            Rows,
        ),
        Id::Pad => (
            RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(1), RtDim::Lit(0))]),
            Rows,
        ),
        Id::Shrink => (
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(3))],
            },
            Rows,
        ),
        Id::Stride => (
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            Rows,
        ),
        Id::Matmul => (
            RiscOp::BlasMatmul {
                batch_dims: vec![],
                m: DimExpr::Concrete(2),
                n: DimExpr::Concrete(2),
                k: DimExpr::Concrete(2),
                accumulator: Prim::F32,
            },
            Rows,
        ),
        Id::Gather => (
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
            Rows,
        ),
        Id::Scatter => (
            RiscOp::ScatterAdd {
                axis: 0,
                batch_rank: 0,
            },
            Rows,
        ),
        Id::ScatterReplace => (
            RiscOp::Scatter {
                axis: 0,
                batch_rank: 0,
            },
            Rows,
        ),
        Id::ScatterElements => (RiscOp::ScatterElements { axis: 0 }, Rows),
        Id::Count => (RiscOp::Count { axes: vec![0] }, NoFloatValue),
        Id::TruncDiv => (RiscOp::TruncDiv, NoFloatValue),
        Id::Mod => (RiscOp::Mod, NoFloatValue),
        Id::BitAnd => (RiscOp::Bitwise(BitwiseKind::And), NoFloatValue),
        Id::BitOr => (RiscOp::Bitwise(BitwiseKind::Or), NoFloatValue),
        Id::BitXor => (RiscOp::Bitwise(BitwiseKind::Xor), NoFloatValue),
        Id::ShiftLeft => (RiscOp::Bitwise(BitwiseKind::ShiftLeft), NoFloatValue),
        Id::ShiftRight => (RiscOp::Bitwise(BitwiseKind::ShiftRight), NoFloatValue),
        Id::CmpLt => (RiscOp::Compare(ComparisonKind::CmpLt), NoFloatValue),
        Id::Lt => (RiscOp::Compare(ComparisonKind::Lt), NoFloatValue),
        Id::Eq => (RiscOp::Compare(ComparisonKind::Eq), NoFloatValue),
        Id::Neq => (RiscOp::Compare(ComparisonKind::Neq), NoFloatValue),
        Id::Gt => (RiscOp::Compare(ComparisonKind::Gt), NoFloatValue),
        Id::Gte => (RiscOp::Compare(ComparisonKind::Gte), NoFloatValue),
        Id::Lte => (RiscOp::Compare(ComparisonKind::Lte), NoFloatValue),
        Id::And => (RiscOp::Logical(LogicalKind::And), NoFloatValue),
        Id::Or => (RiscOp::Logical(LogicalKind::Or), NoFloatValue),
        Id::Not => (RiscOp::Logical(LogicalKind::Not), NoFloatValue),
        Id::GuardedFail => (
            RiscOp::GuardedFail {
                message: String::new(),
                trap_on_true: true,
            },
            NoFloatValue,
        ),
        // A uniform draw's bits come from the key, not from a float operand.
        Id::UniformLike => (RiscOp::UniformLike, NoFloatValue),
        Id::KeyFromSeed => (RiscOp::KeyFromSeed, NoFloatValue),
        Id::SplitKey => (
            RiscOp::Split {
                branch: KeyBranch::Left,
            },
            NoFloatValue,
        ),
        Id::SplitKeys => (
            RiscOp::SplitN {
                count: RtDim::Lit(2),
            },
            NoFloatValue,
        ),
        Id::FoldIn => (RiscOp::FoldIn, NoFloatValue),
        Id::ArgmaxReduce => (RiscOp::Argmax { axis: 0 }, NoFloatValue),
        Id::ArgminReduce => (RiscOp::Argmin { axis: 0 }, NoFloatValue),
        Id::Shape => (RiscOp::Shape { axis: 0 }, NoFloatValue),
        Id::CastTrunc => (
            RiscOp::CastTrunc {
                new_precision: Prim::Int32,
            },
            NoFloatValue,
        ),
    }
}

/// The kernel rows of one atom at one width, each carrying the payload,
/// negative and signaling NaNs through the atom; an invalid operation inside
/// an arithmetic kernel (`inf + -inf`, `0 * inf`); and a one-element group
/// that no addition touches. Widths an atom does not build at in
/// `chelis build` are skipped with the reason beside them.
fn nan_atom_rows(
    atom: chelis_ir::dag::RiscAtomIdentity,
    w: &'static NanWidth,
) -> Vec<NanKernelRow> {
    use chelis_ir::dag::RiscAtomIdentity as Id;
    let name = w.name;
    let neg_inf = w.inf | w.sign();
    let (p, n, s) = (w.payload, w.negative, w.signaling);
    let row = |label: &str,
               params: Vec<(&'static str, Vec<i64>, Vec<u64>)>,
               output: Vec<i64>,
               call: String| NanKernelRow {
        label: format!("{label}_{name}"),
        width: w,
        params,
        output,
        call,
        helper: None,
        finalization: None,
        gradient: false,
    };
    let helped = |label: &str, params, output, call: String, helper: String| NanKernelRow {
        helper: Some(helper),
        finalization: Some(chelis_backend_c::fp_env::NanFinalization::Canonical),
        gradient: true,
        ..row(label, params, output, call)
    };
    let x3 = || vec![("x", vec![3], vec![p, n, s])];
    // `prod_reduce`, the windows and their gradients build at f32 only
    // (chelis#729).
    let f32_only = name == "f32";
    match atom {
        Id::Sum => vec![
            row(
                "sum",
                vec![(
                    "x",
                    vec![4, 2],
                    vec![w.inf, neg_inf, p, w.one, n, w.one, s, w.one],
                )],
                vec![4],
                "sum(x, 1i32)".into(),
            ),
            row(
                "sum_carry",
                vec![("x", vec![3, 1], vec![p, n, s])],
                vec![3],
                "sum(x, 1i32)".into(),
            ),
            // [05-OP-30]'s two spellings of a one-element diagonal sum agree.
            row(
                "trace_carry",
                vec![("x", vec![2, 1, 1], vec![p, n])],
                vec![2],
                "trace(x, 1i32, 2i32)".into(),
            ),
            row(
                "diagonal_sum",
                vec![("x", vec![2, 1, 1], vec![p, n])],
                vec![2],
                "sum(diagonal(x, 1i32, 2i32), 1i32)".into(),
            ),
        ],
        Id::MaxReduce => vec![
            row(
                "max_reduce",
                vec![("x", vec![3, 2], vec![p, w.one, n, w.one, s, w.one])],
                vec![3],
                "max_reduce(x, 1i32)".into(),
            ),
            row(
                "max_reduce_first",
                vec![("x", vec![2, 2], vec![p, n, n, p])],
                vec![2],
                "max_reduce(x, 1i32)".into(),
            ),
        ],
        Id::MinReduce => vec![
            row(
                "min_reduce",
                vec![("x", vec![3, 2], vec![w.one, p, w.one, n, w.one, s])],
                vec![3],
                "min_reduce(x, 1i32)".into(),
            ),
            row(
                "min_reduce_first",
                vec![("x", vec![2, 2], vec![p, n, n, p])],
                vec![2],
                "min_reduce(x, 1i32)".into(),
            ),
        ],
        Id::ProdReduce if f32_only => vec![row(
            "prod",
            vec![("x", vec![3, 2], vec![w.zero, w.inf, p, w.one, n, s])],
            vec![3],
            "prod_reduce(x, 1i32)".into(),
        )],
        Id::ReduceWindowSum if f32_only => vec![
            row(
                "window_sum",
                vec![("x", vec![4], vec![w.inf, neg_inf, p, w.one])],
                vec![3],
                "reduce_window_sum(x, [2i64], [1i64])".into(),
            ),
            row(
                "window_carry",
                x3(),
                vec![3],
                "reduce_window_sum(x, [1i64], [1i64])".into(),
            ),
        ],
        Id::ReduceWindowMean if f32_only => vec![row(
            "window_mean",
            vec![("x", vec![4], vec![w.inf, neg_inf, n, w.one])],
            vec![3],
            "reduce_window_mean(x, [2i64], [1i64])".into(),
        )],
        Id::ReduceWindowMax if f32_only => vec![row(
            "window_max",
            vec![("x", vec![4], vec![p, w.one, n, s])],
            vec![3],
            "reduce_window_max(x, [2i64], [1i64])".into(),
        )],
        Id::ReduceWindowMin if f32_only => vec![row(
            "window_min",
            vec![("x", vec![4], vec![p, w.one, n, s])],
            vec![3],
            "reduce_window_min(x, [2i64], [1i64])".into(),
        )],
        Id::ReduceWindowGrad if f32_only => vec![helped(
            "window_grad",
            vec![("x", vec![4], vec![w.inf, neg_inf, p, w.one])],
            vec![4],
            format!("grad(lwin_{name}, wrt=x)(x)"),
            format!(
                "def lwin_{name}(x: tensor[4, {name}]) -> {name} = {{\n  w = reduce_window_sum(x, [2i64], [1i64])\n  tensor_to_scalar(sum(mul(w, copy(w)), 0i32))\n}}\n"
            ),
        )],
        Id::ProdReduce
        | Id::ReduceWindowSum
        | Id::ReduceWindowMean
        | Id::ReduceWindowMax
        | Id::ReduceWindowMin
        | Id::ReduceWindowGrad => Vec::new(),
        Id::Matmul => vec![row(
            "matmul",
            vec![
                ("a", vec![2, 2], vec![w.zero, w.one, p, w.one]),
                ("b", vec![2, 2], vec![w.inf, w.one, w.one, s]),
            ],
            vec![2, 2],
            "matmul(a, b)".into(),
        )],
        Id::Dropout => vec![row(
            "dropout",
            x3(),
            vec![3],
            format!("dropout(key_from_seed(7i64), x, cast(0.0, {name}))"),
        )],
        Id::DropoutReplay => vec![helped(
            "dropout_grad",
            x3(),
            vec![3],
            format!("grad(ldrop_{name}, wrt=x)(x)"),
            format!(
                "def ldrop_{name}(x: tensor[3, {name}]) -> {name} = tensor_to_scalar(sum(mul(dropout(key_from_seed(7i64), x, cast(0.5, {name})), x), 0i32))\n"
            ),
        )],
        // The checker admits `uniform_like` bounds at f32 only.
        Id::UniformBoundAdjoint if name == "f32" => vec![helped(
            "uniform_low_grad",
            vec![
                ("x", vec![3], vec![p, w.one, n]),
                ("lo", vec![1], vec![w.zero]),
            ],
            vec![1],
            format!("grad(lunif_{name}, wrt=lo)(lo, x)"),
            format!(
                "def lunif_{name}(lo: tensor[1, {name}], x: tensor[3, {name}]) -> {name} = {{\n  b = tensor_to_scalar(sum(lo, 0i32))\n  tensor_to_scalar(sum(mul(uniform_like(key_from_seed(1i64), x, b, cast(2.0, {name})), x), 0i32))\n}}\n"
            ),
        )],
        Id::UniformBoundAdjoint => Vec::new(),
        Id::ReluAdjoint => vec![helped(
            "relu_grad",
            vec![("x", vec![3], vec![p, w.one, n])],
            vec![3],
            format!("grad(lrelu_{name}, wrt=x)(x)"),
            format!(
                "def lrelu_{name}(x: tensor[3, {name}]) -> {name} = tensor_to_scalar(sum(mul(relu(x), x), 0i32))\n"
            ),
        )],
        Id::ExtremaAdjoint => vec![helped(
            "max_elem_grad",
            vec![("x", vec![3], vec![p, w.one, n])],
            vec![3],
            format!("grad(lmax_{name}, wrt=x)(x)"),
            format!(
                "def lmax_{name}(x: tensor[3, {name}]) -> {name} = tensor_to_scalar(sum(mul(max_elem(x, neg(x)), x), 0i32))\n"
            ),
        )],
        // Selection and data movement: every NaN out is a NaN in, bit for bit.
        Id::Where => vec![row(
            "where",
            vec![
                ("x", vec![3], vec![p, n, s]),
                ("y", vec![3], vec![w.one, w.one, w.one]),
            ],
            vec![3],
            "where(cmplt(copy(x), copy(y)), y, x)".into(),
        )],
        Id::Reshape => vec![row(
            "reshape",
            x3(),
            vec![1, 3],
            "reshape(x, [1i64, 3i64])".into(),
        )],
        Id::Permute => vec![row(
            "permute",
            vec![("x", vec![2, 2], vec![p, n, s, w.one])],
            vec![2, 2],
            "permute(x, 1i32, 0i32)".into(),
        )],
        Id::Expand => vec![row(
            "expand",
            vec![("x", vec![1], vec![s])],
            vec![3],
            "expand(x, 0i32, 3i64)".into(),
        )],
        Id::Pad => vec![row(
            "pad",
            x3(),
            vec![4],
            format!("pad(x, [[1i64, 0i64]], cast(0.0, {name}))"),
        )],
        Id::Shrink => vec![row(
            "shrink",
            x3(),
            vec![2],
            "shrink(x, [[1i64, 3i64]])".into(),
        )],
        Id::Stride => vec![row("stride", x3(), vec![2], "stride(x, 2i64)".into())],
        Id::Gather => vec![row(
            "gather",
            x3(),
            vec![4],
            "gather(x, to_tensor([2i64, 0i64, 1i64, 2i64]), 0i32)".into(),
        )],
        Id::ScatterReplace => vec![row(
            "scatter_replace",
            vec![
                ("x", vec![3], vec![w.one, w.one, w.one]),
                ("y", vec![2], vec![s, n]),
            ],
            vec![3],
            "scatter(x, to_tensor([2i64, 0i64]), y, 0i32, \"replace\")".into(),
        )],
        Id::ScatterElements => vec![row(
            "scatter_elements",
            vec![
                ("x", vec![3], vec![w.one, w.one, w.one]),
                ("y", vec![3], vec![p, n, s]),
            ],
            vec![3],
            "scatter_elements(x, to_tensor([2i64, 0i64, 1i64]), y, 0i32)".into(),
        )],
        // chelis#3047: a gather with duplicate indices differentiates to a
        // scatter-add of a finite and a NaN contribution into one element; at
        // f16 and bf16 that sum is taken at f32. The f64 gradient does not
        // build (chelis#729).
        Id::Scatter if name != "f64" => {
            let gather = "gather(x, to_tensor([0i64, 0i64, 2i64, 2i64]), 0i32)";
            vec![helped(
                "gather_grad",
                vec![("x", vec![4], vec![w.one, n, p, w.one])],
                vec![4],
                format!("grad(gather_loss_{name}, wrt=x)(x)"),
                format!(
                    "def gather_loss_{name}(x: tensor[4, {name}]) -> {name} = tensor_to_scalar(sum(mul({gather}, {gather}), 0i32))\n"
                ),
            )]
        }
        _ => Vec::new(),
    }
}

/// chelis#2957 rounds 2 and 3: every emitter that writes a float element
/// through its own store agrees bit for bit with eval through the static
/// library ABI at f16, bf16, f32 and f64, and both lanes finalize NaNs as
/// [04-NUM-2] classifies the atom: arithmetic and conversion to the
/// canonical quiet NaN, [05-OP-12]/[05-OP-40] selection and data movement
/// keeping an input NaN's exact bits. The rows are derived from the atom
/// universe (`RiscAtomIdentity::ALL`) through [`nan_atom_coverage`], and the
/// expected finalization from `fp_env::risc_nan_finalization`, so a new atom
/// or a reclassification changes what this oracle checks. `to_tensor` is a
/// list builtin and `einsum` a runtime kernel, neither with a RISC atom, so
/// their rows are listed by name.
#[test]
fn every_reduction_and_vendor_kernel_finalizes_nan_like_eval_through_the_static_library_abi() {
    use chelis_ir::dag::{RiscAtomDisposition, RiscAtomIdentity};
    let inventory = chelis_backend_c::host_builtin_nan_inventory();
    let mut rows = Vec::new();
    for &atom in RiscAtomIdentity::ALL {
        let (op, coverage) = nan_atom_coverage(atom);
        assert_eq!(
            op.atom_disposition(),
            RiscAtomDisposition::Semantic(atom),
            "the representative of `{}` is another atom",
            atom.as_str()
        );
        let finalization = chelis_backend_c::fp_env::risc_nan_finalization(&op);
        match coverage {
            NanAtomCoverage::Rows => {
                let mut covered = false;
                for w in &NAN_WIDTHS {
                    for mut row in nan_atom_rows(atom, w) {
                        if !row.gradient {
                            row.finalization = finalization;
                        }
                        rows.push(row);
                        covered = true;
                    }
                }
                assert!(
                    covered,
                    "`{}` has no kernel row at any width",
                    atom.as_str()
                );
            }
            NanAtomCoverage::BuiltinInventory => assert!(
                inventory.contains(&(atom.as_str(), finalization)),
                "`{}` is not in the host builtin inventory as {finalization:?}",
                atom.as_str()
            ),
            NanAtomCoverage::CastRows => assert_eq!(
                finalization,
                Some(chelis_backend_c::fp_env::NanFinalization::Canonical),
                "the cast rows expect a canonical conversion"
            ),
            NanAtomCoverage::NoFloatValue => assert_eq!(
                finalization,
                None,
                "`{}` produces a float value; give it kernel rows",
                atom.as_str()
            ),
        }
    }
    for w in &NAN_WIDTHS {
        rows.push(NanKernelRow {
            label: format!("to_tensor_{}", w.name),
            width: w,
            params: vec![("x", vec![3], vec![w.payload, w.negative, w.signaling])],
            output: vec![3],
            call: "to_tensor(to_list(x))".into(),
            helper: None,
            finalization: None,
            gradient: false,
        });
        // `einsum` is a runtime kernel with no RISC atom; its products and
        // sums are arithmetic (chelis#1290 owns its accumulation order).
        rows.push(NanKernelRow {
            label: format!("einsum_{}", w.name),
            width: w,
            params: vec![
                ("a", vec![2, 1], vec![w.payload, w.signaling]),
                ("b", vec![1, 1], vec![w.one]),
            ],
            output: vec![2, 1],
            call: "einsum(\"ij,jk->ik\", a, b)".into(),
            helper: None,
            finalization: Some(chelis_backend_c::fp_env::NanFinalization::Canonical),
            gradient: false,
        });
    }
    let dims = |shape: &[i64]| -> String {
        shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mangle = |name: &str| -> String {
        let hex: String = name.bytes().map(|byte| format!("{byte:02x}")).collect();
        format!("chelis_fn_{hex}")
    };
    let mut program = String::new();
    let mut harness = String::from(
        "#include <stdio.h>\n#include <stdint.h>\n\
         #include \"chelis_runtime.h\"\n#include \"nan_kernels.h\"\n\
         int main(void) {\n",
    );
    for row in &rows {
        let w = row.width;
        let params = row
            .params
            .iter()
            .map(|(param, shape, _)| format!("{param}: tensor[{}, {}]", dims(shape), w.name))
            .collect::<Vec<_>>()
            .join(", ");
        program.push_str(row.helper.as_deref().unwrap_or_default());
        program.push_str(&format!(
            "def t_{}({params}) -> tensor[{}, {}] = {}\n",
            row.label,
            dims(&row.output),
            w.name,
            row.call
        ));
        let mut tensors = Vec::new();
        for (param, shape, bits) in &row.params {
            let data = bits
                .iter()
                .map(|bits| format!("{bits:#x}"))
                .collect::<Vec<_>>()
                .join(", ");
            let var = format!("t_{}_{param}", row.label);
            harness.push_str(&format!(
                "    chelis_tensor *{var};\n    {{ static const {storage} d[{len}] = {{ {data} }}; static const int64_t shape[{rank}] = {{ {shape} }};\n      \
                 {var} = chelis_tensor_entry_borrow({rank}, shape, {dtype}, d, sizeof d); }}\n",
                storage = w.storage,
                len = bits.len(),
                rank = shape.len(),
                shape = dims(shape),
                dtype = w.dtype,
            ));
            tensors.push(var);
        }
        let count: i64 = row.output.iter().product();
        harness.push_str(&format!(
            "    {{ chelis_tensor *r = {}({}); const {storage} *o = (const {storage} *)chelis_tensor_read_view(r).data;\n      \
             for (int i = 0; i < {count}; ++i) printf(\"t_{} %d %llx\\n\", i, (unsigned long long)o[i]); }}\n",
            mangle(&format!("t_{}", row.label)),
            tensors.join(", "),
            row.label,
            storage = w.storage,
        ));
    }
    harness.push_str("    return 0;\n}\n");

    let dir = tempdir().unwrap();
    let file = dir.path().join("nan_kernels.ch");
    let out = dir.path().join("out");
    fs::write(&file, &program).unwrap();
    build(&file, &out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Built static library"));
    let driver = out.join("driver.c");
    fs::write(&driver, &harness).unwrap();
    assert!(
        link_static_library_driver(&out, "libnan_kernels.a", true),
        "the oracle driver must link"
    );
    let run = Process::new(out.join("driver")).output().unwrap();
    assert!(run.status.success(), "{run:?}");
    let stdout = String::from_utf8(run.stdout).unwrap();

    let mut failures = Vec::new();
    for row in &rows {
        let w = row.width;
        let params = row
            .params
            .iter()
            .map(|(param, shape, _)| format!("{param}: tensor[{}, {}]", dims(shape), w.name))
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "{}def main({params}) -> tensor[{}, {}] = {}\n",
            row.helper.as_deref().unwrap_or_default(),
            dims(&row.output),
            w.name,
            row.call
        );
        let inputs: Vec<_> = row
            .params
            .iter()
            .map(|(param, shape, bits)| (*param, w, shape.clone(), bits.clone()))
            .collect();
        let eval = nan_eval_shaped_bits(&source, &inputs, w);
        let mut saw_nan = false;
        let input_bits: Vec<u64> = row
            .params
            .iter()
            .flat_map(|(_, _, bits)| bits.iter().copied())
            .collect();
        for (index, expected) in eval.iter().enumerate() {
            if w.is_nan(*expected) {
                saw_nan = true;
                let admitted = match row.finalization {
                    Some(chelis_backend_c::fp_env::NanFinalization::Canonical) => {
                        *expected == w.canonical
                    }
                    Some(chelis_backend_c::fp_env::NanFinalization::BitPreserving) | None => {
                        input_bits.contains(expected)
                    }
                };
                if !admitted {
                    failures.push(format!(
                        "eval {} [{index}] gave {expected:#x}, not a {:?} NaN",
                        row.label, row.finalization
                    ));
                }
            }
            let prefix = format!("t_{} {index} ", row.label);
            let got = stdout
                .lines()
                .find_map(|line| line.strip_prefix(&prefix))
                .map(|bits| u64::from_str_radix(bits, 16).unwrap());
            if got != Some(*expected) {
                let got = got.map_or("nothing".to_string(), |bits| format!("{bits:#x}"));
                failures.push(format!(
                    "t_{} [{index}]: C {got}, eval {expected:#x}",
                    row.label
                ));
            }
        }
        if !saw_nan {
            failures.push(format!("row {} must produce a NaN to test", row.label));
        }
    }
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---- chelis#2957: eval/C parity through a built static library ----
//
// The oracles below build one `chelis build` static library (the shipped
// profile) and call it from a C driver that passes every input as run-time
// storage bits, so the C compiler cannot fold a call whose argument it cannot
// see. Each result is compared bit for bit with eval.

fn width_named(name: &str) -> &'static NanWidth {
    NAN_WIDTHS
        .iter()
        .find(|width| width.name == name)
        .unwrap_or_else(|| panic!("no float width {name}"))
}

fn mangled(name: &str) -> String {
    let hex: String = name.bytes().map(|byte| format!("{byte:02x}")).collect();
    format!("chelis_fn_{hex}")
}

/// The C expression for a scalar argument given as `width`'s storage bits.
fn c_scalar_arg(width: &NanWidth, bits: u64) -> String {
    match width.name {
        "f32" => format!("f32_of(UINT32_C({bits:#x}))"),
        "f64" => format!("f64_of(UINT64_C({bits:#x}))"),
        _ => format!("(uint16_t){bits:#x}"),
    }
}

/// The C expression for a scalar result's storage bits.
fn c_scalar_bits(width: &NanWidth, call: &str) -> String {
    match width.name {
        "f32" => format!("(unsigned long long)f32_bits({call})"),
        "f64" => format!("(unsigned long long)f64_bits({call})"),
        _ => format!("(unsigned long long){call}"),
    }
}

/// Driver lines calling the scalar `def` once per input, printing
/// `label i bits`.
fn c_scalar_calls(label: &str, def: &str, width: &NanWidth, inputs: &[u64]) -> String {
    inputs
        .iter()
        .enumerate()
        .map(|(index, bits)| {
            let call = format!("{}({})", mangled(def), c_scalar_arg(width, *bits));
            format!(
                "    printf(\"{label} {index} %llx\\n\", {});\n",
                c_scalar_bits(width, &call)
            )
        })
        .collect()
}

/// Driver lines passing `inputs` to the tensor `def` as one rank-1 tensor,
/// printing each result element as `label i bits`.
fn c_tensor_call(label: &str, def: &str, width: &NanWidth, inputs: &[u64]) -> String {
    let n = inputs.len();
    let data = inputs
        .iter()
        .map(|bits| format!("{bits:#x}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "    {{ static const {storage} d[{n}] = {{ {data} }}; static const int64_t shape[1] = {{ {n} }};\n      \
         chelis_tensor *r = {def}(chelis_tensor_entry_borrow(1, shape, {dtype}, d, sizeof d));\n      \
         const {storage} *o = (const {storage} *)chelis_tensor_read_view(r).data;\n      \
         for (int i = 0; i < {n}; ++i) printf(\"{label} %d %llx\\n\", i, (unsigned long long)o[i]); }}\n",
        storage = width.storage,
        dtype = width.dtype,
        def = mangled(def),
    )
}

/// Build `program` as the static library `lib<stem>.a`, link a driver whose
/// `main` runs `body`, run it, and return the build's `Compiler:` line with
/// every printed `label i bits` result keyed by `label i`.
fn run_static_library(
    stem: &str,
    program: &str,
    body: &str,
) -> (String, std::collections::BTreeMap<String, u64>) {
    let dir = tempdir().unwrap();
    let file = dir.path().join(format!("{stem}.ch"));
    let out = dir.path().join("out");
    fs::write(&file, program).unwrap();
    let built = build(&file, &out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Built static library"));
    let compiler = String::from_utf8_lossy(&built.get_output().stdout)
        .lines()
        .find(|line| line.starts_with("Compiler: "))
        .expect("the build reports its C compiler")
        .to_string();
    let driver = format!(
        "#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\n\
         #include \"chelis_runtime.h\"\n#include \"{stem}.h\"\n\
         static float f32_of(uint32_t b) {{ float x; memcpy(&x, &b, 4); return x; }}\n\
         static double f64_of(uint64_t b) {{ double x; memcpy(&x, &b, 8); return x; }}\n\
         static uint32_t f32_bits(float x) {{ uint32_t b; memcpy(&b, &x, 4); return b; }}\n\
         static uint64_t f64_bits(double x) {{ uint64_t b; memcpy(&b, &x, 8); return b; }}\n\
         int main(void) {{\n{body}    return 0;\n}}\n"
    );
    fs::write(out.join("driver.c"), driver).unwrap();
    assert!(
        link_static_library_driver(&out, &format!("lib{stem}.a"), false),
        "the {stem} driver must link"
    );
    let run = Process::new(out.join("driver")).output().unwrap();
    assert!(run.status.success(), "{run:?}");
    let results = String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            let (key, bits) = line.rsplit_once(' ').expect("`label i bits`");
            (key.to_string(), u64::from_str_radix(bits, 16).unwrap())
        })
        .collect();
    (compiler, results)
}

/// The bits the driver printed for `label i`.
fn c_result(results: &std::collections::BTreeMap<String, u64>, label: &str, index: usize) -> u64 {
    *results
        .get(&format!("{label} {index}"))
        .unwrap_or_else(|| panic!("the driver printed no `{label} {index}`"))
}

/// `body` (over `x`) applied by eval to the rank-1 `width` tensor `inputs`.
fn eval_tensor_body(helpers: &str, body: &str, width: &NanWidth, inputs: &[u64]) -> Vec<u64> {
    let (n, name) = (inputs.len(), width.name);
    let source =
        format!("{helpers}def main(x: tensor[{n}, {name}]) -> tensor[{n}, {name}] = {body}\n");
    nan_eval_bits(&source, &[("x", width, inputs.to_vec())], width)
}

/// Compare every `(what, C bits, eval bits)` triple and fail with all
/// disagreements.
fn assert_lanes_agree(oracle: &str, rows: Vec<(String, u64, u64)>) {
    assert!(!rows.is_empty(), "{oracle}: nothing was compared");
    let failures: Vec<String> = rows
        .iter()
        .filter(|(_, c, eval)| c != eval)
        .map(|(what, c, eval)| format!("{what}: C {c:#x}, eval {eval:#x}"))
        .collect();
    assert!(
        failures.is_empty(),
        "{oracle}: {} of {} results disagree:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

fn f32_bits_of(values: &[f64]) -> Vec<u64> {
    values
        .iter()
        .map(|value| u64::from((*value as f32).to_bits()))
        .collect()
}

fn f64_bits_of(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A finite value as a Surf literal of `width` that names it exactly.
fn exact_literal(width: &NanWidth, bits: u64) -> String {
    let (magnitude, negative) = match width.name {
        "f32" => {
            let x = f32::from_bits(u32::try_from(bits).unwrap());
            (format!("{:?}f32", x.abs()), x.is_sign_negative())
        }
        "f64" => {
            let x = f64::from_bits(bits);
            (format!("{:?}f64", x.abs()), x.is_sign_negative())
        }
        other => panic!("no exact literal spelling at {other}"),
    };
    if negative {
        format!("neg({magnitude})")
    } else {
        magnitude
    }
}

/// `chelis eval --json` of `source`, returning root `name`'s tensor bits.
fn cli_eval_bits(source: &str, name: &str) -> Vec<u64> {
    let dir = tempdir().unwrap();
    let file = dir.path().join("reference.ch");
    fs::write(&file, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", file.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let root = value["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == name)
        .unwrap_or_else(|| panic!("no eval root {name}: {value}"));
    root["value"]["value"]["data"]["bits"]
        .as_array()
        .expect("tensor bits")
        .iter()
        .map(|bits| u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap())
        .collect()
}

/// chelis#2952's f32 inputs: the issue's six `normal_cdf` witnesses and its
/// `sigmoid` witness, `1.72889078` (whose `exp(-x)` is the issue's misrounded
/// `expf(-1.72889078)`), then inputs at which macOS libm misrounds the `exp`
/// inside `normal_cdf` and `gelu` (found by scanning each graph's `exp`
/// argument against an MPFR-backed reference).
const COMPOUND_WITNESSES_F32: [f64; 14] = [
    -1.335_994_482_040_405_3,
    -0.461_234_301_328_659_06,
    -1.219_169_020_652_771,
    0.919_759_511_947_631_8,
    -0.335_103_929_042_816_16,
    0.031_051_857_396_960_26,
    0.048_035_141_1,
    1.728_890_78,
    1.851_996_064_186_096_2,
    0.053_997_941_315_174_1,
    1.121_795_654_296_875,
    0.580_531_716_346_740_7,
    1.759_212_017_059_326_2,
    3.211_342_573_165_893_6,
];

/// chelis#2952's f64 inputs: `0x1.93179431561a0p-2` (whose `exp(-x)` is the
/// issue's misrounded `exp(-0x1.93179431561a0p-2)`), then inputs at which macOS
/// libm misrounds the `exp` inside `normal_cdf` and `gelu`.
fn compound_witnesses_f64() -> Vec<f64> {
    let mut values = vec![
        f64::from_bits(0x3fd9_3179_4315_61a0),
        -2.908_605_891_044_485,
        1.511_954_080_226_180_6,
        1.838_723_303_506_749_4,
        0.491_156_805_196_720_1,
        0.081_570_304_728_129_1,
        1.587_021_571_239_189_3,
    ];
    values.extend(COMPOUND_WITNESSES_F32);
    values
}

/// chelis#2952: every zero-ULP compound that embeds `exp` (`sigmoid`, `silu`,
/// `gelu`, `tanh`, `softmax`, `Std.Contracts.normal_cdf`) agrees bit for bit
/// between eval and a built static library at f32 and f64, at the shipped
/// optimization level, with each input supplied at run time as a scalar
/// argument and as a tensor element (softmax: the pair `[0, -x]`, so its
/// `exp(-x)` meets the witness). The inputs are where platform libm misrounds
/// the embedded `exp`.
#[test]
fn compound_activations_match_eval_on_libm_misrounding_inputs_through_the_static_library_abi() {
    const UNARY: [&str; 4] = ["sigmoid", "silu", "gelu", "tanh"];
    let widths = [
        (width_named("f32"), f32_bits_of(&COMPOUND_WITNESSES_F32)),
        (width_named("f64"), f64_bits_of(&compound_witnesses_f64())),
    ];
    let mut program = String::from("import Std.Contracts (normal_cdf)\n");
    let mut body = String::new();
    for (width, inputs) in &widths {
        let (w, n) = (width.name, inputs.len());
        for op in UNARY {
            program.push_str(&format!(
                "def s_{op}_{w}(x: {w}) -> {w} = {op}(x)\n\
                 def t_{op}_{w}(x: tensor[{n}, {w}]) -> tensor[{n}, {w}] = {op}(x)\n"
            ));
            body.push_str(&c_scalar_calls(
                &format!("s_{op}_{w}"),
                &format!("s_{op}_{w}"),
                width,
                inputs,
            ));
            body.push_str(&c_tensor_call(
                &format!("t_{op}_{w}"),
                &format!("t_{op}_{w}"),
                width,
                inputs,
            ));
        }
        program.push_str(&format!(
            "def s_ncdf_{w}(x: {w}) -> {w} = normal_cdf(x)\n\
             def t_softmax_{w}(x: tensor[2, {w}]) -> tensor[2, {w}] = softmax(x, 0i32)\n"
        ));
        body.push_str(&c_scalar_calls(
            &format!("s_ncdf_{w}"),
            &format!("s_ncdf_{w}"),
            width,
            inputs,
        ));
        for (index, bits) in inputs.iter().enumerate() {
            body.push_str(&c_tensor_call(
                &format!("t_softmax_{w}_{index}"),
                &format!("t_softmax_{w}"),
                width,
                &[0, bits ^ width.sign()],
            ));
        }
    }
    let (_, c) = run_static_library("compound_parity", &program, &body);

    let mut rows = Vec::new();
    for (width, inputs) in &widths {
        let w = width.name;
        for op in UNARY {
            let eval = eval_tensor_body("", &format!("{op}(x)"), width, inputs);
            for (index, (input, expected)) in inputs.iter().zip(&eval).enumerate() {
                for lane in ["s", "t"] {
                    let label = format!("{lane}_{op}_{w}");
                    rows.push((
                        format!("{label}({input:#x})"),
                        c_result(&c, &label, index),
                        *expected,
                    ));
                }
            }
        }
        let calls = inputs
            .iter()
            .map(|bits| format!("normal_cdf({})", exact_literal(width, *bits)))
            .collect::<Vec<_>>()
            .join(", ");
        let eval = cli_eval_bits(
            &format!("import Std.Contracts (normal_cdf)\nreference = to_tensor([{calls}])\n"),
            "reference",
        );
        for (index, (input, expected)) in inputs.iter().zip(&eval).enumerate() {
            let label = format!("s_ncdf_{w}");
            rows.push((
                format!("{label}({input:#x})"),
                c_result(&c, &label, index),
                *expected,
            ));
        }
        for (index, bits) in inputs.iter().enumerate() {
            let pair = [0, bits ^ width.sign()];
            let eval = eval_tensor_body("", "softmax(x, 0i32)", width, &pair);
            for (element, expected) in eval.iter().enumerate() {
                let label = format!("t_softmax_{w}_{index}");
                rows.push((
                    format!("{label}[{element}]({pair:#x?})"),
                    c_result(&c, &label, element),
                    *expected,
                ));
            }
        }
    }
    assert_lanes_agree("chelis#2952 compounds", rows);
}

/// The chelis#2961 witnesses: inputs at which a C compiler's fold of a
/// literal, platform libm, and the correctly rounded result disagree.
const HOST_SCALAR_WITNESSES: [(&str, f64); 4] = [
    ("atan", 5.531_991_004_943_848),
    ("exp", 57.802_669_525_146_484),
    ("cos", 29.123_893_737_792_97),
    ("sin", -7_755.117_675_781_25),
];

/// chelis#2961: one built program computes each witness's transcendental as a
/// scalar literal (a constant the C compiler can fold), as a run-time scalar
/// argument (the host scalar emitter), and as a run-time tensor element, at
/// f32 and f64; all three equal eval bit for bit. The test prints the C
/// compiler `chelis build` resolved, so the CI log names the compiler each
/// platform's run used, and on a GitHub Actions Linux runner it requires gcc,
/// the issue's second compiler.
#[test]
fn host_scalar_transcendentals_match_eval_as_literal_runtime_scalar_and_tensor_element() {
    let mut program = String::new();
    let mut body = String::new();
    let mut cases = Vec::new();
    for width in [width_named("f32"), width_named("f64")] {
        let w = width.name;
        for (op, witness) in HOST_SCALAR_WITNESSES {
            let bits = if w == "f32" {
                f32_bits_of(&[witness])[0]
            } else {
                witness.to_bits()
            };
            let label = format!("{op}_{w}");
            program.push_str(&format!(
                "def l_{label}(x: {w}) -> {w} = {op}({literal})\n\
                 def s_{label}(x: {w}) -> {w} = {op}(x)\n\
                 def t_{label}(x: tensor[1, {w}]) -> tensor[1, {w}] = {op}(x)\n",
                literal = exact_literal(width, bits),
            ));
            for lane in ["l", "s"] {
                body.push_str(&c_scalar_calls(
                    &format!("{lane}_{label}"),
                    &format!("{lane}_{label}"),
                    width,
                    &[bits],
                ));
            }
            body.push_str(&c_tensor_call(
                &format!("t_{label}"),
                &format!("t_{label}"),
                width,
                &[bits],
            ));
            cases.push((width, op, label, bits));
        }
    }
    let (compiler, c) = run_static_library("host_scalar_parity", &program, &body);
    println!("chelis#2961 oracle {compiler}");
    if cfg!(target_os = "linux") && std::env::var_os("GITHUB_ACTIONS").is_some() {
        assert!(
            compiler.contains("gcc"),
            "the Linux CI run of this oracle must use gcc: {compiler}"
        );
    }

    let mut rows = Vec::new();
    for (width, op, label, bits) in cases {
        let expected = eval_tensor_body("", &format!("{op}(x)"), width, &[bits])[0];
        for lane in ["l", "s", "t"] {
            rows.push((
                format!("{lane}_{label}({bits:#x})"),
                c_result(&c, &format!("{lane}_{label}"), 0),
                expected,
            ));
        }
    }
    assert_lanes_agree("chelis#2961 host scalar transcendentals", rows);
}

/// chelis#2989's inputs at `width`: the issue's moderate and large magnitudes,
/// the largest finite value, signed zeros, both infinities, and NaN.
fn sin_adjoint_inputs(width: &NanWidth) -> Vec<u64> {
    let finite = [0.7, 1.3, 100.0, 500.0, 1e4, -3.5];
    let mut bits: Vec<u64> = match width.name {
        "f16" => finite
            .iter()
            .map(|x| u64::from(half::f16::from_f32(*x as f32).to_bits()))
            .chain([u64::from(half::f16::MAX.to_bits())])
            .collect(),
        "bf16" => finite
            .iter()
            .map(|x| u64::from(half::bf16::from_f32(*x as f32).to_bits()))
            .chain([u64::from(half::bf16::MAX.to_bits())])
            .collect(),
        "f32" => f32_bits_of(&finite)
            .into_iter()
            .chain([u64::from(f32::MAX.to_bits())])
            .collect(),
        _ => f64_bits_of(&finite)
            .into_iter()
            .chain([f64::MAX.to_bits()])
            .collect(),
    };
    bits.extend([
        width.zero,
        width.sign(),
        width.inf,
        width.inf | width.sign(),
        width.canonical,
    ]);
    bits
}

/// chelis#2989: at every float width, `grad(sum(sin(x)))` and
/// `vmap(grad(sin))` equal the same lane's forward `cos(x)` bit for bit, on
/// large magnitudes, signed zeros and non-finite inputs, in eval and in a
/// built static library; the C forward `cos` equals eval's.
#[test]
fn sin_adjoint_is_the_cos_primitive_in_eval_and_c_at_every_float_width() {
    let mut program = String::new();
    let mut body = String::new();
    let mut cases = Vec::new();
    for width in &NAN_WIDTHS {
        let (w, inputs) = (width.name, sin_adjoint_inputs(width));
        let n = inputs.len();
        let helpers = format!(
            "def sum_sin_{w}(x: tensor[{n}, {w}]) -> tensor[{w}] = sum(sin(x), 0i32)\n\
             def sin_at_{w}(x: tensor[{w}]) -> {w} = tensor_to_scalar(sin(x))\n"
        );
        program.push_str(&helpers);
        for (label, call) in [
            ("cos", "cos(x)".to_string()),
            ("grad", format!("grad(sum_sin_{w})(x)")),
            ("vmap", format!("vmap(grad(sin_at_{w}))(x)")),
        ] {
            program.push_str(&format!(
                "def {label}_{w}(x: tensor[{n}, {w}]) -> tensor[{n}, {w}] = {call}\n"
            ));
            body.push_str(&c_tensor_call(
                &format!("{label}_{w}"),
                &format!("{label}_{w}"),
                width,
                &inputs,
            ));
            cases.push((width, label, call, helpers.clone(), inputs.clone()));
        }
    }
    let (_, c) = run_static_library("sin_adjoint", &program, &body);

    let mut rows = Vec::new();
    for width in &NAN_WIDTHS {
        let inputs = sin_adjoint_inputs(width);
        let forward = eval_tensor_body("", "cos(x)", width, &inputs);
        for (case_width, label, call, helpers, _) in &cases {
            if case_width.name != width.name {
                continue;
            }
            let eval = eval_tensor_body(helpers, call, width, &inputs);
            for (index, input) in inputs.iter().enumerate() {
                let what = format!("{label}_{}({input:#x})", width.name);
                // Each lane's adjoint equals that lane's own forward cos.
                rows.push((
                    format!("C {what} vs C cos"),
                    c_result(&c, &format!("{label}_{}", width.name), index),
                    c_result(&c, &format!("cos_{}", width.name), index),
                ));
                rows.push((
                    format!("eval {what} vs eval cos"),
                    forward[index],
                    eval[index],
                ));
            }
        }
        for (index, input) in inputs.iter().enumerate() {
            rows.push((
                format!("cos_{}({input:#x})", width.name),
                c_result(&c, &format!("cos_{}", width.name), index),
                forward[index],
            ));
        }
    }
    assert_lanes_agree("chelis#2989 sin adjoint", rows);
}

/// [05-OP-46]'s tanh adjoint `g * (1 - y * y)` with `y = tanh(x)` at a
/// half-precision width: each primitive computed at f32 and rounded once to
/// storage ([04-NUM-8]), the cotangent `g` of `sum` being one, and a NaN
/// finalized to the canonical quiet NaN.
fn tanh_adjoint_reference(width: &NanWidth, bits: u16) -> u64 {
    let is_f16 = width.name == "f16";
    // A value rounded once to the width's storage, with its storage bits.
    let round = |v: f32| -> (f32, u16) {
        if is_f16 {
            let r = half::f16::from_f32(v);
            (r.to_f32(), r.to_bits())
        } else {
            let r = half::bf16::from_f32(v);
            (r.to_f32(), r.to_bits())
        }
    };
    let x = if is_f16 {
        half::f16::from_bits(bits).to_f32()
    } else {
        half::bf16::from_bits(bits).to_f32()
    };
    let (y, _) = round(chelis_crmath::tanh_f32(x));
    let (y_sq, _) = round(y * y);
    let (one_minus, _) = round(1.0 - y_sq);
    let (dx, dx_bits) = round(1.0 * one_minus);
    if dx.is_nan() {
        width.canonical
    } else {
        u64::from(dx_bits)
    }
}

/// chelis#2967 item 6: `grad(sum(tanh(x)))` equals [05-OP-46]'s pinned
/// adjoint on all 65,536 f16 and bf16 storage patterns, in eval and in a
/// built static library whose driver generates the inputs at run time.
#[test]
fn tanh_adjoint_is_the_pinned_graph_on_every_half_precision_input_in_eval_and_c() {
    let mut program = String::new();
    let mut body = String::new();
    for w in ["f16", "bf16"] {
        let width = width_named(w);
        program.push_str(&format!(
            "def sum_tanh_{w}(x: tensor[65536, {w}]) -> tensor[{w}] = sum(tanh(x), 0i32)\n\
             def grad_tanh_{w}(x: tensor[65536, {w}]) -> tensor[65536, {w}] = grad(sum_tanh_{w})(x)\n"
        ));
        body.push_str(&format!(
            "    {{ static uint16_t d[65536]; static const int64_t shape[1] = {{ 65536 }};\n      \
             for (int i = 0; i < 65536; ++i) d[i] = (uint16_t)i;\n      \
             chelis_tensor *r = {def}(chelis_tensor_entry_borrow(1, shape, {dtype}, d, sizeof d));\n      \
             const uint16_t *o = (const uint16_t *)chelis_tensor_read_view(r).data;\n      \
             for (int i = 0; i < 65536; ++i) printf(\"grad_tanh_{w} %d %llx\\n\", i, (unsigned long long)o[i]); }}\n",
            def = mangled(&format!("grad_tanh_{w}")),
            dtype = width.dtype,
        ));
    }
    let (_, c) = run_static_library("tanh_adjoint", &program, &body);

    let all: Vec<u64> = (0..=u16::MAX).map(u64::from).collect();
    let mut failures = Vec::new();
    for w in ["f16", "bf16"] {
        let width = width_named(w);
        let eval = eval_tensor_body(
            &format!("def sum_tanh(x: tensor[65536, {w}]) -> tensor[{w}] = sum(tanh(x), 0i32)\n"),
            "grad(sum_tanh)(x)",
            width,
            &all,
        );
        for (index, eval) in eval.iter().enumerate() {
            let want = tanh_adjoint_reference(width, index as u16);
            let got = c_result(&c, &format!("grad_tanh_{w}"), index);
            if *eval != want || got != want {
                failures.push(format!(
                    "{w} x={index:#06x}: eval {eval:#x}, C {got:#x}, pinned {want:#x}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of 131072 half-precision tanh adjoints differ from the pinned graph:\n{}",
        failures.len(),
        failures
            .iter()
            .take(16)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
