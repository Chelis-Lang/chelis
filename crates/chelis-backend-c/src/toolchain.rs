//! Cross-platform toolchain resolution for Chelis CPU codegen consumers.

use std::{
    ffi::OsStr,
    fmt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CodegenRequirements {
    pub wants_openmp: bool,
    pub needs_blas: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlasProvider {
    None,
    OpenBlas,
    Accelerate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeToolchain {
    pub compiler: String,
    pub compile_flags: Vec<String>,
    pub link_flags: Vec<String>,
    pub openmp_enabled: bool,
    pub blas_provider: BlasProvider,
}

/// Legacy helper kept for older tests that only need the compiler name.
pub fn c_compiler() -> String {
    test_toolchain(CodegenRequirements::default()).compiler
}

/// The `chelis build` toolchain: the pinned profile (see [`pinned_toolchain`])
/// for the compiler `CHELIS_CC` names, or else the first platform compiler
/// that runs. The compiler is a declared input of the build, not part of the
/// profile: [`verify_compiler`] records its identity and refuses one that does
/// not apply the profile's floating-point flags.
pub fn runtime_toolchain(requirements: CodegenRequirements) -> NativeToolchain {
    resolve_toolchain(requirements, &["CHELIS_CC"])
}

/// Reference profile for a caller-selected compiler: the pinned profile with
/// OpenMP off, so the agreement gate runs one thread. It does not resolve a
/// compiler or read the environment. Callers spawn the compiler through
/// [`tool_command`] or their own cleared environment.
pub fn strict_reference_toolchain(
    compiler: String,
    requirements: CodegenRequirements,
) -> NativeToolchain {
    pinned_toolchain(compiler, requirements, false)
}

pub fn test_toolchain(requirements: CodegenRequirements) -> NativeToolchain {
    resolve_toolchain(requirements, &["CHELIS_TEST_CC", "CHELIS_CC"])
}

fn resolve_toolchain(requirements: CodegenRequirements, override_vars: &[&str]) -> NativeToolchain {
    let compiler = resolve_compiler(override_vars);
    // OpenMP is gated on `is_real_gcc(&compiler)`. On macOS, `resolve_compiler`
    // returns "clang" (Apple-clang), so this is always false there: Apple-clang
    // does not bundle libomp and `-fopenmp` is unsupported. The Accelerate
    // -framework link in `pinned_toolchain` is therefore decoupled from any
    // OpenMP gating. If a future contributor enables OpenMP on Apple-clang via
    // Homebrew libomp (`-Xpreprocessor -fopenmp -lomp`), revisit this branch
    // but keep the Accelerate link unconditional.
    let openmp_enabled = requirements.wants_openmp && is_real_gcc(&compiler);
    pinned_toolchain(compiler, requirements, openmp_enabled)
}

/// The one host compile/link profile, shared by `chelis build` and the
/// agreement gate. Its flags fix floating-point semantics independently of
/// the build machine: `-O2`, no contraction (spec/08-backends.md), no fast
/// math, and no `-march` (the instruction set is the compiler's configured
/// default, never the build host's CPU). OpenMP is the only optional part; the
/// generated parallel loops are element-wise, so it changes throughput, not
/// values.
fn pinned_toolchain(
    compiler: String,
    requirements: CodegenRequirements,
    openmp_enabled: bool,
) -> NativeToolchain {
    let mut compile_flags = vec![
        "-O2".to_string(),
        "-ffp-contract=off".to_string(),
        "-fno-fast-math".to_string(),
    ];
    let mut link_flags = vec!["-lm".to_string()];
    if !cfg!(target_os = "macos") {
        link_flags.push("-lpthread".to_string());
        link_flags.push("-ldl".to_string());
    }
    if openmp_enabled {
        compile_flags.push("-fopenmp".to_string());
        link_flags.push("-fopenmp".to_string());
    }

    // On macOS, generated C always routes transcendental math (expf/logf/sinf/sqrtf)
    // through Accelerate's vForce intrinsics (vvexpf/vvlogf/vvsinf/vvsqrtf) via
    // `chelis_runtime/include/chelis_math.h`, regardless of whether BLAS is
    // requested.  Linking against the Accelerate framework is therefore
    // mandatory on macOS to resolve those symbols, mirroring what
    // `.github/scripts/smoke_macos_accelerate.py` does for the on-disk
    // compile path.
    let blas_provider = if cfg!(target_os = "macos") {
        link_flags.push("-framework".to_string());
        link_flags.push("Accelerate".to_string());
        if requirements.needs_blas {
            BlasProvider::Accelerate
        } else {
            BlasProvider::None
        }
    } else if requirements.needs_blas {
        link_flags.push("-lopenblas".to_string());
        BlasProvider::OpenBlas
    } else {
        BlasProvider::None
    };

    NativeToolchain {
        compiler,
        compile_flags,
        link_flags,
        openmp_enabled,
        blas_provider,
    }
}

/// The only variables a native compiler, archiver, or linker subprocess
/// inherits; [`tool_command`] clears the rest. Anything else a driver reads can
/// change compiled numerics without appearing on the command line
/// (`CCC_OVERRIDE_OPTIONS`, `NIX_CFLAGS_COMPILE`, `CPATH`, `SDKROOT`,
/// `DEVELOPER_DIR`, `GCC_EXEC_PREFIX`, `COMPILER_PATH`, ...), so this is an
/// allowlist rather than a list of known-bad names.
///
/// - `PATH`: the driver finds its assembler and linker (gcc's `as`/`ld`, the
///   macOS `/usr/bin` shims' `xcrun`) through it, and it is the same `PATH`
///   that selected the compiler, whose identity the build records.
/// - `TMPDIR`: where drivers write intermediate files. It never reaches the
///   output, and sandboxed builds (Nix, macOS app sandboxes) need it.
///
/// Locale variables are deliberately absent so diagnostics are stable. On
/// macOS, `SDKROOT` and `DEVELOPER_DIR` are absent too: the SDK follows the
/// `xcode-select` default, and a different Xcode is selected by naming its
/// compiler in `CHELIS_CC`.
pub const TOOL_ENVIRONMENT: &[&str] = &["PATH", "TMPDIR"];

/// A command for a native build tool with the environment cleared except
/// for [`TOOL_ENVIRONMENT`].
pub fn tool_command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    command.env_clear();
    for name in TOOL_ENVIRONMENT {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
}

/// The compiler a build used, recorded in the build output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerIdentity {
    /// The executable `PATH` resolved, or the path the override named.
    pub path: PathBuf,
    /// The first line the compiler prints for `--version`.
    pub version: String,
}

impl fmt::Display for CompilerIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.path.display(), self.version)
    }
}

/// Resolve `compiler`'s identity and check that, given `compile_flags`, it
/// compiles with the profile's floating-point semantics. A wrapper script is
/// opaque on the command line, so the check reads what the compiler itself
/// predefines under those flags: fast math (`__FAST_MATH__`), finite-only math
/// (`__FINITE_MATH_ONLY__`), or a dropped optimisation level (`__OPTIMIZE__`
/// missing although the profile passes `-O2`) means something outside the
/// profile changed the compile, and the build is refused.
pub fn verify_compiler(
    compiler: &str,
    compile_flags: &[String],
) -> Result<CompilerIdentity, String> {
    let path = resolve_executable(compiler)
        .ok_or_else(|| format!("native compiler `{compiler}` was not found"))?;
    let version = tool_command(&path)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cannot run native compiler `{}`: {error}", path.display()))?;
    if !version.status.success() {
        return Err(format!(
            "native compiler `{}` --version failed with {}",
            path.display(),
            version.status
        ));
    }
    let version = compiler_version_text(&version)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .ok_or_else(|| {
            format!(
                "native compiler `{}` printed no --version text",
                path.display()
            )
        })?
        .to_string();

    let macros = tool_command(&path)
        .args(compile_flags)
        .args(["-dM", "-E", "-x", "c", "-"])
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cannot run native compiler `{}`: {error}", path.display()))?;
    if !macros.status.success() {
        return Err(format!(
            "native compiler `{}` rejected the pinned profile {}: {}",
            path.display(),
            compile_flags.join(" "),
            String::from_utf8_lossy(&macros.stderr).trim()
        ));
    }
    let macros = String::from_utf8_lossy(&macros.stdout);
    let defined = |name: &str| {
        macros.lines().find_map(|line| {
            let rest = line.strip_prefix("#define ")?.strip_prefix(name)?;
            (rest.is_empty() || rest.starts_with(' ')).then(|| rest.trim().to_string())
        })
    };
    let mut violations = Vec::new();
    if defined("__FAST_MATH__").is_some() {
        violations.push("__FAST_MATH__ is defined (fast math)");
    }
    if defined("__FINITE_MATH_ONLY__").is_some_and(|value| value != "0") {
        violations.push("__FINITE_MATH_ONLY__ is nonzero (finite-only math)");
    }
    if compile_flags.iter().any(|flag| flag == "-O2") && defined("__OPTIMIZE__").is_none() {
        violations.push("__OPTIMIZE__ is undefined although the profile passes -O2");
    }
    if !violations.is_empty() {
        return Err(format!(
            "native compiler `{}` does not compile with the pinned floating-point profile \
             ({}): {}; a wrapper or configuration is adding flags, so name a compiler that \
             applies the profile unchanged",
            path.display(),
            compile_flags.join(" "),
            violations.join("; ")
        ));
    }
    Ok(CompilerIdentity { path, version })
}

/// `PATH` lookup as the spawned tool will see it: [`tool_command`] passes
/// `PATH` through, so the resolved file is the one that runs.
fn resolve_executable(program: &str) -> Option<PathBuf> {
    let program = Path::new(program);
    if program.components().count() > 1 {
        return std::path::absolute(program)
            .ok()
            .filter(|path| path.is_file());
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(program))
        .find(|candidate| is_executable_file(candidate))
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    metadata.is_file()
}

fn resolve_compiler(override_vars: &[&str]) -> String {
    for var in override_vars {
        if let Ok(override_cc) = std::env::var(var)
            && !override_cc.is_empty()
        {
            return override_cc;
        }
    }
    let candidates = if cfg!(target_os = "macos") {
        ["clang", "cc", "gcc"]
    } else {
        ["gcc", "clang", "cc"]
    };
    for compiler in candidates {
        if tool_command(compiler)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
        {
            return compiler.into();
        }
    }
    candidates[0].into()
}

pub fn is_real_gcc(bin: &str) -> bool {
    let Ok(output) = tool_command(bin).arg("--version").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let text = compiler_version_text(&output);
    !text.contains("Apple clang") && !text.contains("clang version")
}

pub fn is_apple_clang(bin: &str) -> bool {
    let Ok(output) = tool_command(bin).arg("--version").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let text = compiler_version_text(&output);
    text.contains("Apple clang")
}

fn compiler_version_text(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.trim().is_empty() {
        return stdout.into_owned();
    }
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_reference_uses_portable_profile_and_required_libraries() {
        let requirements = CodegenRequirements {
            wants_openmp: true,
            needs_blas: true,
        };
        let strict = strict_reference_toolchain("pinned-cc".into(), requirements);
        assert_eq!(strict.compiler, "pinned-cc");
        assert_eq!(
            strict.compile_flags,
            ["-O2", "-ffp-contract=off", "-fno-fast-math"]
        );
        assert!(!strict.openmp_enabled);
        assert!(!strict.compile_flags.iter().any(|flag| flag == "-fopenmp"));
        assert!(!strict.link_flags.iter().any(|flag| flag == "-fopenmp"));
        assert!(strict.link_flags.contains(&"-lm".to_string()));

        if cfg!(target_os = "macos") {
            assert_eq!(strict.blas_provider, BlasProvider::Accelerate);
            assert!(
                strict
                    .link_flags
                    .windows(2)
                    .any(|flags| flags[0] == "-framework" && flags[1] == "Accelerate")
            );
        } else {
            assert_eq!(strict.blas_provider, BlasProvider::OpenBlas);
            assert!(strict.link_flags.contains(&"-lpthread".to_string()));
            assert!(strict.link_flags.contains(&"-ldl".to_string()));
            assert!(strict.link_flags.contains(&"-lopenblas".to_string()));
        }

        let no_blas =
            strict_reference_toolchain("pinned-cc".into(), CodegenRequirements::default());
        assert_eq!(no_blas.blas_provider, BlasProvider::None);
        assert!(!no_blas.link_flags.contains(&"-lopenblas".to_string()));
        if cfg!(target_os = "macos") {
            assert!(
                no_blas
                    .link_flags
                    .windows(2)
                    .any(|flags| flags[0] == "-framework" && flags[1] == "Accelerate")
            );
        }
    }

    #[test]
    fn product_profile_is_the_strict_profile() {
        // `chelis build` and the agreement gate compile with one profile; the
        // product build adds only OpenMP, and only for a compiler that ships it.
        for needs_blas in [false, true] {
            let requirements = CodegenRequirements {
                wants_openmp: false,
                needs_blas,
            };
            let runtime = runtime_toolchain(requirements);
            let strict = strict_reference_toolchain(runtime.compiler.clone(), requirements);
            assert_eq!(runtime, strict);
            assert!(
                !runtime
                    .compile_flags
                    .iter()
                    .any(|flag| flag.starts_with("-march"))
            );
        }
        let requirements = CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        };
        let runtime = runtime_toolchain(requirements);
        let strict = strict_reference_toolchain(runtime.compiler.clone(), requirements);
        let without_openmp = |flags: &[String]| {
            flags
                .iter()
                .filter(|flag| *flag != "-fopenmp")
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(without_openmp(&runtime.compile_flags), strict.compile_flags);
        assert_eq!(without_openmp(&runtime.link_flags), strict.link_flags);
    }

    /// Variables that change what a C driver compiles without appearing on
    /// its command line.
    const HOSTILE_ENVIRONMENT: &[(&str, &str)] = &[
        ("CCC_OVERRIDE_OPTIONS", "+-ffast-math +-O0"),
        ("NIX_CFLAGS_COMPILE", "-ffast-math -march=native"),
        ("NIX_LDFLAGS", "-lhostile"),
        ("CPATH", "/nonexistent-shadow-include"),
        ("C_INCLUDE_PATH", "/nonexistent-shadow-include"),
        ("SDKROOT", "/nonexistent-sdk"),
        ("GCC_EXEC_PREFIX", "/nonexistent-gcc/"),
        ("COMPILER_PATH", "/nonexistent-compiler-path"),
    ];

    /// Run `test` in a child test process whose environment carries
    /// `HOSTILE_ENVIRONMENT`, so other parallel tests never observe it.
    fn in_hostile_child(test: &str) -> bool {
        if std::env::var_os("CHELIS_TOOLCHAIN_HOSTILE_CHILD").is_some() {
            return false;
        }
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test])
            .env("CHELIS_TOOLCHAIN_HOSTILE_CHILD", "1")
            .envs(HOSTILE_ENVIRONMENT.iter().copied())
            .status()
            .unwrap();
        assert!(status.success(), "{test} failed in the hostile child");
        true
    }

    #[test]
    fn tool_command_passes_only_the_allowlist() {
        if in_hostile_child("toolchain::tests::tool_command_passes_only_the_allowlist") {
            return;
        }
        let output = tool_command("/usr/bin/env").output().unwrap();
        assert!(output.status.success());
        let names: Vec<String> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| line.split_once('=').map(|(name, _)| name.to_string()))
            .collect();
        assert!(!names.is_empty() || std::env::var_os("PATH").is_none());
        for name in &names {
            assert!(
                TOOL_ENVIRONMENT.contains(&name.as_str()),
                "tool environment leaked {name}: {names:?}"
            );
        }
    }

    #[test]
    fn verify_compiler_ignores_hostile_environment() {
        if in_hostile_child("toolchain::tests::verify_compiler_ignores_hostile_environment") {
            return;
        }
        let toolchain = test_toolchain(CodegenRequirements::default());
        let identity = verify_compiler(&toolchain.compiler, &toolchain.compile_flags)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(identity.path.is_absolute(), "{identity}");
        assert!(!identity.version.is_empty(), "{identity}");
    }

    #[cfg(unix)]
    fn wrapper(dir: &Path, name: &str, extra: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        let compiler = test_toolchain(CodegenRequirements::default()).compiler;
        let real = resolve_executable(&compiler).expect("a C compiler on PATH");
        let path = dir.join(name);
        std::fs::write(
            &path,
            format!("#!/bin/sh\nexec '{}' \"$@\" {extra}\n", real.display()),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[cfg(unix)]
    #[test]
    fn verify_compiler_refuses_a_wrapper_that_changes_the_profile() {
        let dir = tempfile::tempdir().unwrap();
        let flags =
            strict_reference_toolchain(String::new(), CodegenRequirements::default()).compile_flags;
        let plain = wrapper(dir.path(), "plain-cc", "");
        verify_compiler(&plain, &flags).unwrap_or_else(|error| panic!("{error}"));
        for (name, extra, expected) in [
            ("fast-cc", "-ffast-math", "__FAST_MATH__"),
            ("finite-cc", "-ffinite-math-only", "__FINITE_MATH_ONLY__"),
            ("unoptimised-cc", "-O0", "__OPTIMIZE__"),
        ] {
            let hostile = wrapper(dir.path(), name, extra);
            let error = verify_compiler(&hostile, &flags).expect_err(name);
            assert!(error.contains(expected), "{name}: {error}");
        }
        let missing = dir.path().join("missing-cc");
        assert!(verify_compiler(missing.to_str().unwrap(), &flags).is_err());
    }

    #[test]
    fn strict_reference_ignores_hostile_environment() {
        // Keep environment changes in a child process so other parallel tests
        // cannot observe the overrides.
        if std::env::var_os("CHELIS_STRICT_REFERENCE_TEST_CHILD").is_none() {
            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "toolchain::tests::strict_reference_ignores_hostile_environment",
                ])
                .env("CHELIS_STRICT_REFERENCE_TEST_CHILD", "1")
                .env("CHELIS_CC", "hostile-cc")
                .env("CC", "another-hostile-cc")
                .env("CFLAGS", "-ffast-math -march=native -fopenmp")
                .env("LDFLAGS", "-fopenmp -lhostile")
                .env("NIX_CFLAGS_COMPILE", "-ffast-math -march=native")
                .env("NIX_LDFLAGS", "-lhostile")
                .env("OMP_NUM_THREADS", "64")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let requirements = CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        };
        let runtime = runtime_toolchain(requirements);
        assert_eq!(runtime.compiler, "hostile-cc");
        assert!(
            !runtime
                .compile_flags
                .iter()
                .any(|flag| flag.starts_with("-march"))
        );

        let strict = strict_reference_toolchain("pinned-cc".into(), requirements);
        assert_eq!(strict.compiler, "pinned-cc");
        assert_eq!(
            strict.compile_flags,
            ["-O2", "-ffp-contract=off", "-fno-fast-math"]
        );
        assert!(!strict.openmp_enabled);
        assert!(
            !strict
                .link_flags
                .iter()
                .any(|flag| flag == "-fopenmp" || flag == "-lhostile")
        );
    }
}
