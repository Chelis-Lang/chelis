//! Cross-platform toolchain resolution for Chelis CPU codegen consumers.

use chelis_crmath::c_source::{Kernel, kernel_text};
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

/// Link the target's BLAS provider when the generated unit calls BLAS
/// (Accelerate on macOS, OpenBLAS elsewhere). Transcendentals need no
/// library: generated units carry their correctly rounded kernels
/// (spec/design/correctly_rounded_math.md section 4.2).
fn blas_link_flags(
    requirements: CodegenRequirements,
    link_flags: &mut Vec<String>,
) -> BlasProvider {
    if !requirements.needs_blas {
        BlasProvider::None
    } else if cfg!(target_os = "macos") {
        link_flags.push("-framework".to_string());
        link_flags.push("Accelerate".to_string());
        BlasProvider::Accelerate
    } else {
        link_flags.push("-lopenblas".to_string());
        BlasProvider::OpenBlas
    }
}

pub fn test_toolchain(requirements: CodegenRequirements) -> NativeToolchain {
    resolve_toolchain(requirements, &["CHELIS_TEST_CC", "CHELIS_CC"])
}

fn resolve_toolchain(requirements: CodegenRequirements, override_vars: &[&str]) -> NativeToolchain {
    let compiler = resolve_compiler(override_vars);
    // OpenMP is gated on `is_real_gcc(&compiler)`. On macOS, `resolve_compiler`
    // returns "clang" (Apple-clang), so this is always false there: Apple-clang
    // does not bundle libomp and `-fopenmp` is unsupported.
    let openmp_enabled = requirements.wants_openmp && is_real_gcc(&compiler);
    pinned_toolchain(compiler, requirements, openmp_enabled)
}

/// The one host compile/link profile, shared by `chelis build` and the
/// agreement gate. Its flags fix floating-point semantics independently of
/// the build machine: `-O2`, no contraction (spec/08-backends.md), no fast
/// math, and no `-march` (the instruction set is the compiler's configured
/// default, never the build host's CPU). OpenMP is the only optional part. It
/// changes throughput, not values, because two conditions hold: the generated
/// parallel loops are element-wise, so no result depends on how iterations are
/// split, and each loop is a region in which every worker thread installs the
/// pinned floating-point environment before its share
/// (`fp_env::pin_parallel_regions`), so no result depends on which thread, or
/// that thread's prior state, computed it.
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

    let blas_provider = blas_link_flags(requirements, &mut link_flags);

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

/// Why [`verify_compiler`] did not accept a compiler. A missing compiler is
/// its own variant so the caller reports it through the same tool-not-found
/// diagnostic as every other native tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilerCheckError {
    /// Neither the named path nor a `PATH` lookup found an executable.
    NotFound(String),
    /// The compiler ran but did not identify itself or apply the profile.
    Refused(String),
}

impl fmt::Display for CompilerCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(compiler) => write!(f, "native compiler `{compiler}` was not found"),
            Self::Refused(reason) => f.write_str(reason),
        }
    }
}

/// Resolve `compiler`'s identity and check that, given `compile_flags`, it
/// compiles with the profile's floating-point semantics. A wrapper script is
/// opaque on the command line, so the check observes the compiler itself, in
/// two steps. First it reads what the compiler predefines under those flags:
/// fast math (`__FAST_MATH__`), finite-only math (`__FINITE_MATH_ONLY__`), or a
/// dropped optimisation level (`__OPTIMIZE__` missing although the profile
/// passes `-O2`). No macro reveals contraction or reassociation, so it then
/// compiles and runs [`CANARY_MAIN`] with the same flags and compares the bits
/// it prints with [`CANARY_CASES`]. Any difference means something outside the
/// profile changed the compile, and the build is refused.
pub fn verify_compiler(
    compiler: &str,
    compile_flags: &[String],
) -> Result<CompilerIdentity, CompilerCheckError> {
    let path = resolve_executable(compiler)
        .ok_or_else(|| CompilerCheckError::NotFound(compiler.to_string()))?;
    check_compiler(path, compile_flags).map_err(CompilerCheckError::Refused)
}

fn check_compiler(path: PathBuf, compile_flags: &[String]) -> Result<CompilerIdentity, String> {
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
    run_canary(&path, compile_flags)?;
    Ok(CompilerIdentity { path, version })
}

/// The canary's inputs, as `f64` bit patterns it reads from its arguments, so
/// the compiler cannot fold any check away.
const CANARY_INPUTS: [u64; 10] = [
    // A worst case for both kernels: `chelis_cr_exp` and `chelis_cr_tanh`
    // round it wrongly once their polynomial evaluation is contracted.
    0x3fd3_3333_3333_7111,
    // a = b = 1 + 2^-27 and c = -(1 + 2^-26): a*b + c is 0 when the product
    // is rounded and 2^-54 when it is fused.
    0x3ff0_0000_0200_0000,
    0x3ff0_0000_0200_0000,
    0xbff0_0000_0400_0000,
    // p = 2^53 and q = 1: (p + q) - p is 0, and q once reassociated.
    0x4340_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    // n = 5: n / 3 differs from n * (1/3) in the last bit.
    0x4014_0000_0000_0000,
    // z = -0: z + 0 is +0, and -0 once signed zeros are ignored.
    0x8000_0000_0000_0000,
    // s = 2^-1000 and t = 2^-60: s * t is the subnormal 2^-1060, and 0 under
    // flush-to-zero (a `crtfastmath` startup, for example).
    0x0170_0000_0000_0000,
    0x3c30_0000_0000_0000,
];

/// What [`CANARY_MAIN`] prints for [`CANARY_INPUTS`] under the profile: the
/// check, the operation it evaluates, and the result's bits. The kernel rows
/// are the correctly rounded results (checked against 300-bit mpmath).
const CANARY_CASES: [(&str, &str, u64); 7] = [
    ("exp", "chelis_cr_exp(0x1.3333333337111p-2)", 0x3ff5_9905_8c8c_2f76),
    ("tanh", "chelis_cr_tanh(0x1.3333333337111p-2)", 0x3fd2_a4dd_a7d9_4d98),
    ("contraction", "(1+2^-27)*(1+2^-27) - (1+2^-26)", 0),
    ("reassociation", "(2^53 + 1) - 2^53", 0),
    ("reciprocal", "5 / 3", 0x3ffa_aaaa_aaaa_aaab),
    ("signed-zero", "-0 + 0", 0),
    ("subnormal", "2^-1000 * 2^-60", 0x4000),
];

/// The canary's driver. The kernels come from `chelis-crmath`, the same text
/// every generated unit carries.
const CANARY_MAIN: &str = r#"
#include <stdio.h>
#include <stdlib.h>

static double chelis_canary_arg(const char *text) {
    uint64_t bits = strtoull(text, NULL, 16);
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static void chelis_canary_print(const char *check, double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof bits);
    printf("%s %016llx\n", check, (unsigned long long)bits);
}

int main(int argc, char **argv) {
    if (argc != 11) {
        return 2;
    }
    double x = chelis_canary_arg(argv[1]);
    double a = chelis_canary_arg(argv[2]);
    double b = chelis_canary_arg(argv[3]);
    double c = chelis_canary_arg(argv[4]);
    double p = chelis_canary_arg(argv[5]);
    double q = chelis_canary_arg(argv[6]);
    double n = chelis_canary_arg(argv[7]);
    double z = chelis_canary_arg(argv[8]);
    double s = chelis_canary_arg(argv[9]);
    double t = chelis_canary_arg(argv[10]);
    chelis_canary_print("exp", chelis_cr_exp(x));
    chelis_canary_print("tanh", chelis_cr_tanh(x));
    chelis_canary_print("contraction", a * b + c);
    double sum = p + q;
    chelis_canary_print("reassociation", sum - p);
    chelis_canary_print("reciprocal", n / 3.0);
    chelis_canary_print("signed-zero", z + 0.0);
    chelis_canary_print("subnormal", s * t);
    return 0;
}
"#;

/// A scratch directory under the system temporary directory, removed on drop.
struct CanaryDir(PathBuf);

impl CanaryDir {
    fn create() -> Result<Self, String> {
        let base = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.subsec_nanos());
        for attempt in 0..100u32 {
            let dir = base.join(format!(
                "chelis-compiler-canary-{}-{nanos}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(Self(dir)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(format!(
                        "cannot create the compiler canary directory in {}: {error}",
                        base.display()
                    ));
                }
            }
        }
        Err(format!(
            "cannot create the compiler canary directory in {}",
            base.display()
        ))
    }
}

impl Drop for CanaryDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Compile [`CANARY_MAIN`] with `compile_flags`, run it, and refuse the
/// compiler unless it prints exactly [`CANARY_CASES`].
fn run_canary(path: &Path, compile_flags: &[String]) -> Result<(), String> {
    let refuse = |what: String| {
        format!(
            "native compiler `{}` does not compile with the pinned floating-point profile \
             ({}): {what}; a wrapper or configuration is adding flags, so name a compiler \
             that applies the profile unchanged",
            path.display(),
            compile_flags.join(" ")
        )
    };
    let dir = CanaryDir::create()?;
    let source = dir.0.join("chelis-compiler-canary.c");
    let program = dir.0.join("chelis-compiler-canary");
    let text = kernel_text(&[Kernel::ExpF64, Kernel::TanhF64]) + CANARY_MAIN;
    std::fs::write(&source, text)
        .map_err(|error| format!("cannot write the compiler canary: {error}"))?;
    let compiled = tool_command(path)
        .args(compile_flags)
        .arg(&source)
        .arg("-o")
        .arg(&program)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cannot run native compiler `{}`: {error}", path.display()))?;
    if !compiled.status.success() || !program.is_file() {
        return Err(refuse(format!(
            "it did not build the floating-point canary ({}): {}",
            compiled.status,
            String::from_utf8_lossy(&compiled.stderr).trim()
        )));
    }
    let ran = Command::new(&program)
        .env_clear()
        .args(CANARY_INPUTS.map(|bits| format!("{bits:016x}")))
        .stdin(Stdio::null())
        .output()
        .map_err(|error| refuse(format!("its floating-point canary did not run: {error}")))?;
    let printed = String::from_utf8_lossy(&ran.stdout);
    let mut lines = printed.lines();
    let mut violations = Vec::new();
    for (check, operation, expected) in CANARY_CASES {
        let got = lines
            .next()
            .and_then(|line| line.strip_prefix(check)?.strip_prefix(' '))
            .and_then(|bits| u64::from_str_radix(bits, 16).ok());
        match got {
            Some(got) if got == expected => {}
            Some(got) => violations.push(format!(
                "{check}: {operation} gave {:?} (0x{got:016x}) where the profile gives {:?} \
                 (0x{expected:016x})",
                f64::from_bits(got),
                f64::from_bits(expected)
            )),
            None => {
                return Err(refuse(format!(
                    "its floating-point canary exited with {} and printed {:?}",
                    ran.status,
                    printed.trim()
                )));
            }
        }
    }
    if !ran.status.success() || lines.next().is_some() {
        return Err(refuse(format!(
            "its floating-point canary exited with {} and printed {:?}",
            ran.status,
            printed.trim()
        )));
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(refuse(format!(
            "the floating-point canary disagrees with the profile ({})",
            violations.join("; ")
        )))
    }
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
        assert!(
            !no_blas
                .link_flags
                .windows(2)
                .any(|flags| flags[0] == "-framework" && flags[1] == "Accelerate"),
            "a unit without BLAS links no math or BLAS framework"
        );
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
        // Contraction changes values only where the target has a fused
        // multiply-add instruction, which the x86-64 baseline lacks; `-mfma`
        // is what a `-march=native` wrapper would add there.
        let contract = if cfg!(target_arch = "x86_64") {
            "-mfma -ffp-contract=fast"
        } else {
            "-ffp-contract=fast"
        };
        for (name, extra, expected) in [
            ("fast-cc", "-ffast-math", "__FAST_MATH__"),
            ("finite-cc", "-ffinite-math-only", "__FINITE_MATH_ONLY__"),
            ("unoptimised-cc", "-O0", "__OPTIMIZE__"),
            ("contract-cc", contract, "contraction: "),
            ("unsafe-cc", "-funsafe-math-optimizations", "reassociation: "),
            ("reciprocal-cc", "-freciprocal-math", "reciprocal: "),
            ("signed-zero-cc", "-fno-signed-zeros", "signed-zero: "),
            (
                "associative-cc",
                "-fassociative-math -fno-signed-zeros -fno-trapping-math",
                "reassociation: ",
            ),
        ] {
            let hostile = wrapper(dir.path(), name, extra);
            let error = verify_compiler(&hostile, &flags).expect_err(name);
            assert!(
                matches!(&error, CompilerCheckError::Refused(reason) if reason.contains(expected)),
                "{name}: {error}"
            );
        }
        let missing = dir.path().join("missing-cc");
        assert_eq!(
            verify_compiler(missing.to_str().unwrap(), &flags),
            Err(CompilerCheckError::NotFound(
                missing.to_str().unwrap().to_string()
            ))
        );
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
