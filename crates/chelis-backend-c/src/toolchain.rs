//! Cross-platform toolchain resolution for Chelis CPU codegen consumers.

use crate::fp_env;
use chelis_crmath::c_source::{Kernel, kernel_text};
use chelis_crmath::profile;
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
/// passes `-O2`). No macro reveals contraction, reassociation, or a NaN or
/// infinity assumption, so it then compiles and runs the canary
/// ([`canary_source`]) with the same flags and compares the bits it prints with
/// the profile's obligation table (`chelis_crmath::profile`): every kernel row of
/// the MPFR fixtures and every arithmetic, comparison, conversion, and
/// expression-shape row. Any difference means something outside the profile
/// changed the compile, and the build is refused.
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

    let accepted = AcceptedCompiler {
        path: path.clone(),
        version: version.clone(),
        compile_flags: compile_flags.to_vec(),
        executable: path
            .metadata()
            .ok()
            .map(|metadata| (metadata.len(), metadata.modified().ok())),
    };
    if ACCEPTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&accepted)
    {
        return Ok(CompilerIdentity { path, version });
    }

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
    ACCEPTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(accepted);
    Ok(CompilerIdentity { path, version })
}

/// A compiler this process has already accepted for these flags. Compiling the
/// canary costs about half a second (it carries every kernel), so a process that
/// builds repeatedly checks each compiler once. The key is the compiler's identity
/// and argument vector plus the executable's size and modification time, so a wrapper
/// edited in place is checked again. Nothing persists across processes.
#[derive(PartialEq, Eq)]
struct AcceptedCompiler {
    path: PathBuf,
    version: String,
    compile_flags: Vec<String>,
    executable: Option<(u64, Option<std::time::SystemTime>)>,
}

static ACCEPTED: std::sync::Mutex<Vec<AcceptedCompiler>> = std::sync::Mutex::new(Vec::new());

/// The compiler canary: every kernel, generated code's NaN finalization, and the
/// driver `chelis_crmath::profile` generates from the profile's obligation table.
/// The kernel text and the NaN helpers are the bytes generated units carry, so the
/// canary observes what the compiler does to them.
fn canary_source() -> String {
    let mut text = kernel_text(&Kernel::ALL);
    text.push_str(profile::canary_prelude());
    // The NaN helpers' bit casts, which generated units take from the runtime
    // header.
    text.push_str(
        "#define chelis_f32_from_bits chelis_canary_f32\n\
         #define chelis_f64_from_bits chelis_canary_f64\n",
    );
    for line in fp_env::helper_lines() {
        text.push_str(line);
        text.push('\n');
    }
    text.push_str(
        "#define chelis_canary_finalize_f32 __chelis_nan_f32\n\
         #define chelis_canary_finalize_f64 __chelis_nan_f64\n",
    );
    text.push_str(&profile::canary_driver());
    text
}

/// A scratch directory under the system temporary directory, removed on drop.
struct CanaryDir(PathBuf);

impl CanaryDir {
    fn create() -> Result<Self, String> {
        let base = std::env::temp_dir();
        // A process-wide sequence keeps concurrent checks in one process apart;
        // `create_dir` failing on an existing name keeps processes apart.
        static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        for _ in 0..100 {
            let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = base.join(format!(
                "chelis-compiler-canary-{}-{sequence}",
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

/// Compile [`canary_source`] with `compile_flags`, run it on every C row of the
/// obligation table, and refuse the compiler unless it prints exactly the table's
/// bits.
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
    std::fs::write(&source, canary_source())
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
    use std::io::Write as _;
    let mut child = Command::new(&program)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| refuse(format!("its floating-point canary did not run: {error}")))?;
    let input = profile::canary_input();
    let mut stdin = child.stdin.take().expect("the canary's stdin is piped");
    // Write from a thread so a canary that stops reading cannot deadlock against
    // its full stdout pipe.
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let ran = child
        .wait_with_output()
        .map_err(|error| refuse(format!("its floating-point canary did not run: {error}")))?;
    let written = writer.join().expect("the canary input writer does not panic");
    let printed = String::from_utf8_lossy(&ran.stdout);
    if !ran.status.success() || written.is_err() {
        return Err(refuse(format!(
            "its floating-point canary exited with {}",
            ran.status
        )));
    }
    let mismatches = profile::canary_mismatches(&printed).map_err(refuse)?;
    if mismatches.is_empty() {
        return Ok(());
    }
    let reason = format!(
        "the floating-point canary disagrees with the profile on {} of {} rows: {}",
        mismatches.len(),
        profile::canary_rows().count(),
        profile::describe(&mismatches)
    );
    Err(refuse(reason))
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
    fn verify_compiler_accepts_the_plain_compiler_and_reports_a_missing_one() {
        let dir = tempfile::tempdir().unwrap();
        let flags =
            strict_reference_toolchain(String::new(), CodegenRequirements::default()).compile_flags;
        let plain = wrapper(dir.path(), "plain-cc", "");
        verify_compiler(&plain, &flags).unwrap_or_else(|error| panic!("{error}"));
        let missing = dir.path().join("missing-cc");
        assert_eq!(
            verify_compiler(missing.to_str().unwrap(), &flags),
            Err(CompilerCheckError::NotFound(
                missing.to_str().unwrap().to_string()
            ))
        );
    }

    #[cfg(unix)]
    #[test]
    fn verify_compiler_rechecks_a_wrapper_edited_after_acceptance() {
        let dir = tempfile::tempdir().unwrap();
        let flags =
            strict_reference_toolchain(String::new(), CodegenRequirements::default()).compile_flags;
        let edited = wrapper(dir.path(), "edited-cc", "");
        verify_compiler(&edited, &flags).unwrap_or_else(|error| panic!("{error}"));
        verify_compiler(&edited, &flags).unwrap_or_else(|error| panic!("{error}"));
        wrapper(dir.path(), "edited-cc", "-fno-honor-nans");
        let result = verify_compiler(&edited, &flags);
        assert!(
            matches!(&result, Err(CompilerCheckError::Refused(reason)) if reason.contains("canonical-nan")),
            "{result:?}"
        );
    }

    /// What the compiler check must conclude about a wrapper that appends
    /// flags to the profile.
    #[cfg(unix)]
    enum Verdict {
        /// Refused, with this text in the reason.
        Refused(&'static str),
        /// Accepted because the flag changes no compiled instruction of the
        /// canary: its assembly is byte-identical to the plain compiler's.
        Harmless,
    }

    /// Check one wrapper. Contraction changes values only where the target
    /// has a fused multiply-add instruction, which the x86-64 baseline lacks;
    /// `-mfma` is what a `-march=native` wrapper would add there.
    #[cfg(unix)]
    fn check_wrapper(name: &str, extra: &str, verdict: Verdict) {
        let dir = tempfile::tempdir().unwrap();
        let flags =
            strict_reference_toolchain(String::new(), CodegenRequirements::default()).compile_flags;
        let extra = if cfg!(target_arch = "x86_64") && extra.starts_with("-ffp-contract") {
            format!("-mfma {extra}")
        } else {
            extra.to_string()
        };
        let hostile = wrapper(dir.path(), name, &extra);
        let result = verify_compiler(&hostile, &flags);
        match verdict {
            Verdict::Refused(expected) => assert!(
                matches!(&result, Err(CompilerCheckError::Refused(reason)) if reason.contains(expected)),
                "{name} ({extra}) should be refused with `{expected}`: {result:?}"
            ),
            Verdict::Harmless => {
                result.unwrap_or_else(|error| panic!("{name} ({extra}): {error}"));
                let plain = wrapper(dir.path(), "plain-cc", "");
                assert_eq!(
                    canary_assembly(&hostile, &flags, dir.path()),
                    canary_assembly(&plain, &flags, dir.path()),
                    "{name} ({extra}) is accepted, so it must not change the canary's code"
                );
            }
        }
    }

    /// The canary translation unit compiled to assembly by `compiler`.
    #[cfg(unix)]
    fn canary_assembly(compiler: &str, flags: &[String], dir: &Path) -> String {
        let source = dir.join("canary-assembly.c");
        let output = dir.join("canary-assembly.s");
        std::fs::write(&source, canary_source()).unwrap();
        let status = tool_command(compiler)
            .args(flags)
            .args(["-S", "-o"])
            .arg(&output)
            .arg(&source)
            .status()
            .unwrap();
        assert!(status.success(), "{compiler} did not compile the canary");
        std::fs::read_to_string(&output).unwrap()
    }

    /// One test per flag a wrapper might append (chelis#2957 round-1
    /// verification sweep). Each value-changing flag is refused by the check
    /// that observes it; each accepted flag is shown not to change the
    /// canary's compiled code.
    macro_rules! wrapper_verdicts {
        ($($test:ident: $extra:literal => $verdict:expr;)*) => {
            $(
                #[cfg(unix)]
                #[test]
                fn $test() {
                    check_wrapper(stringify!($test), $extra, $verdict);
                }
            )*
        };
    }

    wrapper_verdicts! {
        verify_compiler_wrapper_fast_math: "-ffast-math" => Verdict::Refused("__FAST_MATH__");
        verify_compiler_wrapper_finite_math_only: "-ffinite-math-only" => Verdict::Refused("__FINITE_MATH_ONLY__");
        verify_compiler_wrapper_unoptimised: "-O0" => Verdict::Refused("__OPTIMIZE__");
        verify_compiler_wrapper_no_honor_nans: "-fno-honor-nans" => Verdict::Refused("canonical-nan");
        // Breaks only the kernels' infinity cases (5 rows on Apple clang 21).
        verify_compiler_wrapper_no_honor_infinities: "-fno-honor-infinities" => Verdict::Refused("correct-rounding");
        verify_compiler_wrapper_fp_contract_fast: "-ffp-contract=fast" => Verdict::Refused("no-contraction");
        verify_compiler_wrapper_fp_contract_on: "-ffp-contract=on" => Verdict::Refused("no-contraction");
        verify_compiler_wrapper_unsafe_math: "-funsafe-math-optimizations" => Verdict::Refused("floating-point canary");
        verify_compiler_wrapper_associative_trio: "-fassociative-math -fno-signed-zeros -fno-trapping-math"
            => Verdict::Refused("floating-point canary");
        verify_compiler_wrapper_reciprocal_math: "-freciprocal-math" => Verdict::Refused("no-value-changing-optimization");
        verify_compiler_wrapper_no_signed_zeros: "-fno-signed-zeros" => Verdict::Refused("signed-zero");
        verify_compiler_wrapper_fp_model_fast: "-ffp-model=fast" => Verdict::Refused("floating-point canary");
        verify_compiler_wrapper_fp_eval_method_double: "-ffp-eval-method=double" => Verdict::Refused("did not build");
        // Inert alone: clang and gcc reassociate only when signed zeros and
        // trapping are also given up.
        verify_compiler_wrapper_associative_math: "-fassociative-math" => Verdict::Harmless;
        // Trapping is not observable: [05-OP-46] makes status flags
        // unobservable and the profile installs no trap.
        verify_compiler_wrapper_no_trapping_math: "-fno-trapping-math" => Verdict::Harmless;
    }

    // With SSE2 or AArch64 floating point, FLT_EVAL_METHOD is 0 and there is no
    // excess precision to keep or discard. On x87 the kernels' FLT_EVAL_METHOD
    // guard refuses it instead.
    #[cfg(all(unix, any(target_arch = "aarch64", target_arch = "x86_64")))]
    wrapper_verdicts! {
        verify_compiler_wrapper_excess_precision_fast: "-fexcess-precision=fast" => Verdict::Harmless;
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
