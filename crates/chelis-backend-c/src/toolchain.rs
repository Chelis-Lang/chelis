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
/// macOS the caller's `SDKROOT` and `DEVELOPER_DIR` are absent too; instead
/// [`tool_command`] sets `SDKROOT` to the SDK of the `xcode-select` default
/// ([`selected_sdk`]), the one the `/usr/bin` shims choose. A compiler named
/// directly in `CHELIS_CC`, such as another Xcode's `clang`, has no SDK of its
/// own, so that is how it finds the system headers and libraries.
pub const TOOL_ENVIRONMENT: &[&str] = &["PATH", "TMPDIR"];

/// A command for a native build tool with the environment cleared except
/// for [`TOOL_ENVIRONMENT`], plus `SDKROOT` on macOS ([`selected_sdk`]).
pub fn tool_command(program: impl AsRef<OsStr>) -> Command {
    let mut command = allowlisted_command(program);
    if let Some(sdk) = selected_sdk() {
        command.env("SDKROOT", sdk);
    }
    command
}

fn allowlisted_command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    command.env_clear();
    for name in TOOL_ENVIRONMENT {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
}

/// On macOS, the SDK of the `xcode-select` default: what `xcrun --show-sdk-path`
/// prints under [`TOOL_ENVIRONMENT`] alone, so the caller's `SDKROOT` and
/// `DEVELOPER_DIR` cannot choose it. `None` elsewhere, or when `xcrun` finds no
/// SDK; a compile then fails with the compiler's own diagnostic.
fn selected_sdk() -> Option<&'static OsStr> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    SELECTED_SDK.as_deref()
}

static SELECTED_SDK: std::sync::LazyLock<Option<std::ffi::OsString>> =
    std::sync::LazyLock::new(|| {
        let output = allowlisted_command("/usr/bin/xcrun")
            .arg("--show-sdk-path")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        let path = String::from_utf8(output.stdout).ok()?;
        let path = path.trim();
        (output.status.success() && !path.is_empty()).then(|| path.into())
    });

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

/// The argument vector of a profile link: `compile_flags`, the `inputs`, the
/// profile's `link_flags`, and the output. `chelis build` links executables
/// with it and [`verify_compiler`] links the canary with it, so the canary is
/// linked exactly as a real build is (the profile's `-lm` included).
pub fn link_args(
    compile_flags: &[String],
    inputs: &[&OsStr],
    link_flags: &[String],
    output: &OsStr,
) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = compile_flags.iter().map(Into::into).collect();
    args.extend(inputs.iter().map(|input| input.to_os_string()));
    args.extend(link_flags.iter().map(Into::into));
    args.push("-o".into());
    args.push(output.to_os_string());
    args
}

/// Resolve `compiler`'s identity and check that, given `compile_flags` and
/// `link_flags`, it compiles against the carried runtime archive's C library
/// ([`check_c_library`]) and with the profile's floating-point semantics. A wrapper
/// script is opaque on the command line, so the check observes the compiler
/// itself. After the C library, it reads what the compiler predefines under those flags:
/// fast math (`__FAST_MATH__`), finite-only math (`__FINITE_MATH_ONLY__`), or a
/// dropped optimisation level (`__OPTIMIZE__` missing although the profile
/// passes `-O2`). No macro reveals contraction, reassociation, or a NaN or
/// infinity assumption, so it then compiles and runs the canary
/// ([`canary_source`]) with the same flags, linked by [`link_args`], and compares the bits it prints with
/// the profile's obligation table (`chelis_crmath::profile`): every kernel row of
/// the MPFR fixtures and every arithmetic, comparison, conversion, and
/// expression-shape row. Any difference means something outside the profile
/// changed the compile, and the build is refused.
pub fn verify_compiler(
    compiler: &str,
    compile_flags: &[String],
    link_flags: &[String],
) -> Result<CompilerIdentity, CompilerCheckError> {
    let path = resolve_executable(compiler)
        .ok_or_else(|| CompilerCheckError::NotFound(compiler.to_string()))?;
    check_compiler(path, compile_flags, link_flags).map_err(CompilerCheckError::Refused)
}

fn check_compiler(
    path: PathBuf,
    compile_flags: &[String],
    link_flags: &[String],
) -> Result<CompilerIdentity, String> {
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
        link_flags: link_flags.to_vec(),
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
    check_c_library(&path, compile_flags)?;

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
    run_canary(&path, compile_flags, link_flags)?;
    ACCEPTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(accepted);
    Ok(CompilerIdentity { path, version })
}

/// The Linux C library a runtime archive was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CLibrary {
    Glibc,
    Musl,
}

impl CLibrary {
    fn name(self) -> &'static str {
        match self {
            CLibrary::Glibc => "glibc",
            CLibrary::Musl => "musl",
        }
    }
}

/// The C library of the runtime archive this compiler carries. The archive comes
/// from the compiler's own build (spec/08-backends.md section 2.1), so it is the
/// Rust target environment the compiler was built for; off Linux there is no
/// second C library to tell apart.
const CARRIED_C_LIBRARY: Option<CLibrary> = if cfg!(all(target_os = "linux", target_env = "gnu")) {
    Some(CLibrary::Glibc)
} else if cfg!(all(target_os = "linux", target_env = "musl")) {
    Some(CLibrary::Musl)
} else {
    None
};

/// Refuse a compiler that compiles against another C library than the carried
/// runtime archive's (spec/08-backends.md section 7). Every native build links
/// that archive, and the link would fail on symbols only its own C library
/// defines, so the check runs before anything compiles.
fn check_c_library(path: &Path, compile_flags: &[String]) -> Result<(), String> {
    let Some(carried) = CARRIED_C_LIBRARY else {
        return Ok(());
    };
    match c_library_refusal(path, carried, compiles_against_glibc(path, compile_flags)?) {
        Some(refusal) => Err(refusal),
        None => Ok(()),
    }
}

/// Whether `path` compiles against glibc under `compile_flags`: its `<stdint.h>`
/// defines `__GLIBC__`, which glibc's headers do and musl's do not. musl defines no
/// macro of its own, so the headers tell glibc from every other C library and no
/// more; a compiler like `musl-gcc`, which reports a `-gnu` target but compiles
/// against musl, reads as what its headers are.
fn compiles_against_glibc(path: &Path, compile_flags: &[String]) -> Result<bool, String> {
    use std::io::Write as _;
    let cannot_run =
        |error: std::io::Error| format!("cannot run native compiler `{}`: {error}", path.display());
    let mut child = tool_command(path)
        .args(compile_flags)
        .args(["-dM", "-E", "-x", "c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(cannot_run)?;
    if let Some(mut stdin) = child.stdin.take() {
        match stdin.write_all(b"#include <stdint.h>\n") {
            Ok(()) => {}
            // A compiler that exits without reading its input fails the status
            // check below.
            Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => {}
            Err(error) => return Err(cannot_run(error)),
        }
    }
    let output = child.wait_with_output().map_err(cannot_run)?;
    if !output.status.success() {
        return Err(format!(
            "native compiler `{}` cannot preprocess `#include <stdint.h>`, so its C library \
             headers are missing: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).lines().any(|line| {
        line.strip_prefix("#define __GLIBC__")
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '))
    }))
}

/// The refusal for a compiler whose headers are glibc's (`glibc_headers`) or not
/// while the carried runtime archive was built for `carried`, or nothing when they
/// agree. A musl archive accepts any compiler whose headers are not glibc's, the
/// most the headers tell.
fn c_library_refusal(path: &Path, carried: CLibrary, glibc_headers: bool) -> Option<String> {
    let found = match (carried, glibc_headers) {
        (CLibrary::Glibc, true) | (CLibrary::Musl, false) => return None,
        (CLibrary::Glibc, false) => {
            "a C library other than glibc (its `<stdint.h>` does not define `__GLIBC__`)"
        }
        (CLibrary::Musl, true) => "glibc (its `<stdint.h>` defines `__GLIBC__`)",
    };
    Some(format!(
        "native compiler `{}` compiles against {found}, but this chelis carries a runtime \
         archive built for {carried}, which every native build links; name a C compiler for \
         {carried}, for example with CHELIS_CC",
        path.display(),
        carried = carried.name()
    ))
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
    link_flags: Vec<String>,
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

/// The canary's compile-and-link arguments: [`link_args`] with the canary source
/// as the one input.
fn canary_args(
    compile_flags: &[String],
    link_flags: &[String],
    source: &Path,
    program: &Path,
) -> Vec<std::ffi::OsString> {
    link_args(
        compile_flags,
        &[source.as_os_str()],
        link_flags,
        program.as_os_str(),
    )
}

/// Compile and link [`canary_source`] with the profile's flags, run it on every C
/// row of the obligation table, and refuse the compiler unless it prints exactly
/// the table's bits.
fn run_canary(path: &Path, compile_flags: &[String], link_flags: &[String]) -> Result<(), String> {
    let mismatches = canary_mismatches(path, compile_flags, link_flags)?;
    if mismatches.is_empty() {
        return Ok(());
    }
    let reason = format!(
        "the floating-point canary disagrees with the profile on {} of {} rows: {}",
        mismatches.len(),
        profile::canary_rows().count(),
        profile::describe(&mismatches)
    );
    Err(refusal(path, compile_flags, &reason))
}

/// The refusal text for a compiler whose canary did not reproduce the profile.
fn refusal(path: &Path, compile_flags: &[String], what: &str) -> String {
    format!(
        "native compiler `{}` does not compile with the pinned floating-point profile \
         ({}): {what}; a wrapper or configuration is adding flags, so name a compiler \
         that applies the profile unchanged",
        path.display(),
        compile_flags.join(" ")
    )
}

/// The obligation-table rows the canary, built by `path` with these flags, computes
/// differently from the table. An error when the canary does not build, run, or
/// print one readable result per row.
fn canary_mismatches(
    path: &Path,
    compile_flags: &[String],
    link_flags: &[String],
) -> Result<Vec<profile::Mismatch>, String> {
    let refuse = |what: String| refusal(path, compile_flags, &what);
    let dir = CanaryDir::create()?;
    let source = dir.0.join("chelis-compiler-canary.c");
    let program = dir.0.join("chelis-compiler-canary");
    std::fs::write(&source, canary_source())
        .map_err(|error| format!("cannot write the compiler canary: {error}"))?;
    let compiled = tool_command(path)
        .args(canary_args(compile_flags, link_flags, &source, &program))
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("cannot run native compiler `{}`: {error}", path.display()))?;
    if !compiled.status.success() || !program.is_file() {
        // Not building is not evidence of added flags, so this refusal does not
        // blame a wrapper; the compiler's own diagnostic names what failed, such
        // as a missing builtin or the amalgamation's `#error` guards.
        return Err(format!(
            "native compiler `{}` did not build the floating-point canary, which \
             compiles every correctly rounded kernel under the pinned profile ({}); \
             set CHELIS_CC to a C compiler that builds it ({}): {}",
            path.display(),
            compile_flags.join(" "),
            compiled.status,
            String::from_utf8_lossy(&compiled.stderr).trim()
        ));
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
    let written = writer
        .join()
        .expect("the canary input writer does not panic");
    let printed = String::from_utf8_lossy(&ran.stdout);
    if !ran.status.success() || written.is_err() {
        return Err(refuse(format!(
            "its floating-point canary exited with {}",
            ran.status
        )));
    }
    profile::canary_mismatches(&printed).map_err(refuse)
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
        ("DEVELOPER_DIR", "/nonexistent-developer-dir"),
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
        let variables: Vec<(String, String)> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| {
                let (name, value) = line.split_once('=')?;
                Some((name.to_string(), value.to_string()))
            })
            .collect();
        assert!(!variables.is_empty() || std::env::var_os("PATH").is_none());
        for (name, value) in &variables {
            if name == "SDKROOT" {
                // The xcode-select default's SDK, never the caller's.
                assert_eq!(Some(OsStr::new(value)), selected_sdk(), "{variables:?}");
                continue;
            }
            assert!(
                TOOL_ENVIRONMENT.contains(&name.as_str()),
                "tool environment leaked {name}: {variables:?}"
            );
        }
    }

    /// `/usr/bin/clang` is a shim that hands the compiler it runs an SDK; that
    /// compiler named directly, as `CHELIS_CC` may name another Xcode's, has none.
    #[cfg(target_os = "macos")]
    #[test]
    fn verify_compiler_accepts_an_xcode_clang_named_directly() {
        if in_hostile_child(
            "toolchain::tests::verify_compiler_accepts_an_xcode_clang_named_directly",
        ) {
            return;
        }
        // Under the allowlist, so the hostile DEVELOPER_DIR cannot steer the lookup.
        let found = allowlisted_command("/usr/bin/xcrun")
            .args(["--find", "clang"])
            .output()
            .unwrap();
        assert!(found.status.success(), "xcrun --find clang failed");
        let clang = String::from_utf8(found.stdout).unwrap().trim().to_string();
        assert_ne!(clang, "/usr/bin/clang", "xcrun named the shim itself");
        let toolchain = test_toolchain(CodegenRequirements::default());
        verify_compiler(&clang, &toolchain.compile_flags, &toolchain.link_flags)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn verify_compiler_ignores_hostile_environment() {
        if in_hostile_child("toolchain::tests::verify_compiler_ignores_hostile_environment") {
            return;
        }
        let toolchain = test_toolchain(CodegenRequirements::default());
        let identity = verify_compiler(
            &toolchain.compiler,
            &toolchain.compile_flags,
            &toolchain.link_flags,
        )
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
        let reference = strict_reference_toolchain(String::new(), CodegenRequirements::default());
        let (flags, links) = (reference.compile_flags, reference.link_flags);
        let plain = wrapper(dir.path(), "plain-cc", "");
        verify_compiler(&plain, &flags, &links).unwrap_or_else(|error| panic!("{error}"));
        let missing = dir.path().join("missing-cc");
        assert_eq!(
            verify_compiler(missing.to_str().unwrap(), &flags, &links),
            Err(CompilerCheckError::NotFound(
                missing.to_str().unwrap().to_string()
            ))
        );
    }

    #[cfg(unix)]
    #[test]
    fn verify_compiler_rechecks_a_wrapper_edited_after_acceptance() {
        let dir = tempfile::tempdir().unwrap();
        let reference = strict_reference_toolchain(String::new(), CodegenRequirements::default());
        let (flags, links) = (reference.compile_flags, reference.link_flags);
        let edited = wrapper(dir.path(), "edited-cc", "");
        verify_compiler(&edited, &flags, &links).unwrap_or_else(|error| panic!("{error}"));
        verify_compiler(&edited, &flags, &links).unwrap_or_else(|error| panic!("{error}"));
        // Both clang and gcc accept the flag, and the macro check refuses it.
        wrapper(dir.path(), "edited-cc", "-ffinite-math-only");
        let result = verify_compiler(&edited, &flags, &links);
        assert!(
            matches!(&result, Err(CompilerCheckError::Refused(reason)) if reason.contains("__FINITE_MATH_ONLY__")),
            "{result:?}"
        );
    }

    /// What the compiler check must conclude about a wrapper that appends
    /// flags to the profile.
    #[cfg(unix)]
    enum Verdict {
        /// Refused, with this text in the reason.
        Refused(&'static str),
        /// Accepted, and the flag changes no compiled instruction of the
        /// canary on clang or gcc: its assembly is byte-identical to the plain
        /// compiler's.
        Harmless,
        /// Whether the flag changes a value depends on the compiler (gcc's
        /// `-fno-trapping-math` reschedules code; gcc in ISO C mode treats
        /// `-ffp-contract=on` as `off`). The product rule decides: accepted
        /// exactly when the wrapper's canary reproduces every obligation row,
        /// and otherwise refused naming each broken obligation.
        RowsDecide,
    }

    /// Why the test compiler does not accept `extra` on top of `flags`, when
    /// it rejects the options themselves: an empty translation unit fails to
    /// compile and the diagnostic names one of them. A flag one compiler lacks
    /// (gcc has no `-fno-honor-nans`) cannot reach that compiler's builds, so
    /// its row has nothing to observe there. Any other failure is not a skip.
    #[cfg(unix)]
    fn rejected_flag(flags: &[String], extra: &str) -> Option<String> {
        let compiler = test_toolchain(CodegenRequirements::default()).compiler;
        let probe = tool_command(&compiler)
            .args(flags)
            .args(extra.split_whitespace())
            .args(["-fsyntax-only", "-x", "c", "-"])
            .stdin(Stdio::null())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&probe.stderr);
        (!probe.status.success()
            && extra
                .split_whitespace()
                .any(|flag| stderr.contains(flag.split('=').next().unwrap_or(flag))))
        .then(|| format!("{compiler} rejects `{extra}`: {}", stderr.trim()))
    }

    /// Check one wrapper. Contraction changes values only where the target
    /// has a fused multiply-add instruction, which the x86-64 baseline lacks;
    /// `-mfma` is what a `-march=native` wrapper would add there.
    #[cfg(unix)]
    fn check_wrapper(name: &str, extra: &str, verdict: Verdict) {
        let dir = tempfile::tempdir().unwrap();
        let reference = strict_reference_toolchain(String::new(), CodegenRequirements::default());
        let (flags, links) = (reference.compile_flags, reference.link_flags);
        let extra = if cfg!(target_arch = "x86_64") && extra.starts_with("-ffp-contract") {
            format!("-mfma {extra}")
        } else {
            extra.to_string()
        };
        if let Some(reason) = rejected_flag(&flags, &extra) {
            eprintln!("skipping {name}: {reason}");
            return;
        }
        let hostile = wrapper(dir.path(), name, &extra);
        let result = verify_compiler(&hostile, &flags, &links);
        match verdict {
            Verdict::Refused(expected) => assert!(
                matches!(&result, Err(CompilerCheckError::Refused(reason)) if reason.contains(expected)),
                "{name} ({extra}) should be refused with `{expected}`: {result:?}"
            ),
            Verdict::RowsDecide => {
                let mismatches = canary_mismatches(Path::new(&hostile), &flags, &links)
                    .unwrap_or_else(|error| panic!("{name} ({extra}): {error}"));
                if mismatches.is_empty() {
                    result.unwrap_or_else(|error| panic!("{name} ({extra}): {error}"));
                } else {
                    let broken = profile::describe(&mismatches);
                    assert!(
                        matches!(&result, Err(CompilerCheckError::Refused(reason))
                            if mismatches.iter().all(|m| reason.contains(m.row.obligation.name()))),
                        "{name} ({extra}) breaks {broken}, so it must be refused naming it: {result:?}"
                    );
                }
            }
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
    /// that observes it; a flag is accepted exactly when the wrapper's canary
    /// reproduces every obligation row.
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
        // clang contracts under `on`, which the contraction row refuses; gcc
        // in ISO C mode treats `on` as `off`, so its canary is unchanged.
        verify_compiler_wrapper_fp_contract_on: "-ffp-contract=on" => Verdict::RowsDecide;
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
        verify_compiler_wrapper_no_trapping_math: "-fno-trapping-math" => Verdict::RowsDecide;
    }

    // With SSE2 or AArch64 floating point, FLT_EVAL_METHOD is 0 and there is no
    // excess precision to keep or discard. On x87 the kernels' FLT_EVAL_METHOD
    // guard refuses it instead.
    #[cfg(all(unix, any(target_arch = "aarch64", target_arch = "x86_64")))]
    wrapper_verdicts! {
        verify_compiler_wrapper_excess_precision_fast: "-fexcess-precision=fast" => Verdict::Harmless;
    }

    /// The C library's integer rounding to even, which musl and glibc before 2.25
    /// lack. Generated units must not reach it (spec/design/correctly_rounded_math.md
    /// section 3.3), or no native build links on those C libraries.
    #[cfg(unix)]
    const ROUNDEVEN_FUNCTIONS: &[&str] = &["roundeven", "roundevenf", "roundevenl"];

    /// The first symbol in `assembly` that names one of [`ROUNDEVEN_FUNCTIONS`]:
    /// a whole token, with Mach-O's leading `_` and ELF's `@PLT` or `@GOTPCREL`
    /// allowed.
    #[cfg(unix)]
    fn roundeven_reference(assembly: &str) -> Option<&str> {
        assembly
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .find(|token| ROUNDEVEN_FUNCTIONS.contains(&token.trim_start_matches('_')))
    }

    /// The canary carries every kernel, so its code is every kernel's code: none
    /// may call the C library's `roundeven` under the profile, on any target.
    #[cfg(unix)]
    #[test]
    fn canary_calls_no_c_library_roundeven() {
        let dir = tempfile::tempdir().unwrap();
        let toolchain = test_toolchain(CodegenRequirements::default());
        let assembly = canary_assembly(&toolchain.compiler, &toolchain.compile_flags, dir.path());
        assert_eq!(roundeven_reference(&assembly), None);
    }

    /// Negative control: the scan finds a planted `roundeven` call.
    #[cfg(unix)]
    #[test]
    fn roundeven_scan_reports_a_planted_call() {
        let dir = tempfile::tempdir().unwrap();
        let toolchain = test_toolchain(CodegenRequirements::default());
        let (source, output) = (dir.path().join("planted.c"), dir.path().join("planted.s"));
        std::fs::write(
            &source,
            "double roundeven(double);\ndouble planted(double x) { return roundeven(x); }\n",
        )
        .unwrap();
        // `-fno-builtin`: arm64 would otherwise inline the call as `frintn`.
        let status = tool_command(&toolchain.compiler)
            .args(&toolchain.compile_flags)
            .args(["-fno-builtin", "-S", "-o"])
            .arg(&output)
            .arg(&source)
            .status()
            .unwrap();
        assert!(status.success(), "the planted call did not compile");
        let assembly = std::fs::read_to_string(&output).unwrap();
        assert_eq!(
            roundeven_reference(&assembly).map(|token| token.trim_start_matches('_')),
            Some("roundeven")
        );
    }

    /// A compiler is refused exactly when its headers are not the carried runtime
    /// archive's C library's (spec/08-backends.md section 7), naming the archive's.
    #[test]
    fn c_library_refusal_names_the_archive_library_on_a_mismatch() {
        let path = Path::new("/usr/bin/cc");
        assert_eq!(c_library_refusal(path, CLibrary::Glibc, true), None);
        assert_eq!(c_library_refusal(path, CLibrary::Musl, false), None);
        let refused = c_library_refusal(path, CLibrary::Glibc, false)
            .expect("a glibc archive with a compiler for another C library is refused");
        for part in [
            "`/usr/bin/cc`",
            "a C library other than glibc",
            "does not define `__GLIBC__`",
            "built for glibc",
            "CHELIS_CC",
        ] {
            assert!(refused.contains(part), "{part} missing from: {refused}");
        }
        let refused = c_library_refusal(path, CLibrary::Musl, true)
            .expect("a musl archive with a glibc compiler is refused");
        assert!(
            refused.contains("compiles against glibc") && refused.contains("built for musl"),
            "{refused}"
        );
    }

    /// A directory whose `<stdint.h>` is glibc's (it defines `__GLIBC__`) or not.
    #[cfg(unix)]
    fn fake_c_library_headers(dir: &Path, glibc: bool) -> PathBuf {
        let include = dir.join(if glibc {
            "glibc-include"
        } else {
            "other-include"
        });
        std::fs::create_dir_all(&include).unwrap();
        let define = if glibc { "#define __GLIBC__ 2\n" } else { "" };
        std::fs::write(
            include.join("stdint.h"),
            format!("{define}typedef unsigned long uint64_t;\n"),
        )
        .unwrap();
        include
    }

    /// The probe reads the headers the compiler includes, whatever the build
    /// host's own C library is.
    #[cfg(unix)]
    #[test]
    fn compiles_against_glibc_reads_the_headers_the_compiler_includes() {
        let dir = tempfile::tempdir().unwrap();
        let compiler = test_toolchain(CodegenRequirements::default()).compiler;
        let path = resolve_executable(&compiler).expect("a C compiler on PATH");
        for glibc in [true, false] {
            let include = fake_c_library_headers(dir.path(), glibc);
            let flags = [
                "-nostdinc".to_string(),
                "-isystem".to_string(),
                include.display().to_string(),
            ];
            assert_eq!(compiles_against_glibc(&path, &flags), Ok(glibc));
        }
        if let Some(carried) = CARRIED_C_LIBRARY {
            // The build host's own compiler matches the runtime built on it.
            assert_eq!(
                compiles_against_glibc(&path, &[]),
                Ok(carried == CLibrary::Glibc)
            );
        }
    }

    /// Where the carried runtime names a C library, a compiler whose headers are
    /// another library's is refused before anything compiles; elsewhere the check
    /// does not apply.
    #[cfg(unix)]
    #[test]
    fn verify_compiler_refuses_a_compiler_for_another_c_library() {
        let dir = tempfile::tempdir().unwrap();
        let reference = strict_reference_toolchain(String::new(), CodegenRequirements::default());
        let (flags, links) = (reference.compile_flags, reference.link_flags);
        let include = fake_c_library_headers(dir.path(), CARRIED_C_LIBRARY == Some(CLibrary::Musl));
        let foreign = wrapper(
            dir.path(),
            "foreign-cc",
            &format!("-nostdinc -isystem '{}'", include.display()),
        );
        let result = verify_compiler(&foreign, &flags, &links);
        match CARRIED_C_LIBRARY {
            Some(carried) => assert!(
                matches!(&result, Err(CompilerCheckError::Refused(reason))
                    if reason.contains(&format!("built for {}", carried.name()))),
                "{result:?}"
            ),
            None => assert!(
                !matches!(&result, Err(CompilerCheckError::Refused(reason)) if reason.contains("runtime archive built for")),
                "{result:?}"
            ),
        }
    }

    /// A profile that compiles with OpenMP also links with it, so an
    /// executable link and a static library's printed link requirements both
    /// resolve the parallel regions' `omp_*` and `GOMP_*` symbols on gcc.
    #[test]
    fn an_openmp_profile_links_with_openmp() {
        for needs_blas in [false, true] {
            let requirements = CodegenRequirements {
                wants_openmp: true,
                needs_blas,
            };
            let profile = pinned_toolchain("gcc".into(), requirements, true);
            assert!(profile.compile_flags.contains(&"-fopenmp".to_string()));
            assert!(profile.link_flags.contains(&"-fopenmp".to_string()));
        }
    }

    /// The canary is linked as a real build is: the profile's compile flags,
    /// then its link flags (`-lm` among them, which glibc needs for `fma` and
    /// `sqrt`), for every requirement combination.
    #[test]
    fn canary_links_with_the_profile_link_flags() {
        let (source, program) = (Path::new("canary.c"), Path::new("canary"));
        for wants_openmp in [false, true] {
            for needs_blas in [false, true] {
                let requirements = CodegenRequirements {
                    wants_openmp,
                    needs_blas,
                };
                for profile in [
                    pinned_toolchain("cc".into(), requirements, wants_openmp),
                    strict_reference_toolchain("cc".into(), requirements),
                ] {
                    let mut expected: Vec<std::ffi::OsString> =
                        profile.compile_flags.iter().map(Into::into).collect();
                    expected.push(source.into());
                    expected.extend(profile.link_flags.iter().map(Into::into));
                    expected.extend(["-o".into(), program.into()]);
                    assert!(profile.link_flags.contains(&"-lm".to_string()));
                    assert_eq!(
                        canary_args(&profile.compile_flags, &profile.link_flags, source, program),
                        expected,
                        "{profile:?}"
                    );
                }
            }
        }
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
