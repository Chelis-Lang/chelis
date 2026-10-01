//! Exact evaluator-versus-C observation gate (chelis#763).
//!
//! A caller-supplied scope is provenance data, not proof of a sandbox. Only
//! the Nix check that authored and binds that scope to its closure is an
//! authoritative pinned verdict.

use chelis_backend_c::toolchain::{CodegenRequirements, strict_reference_toolchain};
use chelis_types::agreement::compare_exact_observations;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use walkdir::WalkDir;

const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChelisIdentity {
    store_path: String,
    sha256: String,
    version: String,
    rustc_version: String,
    build_host: String,
    build_target: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompilerIdentity {
    path: String,
    sha256: String,
    version: String,
    target: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LibraryIdentity {
    libc: String,
    libm: String,
    math_provider: String,
    linkage: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExecutionIdentity {
    os: String,
    cpu_isa: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProofInput {
    schema_version: u32,
    source_sha256: String,
    compiler_source_sha256: String,
    corpus_sha256: String,
    flake_lock_sha256: String,
    nixpkgs_revision: String,
    input_derivations: BTreeMap<String, String>,
    chelis: ChelisIdentity,
    compiler: CompilerIdentity,
    libraries: LibraryIdentity,
    nix_system: String,
    execution: ExecutionIdentity,
    environment: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct Invocation {
    file: String,
    compile_link_argv: Vec<String>,
    loaded_libraries: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ProofScope {
    schema_version: u32,
    kind: &'static str,
    source_sha256: Option<String>,
    compiler_source_sha256: Option<String>,
    corpus_sha256: String,
    flake_lock_sha256: Option<String>,
    nixpkgs_revision: Option<String>,
    input_derivations: BTreeMap<String, String>,
    chelis: Option<ChelisIdentity>,
    compiler: Option<CompilerIdentity>,
    libraries: Option<LibraryIdentity>,
    nix_system: Option<String>,
    execution: ExecutionIdentity,
    cpu_model: Option<String>,
    environment: BTreeMap<String, String>,
    profile: StrictProfile,
    invocations: Vec<Invocation>,
}

#[derive(Serialize)]
struct StrictProfile {
    optimization: &'static str,
    fp_contract: &'static str,
    fast_math: bool,
    architecture: &'static str,
    threads: u8,
}

#[derive(Serialize)]
struct ProcessFailure {
    status: Option<i32>,
    signal: Option<i32>,
    timed_out: bool,
    deadline_seconds: u64,
    stderr: String,
}

#[derive(Serialize)]
struct ProgramRecord {
    schema_version: u32,
    record: &'static str,
    file: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    process: Option<ProcessFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_stdout_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compiled_stdout_sha256: Option<String>,
}

impl ProgramRecord {
    fn error(
        file: &str,
        stage: &'static str,
        diagnostic: impl Into<String>,
        process: Option<ProcessFailure>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            record: "program",
            file: file.to_owned(),
            status: "error",
            stage: Some(stage),
            diagnostic: Some(diagnostic.into()),
            process,
            eval_stdout_sha256: None,
            compiled_stdout_sha256: None,
        }
    }
}

#[derive(Serialize)]
struct Summary<'a> {
    schema_version: u32,
    record: &'static str,
    status: &'static str,
    compared: usize,
    passed: usize,
    diverged: usize,
    errors: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostic: Option<String>,
    proof_scope: &'a ProofScope,
}

struct Captured {
    status: Option<i32>,
    signal: Option<i32>,
    timed_out: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Captured {
    fn success(&self) -> bool {
        self.status == Some(0) && !self.timed_out
    }

    fn failure(&self, deadline: Duration) -> ProcessFailure {
        ProcessFailure {
            status: self.status,
            signal: self.signal,
            timed_out: self.timed_out,
            deadline_seconds: deadline.as_secs(),
            stderr: String::from_utf8_lossy(&self.stderr).into_owned(),
        }
    }
}

struct Program {
    path: PathBuf,
    relative: String,
}

fn digest_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hasher.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn corpus_digest(programs: &[Program]) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65536];
    for program in programs {
        hasher.update(program.relative.as_bytes());
        hasher.update([0]);
        let mut source = File::open(&program.path)?;
        loop {
            let size = source.read(&mut buffer)?;
            if size == 0 {
                break;
            }
            hasher.update(&buffer[..size]);
        }
        hasher.update([0]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn discover(path: &Path) -> Result<Vec<Program>, String> {
    let mut programs = Vec::new();
    if path.is_file() && path.extension().is_some_and(|ext| ext == "ch") {
        programs.push(Program {
            path: path.to_path_buf(),
            relative: path.file_name().unwrap().to_string_lossy().into_owned(),
        });
    } else if path.is_dir() {
        for entry in WalkDir::new(path).follow_links(false) {
            let entry = entry.map_err(|error| format!("cannot discover corpus: {error}"))?;
            if entry.path().extension().is_some_and(|ext| ext == "ch") {
                if entry.file_type().is_symlink() {
                    return Err(format!(
                        "corpus source is a symlink: {}",
                        entry.path().display()
                    ));
                }
                if entry.file_type().is_file() {
                    let relative = entry
                        .path()
                        .strip_prefix(path)
                        .map_err(|error| error.to_string())?;
                    let relative = relative
                        .to_str()
                        .ok_or("corpus path is not UTF-8")?
                        .replace('\\', "/");
                    programs.push(Program {
                        path: entry.path().to_path_buf(),
                        relative,
                    });
                }
            }
        }
    } else {
        return Err(format!(
            "expected a .ch file or directory: {}",
            path.display()
        ));
    }
    programs.sort_by(|a, b| a.relative.cmp(&b.relative));
    if programs.is_empty() {
        return Err(format!(
            "empty corpus: no .ch programs under {}",
            path.display()
        ));
    }
    Ok(programs)
}

fn hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn load_input(path: &Path, corpus_hash: &str) -> Result<ProofInput, String> {
    let input: ProofInput =
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("invalid proof-scope input: {error}"))?;
    if input.schema_version != SCHEMA_VERSION
        || [
            &input.source_sha256,
            &input.compiler_source_sha256,
            &input.corpus_sha256,
            &input.flake_lock_sha256,
            &input.chelis.sha256,
            &input.compiler.sha256,
        ]
        .iter()
        .any(|digest| !hex_digest(digest))
    {
        return Err("unsupported proof-scope schema or invalid SHA-256 digest".into());
    }
    if input.corpus_sha256 != corpus_hash {
        return Err("proof-scope corpus SHA-256 disagrees with selected .ch files".into());
    }
    for key in ["chelis", "compiler", "source", "compiler_source", "corpus"] {
        if !input
            .input_derivations
            .get(key)
            .is_some_and(|path| path.starts_with("/nix/store/"))
        {
            return Err(format!(
                "proof-scope input missing Nix store identity for {key}"
            ));
        }
    }
    if input.nixpkgs_revision.is_empty()
        || input.chelis.rustc_version.is_empty()
        || input.chelis.build_host.is_empty()
        || input.chelis.build_target.is_empty()
        || input.libraries.linkage != "dynamic"
        || ["libc", "libm", "math_provider"].iter().any(|key| {
            let value = match *key {
                "libc" => &input.libraries.libc,
                "libm" => &input.libraries.libm,
                _ => &input.libraries.math_provider,
            };
            value.is_empty()
        })
    {
        return Err("proof-scope input is missing required closure identity".into());
    }
    if input.nix_system != "x86_64-linux"
        || input.execution.os != std::env::consts::OS
        || input.execution.cpu_isa != std::env::consts::ARCH
    {
        return Err("proof-scope execution OS/CPU contradicts the selected Nix system".into());
    }
    for (key, expected) in [("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")] {
        if input.environment.get(key).map(String::as_str) != Some(expected) {
            return Err(format!("proof-scope environment requires {key}={expected}"));
        }
    }
    for key in ["PATH", "NIX_CFLAGS_COMPILE", "NIX_LDFLAGS"] {
        if !input
            .environment
            .get(key)
            .is_some_and(|value| !value.is_empty())
        {
            return Err(format!("proof-scope environment is missing {key}"));
        }
    }
    if !input.environment["NIX_CFLAGS_COMPILE"]
        .split_whitespace()
        .any(|flag| flag == "-ffp-contract=off")
        || !input.environment["NIX_CFLAGS_COMPILE"]
            .split_whitespace()
            .any(|flag| flag == "-fno-fast-math")
        || input.environment["NIX_CFLAGS_COMPILE"]
            .split_whitespace()
            .any(|flag| {
                flag.starts_with("-march")
                    || flag == "-ffast-math"
                    || flag == "-Ofast"
                    || flag == "-fopenmp"
                    || flag == "-ffp-contract=fast"
            })
        || input.environment.keys().any(|key| {
            ![
                "PATH",
                "NIX_CFLAGS_COMPILE",
                "NIX_LDFLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
            .contains(&key.as_str())
        })
    {
        return Err("proof-scope environment contradicts the strict reference profile".into());
    }
    if !input.compiler.path.starts_with("/nix/store/")
        || !input.chelis.store_path.starts_with("/nix/store/")
        || !input.libraries.libc.starts_with("/nix/store/")
        || !input.libraries.libm.starts_with("/nix/store/")
        || !input.libraries.math_provider.starts_with("/nix/store/")
    {
        return Err("proof-scope toolchain and libraries must belong to the Nix store".into());
    }
    Ok(input)
}

fn child_environment(input: Option<&ProofInput>) -> BTreeMap<String, String> {
    if let Some(input) = input {
        return input.environment.clone();
    }
    let mut env = BTreeMap::new();
    env.insert("PATH".into(), std::env::var("PATH").unwrap_or_default());
    env.insert("OMP_NUM_THREADS".into(), "1".into());
    env.insert("OPENBLAS_NUM_THREADS".into(), "1".into());
    env
}

fn command(
    binary: &str,
    args: &[String],
    cwd: &Path,
    environment: &BTreeMap<String, String>,
) -> Command {
    let mut command = Command::new(binary);
    command
        .args(args)
        .current_dir(cwd)
        .env_clear()
        .envs(environment);
    command.env("LC_ALL", "C").env("LANG", "C").env("TZ", "UTC");
    command.env("HOME", cwd).env("TMPDIR", cwd);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // A child can spawn grandchildren (cc -> linker, eval -> process). Reap the
    // entire group on a deadline, not just the immediate process.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    command
}

fn capture(
    binary: &str,
    args: &[String],
    cwd: &Path,
    environment: &BTreeMap<String, String>,
    deadline: Duration,
) -> Captured {
    let mut child = match command(binary, args, cwd, environment).spawn() {
        Ok(child) => child,
        Err(error) => {
            return Captured {
                status: None,
                signal: None,
                timed_out: false,
                stdout: Vec::new(),
                stderr: error.to_string().into_bytes(),
            };
        }
    };
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let (sender, receiver) = mpsc::channel();
    let stdout_sender = sender.clone();
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.read_to_end(&mut bytes).map(|_| bytes);
        let _ = stdout_sender.send((false, result));
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stderr.read_to_end(&mut bytes).map(|_| bytes);
        let _ = sender.send((true, result));
    });
    let started = Instant::now();
    let mut read_error = None;
    let (status, mut timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (Some(status), false),
            Ok(None) if started.elapsed() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) => break (None, true),
            Err(error) => {
                read_error = Some(format!("cannot wait for subprocess: {error}"));
                break (None, false);
            }
        }
    };
    let (mut stdout_bytes, mut stderr_bytes) = (None, None);
    if !timed_out && read_error.is_none() {
        while stdout_bytes.is_none() || stderr_bytes.is_none() {
            match receiver.recv_timeout(deadline.saturating_sub(started.elapsed())) {
                Ok((is_stderr, Ok(bytes))) => {
                    if is_stderr {
                        stderr_bytes = Some(bytes);
                    } else {
                        stdout_bytes = Some(bytes);
                    }
                }
                Ok((_, Err(error))) => {
                    read_error = Some(format!("cannot capture subprocess output: {error}"));
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    timed_out = true;
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    read_error = Some("subprocess output reader disconnected".into());
                    break;
                }
            }
        }
    }
    if timed_out || read_error.is_some() {
        // A reaped parent can leave a grandchild holding the stdout pipe.
        // Kill the process group before joining output readers.
        #[cfg(unix)]
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.kill();
        let _ = child.wait();
    }
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    let mut stderr_bytes = stderr_bytes.unwrap_or_default();
    if let Some(error) = &read_error {
        stderr_bytes.extend_from_slice(error.as_bytes());
    }
    #[cfg(unix)]
    let signal = status.as_ref().and_then(|status| {
        use std::os::unix::process::ExitStatusExt;
        status.signal()
    });
    #[cfg(not(unix))]
    let signal = None;
    Captured {
        status: if read_error.is_some() {
            None
        } else {
            status.and_then(|status| status.code())
        },
        signal,
        timed_out,
        stdout: stdout_bytes.unwrap_or_default(),
        stderr: stderr_bytes,
    }
}

fn compiler_query(
    compiler: &str,
    arg: &str,
    cwd: &Path,
    environment: &BTreeMap<String, String>,
    deadline: Duration,
) -> Result<String, String> {
    let result = capture(compiler, &[arg.to_owned()], cwd, environment, deadline);
    if !result.success() {
        return Err(format!(
            "compiler {arg} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    let output = String::from_utf8(result.stdout)
        .map_err(|error| format!("compiler {arg} is not UTF-8: {error}"))?;
    let output = output.trim();
    if output.is_empty() {
        return Err(format!("compiler {arg} returned no target/version"));
    }
    Ok(output.into())
}

fn resolve_executable(name: &str, path: &str) -> Result<PathBuf, String> {
    if name.contains('/') {
        let candidate = Path::new(name);
        if !candidate.is_file() {
            return Err(format!("compiler {name} is not a file"));
        }
        return if candidate.is_absolute() {
            Ok(candidate.to_path_buf())
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(candidate))
                .map_err(|error| format!("compiler {name}: {error}"))
        };
    }
    for directory in std::env::split_paths(path) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            // Some compiler launchers dispatch on argv[0] (e.g. clang
            // symlinked to swiftly). Keep the invoked alias, not its target.
            return if candidate.is_absolute() {
                Ok(candidate)
            } else {
                std::env::current_dir()
                    .map(|cwd| cwd.join(candidate))
                    .map_err(|error| error.to_string())
            };
        }
    }
    Err(format!(
        "compiler {name} was not found on the declared PATH"
    ))
}

fn cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let file = File::open("/proc/cpuinfo").ok()?;
        BufReader::new(file).lines().find_map(|line| {
            line.ok()?
                .strip_prefix("model name\t:")
                .map(|name| name.trim().to_owned())
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn initial_scope(
    hash: &str,
    env: BTreeMap<String, String>,
    input: Option<&ProofInput>,
) -> ProofScope {
    ProofScope {
        schema_version: SCHEMA_VERSION,
        kind: if input.is_some() {
            "declared-nix-closure"
        } else {
            "unpinned-host"
        },
        source_sha256: input.map(|input| input.source_sha256.clone()),
        compiler_source_sha256: input.map(|input| input.compiler_source_sha256.clone()),
        corpus_sha256: hash.into(),
        flake_lock_sha256: input.map(|input| input.flake_lock_sha256.clone()),
        nixpkgs_revision: input.map(|input| input.nixpkgs_revision.clone()),
        input_derivations: input
            .map(|input| input.input_derivations.clone())
            .unwrap_or_default(),
        chelis: input.map(|input| input.chelis.clone()),
        compiler: input.map(|input| input.compiler.clone()),
        libraries: input.map(|input| input.libraries.clone()),
        nix_system: input.map(|input| input.nix_system.clone()),
        execution: input
            .map(|input| input.execution.clone())
            .unwrap_or(ExecutionIdentity {
                os: std::env::consts::OS.into(),
                cpu_isa: std::env::consts::ARCH.into(),
            }),
        cpu_model: cpu_model(),
        environment: env,
        profile: StrictProfile {
            optimization: "-O2",
            fp_contract: "off",
            fast_math: false,
            architecture: "portable (no -march=native)",
            threads: 1,
        },
        invocations: Vec::new(),
    }
}

fn check_identity(
    scope: &mut ProofScope,
    input: Option<&ProofInput>,
    deadline: Duration,
    cwd: &Path,
) -> Result<String, String> {
    let selected_compiler = input
        .map(|input| input.compiler.path.clone())
        .unwrap_or_else(|| {
            chelis_backend_c::toolchain::runtime_toolchain(CodegenRequirements::default()).compiler
        });
    let compiler = resolve_executable(&selected_compiler, &scope.environment["PATH"])?;
    let compiler_str = compiler
        .to_str()
        .ok_or("compiler path is not UTF-8")?
        .to_owned();
    let version = compiler_query(
        &compiler_str,
        "--version",
        cwd,
        &scope.environment,
        deadline,
    )?;
    let target = compiler_query(
        &compiler_str,
        "-dumpmachine",
        cwd,
        &scope.environment,
        deadline,
    )?;
    if target.lines().count() != 1
        || target.chars().any(char::is_whitespace)
        || target.bytes().filter(|byte| *byte == b'-').count() < 2
    {
        return Err(format!(
            "queried compiler target is not a triple: {target:?}"
        ));
    }
    let sha256 =
        digest_file(&compiler).map_err(|error| format!("cannot hash compiler: {error}"))?;
    if let Some(input) = input {
        let chelis_path =
            fs::canonicalize(&input.chelis.store_path).map_err(|error| error.to_string())?;
        let current = std::env::current_exe()
            .and_then(fs::canonicalize)
            .map_err(|error| error.to_string())?;
        if chelis_path != current
            || digest_file(&current).map_err(|error| error.to_string())? != input.chelis.sha256
            || input.chelis.version != format!("chelis {}", env!("CARGO_PKG_VERSION"))
        {
            return Err(
                "Chelis executable digest, path, or version contradicts proof-scope input".into(),
            );
        }
        if compiler_str != input.compiler.path
            && compiler
                != fs::canonicalize(&input.compiler.path).map_err(|error| error.to_string())?
        {
            return Err("selected compiler contradicts proof-scope input".into());
        }
        if version != input.compiler.version
            || target != input.compiler.target
            || sha256 != input.compiler.sha256
        {
            return Err(format!(
                "queried compiler version/target/digest contradicts proof-scope input: queried target {target:?}, declared {:?}",
                input.compiler.target
            ));
        }
        if !target.starts_with("x86_64")
            || !target.contains("linux")
            || !input.chelis.build_host.starts_with("x86_64")
            || !input.chelis.build_host.contains("linux")
            || !input.chelis.build_target.starts_with("x86_64")
            || !input.chelis.build_target.contains("linux")
        {
            return Err(format!(
                "compiler target {target:?} or Rust host/target contradicts x86_64-linux"
            ));
        }
        for library in [
            &input.libraries.libc,
            &input.libraries.libm,
            &input.libraries.math_provider,
        ] {
            if !Path::new(library).exists() {
                return Err(format!(
                    "declared native library closure does not exist: {library}"
                ));
            }
        }
    } else {
        scope.compiler = Some(CompilerIdentity {
            path: compiler_str.clone(),
            sha256,
            version,
            target,
        });
        let chelis = std::env::current_exe().map_err(|error| error.to_string())?;
        scope.chelis = Some(ChelisIdentity {
            store_path: chelis.display().to_string(),
            sha256: digest_file(&chelis).map_err(|error| error.to_string())?,
            version: format!("chelis {}", env!("CARGO_PKG_VERSION")),
            rustc_version: String::new(),
            build_host: String::new(),
            build_target: String::new(),
        });
    }
    Ok(compiler_str)
}

fn failed_stage(
    file: &str,
    stage: &'static str,
    result: &Captured,
    deadline: Duration,
) -> ProgramRecord {
    let diagnosis = if result.timed_out {
        format!("{stage} timed out after {}s", deadline.as_secs())
    } else {
        format!(
            "{stage} failed with status {:?}, signal {:?}",
            result.status, result.signal
        )
    };
    ProgramRecord::error(file, stage, diagnosis, Some(result.failure(deadline)))
}

fn first_difference(reference: &str, candidate: &str) -> String {
    let index = reference
        .as_bytes()
        .iter()
        .zip(candidate.as_bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| reference.len().min(candidate.len()));
    let line = 1 + reference.as_bytes()[..index]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count();
    let column = 1 + reference.as_bytes()[..index]
        .iter()
        .rev()
        .take_while(|byte| **byte != b'\n')
        .count();
    let eval_line = reference.lines().nth(line - 1).unwrap_or("<missing>");
    let compiled_line = candidate.lines().nth(line - 1).unwrap_or("<missing>");
    format!(
        "stdout differs at byte {index} (line {line}, column {column}): eval {eval_line:?}, C {compiled_line:?}; byte lengths eval={} C={}",
        reference.len(),
        candidate.len()
    )
}

fn generated_source_needs_blas(path: &Path) -> io::Result<bool> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(false);
        }
        if line.contains("cblas_sgemm(") || line.contains("\"chelis_blas.h\"") {
            return Ok(true);
        }
    }
}

fn run_case(
    program: &Program,
    executable: &str,
    compiler: &Result<String, String>,
    scope: &mut ProofScope,
    deadline: Duration,
) -> ProgramRecord {
    let scratch = match tempdir() {
        Ok(dir) => dir,
        Err(error) => {
            return ProgramRecord::error(&program.relative, "discovery", error.to_string(), None);
        }
    };
    let local = scratch.path().join("case.ch");
    if let Err(error) = fs::copy(&program.path, &local) {
        return ProgramRecord::error(
            &program.relative,
            "discovery",
            format!("cannot stage program: {error}"),
            None,
        );
    }
    let eval = capture(
        executable,
        &[
            "eval".into(),
            "--file".into(),
            "case.ch".into(),
            "--target".into(),
            "c".into(),
        ],
        scratch.path(),
        &scope.environment,
        deadline,
    );
    if !eval.success() {
        return failed_stage(&program.relative, "eval", &eval, deadline);
    }
    let build = capture(
        executable,
        &[
            "build".into(),
            "--target".into(),
            "c".into(),
            "--output".into(),
            "out".into(),
            "case.ch".into(),
        ],
        scratch.path(),
        &scope.environment,
        deadline,
    );
    if !build.success() {
        return failed_stage(&program.relative, "build", &build, deadline);
    }
    if eval.stdout.is_empty() {
        return ProgramRecord::error(
            &program.relative,
            "discovery",
            "no observable evaluator stdout; library-only and zero-output programs are not comparable",
            None,
        );
    }
    let output = scratch.path().join("out");
    let sources = match fs::read_dir(&output) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "c"))
            .collect::<Vec<_>>(),
        Err(error) => {
            return ProgramRecord::error(
                &program.relative,
                "build",
                format!("C output directory missing: {error}"),
                None,
            );
        }
    };
    if sources.len() != 1 {
        return ProgramRecord::error(
            &program.relative,
            "build",
            format!(
                "expected one emitted C translation unit, got {}",
                sources.len()
            ),
            None,
        );
    }
    let source = &sources[0].path();
    let c_relative = Path::new("out").join(sources[0].file_name());
    let c_relative = match c_relative.to_str() {
        Some(path) => path.to_owned(),
        None => {
            return ProgramRecord::error(
                &program.relative,
                "build",
                "C source path is not UTF-8",
                None,
            );
        }
    };
    let needs_blas = match generated_source_needs_blas(source) {
        Ok(needs_blas) => needs_blas,
        Err(error) => {
            return ProgramRecord::error(
                &program.relative,
                "build",
                format!("cannot inspect emitted C: {error}"),
                None,
            );
        }
    };
    let compiler = match compiler {
        Ok(compiler) => compiler,
        Err(error) => {
            return ProgramRecord::error(
                &program.relative,
                "link",
                error,
                Some(ProcessFailure {
                    status: None,
                    signal: None,
                    timed_out: false,
                    deadline_seconds: deadline.as_secs(),
                    stderr: error.clone(),
                }),
            );
        }
    };
    let toolchain = strict_reference_toolchain(
        compiler.clone(),
        CodegenRequirements {
            wants_openmp: false,
            needs_blas,
        },
    );
    let mut argv = toolchain.compile_flags;
    argv.push("-Iout".into());
    argv.push(c_relative);
    argv.push("out/libchelis_runtime.a".into());
    argv.extend(toolchain.link_flags);
    argv.extend(["-o".into(), "out/case".into()]);
    scope.invocations.push(Invocation {
        file: program.relative.clone(),
        compile_link_argv: std::iter::once(compiler.clone())
            .chain(argv.iter().cloned())
            .collect(),
        loaded_libraries: BTreeMap::new(),
    });
    let link = capture(
        compiler,
        &argv,
        scratch.path(),
        &scope.environment,
        deadline,
    );
    if !link.success() {
        return failed_stage(&program.relative, "link", &link, deadline);
    }
    if scope.kind == "declared-nix-closure" {
        let linked = capture(
            "ldd",
            &["./out/case".into()],
            scratch.path(),
            &scope.environment,
            deadline,
        );
        if !linked.success() {
            return failed_stage(&program.relative, "link", &linked, deadline);
        }
        let text = match std::str::from_utf8(&linked.stdout) {
            Ok(text) => text,
            Err(error) => {
                return ProgramRecord::error(
                    &program.relative,
                    "link",
                    format!("ldd output is not UTF-8: {error}"),
                    None,
                );
            }
        };
        let mut resolved = BTreeMap::new();
        for line in text.lines() {
            let Some((name, location)) = line.trim().split_once(" => ") else {
                continue;
            };
            let Some(path) = location.split_whitespace().next() else {
                continue;
            };
            if !path.starts_with('/') {
                return ProgramRecord::error(
                    &program.relative,
                    "link",
                    format!("unresolved linked library {name}: {line}"),
                    None,
                );
            }
            let actual = match fs::canonicalize(path) {
                Ok(path) => path,
                Err(error) => {
                    return ProgramRecord::error(
                        &program.relative,
                        "link",
                        format!("cannot resolve linked {name}: {error}"),
                        None,
                    );
                }
            };
            if !actual.starts_with("/nix/store/") {
                return ProgramRecord::error(
                    &program.relative,
                    "link",
                    format!(
                        "linked {name} outside pinned Nix store: {}",
                        actual.display()
                    ),
                    None,
                );
            }
            let declared = scope.libraries.as_ref().and_then(|libraries| match name {
                "libc.so.6" => Some(libraries.libc.as_str()),
                "libm.so.6" => Some(libraries.libm.as_str()),
                name if name.contains("openblas") => Some(libraries.math_provider.as_str()),
                _ => None,
            });
            if let Some(declared) = declared {
                match fs::canonicalize(declared) {
                    Ok(expected) if expected == actual => {}
                    _ => {
                        return ProgramRecord::error(
                            &program.relative,
                            "link",
                            format!(
                                "linked {name} at {} contradicts declared {declared}",
                                actual.display()
                            ),
                            None,
                        );
                    }
                }
            }
            resolved.insert(name.to_owned(), actual.display().to_string());
        }
        if !resolved.contains_key("libc.so.6") {
            return ProgramRecord::error(
                &program.relative,
                "link",
                "dynamic C binary loaded no traceable libc.so.6",
                None,
            );
        }
        scope
            .invocations
            .last_mut()
            .expect("recorded compile/link")
            .loaded_libraries = resolved;
    }
    let compiled = capture(
        "./out/case",
        &[],
        scratch.path(),
        &scope.environment,
        deadline,
    );
    if !compiled.success() {
        return failed_stage(&program.relative, "run", &compiled, deadline);
    }
    let eval_text = match std::str::from_utf8(&eval.stdout) {
        Ok(text) => text,
        Err(error) => {
            return ProgramRecord::error(
                &program.relative,
                "compare",
                format!("eval stdout violates UTF-8 observation contract: {error}"),
                None,
            );
        }
    };
    let compiled_text = match std::str::from_utf8(&compiled.stdout) {
        Ok(text) => text,
        Err(error) => {
            return ProgramRecord::error(
                &program.relative,
                "compare",
                format!("compiled stdout violates UTF-8 observation contract: {error}"),
                None,
            );
        }
    };
    let status = if compare_exact_observations(&program.relative, eval_text, compiled_text).is_ok()
    {
        "pass"
    } else {
        "divergence"
    };
    ProgramRecord {
        schema_version: SCHEMA_VERSION,
        record: "program",
        file: program.relative.clone(),
        status,
        stage: (status == "divergence").then_some("compare"),
        diagnostic: (status == "divergence").then(|| first_difference(eval_text, compiled_text)),
        process: None,
        eval_stdout_sha256: Some(format!("{:x}", Sha256::digest(&eval.stdout))),
        compiled_stdout_sha256: Some(format!("{:x}", Sha256::digest(&compiled.stdout))),
    }
}

fn emit_json(record: &impl Serialize) {
    println!(
        "{}",
        serde_json::to_string(record).expect("typed report serializes")
    );
}

fn emit_program(record: &ProgramRecord, json: bool) {
    if json {
        emit_json(record);
    } else {
        println!(
            "{}: {} {}",
            record.file,
            record.status,
            record.diagnostic.as_deref().unwrap_or("")
        );
        if let Some(process) = &record.process
            && !process.stderr.is_empty()
        {
            println!("  stderr: {}", process.stderr.trim_end());
        }
    }
}

fn emit_summary(summary: &Summary<'_>, json: bool) {
    if json {
        emit_json(summary);
    } else {
        println!(
            "summary: {} ({} compared, {} passed, {} diverged, {} errors) {}",
            summary.status,
            summary.compared,
            summary.passed,
            summary.diverged,
            summary.errors,
            summary.diagnostic.as_deref().unwrap_or("")
        );
    }
}

fn setup_error(scope: &ProofScope, json: bool, stage: &'static str, message: String) -> i32 {
    emit_summary(
        &Summary {
            schema_version: SCHEMA_VERSION,
            record: "summary",
            status: "error",
            compared: 0,
            passed: 0,
            diverged: 0,
            errors: 1,
            stage: Some(stage),
            diagnostic: Some(message),
            proof_scope: scope,
        },
        json,
    );
    2
}

pub(super) fn run(
    path: &Path,
    json: bool,
    proof_scope_input: Option<&Path>,
    deadline: Duration,
) -> i32 {
    let environment = child_environment(None);
    let mut scope = initial_scope("", environment, None);
    let programs = match discover(path) {
        Ok(programs) => programs,
        Err(error) => return setup_error(&scope, json, "discovery", error),
    };
    let corpus_hash = match corpus_digest(&programs) {
        Ok(hash) => hash,
        Err(error) => return setup_error(&scope, json, "discovery", error.to_string()),
    };
    let input = match proof_scope_input
        .map(|path| load_input(path, &corpus_hash))
        .transpose()
    {
        Ok(input) => input,
        Err(error) => return setup_error(&scope, json, "discovery", error),
    };
    scope = initial_scope(
        &corpus_hash,
        child_environment(input.as_ref()),
        input.as_ref(),
    );
    let current_exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => return setup_error(&scope, json, "discovery", error.to_string()),
    };
    let executable = match current_exe.to_str() {
        Some(executable) => executable,
        None => {
            return setup_error(
                &scope,
                json,
                "discovery",
                "Chelis executable path is not UTF-8".into(),
            );
        }
    };
    // Verify the actual selected compiler and binary, not the recipe printed
    // by `build`. A contradiction is carried into every program as a link
    // error and never produces an agreement claim.
    let compiler = check_identity(
        &mut scope,
        input.as_ref(),
        deadline,
        path.parent().unwrap_or(Path::new(".")),
    );
    let mut passed = 0;
    let mut diverged = 0;
    let mut errors = 0;
    for program in &programs {
        let record = run_case(program, executable, &compiler, &mut scope, deadline);
        match record.status {
            "pass" => passed += 1,
            "divergence" => diverged += 1,
            _ => errors += 1,
        }
        emit_program(&record, json);
    }
    let compared = passed + diverged;
    let status = if errors > 0 || compared == 0 {
        "error"
    } else if diverged > 0 {
        "divergence"
    } else {
        "pass"
    };
    emit_summary(
        &Summary {
            schema_version: SCHEMA_VERSION,
            record: "summary",
            status,
            compared,
            passed,
            diverged,
            errors,
            stage: (compared == 0).then_some("discovery"),
            diagnostic: (compared == 0).then(|| "zero comparable programs".into()),
            proof_scope: &scope,
        },
        json,
    );
    match status {
        "pass" => 0,
        "divergence" => 1,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::capture;
    use super::first_difference;
    use chelis_types::agreement::compare_exact_observations;
    #[cfg(unix)]
    use std::collections::BTreeMap;
    #[cfg(unix)]
    use std::time::Duration;

    #[cfg(unix)]
    #[test]
    fn descendant_holding_stdout_open_cannot_outlive_deadline() {
        let dir = tempfile::tempdir().unwrap();
        let environment = BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]);
        let result = capture(
            "/bin/sh",
            &["-c".into(), "/bin/sleep 3 &".into()],
            dir.path(),
            &environment,
            Duration::from_secs(1),
        );
        assert!(
            result.timed_out,
            "a reaped shell cannot leave the pipe indefinitely open"
        );
        assert_eq!(result.status, Some(0), "the parent exited before its child");
    }

    #[test]
    fn complete_output_detects_final_newline_and_omitted_root() {
        for (eval, compiled, needle) in [
            ("root = 1\n", "root = 1", "byte lengths"),
            ("a = 1\nb = 2\n", "a = 1\n", "line 2"),
            (
                "shape=[2] data=[true, false]\n",
                "shape=[2] data=[true]\n",
                "line 1",
            ),
            ("n = 9007199254740993\n", "n = 9007199254740992\n", "line 1"),
        ] {
            assert!(compare_exact_observations("program", eval, compiled).is_err());
            assert!(first_difference(eval, compiled).contains(needle));
        }
    }
}
