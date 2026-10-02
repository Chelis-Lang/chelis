//! Native artifact publication shared by all CLI backends.
use std::{
    error::Error,
    ffi::{OsStr, OsString},
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(crate) struct NativeBuild {
    pub target: &'static str,
    pub compiler: OsString,
    pub sources: Vec<PathBuf>,
    pub compile_flags: Vec<String>,
    pub link_flags: Vec<String>,
    pub runtime_archive: PathBuf,
    pub requires_main: bool,
}

pub(crate) fn compiler_override(variable: &str, default: &str) -> OsString {
    std::env::var_os(variable).unwrap_or_else(|| default.into())
}

impl NativeBuild {
    pub fn finish(mut self, emit_c: bool) -> Result<()> {
        for flag in ["-O2", "-ffp-contract=off"] {
            if !self.compile_flags.iter().any(|existing| existing == flag) {
                self.compile_flags.push(flag.into());
            }
        }
        let source = self.sources.first().ok_or("native build has no sources")?;
        let artifact = if self.requires_main {
            source.with_extension("")
        } else {
            let mut name = OsString::from("lib");
            name.push(source.file_stem().ok_or("native source has no stem")?);
            name.push(".a");
            source.with_file_name(name)
        };
        // Absolute paths prevent relative inputs beginning with '-' from
        // being interpreted as tool options, and keep support paths unambiguous.
        self.sources = self
            .sources
            .iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<_>>()?;
        self.runtime_archive = std::path::absolute(&self.runtime_archive)?;
        if artifact.canonicalize().ok().as_ref() == Some(&fs::canonicalize(&self.runtime_archive)?)
        {
            return Err("native artifact would overwrite the carried runtime archive; choose another output filename".into());
        }
        let archiver = compiler_override("CHELIS_AR", "ar");
        if emit_c {
            if self.requires_main {
                let mut command = self.executable_command(&artifact);
                print_command("Compile", &mut command);
            } else {
                let mut archive = Command::new(&archiver);
                archive.arg("rcs").arg(&artifact);
                for source in &self.sources {
                    let object = source.with_extension("o");
                    let mut command = self.object_command(source, &object);
                    print_command("Compile object", &mut command);
                    archive.arg(object);
                }
                print_command("Archive", &mut archive);
                self.print_link_requirements();
            }
            return Ok(());
        }

        // Same filesystem as the destination: rename publishes only a complete
        // artifact. Never give a native tool the previous artifact's pathname.
        let parent = artifact
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let scratch = tempfile::Builder::new()
            .prefix(".chelis-native-")
            .tempdir_in(parent)?;
        let product = scratch.path().join(if self.requires_main {
            "program"
        } else {
            "library.a"
        });
        if self.requires_main {
            run(
                &mut self.executable_command(&product),
                "compile/link",
                self.target,
                &product,
                true,
            )?;
        } else {
            let mut archive = Command::new(archiver);
            archive.arg("rcs").arg(&product);
            for (index, source) in self.sources.iter().enumerate() {
                let object = scratch.path().join(format!("source-{index}.o"));
                run(
                    &mut self.object_command(source, &object),
                    "compile",
                    self.target,
                    &object,
                    false,
                )?;
                archive.arg(object);
            }
            run(&mut archive, "archive", self.target, &product, false)?;
        }
        fs::rename(&product, &artifact)?;
        println!(
            "Built {} {}",
            if self.requires_main {
                "executable"
            } else {
                "static library"
            },
            artifact.display()
        );
        if !self.requires_main {
            self.print_link_requirements();
        }
        Ok(())
    }

    fn executable_command(&self, product: &Path) -> Command {
        let mut command = Command::new(&self.compiler);
        command
            .args(&self.compile_flags)
            .args(&self.sources)
            .arg(&self.runtime_archive)
            .args(&self.link_flags)
            .arg("-o")
            .arg(product);
        command
    }

    fn object_command(&self, source: &Path, object: &Path) -> Command {
        let mut command = Command::new(&self.compiler);
        command
            .args(&self.compile_flags)
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(object);
        command
    }

    fn print_link_requirements(&self) {
        let mut command = Command::new(&self.compiler);
        command.arg(&self.runtime_archive).args(&self.link_flags);
        print_command("Link requirements (after module archive)", &mut command);
    }
}

fn require_product(path: &Path, executable: bool) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("native tool did not produce {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "native tool produced no regular, nonempty artifact at {}",
            path.display()
        )
        .into());
    }
    #[cfg(unix)]
    if executable {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(format!(
                "native tool produced a non-executable artifact at {}",
                path.display()
            )
            .into());
        }
    }
    #[cfg(not(unix))]
    let _ = executable;
    Ok(())
}

fn run(
    command: &mut Command,
    stage: &str,
    target: &str,
    product: &Path,
    executable: bool,
) -> Result<()> {
    let tool = command.get_program().to_string_lossy().into_owned();
    let output = command.output().map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            format!(
                "native {stage}: tool `{tool}` was not found; {}",
                install_guidance(stage, target)
            )
        } else {
            format!("native {stage}: cannot run `{tool}`: {error}")
        }
    })?;
    // Compiler output belongs to diagnostics, not the CLI's success channel.
    io::stderr().write_all(&output.stdout)?;
    io::stderr().write_all(&output.stderr)?;
    if !output.status.success() {
        return Err(format!("native {stage}: `{tool}` failed with {}", output.status).into());
    }
    require_product(product, executable)
        .map_err(|error| format!("native {stage}: `{tool}`: {error}").into())
}

fn install_guidance(stage: &str, target: &str) -> &'static str {
    if stage == "archive" {
        "install the native toolchain's ar (binutils on Linux or xcode-select --install on macOS), or set CHELIS_AR"
    } else if target == "hip" {
        "install ROCm including hipcc, or set CHELIS_HIPCC to its executable path"
    } else if cfg!(target_os = "macos") {
        "install Apple's Command Line Tools with xcode-select --install; check the selected compiler override"
    } else {
        "install gcc/clang (build-essential on Debian/Ubuntu, gcc on Fedora); check the selected compiler override"
    }
}

fn print_command(label: &str, command: &mut Command) {
    let argv = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(shell_word)
        .collect::<Vec<_>>()
        .join(" ");
    println!("{label}: {argv}");
}

fn shell_word(word: &OsStr) -> String {
    let word = word.to_string_lossy();
    if !word.is_empty()
        && word
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_./=-+:".contains(&byte))
    {
        word.into_owned()
    } else {
        format!("'{}'", word.replace('\'', "'\\''"))
    }
}

/// Protect the input before source emission or native publication. The existing
/// CLI output option describes a source path/directory, so check both possible
/// native products before the checked manifest selects one.
pub(crate) fn protect_input(file: &Path, output: Option<&Path>, target: &str) -> Result<()> {
    let out = output.unwrap_or(Path::new("."));
    let extensions: &[&str] = match target {
        "c" => &["c"],
        "hip" => &["c", "cc", "cpp", "cxx"],
        "metal" => &["mm", "cc", "cpp", "cxx"],
        _ => return Err("invalid native target".into()),
    };
    let source = if out
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| extensions.contains(&ext))
    {
        out.to_path_buf()
    } else {
        let mut name = file
            .file_stem()
            .unwrap_or(OsStr::new("chelis_main"))
            .to_os_string();
        name.push(match target {
            "c" => ".c",
            "hip" => "_hip.cpp",
            _ => "_metal.mm",
        });
        out.join(name)
    };
    let mut library = OsString::from("lib");
    library.push(source.file_stem().ok_or("native source has no stem")?);
    library.push(".a");
    let input = fs::canonicalize(file)?;
    #[cfg(unix)]
    let input_metadata = fs::metadata(&input)?;
    for candidate in [
        source.clone(),
        source.with_extension("h"),
        source.with_extension(""),
        source.with_file_name(library),
    ] {
        // Canonical names resolve symlinks, but distinct hard-link names still
        // refer to the same file. Compare filesystem identity where available.
        #[cfg(unix)]
        let same_input = {
            use std::os::unix::fs::MetadataExt;
            candidate.metadata().is_ok_and(|metadata| {
                metadata.dev() == input_metadata.dev() && metadata.ino() == input_metadata.ino()
            })
        };
        #[cfg(not(unix))]
        let same_input = candidate.canonicalize().ok().as_ref() == Some(&input);
        if same_input {
            return Err(format!(
                "build output would overwrite input {}; choose a separate --output directory",
                file.display()
            )
            .into());
        }
    }
    Ok(())
}
