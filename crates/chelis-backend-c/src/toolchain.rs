//! Cross-platform toolchain resolution for Chelis CPU codegen consumers.

use std::process::Command;

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

pub fn runtime_toolchain(requirements: CodegenRequirements) -> NativeToolchain {
    resolve_toolchain(requirements, &["CHELIS_CC"])
}
/// Reference profile for a caller-selected compiler. Unlike the product runtime
/// profile, this does not resolve a compiler or incorporate ambient build flags.
/// Callers spawning the compiler must also sanitize its inherited environment.
pub fn strict_reference_toolchain(
    compiler: String,
    requirements: CodegenRequirements,
) -> NativeToolchain {
    let compile_flags = vec![
        "-O2".to_string(),
        "-ffp-contract=off".to_string(),
        "-fno-fast-math".to_string(),
    ];
    let mut link_flags = vec!["-lm".to_string()];
    if !cfg!(target_os = "macos") {
        link_flags.push("-lpthread".to_string());
        link_flags.push("-ldl".to_string());
    }

    // The generated math header uses Accelerate's vForce symbols on macOS
    // even when codegen does not request BLAS.
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
        openmp_enabled: false,
        blas_provider,
    }
}

pub fn test_toolchain(requirements: CodegenRequirements) -> NativeToolchain {
    resolve_toolchain(requirements, &["CHELIS_TEST_CC", "CHELIS_CC"])
}

fn resolve_toolchain(requirements: CodegenRequirements, override_vars: &[&str]) -> NativeToolchain {
    let compiler = resolve_compiler(override_vars);
    // OpenMP is gated on `is_real_gcc(&compiler)`. On macOS, `resolve_compiler`
    // returns "clang" (Apple-clang), so this is always false there: Apple-clang
    // does not bundle libomp and `-fopenmp` is unsupported. The Accelerate
    // -framework link below is therefore decoupled from any OpenMP gating —
    // it is needed unconditionally on macOS for vForce transcendentals
    // regardless of OpenMP. If a future contributor enables OpenMP on
    // Apple-clang via Homebrew libomp (`-Xpreprocessor -fopenmp -lomp`),
    // revisit this branch but keep the Accelerate link unconditional.
    let openmp_enabled = requirements.wants_openmp && is_real_gcc(&compiler);

    let mut compile_flags = vec!["-march=native".to_string()];
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

fn resolve_compiler(override_vars: &[&str]) -> String {
    for var in override_vars {
        if let Ok(override_cc) = std::env::var(var)
            && !override_cc.is_empty()
        {
            return override_cc;
        }
    }
    if cfg!(target_os = "macos") {
        return "clang".to_string();
    }
    if is_real_gcc("gcc") {
        return "gcc".to_string();
    }
    "gcc".to_string()
}

pub fn is_real_gcc(bin: &str) -> bool {
    let Ok(output) = Command::new(bin).arg("--version").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let text = compiler_version_text(&output);
    !text.contains("Apple clang") && !text.contains("clang version")
}

pub fn is_apple_clang(bin: &str) -> bool {
    let Ok(output) = Command::new(bin).arg("--version").output() else {
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
        assert!(runtime.compile_flags.contains(&"-march=native".to_string()));

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
