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

pub fn test_toolchain(requirements: CodegenRequirements) -> NativeToolchain {
    resolve_toolchain(requirements, &["CHELIS_TEST_CC", "CHELIS_CC"])
}

fn resolve_toolchain(requirements: CodegenRequirements, override_vars: &[&str]) -> NativeToolchain {
    let compiler = resolve_compiler(override_vars);
    let openmp_enabled = requirements.wants_openmp && is_real_gcc(&compiler);

    let mut compile_flags = Vec::new();
    let mut link_flags = vec!["-lm".to_string()];
    if !cfg!(target_os = "macos") {
        link_flags.push("-lpthread".to_string());
        link_flags.push("-ldl".to_string());
    }
    if openmp_enabled {
        compile_flags.push("-fopenmp".to_string());
        link_flags.push("-fopenmp".to_string());
    }

    let blas_provider = if requirements.needs_blas {
        if cfg!(target_os = "macos") {
            link_flags.push("-framework".to_string());
            link_flags.push("Accelerate".to_string());
            BlasProvider::Accelerate
        } else {
            link_flags.push("-lopenblas".to_string());
            BlasProvider::OpenBlas
        }
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
