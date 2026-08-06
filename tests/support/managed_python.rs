use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

const PYTHON_VERSION_PROBE: &str =
    "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 86)";

pub fn managed_python(workspace_root: &Path) -> Result<PathBuf, String> {
    resolve_managed_python_with(
        workspace_root,
        std::env::var_os("PYO3_PYTHON").as_deref(),
        Path::is_file,
        validate_python,
    )
}

fn resolve_managed_python_with(
    workspace_root: &Path,
    configured: Option<&OsStr>,
    is_file: impl Fn(&Path) -> bool,
    validate: impl Fn(&Path) -> Result<(), String>,
) -> Result<PathBuf, String> {
    if let Some(configured) = configured {
        let configured = PathBuf::from(configured);
        let candidate = if configured.is_absolute() {
            configured
        } else {
            workspace_root.join(configured)
        };
        if is_file(&candidate) {
            return validate(&candidate)
                .map(|()| candidate.clone())
                .map_err(|reason| {
                    format!(
                        "PYO3_PYTHON is set, but the configured interpreter at {} \
                     is not a usable Python 3.11+ interpreter: {reason}. The \
                     explicit setting is authoritative and will not fall back \
                     to the checkout environment. Install uv and run \
                     `uv python install 3.11`, then set \
                     `PYO3_PYTHON=\"$(uv python find 3.11)\"` or create the \
                     checkout environment with `uv venv --python 3.11`.",
                        candidate.display()
                    )
                });
        }
        return Err(format!(
            "PYO3_PYTHON is set, but the configured interpreter does not \
             exist at {}. The explicit setting is authoritative and will \
             not fall back to the checkout environment. Install uv and run \
             `uv python install 3.11`, then set \
             `PYO3_PYTHON=\"$(uv python find 3.11)\"` or create the checkout \
             environment with `uv venv --python 3.11`.",
            candidate.display()
        ));
    }

    let candidate = workspace_root.join(".venv/bin/python");
    if is_file(&candidate) {
        return validate(&candidate)
            .map(|()| candidate.clone())
            .map_err(|reason| {
                format!(
                    "the checkout interpreter at {} is not a usable Python 3.11+ \
                 interpreter: {reason}. Install uv and run \
                 `uv python install 3.11`, then set \
                 `PYO3_PYTHON=\"$(uv python find 3.11)\"` or recreate the \
                 checkout environment with `uv venv --python 3.11`.",
                    candidate.display()
                )
            });
    }
    Err(format!(
        "no managed Python interpreter was configured: PYO3_PYTHON is \
         unset and the checkout fallback does not exist at {}. Install uv \
         and run `uv python install 3.11`, then set \
         `PYO3_PYTHON=\"$(uv python find 3.11)\"` or create the checkout \
         environment with `uv venv --python 3.11`.",
        candidate.display()
    ))
}

fn validate_python(candidate: &Path) -> Result<(), String> {
    let output = Command::new(candidate)
        .arg("-c")
        .arg(PYTHON_VERSION_PROBE)
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if !stderr.trim().is_empty() {
        stderr.trim()
    } else {
        stdout.trim()
    };
    if detail.is_empty() {
        Err(format!("the Python version probe exited {}", output.status))
    } else {
        Err(format!(
            "the Python version probe exited {}: {detail}",
            output.status
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_configured_interpreter_wins() {
        let root = Path::new("/checkout");
        let configured = OsStr::new("/shared/uv/python");
        let resolved = resolve_managed_python_with(
            root,
            Some(configured),
            |path| path == Path::new("/shared/uv/python"),
            |_| Ok(()),
        )
        .expect("valid configured interpreter");
        assert_eq!(resolved, Path::new("/shared/uv/python"));
    }

    #[test]
    fn relative_configured_interpreter_resolves_from_workspace() {
        let root = Path::new("/checkout");
        let configured = OsStr::new("../managed/bin/python");
        let resolved = resolve_managed_python_with(
            root,
            Some(configured),
            |path| path == Path::new("/checkout/../managed/bin/python"),
            |_| Ok(()),
        )
        .expect("valid relative configured interpreter");
        assert_eq!(resolved, Path::new("/checkout/../managed/bin/python"));
    }

    #[test]
    fn invalid_explicit_configuration_does_not_fall_back() {
        let root = Path::new("/checkout");
        let configured = OsStr::new("/missing/python");
        let error = resolve_managed_python_with(
            root,
            Some(configured),
            |path| path == Path::new("/checkout/.venv/bin/python"),
            |_| Ok(()),
        )
        .expect_err("invalid PYO3_PYTHON must be authoritative");
        assert!(error.contains("PYO3_PYTHON"), "{error}");
        assert!(error.contains("/missing/python"), "{error}");
    }

    #[test]
    fn existing_non_python_configuration_is_rejected() {
        let root = Path::new("/checkout");
        let configured = std::env::current_exe().expect("current test executable");
        let error = resolve_managed_python_with(
            root,
            Some(configured.as_os_str()),
            Path::is_file,
            validate_python,
        )
        .expect_err("a non-Python executable must be rejected");
        assert!(error.contains("PYO3_PYTHON"), "{error}");
        assert!(error.contains(&configured.display().to_string()), "{error}");
        assert!(error.contains("Python 3.11"), "{error}");
        assert!(error.contains("uv python find 3.11"), "{error}");
    }

    #[test]
    fn absent_configuration_uses_checkout_venv() {
        let root = Path::new("/checkout");
        let resolved = resolve_managed_python_with(
            root,
            None,
            |path| path == Path::new("/checkout/.venv/bin/python"),
            |_| Ok(()),
        )
        .expect("checkout uv environment");
        assert_eq!(resolved, Path::new("/checkout/.venv/bin/python"));
    }

    #[test]
    fn missing_fallback_diagnostic_names_path_and_uv_setup() {
        let root = Path::new("/checkout");
        let error = resolve_managed_python_with(root, None, |_| false, |_| Ok(()))
            .expect_err("missing checkout interpreter");
        assert!(error.contains("/checkout/.venv/bin/python"), "{error}");
        assert!(error.contains("PYO3_PYTHON"), "{error}");
        assert!(error.contains("uv venv --python 3.11"), "{error}");
        assert!(error.contains("uv python find 3.11"), "{error}");
    }
}
