use std::ffi::OsStr;
use std::path::{Path, PathBuf};

pub fn managed_python(workspace_root: &Path) -> Result<PathBuf, String> {
    resolve_managed_python_with(
        workspace_root,
        std::env::var_os("PYO3_PYTHON").as_deref(),
        Path::is_file,
    )
}

fn resolve_managed_python_with(
    workspace_root: &Path,
    configured: Option<&OsStr>,
    is_file: impl Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if let Some(configured) = configured {
        let configured = PathBuf::from(configured);
        let candidate = if configured.is_absolute() {
            configured
        } else {
            workspace_root.join(configured)
        };
        if is_file(&candidate) {
            return Ok(candidate);
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
        return Ok(candidate);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_configured_interpreter_wins() {
        let root = Path::new("/checkout");
        let configured = OsStr::new("/shared/uv/python");
        let resolved = resolve_managed_python_with(root, Some(configured), |path| {
            path == Path::new("/shared/uv/python")
        })
        .expect("valid configured interpreter");
        assert_eq!(resolved, Path::new("/shared/uv/python"));
    }

    #[test]
    fn relative_configured_interpreter_resolves_from_workspace() {
        let root = Path::new("/checkout");
        let configured = OsStr::new("../managed/bin/python");
        let resolved = resolve_managed_python_with(root, Some(configured), |path| {
            path == Path::new("/checkout/../managed/bin/python")
        })
        .expect("valid relative configured interpreter");
        assert_eq!(resolved, Path::new("/checkout/../managed/bin/python"));
    }

    #[test]
    fn invalid_explicit_configuration_does_not_fall_back() {
        let root = Path::new("/checkout");
        let configured = OsStr::new("/missing/python");
        let error = resolve_managed_python_with(root, Some(configured), |path| {
            path == Path::new("/checkout/.venv/bin/python")
        })
        .expect_err("invalid PYO3_PYTHON must be authoritative");
        assert!(error.contains("PYO3_PYTHON"), "{error}");
        assert!(error.contains("/missing/python"), "{error}");
    }

    #[test]
    fn absent_configuration_uses_checkout_venv() {
        let root = Path::new("/checkout");
        let resolved = resolve_managed_python_with(root, None, |path| {
            path == Path::new("/checkout/.venv/bin/python")
        })
        .expect("checkout uv environment");
        assert_eq!(resolved, Path::new("/checkout/.venv/bin/python"));
    }

    #[test]
    fn missing_fallback_diagnostic_names_path_and_uv_setup() {
        let root = Path::new("/checkout");
        let error = resolve_managed_python_with(root, None, |_| false)
            .expect_err("missing checkout interpreter");
        assert!(error.contains("/checkout/.venv/bin/python"), "{error}");
        assert!(error.contains("PYO3_PYTHON"), "{error}");
        assert!(error.contains("uv venv --python 3.11"), "{error}");
        assert!(error.contains("uv python find 3.11"), "{error}");
    }
}
