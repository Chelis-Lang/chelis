//! Toolchain resolution: which installed version answers a `chelis`
//! invocation, decided fresh at every call.
//!
//! # Precedence (first match wins)
//!
//! 1. a leading `+<ver>` CLI argument (for example `chelis +0.13.0 ...`);
//!    stripped from the forwarded args.
//! 2. the `CHELIS_TOOLCHAIN` environment variable.
//! 3. a `chelis-toolchain` file found by walking up from the cwd (its
//!    first non-empty line is a bare version).
//! 4. the nearest `reef.toml` `compiler = "=X.Y.Z"` pin, walking up.
//! 5. the recorded default (`<home>/default`).
//!
//! This is the reasoned order from the packaging design (§5.2): the
//! explicit per-invocation `+ver` and the `CHELIS_TOOLCHAIN` env beat a
//! deliberate directory override (`chelis-toolchain`), which in turn
//! beats the package pin, which beats the bare-machine default.
//!
//! # Two independent walks (deliberate)
//!
//! `chelis-toolchain` (level 3) is "ranked above the package pin"
//! globally, not per directory. So we do a full upward walk for a
//! `chelis-toolchain` file first; only if none exists anywhere above the
//! cwd do we do a second full walk for a `reef.toml` pin. A
//! `chelis-toolchain` two directories up therefore beats a `reef.toml`
//! pin in the cwd, which is what "ranked above the package pin" means.
//!
//! Within the `reef.toml` walk the **nearest** manifest owns the answer:
//! a found `reef.toml` with no usable exact pin stops the walk and falls
//! through to the default (it does not keep climbing to a farther
//! manifest). This mirrors the legacy launcher's behavior.
//!
//! Resolution decides only the version *string* and where it came from.
//! Whether that version is actually installed is a separate check (see
//! [`crate::shim`]) so the not-installed case can be a loud, actionable
//! error instead of a silent fall-through.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::paths::Store;

/// Where a resolved version came from. Carried into diagnostics so a
/// not-installed error can name the exact source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolchainSource {
    /// A leading `+<ver>` CLI argument.
    PlusArg,
    /// The `CHELIS_TOOLCHAIN` environment variable.
    EnvVar,
    /// A `chelis-toolchain` file at this path.
    ToolchainFile(PathBuf),
    /// A `reef.toml` `compiler` pin at this path.
    ReefPin(PathBuf),
    /// The recorded default file at this path.
    Default(PathBuf),
}

impl fmt::Display for ToolchainSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlusArg => write!(f, "the +<ver> argument"),
            Self::EnvVar => write!(f, "the CHELIS_TOOLCHAIN environment variable"),
            Self::ToolchainFile(p) => write!(f, "{}", p.display()),
            Self::ReefPin(p) => write!(f, "the compiler pin in {}", p.display()),
            Self::Default(p) => write!(f, "the recorded default ({})", p.display()),
        }
    }
}

/// A successful resolution: the version, its source, and the args to
/// forward to the toolchain (the leading `+<ver>`, if any, removed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub version: String,
    pub source: ToolchainSource,
    pub forwarded_args: Vec<OsString>,
}

/// Inputs to [`resolve`]. Everything is injected so resolution is pure
/// with respect to the process environment (filesystem reads under
/// `cwd`/`store` are exercised against tempdirs in tests).
pub struct ResolveInput<'a> {
    /// Args after `argv[0]` (the candidate `+<ver>` is `args[0]`).
    pub args: &'a [OsString],
    /// The value of `CHELIS_TOOLCHAIN`, if set.
    pub env_toolchain: Option<String>,
    /// The directory the upward walks start from.
    pub cwd: &'a Path,
    /// The store (for the recorded default).
    pub store: &'a Store,
}

/// Resolve the active toolchain. Returns `None` only when no level
/// produced a version (no `+ver`, no env, no `chelis-toolchain`, no
/// `reef.toml` pin, and no recorded default).
#[must_use]
pub fn resolve(input: &ResolveInput) -> Option<Resolution> {
    // 1. Leading `+<ver>` argument.
    if let Some(first) = input.args.first()
        && let Some(s) = first.to_str()
        && let Some(ver) = s.strip_prefix('+')
        && !ver.is_empty()
    {
        return Some(Resolution {
            version: ver.to_string(),
            source: ToolchainSource::PlusArg,
            forwarded_args: input.args[1..].to_vec(),
        });
    }

    // Levels 2..=5 forward every argument unchanged.
    let forwarded = input.args.to_vec();

    // 2. CHELIS_TOOLCHAIN env.
    if let Some(v) = &input.env_toolchain {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            return Some(Resolution {
                version: trimmed.to_string(),
                source: ToolchainSource::EnvVar,
                forwarded_args: forwarded,
            });
        }
    }

    // 3. chelis-toolchain file (full upward walk).
    if let Some((ver, path)) = find_toolchain_file(input.cwd) {
        return Some(Resolution {
            version: ver,
            source: ToolchainSource::ToolchainFile(path),
            forwarded_args: forwarded,
        });
    }

    // 4. nearest reef.toml compiler pin (full upward walk).
    if let Some((ver, path)) = find_reef_pin(input.cwd) {
        return Some(Resolution {
            version: ver,
            source: ToolchainSource::ReefPin(path),
            forwarded_args: forwarded,
        });
    }

    // 5. recorded default.
    if let Some(ver) = input.store.read_default() {
        return Some(Resolution {
            version: ver,
            source: ToolchainSource::Default(input.store.default_file()),
            forwarded_args: forwarded,
        });
    }

    None
}

/// Walk up from `start`, returning the first `chelis-toolchain` file's
/// first non-empty line (trimmed) and the file path.
fn find_toolchain_file(start: &Path) -> Option<(String, PathBuf)> {
    for dir in ancestors(start) {
        let candidate = dir.join("chelis-toolchain");
        if candidate.is_file() {
            if let Ok(contents) = fs::read_to_string(&candidate)
                && let Some(ver) = first_nonempty_line(&contents)
            {
                return Some((ver, candidate));
            }
            // A present but blank `chelis-toolchain` file owns the
            // answer at this level: stop walking and fall through.
            return None;
        }
    }
    None
}

/// Walk up from `start` to the nearest `reef.toml`. If it carries a
/// usable exact `compiler` pin, return it; otherwise the nearest
/// manifest still owns the answer, so stop and fall through.
fn find_reef_pin(start: &Path) -> Option<(String, PathBuf)> {
    for dir in ancestors(start) {
        let manifest = dir.join("reef.toml");
        if manifest.is_file() {
            let pin = fs::read_to_string(&manifest)
                .ok()
                .and_then(|c| compiler_pin(&c));
            return pin.map(|v| (v, manifest));
        }
    }
    None
}

/// Parse a `compiler` pin out of `reef.toml` text. Reads
/// `package.compiler` (the canonical location) with a top-level
/// `compiler` fallback, strips an optional leading `=` (chelis pins are
/// exact, written `"=X.Y.Z"`), trims, and rejects anything that is not a
/// safe path component (so a malformed or hostile pin falls through
/// rather than being joined into a path).
fn compiler_pin(text: &str) -> Option<String> {
    let value: toml::Value = toml::from_str(text).ok()?;
    let raw = value
        .get("package")
        .and_then(|p| p.get("compiler"))
        .or_else(|| value.get("compiler"))
        .and_then(|c| c.as_str())?;
    let stripped = raw.strip_prefix('=').unwrap_or(raw).trim();
    if stripped.is_empty() || !crate::version::is_safe_path_component(stripped) {
        return None;
    }
    Some(stripped.to_string())
}

/// The first non-empty, comment-free, trimmed line of `text`. Lines
/// starting with `#` are treated as comments.
fn first_nonempty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
}

/// `start` and all of its ancestors, nearest first.
fn ancestors(start: &Path) -> impl Iterator<Item = &Path> {
    start.ancestors()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn osv(items: &[&str]) -> Vec<OsString> {
        items.iter().map(OsString::from).collect()
    }

    /// Build a store whose default file is seeded with `default`.
    fn store_with_default(dir: &Path, default: Option<&str>) -> Store {
        let s = Store::at(dir.join("home"));
        if let Some(d) = default {
            s.write_default(d).unwrap();
        }
        s
    }

    #[test]
    fn plus_arg_wins_and_is_stripped() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_default(tmp.path(), Some("9.9.9"));
        let args = osv(&["+0.13.0", "build", "main.ch"]);
        let input = ResolveInput {
            args: &args,
            env_toolchain: Some("0.1.0".to_string()),
            cwd: tmp.path(),
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.13.0");
        assert_eq!(r.source, ToolchainSource::PlusArg);
        assert_eq!(r.forwarded_args, osv(&["build", "main.ch"]));
    }

    #[test]
    fn bare_plus_is_not_a_version() {
        // A lone "+" is not a `+ver`; resolution falls to the env.
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_default(tmp.path(), None);
        let args = osv(&["+", "build"]);
        let input = ResolveInput {
            args: &args,
            env_toolchain: Some("0.1.0".to_string()),
            cwd: tmp.path(),
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.1.0");
        assert_eq!(r.source, ToolchainSource::EnvVar);
        assert_eq!(r.forwarded_args, osv(&["+", "build"]));
    }

    #[test]
    fn env_beats_files_and_default() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("chelis-toolchain"), "0.5.0\n").unwrap();
        fs::write(
            tmp.path().join("reef.toml"),
            "[package]\ncompiler = \"=0.6.0\"\n",
        )
        .unwrap();
        let store = store_with_default(tmp.path(), Some("0.9.0"));
        let args: Vec<OsString> = vec![];
        let input = ResolveInput {
            args: &args,
            env_toolchain: Some("0.2.0".to_string()),
            cwd: tmp.path(),
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.2.0");
        assert_eq!(r.source, ToolchainSource::EnvVar);
    }

    #[test]
    fn toolchain_file_beats_reef_pin_even_higher_up() {
        // chelis-toolchain in the parent, reef.toml pin in the child.
        // The toolchain file is "ranked above the package pin" globally.
        let tmp = tempfile::tempdir().unwrap();
        let child = tmp.path().join("child");
        fs::create_dir_all(&child).unwrap();
        fs::write(tmp.path().join("chelis-toolchain"), "# comment\n0.5.0\n").unwrap();
        fs::write(
            child.join("reef.toml"),
            "[package]\ncompiler = \"=0.6.0\"\n",
        )
        .unwrap();
        let store = store_with_default(tmp.path(), Some("0.9.0"));
        let args: Vec<OsString> = vec![];
        let input = ResolveInput {
            args: &args,
            env_toolchain: None,
            cwd: &child,
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.5.0");
        match r.source {
            ToolchainSource::ToolchainFile(p) => {
                assert!(p.ends_with("chelis-toolchain"));
            }
            other => panic!("expected ToolchainFile, got {other:?}"),
        }
    }

    #[test]
    fn reef_pin_resolves_when_no_toolchain_file() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            tmp.path().join("a").join("reef.toml"),
            "[package]\nname = \"x\"\ncompiler = \"=0.7.10\"\n",
        )
        .unwrap();
        let store = store_with_default(tmp.path(), Some("0.9.0"));
        let args: Vec<OsString> = vec![];
        let input = ResolveInput {
            args: &args,
            env_toolchain: None,
            cwd: &nested,
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.7.10");
        match r.source {
            ToolchainSource::ReefPin(p) => assert!(p.ends_with("reef.toml")),
            other => panic!("expected ReefPin, got {other:?}"),
        }
    }

    #[test]
    fn nearest_reef_without_pin_falls_to_default_not_farther_manifest() {
        // Outer manifest has a pin; inner manifest (nearer the cwd) has
        // none. The nearer manifest owns the answer: fall to default.
        let tmp = tempfile::tempdir().unwrap();
        let inner = tmp.path().join("inner");
        fs::create_dir_all(&inner).unwrap();
        fs::write(
            tmp.path().join("reef.toml"),
            "[package]\ncompiler = \"=0.6.0\"\n",
        )
        .unwrap();
        fs::write(inner.join("reef.toml"), "[package]\nname = \"x\"\n").unwrap();
        let store = store_with_default(tmp.path(), Some("0.9.0"));
        let args: Vec<OsString> = vec![];
        let input = ResolveInput {
            args: &args,
            env_toolchain: None,
            cwd: &inner,
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.9.0");
        assert!(matches!(r.source, ToolchainSource::Default(_)));
    }

    #[test]
    fn default_is_last_resort() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_default(tmp.path(), Some("0.12.0"));
        let args: Vec<OsString> = vec![];
        let input = ResolveInput {
            args: &args,
            env_toolchain: None,
            cwd: tmp.path(),
            store: &store,
        };
        let r = resolve(&input).unwrap();
        assert_eq!(r.version, "0.12.0");
        assert!(matches!(r.source, ToolchainSource::Default(_)));
    }

    #[test]
    fn nothing_resolves_to_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_default(tmp.path(), None);
        let args: Vec<OsString> = vec![];
        let input = ResolveInput {
            args: &args,
            env_toolchain: None,
            cwd: tmp.path(),
            store: &store,
        };
        assert!(resolve(&input).is_none());
    }

    #[test]
    fn hostile_compiler_pin_is_rejected() {
        // A traversal in the pin must never become a path component.
        assert_eq!(compiler_pin("[package]\ncompiler = \"=../../etc\"\n"), None);
        assert_eq!(compiler_pin("[package]\ncompiler = \"=a/b\"\n"), None);
        // A normal exact pin parses.
        assert_eq!(
            compiler_pin("[package]\ncompiler = \"=0.12.0\"\n").as_deref(),
            Some("0.12.0")
        );
        // A range pin (no leading `=`) is still parsed as a token; the
        // installed-ness check is what rejects a non-installed name.
        assert_eq!(
            compiler_pin("[package]\ncompiler = \"0.12.0\"\n").as_deref(),
            Some("0.12.0")
        );
    }
}
