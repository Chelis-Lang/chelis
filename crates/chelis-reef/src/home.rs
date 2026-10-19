//! One resolver for the Chelis home, the Reef registry root, and the
//! on-disk compiler cache directories.
//!
//! The layout follows `spec/design/chelis_packaging_and_install.md` section 4.
//! Every variable counts as unset when it is empty, so an empty value never
//! names a path relative to the working directory.
//!
//! | State | Resolution order |
//! |---|---|
//! | Chelis home | `$CHELIS_HOME`, else `$HOME/.chelis` |
//! | Reef registry | `$CHELIS_REEF_HOME`, else `$CHELIS_HOME/reef`, else `$HOME/.chelis/reef` |
//! | cache `<name>` | `$CHELIS_REEF_HOME/.cache/<name>`, else `$CHELIS_HOME/reef/.cache/<name>`, else `$XDG_CACHE_HOME/chelis/<name>`, else `$HOME/.cache/chelis/<name>` |
//!
//! A variable that chooses the registry root also holds the caches, under
//! `<registry>/.cache`. Without `CHELIS_REEF_HOME` and `CHELIS_HOME` the
//! caches stay in the XDG cache directory.

use std::ffi::OsString;
use std::path::PathBuf;

/// The environment values that the resolver reads. Empty values are
/// stored as `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HomeEnv {
    reef_home: Option<OsString>,
    chelis_home: Option<OsString>,
    xdg_cache_home: Option<OsString>,
    home: Option<OsString>,
}

fn non_empty(value: Option<OsString>) -> Option<OsString> {
    value.filter(|value| !value.is_empty())
}

impl HomeEnv {
    /// Read the four variables from the process environment.
    pub(crate) fn from_process() -> Self {
        Self::from_lookup(|key| std::env::var_os(key))
    }

    /// Read the four variables through `lookup`.
    pub(crate) fn from_lookup(lookup: impl Fn(&str) -> Option<OsString>) -> Self {
        Self {
            reef_home: non_empty(lookup("CHELIS_REEF_HOME")),
            chelis_home: non_empty(lookup("CHELIS_HOME")),
            xdg_cache_home: non_empty(lookup("XDG_CACHE_HOME")),
            home: non_empty(lookup("HOME")),
        }
    }

    /// `$CHELIS_HOME`, else `$HOME/.chelis`.
    pub(crate) fn chelis_home(&self) -> Result<PathBuf, String> {
        if let Some(chelis_home) = &self.chelis_home {
            return Ok(PathBuf::from(chelis_home));
        }
        match &self.home {
            Some(home) => Ok(PathBuf::from(home).join(".chelis")),
            None => Err("HOME is not set".to_string()),
        }
    }

    /// The registry root that a Chelis variable chooses explicitly:
    /// `$CHELIS_REEF_HOME`, else `$CHELIS_HOME/reef`.
    fn explicit_registry_root(&self) -> Option<PathBuf> {
        if let Some(reef_home) = &self.reef_home {
            return Some(PathBuf::from(reef_home));
        }
        self.chelis_home
            .as_ref()
            .map(|chelis_home| PathBuf::from(chelis_home).join("reef"))
    }

    /// The Reef registry root: the explicit root, else `$HOME/.chelis/reef`.
    pub(crate) fn registry_root(&self) -> Result<PathBuf, String> {
        if let Some(root) = self.explicit_registry_root() {
            return Ok(root);
        }
        Ok(self.chelis_home()?.join("reef"))
    }

    /// The directory of the on-disk cache `name`. Returns `None` when no
    /// variable resolves, and the caller then works without the cache.
    pub(crate) fn cache_dir(&self, name: &str) -> Option<PathBuf> {
        if let Some(root) = self.explicit_registry_root() {
            return Some(root.join(".cache").join(name));
        }
        if let Some(xdg) = &self.xdg_cache_home {
            return Some(PathBuf::from(xdg).join("chelis").join(name));
        }
        self.home
            .as_ref()
            .map(|home| PathBuf::from(home).join(".cache").join("chelis").join(name))
    }
}

/// Resolve the directory of the named on-disk compiler cache from the
/// process environment. See the module documentation for the order.
/// Returns `None` when none of `CHELIS_REEF_HOME`, `CHELIS_HOME`,
/// `XDG_CACHE_HOME`, and `HOME` is set and non-empty.
pub fn cache_dir_for(name: &str) -> Option<PathBuf> {
    HomeEnv::from_process().cache_dir(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn env(pairs: &[(&str, &str)]) -> HomeEnv {
        HomeEnv::from_lookup(|key| {
            pairs
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| OsString::from(value))
        })
    }

    #[test]
    fn default_layout_uses_home_and_the_xdg_cache() {
        let env = env(&[("HOME", "/h")]);
        assert_eq!(env.chelis_home().unwrap(), Path::new("/h/.chelis"));
        assert_eq!(env.registry_root().unwrap(), Path::new("/h/.chelis/reef"));
        assert_eq!(
            env.cache_dir("typecheck").unwrap(),
            Path::new("/h/.cache/chelis/typecheck")
        );
    }

    #[test]
    fn xdg_cache_home_moves_only_the_caches() {
        let env = env(&[("HOME", "/h"), ("XDG_CACHE_HOME", "/x")]);
        assert_eq!(env.registry_root().unwrap(), Path::new("/h/.chelis/reef"));
        assert_eq!(
            env.cache_dir("compiled").unwrap(),
            Path::new("/x/chelis/compiled")
        );
    }

    #[test]
    fn chelis_home_holds_the_registry_and_the_caches() {
        let env = env(&[
            ("HOME", "/h"),
            ("XDG_CACHE_HOME", "/x"),
            ("CHELIS_HOME", "/c"),
        ]);
        assert_eq!(env.chelis_home().unwrap(), Path::new("/c"));
        assert_eq!(env.registry_root().unwrap(), Path::new("/c/reef"));
        assert_eq!(
            env.cache_dir("prepared-graphs").unwrap(),
            Path::new("/c/reef/.cache/prepared-graphs")
        );
    }

    #[test]
    fn chelis_home_needs_no_home() {
        let env = env(&[("CHELIS_HOME", "/c")]);
        assert_eq!(env.registry_root().unwrap(), Path::new("/c/reef"));
        assert_eq!(
            env.cache_dir("typecheck").unwrap(),
            Path::new("/c/reef/.cache/typecheck")
        );
    }

    #[test]
    fn chelis_reef_home_overrides_chelis_home() {
        let env = env(&[
            ("HOME", "/h"),
            ("XDG_CACHE_HOME", "/x"),
            ("CHELIS_HOME", "/c"),
            ("CHELIS_REEF_HOME", "/r"),
        ]);
        assert_eq!(env.chelis_home().unwrap(), Path::new("/c"));
        assert_eq!(env.registry_root().unwrap(), Path::new("/r"));
        assert_eq!(
            env.cache_dir("compiled").unwrap(),
            Path::new("/r/.cache/compiled")
        );
    }

    #[test]
    fn empty_values_count_as_unset() {
        let env = env(&[
            ("HOME", "/h"),
            ("XDG_CACHE_HOME", ""),
            ("CHELIS_HOME", ""),
            ("CHELIS_REEF_HOME", ""),
        ]);
        assert_eq!(env.chelis_home().unwrap(), Path::new("/h/.chelis"));
        assert_eq!(env.registry_root().unwrap(), Path::new("/h/.chelis/reef"));
        assert_eq!(
            env.cache_dir("typecheck").unwrap(),
            Path::new("/h/.cache/chelis/typecheck")
        );
    }

    #[test]
    fn no_variable_gives_no_root() {
        let env = env(&[("HOME", ""), ("CHELIS_REEF_HOME", "")]);
        assert_eq!(env.chelis_home().unwrap_err(), "HOME is not set");
        assert_eq!(env.registry_root().unwrap_err(), "HOME is not set");
        assert_eq!(env.cache_dir("typecheck"), None);
    }
}
