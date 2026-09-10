//! Compiler preprocessing shared by the primary and backend header census.
//!
//! This primitive preserves attribution. Callers still check source closure,
//! line-directive prohibition, configuration invariance and final authority.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Environment<'a> {
    pub compiler: &'a str,
    pub language: &'a str,
    pub arguments: &'a [&'a str],
    pub include_dirs: &'a [PathBuf],
    pub hermetic: bool,
}

impl Environment<'_> {
    pub fn native_c() -> Self {
        Self {
            compiler: "cc",
            language: "c",
            arguments: &[],
            include_dirs: &[],
            hermetic: false,
        }
    }
}

pub fn preprocess_root(
    _include_dir: &Path,
    _root: &str,
    _environment: &Environment<'_>,
) -> Result<BTreeMap<String, String>, String> {
    Err("preprocessor not implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "chelis-census-preprocessor-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("create unique fixture");
            Self(path)
        }
        fn write(&self, path: &str, source: &str) {
            let path = self.0.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, source).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("restore owned preprocessor fixture");
        }
    }

    #[test]
    fn transitive_macro_declarations_keep_their_full_owned_header_path() {
        let fixture = Fixture::new();
        fixture.write("root.h", "#include <first/value.h>\n#include \"second/value.h\"\n");
        fixture.write("first/value.h", "#define PAYLOAD double\nPAYLOAD first_value(void);\n");
        fixture.write("second/value.h", "long long second_value(void);\n");
        let rows = preprocess_root(&fixture.0, "root.h", &Environment::native_c()).unwrap();
        assert!(rows["first/value.h"].contains("double first_value(void)"));
        assert!(rows["second/value.h"].contains("long long second_value(void)"));
        assert!(!rows.contains_key("value.h"), "same basenames cannot collapse identities");
    }

    #[test]
    fn external_same_basename_is_not_attributed_to_the_owned_root() {
        let fixture = Fixture::new();
        fixture.write("owned/root.h", "#include <root.h>\nint owned_value(void);\n");
        fixture.write("sdk/root.h", "double external_value(void);\n");
        // Include search order is explicit. The owned root uses an absolute path.
        let includes = [fixture.0.join("sdk")];
        let environment = Environment {
            include_dirs: &includes,
            ..Environment::native_c()
        };
        let rows = preprocess_root(&fixture.0.join("owned"), "root.h", &environment).unwrap();
        assert!(rows["root.h"].contains("int owned_value(void)"));
        assert!(!rows.values().any(|text| text.contains("external_value")));
    }

    #[test]
    fn declared_cxx_and_objective_cxx_lanes_use_the_real_preprocessor() {
        let fixture = Fixture::new();
        fixture.write("root.h", concat!(
            "#ifdef __cplusplus\nint cxx_surface(void);\n#endif\n",
            "#ifdef __OBJC__\nint objc_surface(void);\n#endif\n"
        ));
        let c = preprocess_root(&fixture.0, "root.h", &Environment::native_c()).unwrap();
        assert!(!c.values().any(|text| text.contains("cxx_surface")));
        for language in ["c++", "objective-c++"] {
            let environment = Environment {
                compiler: "clang",
                language,
                hermetic: true,
                ..Environment::native_c()
            };
            let rows = preprocess_root(&fixture.0, "root.h", &environment).unwrap();
            assert!(rows["root.h"].contains("cxx_surface"));
            assert_eq!(rows["root.h"].contains("objc_surface"), language == "objective-c++");
        }
    }

    #[test]
    fn missing_compiler_and_rejected_input_cannot_be_an_empty_success() {
        let fixture = Fixture::new();
        fixture.write("root.h", "int working_surface(void);\n");
        assert!(preprocess_root(&fixture.0, "root.h", &Environment::native_c()).is_ok());
        let missing = Environment {
            compiler: "/nonexistent/chelis-census-compiler",
            ..Environment::native_c()
        };
        assert!(preprocess_root(&fixture.0, "root.h", &missing).unwrap_err().contains("compiler"));
        fixture.write("root.h", "#error rejected owned header\n");
        assert!(preprocess_root(&fixture.0, "root.h", &Environment::native_c())
            .unwrap_err().contains("rejected owned header"));
    }
}
