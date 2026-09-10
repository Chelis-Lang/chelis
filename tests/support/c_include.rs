//! Local header lookup follows the compiler's quoted-versus-angle search order.

pub fn include_name(_line: &str) -> Option<&str> {
    todo!("parse the include directive")
}

pub fn resolve(
    _including: &str,
    _line: &str,
    _exists: impl Fn(&str) -> bool,
) -> Option<String> {
    todo!("resolve against the including directory and published root")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_headers_prefer_the_including_directory_but_angles_use_the_root() {
        let files = ["detail/value.h", "value.h"];
        let exists = |name: &str| files.contains(&name);
        assert_eq!(
            resolve("detail/root.h", "#include \"value.h\"", exists).as_deref(),
            Some("detail/value.h")
        );
        assert_eq!(
            resolve("detail/root.h", "#include <value.h>", exists).as_deref(),
            Some("value.h")
        );
        assert_eq!(
            resolve("detail/root.h", "#include \"value.h\"", |name| name == "value.h")
                .as_deref(),
            Some("value.h")
        );
        assert!(resolve("detail/root.h", "#include <value.h>", |name| {
            name == "detail/value.h"
        })
        .is_none());
    }

    #[test]
    fn relative_parent_components_preserve_one_canonical_local_identity() {
        for spelling in ["../value.h", ".././value.h", "../nested/../value.h"] {
            assert_eq!(
                resolve("detail/root.h", &format!("#include \"{spelling}\""), |name| {
                    name == "value.h"
                })
                .as_deref(),
                Some("value.h")
            );
        }
        for spelling in ["../../outside.h", "/vendor/outside.h"] {
            assert!(resolve("detail/root.h", &format!("#include \"{spelling}\""), |_| true)
                .is_none());
        }
    }

    #[test]
    fn only_complete_include_directives_produce_a_literal_name() {
        for line in ["#include \"value.h\"", "  # include <value.h>", "#\tinclude\t\"value.h\""] {
            assert_eq!(include_name(line), Some("value.h"));
        }
        for line in ["#included <value.h>", "#include_next <value.h>", "#include VALUE",
                     "#include \"value.h", "#include <>", "#define VALUE value.h"] {
            assert_eq!(include_name(line), None);
        }
    }
}
