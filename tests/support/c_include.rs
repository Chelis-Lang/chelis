//! Local header lookup follows the compiler's quoted-versus-angle search order.

use std::path::{Component, Path};

fn directive(line: &str) -> Option<(&str, bool)> {
    let rest = line.trim_start().strip_prefix('#')?.trim_start();
    let rest = rest.strip_prefix("include")?;
    if !rest.starts_with(|c: char| c.is_whitespace() || c == '"' || c == '<') {
        return None;
    }
    let rest = rest.trim_start();
    let (quoted, close) = match rest.chars().next()? {
        '"' => (true, '"'),
        '<' => (false, '>'),
        _ => return None,
    };
    let (name, _) = rest[1..].split_once(close)?;
    (!name.is_empty()).then_some((name, quoted))
}

pub fn include_name(line: &str) -> Option<&str> {
    directive(line).map(|(name, _)| name)
}

fn normalize(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for part in path.components() {
        match part {
            Component::Normal(value) => parts.push(value.to_str()?),
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop()?;
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

pub fn resolve(including: &str, line: &str, exists: impl Fn(&str) -> bool) -> Option<String> {
    let (name, quoted) = directive(line)?;
    if quoted {
        let parent = Path::new(including).parent()?;
        if let Some(relative) = normalize(&parent.join(name))
            && exists(&relative)
        {
            return Some(relative);
        }
    }
    normalize(Path::new(name)).filter(|candidate| exists(candidate))
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
            resolve("detail/root.h", "#include \"value.h\"", |name| name
                == "value.h")
            .as_deref(),
            Some("value.h")
        );
        assert!(
            resolve("detail/root.h", "#include <value.h>", |name| {
                name == "detail/value.h"
            })
            .is_none()
        );
    }

    #[test]
    fn relative_parent_components_preserve_one_canonical_local_identity() {
        for spelling in ["../value.h", ".././value.h", "../nested/../value.h"] {
            assert_eq!(
                resolve(
                    "detail/root.h",
                    &format!("#include \"{spelling}\""),
                    |name| { name == "value.h" }
                )
                .as_deref(),
                Some("value.h")
            );
        }
        for spelling in ["../../outside.h", "/vendor/outside.h"] {
            assert!(
                resolve("detail/root.h", &format!("#include \"{spelling}\""), |_| {
                    true
                })
                .is_none()
            );
        }
    }

    #[test]
    fn only_complete_include_directives_produce_a_literal_name() {
        for line in [
            "#include \"value.h\"",
            "  # include <value.h>",
            "#\tinclude\t\"value.h\"",
        ] {
            assert_eq!(include_name(line), Some("value.h"));
        }
        for line in [
            "#included <value.h>",
            "#include_next <value.h>",
            "#include VALUE",
            "#include \"value.h",
            "#include <>",
            "#define VALUE value.h",
        ] {
            assert_eq!(include_name(line), None);
        }
    }
}
