//! The `conform` control surface in a shell's `reef.toml`, read **structurally**.
//!
//! Contract §8 gives a shell exactly one knob over its conformance tooling:
//! `conform.local_skills`, the repo-local domain skills `sync` preserves and the
//! §8 audit exempts (chelis#651). Everything else under `conform` is a control
//! the contract does not define, and §8 requires the audit to *report* it rather
//! than ignore it, so a shell can never believe in a knob the tool does not
//! implement (chelis#1262).
//!
//! **Why this module parses instead of scanning.** The first two implementations
//! were hand-rolled line scans over `reef.toml`, and two review rounds produced
//! six ways to write the same declaration past them:
//!
//! | spelling | why the scan missed it |
//! |---|---|
//! | `[conform.skills]` + `exclude = …` | header was not literally `[conform]` |
//! | `skills.exclude = …` | dotted key failed a character allowlist and was dropped |
//! | `"exclude" = …` | quoted key, same allowlist |
//! | `conform = { exclude = … }` | no table header, so the scan never entered scope |
//! | `conform.exclude = …` | same |
//! | `local_skills = ["a["]` then `exclude = …` | a bracket inside a string desynchronized a raw depth counter, swallowing every later key |
//!
//! Each patch closed the spellings in front of it and left the shape of the
//! problem intact, because a line scan has to *guess* at TOML's structure and
//! every guess is a place to differ from it. A parsed document has no spellings:
//! header form, inline table, dotted key, quoted key, sub-table,
//! array-of-tables, and bracket-bearing string values all produce one value, and
//! spelling-independence holds by construction rather than by enumeration. The
//! durable lesson from those rounds still applies to the parse itself: **a
//! checker that silently normalizes or drops what it cannot read is the bypass**,
//! so an unparseable manifest here is a loud failure, never a skip.

use std::collections::BTreeSet;

/// The one key contract §8 defines under `conform`.
pub const LOCAL_SKILLS: &str = "local_skills";

/// The table name that is the control surface.
pub const CONFORM: &str = "conform";

/// What a shell declared under `conform`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ConformDecl {
    /// `conform.local_skills`, the recognized declaration: repo-local domain
    /// skills the shell owns (chelis#651). Empty when absent.
    pub local_skills: Vec<String>,
    /// Every other declaration under the top-level `conform` table, by full
    /// dotted path, sorted. Each is a control the contract does not define.
    pub unrecognized: Vec<String>,
    /// Paths of `conform` tables that are **not** the top-level control surface
    /// (`package.conform`, which is what a dotted `conform.exclude = …` written
    /// after a table header actually declares). These control nothing at all, so
    /// reporting them is the whole point: a shell that wrote one meant the knob.
    pub misplaced: Vec<String>,
}

impl ConformDecl {
    /// Whether the shell declared anything the contract does not define.
    pub fn is_clean(&self) -> bool {
        self.unrecognized.is_empty() && self.misplaced.is_empty()
    }

    /// Every finding, rendered for one diagnostic line.
    pub fn findings(&self) -> Vec<String> {
        let mut out = self.unrecognized.clone();
        out.extend(
            self.misplaced.iter().map(|p| {
                format!("{p} (not the top-level `conform` table, so it controls nothing)")
            }),
        );
        out
    }
}

/// Parse the `conform` control surface out of a `reef.toml` text.
///
/// `Err` carries the TOML parse error. A manifest that does not parse cannot be
/// audited against §8 at all, and the caller must fail loudly rather than treat
/// an unreadable file as an empty declaration.
pub fn parse(reef_toml: &str) -> Result<ConformDecl, String> {
    let doc: toml::Value = toml::from_str(reef_toml).map_err(|e| e.to_string())?;
    let mut decl = ConformDecl::default();
    let mut unrecognized = BTreeSet::new();

    if let Some(value) = doc.get(CONFORM) {
        collect(value, CONFORM, true, &mut decl, &mut unrecognized);
    }
    // A `conform` table anywhere BELOW the top level is not the control surface.
    // It is what `conform.exclude = […]` becomes when written after a table
    // header, so it is the most likely way to intend the knob and miss it.
    if let Some(table) = doc.as_table() {
        for (key, value) in table {
            if key == CONFORM {
                continue;
            }
            find_misplaced(value, key, &mut decl);
        }
    }

    decl.unrecognized = unrecognized.into_iter().collect();
    decl.misplaced.sort();
    decl.misplaced.dedup();
    Ok(decl)
}

/// Walk the `conform` table, recording the recognized key and every other
/// declaration by full dotted path. `control` marks the top-level `conform`
/// table itself, the only place `local_skills` means anything.
fn collect(
    value: &toml::Value,
    path: &str,
    control: bool,
    decl: &mut ConformDecl,
    unrecognized: &mut BTreeSet<String>,
) {
    let Some(table) = value.as_table() else {
        // `[[conform]]`, `conform = 3`, `conform = "x"`: the control surface is
        // a table, and anything else is a declaration in the wrong shape.
        unrecognized.insert(format!("{path} (expected a table)"));
        return;
    };
    if table.is_empty() {
        // An empty `[conform]` declares nothing, which is fine. An empty
        // `[conform.skills]` still declares `skills`, which is not.
        if !control {
            unrecognized.insert(path.to_string());
        }
        return;
    }
    for (key, child) in table {
        let child_path = format!("{path}.{key}");
        if control && key == LOCAL_SKILLS {
            match string_array(child) {
                Some(names) => decl.local_skills = names,
                // The recognized key is recognized by its VALUE TYPE, not by a
                // spelling: `[conform.local_skills]` is a table where the
                // contract defines an array of strings, so it declares a shape
                // §8 does not define and is reported like any other.
                None => {
                    unrecognized.insert(format!("{child_path} (expected an array of strings)"));
                }
            }
            continue;
        }
        if child.is_table() {
            collect(child, &child_path, false, decl, unrecognized);
        } else {
            unrecognized.insert(child_path);
        }
    }
}

/// Record every `conform` table that is not the top-level one.
fn find_misplaced(value: &toml::Value, path: &str, decl: &mut ConformDecl) {
    match value {
        toml::Value::Table(table) => {
            for (key, child) in table {
                let child_path = format!("{path}.{key}");
                if key == CONFORM {
                    decl.misplaced.push(child_path);
                    continue;
                }
                find_misplaced(child, &child_path, decl);
            }
        }
        toml::Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                find_misplaced(item, &format!("{path}[{i}]"), decl);
            }
        }
        _ => {}
    }
}

/// `value` as a list of strings, or `None` if it is not an array of strings.
fn string_array(value: &toml::Value) -> Option<Vec<String>> {
    value
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decl(toml: &str) -> ConformDecl {
        parse(toml).expect("valid toml")
    }

    fn findings(toml: &str) -> Vec<String> {
        decl(toml).findings()
    }

    #[test]
    fn the_recognized_declaration_is_honored() {
        assert_eq!(
            decl("[conform]\nlocal_skills = [\"chelis-std\"]\n").local_skills,
            vec!["chelis-std".to_string()]
        );
        assert!(decl("[conform]\nlocal_skills = [\"a\"]\n").is_clean());
        // Absent entirely.
        assert!(decl("[package]\nname = \"s\"\n").local_skills.is_empty());
        assert!(decl("[package]\nname = \"s\"\n").is_clean());
        // Present but empty.
        assert!(decl("[conform]\nlocal_skills = []\n").is_clean());
        assert!(decl("[conform]\n").is_clean());
    }

    /// The property the structural parse buys: the SPELLING cannot change the
    /// answer, because there are no spellings after parsing. Every row here is
    /// the same declaration written a different way.
    #[test]
    fn every_spelling_of_the_recognized_declaration_is_honored() {
        for text in [
            "[conform]\nlocal_skills = [\"x\"]\n",
            "[conform]\n\"local_skills\" = [\"x\"]\n",
            "[conform]\n'local_skills' = [\"x\"]\n",
            "conform = { local_skills = [\"x\"] }\n",
            "conform.local_skills = [\"x\"]\n",
            "[conform]\nlocal_skills = [\n  \"x\",\n]\n",
            "[ conform ]\nlocal_skills = [\"x\"]\n",
            "[\"conform\"]\nlocal_skills = [\"x\"]\n",
        ] {
            let d = decl(text);
            assert_eq!(d.local_skills, vec!["x".to_string()], "{text:?}");
            assert!(d.is_clean(), "{text:?} -> {:?}", d.findings());
        }
    }

    /// The same property in the other direction: every spelling of the control
    /// §8 denies is reported, including the six that got past a line scan.
    #[test]
    fn every_spelling_of_an_unrecognized_declaration_is_reported() {
        let cases: &[(&str, &str)] = &[
            ("[conform]\nexclude = [\"a\"]\n", "conform.exclude"),
            (
                "[conform.skills]\nexclude = [\"a\"]\n",
                "conform.skills.exclude",
            ),
            (
                "[conform]\nskills.exclude = [\"a\"]\n",
                "conform.skills.exclude",
            ),
            ("[conform]\n\"exclude\" = [\"a\"]\n", "conform.exclude"),
            ("[conform]\n'exclude' = [\"a\"]\n", "conform.exclude"),
            ("conform = { exclude = [\"a\"] }\n", "conform.exclude"),
            ("conform.exclude = [\"a\"]\n", "conform.exclude"),
            (
                "conform.skills.exclude = [\"a\"]\n",
                "conform.skills.exclude",
            ),
            ("[[conform.x]]\nexclude = [\"a\"]\n", "conform.x"),
            (
                "[conform]\nskills = { exclude = [\"a\"] }\n",
                "conform.skills.exclude",
            ),
            // A non-ASCII key is only expressible quoted (TOML bare keys are
            // ASCII), and it survives to the report as written.
            ("[conform]\n\"exclüde\" = [\"a\"]\n", "conform.exclüde"),
            ("[conform]\nlocal_skill = []\n", "conform.local_skill"),
            // A bracket inside a string value used to desync a depth counter and
            // hide every later key. Parsed, it is just a string.
            (
                "[conform]\nlocal_skills = [\"a[\"]\nexclude = [\"b\"]\n",
                "conform.exclude",
            ),
            (
                "[conform]\nlocal_skills = [\"a#b\"]\nexclude = [\"b\"]\n",
                "conform.exclude",
            ),
        ];
        for (text, expected) in cases {
            let f = findings(text);
            assert!(
                f.iter().any(|x| x.starts_with(expected)),
                "{text:?} must report {expected:?}, got {f:?}"
            );
        }
    }

    /// The recognized key is recognized by its VALUE TYPE. A table there is a
    /// shape the contract does not define, which is what closes the last hole a
    /// name-only check leaves: `[conform.local_skills]` laundering its body.
    #[test]
    fn local_skills_must_be_an_array_of_strings() {
        for text in [
            "[conform.local_skills]\nexclude = [\"a\"]\n",
            "[conform]\nlocal_skills = \"chelis-std\"\n",
            "[conform]\nlocal_skills = 3\n",
            "[conform]\nlocal_skills = [\"a\", 3]\n",
            "[conform]\nlocal_skills = { a = 1 }\n",
        ] {
            let d = decl(text);
            assert!(d.local_skills.is_empty(), "{text:?}");
            assert!(
                d.findings()
                    .iter()
                    .any(|f| f.contains("conform.local_skills")),
                "{text:?} -> {:?}",
                d.findings()
            );
        }
    }

    /// A `conform` table under another table controls nothing. It is exactly
    /// what a dotted `conform.exclude` becomes when written after a header, so
    /// it is reported rather than silently doing nothing.
    #[test]
    fn a_conform_table_below_the_top_level_is_reported_as_misplaced() {
        let d = decl("[package]\nname = \"s\"\nconform.exclude = [\"a\"]\n");
        assert_eq!(d.misplaced, vec!["package.conform".to_string()]);
        assert!(!d.is_clean());
        assert!(
            d.findings()[0].contains("controls nothing"),
            "{:?}",
            d.findings()
        );
        // And it does NOT become a local_skills declaration.
        let d = decl("[package]\nname = \"s\"\nconform.local_skills = [\"x\"]\n");
        assert!(d.local_skills.is_empty());
        assert!(!d.is_clean());
    }

    #[test]
    fn a_table_merely_named_like_conform_is_out_of_scope() {
        // Negative parity for the scope rule: widening to `conform*` would fail
        // shells for tables that are not this control surface.
        for text in [
            "[conformx]\nexclude = [\"a\"]\n",
            "[package]\nname = \"conform\"\n",
            "[dependencies]\nconformance = \"1\"\n",
        ] {
            assert!(decl(text).is_clean(), "{text:?} -> {:?}", findings(text));
        }
    }

    #[test]
    fn an_unparseable_manifest_is_an_error_not_an_empty_declaration() {
        // The failure mode this replaces: an unreadable file read as "declares
        // nothing" is a silent pass on a MUST row.
        let err = parse("this is not toml at all\n").unwrap_err();
        assert!(!err.is_empty());
        assert!(parse("[conform]\nlocal_skills = [\n").is_err());
        // An unquoted non-ASCII key is not valid TOML at all. It becomes a loud
        // parse failure rather than a key this tool quietly cannot spell, which
        // is the whole difference from the character-allowlist approach.
        assert!(parse("[conform]\nexclüde = [\"a\"]\n").is_err());
    }

    /// The two `[conform]` tables that actually exist in the ecosystem today,
    /// verbatim (School's and hydronnx's, comments included). A parser change
    /// that reported either of these would fail two conformant shells on a MUST
    /// row, so they are locked as the real-world positive control.
    #[test]
    fn the_conform_tables_shells_actually_ship_are_clean() {
        let school = "[package]\nname = \"school\"\ncompiler = \"=0.17.1\"\n\n\
             [dependencies]\nchelis-std = { version = \"0.4.0\" }\n\n\
             [conform]\n\
             # `chelis reef conform` §8 repo-local domain-skill allowlist (chelis#651,\n\
             # shipped 0.15.2). `chelis-std` is School's downstream-Chelis-authoring domain\n\
             # skill (mirrored from the monorepo), not part of the toolchain's embedded\n\
             # shared set - declared here so `conform sync` preserves it and `conform audit`\n\
             # §8 exempts it, instead of pruning it.\n\
             local_skills = [\"chelis-std\"]\n";
        let d = decl(school);
        assert_eq!(d.local_skills, vec!["chelis-std".to_string()]);
        assert!(d.is_clean(), "{:?}", d.findings());

        let hydronnx = "[package]\nname = \"hydronnx\"\ncompiler = \"=0.18.1\"\n\n\
             [chelis-src]\ncrates = [\"chelis-ir\", \"chelis-types\"]\n\
             pin_commit = \"c8db387d06d538ce8039ac37645a43def48373c9\"\n\n\
             # chelis-std is a hydronnx-local downstream-authoring skill, not one of the\n\
             # shared upstream skills.\n\
             [conform]\nlocal_skills = [\"chelis-std\"]\n";
        let d = decl(hydronnx);
        assert_eq!(d.local_skills, vec!["chelis-std".to_string()]);
        assert!(d.is_clean(), "{:?}", d.findings());
    }

    /// A shell with no `[conform]` table at all is the common case (nine of the
    /// eleven registry shells), and reef.toml carries tables this module must
    /// walk past without comment.
    #[test]
    fn a_manifest_with_no_conform_table_is_clean() {
        let text = "[package]\nname = \"shoals\"\ncompiler = \"=0.18.5\"\n\
             module_prefix = \"Shoals\"\n\
             additional_sources = [\"properties\", \"references\", \"demos\"]\n\n\
             [dependencies]\nchelis-std = { version = \"0.4.0\" }\n\
             nautilus = { version = \"0.7.42\" }\n\n\
             [artifacts.octant]\nversion = \"0.13.0\"\n";
        let d = decl(text);
        assert!(d.local_skills.is_empty());
        assert!(d.is_clean(), "{:?}", d.findings());
    }

    #[test]
    fn findings_are_deduplicated_and_ordered() {
        let d = decl("[conform]\nb = 1\na = 2\n");
        assert_eq!(
            d.unrecognized,
            vec!["conform.a".to_string(), "conform.b".to_string()]
        );
    }
}
