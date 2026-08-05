//! Rule `module-compound-titlecase` — module name components use Title-case
//! for compounds, never ALL-CAPS abbreviations (§6.2). `Hamt` not `HAMT`,
//! `Io` not `IO`, `Json` not `JSON`, `Lstm` not `LSTM`.

use super::module_decl::find_module_decls;
use crate::{Context, Rule, Surface, Violation};

pub struct ModuleCompoundTitlecase;

impl Rule for ModuleCompoundTitlecase {
    fn id(&self) -> &str {
        "module-compound-titlecase"
    }

    fn spec_ref(&self) -> &str {
        "§6.2"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "module compound names use Title-case (Hamt), never ALL-CAPS abbreviations (HAMT)"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for decl in find_module_decls(source) {
            for component in &decl.components {
                if component == "CEarchin" {
                    continue;
                }
                if has_all_caps_run(component) {
                    out.push(Violation {
                        rule_id: self.id().to_string(),
                        spec_ref: self.spec_ref().to_string(),
                        path: ctx.path.to_path_buf(),
                        line: Some(decl.line),
                        col: None,
                        message: format!(
                            "module component `{component}` uses ALL-CAPS abbreviation; rewrite as Title-case per §6.2 (e.g., `{}`)",
                            suggest_titlecase(component)
                        ),
                    });
                }
            }
        }
        out
    }
}

/// True if the component contains 2 or more consecutive uppercase letters.
/// Single capitals (`Hamt`, `Io`) are fine; runs of 2+ (`HAMT`, `IO`,
/// `KLDiv`, `JSON`) are the violation.
fn has_all_caps_run(s: &str) -> bool {
    let mut run = 0;
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            run += 1;
            if run >= 2 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Convert an ALL-CAPS-abbreviation component to its Title-case form.
/// `HAMT` → `Hamt`; `KLDiv` → `KlDiv`; `IO` → `Io`; `JSON` → `Json`.
fn suggest_titlecase(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_uppercase() {
            // Start of a run of uppercase. Emit the first uppercase, then
            // lowercase the rest of the run.
            out.push(c);
            i += 1;
            while i < chars.len() && chars[i].is_ascii_uppercase() {
                // If this is the LAST uppercase in the run AND there are
                // lowercase letters following, keep this one uppercase
                // (it's the start of the next Title-case word, e.g.
                // `KLDiv` → `KlDiv`, the last `D` precedes `iv`).
                let next_is_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();
                if next_is_lower {
                    out.push(chars[i]);
                } else {
                    out.push(chars[i].to_ascii_lowercase());
                }
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_all_caps_hamt_module() {
        assert!(has_all_caps_run("HAMT"));
    }

    #[test]
    fn flags_all_caps_io_module() {
        assert!(has_all_caps_run("IO"));
    }

    #[test]
    fn flags_kldiv_module() {
        assert!(has_all_caps_run("KLDiv"));
    }

    #[test]
    fn accepts_single_cap_pascal() {
        assert!(!has_all_caps_run("Hamt"));
        assert!(!has_all_caps_run("Io"));
        assert!(!has_all_caps_run("KlDiv"));
        assert!(!has_all_caps_run("Json"));
    }

    #[test]
    fn accepts_internal_cap_compound() {
        // OrderBook, CurveFit, GroupBy: each cap starts a new word, no run.
        assert!(!has_all_caps_run("OrderBook"));
        assert!(!has_all_caps_run("CurveFit"));
        assert!(!has_all_caps_run("GroupBy"));
    }

    #[test]
    fn end_to_end_accepts_c_earchin_prefix() {
        use std::path::Path;
        let src = "module CEarchin.Vocabulary\n\ndef x() = 1\n";
        let path = Path::new("c-earchin/src/vocabulary.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let v = ModuleCompoundTitlecase.check(&ctx);
        assert!(v.is_empty(), "CEarchin is the c-earchin module prefix");
    }

    #[test]
    fn suggest_hamt_to_title() {
        assert_eq!(suggest_titlecase("HAMT"), "Hamt");
    }

    #[test]
    fn suggest_io_to_title() {
        assert_eq!(suggest_titlecase("IO"), "Io");
    }

    #[test]
    fn suggest_kldiv_keeps_word_break() {
        // The `D` in `KLDiv` precedes lowercase `iv`; it's the start of the
        // next Title-case word and stays uppercase.
        assert_eq!(suggest_titlecase("KLDiv"), "KlDiv");
    }

    #[test]
    fn suggest_json_to_title() {
        assert_eq!(suggest_titlecase("JSON"), "Json");
    }

    #[test]
    fn end_to_end_flags_coral_internal_hamt() {
        use std::path::Path;
        let src = "module Coral.Internal.HAMT\n\ndef x() = 1\n";
        let path = Path::new("coral/src/internal/hamt.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let v = ModuleCompoundTitlecase.check(&ctx);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("HAMT"));
        assert!(v[0].message.contains("Hamt"));
        assert_eq!(v[0].line, Some(1));
    }

    #[test]
    fn end_to_end_passes_titlecase_module() {
        use std::path::Path;
        let src = "module Coral.Internal.Hamt\n\ndef x() = 1\n";
        let path = Path::new("coral/src/internal/hamt.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let v = ModuleCompoundTitlecase.check(&ctx);
        assert!(v.is_empty());
    }
}
