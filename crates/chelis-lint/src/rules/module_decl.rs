//! Shared `.ch` module-declaration extractor used by the §6.2 and §6.3 rules.
//!
//! Extracts every `^module Foo.Bar.Baz` line from a Surf source file, with
//! the line number and the component list. Both `module-compound-titlecase`
//! and `module-pascal-components` consume this.

use regex::Regex;
use std::sync::OnceLock;

static MODULE_RE: OnceLock<Regex> = OnceLock::new();

fn module_re() -> &'static Regex {
    // `module Foo.Bar.Baz` at the start of a line. Components are PascalCase
    // per §1.1; the regex permits lowercase too so the lint rules can flag
    // them, rather than the regex silently swallowing violators.
    MODULE_RE
        .get_or_init(|| Regex::new(r"(?m)^[ \t]*module[ \t]+([A-Za-z_][A-Za-z0-9_.]*)").unwrap())
}

/// One module declaration extracted from Surf source.
pub struct ModuleDecl {
    /// 1-indexed line number of the `module` line.
    pub line: usize,
    /// Components, e.g. `["Nautilus", "LinAlg"]` for `module Nautilus.LinAlg`.
    pub components: Vec<String>,
}

/// Find every `module Foo.Bar` declaration in `source`.
pub fn find_module_decls(source: &str) -> Vec<ModuleDecl> {
    let mut out = Vec::new();
    for caps in module_re().captures_iter(source) {
        let path = caps.get(1).unwrap().as_str();
        let match_start = caps.get(0).unwrap().start();
        let line = source[..match_start]
            .bytes()
            .filter(|&b| b == b'\n')
            .count()
            + 1;
        let components = path.split('.').map(|s| s.to_string()).collect();
        out.push(ModuleDecl { line, components });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_single_component_module() {
        let decls = find_module_decls("module Nautilus\n");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].components, vec!["Nautilus".to_string()]);
        assert_eq!(decls[0].line, 1);
    }

    #[test]
    fn finds_multi_component_module() {
        let decls = find_module_decls("module Coral.Internal.Hamt\n");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].components, vec!["Coral", "Internal", "Hamt"]);
    }

    #[test]
    fn reports_correct_line_number() {
        let src = "// header\n// header\nmodule Std.Tests.Foo\n";
        let decls = find_module_decls(src);
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].line, 3);
    }

    #[test]
    fn ignores_non_module_keyword_lines() {
        let src = "def x = 1\nimport Foo.Bar\nmodule Real.One\n";
        let decls = find_module_decls(src);
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].components, vec!["Real", "One"]);
    }

    #[test]
    fn permits_lowercase_components_so_rules_can_flag() {
        // The extractor is permissive: it returns `Hellotensor` as a
        // component so the §6.3 rule can flag it. A strict-PascalCase
        // regex would have hidden the violation.
        let decls = find_module_decls("module Hellotensor\n");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].components, vec!["Hellotensor"]);
    }
}
