//! Rule `module-pascal-components` — every component of a module ladder is
//! PascalCase, including each word inside a compound (§6.3). Catches
//! lowercase-after-first compound-looking components like `Examplerootfind`
//! (should be `ExampleRootFind`) and `Hellotensor` (should be `HelloTensor`).
//!
//! This rule is advisory-only. Its "long lowercase run, no internal capital
//! is a suspected compound" heuristic false-positives on real single English
//! words, so it is wired through `registry::non_blocking_rules` and does not
//! fail `chelis lint --check` or the style-gated build/check/eval/validate
//! paths. See
//! `spec/upstream-bugs/module-pascal-components-flags-single-words.md`.

use super::module_decl::find_module_decls;
use crate::{Context, Rule, Severity, Surface, Violation};

pub struct ModulePascalComponents;

impl Rule for ModulePascalComponents {
    fn id(&self) -> &str {
        "module-pascal-components"
    }

    fn spec_ref(&self) -> &str {
        "§6.3"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "module ladder components use PascalCase per word; long lowercase runs without internal caps are suspicious"
    }

    fn severity(&self) -> Severity {
        // Advisory, not blocking. The long-lowercase-run heuristic
        // false-positives on real single English words, so it reports but
        // must not fail `chelis lint --check`. See
        // `spec/upstream-bugs/module-pascal-components-flags-single-words.md`.
        Severity::Advisory
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for decl in find_module_decls(source) {
            for component in &decl.components {
                if let Some(reason) = component_violation(component) {
                    out.push(Violation {
                        rule_id: self.id().to_string(),
                        spec_ref: self.spec_ref().to_string(),
                        path: ctx.path.to_path_buf(),
                        line: Some(decl.line),
                        col: None,
                        message: format!(
                            "module component `{component}` {reason}. Rewrite per-word PascalCase per §6.3, or add to the recognized-single-word allowlist with a §6.3 cross-ref",
                        ),
                    });
                }
            }
        }
        out
    }
}

/// Recognized single English words used as module components in the Chelis
/// ecosystem. Components on this list pass the rule even when they have a
/// long lowercase run. Adding to this list is a documentation act: each
/// entry is asserted to be a single English word that meets §6.3 by being
/// a single PascalCase word, not a hidden compound.
///
/// Seed list comes from a survey of `.ch` files across `chelis`,
/// `nautilus`, `coral`, `shoals`, `octant` (May 2026 ecosystem snapshot).
const KNOWN_SINGLE_WORDS: &[&str] = &[
    // Top-level repo prefixes. Each name is asserted to be a single
    // English PascalCase word under spec §6.1/§6.3. See
    // `docs/investigations/module_pascal_allowlist_diagnosis.md` for the
    // canonical chelis-ecosystem name set and cross-references.
    "Chelis",
    "CEarchin",
    "Nautilus",
    "Coral",
    "Shoals",
    "Octant",
    "Capstone",
    "Hydronnx",
    "Std",
    // Nautilus subsystems
    "Distance",
    "Distributions",
    "Integrate",
    "Interpolation",
    "Optim",
    "Roots",
    "Signal",
    "Special",
    "Stats",
    "Testing",
    // Coral subsystems
    "Frame",
    "Reshape",
    "Window",
    "Internal",
    // Shoals subsystems
    "Pricing",
    "Risk",
    "Curves",
    "Stochastic",
    "Orderbook",
    // Std subsystems
    "Tests",
    "Tensor",
    "Test",
    "Init",
    "Time",
    "Tokenizer",
    "Decimal",
    "Date",
    "Duration",
    "Json",
    "Loss",
    "Math",
    // Octant subsystems
    "Version",
    "Vocabulary",
    // Generic short words
    "Core",
    "Demo",
    "Repl",
    "Cli",
    "Api",
    "Io",
    "Hamt",
    // Test-class module names
    "Join",
    // Recognized single English words at length >= 7 used in identifiers
    // and module names across the ecosystem. Each is a single English word;
    // adding to this list is a documented assertion under §6.3, not a
    // workaround.
    "Activation",
    "Attention",
    "Embedding",
    "Generate",
    "Schedule",
    "Construct",
    "Convolve",
    "Logits",
    "Softmax",
    "Sampling",
    "Compose",
    "Capture",
    "Encoder",
    "Decoder",
    "Process",
    "Convert",
    "Transform",
    "Compute",
    "Predict",
    "Iterate",
    "Visitor",
    "Iterator",
    "Observer",
    "Provider",
    "Consumer",
    "Producer",
    "Builder",
    "Factory",
    "Adapter",
    "Wrapper",
    "Element",
    "Network",
    "Service",
    "Manager",
    "Handler",
    "Controller",
    "Operator",
    "Function",
    "Instance",
    // Reef package additional_sources directory names used as module roots.
    "Properties",
    "References",
    // External library names that the ecosystem mirrors as module
    // components. These are single-word per the upstream library's own
    // canonical naming; treating them as compounds (Safetensors →
    // SafeTensors) would diverge from the upstream's PyPI/HuggingFace
    // identity.
    "Safetensors",
    // Hello-chelis (2026-05) corpus survey: single English words used
    // as module components that today trip the 7-char long-lowercase-run
    // heuristic even though each is canonically a single PascalCase word
    // per §6.3. Asserting these here closes the surface gap pending the
    // Vocabulary-F2 structural follow-up that replaces hand-maintenance
    // with derivation from a canonical vocabulary list.
    "Linearity",
    "Hypothesis",
    "Integration",
    "Optimize",
    // Math/ML single-word domains used as module components across the
    // ecosystem. Each is canonically one English word per §6.3.
    "Matrix",
    "Vector",
    "Random",
    "Statistics",
    "Probability",
    "Geometry",
    "Calculus",
    "Algebra",
    "Topology",
    "Spectrum",
    "Inference",
    "Classification",
    "Regression",
    "Clustering",
    // Calcify shell. `Calcify` is the shell's module prefix; the remaining
    // entries are real single English words that appear as translated-module
    // components when Calcify emits Chelis from single-word Python filenames
    // (e.g. `comprehension.py` -> `Calcify.Comprehension`). Each is one
    // PascalCase English word per §6.3, not a hidden compound; the
    // long-lowercase-run heuristic flags them only because they are long.
    "Calcify",
    "Translation",
    "Comprehension",
    "Conditional",
    "Exception",
];

/// Known PascalCase compound module/type names in the Chelis ecosystem.
/// A component that matches the *lowercase-after-first* form of one of
/// these (e.g., `Linalg` for `LinAlg`, `Groupby` for `GroupBy`) is a
/// definite §6.3 violation — the upstream is using the compound form,
/// the local copy isn't. This catches the sibling-sweep mutation case
/// that the long-lowercase-run heuristic misses.
const KNOWN_PASCAL_COMPOUNDS: &[&str] = &[
    "LinAlg",
    "GroupBy",
    "RmsNorm",
    "CurveFit",
    "OrderBook",
    "CrossEntropy",
    "ApiSmoke",
    "ExampleRootFind",
    "ExampleOdeDemo",
    "ExampleDistributions",
    "ExampleIntegration",
    "ExampleOptim",
    "HelloTensor",
    "KeyValue",
    "ColumnType",
    "GroupedFrame",
    "AggSum",
    "AggMean",
    "AggMax",
    "AggMin",
    "AggCount",
    "RoundUp",
    "RoundDown",
    "RoundHalfEven",
    "RoundHalfUp",
];

/// If `component` is the lowercase-after-first form of a known PascalCase
/// compound, return the canonical compound. Otherwise `None`.
///
/// `LinAlg` → first-cap-form is `Linalg`. So `is_compound_lowercase_form("Linalg")`
/// returns `Some("LinAlg")`.
fn known_compound_lowercase_form(component: &str) -> Option<&'static str> {
    for compound in KNOWN_PASCAL_COMPOUNDS {
        let first_cap_form: String = compound
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c } else { c.to_ascii_lowercase() })
            .collect();
        if component == first_cap_form.as_str() && component != *compound {
            return Some(compound);
        }
    }
    None
}

/// Return `Some(reason)` if the component is a §6.3 violation, `None` if
/// it's clean. Reasons are intended to be human-readable explanations of
/// what the rule flagged.
fn component_violation(s: &str) -> Option<&'static str> {
    // Empty or non-letter-leading components don't reach here; the module
    // declaration regex rejects them.
    if !s.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return Some("does not start with an uppercase letter");
    }
    // Allowlist short-circuits before the compound-lowercase-form check.
    // Some allowlisted single-word names (e.g., `Orderbook`, accepted as
    // the canonical Shoals.Orderbook module name) collide with the
    // lowercase-form of a known compound (`OrderBook` is in
    // KNOWN_PASCAL_COMPOUNDS). The single-word allowlist wins.
    if KNOWN_SINGLE_WORDS.contains(&s) {
        return None;
    }
    // Targeted check for the sibling-sweep mutation case: known PascalCase
    // compounds with their internal capital flattened (e.g., `Linalg` for
    // `LinAlg`). Catches violations the long-lowercase-run heuristic
    // misses for shorter compounds.
    if known_compound_lowercase_form(s).is_some() {
        return Some("matches the lowercase-after-first form of a known PascalCase compound");
    }
    // Heuristic: a component with no internal capital and a long lowercase
    // run (>= 7 lowercase letters in a row) is suspicious. A real single
    // English word at this length should be in the allowlist; if it isn't,
    // the rule fires and the orchestrator either fixes the compound or
    // adds the word to KNOWN_SINGLE_WORDS.
    let has_internal_cap = s.chars().skip(1).any(|c| c.is_ascii_uppercase());
    if has_internal_cap {
        return None;
    }
    let max_lowercase_run = s
        .chars()
        .skip(1)
        .scan(0usize, |run, c| {
            if c.is_ascii_lowercase() {
                *run += 1;
            } else {
                *run = 0;
            }
            Some(*run)
        })
        .max()
        .unwrap_or(0);
    if max_lowercase_run >= 7 {
        Some("looks like a multi-word compound but has no internal capital")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn flags_examplerootfind() {
        assert!(component_violation("Examplerootfind").is_some());
    }

    #[test]
    fn flags_exampleodedemo() {
        assert!(component_violation("Exampleodedemo").is_some());
    }

    #[test]
    fn flags_exampledistributions() {
        assert!(component_violation("Exampledistributions").is_some());
    }

    #[test]
    fn flags_apismoke() {
        assert!(component_violation("Apismoke").is_some());
    }

    #[test]
    fn flags_hellotensor() {
        assert!(component_violation("Hellotensor").is_some());
    }

    #[test]
    fn flags_known_compound_lowercase_form_linalg() {
        // §6.3 sibling-sweep: `Linalg` is the flattened form of `LinAlg`.
        // The 7-char threshold misses it (5 lowercase only) but the
        // known-compound check catches it.
        assert!(component_violation("Linalg").is_some());
    }

    #[test]
    fn flags_known_compound_lowercase_form_groupby() {
        assert!(component_violation("Groupby").is_some());
    }

    #[test]
    fn flags_known_compound_lowercase_form_rmsnorm() {
        assert!(component_violation("Rmsnorm").is_some());
    }

    #[test]
    fn flags_known_compound_lowercase_form_curvefit() {
        assert!(component_violation("Curvefit").is_some());
    }

    #[test]
    fn accepts_canonical_compound_form() {
        assert_eq!(component_violation("LinAlg"), None);
        assert_eq!(component_violation("GroupBy"), None);
        assert_eq!(component_violation("RmsNorm"), None);
    }

    #[test]
    fn accepts_correct_pascal_compound() {
        // Internal caps signal proper word breaks → not flagged.
        assert_eq!(component_violation("ExampleRootFind"), None);
        assert_eq!(component_violation("ExampleOdeDemo"), None);
        assert_eq!(component_violation("ApiSmoke"), None);
        assert_eq!(component_violation("HelloTensor"), None);
        assert_eq!(component_violation("CurveFit"), None);
        assert_eq!(component_violation("GroupBy"), None);
    }

    #[test]
    fn accepts_known_single_words() {
        // Long lowercase runs are fine if the word is recognized.
        assert_eq!(component_violation("Distance"), None);
        assert_eq!(component_violation("Distributions"), None);
        assert_eq!(component_violation("Activation"), None);
        assert_eq!(component_violation("Integrate"), None);
        assert_eq!(component_violation("Interpolation"), None);
        assert_eq!(component_violation("CEarchin"), None);
        assert_eq!(component_violation("Vocabulary"), None);
    }

    #[test]
    fn accepts_short_components_without_allowlist() {
        // Shorter than the threshold; allowlist not required.
        assert_eq!(component_violation("Risk"), None);
        assert_eq!(component_violation("Time"), None);
        // 6 lowercase after the leading cap: Pricing → "ricing" = 6 chars.
        assert_eq!(component_violation("Pricing"), None);
    }

    #[test]
    fn rejects_lowercase_leading() {
        assert!(component_violation("nautilus").is_some());
    }

    #[test]
    fn end_to_end_flags_nautilus_examplerootfind() {
        let src = "module Nautilus.Examplerootfind\n";
        let path = Path::new("nautilus/src/examplerootfind.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let v = ModulePascalComponents.check(&ctx);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("Examplerootfind"));
        assert_eq!(v[0].line, Some(1));
    }

    #[test]
    fn end_to_end_passes_corrected_form() {
        let src = "module Nautilus.ExampleRootFind\n";
        let path = Path::new("nautilus/src/examplerootfind.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let v = ModulePascalComponents.check(&ctx);
        assert!(v.is_empty());
    }

    #[test]
    fn end_to_end_flags_each_offending_component_per_decl() {
        // A pathological declaration with two violating components on one line.
        let src = "module Hellotensor.Apismoke\n";
        let path = Path::new("examples/x.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let v = ModulePascalComponents.check(&ctx);
        assert_eq!(v.len(), 2);
    }

    // §6.3 sibling-sweep: the `KNOWN_SINGLE_WORDS` allowlist is hand-maintained
    // and drifts behind the documented ecosystem name set. The fixtures below
    // pin the gap so subsequent commits in this branch (diagnosis + fix)
    // close it without regression. Each name is asserted to be a single
    // PascalCase ecosystem name per spec §6.1/§6.3 and the canonical-reference
    // shell roster (`spec/design/chelis_canonical_reference.md` §"Shell
    // Ecosystem"). Currently fails because long lowercase runs (>=7 chars
    // after the leading capital) trip the suspicion heuristic at L271 even
    // though the names are single-word and correctly PascalCase per spec.
    #[test]
    fn accepts_capstone_ecosystem_name() {
        // User-reported failure: `Capstone` is 1 leading cap + 7 lowercase,
        // hits the >=7 long-run threshold even though it's a single English
        // word being used as an ecosystem module-prefix name.
        assert_eq!(component_violation("Capstone"), None);
    }

    // Extended-scope §6.3 allowlist gaps surfaced by the hello-chelis
    // corpus survey (2026-05). Each is a single English PascalCase word
    // used as a module-component name; today they trip the 7-char
    // long-lowercase-run heuristic at L275 even though they're correctly
    // formed per §6.3. The allowlist must include them.
    #[test]
    fn accepts_linearity_hypothesis_integration_optimize() {
        assert_eq!(component_violation("Linearity"), None);
        assert_eq!(component_violation("Hypothesis"), None);
        assert_eq!(component_violation("Integration"), None);
        assert_eq!(component_violation("Optimize"), None);
    }

    // §6.3 sibling sweep — additional single English words used as
    // module-components across mathematical, statistical, and ML domains.
    // Each is a canonical single-word PascalCase identifier under §6.3.
    #[test]
    fn accepts_extended_single_word_math_domains() {
        for name in [
            "Tensor",
            "Matrix",
            "Vector",
            "Random",
            "Statistics",
            "Probability",
            "Geometry",
            "Calculus",
            "Algebra",
            "Topology",
            "Spectrum",
            "Inference",
            "Classification",
            "Regression",
            "Clustering",
        ] {
            assert_eq!(
                component_violation(name),
                None,
                "extended single-word `{name}` should pass §6.3",
            );
        }
    }

    // §6.3 Calcify-shell allowlist gap. The Calcify translator emits module
    // ladders of the form `Calcify.<Stem>` where `<Stem>` is the
    // PascalCased Python filename. A single-word filename produces a
    // single-word component (`comprehension.py` -> `Calcify.Comprehension`);
    // each name below is a real single English word that trips the
    // long-lowercase-run heuristic only because it is long.
    #[test]
    fn accepts_calcify_shell_single_words() {
        for name in [
            "Calcify",
            "Translation",
            "Comprehension",
            "Conditional",
            "Exception",
        ] {
            assert_eq!(
                component_violation(name),
                None,
                "Calcify-shell single-word `{name}` should pass §6.3",
            );
        }
    }

    #[test]
    fn module_pascal_components_is_advisory() {
        // The rule reports but must not block `chelis lint --check`. Its
        // heuristic false-positives on real single English words, so it is
        // wired through `registry::non_blocking_rules`.
        assert_eq!(ModulePascalComponents.severity(), Severity::Advisory);
        assert!(!ModulePascalComponents.severity().blocks_check());
    }

    #[test]
    fn negative_long_lowercase_word_without_allowlist_still_fires() {
        // Sanity: a long-lowercase-run identifier not in the allowlist
        // still fires. Use a fictional run that won't be confused with
        // a real ecosystem name.
        assert!(
            component_violation("Bogusnonsense").is_some(),
            "uncurated long-lowercase-run name should still flag",
        );
    }

    #[test]
    fn sibling_sweep_canonical_ecosystem_names_pass() {
        // Sibling sweep: every chelis-ecosystem PascalCase top-level name
        // from `spec/design/chelis_canonical_reference.md` §"Shell Ecosystem"
        // (active shells + post-Phase-3 stubs) plus the canonical
        // chelis-runtime prefix should pass the rule. Today, the entries
        // listed below all pass through the short-lowercase-run path (each
        // has fewer than 7 lowercase chars after the leading cap), so the
        // canary stays green. This test exists to lock the invariant: if
        // any future ecosystem-name addition trips the 7-char threshold,
        // it must be added to `KNOWN_SINGLE_WORDS` rather than tightening
        // the rule's detection logic. The `Capstone`-shaped failure mode
        // is exercised by `accepts_capstone_ecosystem_name` above.
        let canonical_ecosystem = [
            // Compiler-bundled runtime + special-case external prefix.
            "Chelis", "Std", "CEarchin",
            // Currently shipped reef shells (canonical reference §"Shell
            // Ecosystem" — Active and current-phase rows).
            "Nautilus", "Coral", "Shoals", "Octant",
            // Downstream shell repo (ONNX shell; canonical reference
            // §"Shell Ecosystem"). `Hydronnx` is 1 leading cap + 7
            // lowercase ("ydronnx"), tripping the >=7 long-run heuristic
            // like `Capstone`, so it must be in KNOWN_SINGLE_WORDS.
            "Hydronnx",
            // Post-Phase-3 stub shells (canonical reference §"Shell
            // Ecosystem" — Stub / Future rows).
            "School", "Darwin", "Hull", "Beacon",
        ];
        for name in canonical_ecosystem {
            assert_eq!(
                component_violation(name),
                None,
                "ecosystem name `{name}` should pass module-pascal-components per §6.3",
            );
        }
    }
}
