//! Phase 1f — examples + specs agree with the compiler parser AND the
//! executable grammar validator (`chelis-validate`).
//!
//! ## Scope: validator LIBRARY, not the CLI's full surface
//!
//! This file tests the `chelis-validate` LIBRARY — the grammar validator
//! (`validate_{surf,deep,desugared}`) and renderer-vs-parser agreement
//! between the Surf→Deep render pipeline and both the strict compiler
//! parser and the validator. It does NOT exercise the `chelis` CLI's full
//! surface, and it does NOT claim the same coverage as the old subprocess
//! path it replaced.
//!
//! The validator path used to shell out to `chelis validate <mode> <file>`
//! once per corpus entry. Each `#[test]` here iterates the executable +
//! illustrative example corpora plus a handful of spec fixtures, which
//! meant ~30 sequential `chelis` cold-starts inside a single test
//! function — unparallelizable by nextest. The validator is a thin
//! library (`chelis_validate::validate_{surf,deep,desugared}`) and the
//! deep renderer is the same `chelis_surf::desugar` +
//! `chelis_macros::expand_program` + `chelis_deep::printer::print_canonical`
//! pipeline the `chelis deep` CLI uses, so this file now drives both
//! in-process.
//!
//! ## What the in-process path does NOT cover (covered elsewhere)
//!
//! The old subprocess path implicitly ran the CLI style gate
//! (`chelis fmt --check` + the blocking lint rule set). The in-process
//! library calls do NOT: `validate_deep`/`validate_surf` are pure
//! grammar + tag-vocabulary/arity checks. Specifically, **canonical
//! formatting is not enforced in-process** — `validate_deep`'s pest
//! grammar treats `\n` as ordinary `WHITESPACE` (see `deep.pest`:
//! `program = { SOI ~ spacing ~ node+ ~ EOI }`, trailing `spacing`
//! optional), so a Deep fixture WITHOUT a final newline, or with CRLF
//! line endings, validates fine in-process. That is by design: it is a
//! style-gate concern, not a grammar concern, so this file deliberately
//! adds NO in-process missing-newline negative test (it would assert a
//! behavior the library does not have).
//!
//! The CLI style-gate wiring for `chelis validate` — including the
//! missing-final-newline and CRLF rejections the reviewer asked about —
//! is covered by `crates/chelis-cli/tests/style_gate.rs`:
//!
//! - `validate_surf_fails_on_non_canonical_source`
//! - `validate_deep_fails_on_deep_lint_violation`
//! - `validate_deep_bypass_emits_warning_on_stderr`
//! - `validate_deep_allows_lint_directive_without_format_failure`
//! - `validate_deep_still_rejects_missing_final_newline_after_directive_stripping`
//! - `validate_deep_still_rejects_crlf_after_directive_stripping`
//!
//! ## Directive stripping
//!
//! User-authored Deep input (the `chelis-deep` fences in `SKILL.md`)
//! is run through `chelis_validate::strip_deep_lint_directive_lines`
//! before strict-parse + validate, matching what `chelis validate
//! --deep` and `chelis surf <file.dp>` do at `crates/chelis-cli/src/main.rs`.
//! The strip is required: `validate_deep`'s pest grammar rejects a
//! leading `; chelis-lint:` line (locked by a unit test in
//! `chelis-validate`). Rendered-Deep inputs from `render_deep_from_surf`
//! are produced by `chelis_deep::printer::print_canonical`, which emits
//! no `;` comments and therefore no directive lines — so the strip is a
//! no-op there. We still apply it unconditionally at those call sites to
//! lock the "canonical Deep renderers emit no directive comments"
//! invariant: if a future printer regression ever emitted one, the
//! parse/validate assertions would still see directive-free input, but
//! the strip keeps this file's behavior independent of that invariant
//! rather than silently relying on it.

use chelis_deep::parser::parse_str_strict as parse_deep_strict;
use chelis_surf::parser::parse_str as parse_surf;
use std::fs;
use std::path::PathBuf;

#[derive(Debug)]
struct CodeBlock {
    lang: String,
    body: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root should exist")
}

fn examples_dir() -> PathBuf {
    repo_root().join("examples")
}

fn illustrative_dir() -> PathBuf {
    examples_dir().join("illustrative")
}

fn extract_code_blocks(markdown: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut current_lang = None::<String>;
    let mut current_body = Vec::new();

    for line in markdown.lines() {
        if let Some(lang) = &current_lang {
            if line.trim_start().starts_with("```") {
                blocks.push(CodeBlock {
                    lang: lang.clone(),
                    body: current_body.join("\n"),
                });
                current_lang = None;
                current_body.clear();
            } else {
                current_body.push(line.to_string());
            }
            continue;
        }

        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("```") {
            current_lang = Some(rest.trim().to_string());
        }
    }

    assert!(current_lang.is_none(), "unterminated fenced code block");
    blocks
}

/// Render a Surf source string to canonical Deep output, mirroring what
/// `chelis deep <file>` produces (parse Surf → desugar → macro expand →
/// canonical Deep print).
fn render_deep_from_surf(source: &str) -> Result<String, String> {
    let decls =
        chelis_surf::parser::parse_str(source).map_err(|e| format!("surf parse step: {e}"))?;
    let deep = chelis_surf::desugar::desugar_program(&decls);
    let expanded =
        chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
            .map_err(|e| format!("macro-expand step: {e}"))?;
    Ok(chelis_deep::printer::print_canonical(expanded.exprs()))
}

#[test]
fn phase1f_surf_examples_and_specs_agree_with_parser() {
    let mut surf_inputs = Vec::new();

    for dir in [examples_dir(), illustrative_dir()] {
        for entry in fs::read_dir(dir).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("ch") {
                surf_inputs.push((
                    path.display().to_string(),
                    fs::read_to_string(&path).expect("read surf"),
                ));
            }
        }
    }

    surf_inputs.push((
        "spec-surf-fixture".to_string(),
        "module Foo.Bar\nimport Baz(..)\ndef id(x: f32): f32 = x\n".to_string(),
    ));
    surf_inputs.push((
        "spec-semicolon-block-fixture".to_string(),
        "def f(axis) = { y = axis; y }\ndef g() = par { a; b }\n".to_string(),
    ));
    surf_inputs.push((
        "spec-script-fixture".to_string(),
        "x = (x : tensor[32, 784, f32])\ny = relu(x)\n".to_string(),
    ));

    for (label, source) in surf_inputs {
        assert!(
            parse_surf(&source).is_ok(),
            "compiler parser should accept {label}"
        );
        assert!(
            chelis_validate::validate_surf(&source).is_ok(),
            "validator should accept {label}"
        );
    }
}

#[test]
fn phase1f_deep_examples_and_specs_agree_with_strict_parser() {
    let mut deep_inputs = Vec::new();

    for entry in fs::read_dir(examples_dir()).expect("read_dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("ch") {
            let source = fs::read_to_string(&path).expect("read surf");
            let rendered = render_deep_from_surf(&source)
                .unwrap_or_else(|e| panic!("render deep for {}: {e}", path.display()));
            // Lock the "canonical Deep renderers emit no directive
            // comments" invariant explicitly: `print_canonical` must not
            // produce any `; chelis-lint:` line. If a future printer
            // regression ever did, this assertion fails loudly instead of
            // the directive silently riding through validation.
            assert_eq!(
                chelis_validate::strip_deep_lint_directive_lines(&rendered),
                rendered,
                "rendered Deep for {} must contain no lint directive lines",
                path.display()
            );
            deep_inputs.push((format!("deep output for {}", path.display()), rendered));
        }
    }

    deep_inputs.push((
        "spec-deep-fixture".to_string(),
        "(module {} hello_tensor (def {} main (fn {} (params {}) (lit {type: (t-prim {} int32)} 1))))\n"
            .to_string(),
    ));
    for (label, source) in deep_inputs {
        // Defensive: strip directive lines before validate even though
        // canonical renderers (asserted above) and the clean literal
        // fixture emit none. The strip is a unit-tested no-op on
        // directive-free input, so this keeps the call site honest about
        // what `validate_deep` accepts (it rejects a leading directive
        // line) without depending on the no-directive invariant holding.
        let source = chelis_validate::strip_deep_lint_directive_lines(&source);
        assert!(
            parse_deep_strict(&source).is_ok(),
            "strict compiler parser should accept {label}"
        );
        assert!(
            chelis_validate::validate_deep(&source).is_ok(),
            "validator should accept {label}"
        );
    }

    let dotted_deep = render_deep_from_surf("module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n")
        .expect("render dotted module surf");
    assert_eq!(
        chelis_validate::strip_deep_lint_directive_lines(&dotted_deep),
        dotted_deep,
        "rendered dotted-path Deep must contain no lint directive lines"
    );
    let dotted_deep = chelis_validate::strip_deep_lint_directive_lines(&dotted_deep);
    assert!(
        chelis_validate::validate_deep(&dotted_deep).is_ok(),
        "validator should accept canonical Deep with dotted module/import paths"
    );
}

#[test]
fn phase1f_skill_blocks_agree_with_compiler_paths() {
    let skill = fs::read_to_string(repo_root().join("packages/chelis-std/SKILL.md"))
        .expect("read package SKILL");
    let blocks = extract_code_blocks(&skill);

    for (index, block) in blocks.into_iter().enumerate() {
        match block.lang.as_str() {
            "chelis-surf" => {
                assert!(
                    parse_surf(&block.body).is_ok(),
                    "compiler parser should accept SKILL surf block {}",
                    index + 1
                );
                assert!(
                    chelis_validate::validate_surf(&block.body).is_ok(),
                    "validator should accept SKILL surf block {}",
                    index + 1
                );
            }
            "chelis-deep" => {
                let body = chelis_validate::strip_deep_lint_directive_lines(&block.body);
                assert!(
                    parse_deep_strict(&body).is_ok(),
                    "compiler strict parser should accept SKILL deep block {}",
                    index + 1
                );
                if let Err(err) = chelis_validate::validate_deep(&body) {
                    panic!(
                        "validator should accept SKILL deep block {}: {err}",
                        index + 1
                    );
                }
            }
            _ => {}
        }
    }
}

#[test]
fn phase1f_desugar_accepts_executable_examples() {
    for entry in fs::read_dir(examples_dir()).expect("read_dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("ch") {
            let source = fs::read_to_string(&path).expect("read surf");
            assert!(
                chelis_validate::validate_desugared(&source).is_ok(),
                "desugar validator should accept {}",
                path.display()
            );
        }
    }

    let dotted_paths = "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n";
    assert!(
        chelis_validate::validate_desugared(dotted_paths).is_ok(),
        "desugar validator should accept dotted module/import paths"
    );
}

#[test]
fn phase1f_negative_fixtures_fail_in_validator_and_compiler() {
    let bad_surf = [
        "def f(x) = a == b == c\n",
        "def f(x) = if x then y\n",
        "type Option[a] = | Some(a\n",
    ];
    for source in bad_surf {
        assert!(
            parse_surf(source).is_err(),
            "compiler parser should reject {source:?}"
        );
        assert!(
            chelis_validate::validate_surf(source).is_err(),
            "validator should reject {source:?}"
        );
    }

    let bad_deep = [
        "(mystery {} x)\n",
        "(if {} cond then)\n",
        "(var x)\n",
        "(fn {} x body)\n",
    ];
    for source in bad_deep {
        assert!(
            parse_deep_strict(source).is_err(),
            "strict compiler parser should reject {source:?}"
        );
        assert!(
            chelis_validate::validate_deep(source).is_err(),
            "validator should reject {source:?}"
        );
    }
}
