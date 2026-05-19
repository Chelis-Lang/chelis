//! Phase 1f — examples + specs agree with the compiler parser AND the
//! executable grammar validator (`chelis-validate`).
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
//! in-process. Style-gate enforcement is a CLI-only concern (it ran in
//! the old subprocess path via `CHELIS_STYLE_GATE_DISABLE=1`, which the
//! library calls don't see anyway).

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
    let decls = chelis_surf::parser::parse_str(source).map_err(|e| e.to_string())?;
    let deep = chelis_surf::desugar::desugar_program(&decls);
    let expanded =
        chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
            .map_err(|e| e.to_string())?;
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
            deep_inputs.push((format!("deep output for {}", path.display()), rendered));
        }
    }

    deep_inputs.push((
        "spec-deep-fixture".to_string(),
        "(module {} hello_tensor (def {} main (fn {} (params {}) (lit {type: (t-prim {} int32)} 1))))\n"
            .to_string(),
    ));
    for (label, source) in deep_inputs {
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
                assert!(
                    parse_deep_strict(&block.body).is_ok(),
                    "compiler strict parser should accept SKILL deep block {}",
                    index + 1
                );
                if let Err(err) = chelis_validate::validate_deep(&block.body) {
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
