use assert_cmd::Command;
use chelis_deep::parser::parse_str_strict as parse_deep_strict;
use chelis_surf::parser::parse_str as parse_surf;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::process::Output;
use std::sync::OnceLock;
use tempfile::{Builder, NamedTempFile};

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

fn chelis_bin() -> &'static Path {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let debug_bin = repo_root().join("target/debug/chelis");
        if debug_bin.exists() {
            return debug_bin;
        }

        let status = StdCommand::new("cargo")
            .args(["build", "-p", "chelis-cli"])
            .current_dir(repo_root())
            .status()
            .expect("build chelis-cli");
        assert!(status.success(), "cargo build -p chelis-cli failed");
        repo_root().join("target/debug/chelis")
    })
    .as_path()
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

fn run_validate(mode: &str, path: &Path) -> Output {
    Command::new(chelis_bin())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", mode, path.to_str().unwrap()])
        .output()
        .expect("run chelis validate")
}

fn render_deep(path: &Path) -> Vec<u8> {
    Command::new(chelis_bin())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone()
}

fn write_temp_file(ext: &str, contents: &str) -> NamedTempFile {
    let file = Builder::new()
        .suffix(&format!(".{ext}"))
        .tempfile()
        .expect("temp file");
    fs::write(file.path(), contents).expect("write fixture");
    file
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
        let file = write_temp_file("ch", &source);
        assert!(
            run_validate("--surf", file.path()).status.success(),
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
            let output = render_deep(&path);
            deep_inputs.push((
                format!("deep output for {}", path.display()),
                String::from_utf8(output).expect("utf8 deep output"),
            ));
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
        let file = write_temp_file("dp", &source);
        assert!(
            run_validate("--deep", file.path()).status.success(),
            "validator should accept {label}"
        );
    }

    let dotted_module_surf =
        write_temp_file("ch", "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n");
    let dotted_deep =
        String::from_utf8(render_deep(dotted_module_surf.path())).expect("utf8 deep output");
    let dotted_file = write_temp_file("dp", &dotted_deep);
    assert!(
        run_validate("--deep", dotted_file.path()).status.success(),
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
                let file = write_temp_file("ch", &block.body);
                assert!(
                    run_validate("--surf", file.path()).status.success(),
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
                let file = write_temp_file("dp", &block.body);
                let output = run_validate("--deep", file.path());
                assert!(
                    output.status.success(),
                    "validator should accept SKILL deep block {}: {}",
                    index + 1,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
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
            assert!(
                run_validate("--desugar", &path).status.success(),
                "--desugar should accept {}",
                path.display()
            );
        }
    }

    let dotted_paths = write_temp_file("ch", "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n");
    assert!(
        run_validate("--desugar", dotted_paths.path())
            .status
            .success(),
        "--desugar should accept dotted module/import paths"
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
        let file = write_temp_file("ch", source);
        assert!(
            parse_surf(source).is_err(),
            "compiler parser should reject {source:?}"
        );
        assert!(
            !run_validate("--surf", file.path()).status.success(),
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
        let file = write_temp_file("dp", source);
        assert!(
            parse_deep_strict(source).is_err(),
            "strict compiler parser should reject {source:?}"
        );
        assert!(
            !run_validate("--deep", file.path()).status.success(),
            "validator should reject {source:?}"
        );
    }
}
