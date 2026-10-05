use std::fs;
use std::path::PathBuf;

use chelis_deep::parser::{parse_str as parse_deep, parse_str_strict as parse_deep_strict};
use chelis_deep::printer::print_canonical;
use chelis_deep::validate::{WarningKind, validate};
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_source;
use chelis_surf::parser::parse_str as parse_surf;

#[derive(Debug)]
struct CodeBlock {
    lang: String,
    start_line: usize,
    body: String,
}

#[derive(Clone, Copy)]
struct MarkdownExpectations {
    min_surf_blocks: usize,
    min_deep_blocks: usize,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root should exist")
}

fn extract_code_blocks(markdown: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut current_lang = None::<String>;
    let mut current_start = 0usize;
    let mut current_body = Vec::new();

    for (index, line) in markdown.lines().enumerate() {
        let line_no = index + 1;
        if let Some(lang) = &current_lang {
            if line.trim_start().starts_with("```") {
                blocks.push(CodeBlock {
                    lang: lang.clone(),
                    start_line: current_start,
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
            current_start = line_no + 1;
        }
    }

    assert!(
        current_lang.is_none(),
        "unterminated fenced code block in SKILL.md"
    );

    blocks
}

fn collect_markdown_files(dir: &PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_markdown_files(&path));
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn assert_valid_report(label: &str, report: &chelis_types::fitness::FitnessReport) {
    assert!(
        report.score >= 0.9,
        "{label} scored {}, expected >= 0.9. errors: {:?}",
        report.score,
        report.errors
    );
    assert!(
        report.errors.is_empty(),
        "{label} has checker errors despite score {}: {:?}",
        report.score,
        report.errors
    );
}

/// A documented Surf block is what a reader copies into a `.ch` file, and
/// `chelis check`, `eval`, and `build` refuse a file that is not canonically
/// formatted. Parsing and type-checking the block is therefore not enough: it
/// must also be byte-identical to the formatter's output.
fn assert_canonical_surf(label: &str, body: &str) {
    let source = format!("{body}\n");
    let canonical = format_source(&source)
        .unwrap_or_else(|err| panic!("{label} failed to format: {err}\n{body}"));
    assert_eq!(
        canonical, source,
        "{label} is not canonically formatted; `chelis fmt` prints:\n{canonical}"
    );
}

fn validate_markdown_file(path: &PathBuf, expectations: Option<MarkdownExpectations>) {
    let markdown =
        fs::read_to_string(path).unwrap_or_else(|_| panic!("failed to read {}", path.display()));
    let blocks = extract_code_blocks(&markdown);

    let surf_blocks: Vec<_> = blocks
        .iter()
        .filter(|block| block.lang == "chelis-surf")
        .collect();
    let deep_blocks: Vec<_> = blocks
        .iter()
        .filter(|block| block.lang == "chelis-deep")
        .collect();

    if let Some(expectations) = expectations {
        assert!(
            surf_blocks.len() >= expectations.min_surf_blocks,
            "expected at least {} validated Surf examples in {}, found {}",
            expectations.min_surf_blocks,
            path.display(),
            surf_blocks.len()
        );
        assert!(
            deep_blocks.len() >= expectations.min_deep_blocks,
            "expected at least {} validated Deep examples in {}, found {}",
            expectations.min_deep_blocks,
            path.display(),
            deep_blocks.len()
        );
    }

    for (index, block) in surf_blocks.iter().enumerate() {
        let label = format!(
            "Surf example {} at {}:{}",
            index + 1,
            path.display(),
            block.start_line
        );
        let decls = parse_surf(&block.body)
            .unwrap_or_else(|err| panic!("{label} failed to parse: {err}\n{}", block.body));
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        let report = chelis_types::check_program(&deep);
        assert_valid_report(&label, &report);
        assert_canonical_surf(&label, &block.body);

        let printed = print_canonical(&deep);
        parse_deep_strict(&printed)
            .unwrap_or_else(|err| panic!("{label} desugared to invalid Deep: {err}\n{printed}"));
    }

    // A fragment is not type-checked on its own, but one that parses as a
    // program is still shown the way the formatter prints it.
    for block in blocks
        .iter()
        .filter(|block| block.lang == "chelis-surf-fragment")
    {
        if parse_surf(&block.body).is_ok() {
            let label = format!("Surf fragment at {}:{}", path.display(), block.start_line);
            assert_canonical_surf(&label, &block.body);
        }
    }

    for (index, block) in deep_blocks.iter().enumerate() {
        let label = format!(
            "Deep example {} at {}:{}",
            index + 1,
            path.display(),
            block.start_line
        );
        let exprs = parse_deep_strict(&block.body)
            .unwrap_or_else(|err| panic!("{label} failed strict parse: {err}\n{}", block.body));

        let warnings = validate(&exprs);
        let disallowed: Vec<_> = warnings
            .iter()
            .filter(|warning| {
                matches!(
                    warning.kind,
                    WarningKind::UnknownTag
                        | WarningKind::MissingMetadata
                        | WarningKind::Structural
                        | WarningKind::Arity
                )
            })
            .collect();
        assert!(
            disallowed.is_empty(),
            "{label} produced validation warnings: {:?}",
            disallowed
                .iter()
                .map(|warning| &warning.message)
                .collect::<Vec<_>>()
        );

        let reparsed = parse_deep(&block.body)
            .unwrap_or_else(|err| panic!("{label} failed non-strict parse: {err}\n{}", block.body));
        let canonical = print_canonical(&reparsed);
        parse_deep_strict(&canonical).unwrap_or_else(|err| {
            panic!("{label} failed after canonical reprint: {err}\n{canonical}")
        });

        let report = chelis_types::check_program(&exprs);
        assert_valid_report(&label, &report);
    }
}

#[test]
fn skill_examples_parse_validate_and_typecheck() {
    let skill_path = repo_root().join("packages/chelis-std/SKILL.md");
    validate_markdown_file(
        &skill_path,
        Some(MarkdownExpectations {
            min_surf_blocks: 10,
            min_deep_blocks: 10,
        }),
    );
}

#[test]
fn mdbook_examples_parse_validate_and_typecheck() {
    let docs_root = repo_root().join("docs/book/src");
    let markdown_files = collect_markdown_files(&docs_root);
    assert!(
        !markdown_files.is_empty(),
        "expected mdBook markdown files under {}",
        docs_root.display()
    );

    for path in markdown_files {
        validate_markdown_file(&path, None);
    }
}
