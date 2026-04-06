use std::fs;
use std::path::PathBuf;

use chelis_deep::parser::{parse_str as parse_deep, parse_str_strict as parse_deep_strict};
use chelis_deep::printer::print_canonical;
use chelis_deep::validate::{validate, WarningKind};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;

#[derive(Debug)]
struct CodeBlock {
    lang: String,
    start_line: usize,
    body: String,
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

#[test]
fn skill_examples_parse_validate_and_typecheck() {
    let skill_path = repo_root().join("SKILL.md");
    let skill = fs::read_to_string(&skill_path).expect("failed to read SKILL.md");
    let blocks = extract_code_blocks(&skill);

    let surf_blocks: Vec<_> = blocks
        .iter()
        .filter(|block| block.lang == "chelis-surf")
        .collect();
    let deep_blocks: Vec<_> = blocks
        .iter()
        .filter(|block| block.lang == "chelis-deep")
        .collect();

    assert!(
        surf_blocks.len() >= 10,
        "expected at least 10 validated Surf examples, found {}",
        surf_blocks.len()
    );
    assert!(
        deep_blocks.len() >= 10,
        "expected at least 10 validated Deep examples, found {}",
        deep_blocks.len()
    );

    for (index, block) in surf_blocks.iter().enumerate() {
        let label = format!(
            "SKILL Surf example {} at {}:{}",
            index + 1,
            skill_path.display(),
            block.start_line
        );
        let decls = parse_surf(&block.body)
            .unwrap_or_else(|err| panic!("{label} failed to parse: {err}\n{}", block.body));
        let deep = desugar_program(&decls);
        let report = chelis_types::check_program(&deep);
        assert_valid_report(&label, &report);

        let printed = print_canonical(&deep);
        parse_deep_strict(&printed)
            .unwrap_or_else(|err| panic!("{label} desugared to invalid Deep: {err}\n{printed}"));
    }

    for (index, block) in deep_blocks.iter().enumerate() {
        let label = format!(
            "SKILL Deep example {} at {}:{}",
            index + 1,
            skill_path.display(),
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
