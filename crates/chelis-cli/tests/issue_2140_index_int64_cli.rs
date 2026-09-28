//! chelis#2140 through the serialized `chelis check` surface.

use assert_cmd::Command;
use chelis_surf::token::{LiteralSuffix, Token, TokenKind};
use std::fs;
use std::path::Path;
use std::process::{ExitStatus, Output};
use tempfile::tempdir;

struct CheckResult {
    status: ExitStatus,
    report: serde_json::Value,
}

fn check(source: &str) -> CheckResult {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("index_int64.ch");
    fs::write(&path, source).expect("write fixture");
    let Output {
        status,
        stdout,
        stderr,
    } = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let report = serde_json::from_slice(&stdout).unwrap_or_else(|error| {
        panic!(
            "check JSON: {error}; status={status:?}; stdout={}; stderr={}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        )
    });
    CheckResult { status, report }
}

fn assert_accepted(label: &str, source: &str) {
    let CheckResult { status, report } = check(source);
    assert!(status.success(), "{label}: {status:?}: {report}");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{label}: {report}");
    assert!(
        report["errors"].as_array().unwrap().is_empty(),
        "{label}: {report}"
    );
}

fn assert_index_type_rejected(label: &str, source: &str, index_type: &str) {
    let CheckResult { status, report } = check(source);
    assert!(!status.success(), "{label}: {status:?}: {report}");
    assert!(report["score"].as_f64().unwrap() < 1.0, "{label}: {report}");
    let expected = format!("index expects i64 index, got {index_type}");
    assert!(
        report["errors"].as_array().unwrap().iter().any(|error| {
            error["kind"] == "TypeMismatch"
                && error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(&expected))
        }),
        "{label}: expected operation-owned diagnostic {expected:?}: {report}"
    );
}

fn strip_wrapping_parentheses(mut tokens: Vec<&Token>) -> Vec<&Token> {
    loop {
        if tokens.len() < 2
            || !matches!(
                tokens.first().map(|token| &token.kind),
                Some(TokenKind::LParen)
            )
            || !matches!(
                tokens.last().map(|token| &token.kind),
                Some(TokenKind::RParen)
            )
        {
            return tokens;
        }
        let mut depth = 0usize;
        let mut wraps_all = true;
        for (index, token) in tokens.iter().enumerate() {
            match token.kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth -= 1;
                    if depth == 0 && index + 1 != tokens.len() {
                        wraps_all = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if !wraps_all || depth != 0 {
            return tokens;
        }
        tokens = tokens[1..tokens.len() - 1].to_vec();
    }
}

fn direct_numeric_literal(tokens: &[Token], start: usize, end: usize) -> Option<bool> {
    let mut argument = tokens[start..end]
        .iter()
        .filter(|token| !matches!(token.kind, TokenKind::Newline))
        .collect::<Vec<_>>();
    argument = strip_wrapping_parentheses(argument);
    if matches!(
        argument.first().map(|token| &token.kind),
        Some(TokenKind::Minus)
    ) {
        argument.remove(0);
        argument = strip_wrapping_parentheses(argument);
    }
    let [literal] = argument.as_slice() else {
        return None;
    };
    match literal.kind {
        TokenKind::TypedInt(_, LiteralSuffix::I64)
        | TokenKind::IntMinMagnitude(Some(LiteralSuffix::I64)) => Some(true),
        TokenKind::Int(_)
        | TokenKind::IntMinMagnitude(_)
        | TokenKind::Float(_)
        | TokenKind::TypedInt(_, _)
        | TokenKind::TypedFloat(_, _) => Some(false),
        _ => None,
    }
}

fn non_i64_direct_index_literals(source: &str) -> Result<Vec<(usize, String)>, String> {
    let tokens = chelis_surf::lexer::lex(source).map_err(|error| error.to_string())?;
    let mut matches = Vec::new();
    for index in 0..tokens.len() {
        if !matches!(&tokens[index].kind, TokenKind::Ident(name) if name == "index") {
            continue;
        }
        let previous = tokens[..index]
            .iter()
            .rev()
            .find(|token| !matches!(token.kind, TokenKind::Newline));
        if matches!(previous.map(|token| &token.kind), Some(TokenKind::Dot)) {
            continue;
        }
        let Some(open) = tokens[index + 1..]
            .iter()
            .position(|token| !matches!(token.kind, TokenKind::Newline))
            .map(|offset| index + 1 + offset)
        else {
            continue;
        };
        if !matches!(tokens[open].kind, TokenKind::LParen) {
            continue;
        }

        let mut depth = 0usize;
        let mut second_start = None;
        let mut second_end = None;
        for (cursor, token) in tokens.iter().enumerate().skip(open + 1) {
            match token.kind {
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => depth += 1,
                TokenKind::RParen if depth == 0 => {
                    second_end = second_start.map(|_| cursor);
                    break;
                }
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => depth -= 1,
                TokenKind::Comma if depth == 0 => {
                    if second_start.is_some() {
                        second_end = Some(cursor);
                        break;
                    }
                    second_start = Some(cursor + 1);
                }
                _ => {}
            }
        }
        let (Some(second_start), Some(second_end)) = (second_start, second_end) else {
            continue;
        };
        if direct_numeric_literal(&tokens, second_start, second_end) == Some(false) {
            let Some(first) = tokens[second_start..second_end]
                .iter()
                .find(|token| !matches!(token.kind, TokenKind::Newline))
            else {
                continue;
            };
            let Some(last) = tokens[second_start..second_end]
                .iter()
                .rev()
                .find(|token| !matches!(token.kind, TokenKind::Newline))
            else {
                continue;
            };
            matches.push((
                source[..tokens[index].span.offset]
                    .bytes()
                    .filter(|byte| *byte == b'\n')
                    .count()
                    + 1,
                source[first.span.offset..last.span.end()].to_string(),
            ));
        }
    }
    Ok(matches)
}

#[test]
fn direct_index_literal_fixture_oracle_fails_closed() {
    let direct = "index";
    let source = format!(
        "\
-- prose: index([1i64], 99)
text = \"index([1i64], 98)\"
bare = {direct}([1i64], 0)
wrong_width = {direct}([1i64], 1i32)
wrong_family = {direct}([1i64], 2.0f32)
exact = {direct}([1i64], 3i64)
nested = {direct}({direct}([[1i64]], 0i64), (4))
method = values.{direct}(5)
"
    );
    assert_eq!(
        non_i64_direct_index_literals(&source).expect("lex control"),
        vec![
            (3, "0".to_string()),
            (4, "1i32".to_string()),
            (5, "2.0f32".to_string()),
            (7, "(4)".to_string()),
        ]
    );
    let constructed = ["out: i64 = index([1i64], ", "0)\n"].concat();
    assert_eq!(
        non_i64_direct_index_literals(&constructed).expect("lex constructed fixture"),
        vec![(1, "0".to_string())]
    );
}

#[test]
fn tracked_chelis_sources_use_exact_i64_for_direct_index_literals() {
    // This is a source-language migration oracle over actual `.ch` files.
    // Rust/Python-generated rejection controls remain owned by their focused
    // executable tests below rather than an unsound static-string inventory.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tracked = std::process::Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--", "*.ch"])
        .output()
        .expect("list tracked Chelis sources");
    assert!(
        tracked.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&tracked.stderr)
    );

    let mut violations = Vec::new();
    for relative in tracked.stdout.split(|byte| *byte == b'\0') {
        if relative.is_empty() {
            continue;
        }
        let relative = String::from_utf8(relative.to_vec()).expect("UTF-8 tracked path");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let found = non_i64_direct_index_literals(&source)
            .unwrap_or_else(|error| panic!("lex tracked Chelis source {relative}: {error}"));
        for (line, argument) in found {
            violations.push(format!("{relative}:{line}: index argument `{argument}`"));
        }
    }
    violations.sort();
    assert!(
        violations.is_empty(),
        "direct index literal audit requires exact i64:\n{}",
        violations.join("\n")
    );
}

#[test]
fn direct_index_accepts_exact_i64_in_cli() {
    assert_accepted("exact i64 index", "out: i64 = index([1i64], 0i64)\n");
}

#[test]
fn direct_index_rejects_every_other_integer_width_in_cli() {
    // Mutation-equivalent negative control for the former broad predicate.
    for (index_type, literal) in [("i8", "0i8"), ("i16", "0i16"), ("i32", "0i32")] {
        let source = format!("out: i64 = index([1i64], {literal})\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}

#[test]
fn direct_index_rejects_active_non_integer_primitives_in_cli() {
    for index_type in ["f16", "bf16", "f32", "f64", "bool", "string"] {
        let source = format!("def pick(xs: List[i64], i: {index_type}) -> i64 = index(xs, i)\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}
