//! chelis#720 regression matrix for `fold_static_cond`: cast operands must be
//! finalized as sealed f16/bf16 scalars before comparison. Folding through an
//! f32 memo deletes the branch IEEE f16/bf16 semantics require.
//!
//! Sibling of chelis#711 (the integer Const arm of the same fold, threshold
//! 2^53); this one fires at 2049 (f16) / 257 (bf16). The fixes do not
//! overlap: #711's checked-i64 folding does not touch the Cast arm.
//!
//! All branch-presence checks compare f32 BIT PATTERNS in the emitted C
//! (111.0 = 0x42de0000, 222.0 = 0x435e0000), never decimal text - a decimal
//! `111` grep matches the hash constant 0x94D049BB133111EBULL (the exact
//! false-positive the #711 audit recorded).
//!
//! Bounding controls: conditions with an effectful branch (`fail`) route
//! host-lane, do not fold, and the compiled i64 comparison there is exact
//! (locked below). The Phase 3 i8 row also proves that folding cannot hide
//! the required checked-overflow trap.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build to C; return `(emitted_c, run_stdout, run_stderr, run_ok)`.
fn build_and_run_c(program: &str, name: &str) -> Result<(String, String, String, bool), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let emitted = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .map_err(|e| format!("read emitted C: {e}"))?;
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok((
        emitted,
        String::from_utf8_lossy(&run.stdout).into_owned(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        run.status.success(),
    ))
}

fn eval_first_line(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

#[derive(Clone, Copy)]
struct CToken<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

/// Tokenize enough C to recover generated function and statement structure.
/// Comments, strings, and character literals are skipped so braces, marker
/// spellings, and alternate symbol occurrences inside them cannot affect the
/// oracle.
fn c_tokens(source: &str) -> Result<Vec<CToken<'_>>, String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            let comment_start = index;
            index += 2;
            while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/') {
                index += 1;
            }
            if index + 1 == bytes.len() {
                return Err(format!(
                    "unterminated block comment at byte {comment_start}"
                ));
            }
            index += 2;
            continue;
        }
        if matches!(bytes[index], b'"' | b'\'') {
            let literal_start = index;
            let quote = bytes[index];
            index += 1;
            let mut closed = false;
            while index < bytes.len() {
                match bytes[index] {
                    b'\\' => {
                        index += 1;
                        if index < bytes.len() {
                            let escaped =
                                source[index..].chars().next().expect("index is in bounds");
                            index += escaped.len_utf8();
                        }
                    }
                    byte if byte == quote => {
                        index += 1;
                        closed = true;
                        break;
                    }
                    _ => {
                        let character = source[index..].chars().next().expect("index is in bounds");
                        index += character.len_utf8();
                    }
                }
            }
            if !closed {
                return Err(format!("unterminated C literal at byte {literal_start}"));
            }
            continue;
        }

        let start = index;
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
        } else if bytes[index].is_ascii_digit() {
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b'.'))
            {
                index += 1;
            }
        } else {
            let character = source[index..].chars().next().expect("index is in bounds");
            index += character.len_utf8();
        }
        tokens.push(CToken {
            text: &source[start..index],
            start,
            end: index,
        });
    }
    Ok(tokens)
}

fn closing_token(
    tokens: &[CToken<'_>],
    open: usize,
    open_text: &str,
    close_text: &str,
) -> Result<usize, String> {
    if tokens.get(open).map(|token| token.text) != Some(open_text) {
        return Err(format!(
            "expected `{open_text}` at token {open}, found `{}`",
            tokens.get(open).map_or("<end>", |token| token.text)
        ));
    }
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        if token.text == open_text {
            depth += 1;
        } else if token.text == close_text {
            depth = depth
                .checked_sub(1)
                .ok_or_else(|| format!("unmatched `{close_text}` at token {index}"))?;
            if depth == 0 {
                return Ok(index);
            }
        }
    }
    Err(format!(
        "unterminated `{open_text}` beginning at token {open}"
    ))
}

/// Return only the emitted implementation body for the exact authored `pick`
/// definition. Token matching rejects forward declarations, calls, suffixed
/// identifiers, and lookalike occurrences in comments or literals.
fn emitted_pick_body(emitted: &str) -> Result<&str, String> {
    let name = format!("{}__chelis_owned_body", common::authored_c_symbol("pick"));
    let tokens = c_tokens(emitted)?;
    let mut bodies = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text != name || tokens.get(index + 1).map(|next| next.text) != Some("(") {
            continue;
        }
        let close_params = closing_token(&tokens, index + 1, "(", ")")?;
        let Some(open_body) = tokens.get(close_params + 1) else {
            continue;
        };
        if open_body.text != "{" {
            continue;
        }
        let close_body = closing_token(&tokens, close_params + 1, "{", "}")?;
        bodies.push((open_body.end, tokens[close_body].start));
    }
    match bodies.as_slice() {
        [(start, end)] => Ok(&emitted[*start..*end]),
        [] => Err(format!("no definition of exact generated symbol `{name}`")),
        _ => Err(format!(
            "ambiguous generated C: found {} definitions of exact symbol `{name}`",
            bodies.len()
        )),
    }
}

fn c_unsigned_integer(token: &str) -> Option<u64> {
    let lower = token.to_ascii_lowercase();
    let (digits_and_suffix, radix) = lower
        .strip_prefix("0x")
        .map_or((lower.as_str(), 10), |hex| (hex, 16));
    let digits = digits_and_suffix
        .bytes()
        .take_while(|byte| byte.is_ascii_digit() || (radix == 16 && byte.is_ascii_hexdigit()))
        .count();
    if digits == 0
        || !digits_and_suffix[digits..]
            .bytes()
            .all(|byte| matches!(byte, b'u' | b'l'))
    {
        return None;
    }
    u64::from_str_radix(&digits_and_suffix[..digits], radix).ok()
}

fn direct_call_count(tokens: &[CToken<'_>], function: &str) -> Result<usize, String> {
    let mut count = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        if token.text != function || tokens.get(index + 1).map(|next| next.text) != Some("(") {
            continue;
        }
        let close = closing_token(tokens, index + 1, "(", ")")?;
        let starts_statement = index == 0 || matches!(tokens[index - 1].text, ";" | "{" | "}");
        if starts_statement && tokens.get(close + 1).map(|next| next.text) == Some(";") {
            count += 1;
        }
    }
    Ok(count)
}

#[derive(Clone, Copy)]
struct MarkerCounts {
    fail_invocations: usize,
    bits_222: usize,
}

fn marker_counts(tokens: &[CToken<'_>]) -> Result<MarkerCounts, String> {
    Ok(MarkerCounts {
        fail_invocations: direct_call_count(tokens, "chelis_fail")?,
        bits_222: tokens
            .iter()
            .filter(|token| c_unsigned_integer(token.text) == Some(BITS_222_VALUE))
            .count(),
    })
}

fn emitted_pick_marker_counts(emitted: &str) -> Result<MarkerCounts, String> {
    let pick = emitted_pick_body(emitted)?;
    marker_counts(&c_tokens(pick)?)
}

#[derive(Clone, Copy)]
struct ConditionalStructure {
    if_token: usize,
    condition: (usize, usize),
    then_arm: (usize, usize),
    else_arm: (usize, usize),
}

fn pick_conditional_structure(tokens: &[CToken<'_>]) -> Result<ConditionalStructure, String> {
    let mut candidates = Vec::new();
    let mut brace_depth = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        if token.text == "}" {
            brace_depth = brace_depth
                .checked_sub(1)
                .ok_or_else(|| format!("unmatched `}}` at token {index} in `pick`"))?;
        }
        if brace_depth == 0
            && token.text == "if"
            && tokens.get(index + 1).map(|next| next.text) == Some("(")
        {
            let close_condition = closing_token(tokens, index + 1, "(", ")")?;
            let then_open = close_condition + 1;
            if tokens.get(then_open).map(|next| next.text) != Some("{") {
                return Err("generated top-level `if` has no braced `then` arm".to_string());
            }
            let then_close = closing_token(tokens, then_open, "{", "}")?;
            if tokens.get(then_close + 1).map(|next| next.text) != Some("else")
                || tokens.get(then_close + 2).map(|next| next.text) != Some("{")
            {
                return Err("generated top-level `if` has no braced `else` arm".to_string());
            }
            let else_open = then_close + 2;
            let else_close = closing_token(tokens, else_open, "{", "}")?;
            let then_range = (then_open + 1, then_close);
            let else_range = (else_open + 1, else_close);
            let then_markers = marker_counts(&tokens[then_range.0..then_range.1])?;
            let else_markers = marker_counts(&tokens[else_range.0..else_range.1])?;
            if (then_markers.fail_invocations + else_markers.fail_invocations > 0)
                && (then_markers.bits_222 + else_markers.bits_222 > 0)
            {
                candidates.push(ConditionalStructure {
                    if_token: index,
                    condition: (index + 2, close_condition),
                    then_arm: then_range,
                    else_arm: else_range,
                });
            }
        }
        if token.text == "{" {
            brace_depth += 1;
        }
    }
    match candidates.as_slice() {
        [candidate] => Ok(*candidate),
        [] => Err(
            "no top-level generated `if/else` contains both an invoked fail and 222.0 bits"
                .to_string(),
        ),
        _ => Err(format!(
            "ambiguous `pick` structure: {} top-level conditionals contain both markers",
            candidates.len()
        )),
    }
}

fn strip_outer_parens<'a>(mut tokens: &'a [CToken<'a>]) -> Result<&'a [CToken<'a>], String> {
    while tokens.first().map(|token| token.text) == Some("(")
        && tokens.last().map(|token| token.text) == Some(")")
        && closing_token(tokens, 0, "(", ")")? == tokens.len() - 1
    {
        tokens = &tokens[1..tokens.len() - 1];
    }
    Ok(tokens)
}

fn top_level_statement_ranges(tokens: &[CToken<'_>]) -> Result<Vec<(usize, usize)>, String> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        match token.text {
            "(" => paren_depth += 1,
            ")" => {
                paren_depth = paren_depth
                    .checked_sub(1)
                    .ok_or_else(|| format!("unmatched `)` at token {index}"))?;
            }
            "[" => bracket_depth += 1,
            "]" => {
                bracket_depth = bracket_depth
                    .checked_sub(1)
                    .ok_or_else(|| format!("unmatched `]` at token {index}"))?;
            }
            "{" => brace_depth += 1,
            "}" => {
                brace_depth = brace_depth
                    .checked_sub(1)
                    .ok_or_else(|| format!("unmatched `}}` at token {index}"))?;
            }
            ";" if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                if start < index {
                    ranges.push((start, index));
                }
                start = index + 1;
            }
            _ => {}
        }
    }
    if paren_depth != 0 || bracket_depth != 0 || brace_depth != 0 {
        return Err("unterminated delimiter before generated `if`".to_string());
    }
    if start != tokens.len() {
        return Err("unterminated statement before generated `if`".to_string());
    }
    Ok(ranges)
}

fn assignment_rhs<'a>(statement: &'a [CToken<'a>], target: &str) -> Option<&'a [CToken<'a>]> {
    (statement.first().map(|token| token.text) == Some(target)
        && statement.get(1).map(|token| token.text) == Some("="))
    .then_some(&statement[2..])
}

fn exact_initializer(
    statements: &[&[CToken<'_>]],
    target: &str,
    expected: u64,
) -> Result<(), String> {
    let values: Vec<_> = statements
        .iter()
        .filter_map(|statement| assignment_rhs(statement, target))
        .collect();
    let [value] = values.as_slice() else {
        return Err(format!(
            "expected one initializer for comparator operand `{target}`, found {}",
            values.len()
        ));
    };
    let value = strip_outer_parens(value)?;
    if value.len() != 1 || c_unsigned_integer(value[0].text) != Some(expected) {
        return Err(format!(
            "comparator operand `{target}` is not initialized to exact integer {expected}"
        ));
    }
    Ok(())
}

fn validate_exact_lt_condition(
    tokens: &[CToken<'_>],
    structure: ConditionalStructure,
    expected_lhs: u64,
    expected_rhs: u64,
) -> Result<(), String> {
    let condition = strip_outer_parens(&tokens[structure.condition.0..structure.condition.1])?;
    let [condition_var] = condition else {
        return Err("generated `if` condition is not one exact temporary identifier".to_string());
    };
    let statement_ranges = top_level_statement_ranges(&tokens[..structure.if_token])?;
    let statements: Vec<_> = statement_ranges
        .iter()
        .map(|(start, end)| &tokens[*start..*end])
        .collect();
    let producers: Vec<_> = statements
        .iter()
        .enumerate()
        .filter_map(|(index, statement)| {
            assignment_rhs(statement, condition_var.text).map(|rhs| (index, rhs))
        })
        .collect();
    let [(producer_index, producer)] = producers.as_slice() else {
        return Err(format!(
            "expected one assignment producing condition `{}`, found {}",
            condition_var.text,
            producers.len()
        ));
    };
    let producer = strip_outer_parens(producer)?;
    let [lhs, operator, rhs] = producer else {
        return Err(format!(
            "condition `{}` is not one binary comparator",
            condition_var.text
        ));
    };
    if operator.text != "<" {
        return Err(format!(
            "condition `{}` must be produced by exact `<`, found `{}`",
            condition_var.text, operator.text
        ));
    }
    let prior_statements = &statements[..*producer_index];
    exact_initializer(prior_statements, lhs.text, expected_lhs)?;
    exact_initializer(prior_statements, rhs.text, expected_rhs)?;
    Ok(())
}

fn pick_has_expected_control(
    emitted: &str,
    expected_lhs: u64,
    expected_rhs: u64,
) -> Result<(), String> {
    let pick = emitted_pick_body(emitted)?;
    let tokens = c_tokens(pick)?;
    let structure = pick_conditional_structure(&tokens)?;
    validate_exact_lt_condition(&tokens, structure, expected_lhs, expected_rhs)?;
    let then_markers = marker_counts(&tokens[structure.then_arm.0..structure.then_arm.1])?;
    let else_markers = marker_counts(&tokens[structure.else_arm.0..structure.else_arm.1])?;
    if then_markers.fail_invocations != 1
        || then_markers.bits_222 != 0
        || else_markers.fail_invocations != 0
        || else_markers.bits_222 != 1
    {
        return Err(format!(
            "expected exactly one fail invocation in `then` and one 222.0 marker in `else`; got \
             then(fail={}, bits_222={}), else(fail={}, bits_222={})",
            then_markers.fail_invocations,
            then_markers.bits_222,
            else_markers.fail_invocations,
            else_markers.bits_222
        ));
    }
    Ok(())
}

/// f32 bit pattern of the 222.0 branch payload as it appears in emitted
/// `chelis_fill_scalar` calls (111.0 is 0x42de0000; the broken rows assert on
/// the DELETED branch's bits, which is 222.0's).
const BITS_222_VALUE: u64 = 0x435e0000;
const I64_EXACT_LOW: u64 = 9007199254740992;
const I64_EXACT_HIGH: u64 = 9007199254740993;
const TAKEN_FAIL_STDERR: &str = "i64 invariant violated\n";

fn synthetic_control_c(comparator: &str, then_arm: &str, else_arm: &str) -> String {
    let name = format!("{}__chelis_owned_body", common::authored_c_symbol("pick"));
    format!(
        r#"
void {name}(void) {{
    int64_t left;
    left = {I64_EXACT_LOW};
    int64_t right;
    right = {I64_EXACT_HIGH};
    bool condition;
    condition = (left {comparator} right);
    if (condition) {{
        {then_arm}
    }} else {{
        {else_arm}
    }}
}}
"#
    )
}

// ===========================================================================
// chelis#720 - the Cast arm deletes the IEEE-correct branch
// ===========================================================================

/// True f16 rounds cast(2049.0, f16) to 2048, so lt(2048, 2048) is false and
/// the answer is 222. Before the compiled Phase 3 fix, eval printed 222
/// (correct) while the
/// compiled binary prints 111, and 222's bit pattern is ABSENT from the
/// emitted C - the correct branch was deleted at compile time.
#[test]
fn f16_cast_condition_folds_with_f16_semantics() {
    let program = "def pick() -> f32 = if lt(cast(2048.0, f16), cast(2049.0, f16)) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    assert_eq!(
        eval_first_line(program).expect("eval"),
        "222.0",
        "eval is the correct lane here and must stay correct"
    );
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (emitted, stdout, _, ok) = build_and_run_c(program, "fold_f16").expect("C lane");
    assert!(ok);
    assert!(
        emitted_pick_marker_counts(&emitted)
            .expect("extract exact `pick` definition")
            .bits_222
            == 1,
        "the 222 branch (0x435e0000) must exist in the emitted C; it was deleted"
    );
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "222.0",
        "the compiled program must take the IEEE f16 branch; got: {stdout}"
    );
}

/// bf16 sibling at threshold 257 (8-bit mantissa).
#[test]
fn bf16_cast_condition_folds_with_bf16_semantics() {
    let program = "def pick() -> f32 = if lt(cast(256.0, bf16), cast(257.0, bf16)) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    assert_eq!(eval_first_line(program).expect("eval"), "222.0");
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (emitted, stdout, _, ok) = build_and_run_c(program, "fold_bf16").expect("C lane");
    assert!(ok);
    assert!(
        emitted_pick_marker_counts(&emitted)
            .expect("extract exact `pick` definition")
            .bits_222
            == 1,
        "the 222 branch (0x435e0000) must exist in the emitted C; it was deleted"
    );
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "222.0",
        "the compiled program must take the IEEE bf16 branch; got: {stdout}"
    );
}

// ===========================================================================
// chelis#718 - i8 conditions do NOT fold, but the runtime branch diverges
// through the int64_t widening (#714). Contract: the overflow must trap.
// ===========================================================================

/// `add(100i8, 100i8)` overflows i8. Today eval wraps (-56 < 0, prints
/// 111) and compiled C widens (200 < 0, prints 222) - opposite branches at
/// runtime, no deletion (both bit patterns present in the emitted C,
/// verified when this row was probed). The decided contract (#680/#695)
/// says the overflow itself must trap in both lanes.
#[test]
fn int8_overflow_condition_traps_in_both_lanes() {
    let program = "def pick() -> f32 = if lt(add(100i8, 100i8), 0i8) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    match eval_first_line(program) {
        Ok(line) => panic!("eval must trap on the i8 overflow, got: {line}"),
        Err(stderr) => assert!(stderr.contains("overflow"), "got: {stderr}"),
    }
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout, stderr, ok) = build_and_run_c(program, "fold_i8").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "compiled C must trap on the i8 overflow; got ok={ok}, stdout `{stdout}`"
    );
}

// ===========================================================================
// CONTROLS
// ===========================================================================

/// An effectful branch (`fail`) keeps the def in the host lane: no fold,
/// both branches present in the emitted C, and the compiled i64 comparison
/// is EXACT - `lt(2^53, 2^53 + 1)` is true, so the binary must trap with the
/// fail message. The generated C escapes user string bytes, so branch presence
/// is asserted structurally while the executed binary checks the exact payload.
/// (eval takes the wrong branch on the same program - that is chelis#680's
/// known f64 comparison bug, asserted nowhere here.)
#[test]
fn host_lane_fail_branch_survives_and_c_comparison_is_exact() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lt(9007199254740992i64, 9007199254740993i64) \
                   then fail(\"i64 invariant violated\") else 222.0\nout = print(pick())\n";
    let (emitted, _, stderr, ok) = build_and_run_c(program, "fold_fail").expect("C lane");
    pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH)
        .unwrap_or_else(|error| panic!("the host-lane branch structure is wrong: {error}"));
    assert!(
        !ok,
        "2^53 < 2^53 + 1 is true in exact integers; the compiled host lane \
         must take the fail branch. got ok={ok}, stderr: {stderr}"
    );
    assert_eq!(
        stderr, TAKEN_FAIL_STDERR,
        "the taken fail branch must emit only its exact payload"
    );
}

/// Negative control for the same host-lane branch: reversing the exact i64
/// comparison must keep the `fail` arm in the generated control flow without
/// executing it. This distinguishes "both branches survived" from a test that
/// passes only because the positive row happened to trap.
#[test]
fn host_lane_fail_branch_survives_when_exact_condition_is_false() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lt(9007199254740993i64, 9007199254740992i64) \
                   then fail(\"i64 invariant violated\") else 222.0\nout = print(pick())\n";
    let (emitted, stdout, stderr, ok) =
        build_and_run_c(program, "fold_fail_false").expect("C lane");
    pick_has_expected_control(&emitted, I64_EXACT_HIGH, I64_EXACT_LOW)
        .unwrap_or_else(|error| panic!("the host-lane branch structure is wrong: {error}"));
    assert!(
        ok && stdout.lines().next().unwrap_or("").trim() == "222.0",
        "the reversed exact comparison must take the non-failing branch; \
         got ok={ok}, stdout `{stdout}`, stderr `{stderr}`"
    );
    assert_eq!(
        stderr, "",
        "the untaken fail branch must emit no stderr bytes"
    );
}

/// Comparator negative parity for the taken control. `lte` has the same
/// runtime result for these operands, so only structural condition validation
/// can distinguish it from the required exact `lt`.
#[test]
fn structural_controls_reject_lte_in_taken_condition() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lte(9007199254740992i64, 9007199254740993i64) \
                   then fail(\"i64 invariant violated\") else 222.0\nout = print(pick())\n";
    let (emitted, _, stderr, ok) = build_and_run_c(program, "fold_fail_lte_taken").expect("C lane");
    assert!(!ok, "probe setup must still take its failing arm");
    assert_eq!(stderr, TAKEN_FAIL_STDERR);
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).is_err(),
        "the structural oracle must reject `lte` even when runtime branch selection is unchanged"
    );
}

/// Comparator negative parity for the untaken control. Reversed operands keep
/// both `lt` and `lte` false, so runtime output alone cannot catch the mutation.
#[test]
fn structural_controls_reject_lte_in_untaken_condition() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lte(9007199254740993i64, 9007199254740992i64) \
                   then fail(\"i64 invariant violated\") else 222.0\nout = print(pick())\n";
    let (emitted, stdout, stderr, ok) =
        build_and_run_c(program, "fold_fail_lte_untaken").expect("C lane");
    assert!(ok, "probe setup must leave its failing arm untaken");
    assert_eq!(stdout.lines().next(), Some("222.0"));
    assert_eq!(stderr, "");
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_HIGH, I64_EXACT_LOW).is_err(),
        "the structural oracle must reject reversed-operand `lte` despite identical execution"
    );
}

/// Negative parity for marker placement within `pick`: both searched markers
/// are deliberately emitted in the failing `then` arm. A body-wide search
/// accepts this malformed shape even though the surviving `else` arm is
/// 111.0, not 222.0.
#[test]
fn structural_controls_reject_both_markers_in_then_arm() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lt(9007199254740992i64, 9007199254740993i64) \
                   then add(222.0, fail(\"i64 invariant violated\")) else 111.0\n\
                   out = print(pick())\n";
    let (emitted, _, stderr, ok) = build_and_run_c(program, "fold_fail_same_then").expect("C lane");
    assert!(!ok, "probe setup must execute its failing `then` arm");
    assert_eq!(stderr, TAKEN_FAIL_STDERR);
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).is_err(),
        "the structural oracle must reject fail and 222.0 co-located in the `then` arm"
    );
}

/// Symmetric marker-placement mutation: both markers are in the untaken
/// `else` arm while the taken `then` arm survives as 111.0.
#[test]
fn structural_controls_reject_both_markers_in_else_arm() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lt(9007199254740992i64, 9007199254740993i64) \
                   then 111.0 else add(222.0, fail(\"i64 invariant violated\"))\n\
                   out = print(pick())\n";
    let (emitted, stdout, stderr, ok) =
        build_and_run_c(program, "fold_fail_same_else").expect("C lane");
    assert!(ok, "probe setup must leave its failing `else` arm untaken");
    assert_eq!(stdout.lines().next(), Some("111.0"));
    assert_eq!(stderr, "");
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).is_err(),
        "the structural oracle must reject fail and 222.0 co-located in the `else` arm"
    );
}

/// A declaration has the same identifier-plus-parenthesis prefix as a call,
/// but it does not execute. The fail marker must require an invocation.
#[test]
fn structural_controls_reject_fail_prototype_instead_of_invocation() {
    let emitted = synthetic_control_c(
        "<",
        "void chelis_fail(void);",
        "unsigned value = UINT32_C(0x435e0000);",
    );
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).is_err(),
        "a `chelis_fail` prototype must not satisfy the invocation contract"
    );
}

/// Marker multiplicity is exact: a second fail invocation in the failing arm
/// is not equivalent to the one required invocation.
#[test]
fn structural_controls_reject_duplicate_fail_invocations() {
    let emitted = synthetic_control_c(
        "<",
        "chelis_fail();\n        chelis_fail();",
        "unsigned value = UINT32_C(0x435e0000);",
    );
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).is_err(),
        "duplicate fail invocations must be rejected"
    );
}

/// Symmetric marker-multiplicity control for the surviving value arm.
#[test]
fn structural_controls_reject_duplicate_222_markers() {
    let emitted = synthetic_control_c(
        "<",
        "chelis_fail();",
        "unsigned first = UINT32_C(0x435e0000);\n\
         unsigned second = UINT32_C(0X435E0000UL);",
    );
    assert!(
        pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).is_err(),
        "duplicate 222.0 bit markers must be rejected independent of spelling"
    );
}

/// Extraction and tokenization control: only the exact generated definition
/// counts. Forward declarations, symbol lookalikes, comments, literals, and a
/// nested conditional must not create markers or confuse arm boundaries.
#[test]
fn structural_controls_parse_exact_pick_definition() {
    let name = format!("{}__chelis_owned_body", common::authored_c_symbol("pick"));
    let emitted = format!(
        r#"
void {name}(void);
const char *fake = "{name}(void) {{ chelis_fail(); 0x435e0000; }}";
void prefix_{name}(void) {{ chelis_fail(); }}
/* void {name}(void) {{ chelis_fail(); 0x435e0000; }} */
void {name}
(
    void
)
{{
    int64_t left;
    left = 9007199254740992;
    int64_t right;
    right = 9007199254740993;
    bool condition;
    condition
        =
        (
            left
            <
            right
        );
    if
    (
        condition
    )
    {{
        if (0) {{ helper(); }} else {{ helper(); }}
        /* 0x435e0000 belongs to no arm marker. */
        void chelis_fail(void);
        chelis_fail
        (
        );
    }}
    else
    {{
        void chelis_fail
        (
            void
        );
        const char *not_a_call = "chelis_fail(";
        unsigned value = UINT32_C(0x435e0000);
    }}
}}
"#
    );
    pick_has_expected_control(&emitted, I64_EXACT_LOW, I64_EXACT_HIGH).unwrap_or_else(|error| {
        panic!("exact-definition parser rejected valid structure: {error}")
    });
}

/// Negative-parity probe for the structural controls above: unused functions
/// deliberately supply both searched markers to the translation unit, while
/// `pick` supplies neither. Whole-file assertions would pass for the wrong
/// reason; body-scoped assertions must reject both decoys.
#[test]
fn structural_controls_ignore_unused_decoy_functions() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def decoy_fail(_flag: bool) -> f32 = fail(\"unused decoy\")\n\
                   def decoy_bits(_flag: bool) -> f32 = 222.0\n\
                   def pick() -> f32 = 111.0\n\
                   out = print(pick())\n";
    let (emitted, stdout, stderr, ok) =
        build_and_run_c(program, "fold_fail_decoys").expect("C lane");
    let emitted_markers =
        marker_counts(&c_tokens(&emitted).expect("tokenize emitted translation unit"))
            .expect("count emitted markers");
    assert!(
        emitted_markers.fail_invocations > 0,
        "probe setup requires the unused fail decoy in the translation unit"
    );
    assert!(
        emitted_markers.bits_222 > 0,
        "probe setup requires the unused 222 decoy in the translation unit"
    );

    let pick = emitted_pick_body(&emitted).expect("extract exact `pick` definition");
    let pick_tokens = c_tokens(pick).expect("tokenize `pick` body");
    let pick_markers = marker_counts(&pick_tokens).expect("count `pick` markers");
    assert!(
        pick_markers.fail_invocations == 0,
        "the unused fail decoy must not satisfy a pick-body assertion"
    );
    assert!(
        pick_markers.bits_222 == 0,
        "the unused 222 decoy must not satisfy a pick-body assertion"
    );
    assert!(ok, "the decoy probe must execute successfully: {stderr}");
    assert_eq!(stdout.lines().next(), Some("111.0"));
    assert_eq!(stderr, "");
}
