//! Regression tests for chelis#209.
//!
//! `chelis lint --check` panicked on Python files where an em dash
//! (`\u{2014}`, three UTF-8 bytes `E2 80 94`) appeared in a string literal
//! on any line other than line 1. The CLI calls `fix()` on each violation
//! to decide whether to append a `[fix]` marker, and the rule's `fix()`
//! computed replacement-span offsets by passing line-relative byte offsets
//! returned from `dash_spacing` directly into `source[...]` slices.
//!
//! For violations on line 1, line-relative == source-relative, so the bug
//! was masked. For violations deeper in the file the offsets land on
//! arbitrary positions in the source; whenever that position happens to
//! be inside a multi-byte char (commonly: a different em dash earlier in
//! the same file), the slice panics with a non-char-boundary error. When
//! it lands on an ASCII byte the rewrite silently produces wrong text.
//!
//! These tests pin:
//!   * em dash in a user-facing string literal anywhere in the file does
//!     not panic and produces a correctly-located lint diagnostic;
//!   * em dash inside a module / function / class docstring is excluded
//!     by the §8.6 docstring carve-out (multi-line docstrings too);
//!   * non-em-dash content (plain ASCII, en dash) does not fire;
//!   * UTF-8 boundary edge cases (em dash adjacent to other multi-byte
//!     characters, at start/end of the literal, mixed with CJK) do not
//!     panic and report the dash at the correct line/column.

use assert_cmd::Command;
use chelis_lint::rules::no_em_dash_in_public_strings::NoEmDashInPublicStrings;
use chelis_lint::{Context, Rule, Surface};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn py_ctx<'a>(path: &'a Path, src: &'a str) -> Context<'a> {
    Context {
        root: Path::new("/"),
        path,
        source: Some(src),
        surface: Surface::PythonSource,
    }
}

fn run_check_and_fix(src: &str) -> Vec<chelis_lint::Violation> {
    let path = PathBuf::from("test.py");
    let ctx = py_ctx(&path, src);
    let violations = NoEmDashInPublicStrings.check(&ctx);
    // Exercise fix() on every violation just like cmd_lint's
    // fix_available_for_violation path does. This is where chelis#209
    // panicked.
    for v in &violations {
        let _ = NoEmDashInPublicStrings.fix(&ctx, v);
    }
    violations
}

// --- Issue #209 exact repro ---------------------------------------------

#[test]
fn issue209_em_dash_in_module_docstring_does_not_panic() {
    // Exact minimal repro from chelis#209.
    let src = "#!/usr/bin/env python3\n\"\"\"Step 13 (H2 acceptance gate) \u{2014} download reference models.\"\"\"\n";
    let v = run_check_and_fix(src);
    // §8.6 carve-out: docstrings are excluded, so no violation.
    assert!(
        v.is_empty(),
        "module docstring em dash should be excluded by §8.6 carve-out, got {v:?}",
    );
}

#[test]
fn issue209_em_dash_in_print_after_docstring_does_not_panic() {
    // Minimal panic repro for chelis#209. The docstring opens on line
    // 2 with an em dash at source byte 55 (bytes 55..58). The print()
    // on line 5 has an em dash whose line-relative byte offset is 53,
    // so `dash_spacing` returns `after_end = 57` (line-relative) and
    // the bug treated that as a source-absolute offset, slicing
    // `source[57..]` inside the docstring em dash. The leading `x`
    // padding is calibrated to exactly that line-relative offset; do
    // not change it without re-deriving the byte arithmetic.
    let src = "#!/usr/bin/env python3\n\
\"\"\"Step 13 (H2 acceptance gate) \u{2014} narrative.\n\
\"\"\"\n\
\n\
print(\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \u{2014} y\")\n";
    let v = run_check_and_fix(src);
    // The docstring em dash is excluded; the print() em dash fires.
    assert_eq!(
        v.len(),
        1,
        "expected exactly one violation (the print) but got {v:?}",
    );
    let only = &v[0];
    assert_eq!(only.rule_id, "no-em-dash-in-public-strings");
    // The print() em dash sits on line 5 of the fixture.
    assert_eq!(only.line, Some(5), "violation should point at print line");
}

// --- Multi-line docstring positive (no violation, no panic) -------------

#[test]
fn issue209_em_dash_inside_multiline_function_docstring_excluded() {
    let src = "def f(x):\n\
    \"\"\"Compute the thing.\n\
\n\
    The detailed body explains \u{2014} narrative prose \u{2014} that must\n\
    not fire under §8.6.\n\
    \"\"\"\n\
    return x\n";
    let v = run_check_and_fix(src);
    assert!(
        v.is_empty(),
        "function docstring em dash must be excluded, got {v:?}"
    );
}

#[test]
fn issue209_em_dash_inside_multiline_class_docstring_excluded() {
    let src = "class Foo:\n\
    \"\"\"Class docstring.\n\
\n\
    Multi-line narrative prose \u{2014} this is fine under §8.6.\n\
    \"\"\"\n\
    pass\n";
    let v = run_check_and_fix(src);
    assert!(
        v.is_empty(),
        "class docstring em dash must be excluded, got {v:?}"
    );
}

// --- Positive violations at various source positions --------------------

#[test]
fn issue209_em_dash_in_print_at_arbitrary_line() {
    // 200 filler lines before the violation so the violation's
    // line_start is far from byte 0. With the bug, fix() slices
    // source at the line-relative offset (~20), producing wrong or
    // panicking results.
    let mut src = String::from("#!/usr/bin/env python3\n");
    for i in 0..200 {
        src.push_str(&format!("# filler line {i}\n"));
    }
    src.push_str("print(\"hello \u{2014} world\")\n");
    let v = run_check_and_fix(&src);
    assert_eq!(v.len(), 1, "expected exactly one violation, got {v:?}");
    assert_eq!(v[0].line, Some(202));
}

#[test]
fn issue209_multiple_em_dashes_one_per_line() {
    // Three em dashes on three different lines. Each violation must
    // independently fire and survive fix() without panicking.
    let src = "print(\"a \u{2014} b\")\nprint(\"c \u{2014} d\")\nprint(\"e \u{2014} f\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(v.len(), 3, "expected three violations, got {v:?}");
    assert_eq!(v[0].line, Some(1));
    assert_eq!(v[1].line, Some(2));
    assert_eq!(v[2].line, Some(3));
}

#[test]
fn issue209_multiple_em_dashes_one_line() {
    // Two em dashes in one print() literal. Both fire; both pass fix().
    let src = "print(\"one \u{2014} two \u{2014} three\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(v.len(), 2, "expected two violations, got {v:?}");
    assert_eq!(v[0].line, Some(1));
    assert_eq!(v[1].line, Some(1));
}

// --- UTF-8 boundary edges -----------------------------------------------

#[test]
fn issue209_em_dash_adjacent_to_other_multibyte_chars() {
    // Japanese quote brackets around the em dash. Every adjacent
    // glyph is multi-byte, so any line-vs-source confusion lands in
    // the middle of one of them.
    let src = "print(\"\u{300C}\u{2014}\u{300D}\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(v.len(), 1, "expected one violation, got {v:?}");
}

#[test]
fn issue209_em_dash_at_start_of_literal() {
    // Em dash immediately after the opening quote.
    let src = "print(\"\u{2014} trailing\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(v.len(), 1, "expected one violation, got {v:?}");
}

#[test]
fn issue209_em_dash_at_end_of_literal() {
    // Em dash immediately before the closing quote.
    let src = "print(\"leading \u{2014}\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(v.len(), 1, "expected one violation, got {v:?}");
}

#[test]
fn issue209_em_dash_in_string_with_mixed_multibyte_text() {
    // ASCII + accented Latin + CJK + em dash on a non-first line.
    let src = "x = 1\n\
y = 2\n\
print(\"caf\u{00E9} \u{2014} \u{4E2D}\u{6587} test\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(v.len(), 1, "expected one violation, got {v:?}");
    assert_eq!(v[0].line, Some(3));
}

#[test]
fn issue209_em_dash_after_earlier_em_dash_in_docstring() {
    // The chelis#209 panic shape: an excluded em dash early in the
    // file, then a flagged em dash later. The bug used the later
    // violation's line-relative offset as a source offset, and that
    // happened to land inside the earlier em dash.
    let src = "\"\"\"early prose \u{2014} excluded.\"\"\"\n\
\n\
print(\"flagged \u{2014} here\")\n";
    let v = run_check_and_fix(src);
    assert_eq!(
        v.len(),
        1,
        "expected exactly one (print) violation, got {v:?}"
    );
    assert_eq!(v[0].line, Some(3));
}

// --- Negative parity ----------------------------------------------------

#[test]
fn issue209_plain_ascii_docstring_no_violation() {
    let src = "#!/usr/bin/env python3\n\"\"\"Plain ASCII docstring with no em dash.\"\"\"\n";
    let v = run_check_and_fix(src);
    assert!(v.is_empty());
}

#[test]
fn issue209_en_dash_in_docstring_no_violation() {
    // En dash (U+2013) is not an em dash. The rule is em-dash specific.
    let src =
        "#!/usr/bin/env python3\n\"\"\"Step 13 \u{2013} download.\"\"\"\nprint(\"a \u{2013} b\")\n";
    let v = run_check_and_fix(src);
    assert!(v.is_empty(), "en dash must not fire, got {v:?}");
}

#[test]
fn issue209_em_dash_in_python_comment_no_violation() {
    let src = "x = 1\n# Python comment with em dash \u{2014} not a string\nprint(x)\n";
    let v = run_check_and_fix(src);
    assert!(
        v.is_empty(),
        "em dash inside # comment must not fire, got {v:?}"
    );
}

// --- CLI-end-to-end pin: the issue's actual `chelis lint --check ...`
//     command must exit cleanly without panicking. -----------------------

#[test]
fn issue209_cli_lint_check_does_not_panic() {
    // End-to-end CLI repro of chelis#209. The hydronnx file shape was
    // a shebang, a multi-line module docstring whose first line had
    // an em dash, then later code with em dashes in print()
    // arguments. The padding on the print line is calibrated so the
    // bug's line-vs-source offset confusion lands inside the
    // docstring's em dash and panics. Before the fix, `chelis lint
    // --check <path>` aborts with a non-char-boundary panic.
    let dir = tempdir().expect("tempdir");
    let py = dir.path().join("repro.py");
    let src = "#!/usr/bin/env python3\n\
\"\"\"Step 13 (H2 acceptance gate) \u{2014} narrative.\n\
\"\"\"\n\
\n\
print(\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \u{2014} y\")\n";
    fs::write(&py, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--check", py.to_str().unwrap()])
        .output()
        .expect("chelis lint");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "chelis lint --check must not panic, stderr={stderr}",
    );
    // The print() em dash fires (blocking ERROR). The docstring em
    // dash is excluded. lint --check exits non-zero with a
    // diagnostic but must not panic.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("no-em-dash-in-public-strings"),
        "expected the rule id in stdout, got stdout={stdout} stderr={stderr}",
    );
}
