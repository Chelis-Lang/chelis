//! chelis#3839 (parent chelis#1286): a function that captures a top-level
//! value, and binds a local that aliases the capture under a top-level
//! spelling, builds and agrees with `chelis eval`.
//!
//! Ownership lowering mints a captured global's body owner under the name
//! the host body reads it by, and every later local alias of that owner adds
//! its own spelling to the owner's names. The C projection recovered the
//! captured global by matching every one of those names against the retained
//! globals, so an alias spelled like a top-level value (`a = y` for the
//! captured `a`, or `b = a` beside a top-level `b`) gave two matches and
//! `chelis build` panicked with "verified capture names exactly one retained
//! global" on a program the checker accepts. The capture is now resolved by
//! the identity it was minted under.
//!
//! The oracle generates every combination of how the body reads the global
//! (match scrutinees, tuples, lists, calls, conditionals and closures), how a
//! later local shadows or aliases it (the same spelling, another top-level
//! spelling, alias chains, and two captures in one body), and whether the
//! read sits in a nested block. Each read family is one program: it must check, build, run, and
//! print exactly what `chelis eval` prints, and eval must print the value the
//! program computes. The claim is the generated set, not every program that
//! captures a global.

mod common;

use assert_cmd::Command;
use common::{build_and_run, write_file};

/// How the function body reads the captured top-level value `a`.
const READS: [(&str, &str); 10] = [
    (
        "match_pair",
        "c = match (a, a) with {\n    | (p, q) => p\n  }",
    ),
    // Two captures read by one scrutinee.
    (
        "match_both",
        "c = match (a, b) with {\n    | (p, q) => p\n  }",
    ),
    (
        "match_parameter",
        "c = match (a, x) with {\n    | (p, q) => p\n  }",
    ),
    (
        "match_nested",
        "c = match (a, (a, x)) with {\n    | (p, (q, r)) => q\n  }",
    ),
    (
        "match_option",
        "c = match Some(a) with {\n    | Some(p) => p\n    | None => x\n  }",
    ),
    ("tuple", "c = (a, a)"),
    ("list", "c = [a, a]"),
    ("call", "c = add(a, x)"),
    ("conditional", "c = if true then a else x"),
    (
        "closure",
        "f = fn (z: tensor[2, f32]) -> add(z, a)\n  c = f(x)",
    ),
];

/// What a later local does with that spelling, and the value `m` returns for
/// `x = [5, 6]` with `a = [1, 2]` and `b = [3, 4]`.
const SHADOWS: [(&str, &str, &str); 8] = [
    ("no_shadow", "y = a\n  add(y, x)", "[6.0, 8.0]"),
    // The issue's witness: a local reuses the captured global's spelling and
    // aliases the capture.
    (
        "same_name_alias",
        "y = a\n  a = y\n  add(a, x)",
        "[6.0, 8.0]",
    ),
    // An alias spelled like a different top-level value.
    (
        "other_name_alias",
        "y = a\n  b = y\n  add(b, x)",
        "[6.0, 8.0]",
    ),
    ("other_name_direct", "b = a\n  add(b, x)", "[6.0, 8.0]"),
    // An alias chain through both top-level spellings.
    (
        "alias_chain",
        "y = a\n  a = y\n  b = a\n  add(b, x)",
        "[6.0, 8.0]",
    ),
    // Two captures, each aliased under the other's spelling.
    (
        "swapped_captures",
        "y = a\n  a = b\n  b = y\n  add(a, b)",
        "[4.0, 6.0]",
    ),
    // Two captures combined, then both spellings rebound.
    (
        "rebound_captures",
        "y = add(a, b)\n  a = b\n  b = y\n  add(a, b)",
        "[7.0, 10.0]",
    ),
    // A same-spelled local that does not alias the capture.
    (
        "same_name_fresh",
        "a = add(x, x)\n  add(a, x)",
        "[15.0, 18.0]",
    ),
];

fn function(name: &str, read: &str, shadow: &str, nested: bool) -> String {
    let read = if nested {
        format!("w = {{\n    {}\n    a\n  }}", read.replace('\n', "\n  "))
    } else {
        read.to_string()
    };
    format!("def {name}(x: tensor[2, f32]) -> tensor[2, f32] = {{\n  {read}\n  {shadow}\n}}\n")
}

/// One program per read family: every shadow, flat and nested.
fn family(read: &str) -> (String, Vec<String>) {
    let mut source =
        String::from("a = to_tensor([1.0f32, 2.0f32])\nb = to_tensor([3.0f32, 4.0f32])\n");
    let mut expected = Vec::new();
    for (shadow_name, shadow, value) in SHADOWS {
        for nested in [false, true] {
            let name = format!("m_{shadow_name}_{}", if nested { "nested" } else { "flat" });
            source.push_str(&function(&name, read, shadow, nested));
            source.push_str(&format!("r_{name} = {name}(to_tensor([5.0f32, 6.0f32]))\n"));
            expected.push(format!("r_{name} = tensor(shape=[2], data={value})"));
        }
    }
    (source, expected)
}

fn chelis(args: &[&str], path: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .arg(path)
        .output()
        .expect("chelis runs")
}

#[test]
fn issue_3839_captured_globals_with_shadowing_aliases_build_and_agree_with_eval() {
    for (read_name, read) in READS {
        let (source, expected) = family(read);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("{read_name}.ch"));
        write_file(&path, &source);

        let check = chelis(&["check"], &path);
        let report = String::from_utf8_lossy(&check.stdout);
        assert!(
            check.status.success() && report.contains("\"score\": 1"),
            "the checker must accept the {read_name} family:\n{report}\n{source}"
        );

        let eval = chelis(&["eval", "--file"], &path);
        assert!(
            eval.status.success(),
            "eval failed on the {read_name} family: {}\n{source}",
            String::from_utf8_lossy(&eval.stderr)
        );
        let interpreted = String::from_utf8(eval.stdout).expect("UTF-8 stdout");
        for line in &expected {
            assert!(
                interpreted.lines().any(|printed| printed == line),
                "eval of the {read_name} family must print `{line}`:\n{interpreted}"
            );
        }

        let compiled = build_and_run(&source, read_name);
        assert_eq!(
            compiled, interpreted,
            "the {read_name} family's C output must equal eval's:\n{source}"
        );
    }
}
