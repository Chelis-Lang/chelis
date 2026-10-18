//! A list literal of float literals emits as one static table (chelis#3861).
//!
//! Element-by-element emission gave every element two locals, an origin
//! slot and a box statement in one basic block, and C compilers spend
//! superlinear time selecting instructions for such a block: a 4,096-element
//! `f16` literal took 23 s under `clang -O2`. The host emitter now writes the
//! elements' tagged scalars into a function-local `static const` table and
//! boxes and releases them in two loops, so the emitted statements no longer
//! grow with the literal. These tests count those statements and compare
//! the built program's output with `chelis eval`, widened to `f64` so the
//! printed form names every bit.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

const DTYPES: [&str; 4] = ["f16", "bf16", "f32", "f64"];

fn literal_list(dtype: &str, values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value}{dtype}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("x = to_tensor([{items}])\nxs = [{items}]\n")
}

fn chelis(dir: &Path, args: &[&str]) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("run chelis");
    assert!(
        output.status.success(),
        "chelis {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The emitted C for `source`.
fn emitted_c(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("lit.ch"), source).expect("write source");
    chelis(dir.path(), &["build", "--emit-c", "-o", "out", "lit.ch"]);
    fs::read_to_string(dir.path().join("out/lit.c")).expect("read emitted C")
}

/// The statements of the emitted C: lines that end a C statement, which a
/// table entry and a comment never do.
fn statements(c: &str) -> usize {
    c.lines()
        .map(str::trim)
        .filter(|line| line.ends_with(';') && !line.starts_with("//") && !line.starts_with("/*"))
        .count()
}

/// `eval --file` output, and the output of the program `build` compiled.
fn eval_and_run(source: &str) -> (String, String) {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("lit.ch"), source).expect("write source");
    let eval = chelis(dir.path(), &["eval", "--file", "lit.ch"]);
    chelis(dir.path(), &["build", "-o", "out", "lit.ch"]);
    let run = StdCommand::new(dir.path().join("out/lit"))
        .output()
        .expect("run the built program");
    assert!(
        run.status.success(),
        "the built program failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    (eval, String::from_utf8_lossy(&run.stdout).into_owned())
}

#[test]
fn a_float_literal_list_emits_the_same_statements_at_every_length() {
    for dtype in DTYPES {
        let values = |count: usize| {
            (0..count)
                .map(|index| format!("{}.125", index % 97))
                .collect::<Vec<_>>()
        };
        let short = emitted_c(&literal_list(dtype, &values(8)));
        let long = emitted_c(&literal_list(dtype, &values(512)));
        assert_eq!(
            statements(&short),
            statements(&long),
            "{dtype}: the emitted statements grow with the literal"
        );
        for (c, count) in [(&short, 8), (&long, 512)] {
            // Two literals, each boxed in one loop.
            assert_eq!(c.matches("chelis_value_box_scalar(").count(), 2, "{dtype}");
            assert_eq!(
                c.matches("static const chelis_scalar").count(),
                2,
                "{dtype}"
            );
            assert_eq!(
                c.matches(".reserved = {0}, .bits = ").count(),
                2 * count,
                "{dtype}"
            );
        }
    }
}

#[test]
fn table_literals_print_what_eval_prints() {
    let special: [(&str, &[&str]); 4] = [
        (
            "f16",
            &[
                "65504.0",
                "65519.0",
                "0.000000059604645",
                "0.000000029802322",
                "2049.0",
                "2051.0",
                "0.1",
                "0.0",
            ],
        ),
        (
            "bf16",
            &[
                "3.3895313892515355e38",
                "9.183549615799121e-41",
                "1e-40",
                "257.0",
                "259.0",
                "0.1",
                "0.0",
            ],
        ),
        (
            "f32",
            &[
                "3.4028234663852886e38",
                "1.401298464324817e-45",
                "7e-46",
                "16777217.0",
                "0.1",
                "0.0",
            ],
        ),
        (
            "f64",
            &[
                "1.7976931348623157e308",
                "5e-324",
                "9007199254740993.0",
                "0.30000000000000004",
                "0.0",
            ],
        ),
    ];
    for (dtype, values) in special {
        let values = values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        // Widening to f64 is exact, so its printed form names every bit.
        let source = format!(
            "{}wide = cast(x, f64)\nout = print(wide)\nout2 = print(xs)\n",
            literal_list(dtype, &values)
        );
        assert!(
            emitted_c(&source).contains("static const chelis_scalar"),
            "{dtype}"
        );
        let (eval, run) = eval_and_run(&source);
        assert_eq!(run, eval, "{dtype}: the built program and eval disagree");
    }
}

#[test]
fn a_list_with_a_computed_element_is_emitted_element_by_element() {
    // A negated literal is a runtime `neg`, not a folded literal, so the
    // list keeps one box per element, and still prints what eval prints.
    let source = "x = to_tensor([0.5f16, -2.0f16, 3.0f16])\nout = print(x)\n";
    let c = emitted_c(source);
    assert!(!c.contains("static const chelis_scalar"));
    assert_eq!(c.matches("chelis_value_box_scalar(").count(), 3);
    let (eval, run) = eval_and_run(source);
    assert_eq!(run, eval);
}
