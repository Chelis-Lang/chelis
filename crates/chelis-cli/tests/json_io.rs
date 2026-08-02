//! Host-lane JSON I/O acceptance (chelis#890).
//!
//! The QFBench-shaped end-to-end contract: a `.ch` program reads a JSON
//! input file, computes with tensor builtins, rounds with `round_to`,
//! assembles a nested output document, and writes it with `write_file` --
//! no host-language glue anywhere in the loop. The output must be
//! byte-stable across runs (insertion-order keys + shortest-round-trip
//! f64 formatting) and exact in f64.
//!
//! Data is task-neutral (generic instruments), per the chelis#890 plan --
//! nothing here is copied from any benchmark.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

const INPUT_JSON: &str = r#"{
  "portfolio": {"base_currency": "GEN"},
  "instruments": [
    {"id": "alpha", "notional": 1250000.0, "rate": 0.0425},
    {"id": "beta", "notional": 750000.5, "rate": 0.0375}
  ],
  "weights": [0.6, 0.4]
}
"#;

/// The full pipeline in pure chelis: parse -> dot-path accessors ->
/// tensor compute (to_tensor / mul / sum) -> round_to -> nested output
/// assembly (jdict / json_set / jnum) -> to_json -> write_file.
fn solve_source(input_path: &Path, output_path: &Path) -> String {
    let input = input_path.to_str().expect("utf8 path");
    let output = output_path.to_str().expect("utf8 path");
    format!(
        r#"doc = parse_json(read_file("{input}"))
ccy = json_str(doc, "portfolio.base_currency")
rows = json_list(doc, "instruments")
rates = map(fn (row) -> json_f64(row, "rate"), rows)
notionals = map(fn (row) -> json_f64(row, "notional"), rows)
weights = json_f64s(doc, "weights")
total_exposure = tensor_to_scalar(sum(mul(to_tensor(notionals), to_tensor(rates)), 0))
blended_rate = tensor_to_scalar(sum(mul(to_tensor(weights), to_tensor(rates)), 0))
out = jdict([("base_currency", jstr(ccy))])
out2 = json_set(out, "results.total_exposure", jnum(round_to(total_exposure, 2)))
out3 = json_set(out2, "results.blended_rate", jnum(round_to(blended_rate, 6)))
out4 = json_set(out3, "meta.instrument_count", jnum(cast(len(rows), f64)))
done = write_file("{output}", to_json(out4))
"#
    )
}

/// Expected output, computed independently in Rust with the same f64
/// operations the program performs (element-wise mul, sequential sum,
/// decimal rounding via the correctly-rounded fixed-precision formatter).
/// The whole path is f64: 750000.5 * 0.0375 = 28125.01875 survives
/// exactly (an f32 lane would quantize it to 28125.018...).
fn expected_output() -> String {
    let notionals = [1250000.0_f64, 750000.5];
    let rates = [0.0425_f64, 0.0375];
    let weights = [0.6_f64, 0.4];
    let total_exposure: f64 = notionals.iter().zip(&rates).map(|(n, r)| n * r).sum();
    let blended_rate: f64 = weights.iter().zip(&rates).map(|(w, r)| w * r).sum();
    let round =
        |x: f64, places: usize| -> f64 { format!("{x:.places$}").parse().expect("round parse") };
    format!(
        r#"{{"base_currency":"GEN","results":{{"total_exposure":{:?},"blended_rate":{:?}}},"meta":{{"instrument_count":2.0}}}}"#,
        round(total_exposure, 2),
        round(blended_rate, 6),
    )
}

#[test]
fn json_io_end_to_end_is_exact_and_byte_stable() {
    let dir = tempdir().expect("tempdir");
    let input_path = dir.path().join("input.json");
    let output_path = dir.path().join("output.json");
    let solve_path = dir.path().join("solve.ch");
    write_file(&input_path, INPUT_JSON);
    write_file(&solve_path, &solve_source(&input_path, &output_path));

    // Canonicalize through the real formatter, then run through the real
    // style gate (no CHELIS_STYLE_GATE_DISABLE): this is exactly the
    // sequence an agent runs, and the gate passing is part of the
    // acceptance.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", solve_path.to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .success();
    let first_run = fs::read(&output_path).expect("output.json written");
    assert_eq!(
        String::from_utf8(first_run.clone()).expect("utf8 output"),
        expected_output(),
        "output must match the independently-computed f64 values exactly"
    );

    // Byte-stability across runs: delete and regenerate.
    fs::remove_file(&output_path).expect("remove output");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .success();
    let second_run = fs::read(&output_path).expect("output.json rewritten");
    assert_eq!(
        first_run, second_run,
        "two identical runs must produce byte-identical output"
    );
}

/// Loud failure end-to-end: a missing key aborts the eval with a
/// diagnostic naming the builtin, the path, the segment, and the
/// available keys -- the output file is never written (the fx-forward
/// "silently missing output key" class becomes structurally loud).
#[test]
fn json_io_missing_path_fails_eval_loudly() {
    let dir = tempdir().expect("tempdir");
    let input_path = dir.path().join("input.json");
    let output_path = dir.path().join("output.json");
    let solve_path = dir.path().join("solve.ch");
    write_file(&input_path, INPUT_JSON);
    let source = format!(
        r#"doc = parse_json(read_file("{}"))
value = json_f64(doc, "portfolio.settlement_days")
done = write_file("{}", to_json(jnum(value)))
"#,
        input_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
    );
    write_file(&solve_path, &source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", solve_path.to_str().unwrap()])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("json_f64"))
        .stderr(predicate::str::contains("key `settlement_days` not found"))
        .stderr(predicate::str::contains("`base_currency`"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}

/// The JSON builtins are eval-only: `chelis build` rejects them
/// whole-program with the eval-only diagnostic (the same gate as
/// `process_run`), instead of emitting a silently-wrong compiled value.
#[test]
fn json_io_builtins_are_rejected_by_build() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &solve_path,
        "doc = parse_json(\"{\\\"a\\\": 1.5}\")\nvalue = json_f64(doc, \"a\")\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", solve_path.to_str().unwrap()])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            solve_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unsupported: builtin"));
}
