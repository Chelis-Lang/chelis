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

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

const INPUT_JSON: &str = r#"{
  "portfolio": {"base_currency": "GEN"},
  "instruments": [
    {"id": "alpha", "notional": 1250000.0, "rate": 0.0425},
    {"id": "beta", "notional": 750000.5, "rate": 0.0375}
  ],
  "weights": [0.6, 0.4]
}
"#;

/// The full pipeline in pure chelis: parse -> explicit ADT accessors ->
/// tensor compute (to_tensor / mul / sum) -> round_to -> nested output
/// assembly (JsonObject / JsonFloat) -> to_json -> write_file.
fn solve_source(input_path: &Path, output_path: &Path) -> String {
    let input = input_path.to_str().expect("utf8 path");
    let output = output_path.to_str().expect("utf8 path");
    format!(
        r#"module Demo.Main
import Std.Io.Json (Json, JsonFloat, JsonObject, JsonString, json_array, json_float, json_get, json_object, json_string, parse_json, to_json)
def required_object(value: Json, key: string) -> Dict[string, Json] = match json_object(json_get(value, key)) with {{
  | Some(entries) => entries
  | None => fail(string_concat("required JSON object missing at key `", string_concat(key, "`")))
}}
def required_string(value: Json, key: string) -> string = match json_string(json_get(value, key)) with {{
  | Some(text) => text
  | None => fail(string_concat("required JSON string missing at key `", string_concat(key, "`")))
}}
def required_float(value: Json, key: string) -> f64 = match json_float(json_get(value, key)) with {{
  | Some(number) => number
  | None => fail(string_concat("required JSON number missing at key `", string_concat(key, "`")))
}}
doc = parse_json(read_file("{input}"))
portfolio = JsonObject(required_object(doc, "portfolio"))
ccy = required_string(portfolio, "base_currency")
rows = match json_array(json_get(doc, "instruments")) with {{
  | Some(items) => items
  | None => fail("required JSON array missing at key `instruments`")
}}
rates = map(fn (row: Json) -> required_float(row, "rate"), rows)
notionals = map(fn (row: Json) -> required_float(row, "notional"), rows)
weights = match json_array(json_get(doc, "weights")) with {{
  | Some(items) => map(fn (item: Json) -> match json_float(Some(item)) with {{
    | Some(number) => number
    | None => fail("weights contains a non-numeric JSON value")
  }}, items)
  | None => fail("required JSON array missing at key `weights`")
}}
total_exposure = tensor_to_scalar(sum(mul(to_tensor(notionals), to_tensor(rates)), 0))
blended_rate = tensor_to_scalar(sum(mul(to_tensor(weights), to_tensor(rates)), 0))
out = JsonObject(dict_of([
  ("base_currency", JsonString(ccy)),
  ("results", JsonObject(dict_of([
    ("total_exposure", JsonFloat(round_to(total_exposure, 2))),
    ("blended_rate", JsonFloat(round_to(blended_rate, 6)))
  ]))),
  ("meta", JsonObject(dict_of([("instrument_count", JsonFloat(cast(len(rows), f64)))])))
]))
done = write_file("{output}", to_json(out))
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
    let (_dir, reef_home, app_pkg) = make_app("json-io-end-to-end");
    let input_path = app_pkg.join("input.json");
    let output_path = app_pkg.join("output.json");
    let solve_path = app_pkg.join("src/main.ch");
    write_file(&input_path, INPUT_JSON);
    write_file(&solve_path, &solve_source(&input_path, &output_path));

    // Canonicalize through the real formatter, then run through the real
    // style gate (no CHELIS_STYLE_GATE_DISABLE): this is exactly the
    // sequence an agent runs, and the gate passing is part of the
    // acceptance.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["fmt", "--inplace", solve_path.to_str().unwrap()])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
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
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
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
    let (_dir, reef_home, app_pkg) = make_app("json-io-missing-path");
    let input_path = app_pkg.join("input.json");
    let output_path = app_pkg.join("output.json");
    let solve_path = app_pkg.join("src/main.ch");
    write_file(&input_path, INPUT_JSON);
    let source = format!(
        r#"module Demo.Main
import Std.Io.Json (JsonFloat, JsonObject, json_float, json_get, json_object, parse_json, to_json)
doc = parse_json(read_file("{}"))
portfolio = match json_object(json_get(doc, "portfolio")) with {{
  | Some(entries) => JsonObject(entries)
  | None => fail("required JSON object missing at key `portfolio`")
}}
value = match json_float(json_get(portfolio, "settlement_days")) with {{
  | Some(number) => number
  | None => fail("json_float: key `settlement_days` not found; available key: `base_currency`")
}}
done = write_file("{}", to_json(JsonFloat(value)))
"#,
        input_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
    );
    write_file(&solve_path, &source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["fmt", "--inplace", solve_path.to_str().unwrap()])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("json_float"))
        .stderr(predicate::str::contains("key `settlement_days` not found"))
        .stderr(predicate::str::contains("`base_currency`"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}
