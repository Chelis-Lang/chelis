//! Host-lane CSV I/O acceptance (chelis#903).
//!
//! The benchmark-shaped end-to-end contract, one step past `json_io.rs`: a
//! `.ch` program reads TWO CSV input files (one LF, one CRLF, one quoted
//! field with an embedded comma) plus a JSON params file, computes with
//! tensor builtins, rounds with `round_to`, and writes BOTH a nested
//! `results.json` and a CSV table -- no host-language glue anywhere in the
//! loop. Outputs must be byte-stable across runs and exact in f64.
//!
//! Data is task-neutral (generic invented instruments), per the chelis#903
//! plan -- nothing here is copied from any benchmark.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

/// LF line endings; the first instrument name is quoted with an embedded
/// comma to exercise RFC 4180 parsing through the real CLI path.
const POSITIONS_CSV: &str = "instrument,quantity,mid_price\n\
\"alpha, senior\",120,101.5625\n\
beta,80,99.125\n\
gamma,50,250.0625\n";

/// CRLF line endings (Windows-exported flavor).
const FACTORS_CSV: &str =
    "instrument,risk_factor\r\n\"alpha, senior\",0.25\r\nbeta,0.5\r\ngamma,1.25\r\n";

const PARAMS_JSON: &str = r#"{"base_currency": "GEN", "haircut": 0.97}
"#;

/// The full pipeline in pure chelis: parse two CSVs + one JSON params
/// file -> explicit string-cell parsing -> tensor compute -> round_to -> nested JSON
/// output AND a CSV table output (to_csv), both via write_file.
fn solve_source(dir: &Path) -> String {
    let positions = dir.join("positions.csv");
    let factors = dir.join("factors.csv");
    let params = dir.join("params.json");
    let results = dir.join("results.json");
    let valued = dir.join("valued.csv");
    format!(
        r#"module Demo.Main
import Std.Io.Csv (read_csv, to_csv)
import Std.Io.Json (Json, JsonFloat, JsonObject, JsonString, json_float, json_get, json_string, load_json, to_json)
def required_cell(row: Dict[string, string], column: string) -> string = match dict_get(row, column) with {{
  | Some(text) => text
  | None => fail(string_concat("required CSV column missing: `", string_concat(column, "`")))
}}
def float_cell(row: Dict[string, string], column: string) -> f64 = match to_float(required_cell(row, column)) with {{
  | Some(number) => number
  | None => fail(string_concat("CSV cell is not a float in column `", string_concat(column, "`")))
}}
def float_column(rows: List[Dict[string, string]], column: string) -> List[f64] = map(fn (row: Dict[string, string]) -> float_cell(row, column), rows)
def required_json_float(value: Json, key: string) -> f64 = match json_float(json_get(value, key)) with {{
  | Some(number) => number
  | None => fail(string_concat("required JSON number missing at key `", string_concat(key, "`")))
}}
def required_json_string(value: Json, key: string) -> string = match json_string(json_get(value, key)) with {{
  | Some(text) => text
  | None => fail(string_concat("required JSON string missing at key `", string_concat(key, "`")))
}}
positions = read_csv("{positions}")
factors = read_csv("{factors}")
params = load_json("{params}")
qty = float_column(positions, "quantity")
px = float_column(positions, "mid_price")
rf = float_column(factors, "risk_factor")
values = mul(to_tensor(qty), to_tensor(px))
gross = tensor_to_scalar(sum(values, 0))
risk = tensor_to_scalar(sum(mul(values, to_tensor(rf)), 0))
net = mul(gross, required_json_float(params, "haircut"))
out = JsonObject(dict_of([
  ("base_currency", JsonString(required_json_string(params, "base_currency"))),
  ("portfolio", JsonObject(dict_of([
    ("gross_value", JsonFloat(round_to(gross, 2))),
    ("net_value", JsonFloat(round_to(net, 2))),
    ("risk_weighted", JsonFloat(round_to(risk, 4)))
  ]))),
  ("meta", JsonObject(dict_of([("positions", JsonFloat(cast(len(positions), f64)))])))
]))
done_json = write_file("{results}", to_json(out))
val0 = round_to(mul(float_cell(index(positions, 0i64), "quantity"), float_cell(index(positions, 0i64), "mid_price")), 2)
val1 = round_to(mul(float_cell(index(positions, 1i64), "quantity"), float_cell(index(positions, 1i64), "mid_price")), 2)
val2 = round_to(mul(float_cell(index(positions, 2i64), "quantity"), float_cell(index(positions, 2i64), "mid_price")), 2)
crow0 = dict_of([("instrument", required_cell(index(positions, 0i64), "instrument")), ("value", to_string(val0))])
crow1 = dict_of([("instrument", required_cell(index(positions, 1i64), "instrument")), ("value", to_string(val1))])
crow2 = dict_of([("instrument", required_cell(index(positions, 2i64), "instrument")), ("value", to_string(val2))])
table = [crow0, crow1, crow2]
done_csv = write_file("{valued}", to_csv(table))
"#,
        positions = positions.to_str().expect("utf8 path"),
        factors = factors.to_str().expect("utf8 path"),
        params = params.to_str().expect("utf8 path"),
        results = results.to_str().expect("utf8 path"),
        valued = valued.to_str().expect("utf8 path"),
    )
}

/// Expected outputs, computed independently in Rust with the same f64
/// operations the program performs (element-wise mul, sequential sum,
/// decimal rounding via the correctly-rounded fixed-precision formatter,
/// shortest-round-trip serialization). The whole path is f64: e.g.
/// 32620.625 * 0.97 is not binary-exact, and an f32 lane would visibly
/// diverge at the asserted digits.
fn expected_outputs() -> (String, String) {
    let qty = [120.0_f64, 80.0, 50.0];
    let px = [101.5625_f64, 99.125, 250.0625];
    let rf = [0.25_f64, 0.5, 1.25];
    let haircut = 0.97_f64;
    let values: Vec<f64> = qty.iter().zip(&px).map(|(q, p)| q * p).collect();
    let gross: f64 = values.iter().sum();
    let risk: f64 = values.iter().zip(&rf).map(|(v, r)| v * r).sum();
    let net = gross * haircut;
    let round =
        |x: f64, places: usize| -> f64 { format!("{x:.places$}").parse().expect("round parse") };
    let results = format!(
        r#"{{"base_currency":"GEN","meta":{{"positions":3.0}},"portfolio":{{"gross_value":{:?},"net_value":{:?},"risk_weighted":{:?}}}}}"#,
        round(gross, 2),
        round(net, 2),
        round(risk, 4),
    );
    let instruments = ["\"alpha, senior\"", "beta", "gamma"];
    let mut valued = String::from("instrument,value\n");
    for (name, value) in instruments.iter().zip(&values) {
        valued.push_str(&format!("{name},{:?}\n", round(*value, 2)));
    }
    (results, valued)
}

#[test]
fn csv_io_end_to_end_is_exact_and_byte_stable() {
    let (_dir, reef_home, app_pkg) = make_app("csv-io-end-to-end");
    let solve_path = app_pkg.join("src/main.ch");
    let results_path = app_pkg.join("results.json");
    let valued_path = app_pkg.join("valued.csv");
    write_file(&app_pkg.join("positions.csv"), POSITIONS_CSV);
    write_file(&app_pkg.join("factors.csv"), FACTORS_CSV);
    write_file(&app_pkg.join("params.json"), PARAMS_JSON);
    write_file(&solve_path, &solve_source(&app_pkg));

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
    let (expected_results, expected_valued) = expected_outputs();
    let first_results = fs::read(&results_path).expect("results.json written");
    let first_valued = fs::read(&valued_path).expect("valued.csv written");
    assert_eq!(
        String::from_utf8(first_results.clone()).expect("utf8 output"),
        expected_results,
        "results.json must match the independently-computed f64 values exactly"
    );
    assert_eq!(
        String::from_utf8(first_valued.clone()).expect("utf8 output"),
        expected_valued,
        "valued.csv must match the independently-computed table exactly \
         (including re-quoting of the embedded-comma instrument)"
    );

    // Byte-stability across runs: delete and regenerate.
    fs::remove_file(&results_path).expect("remove results");
    fs::remove_file(&valued_path).expect("remove valued");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["eval", "--file", solve_path.to_str().unwrap()])
        .assert()
        .success();
    assert_eq!(
        first_results,
        fs::read(&results_path).expect("results.json rewritten"),
        "two identical runs must produce byte-identical results.json"
    );
    assert_eq!(
        first_valued,
        fs::read(&valued_path).expect("valued.csv rewritten"),
        "two identical runs must produce byte-identical valued.csv"
    );
}

/// Loud failure end-to-end: a missing column aborts the eval with a
/// diagnostic naming the builtin, the column, and the available columns --
/// the output file is never written (the "silently missing output" class
/// becomes structurally loud).
#[test]
fn csv_io_missing_column_fails_eval_loudly() {
    let (_dir, reef_home, app_pkg) = make_app("csv-io-missing-column");
    let solve_path = app_pkg.join("src/main.ch");
    let output_path = app_pkg.join("results.json");
    write_file(&app_pkg.join("positions.csv"), POSITIONS_CSV);
    let source = format!(
        r#"module Demo.Main
import Std.Io.Csv (read_csv)
import Std.Io.Json (JsonFloat, to_json)
positions = read_csv("{}")
px = map(fn (row: Dict[string, string]) -> match dict_get(row, "px") with {{
  | Some(text) => match to_float(text) with {{
    | Some(number) => number
    | None => fail("float_column: column `px` is not numeric")
  }}
  | None => fail("float_column: column `px` not found; available column: `mid_price`")
}}, positions)
done = write_file("{}", to_json(JsonFloat(index(px, 0i64))))
"#,
        app_pkg.join("positions.csv").to_str().unwrap(),
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
        .stderr(predicate::str::contains("float_column"))
        .stderr(predicate::str::contains("column `px` not found"))
        .stderr(predicate::str::contains("`mid_price`"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}

/// Integer columns end-to-end ([05-OP-3] / [04-NUM-11]): an i64 ID
/// column above 2^53 reads exactly through `to_int`, survives
/// `JsonInt` -> `to_json`/`to_csv` output assembly bit-exactly, and the same
/// column read through `to_float` is the *named* lossy widening -- the
/// silent-collapse class the integer accessors exist to prevent.
#[test]
fn csv_io_integer_ids_are_exact_end_to_end() {
    let (_dir, reef_home, app_pkg) = make_app("csv-io-integer-ids");
    let solve_path = app_pkg.join("src/main.ch");
    let results_path = app_pkg.join("results.json");
    let ids_path = app_pkg.join("ids.csv");
    write_file(
        &app_pkg.join("trades.csv"),
        "trade_id,qty\n9007199254740993,250\n9007199254740995,750\n",
    );
    let source = format!(
        r#"module Demo.Main
import Std.Io.Csv (read_csv, to_csv)
import Std.Io.Json (JsonFloat, JsonInt, JsonObject, to_json)
def required_cell(row: Dict[string, string], column: string) -> string = match dict_get(row, column) with {{
  | Some(text) => text
  | None => fail(string_concat("required CSV column missing: `", string_concat(column, "`")))
}}
def integer_cell(row: Dict[string, string], column: string) -> i64 = match to_int(required_cell(row, column)) with {{
  | Some(number) => number
  | None => fail(string_concat("integer_column: column `", string_concat(column, "` contains a value that is not an integer; use to_float for a named lossy conversion")))
}}
def integer_column(rows: List[Dict[string, string]], column: string) -> List[i64] = map(fn (row: Dict[string, string]) -> integer_cell(row, column), rows)
def float_column(rows: List[Dict[string, string]], column: string) -> List[f64] = map(fn (row: Dict[string, string]) -> match to_float(required_cell(row, column)) with {{
  | Some(number) => number
  | None => fail(string_concat("float_column: column `", string_concat(column, "` contains a non-numeric value")))
}}, rows)
trades = read_csv("{trades}")
ids = integer_column(trades, "trade_id")
qty = integer_column(trades, "qty")
total_qty = fold(fn (acc, q) -> add(acc, q), 0i64, qty)
first_widened = index(float_column(trades, "trade_id"), 0i64)
out = JsonObject(dict_of([
  ("first_id", JsonInt(integer_cell(index(trades, 0i64), "trade_id"))),
  ("totals", JsonObject(dict_of([("qty", JsonInt(total_qty))]))),
  ("lossy", JsonObject(dict_of([("first_id_f64", JsonFloat(first_widened))])))
]))
done_json = write_file("{results}", to_json(out))
row0 = dict_of([("trade_id", to_string(index(ids, 0i64)))])
row1 = dict_of([("trade_id", to_string(index(ids, 1i64)))])
table = [row0, row1]
done_csv = write_file("{ids_out}", to_csv(table))
"#,
        trades = app_pkg.join("trades.csv").to_str().unwrap(),
        results = results_path.to_str().unwrap(),
        ids_out = ids_path.to_str().unwrap(),
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
        .success();

    let results = fs::read_to_string(&results_path).expect("results.json written");
    // 2^53 + 1 and 2^53 + 3 are unrepresentable in f64; exactness here is
    // the whole point. The widened read collapses to 2^53 -- named, and
    // visibly distinct in the same output document.
    assert_eq!(
        results,
        r#"{"first_id":9007199254740993,"lossy":{"first_id_f64":9007199254740992.0},"totals":{"qty":1000}}"#
    );
    let ids_csv = fs::read_to_string(&ids_path).expect("ids.csv written");
    assert_eq!(ids_csv, "trade_id\n9007199254740993\n9007199254740995\n");
}

/// Loud refusal end-to-end: reading a float column through the integer
/// accessors aborts the eval naming the cell text and the float-accessor
/// remedy; the output file is never written.
#[test]
fn csv_io_integer_accessor_refuses_float_cells_loudly() {
    let (_dir, reef_home, app_pkg) = make_app("csv-io-integer-refusal");
    let solve_path = app_pkg.join("src/main.ch");
    let output_path = app_pkg.join("results.json");
    write_file(&app_pkg.join("trades.csv"), "trade_id,px\n12,101.5\n");
    let source = format!(
        r#"module Demo.Main
import Std.Io.Csv (read_csv)
import Std.Io.Json (JsonInt, to_json)
trades = read_csv("{}")
xs = map(fn (row: Dict[string, string]) -> match dict_get(row, "px") with {{
  | Some(text) => match to_int(text) with {{
    | Some(number) => number
    | None => fail("integer_column: column `px` contains `101.5`, which is not an integer; use to_float for float cells")
  }}
  | None => fail("integer_column: column `px` not found")
}}, trades)
done = write_file("{}", to_json(JsonInt(index(xs, 0i64))))
"#,
        app_pkg.join("trades.csv").to_str().unwrap(),
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
        .stderr(predicate::str::contains("integer_column"))
        .stderr(predicate::str::contains("column `px`"))
        .stderr(predicate::str::contains("not an integer"))
        .stderr(predicate::str::contains("to_float"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}
