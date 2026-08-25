//! Host-lane CSV I/O acceptance (chelis#903).
//!
//! The QFBench-shaped end-to-end contract, one step past `json_io.rs`: a
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
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

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
/// file -> column accessors -> tensor compute -> round_to -> nested JSON
/// output AND a CSV table output (to_csv), both via write_file.
fn solve_source(dir: &Path) -> String {
    let positions = dir.join("positions.csv");
    let factors = dir.join("factors.csv");
    let params = dir.join("params.json");
    let results = dir.join("results.json");
    let valued = dir.join("valued.csv");
    format!(
        r#"positions = parse_csv(read_file("{positions}"))
factors = parse_csv(read_file("{factors}"))
params = parse_json(read_file("{params}"))
qty = csv_f64s(positions, "quantity")
px = csv_f64s(positions, "mid_price")
rf = csv_f64s(factors, "risk_factor")
values = mul(to_tensor(qty), to_tensor(px))
gross = tensor_to_scalar(sum(values, 0))
risk = tensor_to_scalar(sum(mul(values, to_tensor(rf)), 0))
net = mul(gross, json_f64(params, "haircut"))
out = jdict([("base_currency", jstr(json_str(params, "base_currency")))])
out2 = json_set(out, "portfolio.gross_value", jnum(round_to(gross, 2)))
out3 = json_set(out2, "portfolio.net_value", jnum(round_to(net, 2)))
out4 = json_set(out3, "portfolio.risk_weighted", jnum(round_to(risk, 4)))
out5 = json_set(out4, "meta.positions", jnum(cast(csv_nrows(positions), f64)))
done_json = write_file("{results}", to_json(out5))
val0 = round_to(mul(csv_f64(positions, 0, "quantity"), csv_f64(positions, 0, "mid_price")), 2)
val1 = round_to(mul(csv_f64(positions, 1, "quantity"), csv_f64(positions, 1, "mid_price")), 2)
val2 = round_to(mul(csv_f64(positions, 2, "quantity"), csv_f64(positions, 2, "mid_price")), 2)
crow0 = jdict([("instrument", jstr(csv_str(positions, 0, "instrument"))), ("value", jnum(val0))])
crow1 = jdict([("instrument", jstr(csv_str(positions, 1, "instrument"))), ("value", jnum(val1))])
crow2 = jdict([("instrument", jstr(csv_str(positions, 2, "instrument"))), ("value", jnum(val2))])
table = jdict([("columns", jlist([jstr("instrument"), jstr("value")])), ("rows", jlist([crow0, crow1, crow2]))])
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
        r#"{{"base_currency":"GEN","portfolio":{{"gross_value":{:?},"net_value":{:?},"risk_weighted":{:?}}},"meta":{{"positions":3.0}}}}"#,
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
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let results_path = dir.path().join("results.json");
    let valued_path = dir.path().join("valued.csv");
    write_file(&dir.path().join("positions.csv"), POSITIONS_CSV);
    write_file(&dir.path().join("factors.csv"), FACTORS_CSV);
    write_file(&dir.path().join("params.json"), PARAMS_JSON);
    write_file(&solve_path, &solve_source(dir.path()));

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
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let output_path = dir.path().join("results.json");
    write_file(&dir.path().join("positions.csv"), POSITIONS_CSV);
    let source = format!(
        r#"positions = parse_csv(read_file("{}"))
px = csv_f64s(positions, "px")
done = write_file("{}", to_json(jnum(index(px, 0))))
"#,
        dir.path().join("positions.csv").to_str().unwrap(),
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
        .stderr(predicate::str::contains("csv_f64s"))
        .stderr(predicate::str::contains("column `px` not found"))
        .stderr(predicate::str::contains("`mid_price`"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}

/// The CSV builtins are eval-only: `chelis build` rejects them when the
/// retained compile target reaches them, instead of emitting a silently-wrong
/// compiled value. Build checks selected definitions before pruning.
#[test]
fn csv_io_builtins_are_rejected_by_build() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &solve_path,
        "c = parse_csv(\"px\\n1.5\\n\")\nvalue = csv_f64(c, 0, \"px\")\n",
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

/// Integer columns end-to-end ([05-OP-3] / [04-NUM-11]): an int64 ID
/// column above 2^53 reads exactly through `csv_ints`/`csv_int`, survives
/// `jint` -> `to_json`/`to_csv` output assembly bit-exactly, and the same
/// column read through `csv_f64s` is the *named* lossy widening -- the
/// silent-collapse class the integer accessors exist to prevent.
#[test]
fn csv_io_integer_ids_are_exact_end_to_end() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let results_path = dir.path().join("results.json");
    let ids_path = dir.path().join("ids.csv");
    write_file(
        &dir.path().join("trades.csv"),
        "trade_id,qty\n9007199254740993,250\n9007199254740995,750\n",
    );
    let source = format!(
        r#"trades = parse_csv(read_file("{trades}"))
ids = csv_ints(trades, "trade_id")
qty = csv_ints(trades, "qty")
total_qty = fold(fn (acc, q) -> add(acc, q), 0i64, qty)
first_widened = index(csv_f64s(trades, "trade_id"), 0)
out = jdict([("first_id", jint(csv_int(trades, 0, "trade_id")))])
out2 = json_set(out, "totals.qty", jint(total_qty))
out3 = json_set(out2, "lossy.first_id_f64", jnum(first_widened))
done_json = write_file("{results}", to_json(out3))
row0 = jdict([("trade_id", jint(index(ids, 0)))])
row1 = jdict([("trade_id", jint(index(ids, 1)))])
table = jdict([("columns", jlist([jstr("trade_id")])), ("rows", jlist([row0, row1]))])
done_csv = write_file("{ids_out}", to_csv(table))
"#,
        trades = dir.path().join("trades.csv").to_str().unwrap(),
        results = results_path.to_str().unwrap(),
        ids_out = ids_path.to_str().unwrap(),
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
        .success();

    let results = fs::read_to_string(&results_path).expect("results.json written");
    // 2^53 + 1 and 2^53 + 3 are unrepresentable in f64; exactness here is
    // the whole point. The widened read collapses to 2^53 -- named, and
    // visibly distinct in the same output document.
    assert_eq!(
        results,
        r#"{"first_id":9007199254740993,"totals":{"qty":1000},"lossy":{"first_id_f64":9007199254740992.0}}"#
    );
    let ids_csv = fs::read_to_string(&ids_path).expect("ids.csv written");
    assert_eq!(ids_csv, "trade_id\n9007199254740993\n9007199254740995\n");
}

/// Loud refusal end-to-end: reading a float column through the integer
/// accessors aborts the eval naming the cell text and the float-accessor
/// remedy; the output file is never written.
#[test]
fn csv_io_integer_accessor_refuses_float_cells_loudly() {
    let dir = tempdir().expect("tempdir");
    let solve_path = dir.path().join("solve.ch");
    let output_path = dir.path().join("results.json");
    write_file(&dir.path().join("trades.csv"), "trade_id,px\n12,101.5\n");
    let source = format!(
        r#"trades = parse_csv(read_file("{}"))
xs = csv_ints(trades, "px")
done = write_file("{}", to_json(jint(index(xs, 0))))
"#,
        dir.path().join("trades.csv").to_str().unwrap(),
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
        .stderr(predicate::str::contains("csv_ints"))
        .stderr(predicate::str::contains("column `px`"))
        .stderr(predicate::str::contains("not an integer"))
        .stderr(predicate::str::contains("csv_f64/csv_f64s"));
    assert!(
        !output_path.exists(),
        "a failed pipeline must not leave a partial output file"
    );
}
