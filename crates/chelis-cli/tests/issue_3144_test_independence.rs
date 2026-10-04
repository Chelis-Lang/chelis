//! chelis#3144: `chelis test` evaluates every test of a file against one
//! prepared program and shares the program's derived facts among them, but
//! never a test's state. A test's keys, effects, output and failure are its
//! own: each test reports in the full file exactly what it reports alone.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{make_app, write_file};

/// Test A draws from a keyed stream, prints and fails. B and C pass after
/// it, and C draws from the same seed as A. If A's draw advanced any shared
/// stream, C would see a different value.
const TESTS: &str = "module Demo.Tests.Independence\n\
banner = print(\"module init\")\n\
def draw(seed: i64) -> f32 = tensor_to_scalar(sum(dropout(key_from_seed(seed), \
to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]), 0.5f32), 0i32))\n\
def test_a_fails_first() -> unit ! { Test, IO } = {\n\
  _ = print(\"a\")\n\
  _ = draw(7i64)\n\
  test_assert(false, \"a fails\")\n\
}\n\
def test_b_passes_after_a_failure() -> unit ! { Test, IO } = {\n\
  _ = print(\"b\")\n\
  test_assert_eq(draw(7i64), draw(7i64), \"one seed, one stream\")\n\
}\n\
def test_c_sees_the_same_stream() -> unit ! { Test, IO } = {\n\
  _ = print(\"c\")\n\
  test_assert_eq(draw(7i64), 30.0f32, \"the same seed draws the same stream\")\n\
}\n";

const NAMES: [&str; 3] = [
    "test_a_fails_first",
    "test_b_passes_after_a_failure",
    "test_c_sees_the_same_stream",
];

/// The JSON records `chelis test` prints for the file, without the summary.
fn records(filter: Option<&str>) -> Vec<String> {
    let (_dir, reef_home, app_pkg) = make_app("independence-3144");
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    write_file(&app_pkg.join("tests/independence.ch"), TESTS);
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(&app_pkg)
        .args(["test", "--batch-mode", "file", "--json"]);
    if let Some(filter) = filter {
        command.args(["--filter", filter]);
    }
    let output = command
        .arg("tests/independence.ch")
        .output()
        .expect("chelis test runs");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.starts_with("{\"summary\""))
        .map(str::to_string)
        .collect()
}

#[test]
fn each_test_of_a_file_reports_what_it_reports_alone() {
    let together = records(None);
    assert_eq!(
        together,
        vec![
            "{\"file\":\"tests/independence.ch\",\"test\":\"test_a_fails_first\",\"status\":\"fail\",\"message\":\"assert failed: a fails\"}",
            "{\"file\":\"tests/independence.ch\",\"test\":\"test_b_passes_after_a_failure\",\"status\":\"pass\"}",
            "{\"file\":\"tests/independence.ch\",\"test\":\"test_c_sees_the_same_stream\",\"status\":\"pass\"}",
        ],
        "A fails, and B and C still pass after it"
    );
    let alone: Vec<String> = NAMES.iter().flat_map(|name| records(Some(name))).collect();
    assert_eq!(together, alone, "no test's state reaches another test");
}
