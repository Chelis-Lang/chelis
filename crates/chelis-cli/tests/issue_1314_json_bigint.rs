//! Executable acceptance for chelis#1314's source-faithful JSON integer lane.
//!
//! These calls resolve through the packaged `Std.Io.Json` module. They pin
//! exact out-of-int64 ingestion, accessor refusal, canonical serialization,
//! and the loud/non-throwing serializer pair at the public CLI surface.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

#[test]
fn json_bigint_preserves_exact_tokens_and_refuses_numeric_accessors() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1314-json-bigint");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (JsonBigInt, parse_json, to_json, try_to_json, json_bigint, json_int, json_float)

positive = parse_json("9223372036854775808")
negative = parse_json("-9223372036854775809")
positive_text = json_bigint(Some(positive))
negative_text = json_bigint(Some(negative))
positive_as_int = json_int(Some(positive))
positive_as_float = json_float(Some(positive))
serialized = to_json(JsonBigInt("9223372036854775808"))
round_trip = json_bigint(Some(parse_json(serialized)))
invalid_leading_zero = try_to_json(JsonBigInt("09223372036854775808"))
invalid_in_range = try_to_json(JsonBigInt("9223372036854775807"))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "positive_text = Some(9223372036854775808)",
        ))
        .stdout(predicate::str::contains(
            "negative_text = Some(-9223372036854775809)",
        ))
        .stdout(predicate::str::contains("positive_as_int = None"))
        .stdout(predicate::str::contains("positive_as_float = None"))
        .stdout(predicate::str::contains("serialized = 9223372036854775808"))
        .stdout(predicate::str::contains(
            "round_trip = Some(9223372036854775808)",
        ))
        .stdout(predicate::str::contains("invalid_leading_zero = None"))
        .stdout(predicate::str::contains("invalid_in_range = None"));
}

#[test]
fn to_json_rejects_invalid_constructed_bigint_loudly() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1314-json-bigint-domain");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (JsonBigInt, to_json)

bad = to_json(JsonBigInt("9223372036854775807"))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "to_json failed: invalid JsonBigInt storage",
        ));
}
