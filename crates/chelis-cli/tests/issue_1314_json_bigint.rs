//! Executable acceptance for chelis#1314's source-faithful JSON integer lane.
//!
//! These calls resolve through the packaged `Std.Io.Json` module. They pin
//! exact out-of-int64 ingestion, accessor refusal, canonical serialization,
//! and the loud/non-throwing serializer pair at the public CLI surface.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, make_app, write_file};

fn eval_app_stdout(reef_home: &std::path::Path, app_pkg: &std::path::Path) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("eval should run");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

#[test]
fn json_bigint_preserves_exact_tokens_and_refuses_numeric_accessors() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1314-json-bigint");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (JsonBigInt, parse_json, try_parse_json, to_json, try_to_json, json_bigint, json_int, json_float)

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
max_finite = try_parse_json("1.7976931348623157e308")
positive_float_overflow = try_parse_json("1.7976931348623159e308")
negative_float_overflow = try_parse_json("-1.7976931348623159e308")
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
        .stdout(predicate::str::contains("invalid_in_range = None"))
        .stdout(predicate::str::contains(
            "max_finite = Some(JsonFloat(1.7976931348623157e308))",
        ))
        .stdout(predicate::str::contains("positive_float_overflow = None"))
        .stdout(predicate::str::contains("negative_float_overflow = None"));
}

#[test]
fn parse_json_rejects_float_overflow_loudly() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1314-json-float-overflow");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (parse_json)

bad = parse_json("1e400")
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
            "parse_json failed: malformed JSON",
        ));
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

#[test]
fn json_object_serialization_is_recursive_canonical_unicode_order_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1314-json-object-order");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (JsonNull, JsonInt, JsonFloat, JsonObject, parse_json, to_json)

ba = to_json(JsonObject(dict_of([
  ("b", JsonInt(cast(1, int64))),
  ("a", JsonFloat(2.0f64))
])))
ab = to_json(JsonObject(dict_of([
  ("a", JsonFloat(2.0f64)),
  ("b", JsonInt(cast(1, int64)))
])))
equal_mappings = eq(ba, ab)
empty_object = to_json(parse_json("{}"))
nested = to_json(JsonObject(dict_of([
  ("outer", JsonObject(dict_of([
    ("z", JsonInt(cast(3, int64))),
    ("m", JsonInt(cast(4, int64)))
  ]))),
  ("a", JsonNull)
])))
unicode_and_escaped = to_json(JsonObject(dict_of([
  ("😀", JsonInt(cast(4, int64))),
  ("é", JsonInt(cast(3, int64))),
  ("a\\", JsonInt(cast(2, int64))),
  ("a\"", JsonInt(cast(1, int64)))
])))
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "ba = {\"a\":2.0,\"b\":1}",
        "ab = {\"a\":2.0,\"b\":1}",
        "equal_mappings = true",
        "empty_object = {}",
        "nested = {\"a\":null,\"outer\":{\"m\":4,\"z\":3}}",
        "unicode_and_escaped = {\"a\\\"\":1,\"a\\\\\":2,\"é\":3,\"😀\":4}",
    ] {
        assert!(
            eval.contains(expected),
            "eval missing `{expected}`:\n{eval}"
        );
        assert!(
            compiled.contains(expected),
            "compiled C missing `{expected}`:\n{compiled}"
        );
    }
}

/// PR #1302 red-team finding P0-1: every earlier bigint *ingestion* test
/// ran the eval lane only, so compiled `parse_json` of any out-of-int64
/// integer token heap-corrupted (a block-close release of the
/// parameter-aliasing `digits` binding inside `canonical_bigint_text`
/// freed the caller's string) while the suite stayed green. This drives
/// the full ingestion boundary through BOTH lanes and requires identical
/// output: MAX/MIN stay exact ints, MAX+1/MIN-1 and a 60-digit token
/// round-trip verbatim as bigints, the `json_bigint` accessor refuses an
/// in-range int, leading zeros are refused, `-0` collapses to `0`, and a
/// constructed in-range `JsonBigInt` refuses to serialize.
#[test]
fn bigint_ingestion_round_trips_identically_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1302-bigint-ingestion");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (Json, JsonBigInt, parse_json, try_parse_json, to_json, try_to_json, json_bigint)

def opt_text(value: Option[string], fallback: string) -> string =
  match value with {
    | Some(text) => text
    | None => fallback
  }

def parse_refused(text: string) -> string =
  match try_parse_json(text) with {
    | Some(_) => "PARSED"
    | None => "REFUSED"
  }

over_max = to_json(parse_json("9223372036854775808"))
under_min = to_json(parse_json("-9223372036854775809"))
at_max = to_json(parse_json("9223372036854775807"))
at_min = to_json(parse_json("-9223372036854775808"))
wide = to_json(parse_json("123456789012345678901234567890123456789012345678901234567890"))
accessor = opt_text(json_bigint(Some(parse_json("123456789012345678901234567890"))), "NONE")
accessor_refuses_int = opt_text(json_bigint(Some(parse_json("7"))), "NONE")
leading_zero = parse_refused("-012")
negative_zero = to_json(parse_json("-0"))
in_range_ctor = opt_text(try_to_json(JsonBigInt("42")), "REFUSED")
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "over_max = 9223372036854775808",
        "under_min = -9223372036854775809",
        "at_max = 9223372036854775807",
        "at_min = -9223372036854775808",
        "wide = 123456789012345678901234567890123456789012345678901234567890",
        "accessor = 123456789012345678901234567890",
        "accessor_refuses_int = NONE",
        "leading_zero = REFUSED",
        "negative_zero = 0",
        "in_range_ctor = REFUSED",
    ] {
        assert!(
            eval.contains(expected),
            "eval missing `{expected}`:\n{eval}"
        );
        assert!(
            compiled.contains(expected),
            "compiled C missing `{expected}`:\n{compiled}"
        );
    }
}
