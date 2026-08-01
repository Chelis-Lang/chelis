module Std.Tests.Io.Json
import Std.Io.Json (Json, JsonNull, JsonBool, parse_json, try_parse_json, json_get, json_string, json_int, json_array, json_is_null)
import Std.Test (assert_eq_int, assert_eq_string, assert_true, assert_false, fail)
def test_parse_null_returns_json_null() -> () ! { Test } =
  match parse_json("null") with {
    | JsonNull => assert_true(true, "parse_json(\"null\") matches JsonNull")
    | _ => fail("parse_json(\"null\") did not match JsonNull")
  }
def test_parse_true_returns_json_bool_true() -> () ! { Test } =
  match parse_json("true") with {
    | JsonBool(flag) => assert_true(flag, "parse_json(\"true\") yields JsonBool(true)")
    | _ => fail("parse_json(\"true\") did not match JsonBool")
  }
def test_parse_false_returns_json_bool_false() -> () ! { Test } =
  match parse_json("false") with {
    | JsonBool(flag) => assert_false(flag, "parse_json(\"false\") yields JsonBool(false)")
    | _ => fail("parse_json(\"false\") did not match JsonBool")
  }
def test_parse_int_returns_json_int() -> () ! { Test } = {
  parsed = parse_json("42")
  match json_int(Some(parsed)) with {
    | Some(n) => assert_eq_int(n, cast(42, int64), "parse_json(\"42\") yields JsonInt(42)")
    | None => fail("parse_json(\"42\") did not yield JsonInt")
  }
}
def test_parse_string_returns_json_string() -> () ! { Test } = {
  parsed = parse_json("\"hello\"")
  match json_string(Some(parsed)) with {
    | Some(text) => assert_eq_string(text, "hello", "parse_json(\"\\\"hello\\\"\") yields JsonString(\"hello\")")
    | None => fail("parse_json(\"\\\"hello\\\"\") did not yield JsonString")
  }
}
def test_parse_array_three_ints() -> () ! { Test } = {
  parsed = parse_json("[1, 2, 3]")
  match json_array(Some(parsed)) with {
    | Some(items) => {
    _ = assert_eq_int(cast(len(items), int64), cast(3, int64), "[1, 2, 3] has length 3")
    _ = match json_int(Some(index(items, cast(0, int64)))) with {
      | Some(n) => assert_eq_int(n, cast(1, int64), "items[0] == JsonInt(1)")
      | None => fail("items[0] is not a JsonInt")
    }
    _ = match json_int(Some(index(items, cast(1, int64)))) with {
      | Some(n) => assert_eq_int(n, cast(2, int64), "items[1] == JsonInt(2)")
      | None => fail("items[1] is not a JsonInt")
    }
    match json_int(Some(index(items, cast(2, int64)))) with {
      | Some(n) => assert_eq_int(n, cast(3, int64), "items[2] == JsonInt(3)")
      | None => fail("items[2] is not a JsonInt")
    }
  }
    | None => fail("parse_json(\"[1, 2, 3]\") did not yield JsonArray")
  }
}
def test_parse_object_with_int_value() -> () ! { Test } = {
  parsed = parse_json("{\"a\": 1}")
  match json_int(json_get(parsed, "a")) with {
    | Some(n) => assert_eq_int(n, cast(1, int64), "{\"a\": 1}.a == JsonInt(1)")
    | None => fail("json_get(parse_json(\"{\\\"a\\\": 1}\"), \"a\") did not yield JsonInt")
  }
}
def test_parse_nested_object_null_matches_json_is_null() -> () ! { Test } = {
  parsed = parse_json("{\"x\": null}")
  assert_true(json_is_null(json_get(parsed, "x")), "{\"x\": null}.x is JsonNull via json_is_null")
}
def test_try_parse_json_malformed_returns_none() -> () ! { Test } =
  match try_parse_json("{not json}") with {
    | Some(_) => fail("try_parse_json(\"{not json}\") must return None")
    | None => assert_true(true, "try_parse_json on malformed input returns None")
  }
def test_try_parse_json_unterminated_string_returns_none() -> () ! { Test } =
  match try_parse_json("\"unterminated") with {
    | Some(_) => fail("try_parse_json on unterminated string must return None")
    | None => assert_true(true, "try_parse_json on unterminated string returns None")
  }
def test_try_parse_json_trailing_garbage_returns_none() -> () ! { Test } =
  match try_parse_json("42 trailing") with {
    | Some(_) => fail("try_parse_json must reject trailing garbage after a complete value")
    | None => assert_true(true, "try_parse_json on trailing garbage returns None")
  }
