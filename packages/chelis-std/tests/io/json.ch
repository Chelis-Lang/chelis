module Std.Tests.Io.Json
import Std.Io.Json (Json, JsonNull, JsonBool, JsonInt, JsonFloat, JsonString, JsonArray, JsonObject, parse_json, try_parse_json, load_json, try_load_json, to_json, try_to_json, write_json, try_write_json, json_get, json_string, json_int, json_float, json_bool, json_array, json_is_null)
import Std.Test (assert_eq_int, assert_eq_string, assert_true, assert_false, fail)
def test_parse_null_returns_json_null() -> unit ! { Test } = {
  match parse_json("null") with {
    | JsonNull => assert_true(true, "parse_json(\"null\") matches JsonNull")
    | _ => fail("parse_json(\"null\") did not match JsonNull")
  }
}
def test_parse_true_returns_json_bool_true() -> unit ! { Test } = {
  match parse_json("true") with {
    | JsonBool(flag) => assert_true(flag, "parse_json(\"true\") yields JsonBool(true)")
    | _ => fail("parse_json(\"true\") did not match JsonBool")
  }
}
def test_parse_false_returns_json_bool_false() -> unit ! { Test } = {
  match parse_json("false") with {
    | JsonBool(flag) => assert_false(flag, "parse_json(\"false\") yields JsonBool(false)")
    | _ => fail("parse_json(\"false\") did not match JsonBool")
  }
}
def test_parse_int_returns_json_int() -> unit ! { Test } = {
  parsed = parse_json("42")
  match json_int(Some(parsed)) with {
    | Some(n) => assert_eq_int(n, cast(42, int64), "parse_json(\"42\") yields JsonInt(42)")
    | None => fail("parse_json(\"42\") did not yield JsonInt")
  }
}
def test_parse_string_returns_json_string() -> unit ! { Test } = {
  parsed = parse_json("\"hello\"")
  match json_string(Some(parsed)) with {
    | Some(text) => assert_eq_string(text, "hello", "parse_json(\"\\\"hello\\\"\") yields JsonString(\"hello\")")
    | None => fail("parse_json(\"\\\"hello\\\"\") did not yield JsonString")
  }
}
def test_parse_array_three_ints() -> unit ! { Test } = {
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
def test_parse_object_with_int_value() -> unit ! { Test } = {
  parsed = parse_json("{\"a\": 1}")
  match json_int(json_get(parsed, "a")) with {
    | Some(n) => assert_eq_int(n, cast(1, int64), "{\"a\": 1}.a == JsonInt(1)")
    | None => fail("json_get(parse_json(\"{\\\"a\\\": 1}\"), \"a\") did not yield JsonInt")
  }
}
def test_parse_nested_object_null_matches_json_is_null() -> unit ! { Test } = {
  parsed = parse_json("{\"x\": null}")
  assert_true(json_is_null(json_get(parsed, "x")), "{\"x\": null}.x is JsonNull via json_is_null")
}
def test_try_parse_json_malformed_returns_none() -> unit ! { Test } = {
  match try_parse_json("{not json}") with {
    | Some(_) => fail("try_parse_json(\"{not json}\") must return None")
    | None => assert_true(true, "try_parse_json on malformed input returns None")
  }
}
def test_try_parse_json_unterminated_string_returns_none() -> unit ! { Test } = {
  match try_parse_json("\"unterminated") with {
    | Some(_) => fail("try_parse_json on unterminated string must return None")
    | None => assert_true(true, "try_parse_json on unterminated string returns None")
  }
}
def test_try_parse_json_trailing_garbage_returns_none() -> unit ! { Test } = {
  match try_parse_json("42 trailing") with {
    | Some(_) => fail("try_parse_json must reject trailing garbage after a complete value")
    | None => assert_true(true, "try_parse_json on trailing garbage returns None")
  }
}
def test_to_json_scalars() -> unit ! { Test } = {
  _ = assert_eq_string(to_json(JsonNull), "null", "to_json(JsonNull) == null")
  _ = assert_eq_string(to_json(JsonBool(true)), "true", "to_json(JsonBool(true)) == true")
  _ = assert_eq_string(to_json(JsonBool(false)), "false", "to_json(JsonBool(false)) == false")
  assert_eq_string(to_json(JsonInt(cast(42, int64))), "42", "to_json(JsonInt(42)) == 42")
}
def test_to_json_float_is_shortest_round_trip() -> unit ! { Test } = { assert_eq_string(to_json(JsonFloat(0.15110743269565682f64)), "0.15110743269565682", "17-significant-digit f64 survives to_json byte-exactly") }
def test_to_json_string_escapes_specials() -> unit ! { Test } = { assert_eq_string(to_json(JsonString("a\"b\\c\nd\te\rf")), "\"a\\\"b\\\\c\\nd\\te\\rf\"", "quote, backslash, and control whitespace are escaped") }
def test_to_json_array_and_empty_containers() -> unit ! { Test } = {
  _ = assert_eq_string(to_json(JsonArray([JsonFloat(1.5f64), JsonNull])), "[1.5,null]", "array renders compact with null")
  _ = assert_eq_string(to_json(JsonArray([])), "[]", "empty array renders []")
  assert_eq_string(to_json(JsonObject(dict_of([]))), "{}", "empty object renders {}")
}
def test_to_json_object_preserves_insertion_order() -> unit ! { Test } = {
  doc = JsonObject(dict_of([("b", JsonInt(cast(1, int64))), ("a", JsonInt(cast(2, int64)))]))
  assert_eq_string(to_json(doc), "{\"b\":1,\"a\":2}", "object keys render in insertion order")
}
def test_try_to_json_non_finite_returns_none() -> unit ! { Test } = {
  _ = match try_to_json(JsonFloat(div(0.0f64, 0.0f64))) with {
    | Some(_) => fail("try_to_json(NaN) must return None")
    | None => assert_true(true, "NaN -> None")
  }
  _ = match try_to_json(JsonFloat(div(1.0f64, 0.0f64))) with {
    | Some(_) => fail("try_to_json(inf) must return None")
    | None => assert_true(true, "inf -> None")
  }
  match try_to_json(JsonFloat(div(-1.0f64, 0.0f64))) with {
    | Some(_) => fail("try_to_json(-inf) must return None")
    | None => assert_true(true, "-inf -> None")
  }
}
def test_to_json_round_trips_through_parse_json() -> unit ! { Test } = {
  doc = JsonObject(dict_of([("cap", JsonFloat(0.15110743269565682f64)), ("name", JsonString("a\"b\\c")), ("n", JsonInt(cast(3, int64)))]))
  parsed = parse_json(to_json(doc))
  _ = match json_float(json_get(parsed, "cap")) with {
    | Some(x) => assert_true(eq(x, 0.15110743269565682f64), "float field round-trips bit-exactly")
    | None => fail("cap did not round-trip as a float")
  }
  _ = match json_string(json_get(parsed, "name")) with {
    | Some(s) => assert_eq_string(s, "a\"b\\c", "escaped string round-trips")
    | None => fail("name did not round-trip as a string")
  }
  match json_int(json_get(parsed, "n")) with {
    | Some(n) => assert_eq_int(n, cast(3, int64), "int field round-trips")
    | None => fail("n did not round-trip as an int")
  }
}
def test_try_to_json_array_with_non_finite_returns_none() -> unit ! { Test } = {
  match try_to_json(JsonArray([JsonFloat(1.0f64), JsonFloat(div(0.0f64, 0.0f64))])) with {
    | Some(_) => fail("a NaN inside an array must propagate None through try_to_json")
    | None => assert_true(true, "NaN in array -> None")
  }
}
def test_try_to_json_object_with_non_finite_returns_none() -> unit ! { Test } = {
  doc = JsonObject(dict_of([("ok", JsonFloat(1.5f64)), ("bad", JsonFloat(div(1.0f64, 0.0f64)))]))
  match try_to_json(doc) with {
    | Some(_) => fail("an inf inside an object must propagate None through try_to_json")
    | None => assert_true(true, "inf in object -> None")
  }
}
def test_try_load_json_missing_path_returns_none() -> unit ! { Test, IO } = {
  match try_load_json("/tmp/chelis_std_test_json_definitely_missing_z4w8q1.json") with {
    | Some(_) => fail("try_load_json on a missing path must be None, got Some")
    | None => assert_true(true, "missing path -> None (not a crash)")
  }
}
def test_to_json_long_string_renders_escaped() -> unit ! { Test } = {
  tail = fold(fn (acc: string, i: int64) -> string_concat(acc, "ab"), "", range(cast(0, int64), cast(2048, int64)))
  big = string_concat("a\"b\\c", tail)
  expected = string_concat("\"a\\\"b\\\\c", string_concat(tail, "\""))
  assert_eq_string(to_json(JsonString(big)), expected, "4 KiB string with escapes renders byte-exactly (parse-back of long strings is capped by the reader's recursion, chelis#953/#954 territory)")
}
def test_try_write_json_non_finite_returns_none() -> unit ! { Test, IO } = {
  doc = JsonObject(dict_of([("bad", JsonFloat(div(0.0f64, 0.0f64)))]))
  match try_write_json("/tmp/chelis_std_test_json_try_write.json", doc) with {
    | Some(_) => fail("try_write_json with a NaN field must return None")
    | None => assert_true(true, "non-finite doc -> None through the write facade")
  }
}
def test_write_json_reads_back() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_json_write.json"
  _ = write_json(path, JsonObject(dict_of([("ok", JsonBool(true))])))
  doc = load_json(path)
  match json_bool(json_get(doc, "ok")) with {
    | Some(flag) => assert_true(flag, "write_json output loads back with ok == true")
    | None => fail("write_json output did not load back")
  }
}
