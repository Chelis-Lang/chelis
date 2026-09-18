module Std.Tests.Io.Json
import Std.Io.Json (Json, JsonNull, JsonBool, JsonInt, JsonBigInt, JsonFloat, JsonString, JsonArray, JsonObject, parse_json, try_parse_json, load_json, try_load_json, to_json, try_to_json, write_json, try_write_json, json_get, json_string, json_int, json_bigint, json_float, json_bool, json_array, json_is_null)
import Std.Test (assert_eq, assert_true, assert_false, fail)
def test_parse_null_returns_json_null() -> unit ! { Test } =
  match parse_json("null") with {
    | JsonNull => assert_true(true, "parse_json(\"null\") matches JsonNull")
    | _ => fail("parse_json(\"null\") did not match JsonNull")
  }
def test_parse_true_returns_json_bool_true() -> unit ! { Test } =
  match parse_json("true") with {
    | JsonBool(flag) => assert_true(flag, "parse_json(\"true\") yields JsonBool(true)")
    | _ => fail("parse_json(\"true\") did not match JsonBool")
  }
def test_parse_false_returns_json_bool_false() -> unit ! { Test } =
  match parse_json("false") with {
    | JsonBool(flag) => assert_false(flag, "parse_json(\"false\") yields JsonBool(false)")
    | _ => fail("parse_json(\"false\") did not match JsonBool")
  }
def test_parse_int_returns_json_int() -> unit ! { Test } = {
  parsed = parse_json("42")
  match json_int(Some(parsed)) with {
    | Some(n) => assert_eq(n, cast(42, i64), "parse_json(\"42\") yields JsonInt(42)")
    | None => fail("parse_json(\"42\") did not yield JsonInt")
  }
}
def test_parse_out_of_int64_integer_preserves_exact_bigint_spelling() -> unit ! { Test } = {
  positive = parse_json("9223372036854775808")
  negative = parse_json("-9223372036854775809")
  _ = match json_bigint(Some(positive)) with {
    | Some(text) => assert_eq(text, "9223372036854775808", "positive out-of-int64 token is byte-exact")
    | None => fail("positive out-of-int64 token did not yield JsonBigInt")
  }
  match json_bigint(Some(negative)) with {
    | Some(text) => assert_eq(text, "-9223372036854775809", "negative out-of-int64 token is byte-exact")
    | None => fail("negative out-of-int64 token did not yield JsonBigInt")
  }
}
def test_bigint_accessors_refuse_cross_variant_coercion() -> unit ! { Test } = {
  value = Some(JsonBigInt("9223372036854775808"))
  _ = match json_int(value) with {
    | Some(_) => fail("json_int must refuse JsonBigInt")
    | None => assert_true(true, "json_int(JsonBigInt) -> None")
  }
  _ = match json_float(value) with {
    | Some(_) => fail("json_float must refuse JsonBigInt")
    | None => assert_true(true, "json_float(JsonBigInt) -> None")
  }
  match json_bigint(Some(JsonInt(cast(1, i64)))) with {
    | Some(_) => fail("json_bigint must refuse JsonInt")
    | None => assert_true(true, "json_bigint(JsonInt) -> None")
  }
}
def test_parse_string_returns_json_string() -> unit ! { Test } = {
  parsed = parse_json("\"hello\"")
  match json_string(Some(parsed)) with {
    | Some(text) => assert_eq(text, "hello", "parse_json(\"\\\"hello\\\"\") yields JsonString(\"hello\")")
    | None => fail("parse_json(\"\\\"hello\\\"\") did not yield JsonString")
  }
}
def test_parse_unicode_escapes_and_surrogate_pairs() -> unit ! { Test } = {
  parsed = parse_json("\"\\u00e9\\ud83d\\ude00\"")
  match json_string(Some(parsed)) with {
    | Some(text) => assert_eq(text, "é😀", "BMP and surrogate-pair escapes decode to Unicode scalars")
    | None => fail("unicode escape input did not yield JsonString")
  }
}
def test_try_parse_json_rejects_malformed_unicode_escapes_and_raw_controls() -> unit ! { Test } = {
  invalid = ["\"\\u12x4\"", "\"\\ud83d\"", "\"\\ude00\"", "\"\\ud83d\\u0041\"", "\"a\u{b}b\""]
  fold(fn (acc: unit, text: string) -> match try_parse_json(text) with {
    | Some(_) => fail(string_concat("try_parse_json must reject malformed Unicode/control input: ", text))
    | None => assert_true(true, "malformed Unicode/control input rejected")
  }, (), invalid)
}
def test_parse_array_three_ints() -> unit ! { Test } = {
  parsed = parse_json("[1, 2, 3]")
  match json_array(Some(parsed)) with {
    | Some(items) => {
    _ = assert_eq(cast(len(items), i64), cast(3, i64), "[1, 2, 3] has length 3")
    _ = match json_int(Some(index(items, cast(0, i64)))) with {
      | Some(n) => assert_eq(n, cast(1, i64), "items[0] == JsonInt(1)")
      | None => fail("items[0] is not a JsonInt")
    }
    _ = match json_int(Some(index(items, cast(1, i64)))) with {
      | Some(n) => assert_eq(n, cast(2, i64), "items[1] == JsonInt(2)")
      | None => fail("items[1] is not a JsonInt")
    }
    match json_int(Some(index(items, cast(2, i64)))) with {
      | Some(n) => assert_eq(n, cast(3, i64), "items[2] == JsonInt(3)")
      | None => fail("items[2] is not a JsonInt")
    }
  }
    | None => fail("parse_json(\"[1, 2, 3]\") did not yield JsonArray")
  }
}
def test_parse_object_with_int_value() -> unit ! { Test } = {
  parsed = parse_json("{\"a\": 1}")
  match json_int(json_get(parsed, "a")) with {
    | Some(n) => assert_eq(n, cast(1, i64), "{\"a\": 1}.a == JsonInt(1)")
    | None => fail("json_get(parse_json(\"{\\\"a\\\": 1}\"), \"a\") did not yield JsonInt")
  }
}
def test_parse_nested_object_null_matches_json_is_null() -> unit ! { Test } = {
  parsed = parse_json("{\"x\": null}")
  assert_true(json_is_null(json_get(parsed, "x")), "{\"x\": null}.x is JsonNull via json_is_null")
}
def test_try_parse_json_malformed_returns_none() -> unit ! { Test } =
  match try_parse_json("{not json}") with {
    | Some(_) => fail("try_parse_json(\"{not json}\") must return None")
    | None => assert_true(true, "try_parse_json on malformed input returns None")
  }
def test_try_parse_json_unterminated_string_returns_none() -> unit ! { Test } =
  match try_parse_json("\"unterminated") with {
    | Some(_) => fail("try_parse_json on unterminated string must return None")
    | None => assert_true(true, "try_parse_json on unterminated string returns None")
  }
def test_try_parse_json_trailing_garbage_returns_none() -> unit ! { Test } =
  match try_parse_json("42 trailing") with {
    | Some(_) => fail("try_parse_json must reject trailing garbage after a complete value")
    | None => assert_true(true, "try_parse_json on trailing garbage returns None")
  }
def test_to_json_scalars() -> unit ! { Test } = {
  _ = assert_eq(to_json(JsonNull), "null", "to_json(JsonNull) == null")
  _ = assert_eq(to_json(JsonBool(true)), "true", "to_json(JsonBool(true)) == true")
  _ = assert_eq(to_json(JsonBool(false)), "false", "to_json(JsonBool(false)) == false")
  assert_eq(to_json(JsonInt(cast(42, i64))), "42", "to_json(JsonInt(42)) == 42")
}
def test_to_json_canonical_bigint_is_verbatim_and_round_trips_as_bigint() -> unit ! { Test } = {
  positive = JsonBigInt("9223372036854775808")
  negative = JsonBigInt("-9223372036854775809")
  _ = assert_eq(to_json(positive), "9223372036854775808", "canonical positive JsonBigInt emits verbatim")
  _ = assert_eq(to_json(negative), "-9223372036854775809", "canonical negative JsonBigInt emits verbatim")
  match parse_json(to_json(positive)) with {
    | JsonBigInt(text) => assert_eq(text, "9223372036854775808", "JsonBigInt round-trips with its variant and bytes intact")
    | _ => fail("canonical JsonBigInt did not round-trip as JsonBigInt")
  }
}
def test_try_to_json_rejects_noncanonical_or_in_range_bigint_storage() -> unit ! { Test } = {
  invalid = ["", "-", "0", "-0", "01", "-01", "+9223372036854775808", "9223372036854775807", "-9223372036854775808", "1.0"]
  fold(fn (acc: unit, text: string) -> match try_to_json(JsonBigInt(text)) with {
    | Some(_) => fail(string_concat("try_to_json must reject invalid JsonBigInt storage: ", text))
    | None => assert_true(true, string_concat("invalid JsonBigInt rejected: ", text))
  }, (), invalid)
}
def test_to_json_float_is_shortest_round_trip() -> unit ! { Test } = assert_eq(to_json(JsonFloat(0.15110743269565682f64)), "0.15110743269565682", "17-significant-digit f64 survives to_json byte-exactly")
def test_to_json_string_escapes_specials() -> unit ! { Test } = assert_eq(to_json(JsonString("a\"b\\c\nd\te\rf")), "\"a\\\"b\\\\c\\nd\\te\\rf\"", "quote, backslash, and control whitespace are escaped")
def test_to_json_string_escapes_every_c0_control() -> unit ! { Test } = assert_eq(to_json(JsonString("\0\u{8}\u{b}\u{c}\u{1f}")), "\"\\u0000\\b\\u000b\\f\\u001f\"", "every C0 control is emitted as RFC-valid JSON")
def test_to_json_array_and_empty_containers() -> unit ! { Test } = {
  _ = assert_eq(to_json(JsonArray([JsonFloat(1.5f64), JsonNull])), "[1.5,null]", "array renders compact with null")
  _ = assert_eq(to_json(JsonArray([])), "[]", "empty array renders []")
  assert_eq(to_json(JsonObject(dict_of([]))), "{}", "empty object renders {}")
}
def test_to_json_object_uses_recursive_canonical_unicode_key_order() -> unit ! { Test } = {
  ba = JsonObject(dict_of([("b", JsonInt(cast(1, i64))), ("a", JsonFloat(2.0f64))]))
  ab = JsonObject(dict_of([("a", JsonFloat(2.0f64)), ("b", JsonInt(cast(1, i64)))]))
  ba_bytes = to_json(ba)
  ab_bytes = to_json(ab)
  _ = assert_eq(ba_bytes, "{\"a\":2.0,\"b\":1}", "canonical order is independent of insertion history and preserves JsonFloat/JsonInt spelling")
  _ = assert_eq(ba_bytes, ab_bytes, "equal mappings serialize to identical bytes")
  nested = JsonObject(dict_of([("outer", JsonObject(dict_of([("z", JsonInt(cast(3, i64))), ("m", JsonInt(cast(4, i64)))]))), ("a", JsonNull)]))
  _ = assert_eq(to_json(nested), "{\"a\":null,\"outer\":{\"m\":4,\"z\":3}}", "canonical key ordering applies recursively")
  unicode_and_escaped = JsonObject(dict_of([("😀", JsonInt(cast(4, i64))), ("é", JsonInt(cast(3, i64))), ("a\\", JsonInt(cast(2, i64))), ("a\"", JsonInt(cast(1, i64)))]))
  assert_eq(to_json(unicode_and_escaped), "{\"a\\\"\":1,\"a\\\\\":2,\"é\":3,\"😀\":4}", "keys order by Unicode scalar values before JSON escaping")
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
  doc = JsonObject(dict_of([("cap", JsonFloat(0.15110743269565682f64)), ("name", JsonString("a\"b\\c")), ("n", JsonInt(cast(3, i64)))]))
  parsed = parse_json(to_json(doc))
  _ = match json_float(json_get(parsed, "cap")) with {
    | Some(x) => assert_true(eq(x, 0.15110743269565682f64), "float field round-trips bit-exactly")
    | None => fail("cap did not round-trip as a float")
  }
  _ = match json_string(json_get(parsed, "name")) with {
    | Some(s) => assert_eq(s, "a\"b\\c", "escaped string round-trips")
    | None => fail("name did not round-trip as a string")
  }
  match json_int(json_get(parsed, "n")) with {
    | Some(n) => assert_eq(n, cast(3, i64), "int field round-trips")
    | None => fail("n did not round-trip as an int")
  }
}
def test_try_to_json_array_with_non_finite_returns_none() -> unit ! { Test } =
  match try_to_json(JsonArray([JsonFloat(1.0f64), JsonFloat(div(0.0f64, 0.0f64))])) with {
    | Some(_) => fail("a NaN inside an array must propagate None through try_to_json")
    | None => assert_true(true, "NaN in array -> None")
  }
def test_try_to_json_object_with_non_finite_returns_none() -> unit ! { Test } = {
  doc = JsonObject(dict_of([("ok", JsonFloat(1.5f64)), ("bad", JsonFloat(div(1.0f64, 0.0f64)))]))
  match try_to_json(doc) with {
    | Some(_) => fail("an inf inside an object must propagate None through try_to_json")
    | None => assert_true(true, "inf in object -> None")
  }
}
def test_try_load_json_missing_path_returns_none() -> unit ! { Test, IO } =
  match try_load_json("/tmp/chelis_std_test_json_definitely_missing_z4w8q1.json") with {
    | Some(_) => fail("try_load_json on a missing path must be None, got Some")
    | None => assert_true(true, "missing path -> None (not a crash)")
  }
def test_to_json_long_string_renders_escaped() -> unit ! { Test } = {
  tail = fold(fn (acc: string, i: i64) -> string_concat(acc, "ab"), "", range(cast(0, i64), cast(2048, i64)))
  big = string_concat("a\"b\\c", tail)
  expected = string_concat("\"a\\\"b\\\\c", string_concat(tail, "\""))
  assert_eq(to_json(JsonString(big)), expected, "4 KiB string with escapes renders byte-exactly (parse-back of long strings is capped by the reader's recursion, chelis#953/#954 territory)")
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
-- to_string renders f64 via the Rust Debug formatter, which always emits a
-- decimal point, so a whole-valued JsonFloat stays a float and round-trips as
-- JsonFloat, never collapsing to JsonInt; this pins the int/float boundary the
-- prelude JInt/JNum split (chelis#891) makes load-bearing.
def test_to_json_whole_valued_float_keeps_its_point() -> unit ! { Test } = {
  _ = assert_eq(to_json(JsonFloat(2.0f64)), "2.0", "a whole-valued float renders with a decimal point")
  _ = assert_eq(to_json(JsonInt(cast(2, i64))), "2", "an int renders without a decimal point")
  _ = match json_float(Some(parse_json("2.0"))) with {
    | Some(x) => assert_true(eq(x, 2.0f64), "\"2.0\" parses back as a float")
    | None => fail("\"2.0\" did not parse back as a float")
  }
  _ = match json_int(Some(parse_json("2.0"))) with {
    | Some(_) => fail("a whole-valued float must not read back as an int")
    | None => assert_true(true, "json_int refuses the whole-valued float")
  }
  match json_int(Some(parse_json("2"))) with {
    | Some(n) => assert_eq(n, cast(2, i64), "\"2\" parses back as an exact int")
    | None => fail("\"2\" did not parse back as an int")
  }
}
