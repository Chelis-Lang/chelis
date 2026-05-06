module Std.Io.Json
export (Json, load_json, parse_json, try_load_json, try_parse_json, json_get, json_string, json_int, json_float, json_bool, json_array, json_object, json_is_null)
type Json =
  | JsonNull
  | JsonBool(bool)
  | JsonInt(int64)
  | JsonFloat(f64)
  | JsonString(string)
  | JsonArray(List[Json])
  | JsonObject(Dict[string, Json])
def load_json(path: string) -> Json = { match try_load_json(path) with {
  | Some(value) => value
  | None => fail(string_concat("load_json failed for ", path))
} }
def try_load_json(path: string) -> Option[Json] = try_parse_json(read_file(path))
def parse_json(text: string) -> Json = { match try_parse_json(text) with {
  | Some(value) => value
  | None => fail("parse_json failed: malformed JSON")
} }
def try_parse_json(text: string) -> Option[Json] = {
  start = skip_ws(text, cast(0, int64))
  match parse_value(text, start) with {
    | Some(pair) => {
    (value, next) = pair
    if eq(skip_ws(text, next), string_len(text)) then Some(value) else None
  }
    | None => None
  }
}
def json_get(value: Json, key: string) -> Option[Json] = { match value with {
  | JsonObject(entries) => dict_get(entries, key)
  | _ => None
} }
def json_string(value: Option[Json]) -> Option[string] = { match value with {
  | Some(inner) => match inner with {
  | JsonString(text) => Some(text)
  | _ => None
}
  | None => None
} }
def json_int(value: Option[Json]) -> Option[int64] = { match value with {
  | Some(inner) => match inner with {
  | JsonInt(n) => Some(n)
  | _ => None
}
  | None => None
} }
def json_float(value: Option[Json]) -> Option[f64] = { match value with {
  | Some(inner) => match inner with {
  | JsonFloat(n) => Some(n)
  | JsonInt(n) => Some(cast(n, f64))
  | _ => None
}
  | None => None
} }
def json_bool(value: Option[Json]) -> Option[bool] = { match value with {
  | Some(inner) => match inner with {
  | JsonBool(flag) => Some(flag)
  | _ => None
}
  | None => None
} }
def json_array(value: Option[Json]) -> Option[List[Json]] = { match value with {
  | Some(inner) => match inner with {
  | JsonArray(items) => Some(items)
  | _ => None
}
  | None => None
} }
def json_object(value: Option[Json]) -> Option[Dict[string, Json]] = { match value with {
  | Some(inner) => match inner with {
  | JsonObject(entries) => Some(entries)
  | _ => None
}
  | None => None
} }
def json_is_null(value: Option[Json]) -> bool = { match value with {
  | Some(inner) => match inner with {
  | JsonNull => true
  | _ => false
}
  | None => false
} }
def parse_value(text: string, idx: int64) -> Option[(Json, int64)] = { if gte(idx, string_len(text)) then None else {
  ch = char_at(text, idx)
  if eq(ch, "{") then { match parse_object(text, add(idx, cast(1, int64))) with {
    | Some(pair) => {
    (entries, next) = pair
    Some((JsonObject(entries), next))
  }
    | None => None
  } } else if eq(ch, "[") then { match parse_array(text, add(idx, cast(1, int64))) with {
    | Some(pair) => {
    (items, next) = pair
    Some((JsonArray(items), next))
  }
    | None => None
  } } else if eq(ch, "\"") then { match parse_string(text, idx) with {
    | Some(pair) => {
    (value, next) = pair
    Some((JsonString(value), next))
  }
    | None => None
  } } else if starts_with_at(text, idx, "true") then Some((JsonBool(true), add(idx, cast(4, int64)))) else if starts_with_at(text, idx, "false") then Some((JsonBool(false), add(idx, cast(5, int64)))) else if starts_with_at(text, idx, "null") then Some((JsonNull, add(idx, cast(4, int64)))) else if or(eq(ch, "-"), is_digit(ch)) then parse_number(text, idx) else None
} }
def parse_object(text: string, idx: int64) -> Option[(Dict[string, Json], int64)] = {
  next = skip_ws(text, idx)
  if gte(next, string_len(text)) then None else if eq(char_at(text, next), "}") then Some((dict_of([]), add(next, cast(1, int64)))) else match parse_string(text, next) with {
    | Some(key_pair) => {
    (key, after_key) = key_pair
    colon = skip_ws(text, after_key)
    if gte(colon, string_len(text)) then None else if neq(char_at(text, colon), ":") then None else match parse_value(text, skip_ws(text, add(colon, cast(1, int64)))) with {
      | Some(value_pair) => {
      (value, after_value) = value_pair
      parse_object_rest(text, after_value, [(key, value)])
    }
      | None => None
    }
  }
    | None => None
  }
}
def parse_object_rest(text: string, idx: int64, entries: List[(string, Json)]) -> Option[(Dict[string, Json], int64)] = {
  next = skip_ws(text, idx)
  if gte(next, string_len(text)) then None else {
    ch = char_at(text, next)
    if eq(ch, "}") then Some((dict_of(entries), add(next, cast(1, int64)))) else if eq(ch, ",") then {
      key_idx = skip_ws(text, add(next, cast(1, int64)))
      match parse_string(text, key_idx) with {
        | Some(key_pair) => {
        (key, after_key) = key_pair
        colon = skip_ws(text, after_key)
        if gte(colon, string_len(text)) then None else if neq(char_at(text, colon), ":") then None else match parse_value(text, skip_ws(text, add(colon, cast(1, int64)))) with {
          | Some(value_pair) => {
          (value, after_value) = value_pair
          parse_object_rest(text, after_value, append(entries, (key, value)))
        }
          | None => None
        }
      }
        | None => None
      }
    } else None
  }
}
def parse_array(text: string, idx: int64) -> Option[(List[Json], int64)] = {
  next = skip_ws(text, idx)
  if gte(next, string_len(text)) then None else if eq(char_at(text, next), "]") then Some(([], add(next, cast(1, int64)))) else match parse_value(text, next) with {
    | Some(pair) => {
    (value, after_value) = pair
    parse_array_rest(text, after_value, [value])
  }
    | None => None
  }
}
def parse_array_rest(text: string, idx: int64, items: List[Json]) -> Option[(List[Json], int64)] = {
  next = skip_ws(text, idx)
  if gte(next, string_len(text)) then None else {
    ch = char_at(text, next)
    if eq(ch, "]") then Some((items, add(next, cast(1, int64)))) else if eq(ch, ",") then match parse_value(text, skip_ws(text, add(next, cast(1, int64)))) with {
      | Some(pair) => {
      (value, after_value) = pair
      parse_array_rest(text, after_value, append(items, value))
    }
      | None => None
    } else None
  }
}
def parse_string(text: string, idx: int64) -> Option[(string, int64)] = { if gte(idx, string_len(text)) then None else if neq(char_at(text, idx), "\"") then None else parse_string_chars(text, add(idx, cast(1, int64)), "") }
def parse_string_chars(text: string, idx: int64, acc: string) -> Option[(string, int64)] = { if gte(idx, string_len(text)) then None else {
  ch = char_at(text, idx)
  if eq(ch, "\"") then Some((acc, add(idx, cast(1, int64)))) else if eq(ch, "\\") then {
    esc_idx = add(idx, cast(1, int64))
    if gte(esc_idx, string_len(text)) then None else match decode_escape(char_at(text, esc_idx)) with {
      | Some(value) => parse_string_chars(text, add(idx, cast(2, int64)), string_concat(acc, value))
      | None => None
    }
  } else parse_string_chars(text, add(idx, cast(1, int64)), string_concat(acc, ch))
} }
def decode_escape(ch: string) -> Option[string] = { if eq(ch, "\"") then Some("\"") else if eq(ch, "\\") then Some("\\") else if eq(ch, "/") then Some("/") else if eq(ch, "n") then Some("\n") else if eq(ch, "r") then Some("\r") else if eq(ch, "t") then Some("\t") else None }
def parse_number(text: string, idx: int64) -> Option[(Json, int64)] = {
  end = scan_number_end(text, idx)
  raw = string_slice(text, idx, sub(end, idx))
  if or(string_contains(raw, "."), or(string_contains(raw, "e"), string_contains(raw, "E"))) then match to_float(raw) with {
    | Some(value) => Some((JsonFloat(value), end))
    | None => None
  } else match to_int(raw) with {
    | Some(value) => Some((JsonInt(value), end))
    | None => None
  }
}
def scan_number_end(text: string, idx: int64) -> int64 = { if gte(idx, string_len(text)) then idx else {
  ch = char_at(text, idx)
  if is_number_char(ch) then scan_number_end(text, add(idx, cast(1, int64))) else idx
} }
def is_number_char(ch: string) -> bool = or(is_digit(ch), or(eq(ch, "-"), or(eq(ch, "+"), or(eq(ch, "."), or(eq(ch, "e"), eq(ch, "E"))))))
def skip_ws(text: string, idx: int64) -> int64 = { if gte(idx, string_len(text)) then idx else {
  ch = char_at(text, idx)
  if is_ws(ch) then skip_ws(text, add(idx, cast(1, int64))) else idx
} }
def starts_with_at(text: string, idx: int64, prefix: string) -> bool = eq(string_slice(text, idx, string_len(prefix)), prefix)
def char_at(text: string, idx: int64) -> string = string_slice(text, idx, cast(1, int64))
def is_ws(ch: string) -> bool = or(eq(ch, " "), or(eq(ch, "\n"), or(eq(ch, "\r"), eq(ch, "\t"))))
def is_digit(ch: string) -> bool = or(eq(ch, "0"), or(eq(ch, "1"), or(eq(ch, "2"), or(eq(ch, "3"), or(eq(ch, "4"), or(eq(ch, "5"), or(eq(ch, "6"), or(eq(ch, "7"), or(eq(ch, "8"), eq(ch, "9"))))))))))
