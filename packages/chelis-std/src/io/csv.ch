module Std.Io.Csv
export (read_csv, try_read_csv, to_csv, try_to_csv, write_csv, try_write_csv)
import Std.Text (join)
def read_csv(path: string) -> List[Dict[string, string]] =
  match try_read_csv(path) with {
    | Some(rows) => rows
    | None => fail(string_concat("read_csv failed for ", path))
  }
def try_read_csv(path: string) -> Option[List[Dict[string, string]]] =
  if not(file_exists(path)) then None else {
    raw_lines = read_lines(path)
    lines = filter(fn (line: string) -> gt(string_len(line), cast(0, int64)), raw_lines)
    if eq(len(lines), cast(0, int64)) then Some([]) else match parse_line(index(lines, cast(0, int64))) with {
      | Some(headers) => parse_rows(headers, drop(lines, cast(1, int64)))
      | None => None
    }
  }
def to_csv(rows: List[Dict[string, string]]) -> string =
  match try_to_csv(rows) with {
    | Some(text) => text
    | None => fail(string_concat("to_csv failed at row ", string_concat(to_string(first_invalid_row(rows)), ": every row must have exactly the header row's keys, and no field or header may contain CR or LF")))
  }
def try_to_csv(rows: List[Dict[string, string]]) -> Option[string] =
  if eq(len(rows), cast(0, int64)) then Some("") else {
    headers = dict_keys(index(rows, cast(0, int64)))
    if and(fields_ok(headers), rows_ok(headers, rows)) then Some(render_all(headers, rows)) else None
  }
def write_csv(path: string, rows: List[Dict[string, string]]) -> unit =
  match try_to_csv(rows) with {
    | Some(text) => write_file(path, text)
    | None => fail(string_concat("write_csv failed for ", string_concat(path, string_concat(" at row ", string_concat(to_string(first_invalid_row(rows)), ": every row must have exactly the header row's keys, and no field or header may contain CR or LF")))))
  }
def try_write_csv(path: string, rows: List[Dict[string, string]]) -> Option[unit] =
  match try_to_csv(rows) with {
    | Some(text) => Some(write_file(path, text))
    | None => None
  }
def rows_ok(headers: List[string], rows: List[Dict[string, string]]) -> bool = fold(fn (acc: bool, row: Dict[string, string]) -> and(acc, row_ok(headers, row)), true, rows)
def row_ok(headers: List[string], row: Dict[string, string]) -> bool =
  if neq(len(row), len(headers)) then false else fold(fn (acc: bool, h: string) -> and(acc, match dict_get(row, h) with {
    | Some(v) => field_ok(v)
    | None => false
  }), true, headers)
def field_ok(text: string) -> bool = not(or(string_contains(text, "\n"), string_contains(text, "\r")))
def fields_ok(values: List[string]) -> bool = fold(fn (acc: bool, v: string) -> and(acc, field_ok(v)), true, values)
def render_all(headers: List[string], rows: List[Dict[string, string]]) -> string = string_concat(render_line(map(fn (h: string) -> render_field(h), headers)), string_concat("\n", string_concat(join(map(fn (row: Dict[string, string]) -> render_row(headers, row), rows), "\n"), "\n")))
def render_row(headers: List[string], row: Dict[string, string]) -> string =
  render_line(map(fn (h: string) -> render_field(match dict_get(row, h) with {
    | Some(v) => v
    | None => ""
  }), headers))
def render_line(fields: List[string]) -> string = {
  line = join(fields, ",")
  if eq(string_len(line), cast(0, int64)) then "\"\"" else line
}
def render_field(text: string) -> string = if or(string_contains(text, ","), string_contains(text, "\"")) then string_concat("\"", string_concat(double_quotes(text), "\"")) else text
def double_quotes(text: string) -> string = if not(string_contains(text, "\"")) then text else fold(fn (acc: string, idx: int64) -> string_concat(acc, if eq(string_slice(text, idx, cast(1, int64)), "\"") then "\"\"" else string_slice(text, idx, cast(1, int64))), "", range(cast(0, int64), string_len(text)))
def first_invalid_row(rows: List[Dict[string, string]]) -> int64 =
  if eq(len(rows), cast(0, int64)) then cast(-1, int64) else {
    headers = dict_keys(index(rows, cast(0, int64)))
    if not(fields_ok(headers)) then cast(0, int64) else {
      scan = fold(fn (acc: (int64, int64), row: Dict[string, string]) -> if gte(acc.1, cast(0, int64)) then (add(acc.0, cast(1, int64)), acc.1) else if row_ok(headers, row) then (add(acc.0, cast(1, int64)), cast(-1, int64)) else (add(acc.0, cast(1, int64)), acc.0), (cast(0, int64), cast(-1, int64)), rows)
      scan.1
    }
  }
-- Row parsing runs on the linear combinator lane. The previous shape
-- recursed one line at a time through `append(rows, ...)` and
-- `drop(lines, 1)`; both deep-clone their list argument, so reading r
-- rows allocated O(r^2) list bytes and every intermediate generation
-- stayed live until the recursion bottomed out. `map` and `fold` lower
-- to a capacity-reserved list plus in-place pushes (chelis#943/#949),
-- so the same read is O(r).
--
-- `map` cannot short-circuit, so the None contract needs three passes
-- rather than one: parse every line once, decide validity with a `fold`
-- over the parsed rows, and only build the dicts once every row is known
-- good. The intermediate holds retained handles to the very field strings
-- the dicts will hold, so it costs one pointer per field, not a second
-- copy of the text.
--
-- The intermediate is `List[List[string]]`, not `List[Option[List[string]]]`,
-- because the C backend refuses to box an ADT as a list element
-- ("unresolved host type `Option(List(String))` on boxing a resolved host
-- value"), which the eval lane accepts. `[]` stands in for a line that did
-- not parse, and it is unambiguous rather than merely convenient:
-- `parse_line_chars` always appends its final `current` field, so a
-- successful parse returns at least one field and can never be empty.
def parse_rows(headers: List[string], lines: List[string]) -> Option[List[Dict[string, string]]] = {
  parsed = map(fn (line: string) -> fields_of(line), lines)
  if parsed_ok(len(headers), parsed) then Some(map(fn (fields: List[string]) -> dict_of(zip(headers, fields)), parsed)) else None
}
def fields_of(line: string) -> List[string] =
  match parse_line(line) with {
    | Some(fields) => fields
    | None => []
  }
-- One test rejects both failure modes, and it is complete rather than
-- merely convenient. `width` is `len(headers)`, and `headers` came from a
-- `parse_line` that returned `Some`, so `width` is at least 1; the `[]`
-- standing for a line that did not parse has length 0 and can therefore
-- never equal it. An explicit `len(fields) > 0` conjunct alongside this
-- would be unfalsifiable -- no input can make it the deciding term -- so it
-- is left out rather than shipped as a guard no test can fire.
def parsed_ok(width: int64, parsed: List[List[string]]) -> bool = fold(fn (acc: bool, fields: List[string]) -> and(acc, eq(len(fields), width)), true, parsed)
def parse_line(line: string) -> Option[List[string]] = parse_line_chars(line, cast(0, int64), false, "", [])
def parse_line_chars(line: string, idx: int64, in_quotes: bool, current: string, fields: List[string]) -> Option[List[string]] =
  if gte(idx, string_len(line)) then if in_quotes then None else Some(append(fields, current)) else {
    ch = string_slice(line, idx, cast(1, int64))
    if eq(ch, "\"") then if in_quotes then {
      next = add(idx, cast(1, int64))
      if and(lt(next, string_len(line)), eq(string_slice(line, next, cast(1, int64)), "\"")) then parse_line_chars(line, add(idx, cast(2, int64)), true, string_concat(current, "\""), fields) else parse_line_chars(line, add(idx, cast(1, int64)), false, current, fields)
    } else parse_line_chars(line, add(idx, cast(1, int64)), true, current, fields) else if and(eq(ch, ","), not(in_quotes)) then parse_line_chars(line, add(idx, cast(1, int64)), false, "", append(fields, current)) else parse_line_chars(line, add(idx, cast(1, int64)), in_quotes, string_concat(current, ch), fields)
  }
