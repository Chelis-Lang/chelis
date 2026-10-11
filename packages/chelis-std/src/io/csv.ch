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
    lines = filter(fn (line: string) -> gt(string_len(line), cast(0, i64)), raw_lines)
    if eq(len(lines), cast(0, i64)) then Some([]) else match parse_line(index(lines, cast(0, i64))) with {
      | Some(headers) => parse_rows(headers, skip(lines, cast(1, i64)))
      | None => None
    }
  }
def to_csv(rows: List[Dict[string, string]]) -> string =
  match try_to_csv(rows) with {
    | Some(text) => text
    | None => fail(string_concat("to_csv failed at row ", string_concat(to_string(first_invalid_row(rows)), ": every row must have exactly the header row's keys, and no field or header may contain CR or LF")))
  }
def try_to_csv(rows: List[Dict[string, string]]) -> Option[string] =
  if eq(len(rows), cast(0, i64)) then Some("") else {
    headers = dict_keys(index(rows, cast(0, i64)))
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
  if eq(string_len(line), cast(0, i64)) then "\"\"" else line
}
def render_field(text: string) -> string = if or(string_contains(text, ","), string_contains(text, "\"")) then string_concat("\"", string_concat(double_quotes(text), "\"")) else text
def double_quotes(text: string) -> string =
  if not(string_contains(text, "\"")) then text else if lte(string_len(text), cast(64, i64)) then fold(fn (acc: string, idx: i64) -> string_concat(acc, if eq(string_slice(text, idx, cast(1, i64)), "\"") then "\"\"" else string_slice(text, idx, cast(1, i64))), "", range(cast(0, i64), string_len(text))) else {
    middle = floor_div(string_len(text), cast(2, i64))
    left = string_slice(text, cast(0, i64), middle)
    right = string_slice(text, middle, sub(string_len(text), middle))
    string_concat(double_quotes(left), double_quotes(right))
  }
def first_invalid_row(rows: List[Dict[string, string]]) -> i64 =
  if eq(len(rows), cast(0, i64)) then cast(-1, i64) else {
    headers = dict_keys(index(rows, cast(0, i64)))
    if not(fields_ok(headers)) then cast(0, i64) else {
      scan = fold(fn (acc: (i64, i64), row: Dict[string, string]) -> if gte(acc.1, cast(0, i64)) then (add(acc.0, cast(1, i64)), acc.1) else if row_ok(headers, row) then (add(acc.0, cast(1, i64)), cast(-1, i64)) else (add(acc.0, cast(1, i64)), acc.0), (cast(0, i64), cast(-1, i64)), rows)
      scan.1
    }
  }
-- Row parsing runs on the linear combinator lane. The pre-#1213 shape
-- recursed one line at a time through `append(rows, ...)` and
-- `skip(lines, 1)`; both deep-clone their list argument, so reading r
-- rows allocated O(r^2) list bytes and every intermediate generation
-- stayed live until the recursion bottomed out. `map` and `fold` lower
-- to a capacity-reserved list plus in-place pushes (chelis#943/#949),
-- so the same read is O(r).
--
-- The validity pass carries only a bool, and its body is an `if` on the
-- accumulator rather than an `and(...)` call: a call evaluates both
-- arguments, while `if` evaluates one branch, so after the first invalid
-- row every later line is skipped without being parsed. A malformed row
-- therefore leaves the rest of the file untouched, including long lines.
--
-- On success every line is parsed a second time to build the dicts,
-- doubling a linear constant. Both single-parse alternatives lose more:
-- a fold that appends each row's fields to an accumulator list
-- deep-clones the accumulator per step on the eval lane (the exact
-- O(r^2) this change removed), and a `map` into an intermediate
-- `List[List[string]]` parses every line unconditionally, which is the
-- short-circuit regression this shape exists to prevent.
def parse_rows(headers: List[string], lines: List[string]) -> Option[List[Dict[string, string]]] = {
  width = len(headers)
  ok = fold(fn (acc: bool, line: string) -> if acc then line_ok(width, line) else false, true, lines)
  if ok then Some(map(fn (line: string) -> dict_of(zip(headers, fields_of(line))), lines)) else None
}
-- One test rejects both failure modes: `width` is `len(headers)`, and
-- `headers` came from a `parse_line` that returned `Some`, so `width` is
-- at least 1, while an unparsable line takes the `None` arm directly.
def line_ok(width: i64, line: string) -> bool =
  match parse_line(line) with {
    | Some(fields) => eq(len(fields), width)
    | None => false
  }
-- Only reached from the success arm of `parse_rows`, where every line
-- already passed `line_ok`, so the `None` arm is unreachable there; `[]`
-- keeps the function total without smuggling a sentinel into results.
def fields_of(line: string) -> List[string] =
  match parse_line(line) with {
    | Some(fields) => fields
    | None => []
  }
-- Quote-free segments without a comma are complete fields. Split other
-- segments in halves so only bounded leaves scan one scalar at a time.
def split_unquoted_small(line: string) -> List[string] = {
  state = fold(fn (acc: (string, List[string]), idx: i64) -> {
    ch = string_slice(line, idx, cast(1, i64))
    if eq(ch, ",") then ("", append(acc.1, acc.0)) else (string_concat(acc.0, ch), acc.1)
  }, ("", []), range(cast(0, i64), string_len(line)))
  append(state.1, state.0)
}
def split_unquoted(line: string) -> List[string] =
  if not(string_contains(line, ",")) then [line] else if lte(string_len(line), cast(64, i64)) then split_unquoted_small(line) else {
    middle = floor_div(string_len(line), cast(2, i64))
    left = split_unquoted(string_slice(line, cast(0, i64), middle))
    right = split_unquoted(string_slice(line, middle, sub(string_len(line), middle)))
    joined = string_concat(index(left, sub(len(left), cast(1, i64))), index(right, cast(0, i64)))
    concat(append(take(left, sub(len(left), cast(1, i64))), joined), skip(right, cast(1, i64)))
  }
-- Each leaf scans at most 64 scalars; a pending closing quote lets doubled
-- quotes span a leaf boundary. Parent nodes join only their touching fields,
-- so long quoted fields are never copied once per input scalar.
def scan_quoted_small(line: string, inside: bool, pending: bool) -> (bool, bool, List[string]) = {
  state = fold(fn (acc: (bool, bool, string, List[string]), idx: i64) -> {
    ch = string_slice(line, idx, cast(1, i64))
    if eq(ch, "\"") then if acc.1 then (true, false, string_concat(acc.2, "\""), acc.3) else (true, acc.0, acc.2, acc.3) else {
      next_inside = and(acc.0, not(acc.1))
      if and(eq(ch, ","), not(next_inside)) then (false, false, "", append(acc.3, acc.2)) else (next_inside, false, string_concat(acc.2, ch), acc.3)
    }
  }, (inside, pending, "", []), range(cast(0, i64), string_len(line)))
  (state.0, state.1, append(state.3, state.2))
}
def scan_quoted(line: string, inside: bool, pending: bool) -> (bool, bool, List[string]) =
  if not(string_contains(line, "\"")) then if and(inside, not(pending)) then (true, false, [line]) else (false, false, split_unquoted(line)) else if lte(string_len(line), cast(64, i64)) then scan_quoted_small(line, inside, pending) else {
    middle = floor_div(string_len(line), cast(2, i64))
    left = scan_quoted(string_slice(line, cast(0, i64), middle), inside, pending)
    right = scan_quoted(string_slice(line, middle, sub(string_len(line), middle)), left.0, left.1)
    joined = string_concat(index(left.2, sub(len(left.2), cast(1, i64))), index(right.2, cast(0, i64)))
    (right.0, right.1, concat(append(take(left.2, sub(len(left.2), cast(1, i64))), joined), skip(right.2, cast(1, i64))))
  }
def parse_line(line: string) -> Option[List[string]] =
  if not(string_contains(line, "\"")) then Some(split_unquoted(line)) else {
    state = scan_quoted(line, false, false)
    if and(state.0, not(state.1)) then None else Some(state.2)
  }
