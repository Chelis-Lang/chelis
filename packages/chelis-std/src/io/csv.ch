module Std.Io.Csv
export (read_csv, try_read_csv)
def read_csv(path: string) -> List[Dict[string, string]] = { match try_read_csv(path) with {
  | Some(rows) => rows
  | None => fail(string_concat("read_csv failed for ", path))
} }
def try_read_csv(path: string) -> Option[List[Dict[string, string]]] = { if not(file_exists(path)) then None else {
  raw_lines = read_lines(path)
  lines = filter(fn (line: string) -> gt(string_len(line), cast(0, int64)), raw_lines)
  if eq(len(lines), cast(0, int64)) then Some([]) else match parse_line(index(lines, cast(0, int64))) with {
    | Some(headers) => parse_rows(headers, drop(lines, cast(1, int64)), [])
    | None => None
  }
} }
def parse_rows(headers: List[string], lines: List[string], rows: List[Dict[string, string]]) -> Option[List[Dict[string, string]]] = { if eq(len(lines), cast(0, int64)) then Some(rows) else {
  line = index(lines, cast(0, int64))
  match parse_line(line) with {
    | Some(fields) => if neq(len(headers), len(fields)) then None else parse_rows(headers, drop(lines, cast(1, int64)), append(rows, dict_of(zip(headers, fields))))
    | None => None
  }
} }
def parse_line(line: string) -> Option[List[string]] = parse_line_chars(line, cast(0, int64), false, "", [])
def parse_line_chars(line: string, idx: int64, in_quotes: bool, current: string, fields: List[string]) -> Option[List[string]] = { if gte(idx, string_len(line)) then if in_quotes then None else Some(append(fields, current)) else {
  ch = string_slice(line, idx, cast(1, int64))
  if eq(ch, "\"") then if in_quotes then {
    next = add(idx, cast(1, int64))
    if and(lt(next, string_len(line)), eq(string_slice(line, next, cast(1, int64)), "\"")) then parse_line_chars(line, add(idx, cast(2, int64)), true, string_concat(current, "\""), fields) else parse_line_chars(line, add(idx, cast(1, int64)), false, current, fields)
  } else parse_line_chars(line, add(idx, cast(1, int64)), true, current, fields) else if and(eq(ch, ","), not(in_quotes)) then parse_line_chars(line, add(idx, cast(1, int64)), false, "", append(fields, current)) else parse_line_chars(line, add(idx, cast(1, int64)), in_quotes, string_concat(current, ch), fields)
} }
