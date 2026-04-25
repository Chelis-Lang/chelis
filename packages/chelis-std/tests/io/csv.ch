module Std.Tests.IO.Csv
import Std.IO.Csv (read_csv, try_read_csv)
import Std.Test (assert_eq_int, assert_eq_string, assert_true, fail)

-- Tests for `Std.IO.Csv`. The module's public surface is file-only:
-- `read_csv(path) -> List[Dict[string, string]]` and
-- `try_read_csv(path) -> Option[List[Dict[string, string]]]`. There is no
-- string-parsing entrypoint exported (the internal `parse_line` is private).
-- We therefore round-trip via `write_file` + the public reader. Every test
-- uses a unique fixture path under `/tmp` so test order is irrelevant.
--
-- Documented gap: `try_read_csv("/nonexistent/path")` does NOT return `None`.
-- The underlying `read_lines` raises an evaluator error when the path is
-- missing, so the failure surfaces as an evaluation crash rather than `None`.
-- Covering that case requires either (a) a `parse_csv(text)` string entrypoint,
-- or (b) wrapping the `read_lines` call in `file_exists` inside `try_read_csv`.
-- Both are upstream changes to `Std.IO.Csv` and out of scope for the self-test
-- suite. See phase 3t A1 Wave 2 follow-up.

def check_field(rows: List[Dict[string, string]], i: int64, k: string, expected: string, label: string) -> unit ! { Test } = { match dict_get(index(rows, i), k) with {
  | Some(v) => assert_eq_string(v, expected, label)
  | None => fail(string_concat("missing key in row: ", label))
} }

def test_round_trip_two_rows() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_two_rows.csv"
  _ = write_file(path, "name,age\nAlice,30\nBob,25")
  rows = read_csv(path)
  _ = assert_eq_int(len(rows), cast(2, int64), "two-row CSV yields 2 dicts")
  _ = check_field(rows, cast(0, int64), "name", "Alice", "row0.name == Alice")
  _ = check_field(rows, cast(0, int64), "age", "30", "row0.age == 30")
  _ = check_field(rows, cast(1, int64), "name", "Bob", "row1.name == Bob")
  check_field(rows, cast(1, int64), "age", "25", "row1.age == 25")
}

def test_try_read_csv_returns_some_for_valid_file() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_try_some.csv"
  _ = write_file(path, "k,v\nfoo,bar")
  match try_read_csv(path) with {
    | Some(rows) => assert_eq_int(len(rows), cast(1, int64), "try_read_csv Some([1 row])")
    | None => fail("try_read_csv on a valid file must be Some, got None")
  }
}

def test_empty_file_returns_some_empty() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_empty.csv"
  _ = write_file(path, "")
  match try_read_csv(path) with {
    | Some(rows) => assert_eq_int(len(rows), cast(0, int64), "empty file -> Some([])")
    | None => fail("try_read_csv on empty file must be Some([]), got None")
  }
}

def test_header_only_returns_zero_rows() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_header_only.csv"
  _ = write_file(path, "name,age")
  rows = read_csv(path)
  assert_eq_int(len(rows), cast(0, int64), "header-only CSV -> 0 rows")
}

def test_mismatched_column_count_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_mismatch.csv"
  _ = write_file(path, "a,b,c\nx,y")
  match try_read_csv(path) with {
    | Some(_) => fail("mismatched col count must yield None, got Some")
    | None => assert_true(true, "mismatched col count -> None")
  }
}

def test_quoted_field_with_embedded_comma() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_quoted.csv"
  _ = write_file(path, "k,v\n\"a,b\",x")
  rows = read_csv(path)
  _ = assert_eq_int(len(rows), cast(1, int64), "quoted-comma CSV -> 1 row")
  _ = check_field(rows, cast(0, int64), "k", "a,b", "embedded comma preserved inside quotes")
  check_field(rows, cast(0, int64), "v", "x", "trailing field after quoted preserved")
}

def test_quoted_field_with_escaped_quote() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_escaped_quote.csv"
  _ = write_file(path, "k,v\n\"he said \"\"hi\"\"\",1")
  rows = read_csv(path)
  _ = assert_eq_int(len(rows), cast(1, int64), "escaped-quote CSV -> 1 row")
  check_field(rows, cast(0, int64), "k", "he said \"hi\"", "doubled \"\" decodes to literal \"")
}

def test_unterminated_quote_returns_none() -> unit ! { Test, IO } = {
  -- Header has one column so the unterminated-quote branch is the ONLY way
  -- to reach None. If the parser silently closed the quote at EOF instead of
  -- returning None, fields would be ["hi extra"], col count would still
  -- match the 1-column header, and the column-mismatch guard would not catch
  -- the bug. Single-column header pins the actual unterminated-quote path.
  path = "/tmp/chelis_std_test_csv_unterm.csv"
  _ = write_file(path, "k\n\"hi extra")
  match try_read_csv(path) with {
    | Some(_) => fail("unterminated quote must yield None, got Some")
    | None => assert_true(true, "unterminated quote -> None")
  }
}
