module Std.Tests.Io.Csv
import Std.Io.Csv (read_csv, try_read_csv, to_csv, try_to_csv, write_csv, try_write_csv)
import Std.Test (assert_eq, assert_true, fail)
def check_field(rows: List[Dict[string, string]], i: i64, k: string, expected: string, label: string) -> unit ! { Test } =
  match dict_get(index(rows, i), k) with {
    | Some(v) => assert_eq(v, expected, label)
    | None => fail(string_concat("missing key in row: ", label))
  }
def test_round_trip_two_rows() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_two_rows.csv"
  _ = write_file(path, "name,age\nAlice,30\nBob,25")
  rows = read_csv(path)
  _ = assert_eq(len(rows), cast(2, i64), "two-row CSV yields 2 dicts")
  _ = check_field(rows, cast(0, i64), "name", "Alice", "row0.name == Alice")
  _ = check_field(rows, cast(0, i64), "age", "30", "row0.age == 30")
  _ = check_field(rows, cast(1, i64), "name", "Bob", "row1.name == Bob")
  check_field(rows, cast(1, i64), "age", "25", "row1.age == 25")
}
def test_try_read_csv_returns_some_for_valid_file() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_try_some.csv"
  _ = write_file(path, "k,v\nfoo,bar")
  match try_read_csv(path) with {
    | Some(rows) => assert_eq(len(rows), cast(1, i64), "try_read_csv Some([1 row])")
    | None => fail("try_read_csv on a valid file must be Some, got None")
  }
}
def test_empty_file_returns_some_empty() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_empty.csv"
  _ = write_file(path, "")
  match try_read_csv(path) with {
    | Some(rows) => assert_eq(len(rows), cast(0, i64), "empty file -> Some([])")
    | None => fail("try_read_csv on empty file must be Some([]), got None")
  }
}
def test_header_only_returns_zero_rows() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_header_only.csv"
  _ = write_file(path, "name,age")
  rows = read_csv(path)
  assert_eq(len(rows), cast(0, i64), "header-only CSV -> 0 rows")
}
def test_mismatched_column_count_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_mismatch.csv"
  _ = write_file(path, "a,b,c\nx,y")
  match try_read_csv(path) with {
    | Some(_) => fail("mismatched col count must yield None, got Some")
    | None => assert_true(true, "mismatched col count -> None")
  }
}
def test_extra_column_count_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_too_many.csv"
  _ = write_file(path, "a,b\nx,y,z")
  match try_read_csv(path) with {
    | Some(_) => fail("a row with MORE fields than the header must yield None, got Some")
    | None => assert_true(true, "wider-than-header row -> None")
  }
}
def test_short_row_in_last_position_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_ragged_tail.csv"
  _ = write_file(path, "a,b\n1,2\n3,4\n5")
  match try_read_csv(path) with {
    | Some(_) => fail("a short row in the LAST position must yield None, got Some")
    | None => assert_true(true, "trailing short row -> None (validity is not decided by the first data row alone)")
  }
}
def test_short_row_in_first_position_followed_by_valid_rows_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_ragged_head.csv"
  _ = write_file(path, "a,b\n1\n2,3\n4,5")
  match try_read_csv(path) with {
    | Some(_) => fail("a short FIRST data row must yield None even when later rows are well-formed, got Some")
    | None => assert_true(true, "leading short row -> None (a later valid row does not clear the verdict)")
  }
}
def test_wide_row_in_middle_position_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_ragged_middle.csv"
  _ = write_file(path, "a,b\n1,2\n3,4,5\n6,7")
  match try_read_csv(path) with {
    | Some(_) => fail("a wide row between two well-formed rows must yield None, got Some")
    | None => assert_true(true, "interior wide row -> None")
  }
}
def test_unterminated_quote_in_last_row_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_unterm_tail.csv"
  _ = write_file(path, "k\nok\n\"dangling")
  match try_read_csv(path) with {
    | Some(_) => fail("an unterminated quote in a NON-first data row must yield None, got Some")
    | None => assert_true(true, "trailing unterminated quote -> None (an unparsable line is not a placeholder row)")
  }
}
def kib4_line() -> string = {
  s1 = "x"
  s2 = string_concat(s1, s1)
  s4 = string_concat(s2, s2)
  s8 = string_concat(s4, s4)
  s16 = string_concat(s8, s8)
  s32 = string_concat(s16, s16)
  s64 = string_concat(s32, s32)
  s128 = string_concat(s64, s64)
  s256 = string_concat(s128, s128)
  s512 = string_concat(s256, s256)
  s1024 = string_concat(s512, s512)
  s2048 = string_concat(s1024, s1024)
  string_concat(s2048, s2048)
}
-- A malformed first data row makes the whole file invalid even when a
-- later row is long. The CLI regression also uses an execution budget to
-- check that the later row is skipped rather than parsed eagerly.
def test_short_first_row_before_long_line_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_short_then_long.csv"
  _ = write_file(path, string_concat("a,b\n1\n", kib4_line()))
  match try_read_csv(path) with {
    | Some(_) => fail("a short first data row must yield None even with a later long row, got Some")
    | None => assert_true(true, "short first row -> None even with a 4 KiB later line")
  }
}
def test_quoted_field_with_embedded_comma() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_quoted.csv"
  _ = write_file(path, "k,v\n\"a,b\",x")
  rows = read_csv(path)
  _ = assert_eq(len(rows), cast(1, i64), "quoted-comma CSV -> 1 row")
  _ = check_field(rows, cast(0, i64), "k", "a,b", "embedded comma preserved inside quotes")
  check_field(rows, cast(0, i64), "v", "x", "trailing field after quoted preserved")
}
def test_quoted_field_with_escaped_quote() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_escaped_quote.csv"
  _ = write_file(path, "k,v\n\"he said \"\"hi\"\"\",1")
  rows = read_csv(path)
  _ = assert_eq(len(rows), cast(1, i64), "escaped-quote CSV -> 1 row")
  check_field(rows, cast(0, i64), "k", "he said \"hi\"", "doubled \"\" decodes to literal \"")
}
def test_try_read_csv_missing_path_returns_none() -> unit ! { Test, IO } =
  match try_read_csv("/tmp/chelis_std_test_csv_definitely_missing_x9q7p2.csv") with {
    | Some(_) => fail("try_read_csv on missing path must be None, got Some")
    | None => assert_true(true, "missing path -> None (not a crash)")
  }
def test_unterminated_quote_returns_none() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_unterm.csv"
  _ = write_file(path, "k\n\"hi extra")
  match try_read_csv(path) with {
    | Some(_) => fail("unterminated quote must yield None, got Some")
    | None => assert_true(true, "unterminated quote -> None")
  }
}
def test_to_csv_renders_header_and_rows() -> unit ! { Test } = {
  rows = [dict_of([("name", "Alice"), ("age", "30")]), dict_of([("name", "Bob"), ("age", "25")])]
  assert_eq(to_csv(rows), "name,age\nAlice,30\nBob,25\n", "two rows render with header, LF endings, trailing newline")
}
def test_to_csv_quotes_comma_and_doubles_quotes() -> unit ! { Test } = {
  rows = [dict_of([("k", "a,b"), ("v", "he said \"hi\"")])]
  assert_eq(to_csv(rows), "k,v\n\"a,b\",\"he said \"\"hi\"\"\"\n", "comma field is quoted and embedded quotes are doubled")
}
def test_to_csv_empty_rows_renders_empty_string() -> unit ! { Test } = assert_eq(to_csv([]), "", "no rows -> empty output, mirroring try_read_csv on an empty file")
def test_try_to_csv_mismatched_key_sets_returns_none() -> unit ! { Test } = {
  rows = [dict_of([("a", "1")]), dict_of([("b", "2")])]
  match try_to_csv(rows) with {
    | Some(_) => fail("row with a different key set must yield None")
    | None => assert_true(true, "mismatched key set -> None")
  }
}
def test_try_to_csv_extra_key_returns_none() -> unit ! { Test } = {
  rows = [dict_of([("a", "1")]), dict_of([("a", "1"), ("b", "2")])]
  match try_to_csv(rows) with {
    | Some(_) => fail("row with an extra key must yield None")
    | None => assert_true(true, "extra key -> None")
  }
}
def test_try_to_csv_newline_field_returns_none() -> unit ! { Test } = {
  rows = [dict_of([("k", "line one\nline two")])]
  match try_to_csv(rows) with {
    | Some(_) => fail("a field containing LF must yield None: the line-based reader cannot round-trip it")
    | None => assert_true(true, "LF field -> None")
  }
}
def test_try_to_csv_cr_header_returns_none() -> unit ! { Test } = {
  rows = [dict_of([("bad\rheader", "x")])]
  match try_to_csv(rows) with {
    | Some(_) => fail("a header containing CR must yield None")
    | None => assert_true(true, "CR header -> None")
  }
}
def test_empty_single_column_value_round_trips() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_empty_value.csv"
  rows = [dict_of([("a", "x")]), dict_of([("a", "")])]
  _ = write_csv(path, rows)
  back = read_csv(path)
  _ = assert_eq(len(back), cast(2, i64), "empty single-column value survives round-trip as a quoted empty field, not a dropped blank line")
  check_field(back, cast(1, i64), "a", "", "row1.a reads back as the empty string")
}
def test_empty_header_round_trips() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_empty_header.csv"
  rows = [dict_of([("", "x")])]
  _ = write_csv(path, rows)
  back = read_csv(path)
  _ = assert_eq(len(back), cast(1, i64), "empty header renders quoted and is not promoted over by the data row")
  check_field(back, cast(0, i64), "", "x", "value keyed by the empty header reads back")
}
def test_to_csv_handles_multi_kilobyte_fields() -> unit ! { Test } = {
  tail = fold(fn (acc: string, i: i64) -> string_concat(acc, "ab"), "", range(cast(0, i64), cast(2048, i64)))
  field = string_concat("a\"b,", tail)
  rows = [dict_of([("k", field), ("v", "plain")])]
  expected = string_concat("k,v\n\"a\"\"b,", string_concat(tail, "\",plain\n"))
  assert_eq(to_csv(rows), expected, "4 KiB quoted field renders byte-exactly")
}
def test_try_write_csv_mismatched_rows_returns_none() -> unit ! { Test, IO } = {
  rows = [dict_of([("a", "1")]), dict_of([("b", "2")])]
  match try_write_csv("/tmp/chelis_std_test_csv_try_write.csv", rows) with {
    | Some(_) => fail("try_write_csv on mismatched key sets must return None")
    | None => assert_true(true, "mismatched key set -> None through the write facade")
  }
}
def test_write_csv_round_trips_through_read_csv() -> unit ! { Test, IO } = {
  path = "/tmp/chelis_std_test_csv_write_round_trip.csv"
  rows = [dict_of([("k", "a,b"), ("v", "x")]), dict_of([("k", "plain"), ("v", "he said \"hi\"")])]
  _ = write_csv(path, rows)
  back = read_csv(path)
  _ = assert_eq(len(back), cast(2, i64), "written CSV reads back with 2 rows")
  _ = check_field(back, cast(0, i64), "k", "a,b", "quoted comma field round-trips")
  _ = check_field(back, cast(0, i64), "v", "x", "plain field round-trips")
  check_field(back, cast(1, i64), "v", "he said \"hi\"", "doubled quotes round-trip")
}
