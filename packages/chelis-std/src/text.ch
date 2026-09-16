module Std.Text
export (join)
def join(parts: List[string], sep: string) -> string = if eq(len(parts), cast(0, i64)) then "" else fold(fn (acc: string, part: string) -> string_concat(acc, string_concat(sep, part)), index(parts, cast(0, i64)), drop(parts, cast(1, i64)))
