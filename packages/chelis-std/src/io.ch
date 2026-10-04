module Std.Io
export (read_text, write_text, read_trimmed_lines, read_head_bytes, exists, list, mmap_size)
def read_text(path: string) -> string ! { IO } = read_file(path)
def write_text(path: string, contents: string) -> unit ! { IO } = write_file(path, contents)
def read_trimmed_lines(path: string) -> List[string] ! { IO } = filter(fn (line: string) -> (line |> string_len |> gt(cast(0, i64))), map(fn (line: string) -> string_trim(line), read_lines(path)))
def read_head_bytes(path: string, width: i64) -> List[i64] ! { IO } = path |> mmap_file |> mmap_read(cast(0, i64), width)
def exists(path: string) -> bool ! { IO } = file_exists(path)
def list(path: string) -> List[string] ! { IO } = list_dir(path)
def mmap_size(path: string) -> i64 ! { IO } = path |> mmap_file |> mmap_len
