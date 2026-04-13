module Std.IO
export (read_text, write_text, read_trimmed_lines, read_head_bytes, exists, list, mmap_size)
def read_text(path: string) -> string = read_file(path)
def write_text(path: string, contents: string) -> unit = write_file(path, contents)
def read_trimmed_lines(path: string) -> List[string] = filter(fn (line: string) -> gt(string_len(line), cast(0, int64)), map(fn (line: string) -> string_trim(line), read_lines(path)))
def read_head_bytes(path: string, width: int64) -> List[int64] = mmap_read(mmap_file(path), cast(0, int64), width)
def exists(path: string) -> bool = file_exists(path)
def list(path: string) -> List[string] = list_dir(path)
def mmap_size(path: string) -> int64 = mmap_len(mmap_file(path))
