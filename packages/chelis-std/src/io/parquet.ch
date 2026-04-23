module Std.IO.Parquet
export (read_parquet, write_parquet)
def read_parquet(path: string) -> List[Dict[string, string]] = fail(string_concat("read_parquet: Parquet I/O not yet implemented: ", path))
def write_parquet(path: string, rows: List[Dict[string, string]]) -> unit = fail(string_concat("write_parquet: Parquet I/O not yet implemented: ", path))
