module Std.Io.Parquet
export (read_parquet, write_parquet)
sig read_parquet: string -> List[Dict[string, string]]
sig write_parquet: string -> List[Dict[string, string]] -> unit
def read_parquet(path: string) -> List[Dict[string, string]] = fail(string_concat("Std.Io.Parquet.read_parquet is not implemented: ", path))
def write_parquet(path: string, rows: List[Dict[string, string]]) -> unit = fail(string_concat("Std.Io.Parquet.write_parquet is not implemented: ", path))
