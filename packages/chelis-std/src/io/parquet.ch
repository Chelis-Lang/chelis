module Std.Io.Parquet
export (read_parquet, write_parquet)
sig read_parquet: string -> List[Dict[string, string]]
sig write_parquet: string -> List[Dict[string, string]] -> ()
