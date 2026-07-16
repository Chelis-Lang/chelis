module M.Main
def f(xs: [int64]) -> ([int64], [int64]) = partition(xs, fn (v: int64) -> bool = gt(v, cast(2, int64)))
out = print(f([cast(1, int64), cast(3, int64), cast(2, int64), cast(4, int64)]))
