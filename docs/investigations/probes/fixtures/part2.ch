xs: List[int64] = [cast(1, int64), cast(3, int64), cast(2, int64), cast(4, int64)]
buckets = partition(fn (x: int64) -> gt(x, cast(2, int64)), xs)
out = print(buckets)
