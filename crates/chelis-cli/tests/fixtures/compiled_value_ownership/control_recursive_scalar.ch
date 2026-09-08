def step(state: int64, index: int64) -> int64 = if gte(index, 288i64) then state else step(add(state, 1i64), add(index, 1i64))
value = step(0i64, 0i64)
out: List[int64] = [value, value]
