def step(state: i64, index: i64) -> i64 = if gte(index, 288i64) then state else step(add(state, 1i64), add(index, 1i64))
value = step(0i64, 0i64)
out: List[i64] = [value, value]
