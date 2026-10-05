module RecursiveCasts
def step[p: Int](x: p, n: i64) -> p = if eq(n, 0i64) then x else (x |> add(cast(1, p)) |> step(sub(n, 1i64)))
def sum_steps[p: Float](x: p, n: i64) -> p = if eq(n, 0i64) then x else (x |> add(cast(n, p)) |> sum_steps(sub(n, 1i64)))
i32 = step(1i32, 3i64)
i64 = step(9007199254740993i64, 3i64)
f32 = sum_steps(0.5f32, 3i64)
f64 = sum_steps(0.5f64, 3i64)
