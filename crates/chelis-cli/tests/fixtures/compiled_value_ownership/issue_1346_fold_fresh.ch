values: List[i64] = [1i64, 2i64]
base = (0.0, 0.0)
picked = fold(fn (acc: (f32, f32), value: i64) -> (cast(value, f32), cast(value, f32)), base, values)
