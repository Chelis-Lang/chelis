def pick() -> f32 = if lt(cast(2048.0, f16), cast(2049.0, f16)) then 111.0 else 222.0
out = print(pick())
