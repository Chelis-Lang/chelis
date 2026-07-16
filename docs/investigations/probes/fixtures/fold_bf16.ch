def pick() -> f32 = if lt(cast(256.0, bf16), cast(257.0, bf16)) then 111.0 else 222.0
out = pick()
