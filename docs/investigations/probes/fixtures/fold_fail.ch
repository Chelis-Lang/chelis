def pick() -> f32 = if lt(9007199254740992i64, 9007199254740993i64) then fail("int64 invariant violated") else 222.0
out = pick()
