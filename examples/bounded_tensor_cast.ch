def convert[p: Numeric](values: tensor[3, int32], witness: p) -> tensor[3, p] = cast(values, p)
as_float = convert(to_tensor([1, 2, 3]), 0.0f64)
as_integer = convert(to_tensor([1, 2, 3]), 0i64)
out = print((as_float, as_integer))
