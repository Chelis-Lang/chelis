def f() -> f32 = with seed(42) { to_tensor([1.0, 2.0]) }
out = print(f())
