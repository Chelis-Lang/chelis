module M.Main
def f(x: tensor[6, f32], w: int32, s: int32) -> tensor[5, f32] = reduce_window_max(x, [w], [s])
out = print(f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0]), 2, 1))
