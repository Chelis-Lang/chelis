module M.Main
def f(x: tensor[6, f32]) -> tensor[5, f32] = reduce_window_max(x, [2], [1])
out = print(f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])))
