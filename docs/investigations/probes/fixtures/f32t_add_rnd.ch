module M.Main
def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = add(x, y)
out = print(f(to_tensor([0.1, 1.0]), to_tensor([0.2, 2.0])))
