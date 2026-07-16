module M.Main
def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = max_elem(x, y)
out = print(f(to_tensor([1.5, 0.25]), to_tensor([0.5, 2.5])))
