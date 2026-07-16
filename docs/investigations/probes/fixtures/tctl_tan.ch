module M.Main
def f(x: tensor[2, f32]) -> tensor[2, f32] = tan(x)
out = print(f(to_tensor([1.5, 0.25])))
