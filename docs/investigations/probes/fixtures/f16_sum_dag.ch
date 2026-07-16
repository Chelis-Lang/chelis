module M.Main
def f(x: tensor[2, 2, f32]) -> tensor[2, f16] = sum(cast(x, f16), 0)
out = print(f(to_tensor([[2048.0, 0.5], [1.0, 0.25]])))
