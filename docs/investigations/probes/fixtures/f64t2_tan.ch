module M.Main
def f(x: tensor[2, f64]) -> tensor[2, f64] = tan(x)
out = print(f(to_tensor([cast(1.5, f64), cast(3.0, f64)])))
