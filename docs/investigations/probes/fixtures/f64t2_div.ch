module M.Main
def f(x: tensor[2, f64], y: tensor[2, f64]) -> tensor[2, f64] = div(x, y)
out = print(f(to_tensor([cast(1.0, f64), cast(2.0, f64)]), to_tensor([cast(3.0, f64), cast(3.0, f64)])))
