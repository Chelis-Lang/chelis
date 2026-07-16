module M.Main
def f(x: tensor[2, f64], y: tensor[2, f64]) -> tensor[2, f64] = div(x, y)
out = print(f(to_tensor([1.0, 2.0]), to_tensor([3.0, 3.0])))
