module M.Main
def f(x: tensor[2, f64]) -> tensor[2, f64] = exp(x)
out = print(f(to_tensor([1.5, 3.0])))
