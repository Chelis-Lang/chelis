module M.Main
def f(x: tensor[2, f32]) -> string = to_string(x)
out = print(f(to_tensor([1.5, 2.5])))
