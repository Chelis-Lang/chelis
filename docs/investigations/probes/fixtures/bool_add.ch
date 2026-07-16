module M.Main
def f(x: tensor[2, bool], y: tensor[2, bool]) -> tensor[2, bool] = add(x, y)
out = print(f(to_tensor([true, false]), to_tensor([true, true])))
