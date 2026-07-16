module M.Main
def f(x: tensor[4, int64]) -> tensor[int64] = min_reduce(x, 0)
out = print(f(to_tensor([cast(100, int64), cast(400, int64), cast(200, int64), cast(50, int64)])))
