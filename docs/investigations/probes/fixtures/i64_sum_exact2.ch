module M.Main
def f(x: tensor[2, int64]) -> tensor[int64] = sum(x, 0)
out = print(to_list(f(to_tensor([cast(9007199254740992, int64), cast(1, int64)]))))
