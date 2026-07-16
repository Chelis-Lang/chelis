module M.Main
def f(x: tensor[2, int64]) -> int64 = sub(to_list(sum(x, 0))[0], cast(9007199254740992, int64))
out = print(f(to_tensor([cast(9007199254740992, int64), cast(1, int64)])))
