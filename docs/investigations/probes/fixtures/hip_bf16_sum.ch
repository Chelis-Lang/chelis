def f(a: tensor[2, 2, bf16]) -> tensor[2, bf16] = sum(a, 0)
