def bf16_matmul(x: tensor[8, 16, bf16], y: tensor[16, 32, bf16]) -> tensor[8, 32, bf16] = matmul(x, y)
