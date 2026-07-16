# Which f32 tensor ops skip f32 rounding in eval? 1/3 and 2/3 shapes expose it.
UNARY = ["recip", "exp", "log", "sqrt", "sin", "cos", "tan", "atan", "sigmoid", "relu", "tanh"]
BINARY = ["div", "add", "mul", "sub"]
ROWS = []
for op in UNARY:
    ROWS.append(
        (
            f"f32t_{op}",
            "module M.Main\n"
            f"def f(x: tensor[2, f32]) -> tensor[2, f32] = {op}(x)\n"
            "out = print(f(to_tensor([1.5, 3.0])))\n",
        )
    )
for op in BINARY:
    ROWS.append(
        (
            f"f32t_{op}",
            "module M.Main\n"
            f"def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = {op}(x, y)\n"
            "out = print(f(to_tensor([1.0, 2.0]), to_tensor([3.0, 3.0])))\n",
        )
    )
