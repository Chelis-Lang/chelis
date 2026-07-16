# Sweep 1 battery: narrow floats (f16/bf16) and narrow ints (int8/16/32) scalars + tensors.
ROWS = [
    # --- f16 scalar boundary/rounding, both lanes ---
    ("f16_add_2048_1", "add(cast(2048.0, f16), cast(1.0, f16))", "f16"),
    ("f16_add_chain", "add(add(cast(2048.0, f16), cast(1.0, f16)), cast(1.0, f16))", "f16"),
    ("f16_frac", "add(cast(0.5, f16), cast(0.25, f16))", "f16"),
    ("f16_mul_frac", "mul(cast(0.1, f16), cast(0.1, f16))", "f16"),
    ("f16_overflow_inf", "mul(cast(65504.0, f16), cast(2.0, f16))", "f16"),
    ("f16_cast_2049", "cast(2049.0, f16)", "f16"),
    ("f16_sub", "sub(cast(1.5, f16), cast(0.25, f16))", "f16"),
    ("f16_div", "div(cast(1.0, f16), cast(3.0, f16))", "f16"),
    ("f16_neg", "neg(cast(1.5, f16))", "f16"),
    ("f16_abs", "abs(cast(-1.5, f16))", "f16"),
    ("f16_maxelem", "max_elem(cast(1.5, f16), cast(2.5, f16))", "f16"),
    ("f16_sqrt", "sqrt(cast(2.0, f16))", "f16"),
    # --- bf16 scalar ---
    ("bf16_add_256_1", "add(cast(256.0, bf16), cast(1.0, bf16))", "bf16"),
    ("bf16_frac", "add(cast(0.5, bf16), cast(0.25, bf16))", "bf16"),
    ("bf16_mul_frac", "mul(cast(0.1, bf16), cast(0.1, bf16))", "bf16"),
    ("bf16_cast_257", "cast(257.0, bf16)", "bf16"),
    # --- f16/bf16 tensor lane (tensor storage has CHELIS_F16/BF16) ---
    (
        "f16_tensor_roundtrip",
        "module M.Main\n"
        "def f() -> tensor[2, f16] = to_tensor([cast(2049.0, f16), cast(0.75, f16)])\n"
        "out = print(f())\n",
    ),
    (
        "f16_tensor_add_boundary",
        "module M.Main\n"
        "def f(x: tensor[2, f16], y: tensor[2, f16]) -> tensor[2, f16] = add(x, y)\n"
        "out = print(f(to_tensor([cast(2048.0, f16), cast(0.5, f16)]), to_tensor([cast(1.0, f16), cast(0.25, f16)])))\n",
    ),
    (
        "bf16_tensor_add_boundary",
        "module M.Main\n"
        "def f(x: tensor[2, bf16], y: tensor[2, bf16]) -> tensor[2, bf16] = add(x, y)\n"
        "out = print(f(to_tensor([cast(256.0, bf16), cast(0.5, bf16)]), to_tensor([cast(1.0, bf16), cast(0.25, bf16)])))\n",
    ),
    (
        "f16_tensor_sum",
        "module M.Main\n"
        "def f(x: tensor[4, f16]) -> f16 = sum(x)\n"
        "out = print(f(to_tensor([cast(2048.0, f16), cast(1.0, f16), cast(1.0, f16), cast(1.0, f16)])))\n",
    ),
    # --- narrow int scalars: width semantics across lanes ---
    ("int8_add_overflow", "add(cast(100, int8), cast(100, int8))", "int8"),
    ("int16_add_overflow", "add(cast(30000, int16), cast(30000, int16))", "int16"),
    ("int32_add_overflow", "add(cast(2000000000, int32), cast(2000000000, int32))", "int32"),
    ("int8_mul_overflow", "mul(cast(16, int8), cast(16, int8))", "int8"),
    ("int8_neg_min", "neg(cast(-128, int8))", "int8"),
    # --- f8e4m3: documented as rejected; is it? ---
    ("f8_cast", "cast(1.0, f8e4m3)", "f8e4m3"),
    (
        "f8_tensor",
        "module M.Main\n"
        "def f() -> tensor[2, f8e4m3] = to_tensor([cast(1.0, f8e4m3), cast(2.0, f8e4m3)])\n"
        "out = print(f())\n",
    ),
]
