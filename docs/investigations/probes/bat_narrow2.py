# Sweep 1 battery 2: DAG kernel lane for f16/bf16 (tensor cast entry), comparisons,
# and the remaining scalar op surface.
ROWS = [
    # tensor cast into f16/bf16 (avoids the to_tensor host literal abort?)
    (
        "f16_tensor_via_cast",
        "module M.Main\n"
        "def f(x: tensor[2, f32]) -> tensor[2, f16] = cast(x, f16)\n"
        "out = print(f(to_tensor([2049.0, 0.75])))\n",
    ),
    (
        "f16_tensor_add_via_cast",
        "module M.Main\n"
        "def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f16] = add(cast(x, f16), cast(y, f16))\n"
        "out = print(f(to_tensor([2048.0, 0.5]), to_tensor([1.0, 0.25])))\n",
    ),
    (
        "bf16_tensor_add_via_cast",
        "module M.Main\n"
        "def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, bf16] = add(cast(x, bf16), cast(y, bf16))\n"
        "out = print(f(to_tensor([256.0, 0.5]), to_tensor([1.0, 0.25])))\n",
    ),
    (
        "f16_tensor_mul_via_cast",
        "module M.Main\n"
        "def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f16] = mul(cast(x, f16), cast(y, f16))\n"
        "out = print(f(to_tensor([0.1, 3.0]), to_tensor([0.1, 7.0])))\n",
    ),
    # comparisons at the f16 boundary: 2049 and 2050 both round to 2048 in true f16
    ("f16_lt_collapsed", "lt(cast(2049.0, f16), cast(2050.0, f16))", "bool"),
    ("f16_gte_collapsed", "gte(cast(2049.0, f16), cast(2050.0, f16))", "bool"),
    ("bf16_lt_collapsed", "lt(cast(257.0, bf16), cast(258.0, bf16))", "bool"),
    # scalar transcendentals / activations on f16
    ("f16_exp", "exp(cast(1.0, f16))", "f16"),
    ("f16_tanh", "tanh(cast(1.0, f16))", "f16"),
    ("f16_floor", "floor(cast(1.5, f16))", "f16"),
    ("f16_relu", "relu(cast(1.5, f16))", "f16"),
    # mod on f16 (mod is Int64-typed per infer_builtin_host_type_from_arg_tys!)
    ("f16_mod", "mod(cast(5.5, f16), cast(2.0, f16))", "f16"),
    # f32_mod control
    ("f32_mod", "mod(cast(5.5, f32), cast(2.0, f32))", "f32"),
    # int8 tensor ops: does the tensor lane wrap at width in each lane?
    (
        "int8_tensor_add_overflow",
        "module M.Main\n"
        "def f(x: tensor[2, int8], y: tensor[2, int8]) -> tensor[2, int8] = add(x, y)\n"
        "out = print(f(to_tensor([cast(100, int8), cast(1, int8)]), to_tensor([cast(100, int8), cast(2, int8)])))\n",
    ),
    (
        "int16_tensor_add_overflow",
        "module M.Main\n"
        "def f(x: tensor[2, int16], y: tensor[2, int16]) -> tensor[2, int16] = add(x, y)\n"
        "out = print(f(to_tensor([cast(30000, int16), cast(1, int16)]), to_tensor([cast(30000, int16), cast(2, int16)])))\n",
    ),
    # f16 scalar cast alone (re-probe with fixed driver)
    ("f16_cast_2049_redo", "cast(2049.0, f16)", "f16"),
    ("bf16_cast_257_redo", "cast(257.0, bf16)", "bf16"),
    # what about casting f16 back up to f32? Does the narrowing get applied then?
    ("f16_roundtrip_f32", "cast(cast(2049.0, f16), f32)", "f32"),
]
