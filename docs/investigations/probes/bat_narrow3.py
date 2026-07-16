# Sweep 1 battery 3: comparisons at the rounding collapse point + remaining scalar ops.
ROWS = [
    # true f16: cast(2049)->2048, so 2048 < 2049 must be FALSE after rounding
    (
        "f16_lt_collapse",
        "module M.Main\ndef run() -> bool = lt(cast(2048.0, f16), cast(2049.0, f16))\nout = print(run())\n",
    ),
    (
        "bf16_lt_collapse",
        "module M.Main\ndef run() -> bool = lt(cast(256.0, bf16), cast(257.0, bf16))\nout = print(run())\n",
    ),
    # scalar transcendentals / activations on f16 through print (host lane)
    (
        "f16_exp",
        "module M.Main\ndef run() -> f16 = exp(cast(1.0, f16))\nout = print(run())\n",
    ),
    (
        "f16_tanh",
        "module M.Main\ndef run() -> f16 = tanh(cast(1.0, f16))\nout = print(run())\n",
    ),
    (
        "f16_floor",
        "module M.Main\ndef run() -> f16 = floor(cast(1.5, f16))\nout = print(run())\n",
    ),
    (
        "f16_relu",
        "module M.Main\ndef run() -> f16 = relu(cast(1.5, f16))\nout = print(run())\n",
    ),
    (
        "f16_mod",
        "module M.Main\ndef run() -> f16 = mod(cast(5.5, f16), cast(2.0, f16))\nout = print(run())\n",
    ),
    (
        "f32_mod_ctl",
        "module M.Main\ndef run() -> f32 = mod(cast(5.5, f32), cast(2.0, f32))\nout = print(run())\n",
    ),
    # int8/int16 tensor add overflow, both lanes
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
    # scalar casts re-probe with fixed driver
    (
        "f16_cast_2049_redo",
        "module M.Main\ndef run() -> f16 = cast(2049.0, f16)\nout = print(run())\n",
    ),
    (
        "f16_roundtrip_f32",
        "module M.Main\ndef run() -> f32 = cast(cast(2049.0, f16), f32)\nout = print(run())\n",
    ),
    # to_list exit path for an f16 tensor (C lane misread probe beyond print)
    (
        "f16_to_list",
        "module M.Main\n"
        "def f(x: tensor[2, f32]) -> tensor[2, f16] = cast(x, f16)\n"
        "out = print(to_list(f(to_tensor([2049.0, 0.75]))))\n",
    ),
]
