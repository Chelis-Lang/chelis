# Sweep 3: bitwise beyond int64, reductions across dtypes, Bool everywhere.
ROWS = []

# --- bitwise at every width, both lanes ---
for ty in ("int8", "int16", "int32", "int64"):
    ROWS.append(
        (
            f"bitand_{ty}",
            f"module M.Main\ndef run() -> {ty} = bitand(cast(12, {ty}), cast(10, {ty}))\nout = print(run())\n",
        )
    )
    ROWS.append(
        (
            f"shl_{ty}",
            f"module M.Main\ndef run() -> {ty} = shl(cast(1, {ty}), cast(4, {ty}))\nout = print(run())\n",
        )
    )
# shl past the width: int8 1 << 7 = -128 wrapped, 1 << 8 = ? (UB-ish)
ROWS.append(
    (
        "shl_i8_to_sign",
        "module M.Main\ndef run() -> int8 = shl(cast(1, int8), cast(7, int8))\nout = print(run())\n",
    )
)
ROWS.append(
    (
        "shl_i8_past_width",
        "module M.Main\ndef run() -> int8 = shl(cast(1, int8), cast(9, int8))\nout = print(run())\n",
    )
)
ROWS.append(
    (
        "shr_i64_neg",
        "module M.Main\ndef run() -> int64 = shr(cast(-16, int64), cast(2, int64))\nout = print(run())\n",
    )
)

# --- bool dtype ---
ROWS.append(
    (
        "bool_scalar_ops",
        "module M.Main\ndef run() -> bool = and(true, not(false))\nout = print(run())\n",
    )
)
ROWS.append(
    (
        "bool_tensor_roundtrip",
        "module M.Main\ndef f() -> tensor[3, bool] = to_tensor([true, false, true])\nout = print(f())\n",
    )
)
ROWS.append(
    (
        "bool_tensor_to_list",
        "module M.Main\nout = print(to_list(to_tensor([true, false, true])))\n",
    )
)
ROWS.append(
    (
        "bool_tensor_not",
        "module M.Main\ndef f(x: tensor[3, bool]) -> tensor[3, bool] = not(x)\nout = print(f(to_tensor([true, false, true])))\n",
    )
)
ROWS.append(
    (
        "bool_cast_int",
        "module M.Main\ndef run() -> int64 = cast(true, int64)\nout = print(run())\n",
    )
)
ROWS.append(
    (
        "cmplt_bool_tensor",
        "module M.Main\ndef f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, bool] = cmplt(x, y)\nout = print(f(to_tensor([1.0, 3.0]), to_tensor([2.0, 2.0])))\n",
    )
)

# --- reductions across dtypes, both lanes ---
for op, expect_desc in (("sum", ""), ("max_reduce", ""), ("min_reduce", ""), ("prod_reduce", ""), ("mean", ""), ("argmax_reduce", ""), ("argmin_reduce", "")):
    ROWS.append(
        (
            f"red_{op}_f32",
            "module M.Main\n"
            f"def f(x: tensor[4, f32]) -> tensor[{'int64' if 'arg' in op else 'f32'}] = {op}(x, 0)\n"
            "out = print(f(to_tensor([1.5, 4.5, 2.5, 0.5])))\n",
        )
    )
for op in ("sum", "max_reduce", "prod_reduce"):
    ROWS.append(
        (
            f"red_{op}_i64",
            "module M.Main\n"
            f"def f(x: tensor[4, int64]) -> tensor[int64] = {op}(x, 0)\n"
            "out = print(f(to_tensor([cast(100, int64), cast(400, int64), cast(200, int64), cast(50, int64)])))\n",
        )
    )
# int64 sum exactness at 2^53 (tensor reduction via f64 accumulation?)
ROWS.append(
    (
        "red_sum_i64_2p53",
        "module M.Main\n"
        "def f(x: tensor[2, int64]) -> tensor[int64] = sum(x, 0)\n"
        "out = print(f(to_tensor([cast(9007199254740992, int64), cast(1, int64)])))\n",
    )
)
