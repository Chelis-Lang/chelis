# Issue-B edges: the newly-found stub ops at integer dtypes, plus mod/div controls.
ROWS = []
for op, x, y in [
    ("max_elem", "7", "3"),
    ("min_elem", "7", "3"),
    ("mod", "7", "3"),
    ("div", "7", "2"),
    ("floor_div", "7", "2"),
    ("trunc_div", "-7", "2"),
]:
    ROWS.append(
        (
            f"i64_{op}",
            f"module M.Main\ndef run() -> int64 = {op}(cast({x}, int64), cast({y}, int64))\nout = print(run())\n",
        )
    )
for op, x in [("floor", "5"), ("ceil", "5"), ("round", "5"), ("recip", "4"), ("abs", "-5"), ("neg", "5")]:
    ROWS.append(
        (
            f"i64_{op}",
            f"module M.Main\ndef run() -> int64 = {op}(cast({x}, int64))\nout = print(run())\n",
        )
    )
# int32 spot checks on the stub family
ROWS.append(
    (
        "i32_max_elem",
        "module M.Main\ndef run() -> int32 = max_elem(cast(7, int32), cast(3, int32))\nout = print(run())\n",
    )
)
