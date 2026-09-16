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
            f"module M.Main\ndef run() -> i64 = {op}(cast({x}, i64), cast({y}, i64))\nout = print(run())\n",
        )
    )
for op, x in [("floor", "5"), ("ceil", "5"), ("round", "5"), ("recip", "4"), ("abs", "-5"), ("neg", "5")]:
    ROWS.append(
        (
            f"i64_{op}",
            f"module M.Main\ndef run() -> i64 = {op}(cast({x}, i64))\nout = print(run())\n",
        )
    )
# i32 spot checks on the stub family
ROWS.append(
    (
        "i32_max_elem",
        "module M.Main\ndef run() -> i32 = max_elem(cast(7, i32), cast(3, i32))\nout = print(run())\n",
    )
)
