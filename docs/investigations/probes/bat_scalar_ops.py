# Bound the scalar-op stub hole: every scalar op at f32 and f64, eval vs C.
UNARY = [
    ("abs", "-1.5"),
    ("neg", "1.5"),
    ("sqrt", "2.0"),
    ("exp", "1.0"),
    ("log", "2.0"),
    ("sin", "1.0"),
    ("cos", "1.0"),
    ("tan", "1.0"),
    ("atan", "1.0"),
    ("floor", "1.5"),
    ("ceil", "1.5"),
    ("round", "1.5"),
    ("relu", "1.5"),
    ("sigmoid", "1.0"),
    ("tanh", "1.0"),
    ("silu", "1.0"),
    ("gelu", "1.0"),
    ("recip", "4.0"),
]
BINARY = [
    ("add", "1.5", "0.25"),
    ("sub", "1.5", "0.25"),
    ("mul", "1.5", "0.25"),
    ("div", "1.5", "0.25"),
    ("max_elem", "1.5", "0.25"),
    ("min_elem", "1.5", "0.25"),
    ("pow", "2.0", "3.0"),
]

ROWS = []
for ty in ("f32", "f64"):
    for op, x in UNARY:
        ROWS.append(
            (
                f"{ty}_{op}",
                f"module M.Main\ndef run() -> {ty} = {op}(cast({x}, {ty}))\nout = print(run())\n",
            )
        )
    for op, x, y in BINARY:
        ROWS.append(
            (
                f"{ty}_{op}",
                f"module M.Main\ndef run() -> {ty} = {op}(cast({x}, {ty}), cast({y}, {ty}))\nout = print(run())\n",
            )
        )
