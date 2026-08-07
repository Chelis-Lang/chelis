def choose[a](x: a, n: int32) -> a = if (n <= 0) then x else choose(x, (n - 1))
def concrete() -> int32 = if choose(true, 2) then choose(5, 3) else 0
out = print(concrete())
