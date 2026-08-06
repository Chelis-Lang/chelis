type Box[a] =
  | Empty
  | Full { value: a }
def depth[a](box: Box[a], n: int32) -> int32 = if (n <= 0) then 0 else (depth(box, (n - 1)) + 1)
def concrete() -> int32 = (depth(Full { value: cast(7, int32) }, 2) + depth(Full { value: true }, 3))
out = print(concrete())
