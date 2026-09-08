def apply(callback: int8 -> int8, value: int8) -> int8 = callback(value)
out = apply(fn (item: int8) -> add(item, cast(1, int8)), cast(6, int8))
