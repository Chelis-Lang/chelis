def apply(callback: i8 -> i8, value: i8) -> i8 = callback(value)
out = apply(fn (item: i8) -> add(item, cast(1, i8)), cast(6, i8))
