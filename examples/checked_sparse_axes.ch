-- Nontrailing Gather keeps exact i64 values, including values above 2^53.
base: tensor[2, 3, i64] = [[9007199254740993i64, 9007199254740994i64, 9007199254740995i64], [9007199254740996i64, 9007199254740997i64, 9007199254740998i64]]
indices: tensor[2, i64] = [2i64, 2i64]
selected: tensor[2, 2, i64] = gather(&base, &indices, 1)
