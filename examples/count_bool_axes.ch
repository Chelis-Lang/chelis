-- First-class bool cardinality across two axes.
-- The result is tensor[3, int64] with values [3, 3, 2].
mask: tensor[2, 3, 2, bool] = [[[true, false], [true, true], [false, false]], [[true, true], [false, true], [true, true]]]
counted: tensor[3, int64] = count(&mask, 2, 0)
-- A nontrailing reduction preserves result-axis order and exact int64 values.
values: tensor[2, 3, 2, int64] = [[[1i64, 2i64], [3i64, 4i64], [5i64, 6i64]], [[7i64, 8i64], [9i64, 10i64], [11i64, 12i64]]]
summed: tensor[2, 2, int64] = sum(&values, 1)
