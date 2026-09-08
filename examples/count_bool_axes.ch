-- First-class bool cardinality across two axes.
-- The result is tensor[3, int64] with values [3, 3, 2].
mask: tensor[2, 3, 2, bool] = [[[true, false], [true, true], [false, false]], [[true, true], [false, true], [true, true]]]
counted: tensor[3, int64] = count(&mask, 2, 0)
