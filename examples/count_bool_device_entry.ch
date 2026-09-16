-- Parameterized device entry for [05-OP-29] multi-axis bool cardinality.
-- The result shape is tensor[3, i64]. HIP and Metal lower this definition
-- directly to their dedicated Count kernels.
def main(mask: tensor[2, 3, 5, bool]) -> tensor[3, i64] = count(&mask, 2, 0)
