-- Each value-root call supplies its own concrete dtype and tensor extent.
-- The comparison's bool result does not determine the operand dtype.
def above[p: Float](xs: tensor[3, p]) -> tensor[3, bool] = gt(xs, insert(scalar_to_tensor(cast(1.5, p)), 0i32, shape(xs, 0i32)))
out32 = [1.0f32, 2.0f32, 3.0f32] |> to_tensor |> above
out64 = [1.0f64, 2.0f64, 3.0f64] |> to_tensor |> above
