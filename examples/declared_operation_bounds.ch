-- A generic definition states the dtype family its operations require.
sig average3[p: Float]: tensor[3, p] -> tensor[p]
def average3(values) = mean(values, 0i32)
-- A wrapper preserves that contract, even without naming the primitive.
def summarize[p: Float](values: tensor[3, p]) -> tensor[p] = average3(values)
float_result = to_tensor([1.0f32, 2.0f32, 6.0f32]) |> summarize
double_result = to_tensor([1.0f64, 2.0f64, 6.0f64]) |> summarize
-- Window reductions carry the same operation contract through their
-- specialized shape checker.
def moving_sum[p: Numeric](values: tensor[3, p]) -> tensor[2, p] = reduce_window_sum(values, [2i64], [1i64])
window_result = to_tensor([1.0f32, 2.0f32, 4.0f32]) |> moving_sum
-- A local hole binds inside its declaration; it does not publish a new bound.
def local_sine(x: f32) -> f32 = {
  op = fn (value) -> sin(value)
  op(x)
}
local_result = local_sine(0.0f32)
