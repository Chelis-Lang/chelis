def finite_positive[p: Float](x: p) -> bool = and(is_finite(x), gt(x, cast(0i64, p)))
def classify(x: tensor[4, f32]) -> tensor[4, bool] = is_finite(x)
values: tensor[4, f32] = to_tensor([0.0f32, -0.0f32, 1.0f32, -1.0f32])
finite = classify(values)
positive = finite_positive(1.0f64)
negative = finite_positive(-1.0f64)
nan = is_nan(1.0f32)
infinite = is_infinite(1.0f16)
predicate = is_finite
aliased = predicate(to_tensor([1.0f32, div(0.0f32, 0.0f32)]))
rank_zero = predicate(scalar_to_tensor(1.0f16))
def captured_finite(x: f32) -> bool = predicate(x)
captured = captured_finite(1.0f32)
runtime_shaped = predicate(stride(values, 2i64))
wrapped_runtime = Some(predicate(stride(values, 2i64)))
def choose_runtime(flag: bool, x: tensor[4, f32]) -> tensor[*, bool] = if flag then predicate(stride(x, 2i64)) else runtime_shaped
chosen_runtime = choose_runtime(true, values)
bound_runtime = choose_runtime(false, values)
def select_finite(flag: bool, x: tensor[4, f32]) -> tensor[4, bool] = {
  f: &tensor[4, f32] -> tensor[4, bool] = if flag then {
    local = predicate
    local
  } else {
    local = is_finite
    local
  }
  alias = f
  alias(x)
}
selected_finite = select_finite(true, values)
selected_finite_other = select_finite(false, values)
def finite_magnitude[r, p: Float](x: tensor[..r, p]) -> tensor[..r, bool] = {
  magnitude = abs(x)
  is_finite(magnitude)
}
composed_finite = finite_magnitude(values)
