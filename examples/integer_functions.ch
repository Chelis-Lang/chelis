module IntegerFunctions
def anchor() -> i32 = 7
def user() -> i32 = anchor()
def is_zero_i64(x: i64) -> bool = {
  classify = fn (value) -> match value with {
    | 0 => true
    | _ => false
  }
  classify(x)
}
zero_checked = 0i64 |> is_zero_i64
def weighted_loss(x: tensor[4, f32]) -> tensor[f32] = {
  weights = cast(abs(to_tensor([-100i64, 200i64, -300i64, 400i64])), f32)
  x |> mul(weights) |> sum(0i32)
}
def integer_weight_gradient(x: tensor[4, f32]) -> tensor[4, f32] = grad(weighted_loss, wrt=x)(x)
weight_gradient = [0.1f32, 0.2f32, 0.3f32, 0.4f32] |> to_tensor |> integer_weight_gradient
