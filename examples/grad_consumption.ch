type Model =
  | Model { value: tensor[2, f32] }
def zero(m: &Model) -> f32 = 0.0f32
def loss(m: &Model) -> f32 = tensor_to_scalar(sum(mul(m.value, m.value), 0i32))
def zeros(m: Model) -> Model = grad(zero)(m)
def active(m: Model) -> Model = grad(loss)(m)
def inspect(m: &Model) -> f32 = tensor_to_scalar(sum(m.value, 0i32))
def inspect_again(m: &Model) -> f32 = inspect(m)
m = Model { value: to_tensor([1.0f32, 2.0f32]) }
z = zeros(m)
a = active(m)
b = active(Model { value: to_tensor([3.0f32, 5.0f32]) })
zero_sum = inspect(z)
active_sum = inspect(a)
second_sum = inspect_again(b)
primal_sum = inspect(m)
zero_again = inspect_again(z)
active_again = inspect_again(a)
