module Examples.SoftmaxAdjoint
def loss(x: tensor[4, f32]) -> tensor[f32] = sum(mul(softmax(x, 0i32), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])), 0i32)
x = to_tensor([0.1f32, 0.7f32, 1.3f32, 2.9f32])
y = softmax(x, 0i32)
w = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])
actual = grad(loss)(x)
expected = mul(y, sub(w, insert(cast(sum(mul(w, y), 0i32), f32), 0i32, 4i64)))
