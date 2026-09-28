-- Discrete coefficients remain exact forward values under transformations.
def coefficient(a: i32) -> i32 = bitxor(bitor(bitand(a, 3i32), shl(1i32, 2i32)), shr(8i32, 2i32))
def loss(x: tensor[f32]) -> tensor[f32] = mul(x, scalar_to_tensor(cast(coefficient(6i32), f32)))
derivative = grad(loss)(scalar_to_tensor(1.0f32))
mapped = vmap(loss)(to_tensor([1.0f32, 2.0f32]))
def toggle(x: tensor[i64]) -> tensor[i64] = scalar_to_tensor(bitxor(tensor_to_scalar(x), 1i64))
exact = vmap(toggle)(to_tensor([9007199254740992i64, -1i64]))
