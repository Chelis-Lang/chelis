-- The callable's named result agrees with its first tensor argument.
def broad(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = y
def invoke(f: tensor[seq, f32] -> tensor[*, f32] -> tensor[seq, f32], x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = f(x, y)
direct = to_tensor([1.0f32, 2.0f32]) |> broad(to_tensor([4.0f32, 5.0f32]))
retained = invoke(broad, to_tensor([1.0f32, 2.0f32]), to_tensor([4.0f32, 5.0f32]))
