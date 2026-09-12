module Annotated_Concat_Softmax
def join_columns[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([x, y], 1i32)
def probabilities[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) -> tensor[s, *, f32] = softmax(join_columns(x, y), -1)
output = probabilities(to_tensor([[0.0, 0.0], [0.0, 0.0]]), to_tensor([[0.0, 0.0], [0.0, 0.0]]))
