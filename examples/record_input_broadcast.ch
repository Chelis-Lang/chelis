-- A multi-input forward takes its arguments as a record, so every
-- symbolic-extent broadcast inside it sizes an axis from a field's shape
-- (chelis#1266). `shape(inputs.features, 0i32)` is admissible exactly where
-- `shape(x, 0i32)` is: spec/04-type-system.md §4.7.2 admits an extent by
-- what supplies it, never by how the read is spelled.
type ForwardInputs =
  | ForwardInputs { features: tensor[batch, 4, f32], bias: tensor[1, 4, f32] }
sig forward: ForwardInputs -> tensor[batch, 4, f32]
def forward(inputs: ForwardInputs) = add(inputs.features, expand(inputs.bias, 0i32, shape(inputs.features, 0i32)))
def main() -> tensor[batch, 4, f32] = forward(ForwardInputs { features: to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]), bias: to_tensor([[0.25f32, 0.25f32, 0.25f32, 0.25f32]]) })
