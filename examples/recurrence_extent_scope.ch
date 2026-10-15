module Repro.RecurrenceScope
type Model =
  | Model { blocks: List[f32] }
def stage[n](x: tensor[n, f32], blocks: List[f32]) -> tensor[n, f32] = fold(fn (acc: tensor[n, f32], v: f32) -> add(acc, insert(scalar_to_tensor(v), 0i32, shape(acc, 0i32))), x, blocks)
def loss(p: Model) -> f32 = {
  first = stage(to_tensor([1.0f32, 2.0f32]), p.blocks)
  second = stage(to_tensor([3.0f32, 4.0f32, 5.0f32]), p.blocks)
  add(tensor_to_scalar(sum(first, 0i32)), tensor_to_scalar(sum(second, 0i32)))
}
def step(p: Model) -> Model = grad(loss)(p)
def main() -> (f32, Model) = {
  p = Model { blocks: [2.0f32, 3.0f32] }
  (loss(p), step(p))
}
