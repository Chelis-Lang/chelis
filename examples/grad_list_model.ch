-- A model keeps the runtime length and order of its block gradients.
import Std.Index (list_index)
type Block =
  | Block { weight: f32 }
type Model =
  | Model { blocks: List[Block] }
def loss(model: Model) -> f32 = {
  first = list_index(model.blocks, 0i64)
  second = list_index(model.blocks, 1i64)
  add(mul(first.weight, first.weight), mul(second.weight, second.weight))
}
def step(model: Model) -> Model = grad(loss)(model)
out = step(Model { blocks: [Block { weight: 3.0f32 }, Block { weight: 4.0f32 }] })
