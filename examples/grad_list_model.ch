-- Each block gradient retains its tensor shape and its position in the List.
import Std.Index (list_index)
type Block =
  | Block { weight: tensor[*, f32] }
type Model =
  | Model { blocks: List[Block] }
def loss(model: Model) -> f32 = {
  first = list_index(model.blocks, 0i64)
  second = list_index(model.blocks, 1i64)
  a = tensor_to_scalar(sum(mul(first.weight, first.weight), 0i32))
  b = tensor_to_scalar(sum(mul(second.weight, second.weight), 0i32))
  add(a, b)
}
def step(model: Model) -> Model = grad(loss)(model)
out = step(Model { blocks: [Block { weight: to_tensor([3.0f32, 4.0f32]) }, Block { weight: to_tensor([5.0f32]) }] })
