module Std.Tensor.Construct
export (linspace, arange, stack, squeeze, unsqueeze)
def linspace[n](start: f32, stop: f32, count: int32) -> tensor[n, f32] = if lte(count, cast(1, int32)) then to_tensor([start]) else to_tensor(map(fn (i: int64) -> add(start, mul(sub(stop, start), div(cast(cast(i, int32), f32), cast(sub(count, cast(1, int32)), f32)))), range(cast(0, int64), cast(count, int64))))
def arange[n](start: int32, stop: int32) -> tensor[n, int32] = to_tensor(map(fn (i: int64) -> cast(cast(i, int32), int32), range(cast(start, int64), cast(stop, int64))))
def stack[d](xs: List[tensor[d, f32]]) = {
  concat(map(fn (x: tensor[d, f32]) -> {
    size = cast(shape(copy(x), cast(0, int32)), int64)
    reshape(x, [cast(1, int64), size])
  }, xs), cast(0, int32))
}
def squeeze[a, b](x: tensor[a, 1, b, f32]) = {
  outer = cast(shape(copy(x), cast(0, int32)), int64)
  inner = cast(shape(copy(x), cast(2, int32)), int64)
  reshape(x, [outer, inner])
}
def unsqueeze[a, b](x: tensor[a, b, f32]) = {
  outer = cast(shape(copy(x), cast(0, int32)), int64)
  inner = cast(shape(copy(x), cast(1, int32)), int64)
  reshape(x, [outer, cast(1, int64), inner])
}
