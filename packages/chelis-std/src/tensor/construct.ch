module Std.Tensor.Construct
export (linspace, arange, stack, squeeze, unsqueeze)
sig linspace[p: Float]: p -> p -> int64 -> tensor[n, p]
def linspace(start, stop, count) =
  if lte(count, cast(0, int64)) then fail("linspace: count must be at least 1") else if eq(count, cast(1, int64)) then to_tensor([start]) else {
    empty = drop([start], cast(1, int64))
    to_tensor(linspace_values(start, stop, count, cast(0, int64), empty))
  }
def linspace_values[p: Float](start: p, stop: p, count: int64, idx: int64, out: List[p]) -> List[p] =
  if gte(idx, count) then out else {
    weight = div(cast(idx, p), cast(sub(count, cast(1, int64)), p))
    value = if eq(idx, cast(0, int64)) then start else if eq(idx, sub(count, cast(1, int64))) then stop else add(start, mul(sub(stop, start), weight))
    linspace_values(start, stop, count, add(idx, cast(1, int64)), append(out, value))
  }
sig arange[p: Int]: p -> p -> tensor[n, p]
def arange(start, stop) = {
  empty = drop([start], cast(1, int64))
  to_tensor(arange_values(start, stop, empty))
}
def arange_values[p: Int](current: p, stop: p, out: List[p]) -> List[p] = if gte(current, stop) then out else arange_values(add(current, cast(1, p)), stop, append(out, current))
sig stack: List[tensor[..pre, ..post, p]] -> int32 -> tensor[..pre, rows, ..post, p]
def stack(xs, axis) = concat(map(fn (x) -> unsqueeze(x, axis), xs), axis)
sig squeeze: &tensor[..pre, 1, ..post, p] -> int32 -> tensor[..pre, ..post, p]
def squeeze(x, axis) = {
  input_rank = cast(rank(x), int32)
  normalized = if gt(cast(0, int32), axis) then add(axis, input_rank) else axis
  target = shape_without_axis(x, cast(0, int32), input_rank, normalized, drop([cast(0, int64)], cast(1, int64)))
  reshape(x, target)
}
sig unsqueeze: &tensor[..pre, ..post, p] -> int32 -> tensor[..pre, 1, ..post, p]
def unsqueeze(x, axis) = {
  input_rank = cast(rank(x), int32)
  result_rank = add(input_rank, cast(1, int32))
  normalized = if gt(cast(0, int32), axis) then add(axis, result_rank) else axis
  target = shape_with_axis(x, cast(0, int32), input_rank, normalized, drop([cast(0, int64)], cast(1, int64)))
  reshape(x, target)
}
def shape_without_axis[p](x: &tensor[..r, p], cursor: int32, input_rank: int32, removed: int32, out: List[int64]) -> List[int64] = if gte(cursor, input_rank) then out else shape_without_axis(x, add(cursor, cast(1, int32)), input_rank, removed, if eq(cursor, removed) then out else append(out, shape(x, cursor)))
def shape_with_axis[p](x: &tensor[..r, p], cursor: int32, input_rank: int32, inserted: int32, out: List[int64]) -> List[int64] = if gt(cursor, input_rank) then out else if eq(cursor, inserted) then shape_with_axis(x, cursor, input_rank, sub(inserted, cast(1, int32)), append(out, cast(1, int64))) else if eq(cursor, input_rank) then out else shape_with_axis(x, add(cursor, cast(1, int32)), input_rank, inserted, append(out, shape(x, cursor)))
