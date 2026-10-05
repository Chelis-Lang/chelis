module Std.Tensor.Construct
export (linspace, arange, stack, squeeze, unsqueeze)
sig linspace[n, p: Float]: p -> p -> i64 -> tensor[n, p]
def linspace(start, stop, count) =
  if not(and(eq(sub(start, start), cast(0, p)), eq(sub(stop, stop), cast(0, p)))) then fail("linspace: endpoints must be finite") else if lte(count, cast(0, i64)) then fail("linspace: count must be at least 1") else if eq(count, cast(1, i64)) then to_tensor([start]) else {
    denominator = sub(count, cast(1, i64))
    to_tensor(map(fn (idx) -> if eq(idx, cast(0, i64)) then start else if eq(idx, denominator) then stop else add(start, mul(sub(stop, start), linspace_weight(idx, denominator, cast(0, p), cast(0.5, p)))), range(cast(0, i64), count)))
  }
-- Generate the binary fraction with exact i64 remainders. Each stored prefix
-- is exact. Stop before adding an unrepresentable half-place, then use the
-- remainder and final bit for one ties-to-even rounding. Comparing remainder
-- with denominator-remainder avoids overflowing an i64 doubling. A positive
-- i64 ratio needs at most 63 leading places plus 53 significand places; fold
-- those 117 steps so evaluation never recurses with floating precision.
def linspace_weight[p: Float](remainder: i64, denominator: i64, value: p, place: p) -> p = {
  rounded = fold(fn (state: (i64, p, p, bool), unused: i64) -> if state.3 then state else if eq(state.0, cast(0, i64)) then (state.0, state.1, state.2, true) else {
    complement = sub(denominator, state.0)
    digit = gte(state.0, complement)
    next_remainder = if digit then sub(state.0, complement) else add(state.0, state.0)
    prefix = if digit then add(state.1, state.2) else state.1
    half_place = div(state.2, cast(2, p))
    if or(eq(half_place, cast(0, p)), neq(sub(add(prefix, half_place), prefix), half_place)) then {
      remaining_complement = sub(denominator, next_remainder)
      round_up = or(gt(next_remainder, remaining_complement), and(eq(next_remainder, remaining_complement), digit))
      (next_remainder, if round_up then add(prefix, state.2) else prefix, half_place, true)
    } else (next_remainder, prefix, half_place, false)
  }, (remainder, value, place, false), range(cast(0, i64), cast(117, i64)))
  if rounded.3 then rounded.1 else fail("linspace: exact weight exceeded integer precision bound")
}
sig arange[n, p: Int]: p -> p -> tensor[n, p]
def arange(start, stop) = {
  empty = skip([start], cast(1, i64))
  to_tensor(arange_values(start, stop, empty))
}
def arange_values[p: Int](current: p, stop: p, out: List[p]) -> List[p] = if gte(current, stop) then out else arange_values(add(current, cast(1, p)), stop, append(out, current))
sig stack[pre, post, p]: List[tensor[..pre, ..post, p]] -> i32 -> tensor[..pre, rows, ..post, p]
def stack(xs, axis) = concat(map(fn (x) -> unsqueeze(x, axis), xs), axis)
sig squeeze[pre, post, p]: &tensor[..pre, 1, ..post, p] -> i32 -> tensor[..pre, ..post, p]
def squeeze(x, axis) = {
  input_rank = cast(rank(x), i32)
  normalized = if gt(cast(0, i32), axis) then add(axis, input_rank) else axis
  target = shape_without_axis(x, cast(0, i32), input_rank, normalized, skip([cast(0, i64)], cast(1, i64)))
  reshape(x, target)
}
sig unsqueeze[pre, post, p]: &tensor[..pre, ..post, p] -> i32 -> tensor[..pre, 1, ..post, p]
def unsqueeze(x, axis) = {
  input_rank = cast(rank(x), i32)
  result_rank = add(input_rank, cast(1, i32))
  normalized = if gt(cast(0, i32), axis) then add(axis, result_rank) else axis
  target = shape_with_axis(x, cast(0, i32), input_rank, normalized, skip([cast(0, i64)], cast(1, i64)))
  reshape(x, target)
}
def shape_without_axis[r, p](x: &tensor[..r, p], cursor: i32, input_rank: i32, removed: i32, out: List[i64]) -> List[i64] = if gte(cursor, input_rank) then out else shape_without_axis(x, add(cursor, cast(1, i32)), input_rank, removed, if eq(cursor, removed) then out else append(out, shape(x, cursor)))
def shape_with_axis[r, p](x: &tensor[..r, p], cursor: i32, input_rank: i32, inserted: i32, out: List[i64]) -> List[i64] = if gt(cursor, input_rank) then out else if eq(cursor, inserted) then shape_with_axis(x, cursor, input_rank, sub(inserted, cast(1, i32)), append(out, cast(1, i64))) else if eq(cursor, input_rank) then out else shape_with_axis(x, add(cursor, cast(1, i32)), input_rank, inserted, append(out, shape(x, cursor)))
