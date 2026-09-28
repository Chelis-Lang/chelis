ids = to_tensor([9007199254740993i64, 9007199254740995i64])
reshaped_ids = reshape(ids, [1i64, 2i64])
empty_values: List[f64] = []
empty_tensor = to_tensor(empty_values)
empty_rows: List[List[f64]] = [[], []]
empty_matrix = to_tensor(empty_rows)
wide_empty = reshape(empty_tensor, [2147483648i64, 0i64])
small = to_tensor([1.0f32, 2.0f32])
increment = to_tensor([3.0f32, 4.0f32])
shifted = (neg(small) + increment)
exact_integers = cast(shifted, i64)
transposed_ids = permute(reshaped_ids, 1i32, 0i32)
repeated_ids = expand(reshaped_ids, 0i32, 3i64)
inserted_ids = insert(ids, 0i32, 2i64)
padded_ids = pad(reshaped_ids, [[1i64, 0i64], [0i64, 1i64]], 0i64)
trimmed_ids = shrink(padded_ids, [[1i64, 2i64], [0i64, 3i64]])
sampled_ids = stride(trimmed_ids, 1i64, 2i64)
-- A locally computed tensor extent keeps insert's own runtime guard identity.
sig repeat_local[n]: tensor[1, i64] -> tensor[n, i64] -> tensor[3, 1, i64]
def repeat_local(value, lengths) = {
  shorter = shrink(lengths, [[0i64, sub(shape(lengths, 0i32), 1i64)]])
  insert(value, 0i32, shape(shorter, 0i32))
}
local_insert = repeat_local(to_tensor([9007199254740993i64]), to_tensor([1i64, 2i64, 3i64, 4i64]))
-- Separate signatures retain dtype evidence for computed empty Lists.
sig empty_like[n, p: Numeric]: p -> tensor[n, p]
def empty_like(x) = to_tensor(skip([x], 1i64))
constructed_empty_f64 = empty_like(0.0f64)
constructed_empty_int64 = empty_like(0i64)
