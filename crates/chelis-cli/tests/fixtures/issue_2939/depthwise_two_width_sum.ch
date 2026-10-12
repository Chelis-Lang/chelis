module Demo.Main
def depthwise7x7[a, c, h, w](x: &tensor[a, c, h, w, f32], kernel: &tensor[c, 1, 7, 7, f32]) -> tensor[a, c, h, w, f32] = {
  n_a = shape(x, 0i32)
  n_c = shape(x, 1i32)
  n_h = shape(x, 2i32)
  n_w = shape(x, 3i32)
  padded = pad(copy(x), [[0i64, 0i64], [0i64, 0i64], [3i64, 3i64], [3i64, 3i64]], 0.0f32)
  offsets = to_tensor(range(0i64, 7i64))
  h_idx = add(insert(to_tensor(range(0i64, n_h)), 1i32, 7i64), insert(&offsets, 0i32, n_h))
  w_idx = add(insert(to_tensor(range(0i64, n_w)), 1i32, 7i64), insert(offsets, 0i32, n_w))
  windows = permute(gather(gather(padded, h_idx, 2i32), w_idx, 4i32), 0i32, 1i32, 2i32, 4i32, 3i32, 5i32)
  weights = insert(insert(insert(reshape(copy(kernel), [n_c, 7i64, 7i64]), 0i32, n_a), 2i32, n_h), 3i32, n_w)
  sum(sum(mul(windows, weights), 5i32), 4i32)
}
def main() -> f32 = {
  x2 = reshape(insert(scalar_to_tensor(1.0f32), 0i32, 18i64), [1i64, 2i64, 3i64, 3i64])
  w2 = reshape(insert(scalar_to_tensor(0.1f32), 0i32, 98i64), [2i64, 1i64, 7i64, 7i64])
  x4 = reshape(insert(scalar_to_tensor(2.0f32), 0i32, 36i64), [1i64, 4i64, 3i64, 3i64])
  w4 = reshape(insert(scalar_to_tensor(0.1f32), 0i32, 196i64), [4i64, 1i64, 7i64, 7i64])
  a = tensor_to_scalar(sum(reshape(depthwise7x7(x2, w2), [18i64]), 0i32))
  b = tensor_to_scalar(sum(reshape(depthwise7x7(x4, w4), [36i64]), 0i32))
  add(a, b)
}
