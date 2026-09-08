module Std.Tests.Runtime.Permute
import Std.Test (assert_close_tensor)
def test_permute_2x2_transpose() -> unit ! { Test } = {
  a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]], cast(2, int64), cast(0.0, f32))
  t = permute(a, 1, 0)
  flat = reshape(t, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(3.0, f32), cast(2.0, f32), cast(4.0, f32)]), cast(1e-6, f32), "transpose [[1,2],[3,4]] = [[1,3],[2,4]]")
}
def test_permute_2x3_transpose() -> unit ! { Test } = {
  a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
  t = permute(a, 1, 0)
  flat = reshape(t, [cast(6, int64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(4.0, f32), cast(2.0, f32), cast(5.0, f32), cast(3.0, f32), cast(6.0, f32)]), cast(1e-6, f32), "transpose 2x3 -> 3x2 column-major flatten")
}
def test_permute_identity_keeps_values() -> unit ! { Test } = {
  a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]], cast(2, int64), cast(0.0, f32))
  t = permute(a, 0, 1)
  flat = reshape(t, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]), cast(1e-6, f32), "permute identity is no-op")
}
