module Std.Tests.Runtime.Matmul
import Std.Test (assert_close_tensor)
def test_matmul_identity_passthrough() -> unit ! { Test } = {
  a = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
  b = pad_sequences_to([[cast(3.0, f32), cast(5.0, f32)], [cast(7.0, f32), cast(11.0, f32)]], cast(2, int64), cast(0.0, f32))
  out = matmul(a, b)
  flat = reshape(out, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(3.0, f32), cast(5.0, f32), cast(7.0, f32), cast(11.0, f32)]), cast(0.000001, f32), "I*B=B")
}
def test_matmul_2x2_2x2_basic() -> unit ! { Test } = {
  a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]], cast(2, int64), cast(0.0, f32))
  b = pad_sequences_to([[cast(5.0, f32), cast(6.0, f32)], [cast(7.0, f32), cast(8.0, f32)]], cast(2, int64), cast(0.0, f32))
  out = matmul(a, b)
  flat = reshape(out, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(19.0, f32), cast(22.0, f32), cast(43.0, f32), cast(50.0, f32)]), cast(0.000001, f32), "[[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]")
}
def test_matmul_rectangular_2x3_3x2() -> unit ! { Test } = {
  a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
  b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
  out = matmul(a, b)
  flat = reshape(out, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(4.0, f32), cast(5.0, f32), cast(10.0, f32), cast(11.0, f32)]), cast(0.000001, f32), "rectangular 2x3 * 3x2 sum check")
}
