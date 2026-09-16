module Std.Tests.Tensor.ReshapeRank
import Std.Test (assert_close_tensor, assert_eq)
def test_reshape_rank1_to_rank3_concrete_dims() -> unit ! { Test } = {
  flat = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)])
  shaped = (reshape(flat, [cast(2, i64), cast(1, i64), cast(3, i64)]) : tensor[2, 1, 3, f32])
  d0 = cast(shape(copy(shaped), cast(0, i32)), i64)
  d1 = cast(shape(copy(shaped), cast(1, i32)), i64)
  d2 = cast(shape(copy(shaped), cast(2, i32)), i64)
  _ = assert_eq(d0, cast(2, i64), "reshape rank-3 d0 = 2")
  _ = assert_eq(d1, cast(1, i64), "reshape rank-3 d1 = 1")
  _ = assert_eq(d2, cast(3, i64), "reshape rank-3 d2 = 3")
  back = reshape(shaped, [cast(6, i64)])
  expected = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)])
  assert_close_tensor(back, expected, cast(1e-6, f32), "reshape rank-3 back to rank-1 preserves elements")
}
