module Std.Tests.Tensor.ReshapeRank
import Std.Test (assert_close_tensor, assert_eq_int)
def test_reshape_rank1_to_rank3_concrete_dims() -> unit ! { Test } = {
  flat = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)])
  shaped = (reshape(flat, [cast(2, int64), cast(1, int64), cast(3, int64)]) : tensor[2, 1, 3, f32])
  d0 = cast(shape(copy(shaped), cast(0, int32)), int64)
  d1 = cast(shape(copy(shaped), cast(1, int32)), int64)
  d2 = cast(shape(copy(shaped), cast(2, int32)), int64)
  _ = assert_eq_int(d0, cast(2, int64), "reshape rank-3 d0 = 2")
  _ = assert_eq_int(d1, cast(1, int64), "reshape rank-3 d1 = 1")
  _ = assert_eq_int(d2, cast(3, int64), "reshape rank-3 d2 = 3")
  back = reshape(shaped, [cast(6, int64)])
  expected = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)])
  assert_close_tensor(back, expected, cast(1e-6, f32), "reshape rank-3 back to rank-1 preserves elements")
}
