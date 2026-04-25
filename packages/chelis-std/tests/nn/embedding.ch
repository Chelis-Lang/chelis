module Std.Tests.Nn.Embedding
import Std.Nn.Embedding (forward)
import Std.Test (assert_close_tensor, assert_eq_int)
def test_single_token_lookup_returns_row() -> unit ! { Test } = {
  table = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(1.0, f32)], [cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)]], cast(3, int64), cast(0.0, f32))
  ids = pad_sequences_to([[cast(0, int64)]], cast(1, int64), cast(0, int64))
  out = forward(ids, table)
  flat = reshape(out, [cast(3, int64)])
  expected = to_tensor([cast(1.0, f32), cast(0.0, f32), cast(0.0, f32)])
  assert_close_tensor(flat, expected, cast(0.000001, f32), "embed[ids=[[0]]] equals table row 0")
}
def test_distinct_rows_preserved() -> unit ! { Test } = {
  table = pad_sequences_to([[cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)], [cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(3.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(4.0, f32), cast(4.0, f32)]], cast(3, int64), cast(0.0, f32))
  ids = pad_sequences_to([[cast(0, int64), cast(2, int64)]], cast(2, int64), cast(0, int64))
  out = forward(ids, table)
  flat = reshape(out, [cast(6, int64)])
  expected = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(3.0, f32), cast(3.0, f32), cast(3.0, f32)])
  assert_close_tensor(flat, expected, cast(0.000001, f32), "embed[ids=[[0,2]]] equals rows A and C")
}
def test_repeated_id_returns_same_row_twice() -> unit ! { Test } = {
  table = pad_sequences_to([[cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)], [cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(3.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(4.0, f32), cast(4.0, f32)]], cast(3, int64), cast(0.0, f32))
  ids = pad_sequences_to([[cast(1, int64), cast(1, int64)]], cast(2, int64), cast(0, int64))
  out = forward(ids, table)
  flat = reshape(out, [cast(6, int64)])
  expected = to_tensor([cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)])
  assert_close_tensor(flat, expected, cast(0.000001, f32), "embed[ids=[[1,1]]] equals row B twice")
}
def test_shape_contract_batch_seq_hidden() -> unit ! { Test } = {
  table = pad_sequences_to([[cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)]], cast(4, int64), cast(0.0, f32))
  ids = pad_sequences_to([[cast(0, int64), cast(1, int64), cast(2, int64)], [cast(3, int64), cast(4, int64), cast(5, int64)]], cast(3, int64), cast(0, int64))
  out = forward(ids, table)
  d0 = cast(shape(copy(out), cast(0, int32)), int64)
  d1 = cast(shape(copy(out), cast(1, int32)), int64)
  d2 = cast(shape(copy(out), cast(2, int32)), int64)
  _ = assert_eq_int(d0, cast(2, int64), "embed shape axis 0 = batch = 2")
  _ = assert_eq_int(d1, cast(3, int64), "embed shape axis 1 = seq = 3")
  assert_eq_int(d2, cast(4, int64), "embed shape axis 2 = hidden = 4")
}
