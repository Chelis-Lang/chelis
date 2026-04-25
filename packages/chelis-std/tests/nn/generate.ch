module Std.Tests.Nn.Generate
import Std.Nn.Generate (KVCache, generate)
import Std.Test (assert_eq_int)
def keep_cache[p](cache: Option[KVCache[p]]) -> KVCache[p] = match cache with {
  | Some(value) => value
  | None => KVCache([])
}
def model_always_token_0[batch, seq](input_ids: tensor[batch, seq, int64], cache: Option[KVCache[tensor[kv, f32]]]) = {
  logits = (pad_sequences_to([[cast(10.0, f32), cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]], cast(4, int64), cast(0.0, f32)) : tensor[1, 4, f32])
  (logits, keep_cache(cache))
}
def test_greedy_trivial_model_appends_zeros() -> unit ! { Test } = {
  context = (pad_sequences_to([[cast(5, int64)]], cast(1, int64), cast(0, int64)) : tensor[1, 1, int64])
  out = (generate(model_always_token_0, context, cast(3, int64)) : tensor[1, 4, int64])
  flat = reshape(out, [cast(4, int64)])
  ids = to_list(flat)
  _ = assert_eq_int(len(ids), cast(4, int64), "greedy out length = 1 (context) + 3 (new) = 4")
  _ = assert_eq_int(index(ids, cast(0, int64)), cast(5, int64), "context start token preserved at index 0")
  _ = assert_eq_int(index(ids, cast(1, int64)), cast(0, int64), "first generated token is 0 (argmax)")
  _ = assert_eq_int(index(ids, cast(2, int64)), cast(0, int64), "second generated token is 0 (argmax)")
  assert_eq_int(index(ids, cast(3, int64)), cast(0, int64), "third generated token is 0 (argmax)")
}
def test_greedy_max_tokens_length_honored() -> unit ! { Test } = {
  context = (pad_sequences_to([[cast(7, int64), cast(8, int64)]], cast(2, int64), cast(0, int64)) : tensor[1, 2, int64])
  out = (generate(model_always_token_0, context, cast(5, int64)) : tensor[1, 7, int64])
  ids = to_list(reshape(out, [cast(7, int64)]))
  _ = assert_eq_int(len(ids), cast(7, int64), "seq length = context(2) + max_tokens(5) = 7")
  _ = assert_eq_int(index(ids, cast(2, int64)), cast(0, int64), "appended token 1 is 0")
  _ = assert_eq_int(index(ids, cast(3, int64)), cast(0, int64), "appended token 2 is 0")
  _ = assert_eq_int(index(ids, cast(4, int64)), cast(0, int64), "appended token 3 is 0")
  _ = assert_eq_int(index(ids, cast(5, int64)), cast(0, int64), "appended token 4 is 0")
  assert_eq_int(index(ids, cast(6, int64)), cast(0, int64), "appended token 5 is 0")
}
def test_greedy_preserves_starting_context() -> unit ! { Test } = {
  context = (pad_sequences_to([[cast(11, int64), cast(22, int64), cast(33, int64)]], cast(3, int64), cast(0, int64)) : tensor[1, 3, int64])
  out = (generate(model_always_token_0, context, cast(2, int64)) : tensor[1, 5, int64])
  ids = to_list(reshape(out, [cast(5, int64)]))
  _ = assert_eq_int(len(ids), cast(5, int64), "total length = 3 + 2 = 5")
  _ = assert_eq_int(index(ids, cast(0, int64)), cast(11, int64), "context[0] preserved")
  _ = assert_eq_int(index(ids, cast(1, int64)), cast(22, int64), "context[1] preserved")
  assert_eq_int(index(ids, cast(2, int64)), cast(33, int64), "context[2] preserved")
}
def test_greedy_is_deterministic_across_calls() -> unit ! { Test } = {
  context = (pad_sequences_to([[cast(4, int64), cast(2, int64)]], cast(2, int64), cast(0, int64)) : tensor[1, 2, int64])
  out_a = (generate(model_always_token_0, copy(context), cast(3, int64)) : tensor[1, 5, int64])
  out_b = (generate(model_always_token_0, context, cast(3, int64)) : tensor[1, 5, int64])
  ids_a = to_list(reshape(out_a, [cast(5, int64)]))
  ids_b = to_list(reshape(out_b, [cast(5, int64)]))
  _ = assert_eq_int(len(ids_a), len(ids_b), "two calls produce same length")
  pairs = zip(ids_a, ids_b)
  hits = fold(fn (acc: int64, p: (int64, int64)) -> if eq(p.0, p.1) then add(acc, cast(1, int64)) else acc, cast(0, int64), pairs)
  assert_eq_int(hits, len(pairs), "every position equal across two greedy calls (determinism)")
}
