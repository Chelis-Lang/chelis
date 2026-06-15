module Std.Tests.Tokenizer
import Std.Tokenizer (Tokenizer, BpeTokenizer, batch_encode, decode, encode)
import Std.Test (assert_eq_int, assert_eq_string)
def make_ab_tokenizer() -> Tokenizer = {
  vocab = dict_of([("a", cast(1, int64)), ("b", cast(2, int64))])
  merges: Dict[string, int64] = dict_of(([] : List[(string, int64)]))
  inverse = dict_of([(cast(1, int64), "a"), (cast(2, int64), "b")])
  BpeTokenizer(vocab, merges, inverse, cast(0, int64))
}
def test_encode_known_vocab_string_returns_ids_in_order() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  ids = encode(tok, "ab")
  _ = assert_eq_int(len(ids), cast(2, int64), "encode(\"ab\") length is 2")
  _ = assert_eq_int(index(ids, cast(0, int64)), cast(1, int64), "encode(\"ab\")[0] == 1 (id for \"a\")")
  assert_eq_int(index(ids, cast(1, int64)), cast(2, int64), "encode(\"ab\")[1] == 2 (id for \"b\")")
}
def test_decode_single_known_id_returns_token_string() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  text = decode(tok, [cast(1, int64)])
  assert_eq_string(text, "a", "decode([1]) on {1->\"a\"} == \"a\"")
}
def test_decode_multiple_known_ids_concatenates_in_order() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  text = decode(tok, [cast(1, int64), cast(2, int64)])
  assert_eq_string(text, "ab", "decode([1, 2]) concatenates to \"ab\"")
}
def test_encode_decode_round_trip_on_known_vocab() -> unit ! { Test } = {
  encoded = encode(make_ab_tokenizer(), "ab")
  decoded = decode(make_ab_tokenizer(), encoded)
  assert_eq_string(decoded, "ab", "decode(encode(\"ab\")) round-trips to \"ab\"")
}
def test_encode_unknown_char_maps_to_unk_id() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  ids = encode(tok, "z")
  _ = assert_eq_int(len(ids), cast(1, int64), "encode(\"z\") length is 1")
  assert_eq_int(index(ids, cast(0, int64)), cast(0, int64), "encode(\"z\") yields unk_id (0) for unknown character")
}
def test_batch_encode_produces_uniform_width_tensor() -> unit ! { Test } = {
  batch = batch_encode(make_ab_tokenizer(), ["a", "ab"], cast(2, int64), cast(0, int64))
  d0 = cast(shape(copy(batch), cast(0, int32)), int64)
  d1 = cast(shape(batch, cast(1, int32)), int64)
  _ = assert_eq_int(d0, cast(2, int64), "batch_encode shape[0] == N (2 inputs)")
  assert_eq_int(d1, cast(2, int64), "batch_encode shape[1] == max_length (2)")
}
