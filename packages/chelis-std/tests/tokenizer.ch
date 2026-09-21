module Std.Tests.Tokenizer
import Std.Tokenizer (Tokenizer, BpeTokenizer, batch_encode, decode, encode, try_load_tokenizer)
import Std.Text (join)
import Std.Test (assert_eq, assert_true, fail)
def make_ab_tokenizer() -> Tokenizer = {
  vocab = dict_of([("a", cast(1, i64)), ("b", cast(2, i64))])
  merges: Dict[string, i64] = dict_of(([] : List[(string, i64)]))
  inverse = dict_of([(cast(1, i64), "a"), (cast(2, i64), "b")])
  BpeTokenizer(vocab, merges, inverse, cast(0, i64))
}
def test_encode_known_vocab_string_returns_ids_in_order() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  ids = encode(tok, "ab")
  _ = assert_eq(len(ids), cast(2, i64), "encode(\"ab\") length is 2")
  _ = assert_eq(index(ids, cast(0, i64)), cast(1, i64), "encode(\"ab\")[0] == 1 (id for \"a\")")
  assert_eq(index(ids, cast(1, i64)), cast(2, i64), "encode(\"ab\")[1] == 2 (id for \"b\")")
}
def test_decode_single_known_id_returns_token_string() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  text = decode(tok, [cast(1, i64)])
  assert_eq(text, "a", "decode([1]) on {1->\"a\"} == \"a\"")
}
def test_decode_multiple_known_ids_concatenates_in_order() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  text = decode(tok, [cast(1, i64), cast(2, i64)])
  assert_eq(text, "ab", "decode([1, 2]) concatenates to \"ab\"")
}
def test_encode_decode_round_trip_on_known_vocab() -> unit ! { Test } = {
  encoded = encode(make_ab_tokenizer(), "ab")
  decoded = decode(make_ab_tokenizer(), encoded)
  assert_eq(decoded, "ab", "decode(encode(\"ab\")) round-trips to \"ab\"")
}
def test_encode_unknown_char_maps_to_unk_id() -> unit ! { Test } = {
  tok = make_ab_tokenizer()
  ids = encode(tok, "z")
  _ = assert_eq(len(ids), cast(1, i64), "encode(\"z\") length is 1")
  assert_eq(index(ids, cast(0, i64)), cast(0, i64), "encode(\"z\") yields unk_id (0) for unknown character")
}
def test_batch_encode_produces_uniform_width_tensor() -> unit ! { Test } = {
  batch = batch_encode(make_ab_tokenizer(), ["a", "ab"], cast(2, i64), cast(0, i64))
  d0 = cast(shape(copy(batch), cast(0, i32)), i64)
  d1 = cast(shape(batch, cast(1, i32)), i64)
  _ = assert_eq(d0, cast(2, i64), "batch_encode shape[0] == N (2 inputs)")
  assert_eq(d1, cast(2, i64), "batch_encode shape[1] == max_length (2)")
}
-- `try_load_tokenizer` and the three entry builders behind it had no
-- coverage before the chelis#1213-shaped rewrite that replaced their
-- recursive `skip` cursors with `fold`/`map`. Every test above builds a
-- `BpeTokenizer` directly and never reaches the JSON path.
def bpe_json(vocab: string, merges: string) -> string = string_concat("{\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"vocab\":", string_concat(vocab, string_concat(",\"merges\":", string_concat(merges, "}}"))))
def written(path: string, text: string) -> string ! { IO } = {
  _ = write_file(path, text)
  path
}
def test_try_load_tokenizer_round_trips_vocab_and_merges() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_ok.json", bpe_json("{\"<unk>\":0,\"b\":1,\"ab\":2,\"a\":3}", "[\"a b\"]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => {
    ids = encode(tok, "ab")
    _ = assert_eq(len(ids), cast(1, i64), "\"a b\" merges before lookup, so \"ab\" is one id")
    _ = assert_eq(index(ids, cast(0, i64)), cast(2, i64), "the merged token takes its vocab id")
    assert_eq(decode(tok, [cast(0, i64), cast(1, i64), cast(2, i64), cast(3, i64)]), "<unk>baba", "the inverse map decodes every vocab id, including the lowest-sorted key")
  }
    | None => fail("a well-formed BPE document must load")
  }
}
-- The two merges overlap on "abc" and only their ranks decide which one
-- applies: rank 0 is "b c", so the result is ["a", "bc"] = [1, 5]. Ranking
-- them equally, or in the other order, would merge "a b" first and yield
-- ["ab", "c"] = [4, 3]. This is what pins `enumerate`'s index to the merge
-- file's order after the counter parameter was removed.
def test_try_load_tokenizer_assigns_merge_ranks_in_file_order() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_ranks.json", bpe_json("{\"<unk>\":0,\"a\":1,\"b\":2,\"c\":3,\"ab\":4,\"bc\":5}", "[\"b c\",\"a b\"]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => {
    ids = encode(tok, "abc")
    _ = assert_eq(len(ids), cast(2, i64), "one merge applies, leaving two tokens")
    _ = assert_eq(index(ids, cast(0, i64)), cast(1, i64), "the rank-0 \"b c\" merge runs first, so \"a\" stays whole")
    assert_eq(index(ids, cast(1, i64)), cast(5, i64), "and \"bc\" takes its own vocab id")
  }
    | None => fail("a well-formed BPE document must load")
  }
}
def test_try_load_tokenizer_rejects_a_non_integer_vocab_value() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_bad_vocab.json", bpe_json("{\"a\":\"three\"}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(_) => fail("a string vocab id must reject the whole document")
    | None => assert_true(true, "a non-integer vocab value yields None")
  }
}
def test_try_load_tokenizer_rejects_a_merge_line_without_a_space() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_bad_merge.json", bpe_json("{\"<unk>\":0,\"a\":1}", "[\"a b\",\"nospace\"]"))
  match try_load_tokenizer(path) with {
    | Some(_) => fail("a merge line with no space must reject the whole document")
    | None => assert_true(true, "an unparsable merge line yields None")
  }
}
def test_try_load_tokenizer_rejects_a_non_string_merge_entry() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_bad_merge_type.json", bpe_json("{\"<unk>\":0,\"a\":1}", "[\"a b\",7]"))
  match try_load_tokenizer(path) with {
    | Some(_) => fail("a numeric merge entry must reject the whole document")
    | None => assert_true(true, "a non-string merge entry yields None")
  }
}
def test_try_load_tokenizer_accepts_an_empty_vocab_and_merge_list() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_empty.json", bpe_json("{}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(len(encode(tok, "ab")), cast(2, i64), "an empty vocab maps every character to unk")
    | None => fail("an empty vocab and merge list is well-formed")
  }
}
-- The eight cases below pin `Std.Tokenizer`'s behaviour on inputs a real
-- tokenizer.json can carry and that the seven above do not reach. They also
-- serve as the differential oracle for the [05-OP-54]/[05-OP-67] rewrite of
-- the three JSON entry builders: this file is source, and `Std.*` resolves
-- through the bundle embedded in whichever `chelis` runs it, so running it
-- under a pre-rewrite and a post-rewrite binary compares the two
-- implementations on identical input. Every value here was taken under both
-- and agreed.
def test_try_load_tokenizer_keeps_the_last_duplicate_vocab_key() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_dupvocab.json", bpe_json("{\"<unk>\":0,\"a\":1,\"a\":2}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(encode_csv(tok, "a"), "2", "a repeated vocab key resolves to its last binding")
    | None => fail("a duplicate vocab key is well-formed JSON and must load")
  }
}
def test_try_load_tokenizer_keeps_the_first_rank_for_a_duplicate_merge() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_dupmerge.json", bpe_json("{\"<unk>\":0,\"a\":1,\"b\":2,\"ab\":3}", "[\"a b\",\"a b\"]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(encode_csv(tok, "ab"), "3", "a repeated merge line still merges once")
    | None => fail("a duplicate merge line is well-formed and must load")
  }
}
def test_try_load_tokenizer_rejects_a_float_vocab_value() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_floatval.json", bpe_json("{\"a\":1.5}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(_) => fail("a float vocab id is not an i64 and must reject the document")
    | None => assert_true(true, "a float vocab value yields None")
  }
}
def test_try_load_tokenizer_accepts_a_negative_vocab_id() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_negid.json", bpe_json("{\"<unk>\":0,\"a\":-4}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(encode_csv(tok, "ab"), "-4,0", "a negative id is an exact i64, not a rejection")
    | None => fail("a negative vocab id is representable and must load")
  }
}
def test_try_load_tokenizer_preserves_an_id_beyond_f64_integer_precision() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_bigint.json", bpe_json("{\"<unk>\":0,\"a\":9007199254740993}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(encode_csv(tok, "ab"), "9007199254740993,0", "2^53+1 survives ingestion exactly, with no float funnel")
    | None => fail("2^53+1 is an exact i64 and must load")
  }
}
def test_try_load_tokenizer_splits_a_merge_line_at_its_first_space() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_multispace.json", bpe_json("{\"<unk>\":0,\"a\":1,\"b\":2}", "[\"a  b\"]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(encode_csv(tok, "ab"), "1,2", "the key is split at the first space, so \"a  b\" does not merge \"ab\"")
    | None => fail("a multi-space merge line is well-formed and must load")
  }
}
def test_try_load_tokenizer_rejects_a_malformed_leading_merge_line() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_leadbad.json", bpe_json("{\"<unk>\":0,\"a\":1}", "[\"nospace\",\"a b\"]"))
  match try_load_tokenizer(path) with {
    | Some(_) => fail("a malformed first merge line must reject the whole document")
    | None => assert_true(true, "position does not matter: a leading bad line rejects like a trailing one")
  }
}
def test_try_load_tokenizer_handles_non_ascii_vocab_tokens() -> unit ! { Test, IO } = {
  path = written("/tmp/chelis_std_test_tokenizer_unicode.json", bpe_json("{\"<unk>\":0,\"é\":2}", "[]"))
  match try_load_tokenizer(path) with {
    | Some(tok) => assert_eq(encode_csv(tok, "éb"), "2,0", "a non-ASCII token is one Unicode scalar value, not its bytes")
    | None => fail("a non-ASCII vocab token must load")
  }
}
def encode_csv(tokenizer: Tokenizer, text: string) -> string = join(map(fn (id: i64) -> to_string(id), encode(tokenizer, text)), ",")
def test_try_load_tokenizer_missing_path_returns_none() -> unit ! { Test, IO } =
  match try_load_tokenizer("/tmp/chelis_std_test_tokenizer_absent.json") with {
    | Some(_) => fail("a missing path must return None")
    | None => assert_true(true, "a missing path yields None")
  }
