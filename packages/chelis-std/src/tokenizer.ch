module Std.Tokenizer
import Std.Io.Json (Json, json_array, json_get, json_int, json_object, json_string, try_load_json)
export (Tokenizer, load_tokenizer, try_load_tokenizer, encode, decode, batch_encode)
type Tokenizer =
  | BpeTokenizer(Dict[string, i64], Dict[string, i64], Dict[i64, string], i64)
def seed_strings() -> List[string] = [""]
def seed_ints() -> List[i64] = [cast(0, i64)]
def load_tokenizer(path: string) -> Tokenizer ! { IO } =
  match try_load_tokenizer(path) with {
    | Some(tokenizer) => tokenizer
    | None => fail(string_concat("load_tokenizer failed for ", path))
  }
def try_load_tokenizer(path: string) -> Option[Tokenizer] ! { IO } =
  match try_load_json(path) with {
    | Some(root) => match json_object(json_get(root, "model")) with {
    | Some(model) => match json_string(dict_get(model, "type")) with {
    | Some(kind) => if neq(kind, "BPE") then None else match json_object(dict_get(model, "vocab")) with {
    | Some(vocab_json) => match json_array(dict_get(model, "merges")) with {
    | Some(merges_json) => match vocab_from_json(dict_entries(vocab_json)) with {
    | Some(vocab_entries) => match merges_from_json(merges_json) with {
    | Some(merge_entries) => {
    vocab = dict_of(vocab_entries)
    merges = dict_of(merge_entries)
    inverse = dict_of(invert_vocab_entries(vocab_entries))
    unk_id = match json_string(dict_get(model, "unk_token")) with {
      | Some(token) => match dict_get(vocab, token) with {
      | Some(value) => value
      | None => cast(0, i64)
    }
      | None => cast(0, i64)
    }
    Some(BpeTokenizer(vocab, merges, inverse, unk_id))
  }
    | None => None
  }
    | None => None
  }
    | None => None
  }
    | None => None
  }
    | None => None
  }
    | None => None
  }
    | None => None
  }
def encode(tokenizer: Tokenizer, text: string) -> List[i64] =
  match tokenizer with {
    | BpeTokenizer(vocab, merges, _, unk_id) => map(fn (token: string) -> match dict_get(vocab, token) with {
    | Some(value) => value
    | None => unk_id
  }, apply_bpe(split_chars(text, cast(0, i64), skip(seed_strings(), cast(1, i64))), merges))
    | _ => skip(seed_ints(), cast(1, i64))
  }
def decode(tokenizer: Tokenizer, ids: List[i64]) -> string =
  match tokenizer with {
    | BpeTokenizer(_, _, inverse, unk_id) => {
    unk = match dict_get(inverse, unk_id) with {
      | Some(value) => value
      | None => ""
    }
    fold(fn (acc: string, id: i64) -> string_concat(acc, match dict_get(inverse, id) with {
      | Some(value) => value
      | None => unk
    }), "", ids)
  }
    | _ => ""
  }
def batch_encode(tokenizer: Tokenizer, texts: List[string], max_length: i64, pad_value: i64) -> tensor[batch, seq, i64] = pad_sequences_to(map(fn (text: string) -> encode(tokenizer, text), texts), max_length, pad_value)
-- The three entry builders ran on the linear combinator lane after
-- chelis#1213's CSV rewrite showed what the recursive shape cost. Each one
-- used to walk its input with `skip(entries, 1)` and an `append(out, pair)`
-- accumulator: `skip` copies the whole remaining List per step and `append`
-- copies the whole accumulator per step, so reading n entries allocated
-- O(n^2) list bytes and recursed n frames deep. `map` lowers to a
-- capacity-reserved list plus in-place pushes (chelis#943/#949) and recurses
-- not at all.
--
-- The `Option` is what the recursion carried for free and a `map` cannot:
-- `map`'s callback is total. It is split the way `Std.Io.Csv.parse_rows`
-- splits it. A `fold` carries one bool and its body is an `if` on the
-- accumulator rather than an `and(...)` call, because a call evaluates both
-- arguments while `if` evaluates one branch; after the first malformed entry
-- no later entry is parsed. The success arm then runs a total `map` over
-- entries already known to parse. Valid input is parsed twice, doubling a
-- linear constant; the recursive shape it replaces was quadratic.
def vocab_from_json(entries: List[(string, Json)]) -> Option[List[(string, i64)]] = if fold(fn (acc: bool, entry: (string, Json)) -> if acc then vocab_entry_ok(entry) else false, true, entries) then Some(map(fn (entry: (string, Json)) -> vocab_pair(entry), entries)) else None
def vocab_entry_ok(entry: (string, Json)) -> bool =
  match json_int(Some(entry.1)) with {
    | Some(_) => true
    | None => false
  }
-- Only reached from the success arm of `vocab_from_json`, where every entry
-- already passed `vocab_entry_ok`, so the `None` arm is unreachable there;
-- the zero keeps the function total without smuggling a sentinel into a
-- reachable result.
def vocab_pair(entry: (string, Json)) -> (string, i64) =
  match json_int(Some(entry.1)) with {
    | Some(id) => (entry.0, id)
    | None => (entry.0, cast(0, i64))
  }
def invert_vocab_entries(entries: List[(string, i64)]) -> List[(i64, string)] = map(fn (entry: (string, i64)) -> (entry.1, entry.0), entries)
-- The merge rank was a counter threaded through the recursion. `enumerate`
-- supplies the same zero-based i64 index ([05-OP-55]) without one.
def merges_from_json(merges: List[Json]) -> Option[List[(string, i64)]] = if fold(fn (acc: bool, line: Json) -> if acc then merge_line_ok(line) else false, true, merges) then Some(map(fn (ranked: (i64, Json)) -> merge_pair(ranked), enumerate(merges))) else None
def merge_line_ok(line: Json) -> bool =
  match json_string(Some(line)) with {
    | Some(text) => match merge_key_from_line(text) with {
    | Some(_) => true
    | None => false
  }
    | None => false
  }
-- Unreachable `None` arms for the same reason as `vocab_pair`: every line
-- reaching here already passed `merge_line_ok`.
def merge_pair(ranked: (i64, Json)) -> (string, i64) =
  match json_string(Some(ranked.1)) with {
    | Some(text) => match merge_key_from_line(text) with {
    | Some(key) => (key, ranked.0)
    | None => ("", ranked.0)
  }
    | None => ("", ranked.0)
  }
def merge_key_from_line(line: string) -> Option[string] =
  match find_space(line, cast(0, i64)) with {
    | Some(idx) => {
    lhs = string_slice(line, cast(0, i64), idx)
    rhs = string_slice(line, add(idx, cast(1, i64)), sub(string_len(line), add(idx, cast(1, i64))))
    Some(merge_key(lhs, rhs))
  }
    | None => None
  }
def find_space(line: string, idx: i64) -> Option[i64] = if gte(idx, string_len(line)) then None else if eq(string_slice(line, idx, cast(1, i64)), " ") then Some(idx) else find_space(line, add(idx, cast(1, i64)))
def split_chars(text: string, idx: i64, out: List[string]) -> List[string] = if gte(idx, string_len(text)) then out else split_chars(text, add(idx, cast(1, i64)), append(out, string_slice(text, idx, cast(1, i64))))
def apply_bpe(tokens: List[string], merges: Dict[string, i64]) -> List[string] =
  match best_pair(tokens, merges, cast(0, i64)) with {
    | Some(pair) => {
    (key, _) = pair
    apply_bpe(merge_once(tokens, key, cast(0, i64), skip(seed_strings(), cast(1, i64))), merges)
  }
    | None => tokens
  }
def best_pair(tokens: List[string], merges: Dict[string, i64], idx: i64) -> Option[(string, i64)] =
  if gte(add(idx, cast(1, i64)), len(tokens)) then None else {
    key = merge_key(index(tokens, idx), index(tokens, add(idx, cast(1, i64))))
    current = match dict_get(merges, key) with {
      | Some(merge_rank) => Some((key, merge_rank))
      | None => None
    }
    choose_better(current, best_pair(tokens, merges, add(idx, cast(1, i64))))
  }
def choose_better(lhs: Option[(string, i64)], rhs: Option[(string, i64)]) -> Option[(string, i64)] =
  match lhs with {
    | Some(left) => {
    (_, left_rank) = left
    match rhs with {
      | Some(right) => {
      (_, right_rank) = right
      if lte(left_rank, right_rank) then Some(left) else Some(right)
    }
      | None => Some(left)
    }
  }
    | None => rhs
  }
def merge_once(tokens: List[string], target: string, idx: i64, out: List[string]) -> List[string] =
  if gte(idx, len(tokens)) then out else if gte(add(idx, cast(1, i64)), len(tokens)) then append(out, index(tokens, idx)) else {
    lhs = index(tokens, idx)
    rhs = index(tokens, add(idx, cast(1, i64)))
    if eq(merge_key(lhs, rhs), target) then merge_once(tokens, target, add(idx, cast(2, i64)), append(out, string_concat(lhs, rhs))) else merge_once(tokens, target, add(idx, cast(1, i64)), append(out, lhs))
  }
def merge_key(lhs: string, rhs: string) -> string = string_concat(lhs, string_concat("\t", rhs))
