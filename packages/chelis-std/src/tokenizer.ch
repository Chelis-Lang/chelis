module Std.Tokenizer
import Std.Io.Json (JsonValue, json_array, json_get, json_int, json_object, json_string, try_load_json)
export (load_tokenizer, try_load_tokenizer, encode, decode, batch_encode)
type Tokenizer =
  | BpeTokenizer(Dict[string, int64], Dict[string, int64], Dict[int64, string], int64)
def seed_strings() -> List[string] = [""]
def seed_ints() -> List[int64] = [cast(0, int64)]
def seed_string_int_pairs() -> List[(string, int64)] = zip(seed_strings(), seed_ints())
def seed_int_string_pairs() -> List[(int64, string)] = zip(seed_ints(), seed_strings())
def load_tokenizer(path: string) -> Tokenizer = {
  match try_load_tokenizer(path) with {
    | Some(tokenizer) => tokenizer
    | None => fail(string_concat("load_tokenizer failed for ", path))
  }
}
def try_load_tokenizer(path: string) -> Option[Tokenizer] = {
  match try_load_json(path) with {
    | Some(root) => match json_object(json_get(root, "model")) with {
    | Some(model) => match json_string(dict_get(model, "type")) with {
    | Some(kind) => if neq(kind, "BPE") then None else {
    match json_object(dict_get(model, "vocab")) with {
      | Some(vocab_json) => match json_array(dict_get(model, "merges")) with {
      | Some(merges_json) => {
      match vocab_from_json(dict_entries(vocab_json), drop(seed_string_int_pairs(), cast(1, int64))) with {
        | Some(vocab_entries) => {
        match merges_from_json(merges_json, cast(0, int64), drop(seed_string_int_pairs(), cast(1, int64))) with {
          | Some(merge_entries) => {
          vocab = dict_of(vocab_entries)
          merges = dict_of(merge_entries)
          inverse = dict_of(invert_vocab_entries(vocab_entries, drop(seed_int_string_pairs(), cast(1, int64))))
          unk_id = match json_string(dict_get(model, "unk_token")) with {
            | Some(token) => match dict_get(vocab, token) with {
            | Some(value) => value
            | None => cast(0, int64)
          }
            | None => cast(0, int64)
          }
          Some(BpeTokenizer(vocab, merges, inverse, unk_id))
        }
          | None => None
        }
      }
        | None => None
      }
    }
      | None => None
    }
      | None => None
    }
  }
    | None => None
  }
    | None => None
  }
    | None => None
  }
}
def encode(tokenizer: Tokenizer, text: string) -> List[int64] = {
  match tokenizer with {
    | BpeTokenizer(vocab, merges, _, unk_id) => map(fn (token: string) -> match dict_get(vocab, token) with {
    | Some(value) => value
    | None => unk_id
  }, apply_bpe(split_chars(text, cast(0, int64), drop(seed_strings(), cast(1, int64))), merges))
    | _ => drop(seed_ints(), cast(1, int64))
  }
}
def decode(tokenizer: Tokenizer, ids: List[int64]) -> string = {
  match tokenizer with {
    | BpeTokenizer(_, _, inverse, unk_id) => {
    unk = match dict_get(inverse, unk_id) with {
      | Some(value) => value
      | None => ""
    }
    fold(fn (acc: string, id: int64) -> string_concat(acc, match dict_get(inverse, id) with {
      | Some(value) => value
      | None => unk
    }), "", ids)
  }
    | _ => ""
  }
}
def batch_encode(tokenizer: Tokenizer, texts: List[string], max_length: int64, pad_value: int64) -> tensor[batch, seq, int64] = pad_sequences_to(map(fn (text: string) -> encode(tokenizer, text), texts), max_length, pad_value)
def vocab_from_json(entries: List[(string, JsonValue)], out: List[(string, int64)]) -> Option[List[(string, int64)]] = {
  if eq(len(entries), cast(0, int64)) then Some(out) else {
    entry = index(entries, cast(0, int64))
    (token, value) = entry
    match json_int(Some(value)) with {
      | Some(id) => {
      pair = (token, id)
      vocab_from_json(drop(entries, cast(1, int64)), append(out, pair))
    }
      | None => None
    }
  }
}
def invert_vocab_entries(entries: List[(string, int64)], out: List[(int64, string)]) -> List[(int64, string)] = {
  if eq(len(entries), cast(0, int64)) then out else {
    entry = index(entries, cast(0, int64))
    (token, id) = entry
    pair = (id, token)
    invert_vocab_entries(drop(entries, cast(1, int64)), append(out, pair))
  }
}
def merges_from_json(merges: List[JsonValue], index0: int64, out: List[(string, int64)]) -> Option[List[(string, int64)]] = {
  if eq(len(merges), cast(0, int64)) then Some(out) else match json_string(Some(index(merges, cast(0, int64)))) with {
    | Some(line) => match merge_key_from_line(line) with {
    | Some(key) => {
    pair = (key, index0)
    merges_from_json(drop(merges, cast(1, int64)), add(index0, cast(1, int64)), append(out, pair))
  }
    | None => None
  }
    | None => None
  }
}
def merge_key_from_line(line: string) -> Option[string] = {
  match find_space(line, cast(0, int64)) with {
    | Some(idx) => {
    lhs = string_slice(line, cast(0, int64), idx)
    rhs = string_slice(line, add(idx, cast(1, int64)), sub(string_len(line), add(idx, cast(1, int64))))
    Some(merge_key(lhs, rhs))
  }
    | None => None
  }
}
def find_space(line: string, idx: int64) -> Option[int64] = { if gte(idx, string_len(line)) then None else if eq(string_slice(line, idx, cast(1, int64)), " ") then Some(idx) else find_space(line, add(idx, cast(1, int64))) }
def split_chars(text: string, idx: int64, out: List[string]) -> List[string] = { if gte(idx, string_len(text)) then out else split_chars(text, add(idx, cast(1, int64)), append(out, string_slice(text, idx, cast(1, int64)))) }
def apply_bpe(tokens: List[string], merges: Dict[string, int64]) -> List[string] = {
  match best_pair(tokens, merges, cast(0, int64)) with {
    | Some(pair) => {
    (key, _) = pair
    apply_bpe(merge_once(tokens, key, cast(0, int64), drop(seed_strings(), cast(1, int64))), merges)
  }
    | None => tokens
  }
}
def best_pair(tokens: List[string], merges: Dict[string, int64], idx: int64) -> Option[(string, int64)] = {
  if gte(add(idx, cast(1, int64)), len(tokens)) then None else {
    key = merge_key(index(tokens, idx), index(tokens, add(idx, cast(1, int64))))
    current = match dict_get(merges, key) with {
      | Some(merge_rank) => Some((key, merge_rank))
      | None => None
    }
    choose_better(current, best_pair(tokens, merges, add(idx, cast(1, int64))))
  }
}
def choose_better(lhs: Option[(string, int64)], rhs: Option[(string, int64)]) -> Option[(string, int64)] = {
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
}
def merge_once(tokens: List[string], target: string, idx: int64, out: List[string]) -> List[string] = {
  if gte(idx, len(tokens)) then out else if gte(add(idx, cast(1, int64)), len(tokens)) then append(out, index(tokens, idx)) else {
    lhs = index(tokens, idx)
    rhs = index(tokens, add(idx, cast(1, int64)))
    if eq(merge_key(lhs, rhs), target) then merge_once(tokens, target, add(idx, cast(2, int64)), append(out, string_concat(lhs, rhs))) else merge_once(tokens, target, add(idx, cast(1, int64)), append(out, lhs))
  }
}
def merge_key(lhs: string, rhs: string) -> string = { string_concat(lhs, string_concat("\t", rhs)) }
