module Std.Nn.Generate
export (GenerateConfig, KVCache, generate, generate_with, greedy_next_tokens, sample_next_tokens)
type GenerateConfig =
  | GenerateConfig { max_tokens: int64, temperature: f32, top_k: int64, top_p: f32 }
type KVCache[a] =
  | KVCache(List[a])
-- The `generate*` functions below take a `model` callback as their first
-- parameter, so a standalone `sig` for them needs a parenthesized
-- function-typed argument: `(model_fn) -> context -> ... -> result`. `chelis
-- fmt` currently strips the grouping parens (`(a -> b) -> c` is rewritten to
-- `a -> b -> c`, which changes arity), so those sigs can't be written
-- fmt-clean today. The inline `model:` annotations on the defs are
-- authoritative; restore explicit sigs once the formatter preserves the parens.
sig ids_to_batch_tensor: List[int64] -> tensor[rows, cols, int64]
def generate[batch, seq, vocab, kv](model: tensor[batch, seq, int64] -> Option[KVCache[kv]] -> (tensor[batch, vocab, f32], KVCache[kv]), context: tensor[batch, seq, int64], max_tokens: int64) = generate_greedy_loop(model, context, None, max_tokens)
def generate_with[batch, seq, vocab, kv](model: tensor[batch, seq, int64] -> Option[KVCache[kv]] -> (tensor[batch, vocab, f32], KVCache[kv]), context: tensor[batch, seq, int64], config: GenerateConfig) = generate_loop(model, context, None, config.max_tokens, config)
def generate_greedy_loop[batch, seq, vocab, kv](model: tensor[batch, seq, int64] -> Option[KVCache[kv]] -> (tensor[batch, vocab, f32], KVCache[kv]), current: tensor[batch, seq, int64], cache: Option[KVCache[kv]], remaining: int64) = {
  if lte(remaining, cast(0, int64)) then current else {
    step = model(copy(current), cache)
    next_ids = greedy_next_tokens(step.0)
    generate_greedy_loop(model, concat([current, ids_to_batch_tensor(next_ids)], cast(1, int32)), Some(step.1), sub(remaining, cast(1, int64)))
  }
}
def greedy_next_tokens[batch, vocab](logits: &tensor[batch, vocab, f32]) -> List[int64] = {
  batch_size = cast(shape(logits, cast(0, int32)), int64)
  rows = split(logits, cast(0, int32), one_sizes(batch_size))
  map(fn (row: tensor[piece, vocab, f32]) -> {
    out = greedy_next_token(row)
    _ = drop(row)
    out
  }, rows)
}
def sample_next_tokens[batch, vocab](logits: &tensor[batch, vocab, f32], config: GenerateConfig) -> List[int64] = {
  batch_size = cast(shape(logits, cast(0, int32)), int64)
  rows = split(logits, cast(0, int32), one_sizes(batch_size))
  map(fn (row: tensor[piece, vocab, f32]) -> {
    out = sample_next_token(row, config)
    _ = drop(row)
    out
  }, rows)
}
def generate_loop[batch, seq, vocab, kv](model: tensor[batch, seq, int64] -> Option[KVCache[kv]] -> (tensor[batch, vocab, f32], KVCache[kv]), current: tensor[batch, seq, int64], cache: Option[KVCache[kv]], remaining: int64, config: GenerateConfig) = { if lte(remaining, cast(0, int64)) then current else generate_loop_step(model, current, cache, remaining, config) }
def generate_loop_step[batch, seq, vocab, kv](model: tensor[batch, seq, int64] -> Option[KVCache[kv]] -> (tensor[batch, vocab, f32], KVCache[kv]), current: tensor[batch, seq, int64], cache: Option[KVCache[kv]], remaining: int64, config: GenerateConfig) = {
  step = model(copy(current), cache)
  next_ids = if lte(config.temperature, cast(0.0, f32)) then greedy_next_tokens(step.0) else sample_next_tokens(step.0, config)
  generate_loop(model, concat([current, ids_to_batch_tensor(next_ids)], cast(1, int32)), Some(step.1), sub(remaining, cast(1, int64)), config)
}
def ids_to_batch_tensor(ids: List[int64]) = pad_sequences_to(map(fn (id: int64) -> [id], ids), cast(1, int64), cast(0, int64))
def greedy_next_token[piece, vocab](logits: &tensor[piece, vocab, f32]) -> int64 = {
  pair = sort(logits, cast(1, int32))
  ids = tensor_row_to_ints(pair.1)
  index(ids, sub(len(ids), cast(1, int64)))
}
def sample_next_token[piece, vocab](logits: &tensor[piece, vocab, f32], config: GenerateConfig) -> int64 = {
  match config with {
    | GenerateConfig { temperature: temperature, top_k: top_k, top_p: top_p, max_tokens: max_tokens } => {
    probs = sorted_probs(logits, temperature)
    ids = sorted_ids(logits, temperature)
    start = top_p_start(probs, top_k_start(len(probs), top_k), top_p)
    sample_from_sorted(ids, probs, start, mul(random_unit(), suffix_sum(probs, start)), cast(0.0, f32))
  }
  }
}
def sorted_probs[piece, vocab](logits: &tensor[piece, vocab, f32], temperature: f32) -> List[f32] = {
  sorted = tensor_row_to_floats(sort(logits, cast(1, int32)).0)
  scaled = map(fn (x: f32) -> exp(div(x, temperature)), sorted)
  total = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), scaled)
  map(fn (x: f32) -> div(x, total), scaled)
}
def sorted_ids[piece, vocab](logits: &tensor[piece, vocab, f32], temperature: f32) -> List[int64] = tensor_row_to_ints(sort(logits, cast(1, int32)).1)
def tensor_row_to_floats[piece, vocab](row: &tensor[piece, vocab, f32]) -> List[f32] = {
  vocab_size = cast(shape(row, cast(1, int32)), int64)
  cols = split(row, cast(1, int32), one_sizes(vocab_size))
  map(fn (col: tensor[piece, unit, f32]) -> {
    traced = trace(col, cast(0, int32), cast(1, int32))
    out = tensor_to_scalar(traced)
    _ = drop(traced)
    _ = drop(col)
    out
  }, cols)
}
def tensor_row_to_ints[piece, vocab](row: &tensor[piece, vocab, int64]) -> List[int64] = {
  vocab_size = cast(shape(row, cast(1, int32)), int64)
  cols = split(row, cast(1, int32), one_sizes(vocab_size))
  map(fn (col: tensor[piece, unit, int64]) -> {
    traced = trace(col, cast(0, int32), cast(1, int32))
    out = tensor_to_scalar(traced)
    _ = drop(traced)
    _ = drop(col)
    out
  }, cols)
}
def sample_from_sorted(ids: List[int64], probs: List[f32], start: int64, target: f32, acc: f32) -> int64 = pick_sorted_desc(ids, probs, sub(len(ids), cast(1, int64)), start, target, acc)
def one_sizes(count: int64) -> List[int64] = if lte(count, cast(0, int64)) then [] else append(one_sizes(sub(count, cast(1, int64))), cast(1, int64))
def top_k_start(size: int64, top_k: int64) -> int64 = if or(lte(top_k, cast(0, int64)), gte(top_k, size)) then cast(0, int64) else sub(size, top_k)
def top_p_start(probs: List[f32], keep_start: int64, top_p: f32) -> int64 = top_p_start_from(probs, keep_start, sub(len(probs), cast(1, int64)), top_p, cast(0.0, f32))
def top_p_start_from(probs: List[f32], keep_start: int64, idx: int64, limit: f32, acc: f32) -> int64 = if lt(idx, keep_start) then keep_start else if gte(add(acc, index(probs, idx)), limit) then idx else top_p_start_from(probs, keep_start, sub(idx, cast(1, int64)), limit, add(acc, index(probs, idx)))
def suffix_sum(probs: List[f32], start: int64) -> f32 = if gte(start, len(probs)) then cast(0.0, f32) else add(index(probs, start), suffix_sum(probs, add(start, cast(1, int64))))
def pick_sorted_desc(ids: List[int64], probs: List[f32], idx: int64, start: int64, target: f32, acc: f32) -> int64 = if lte(idx, start) then index(ids, start) else if gte(add(acc, index(probs, idx)), target) then index(ids, idx) else pick_sorted_desc(ids, probs, sub(idx, cast(1, int64)), start, target, add(acc, index(probs, idx)))
def random_unit() -> f32 = tensor_to_scalar(uniform_like(trace(pad_sequences_to([[0.0]], cast(1, int64), cast(0.0, f32)), cast(0, int32), cast(1, int32)), 0.0, 1.0))
