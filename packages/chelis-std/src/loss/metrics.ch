module Std.Loss.Metrics
export (accuracy, perplexity)
def accuracy[batch, classes, p](logits: &tensor[batch, classes, p], labels: &tensor[batch, int64]) -> f32 = {
  batch_size = cast(shape(logits, cast(0, int32)), int64)
  rows = split(logits, cast(0, int32), one_sizes_acc(batch_size))
  preds = map(fn (row) -> {
    out = row_argmax(row)
    _ = drop(row)
    out
  }, rows)
  label_list = tensor_ints_to_list(labels)
  hits = fold(fn (acc: int64, pair: (int64, int64)) -> if eq(pair.0, pair.1) then add(acc, cast(1, int64)) else acc, cast(0, int64), zip(preds, label_list))
  div(cast(hits, f32), cast(batch_size, f32))
}
def perplexity(loss: f32) -> f32 = exp(loss)
def row_argmax[piece, classes, p](row: &tensor[piece, classes, p]) -> int64 = {
  pair = sort(row, cast(1, int32))
  ids = tensor_int_row_to_list(pair.1)
  index(ids, sub(len(ids), cast(1, int64)))
}
def tensor_int_row_to_list[piece, classes](row: &tensor[piece, classes, int64]) -> List[int64] = {
  col_size = cast(shape(row, cast(1, int32)), int64)
  cols = split(row, cast(1, int32), one_sizes_acc(col_size))
  map(fn (col: tensor[piece, unit, int64]) -> {
    traced = trace(col, cast(0, int32), cast(1, int32))
    out = tensor_to_scalar(traced)
    _ = drop(traced)
    _ = drop(col)
    out
  }, cols)
}
def tensor_ints_to_list[batch](labels: &tensor[batch, int64]) -> List[int64] = to_list(labels)
def one_sizes_acc(count: int64) -> List[int64] = if lte(count, cast(0, int64)) then [] else append(one_sizes_acc(sub(count, cast(1, int64))), cast(1, int64))
