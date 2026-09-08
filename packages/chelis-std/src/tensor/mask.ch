module Std.Tensor.Mask
export (where_indices)
def where_indices[hits](mask: &tensor[..r, bool]) -> tensor[hits, int64] = {
  pairs = enumerate(to_list(mask))
  hits = filter(fn (pair: (int64, bool)) -> pair.1, pairs)
  ids = map(fn (pair: (int64, bool)) -> pair.0, hits)
  to_tensor(ids)
}
