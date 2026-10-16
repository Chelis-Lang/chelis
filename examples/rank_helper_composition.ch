module RankHelperComposition
def negate[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)
def wrapped[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = negate(x)
out = wrapped(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f32))
