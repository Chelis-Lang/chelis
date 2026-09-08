sig bias_broadcast: &tensor[n, 4, f32] -> &tensor[4, f32] -> tensor[n, 4, f32]
def bias_broadcast(x, b) -> tensor[n, 4, f32] = insert(b, 0, shape(x, cast(0, int32)))
sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) -> tensor[n, 4, f32] = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
sig flatten_two: &tensor[n, m, f32] -> tensor[n, m, f32]
def flatten_two(x) -> tensor[n, m, f32] = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(shape(x, cast(1, int32)), int64)])
sig fixed_dims: &tensor[n, 4, f32] -> tensor[4, 4, f32]
def fixed_dims(x) -> tensor[4, 4, f32] = reshape(x, [cast(4, int64), cast(4, int64)])
