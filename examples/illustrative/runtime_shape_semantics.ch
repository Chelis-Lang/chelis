sig bias_broadcast[n]: &tensor[n, 4, f32] -> &tensor[4, f32] -> tensor[n, 4, f32]
def bias_broadcast(x, b) -> tensor[n, 4, f32] = insert(b, 0, shape(x, cast(0, i32)))
sig flatten_batch[n]: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) -> tensor[n, 4, f32] = reshape(x, [x |> shape(cast(0, i32)) |> cast(i64), cast(4, i64)])
sig flatten_two[n, m]: &tensor[n, m, f32] -> tensor[n, m, f32]
def flatten_two(x) -> tensor[n, m, f32] = reshape(x, [x |> shape(cast(0, i32)) |> cast(i64), x |> shape(cast(1, i32)) |> cast(i64)])
sig fixed_dims[n]: &tensor[n, 4, f32] -> tensor[4, 4, f32]
def fixed_dims(x) -> tensor[4, 4, f32] = reshape(x, [cast(4, i64), cast(4, i64)])
