def embed_lookup(table: tensor[1000, 128, f32], indices: tensor[64, i64]) -> tensor[64, 128, f32] = gather(table, indices, 0)
def overwrite_rows(table: tensor[3, 2, f32], indices: tensor[4, i32], updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = scatter_replace(table, indices, updates, 0)
def wrap_embed(table: tensor[1000, 128, f32], indices: tensor[64, i64]) -> tensor[64, 128, f32] = embed_lookup(table, indices)
