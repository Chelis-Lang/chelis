def overwrite_rows(table: tensor[3, 2, f32], indices: tensor[4, int32], updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = scatter_replace(table, indices, updates, 0)
