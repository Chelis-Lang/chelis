-- Each index row selects positions within its matching data row.
def select_row(values: tensor[4, f32], indices: tensor[2, i64]) -> tensor[2, f32] = gather(values, indices, 0i32)
values = to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]])
indices = to_tensor([[0i64, 2i64], [1i64, 3i64]])
selected = vmap(select_row)(values, indices)
