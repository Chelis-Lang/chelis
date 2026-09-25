-- Declared result extents belong to their producing inserts.
-- Reading an input shape does not move a result guard to function entry.
def fill_grid(b: tensor[f32], z: tensor[rows, f32], a: tensor[cols, f32]) -> tensor[4, 3, f32] = insert(insert(b, 0i32, shape(a, 0i32)), 0i32, shape(z, 0i32))
def main() = fill_grid(scalar_to_tensor(7.0f32), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]))
