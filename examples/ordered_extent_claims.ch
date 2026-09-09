-- The row witness z precedes the column witness a in this signature.
-- If both claims fail, the row claim is checked first on every host call path.
def fill_grid(b: tensor[f32], z: tensor[rows, f32], a: tensor[cols, f32]) -> tensor[4, 3, f32] = insert(insert(b, 0i32, shape(a, 0i32)), 0i32, shape(z, 0i32))
def main() = fill_grid(scalar_to_tensor(7.0f32), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]))
