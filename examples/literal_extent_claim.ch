-- The declared result requires four elements even when its size comes from x.
def fill_four(seed: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = insert(seed, 0i32, shape(x, 0i32))
def main() = fill_four(scalar_to_tensor(7.0f32), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
