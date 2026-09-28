def main(x: tensor[2, f32]) -> tensor[2, f32] = with device("cpu") { mul(x, x) }
