sig train_step: tensor[32, f32] -> tensor[32, f32] ! { Random, Resource("gpu:0") }
def train_step(x: tensor[32, f32]) -> tensor[32, f32] ! { Random, Resource("gpu:0") } = with device("gpu:0") { with seed(42) { dropout(x, 0.5) } }
