-- A `Resource` handler scopes device placement. Randomness is not an effect:
-- the draw takes its key as an argument.
sig train_step: key -> tensor[32, f32] -> tensor[32, f32] ! { Resource("gpu:0") }
def train_step(k: key, x: tensor[32, f32]) -> tensor[32, f32] ! { Resource("gpu:0") } = with device("gpu:0") { dropout(k, x, 0.5) }
