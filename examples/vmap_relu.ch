def process(x: tensor[features, f32]) -> tensor[features, f32] =
  relu(x)
def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] =
  vmap(process, axis=0)(xs)