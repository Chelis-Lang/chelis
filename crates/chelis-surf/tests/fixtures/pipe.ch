def process(x: tensor[features, f32]): tensor[features, f32] =
  x |> normalize |> relu |> softmax
