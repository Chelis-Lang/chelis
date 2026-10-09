module PipelineRejected.Main
def first(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = drop(x)
  realize(x)
}
def second(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = drop(x)
  sigmoid(x)
}
