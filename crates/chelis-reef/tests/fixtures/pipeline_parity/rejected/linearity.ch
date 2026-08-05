module PipelineRejected.Main
def first(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = realize(x)
  add(x, y)
}
def second(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = realize(x)
  mul(x, y)
}
