module PipelineParity.Main
export (Choice, choose, identity)
type Choice =
  | Left(i32)
  | Right(i32)
def choose(x: i32) -> Choice = Left(x)
def identity(x: tensor[n, f32]) -> tensor[n, f32] = x
