module PipelineParity.Main
export (Choice, choose, identity)
type Choice =
  | Left(int32)
  | Right(int32)
def choose(x: int32) -> Choice = Left(x)
def identity(x: tensor[n, f32]) -> tensor[n, f32] = x
