module ReleasePipe.Tests.PipeStage

type Sign =
  | Neg
  | Pos

def choose(s: Sign) -> f32 = match s with {
  | Neg => cast(0.0, f32)
  | Pos => cast(1.0, f32)
}

def pipe_choose(s: Sign) -> f32 = s |> choose

def test_pipe_stage_host_runtime() -> unit = test_assert(pipe_choose(Pos) == cast(1.0, f32), "pipe stage host runtime")
