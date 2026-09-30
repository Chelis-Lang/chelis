type State =
  | State { seed: i64 }
def init(seed: i64) -> State = State { seed }
def wrapped(x: tensor[2, 1, f32], state: State) -> (tensor[2, 1, f32], State) =
  match state with {
    | State { seed } => (x, State { seed })
  }
def main() -> tensor[2, 1, f32] = {
  x = [1.0f32, 2.0f32] |> to_tensor |> reshape([2i64, 1i64])
  pair = wrapped(x, init(0i64))
  pair.0
}
