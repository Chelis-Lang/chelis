type Reading =
  | Valid(f32)
  | Missing
def value_or(reading: Reading, default: f32) -> f32 =
  match reading with {
    | Valid(x) => x
    | Missing => default
  }
def map_reading(f: f32 -> f32, reading: Reading) -> Reading =
  match reading with {
    | Valid(x) => x |> f |> Valid
    | Missing => Missing
  }
def unwrap_or(opt: Option[f32], default: f32) -> f32 =
  match opt with {
    | Some(x) => x
    | None => default
  }
