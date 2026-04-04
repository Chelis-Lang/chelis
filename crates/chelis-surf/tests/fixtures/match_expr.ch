type Option a =
  | Some a
  | None

def unwrap_or(opt: Option f32, default: f32): f32 =
  match opt with {
    | Some x -> x
    | None -> default
  }
