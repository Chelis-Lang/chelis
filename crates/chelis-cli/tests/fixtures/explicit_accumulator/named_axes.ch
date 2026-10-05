module Probe.NamedAxes
def small(x: tensor[seq, head, i8]) -> tensor[i32] = sum(x, seq, head)
def wide(x: tensor[seq, head, i8]) -> tensor[i64] = sum(x, seq, head, accumulator=i64)
def float_total(x: tensor[seq, head, f32]) -> tensor[f64] = sum(x, seq, head, accumulator=f64)
def main() -> unit ! { IO } = {
  bytes: tensor[2, 3, i8] = to_tensor([[100i8, 100i8, 100i8], [100i8, 100i8, 100i8]])
  floats: tensor[2, 3, f32] = to_tensor([[16777216.0f32, 1.0f32, 1.0f32], [0.0f32, 0.0f32, 0.0f32]])
  _ = print(small(copy(bytes)))
  _ = print(wide(bytes))
  _ = print(float_total(floats))
  ()
}
out = main()
