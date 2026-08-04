module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Probability = Probability { value: x }
def broken(x: f32) -> f32 = to_tensor([x])
