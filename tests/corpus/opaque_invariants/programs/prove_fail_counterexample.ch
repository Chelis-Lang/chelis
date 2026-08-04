module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Option[Probability] = if (x >= 0.0) then Some(Probability { value: x }) else None
