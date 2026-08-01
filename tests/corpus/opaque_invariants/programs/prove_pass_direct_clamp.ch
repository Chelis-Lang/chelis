module Stats.Prob
export (clamp_prob)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }
