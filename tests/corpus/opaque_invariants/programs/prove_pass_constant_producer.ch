module Stats.Prob
export (half)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def half() -> Probability = Probability { value: 0.5 }
