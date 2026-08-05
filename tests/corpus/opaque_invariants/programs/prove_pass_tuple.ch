module Stats.Prob
export (mk)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def mk(x: f32) -> Option[(Probability, f32)] = if ((x >= 0.0) && (x <= 1.0)) then Some((Probability { value: x }, x)) else None
