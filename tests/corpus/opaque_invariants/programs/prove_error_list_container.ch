module Stats.Prob
export (many)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def many(x: f32) -> List[Probability] = Cons(Probability { value: x }, Nil)
