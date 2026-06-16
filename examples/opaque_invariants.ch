module Stats.Opaque
export (probability, scale, combine, prob_value)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] = if ((x >= 0.0) && (x <= 1.0)) then Some(Probability { value: x }) else None
def scale(p: Probability, factor: Probability) -> Probability = Probability { value: (p.value * factor.value) }
def combine(p: Probability, q: Probability) -> Probability = Probability { value: (p.value * q.value) }
def prob_value(p: Probability) -> f32 = p.value
@property prob_value_in_unit_interval forall(p: Probability):
  ((prob_value(p) >= 0.0) && (prob_value(p) <= 1.0))
