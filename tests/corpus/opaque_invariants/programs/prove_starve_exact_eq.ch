module M.Exact
@opaque
@invariant(p) p.value == 0.5
type Exact =
  | Exact { value: f32 }
def mk(x: f32) -> Exact = Exact { value: x }
def exact_value(p: Exact) -> f32 = p.value
@property always forall(p: Exact):
  exact_value(p) >= 0.0
