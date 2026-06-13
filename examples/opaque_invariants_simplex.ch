module Stats.Simplex
export (make_simplex)
@opaque
@invariant(p) ((sum(p.weights) >= (1.0 - eps)) && (sum(p.weights) <= (1.0 + eps)))
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.0001
def make_simplex(a: f32, b: f32, c: f32) -> Simplex = {
  s = (((abs(a) + abs(b)) + abs(c)) + 0.001)
  Simplex { weights: to_tensor([(abs(a) / s), (abs(b) / s), ((abs(c) + 0.001) / s)]) }
}
@property simplex_binder_is_generated forall(p: Simplex):
  true
