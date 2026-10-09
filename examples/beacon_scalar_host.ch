def affine(x: f64) -> f64 = (x + 1.0f64)
@property bounded_affine forall(x: f64) where x >= -1.0f64, x <= 1.0f64:
  (affine(x) <= 2.0f64)
