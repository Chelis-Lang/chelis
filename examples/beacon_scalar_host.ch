def affine(x: f64) -> f64 = (x + 1.0f64)
@property bounded_affine forall(x: f64) where x >= -1.0f64, x <= 1.0f64:
  ((affine(x) >= -1.0f64) && (affine(x) <= 3.0f64))
def affine_f32(x: f32) -> f32 = (x + 0.1f32)
@property bounded_affine_f32 forall(x: f32) where x >= 0.1f32, x <= 0.2f32:
  ((affine_f32(x) >= 0.0f32) && (affine_f32(x) <= 1.0f32))
