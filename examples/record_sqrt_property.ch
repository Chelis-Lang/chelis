module Examples.RecordSqrtProperty
type Radius =
  | Radius { squared: f32 }
def radius(p: Radius) -> f32 = sqrt(p.squared)
@property radius_nonnegative forall(x: f32) where x >= 0.0f32:
  (radius(Radius { squared: x }) >= 0.0f32)
