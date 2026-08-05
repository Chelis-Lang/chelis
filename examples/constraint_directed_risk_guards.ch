module Risk.ConstraintGuards
@property confidence_tail_order forall(alpha1: f32, alpha2: f32) where alpha1 > 0.99, alpha1 < alpha2, alpha2 < 1.0:
  ((1.0 - alpha2) < (1.0 - alpha1))
