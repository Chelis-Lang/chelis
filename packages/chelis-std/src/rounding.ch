module Std.Rounding
export (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
-- Std.Rounding: the rounding modes every standard-library conversion that
-- drops precision takes, governed by [05-OP-74]. For an exact value `v` and a
-- positive quantum `q`, each mode selects a multiple `k * q` of the quantum.
type Rounding =
  | RoundTowardNegative
  | RoundTowardPositive
  | RoundTowardZero
  | RoundAwayFromZero
  | RoundTiesToEven
  | RoundTiesToAway
  | RejectInexact
