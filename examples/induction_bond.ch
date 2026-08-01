module Examples.InductionBond
def bond_value(n: int32, coupon: f64, discount: f64) -> f64 = if (n <= 0) then cast(1.0, f64) else (coupon + (discount * bond_value((n - 1), coupon, discount)))
@property bond_value_nonnegative forall(n: int32, coupon: f64, discount: f64) where (n >= 0), (coupon >= cast(0.0, f64)), (discount >= cast(0.0, f64)):
  (bond_value(n, coupon, discount) >= cast(0.0, f64))
