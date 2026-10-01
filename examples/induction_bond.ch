module Examples.InductionBond
def bond_value(n: i32, coupon: f64, discount: f64) -> f64 = if (n <= 0) then 1.0f64 else (coupon + (discount * bond_value((n - 1), coupon, discount)))
@property bond_value_nonnegative forall(n: i32, coupon: f64, discount: f64) where n >= 0, coupon >= 0.0f64, discount >= 0.0f64:
  (bond_value(n, coupon, discount) >= 0.0f64)
