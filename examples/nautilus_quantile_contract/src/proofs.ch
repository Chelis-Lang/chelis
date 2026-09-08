module QuantileContract.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[5, f32] = (to_tensor([8.0, 1.0, 5.0, 3.0, 2.0]) : tensor[5, f32])
@property example_quantiles_are_monotone forall(p: f32, q: f32) where 0.0 <= p, p <= q, q <= 1.0:
  (quantile_vec(observations(), p) <= quantile_vec(observations(), q))
  with contract = "std.quantile.monotonicity"
