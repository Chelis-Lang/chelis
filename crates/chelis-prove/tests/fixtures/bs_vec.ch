export (bs_call_vec, bs_put_vec)
-- Vectorized Black-Scholes pricer: pure-tensor-DAG, elementwise over tensor[n].
-- A-S erf coefficients passed as tensor params (Beacon verifies over point intervals).
-- No vmap, no scalar_to_tensor, no expand, no shape(). WireDag root for Beacon.
--
-- The branch structure (CmpLt + mask*a + (1-mask)*b) is the inlined Abramowitz-Stegun
-- sign-fold and small-x guard -- same computation as Shoals.Pricing.erf64.
-- Elementwise select: where mask=1 return a, else b
-- mask is f32 (0.0 or 1.0 from ... |> lt |> cast(f32))
def sel[n](mask: &tensor[n, f32], a: tensor[n, f32], b: tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] = {
  one = add(half, half)
  inv = sub(one, mask)
  mask |> mul(a) |> add(mul(inv, b))
}
-- Elementwise abs via sign-fold mask
def abs_v[n](x: &tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] = {
  zero = sub(x, x)
  mask = x |> lt(zero) |> cast(f32)
  sel(&mask, neg(x), copy(x), half)
}
-- erf (Abramowitz-Stegun 7.1.26) with coefficients as parameters.
-- a1..a5, p are the standard A-S constants; twosqrtpi = 2/sqrt(pi).
-- small_thresh is the small-x guard threshold (1e-5).
def erf_vec[n](x: &tensor[n, f32], a1: &tensor[n, f32], a2: &tensor[n, f32], a3: &tensor[n, f32], a4: &tensor[n, f32], a5: &tensor[n, f32], p: &tensor[n, f32], twosqrtpi: &tensor[n, f32], small_thresh: &tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] = {
  one = add(half, half)
  zero = sub(x, x)
  ax = abs_v(x, half)
  -- small-x linear: erf(x) ~ x * 2/sqrt(pi)
  small_result = mul(x, twosqrtpi)
  -- main polynomial
  t = div(&one, add(&one, mul(p, &ax)))
  poly = mul(&t, add(a1, mul(&t, add(a2, mul(&t, add(a3, mul(&t, add(a4, mul(&t, a5)))))))))
  e = exp(neg(mul(&ax, &ax)))
  y_pos = sub(one, mul(poly, e))
  -- sign fold
  neg_mask = x |> lt(zero) |> cast(f32)
  y_signed = sel(&neg_mask, neg(&y_pos), y_pos, half)
  -- small-x guard
  is_small = ax |> lt(small_thresh) |> cast(f32)
  sel(&is_small, small_result, y_signed, half)
}
-- Normal CDF: N(x) = 0.5 * (1 + erf(x / sqrt(2)))
def n_cdf_vec[n](x: &tensor[n, f32], half: &tensor[n, f32], inv_sqrt2: &tensor[n, f32], a1: &tensor[n, f32], a2: &tensor[n, f32], a3: &tensor[n, f32], a4: &tensor[n, f32], a5: &tensor[n, f32], p: &tensor[n, f32], twosqrtpi: &tensor[n, f32], small_thresh: &tensor[n, f32]) -> tensor[n, f32] = {
  one = add(half, half)
  scaled = mul(x, inv_sqrt2)
  erf_val = erf_vec(&scaled, a1, a2, a3, a4, a5, p, twosqrtpi, small_thresh, half)
  mul(half, add(one, erf_val))
}
-- Black-Scholes d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
def bs_d1[n](s: &tensor[n, f32], k: &tensor[n, f32], r: &tensor[n, f32], sigma: &tensor[n, f32], t: &tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] = {
  num = add(log(div(s, k)), mul(add(r, mul(half, mul(sigma, sigma))), t))
  div(num, mul(sigma, sqrt(t)))
}
-- Black-Scholes d2 = d1 - sigma*sqrt(T)
def bs_d2[n](s: &tensor[n, f32], k: &tensor[n, f32], r: &tensor[n, f32], sigma: &tensor[n, f32], t: &tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] = s |> bs_d1(k, r, sigma, t, half) |> sub(mul(sigma, sqrt(t)))
-- Vectorized BS call: S*N(d1) - K*exp(-rT)*N(d2)
def bs_call_vec[n](s: tensor[n, f32], k: tensor[n, f32], r: tensor[n, f32], sigma: tensor[n, f32], t: tensor[n, f32], half: tensor[n, f32], inv_sqrt2: tensor[n, f32], a1: tensor[n, f32], a2: tensor[n, f32], a3: tensor[n, f32], a4: tensor[n, f32], a5: tensor[n, f32], p: tensor[n, f32], twosqrtpi: tensor[n, f32], small_thresh: tensor[n, f32]) -> tensor[n, f32] = {
  d1 = bs_d1(&s, &k, &r, &sigma, &t, &half)
  d2 = bs_d2(&s, &k, &r, &sigma, &t, &half)
  nd1 = n_cdf_vec(&d1, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  nd2 = n_cdf_vec(&d2, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  disc = exp(neg(mul(&r, &t)))
  s |> mul(nd1) |> sub(mul(k, mul(disc, nd2)))
}
-- Vectorized BS put: K*exp(-rT)*N(-d2) - S*N(-d1)
def bs_put_vec[n](s: tensor[n, f32], k: tensor[n, f32], r: tensor[n, f32], sigma: tensor[n, f32], t: tensor[n, f32], half: tensor[n, f32], inv_sqrt2: tensor[n, f32], a1: tensor[n, f32], a2: tensor[n, f32], a3: tensor[n, f32], a4: tensor[n, f32], a5: tensor[n, f32], p: tensor[n, f32], twosqrtpi: tensor[n, f32], small_thresh: tensor[n, f32]) -> tensor[n, f32] = {
  d1 = bs_d1(&s, &k, &r, &sigma, &t, &half)
  d2 = bs_d2(&s, &k, &r, &sigma, &t, &half)
  nd1_arg = neg(d1)
  nd2_arg = neg(d2)
  nd1 = n_cdf_vec(&nd1_arg, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  nd2 = n_cdf_vec(&nd2_arg, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  disc = exp(neg(mul(&r, &t)))
  k |> mul(mul(disc, nd2)) |> sub(mul(s, nd1))
}
