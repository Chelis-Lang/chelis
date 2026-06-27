export (bs_call_vec, bs_put_vec)
-- Vectorized Black-Scholes pricer: pure-tensor-DAG, elementwise over tensor[n].
-- A-S erf coefficients passed as tensor params (Beacon verifies over point intervals).
-- No vmap, no scalar_to_tensor, no expand, no shape(). WireDag root for Beacon.
--
-- The branch structure (CmpLt + mask*a + (1-mask)*b) is the inlined Abramowitz-Stegun
-- sign-fold and small-x guard -- same computation as Shoals.Pricing.erf64.

-- Elementwise select: where mask=1 return a, else b
-- mask is f32 (0.0 or 1.0 from cast(lt(...), f32))
def sel[n](mask: &tensor[n, f32], a: tensor[n, f32], b: tensor[n, f32]) -> tensor[n, f32] = {
  one = div(copy(mask), copy(mask))
  inv = sub(one, copy(mask))
  add(mul(copy(mask), a), mul(inv, b))
}

-- Elementwise abs via sign-fold mask
def abs_v[n](x: &tensor[n, f32]) -> tensor[n, f32] = {
  zero = sub(copy(x), copy(x))
  mask = cast(lt(copy(x), zero), f32)
  sel(&mask, neg(copy(x)), copy(x))
}

-- erf (Abramowitz-Stegun 7.1.26) with coefficients as parameters.
-- a1..a5, p are the standard A-S constants; twosqrtpi = 2/sqrt(pi).
-- small_thresh is the small-x guard threshold (1e-5).
def erf_vec[n](x: &tensor[n, f32], a1: &tensor[n, f32], a2: &tensor[n, f32], a3: &tensor[n, f32], a4: &tensor[n, f32], a5: &tensor[n, f32], p: &tensor[n, f32], twosqrtpi: &tensor[n, f32], small_thresh: &tensor[n, f32]) -> tensor[n, f32] = {
  one = div(copy(x), copy(x))
  zero = sub(copy(x), copy(x))
  ax = abs_v(x)
  -- small-x linear: erf(x) ~ x * 2/sqrt(pi)
  small_result = mul(copy(x), copy(twosqrtpi))
  -- main polynomial
  t = div(copy(&one), add(copy(&one), mul(copy(p), copy(&ax))))
  poly = mul(copy(&t), add(copy(a1), mul(copy(&t), add(copy(a2), mul(copy(&t), add(copy(a3), mul(copy(&t), add(copy(a4), mul(copy(&t), copy(a5))))))))))
  e = exp(neg(mul(copy(&ax), copy(&ax))))
  y_pos = sub(one, mul(poly, e))
  -- sign fold
  neg_mask = cast(lt(copy(x), zero), f32)
  y_signed = sel(&neg_mask, neg(copy(&y_pos)), y_pos)
  -- small-x guard
  is_small = cast(lt(ax, copy(small_thresh)), f32)
  sel(&is_small, small_result, y_signed)
}

-- Normal CDF: N(x) = 0.5 * (1 + erf(x / sqrt(2)))
def n_cdf_vec[n](x: &tensor[n, f32], half: &tensor[n, f32], inv_sqrt2: &tensor[n, f32], a1: &tensor[n, f32], a2: &tensor[n, f32], a3: &tensor[n, f32], a4: &tensor[n, f32], a5: &tensor[n, f32], p: &tensor[n, f32], twosqrtpi: &tensor[n, f32], small_thresh: &tensor[n, f32]) -> tensor[n, f32] = {
  one = div(copy(x), copy(x))
  scaled = mul(copy(x), copy(inv_sqrt2))
  erf_val = erf_vec(&scaled, a1, a2, a3, a4, a5, p, twosqrtpi, small_thresh)
  mul(copy(half), add(one, erf_val))
}

-- Black-Scholes d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
def bs_d1[n](s: &tensor[n, f32], k: &tensor[n, f32], r: &tensor[n, f32], sigma: &tensor[n, f32], t: &tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] = {
  num = add(log(div(copy(s), copy(k))), mul(add(copy(r), mul(copy(half), mul(copy(sigma), copy(sigma)))), copy(t)))
  div(num, mul(copy(sigma), sqrt(copy(t))))
}

-- Black-Scholes d2 = d1 - sigma*sqrt(T)
def bs_d2[n](s: &tensor[n, f32], k: &tensor[n, f32], r: &tensor[n, f32], sigma: &tensor[n, f32], t: &tensor[n, f32], half: &tensor[n, f32]) -> tensor[n, f32] =
  sub(bs_d1(s, k, r, sigma, t, half), mul(copy(sigma), sqrt(copy(t))))

-- Vectorized BS call: S*N(d1) - K*exp(-rT)*N(d2)
def bs_call_vec[n](s: tensor[n, f32], k: tensor[n, f32], r: tensor[n, f32], sigma: tensor[n, f32], t: tensor[n, f32], half: tensor[n, f32], inv_sqrt2: tensor[n, f32], a1: tensor[n, f32], a2: tensor[n, f32], a3: tensor[n, f32], a4: tensor[n, f32], a5: tensor[n, f32], p: tensor[n, f32], twosqrtpi: tensor[n, f32], small_thresh: tensor[n, f32]) -> tensor[n, f32] = {
  d1 = bs_d1(&s, &k, &r, &sigma, &t, &half)
  d2 = bs_d2(&s, &k, &r, &sigma, &t, &half)
  nd1 = n_cdf_vec(&d1, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  nd2 = n_cdf_vec(&d2, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  disc = exp(neg(mul(copy(&r), copy(&t))))
  sub(mul(s, nd1), mul(k, mul(disc, nd2)))
}

-- Vectorized BS put: K*exp(-rT)*N(-d2) - S*N(-d1)
def bs_put_vec[n](s: tensor[n, f32], k: tensor[n, f32], r: tensor[n, f32], sigma: tensor[n, f32], t: tensor[n, f32], half: tensor[n, f32], inv_sqrt2: tensor[n, f32], a1: tensor[n, f32], a2: tensor[n, f32], a3: tensor[n, f32], a4: tensor[n, f32], a5: tensor[n, f32], p: tensor[n, f32], twosqrtpi: tensor[n, f32], small_thresh: tensor[n, f32]) -> tensor[n, f32] = {
  d1 = bs_d1(&s, &k, &r, &sigma, &t, &half)
  d2 = bs_d2(&s, &k, &r, &sigma, &t, &half)
  nd1_arg = neg(d1)
  nd2_arg = neg(d2)
  nd1 = n_cdf_vec(&nd1_arg, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  nd2 = n_cdf_vec(&nd2_arg, &half, &inv_sqrt2, &a1, &a2, &a3, &a4, &a5, &p, &twosqrtpi, &small_thresh)
  disc = exp(neg(mul(copy(&r), copy(&t))))
  sub(mul(k, mul(disc, nd2)), mul(s, nd1))
}
