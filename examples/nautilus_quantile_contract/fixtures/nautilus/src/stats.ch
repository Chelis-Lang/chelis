module Nautilus.Stats
export (quantile_vec)
def zero_f() -> f32 = cast(0.0, f32)
def one_f() -> f32 = cast(1.0, f32)
def one_i() -> int64 = cast(1, int64)
def quantile_vec[n](v: &tensor[n, f32], q: f32) -> f32 = {
  n_i = numel(v)
  n_f = cast(n_i, f32)
  q_clamped = if lt(q, zero_f()) then zero_f() else if gt(q, one_f()) then one_f() else q
  sorted_pair = sort(v, 0)
  sorted_v = sorted_pair.0
  lst = to_list(sorted_v)
  enum_lst = enumerate(lst)
  pos = mul(q_clamped, sub(n_f, one_f()))
  lo_idx_f = pos
  hi_idx_f = add(pos, one_f())
  lo_idx_i = cast(lo_idx_f, int64)
  lo_idx_back = cast(lo_idx_i, f32)
  frac = sub(pos, lo_idx_back)
  hi_idx_i = add(lo_idx_i, one_i())
  last_idx = sub(n_i, one_i())
  hi_idx_clamped = if gt(hi_idx_i, last_idx) then last_idx else hi_idx_i
  picked = fold(fn (acc: (f32, f32), pair: (int64, f32)) -> {
    i = pair.0
    x = pair.1
    take_lo = eq(i, lo_idx_i)
    take_hi = eq(i, hi_idx_clamped)
    lo_val = if take_lo then x else acc.0
    hi_val = if take_hi then x else acc.1
    (lo_val, hi_val)
  }, (zero_f(), zero_f()), enum_lst)
  add(picked.0, mul(frac, sub(picked.1, picked.0)))
}
