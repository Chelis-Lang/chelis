module Std.Loss.Bce
export (bce_with_logits)
sig bce_with_logits: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]
def bce_with_logits(z, y) = {
  z_list = to_list(z)
  y_list = to_list(y)
  if neq(len(z_list), len(y_list)) then fail("bce_with_logits: z and y have different lengths") else to_tensor(map(fn (pair: (f32, f32)) -> {
    zv = pair.0
    yv = pair.1
    zero = cast(0.0, f32)
    one = cast(1.0, f32)
    mz = if gt(zv, zero) then zv else zero
    az = if gt(zv, zero) then zv else neg(zv)
    sub(add(mz, log(add(one, exp(neg(az))))), mul(zv, yv))
  }, zip(z_list, y_list)))
}
