module Examples.LiteralAscriptions
same = (1.5f64 : f64)
widened = (cast(1.1f32, f64) : f64)
adopted = (cast(1.1, f64) : f64)
local = {
  value: f64 = 1.5f64
  value
}
def keep_i32(value: i32) -> i32 = value
minimum_i32 = keep_i32(-2147483648i32)
minimum_default = keep_i32(-2147483648)
