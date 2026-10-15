module Examples.LiteralAscriptions
same = (1.5f64 : f64)
widened = (cast(1.1f32, f64) : f64)
adopted = (cast(1.1, f64) : f64)
local = {
  value: f64 = 1.5f64
  value
}
@property integer_input_minimum forall(x: i32):
  (x >= cast(-2147483648, i32))
def declared_wide(dummy: i32) -> f64 = 1.1
@property declared_literal forall(x: i32):
  (declared_wide(x) == 1.1f64)
