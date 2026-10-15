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
def literal_coefficient(x: f32) -> f32 = (x * 1.1f32)
gradient_matches = (grad(literal_coefficient, wrt=x)(0.0f32) == 1.1f32)
gradient_differs = (grad(literal_coefficient, wrt=x)(0.0f32) != 1.1f32)
def subnormal_coefficient(x: f32) -> f32 = (x * 1e-45f32)
def underflow_coefficient(x: f32) -> f32 = (x * 1e-46f32)
subnormal_gradient_matches = (grad(subnormal_coefficient, wrt=x)(0.0f32) == 1e-45f32)
underflow_gradient_matches = (grad(underflow_coefficient, wrt=x)(0.0f32) == 0.0f32)
@property gradient_literal forall(x: f32):
  (grad(literal_coefficient, wrt=x)(x) == 1.1f32)
