module Example.ScalarSpecialValues
import Std.Scalar (abs, max, min)
magnitude = abs(-0.0f64)
negative_infinity_magnitude = abs(neg(div(1.0f64, 0.0f64)))
first_zero_maximum = max(-0.0f64, 0.0f64)
first_zero_minimum = min(0.0f64, -0.0f64)
leading_nan = max(div(0.0f64, 0.0f64), 1.0f64)
