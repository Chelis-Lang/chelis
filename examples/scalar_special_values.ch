module Example.ScalarSpecialValues
import Std.Scalar (abs)
magnitude = abs(-0.0f64)
negative_infinity_magnitude = abs(neg(div(1.0f64, 0.0f64)))
first_zero_maximum = max_elem(-0.0f64, 0.0f64)
first_zero_minimum = min_elem(0.0f64, -0.0f64)
leading_nan = max_elem(div(0.0f64, 0.0f64), 1.0f64)
