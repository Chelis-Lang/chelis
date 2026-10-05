module Examples.Linspace
import Std.Tensor.Construct (linspace)
def exact_bf16_sample() -> bf16 = index(to_list(linspace(0.0bf16, 1.0bf16, 300i64)), 257i64)
grid = linspace(-2.0f64, 3.0f64, 3i64)
