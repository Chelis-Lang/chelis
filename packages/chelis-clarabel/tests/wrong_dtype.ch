module Clarabel.TestWrongDtype
import Clarabel.Qp (Settings, ZeroCone, solve)
def main() = solve(to_tensor([[2.0f32]]), to_tensor([-4.0f64]), to_tensor([[1.0f64]]), to_tensor([1.0f64]), [ZeroCone(1i64)], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 })
