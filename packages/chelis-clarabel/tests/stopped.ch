module Clarabel.TestStopped
import Clarabel.Qp (MaxIterationsStatus, NonnegativeCone, Stopped, Settings, solve)
def main() -> bool =
  match solve(to_tensor([[2.0f64]]), to_tensor([0.0f64]), to_tensor([[-1.0f64]]), to_tensor([-1.0f64]), [NonnegativeCone(1i64)], Settings { max_iterations: 0i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 }) with {
    | Stopped { status: MaxIterationsStatus } => true
    | _ => false
  }
