module ClarabelExample.Main
import Clarabel.Qp (NonnegativeCone, Solved, Settings, solve)
def main() -> bool =
  match solve(to_tensor([[2.0f64]]), to_tensor([0.0f64]), to_tensor([[-1.0f64]]), to_tensor([-1.0f64]), [NonnegativeCone(1i64)], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 }) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => gt(tensor_to_scalar(sum(primal, 0i32)), 0.99f64)
    | _ => false
  }
