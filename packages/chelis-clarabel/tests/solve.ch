module Clarabel.TestSolve
import Clarabel.Qp (Solved, solve, Settings)
empty_values: List[f64] = []
def main() -> bool =
  match solve(to_tensor([[2.0f64]]), to_tensor([-4.0f64]), reshape(to_tensor(empty_values), [0i64, 1i64]), to_tensor(empty_values), [], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 }) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => gt(tensor_to_scalar(sum(primal, 0i32)), 1.9f64)
    | _ => false
  }
