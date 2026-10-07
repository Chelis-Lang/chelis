module ClarabelExample.TestIdealWrapper
import Clarabel.Qp (Solved, solve, Settings)
empty_values: List[f64] = []
def optimize(q: &tensor[1, f64]) = solve(to_tensor([[2.0f64]]), q, reshape(to_tensor(empty_values), [0i64, 1i64]), to_tensor(empty_values), [], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 })
@property wrapper_result_is_optimal forall(dummy: f64):
  match optimize(to_tensor([-4.0f64])) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => (tensor_to_scalar(sum(primal, 0i32)) == 2.0f64)
    | _ => true
  }
  with contract = "clarabel.qp.ideal_optimality"
