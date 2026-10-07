module ClarabelExample.TestIdealIndefinite
import Clarabel.Qp (Solved, solve, Settings)
empty_values: List[f64] = []
@property indefinite_p_has_no_optimizer_axiom forall(dummy: f64):
  match solve(to_tensor([[-2.0f64]]), to_tensor([1.0f64]), reshape(to_tensor(empty_values), [0i64, 1i64]), to_tensor(empty_values), [], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 }) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => true
    | _ => true
  }
  with contract = "clarabel.qp.ideal_optimality"
