module ClarabelExample.TestIdealSymbolicWrongArgument
import Clarabel.Qp (Solved, solve, Settings)
empty_values: List[f64] = []
def quality(q: tensor[2, f64], step: tensor[2, f64]) -> f64 = ((0.0f64 - (0.5f64 * tensor_to_scalar(sum(mul(step, step), 0i32)))) - tensor_to_scalar(sum(mul(q, step), 0i32)))
@property wrong_argument forall(q: tensor[2, f64], other_q: tensor[2, f64], baseline: tensor[2, f64]):
  match solve(to_tensor([[1.0f64, 0.0f64], [0.0f64, 1.0f64]]), q, reshape(to_tensor(empty_values), [0i64, 2i64]), to_tensor(empty_values), [], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 }) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => (quality(other_q, primal) >= quality(other_q, baseline))
    | _ => true
  }
  with contract = "clarabel.qp.ideal_optimality"
