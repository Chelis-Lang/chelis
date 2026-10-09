module ClarabelExample.TestIdealSymbolicConstrainedQuality
import Clarabel.Qp (NonnegativeCone, Solved, solve, Settings)
def apply_step(state: tensor[2, f64], step: tensor[2, f64]) -> tensor[2, f64] = add(state, step)
def quality(updated: tensor[2, f64], state: tensor[2, f64], p: tensor[2, 2, f64], q: tensor[2, f64]) -> f64 = {
  step = sub(updated, state)
  weighted = reshape(matmul(p, reshape(step, [2i64, 1i64])), [2i64])
  ((0.0f64 - (0.5f64 * tensor_to_scalar(sum(mul(step, weighted), 0i32)))) - tensor_to_scalar(sum(mul(q, step), 0i32)))
}
def row_value(a: tensor[1, 2, f64], vector: tensor[2, f64]) -> f64 = tensor_to_scalar(sum(reshape(matmul(a, reshape(vector, [2i64, 1i64])), [1i64]), 0i32))
@property constrained_quality forall(basis: tensor[2, 2, f64], q: tensor[2, f64], a: tensor[1, 2, f64], b: tensor[1, f64], state: tensor[2, f64], baseline: tensor[2, f64]) where row_value(a, baseline) <= tensor_to_scalar(sum(b, 0i32)):
  match solve(matmul(permute(basis, 1i32, 0i32), basis), q, a, b, [NonnegativeCone(1i64)], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-8f64, relative_gap_tolerance: 1e-8f64, feasibility_tolerance: 1e-8f64 }) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => (quality(apply_step(state, primal), state, matmul(permute(basis, 1i32, 0i32), basis), q) >= quality(apply_step(state, baseline), state, matmul(permute(basis, 1i32, 0i32), basis), q))
    | _ => true
  }
  with contract = "clarabel.qp.ideal_optimality"
