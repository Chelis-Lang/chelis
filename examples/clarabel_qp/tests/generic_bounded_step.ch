module ClarabelExample.TestGenericBoundedStep
import Clarabel.Qp (NonnegativeCone, Settings, Solved, solve)
def identity[n](template: &tensor[n, f64]) -> tensor[n, n, f64] = {
  indices = range(0i64, len(to_list(template)))
  rows = map(fn (row: i64) -> map(fn (column: i64) -> if eq(row, column) then 1.0f64 else 0.0f64, indices), indices)
  to_tensor(rows)
}
def bounded_step[n](p: &tensor[n, n, f64], g: &tensor[n, f64], theta: &tensor[n, f64], lo: &tensor[n, f64], hi: &tensor[n, f64]) = {
  count = len(to_list(theta))
  unit = identity(theta)
  a = concat([copy(unit), neg(unit)], 0i32)
  b = concat([sub(hi, theta), sub(theta, lo)], 0i32)
  solve(p, neg(g), a, b, [NonnegativeCone(mul(2i64, count))], Settings { max_iterations: 200i64, absolute_gap_tolerance: 1e-9f64, relative_gap_tolerance: 1e-9f64, feasibility_tolerance: 1e-9f64 })
}
def main() -> bool =
  match bounded_step(to_tensor([[2.0f64, 1.0f64], [1.0f64, 2.0f64]]), to_tensor([-1.0f64, 1.0f64]), to_tensor([0.0f64, 0.0f64]), to_tensor([0.0f64, 0.0f64]), to_tensor([2.0f64, 2.0f64])) with {
    | Solved { primal, dual, slack, iterations, primal_residual, dual_residual } => {
    values = to_list(primal)
    and(lt(abs(index(values, 0i64)), 1e-6f64), lt(abs(sub(index(values, 1i64), 0.5f64)), 1e-6f64))
  }
    | _ => false
  }
