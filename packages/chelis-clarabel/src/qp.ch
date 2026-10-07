module Clarabel.Qp
export (Cone, ZeroCone, NonnegativeCone, SecondOrderCone, ExponentialCone, PowerCone, GeneralizedPowerCone, Settings, SolveStatus, SolvedStatus, AlmostSolvedStatus, PrimalInfeasibleStatus, DualInfeasibleStatus, AlmostPrimalInfeasibleStatus, AlmostDualInfeasibleStatus, MaxIterationsStatus, MaxTimeStatus, NumericalErrorStatus, InsufficientProgressStatus, CallbackTerminatedStatus, UnsolvedStatus, SolveResult, Solved, Stopped, solve)
type Cone =
  | ZeroCone(i64)
  | NonnegativeCone(i64)
  | SecondOrderCone(i64)
  | ExponentialCone
  | PowerCone(f64)
  | GeneralizedPowerCone(List[f64], i64)
type Settings =
  | Settings { max_iterations: i64, absolute_gap_tolerance: f64, relative_gap_tolerance: f64, feasibility_tolerance: f64 }
type SolveStatus =
  | SolvedStatus
  | AlmostSolvedStatus
  | PrimalInfeasibleStatus
  | DualInfeasibleStatus
  | AlmostPrimalInfeasibleStatus
  | AlmostDualInfeasibleStatus
  | MaxIterationsStatus
  | MaxTimeStatus
  | NumericalErrorStatus
  | InsufficientProgressStatus
  | CallbackTerminatedStatus
  | UnsolvedStatus
type SolveResult[n, m] =
  | Solved { primal: tensor[n, f64], dual: tensor[m, f64], slack: tensor[m, f64], iterations: i64, primal_residual: f64, dual_residual: f64 }
  | Stopped { status: SolveStatus }
sig solve[n, m]: &tensor[n, n, f64] -> &tensor[n, f64] -> &tensor[m, n, f64] -> &tensor[m, f64] -> List[Cone] -> Settings -> SolveResult[n, m]
def solve(p, q, a, b, cones, settings) = fail("Clarabel native provider unavailable")
