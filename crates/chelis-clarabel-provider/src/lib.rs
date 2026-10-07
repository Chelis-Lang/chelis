#![cfg(feature = "solver")]

//! Validated, non-SDP Clarabel QP adapter.
//!
//! This crate is the solver portion of the planned optional native provider.
//! It does not export a Chelis function or a C ABI by itself.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputError(String);

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputError {}

fn invalid(message: impl Into<String>) -> InputError {
    InputError(message.into())
}

/// Row-major matrix whose shape and finite binary64 elements were checked.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, InputError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("matrix element count overflows usize"))?;
        if data.len() != expected {
            return Err(invalid(format!(
                "matrix shape [{rows},{cols}] requires {expected} elements, got {}",
                data.len()
            )));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(invalid("matrix contains a non-finite f64 value"));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    fn to_csc(&self, upper_triangle_only: bool) -> CscMatrix<f64> {
        let mut colptr = Vec::with_capacity(self.cols + 1);
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        colptr.push(0);
        for col in 0..self.cols {
            let end = if upper_triangle_only {
                self.rows.min(col + 1)
            } else {
                self.rows
            };
            for row in 0..end {
                let value = self.at(row, col);
                if value != 0.0 {
                    rowval.push(row);
                    nzval.push(value);
                }
            }
            colptr.push(nzval.len());
        }
        CscMatrix::new(self.rows, self.cols, colptr, rowval, nzval)
    }
}

/// Binary64 vector; user inputs require finite elements.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector {
    data: Vec<f64>,
}

impl DenseVector {
    pub fn new(data: Vec<f64>) -> Result<Self, InputError> {
        if data.iter().any(|value| !value.is_finite()) {
            return Err(invalid("vector contains a non-finite f64 value"));
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ConeKind {
    Zero(usize),
    Nonnegative(usize),
    SecondOrder(usize),
    Exponential,
    Power(f64),
    GeneralizedPower(Vec<f64>, usize),
}

/// A non-PSD Clarabel cone with validated parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Cone(ConeKind);

impl Cone {
    pub fn zero(dimension: usize) -> Result<Self, InputError> {
        if dimension == 0 {
            return Err(invalid("zero cone dimension must be positive"));
        }
        Ok(Self(ConeKind::Zero(dimension)))
    }

    pub fn nonnegative(dimension: usize) -> Result<Self, InputError> {
        if dimension == 0 {
            return Err(invalid("nonnegative cone dimension must be positive"));
        }
        Ok(Self(ConeKind::Nonnegative(dimension)))
    }

    pub fn second_order(dimension: usize) -> Result<Self, InputError> {
        if dimension < 2 {
            return Err(invalid("second-order cone dimension must be at least two"));
        }
        Ok(Self(ConeKind::SecondOrder(dimension)))
    }

    pub fn exponential() -> Self {
        Self(ConeKind::Exponential)
    }

    pub fn power(exponent: f64) -> Result<Self, InputError> {
        if !exponent.is_finite() || !(0.0..1.0).contains(&exponent) || exponent == 0.0 {
            return Err(invalid(
                "power-cone exponent must be finite and strictly between zero and one",
            ));
        }
        Ok(Self(ConeKind::Power(exponent)))
    }

    pub fn generalized_power(
        exponents: Vec<f64>,
        norm_dimension: usize,
    ) -> Result<Self, InputError> {
        if exponents.len() < 2 || norm_dimension == 0 {
            return Err(invalid(
                "generalized-power cone requires at least two exponents and a positive norm dimension",
            ));
        }
        if exponents
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0)
        {
            return Err(invalid(
                "generalized-power cone exponents must be finite and strictly between zero and one",
            ));
        }
        let sum: f64 = exponents.iter().sum();
        if (sum - 1.0).abs() > 1e-12 {
            return Err(invalid(
                "generalized-power cone exponents must sum to one within 1e-12",
            ));
        }
        Ok(Self(ConeKind::GeneralizedPower(exponents, norm_dimension)))
    }

    pub fn dimension(&self) -> usize {
        match &self.0 {
            ConeKind::Zero(n) | ConeKind::Nonnegative(n) | ConeKind::SecondOrder(n) => *n,
            ConeKind::Exponential | ConeKind::Power(_) => 3,
            ConeKind::GeneralizedPower(exponents, norm_dimension) => {
                exponents.len() + *norm_dimension
            }
        }
    }

    fn to_clarabel(&self) -> SupportedConeT<f64> {
        match &self.0 {
            ConeKind::Zero(n) => SupportedConeT::ZeroConeT(*n),
            ConeKind::Nonnegative(n) => SupportedConeT::NonnegativeConeT(*n),
            ConeKind::SecondOrder(n) => SupportedConeT::SecondOrderConeT(*n),
            ConeKind::Exponential => SupportedConeT::ExponentialConeT(),
            ConeKind::Power(alpha) => SupportedConeT::PowerConeT(*alpha),
            ConeKind::GeneralizedPower(alphas, n) => {
                SupportedConeT::GenPowerConeT(alphas.clone(), *n)
            }
        }
    }
}

/// Checked QP data. Convexity requires `P` positive semidefinite; this is a
/// mathematical caller premise and is not certified by the float adapter.
#[derive(Debug, Clone)]
pub struct Problem {
    p: DenseMatrix,
    q: DenseVector,
    a: DenseMatrix,
    b: DenseVector,
    cones: Vec<Cone>,
}

impl Problem {
    pub fn new(
        p: DenseMatrix,
        q: DenseVector,
        a: DenseMatrix,
        b: DenseVector,
        cones: Vec<Cone>,
    ) -> Result<Self, InputError> {
        let n = q.len();
        let m = b.len();
        if n == 0 || p.rows() != n || p.cols() != n || a.rows() != m || a.cols() != n {
            return Err(invalid(format!(
                "QP dimensions require P[{n},{n}], q[{n}], A[{m},{n}], b[{m}] with n positive"
            )));
        }
        for col in 0..n {
            for row in 0..col {
                if p.at(row, col) != p.at(col, row) {
                    return Err(invalid(format!("P is not symmetric at [{row},{col}]")));
                }
            }
        }
        let cone_dimension = cones.iter().try_fold(0usize, |sum, cone| {
            sum.checked_add(cone.dimension())
                .ok_or_else(|| invalid("cone dimensions overflow usize"))
        })?;
        if cone_dimension != m {
            return Err(invalid(format!(
                "cone dimensions sum to {cone_dimension}, but b and A have {m} rows"
            )));
        }
        Ok(Self { p, q, a, b, cones })
    }
}

/// Deliberately small settings subset. Other Clarabel settings stay pinned to
/// the upstream defaults apart from quiet, one-thread, no-time-limit execution.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    max_iterations: u32,
    absolute_gap_tolerance: f64,
    relative_gap_tolerance: f64,
    feasibility_tolerance: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            absolute_gap_tolerance: 1e-8,
            relative_gap_tolerance: 1e-8,
            feasibility_tolerance: 1e-8,
        }
    }
}

impl Settings {
    pub fn new(
        max_iterations: u32,
        absolute_gap_tolerance: f64,
        relative_gap_tolerance: f64,
        feasibility_tolerance: f64,
    ) -> Result<Self, InputError> {
        if [
            absolute_gap_tolerance,
            relative_gap_tolerance,
            feasibility_tolerance,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid("solver tolerances must be finite and positive"));
        }
        Ok(Self {
            max_iterations,
            absolute_gap_tolerance,
            relative_gap_tolerance,
            feasibility_tolerance,
        })
    }

    fn to_clarabel(self) -> DefaultSettings<f64> {
        DefaultSettings {
            max_iter: self.max_iterations,
            tol_gap_abs: self.absolute_gap_tolerance,
            tol_gap_rel: self.relative_gap_tolerance,
            tol_feas: self.feasibility_tolerance,
            max_threads: 1,
            time_limit: f64::INFINITY,
            verbose: false,
            direct_solve_method: "qdldl".to_owned(),
            ..DefaultSettings::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Solved,
    AlmostSolved,
    PrimalInfeasible,
    DualInfeasible,
    AlmostPrimalInfeasible,
    AlmostDualInfeasible,
    MaxIterations,
    MaxTime,
    NumericalError,
    InsufficientProgress,
    CallbackTerminated,
    Unsolved,
}

impl From<SolverStatus> for Status {
    fn from(value: SolverStatus) -> Self {
        match value {
            SolverStatus::Solved => Self::Solved,
            SolverStatus::AlmostSolved => Self::AlmostSolved,
            SolverStatus::PrimalInfeasible => Self::PrimalInfeasible,
            SolverStatus::DualInfeasible => Self::DualInfeasible,
            SolverStatus::AlmostPrimalInfeasible => Self::AlmostPrimalInfeasible,
            SolverStatus::AlmostDualInfeasible => Self::AlmostDualInfeasible,
            SolverStatus::MaxIterations => Self::MaxIterations,
            SolverStatus::MaxTime => Self::MaxTime,
            SolverStatus::NumericalError => Self::NumericalError,
            SolverStatus::InsufficientProgress => Self::InsufficientProgress,
            SolverStatus::CallbackTerminated => Self::CallbackTerminated,
            SolverStatus::Unsolved => Self::Unsolved,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    status: Status,
    primal: DenseVector,
    dual: DenseVector,
    slack: DenseVector,
    iterations: u32,
    primal_residual: f64,
    dual_residual: f64,
}

impl SolveResult {
    pub fn status(&self) -> Status {
        self.status
    }
    pub fn primal(&self) -> &DenseVector {
        &self.primal
    }
    pub fn dual(&self) -> &DenseVector {
        &self.dual
    }
    pub fn slack(&self) -> &DenseVector {
        &self.slack
    }
    pub fn iterations(&self) -> u32 {
        self.iterations
    }
    pub fn primal_residual(&self) -> f64 {
        self.primal_residual
    }
    pub fn dual_residual(&self) -> f64 {
        self.dual_residual
    }
}

struct SolverOutput<'a> {
    primal: &'a [f64],
    dual: &'a [f64],
    slack: &'a [f64],
    primal_residual: f64,
    dual_residual: f64,
}

fn validate_solved_output(
    status: Status,
    output: &SolverOutput<'_>,
    dimensions: (usize, usize),
) -> Result<(), InputError> {
    if status != Status::Solved {
        return Ok(());
    }
    let (n, m) = dimensions;
    if output.primal.len() != n || output.dual.len() != m || output.slack.len() != m {
        return Err(invalid(
            "Clarabel returned solved vectors with incorrect extents",
        ));
    }
    if output
        .primal
        .iter()
        .chain(output.dual)
        .chain(output.slack)
        .any(|value| !value.is_finite())
        || !output.primal_residual.is_finite()
        || !output.dual_residual.is_finite()
    {
        return Err(invalid("Clarabel returned a non-finite solved value"));
    }
    Ok(())
}

/// Solve one validated QP. `Solved` remains an approximate numerical result;
/// no exact-optimality proposition is implied by this Rust return value.
pub fn solve(problem: &Problem, settings: &Settings) -> Result<SolveResult, InputError> {
    let p = problem.p.to_csc(true);
    let a = problem.a.to_csc(false);
    p.check_format()
        .map_err(|error| invalid(format!("P CSC format: {error}")))?;
    a.check_format()
        .map_err(|error| invalid(format!("A CSC format: {error}")))?;
    let cones: Vec<_> = problem.cones.iter().map(Cone::to_clarabel).collect();
    let mut solver = DefaultSolver::new(
        &p,
        problem.q.as_slice(),
        &a,
        problem.b.as_slice(),
        &cones,
        settings.to_clarabel(),
    )
    .map_err(|error| invalid(format!("Clarabel rejected the QP: {error}")))?;
    solver.solve();
    let solution = &solver.solution;
    let status = Status::from(solution.status);
    validate_solved_output(
        status,
        &SolverOutput {
            primal: &solution.x,
            dual: &solution.z,
            slack: &solution.s,
            primal_residual: solution.r_prim,
            dual_residual: solution.r_dual,
        },
        (problem.q.len(), problem.b.len()),
    )?;
    Ok(SolveResult {
        status,
        primal: DenseVector {
            data: solution.x.clone(),
        },
        dual: DenseVector {
            data: solution.z.clone(),
        },
        slack: DenseVector {
            data: solution.s.clone(),
        },
        iterations: solution.iterations,
        primal_residual: solution.r_prim,
        dual_residual: solution.r_dual,
    })
}

#[cfg(test)]
mod output_validation_tests {
    use super::*;

    #[test]
    fn solved_result_requires_declared_extents_and_finite_diagnostics() {
        let valid = SolverOutput {
            primal: &[1.0],
            dual: &[],
            slack: &[],
            primal_residual: 0.0,
            dual_residual: 0.0,
        };
        assert!(validate_solved_output(Status::Solved, &valid, (1, 0)).is_ok());
        assert!(
            validate_solved_output(
                Status::Solved,
                &SolverOutput {
                    primal: &[],
                    ..valid
                },
                (1, 0)
            )
            .is_err()
        );
        assert!(validate_solved_output(Status::Solved, &valid, (1, 1)).is_err());
        assert!(
            validate_solved_output(
                Status::Solved,
                &SolverOutput {
                    primal_residual: f64::NAN,
                    ..valid
                },
                (1, 0)
            )
            .is_err()
        );
        assert!(
            validate_solved_output(
                Status::Solved,
                &SolverOutput {
                    dual_residual: f64::INFINITY,
                    ..valid
                },
                (1, 0)
            )
            .is_err()
        );
        assert!(
            validate_solved_output(
                Status::Solved,
                &SolverOutput {
                    primal: &[f64::NAN],
                    ..valid
                },
                (1, 0)
            )
            .is_err()
        );
        assert!(
            validate_solved_output(
                Status::MaxIterations,
                &SolverOutput {
                    primal: &[],
                    primal_residual: f64::NAN,
                    ..valid
                },
                (1, 0)
            )
            .is_ok()
        );
    }
}
