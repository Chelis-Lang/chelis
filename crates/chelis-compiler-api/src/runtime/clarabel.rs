//! The evaluator's opt-in adapter for the checked Clarabel package function.
//! All values cross this boundary with their admitted runtime dtype and shape.

use super::{RuntimeTensorValue, RuntimeValue, Values};
use chelis_clarabel_provider::{Cone, DenseMatrix, DenseVector, Problem, Settings, Status, solve};
use chelis_types::types::Prim;

pub(super) const SOLVE: &str = "pkg__chelis__clarabel__Clarabel__Qp__solve";
const CTOR: &str = "Pkg__chelis__clarabel__Clarabel__Qp__";

fn constructor(name: &str, fields: Vec<RuntimeValue>, names: &[&str]) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: format!("{CTOR}{name}"),
        source_name: name.to_owned(),
        fields: Values::from(fields),
        field_names: (!names.is_empty())
            .then(|| names.iter().map(|name| (*name).to_owned()).collect()),
    }
}

fn tensor_arg(
    args: &[RuntimeValue],
    index: usize,
    rank: usize,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    let Some(RuntimeValue::Tensor(tensor)) = args.get(index) else {
        return Err(format!(
            "Clarabel.solve argument {index} must be an f64 tensor"
        ));
    };
    if tensor.precision != Prim::F64 || tensor.value.shape.len() != rank {
        return Err(format!(
            "Clarabel.solve argument {index} must have rank {rank} and dtype f64"
        ));
    }
    Ok((tensor.value.shape.clone(), tensor.value.to_f64_lossy_vec()))
}

fn integer(value: &RuntimeValue, context: &str) -> Result<i64, String> {
    match value {
        RuntimeValue::Scalar(payload) if payload.dtype() == Prim::Int64 => Ok(payload.as_i64()),
        _ => Err(format!("{context} must be i64")),
    }
}

fn float(value: &RuntimeValue, context: &str) -> Result<f64, String> {
    match value {
        RuntimeValue::Scalar(payload) if payload.dtype() == Prim::F64 => Ok(payload.as_f64_lossy()),
        _ => Err(format!("{context} must be f64")),
    }
}

fn cone(value: &RuntimeValue) -> Result<Cone, String> {
    let RuntimeValue::Adt { ctor, fields, .. } = value else {
        return Err("Clarabel.solve cone must be a Cone value".to_owned());
    };
    let name = ctor
        .strip_prefix(CTOR)
        .ok_or_else(|| "Clarabel.solve cone has the wrong package identity".to_owned())?;
    let dimension = |at: usize| -> Result<usize, String> {
        let n = integer(
            fields.get(at).ok_or("cone dimension missing")?,
            "cone dimension",
        )?;
        usize::try_from(n).map_err(|_| "cone dimension must be nonnegative".to_owned())
    };
    let parsed = match (name, fields.len()) {
        ("ZeroCone", 1) => Cone::zero(dimension(0)?),
        ("NonnegativeCone", 1) => Cone::nonnegative(dimension(0)?),
        ("SecondOrderCone", 1) => Cone::second_order(dimension(0)?),
        ("ExponentialCone", 0) => return Ok(Cone::exponential()),
        ("PowerCone", 1) => Cone::power(float(&fields[0], "power exponent")?),
        ("GeneralizedPowerCone", 2) => {
            let RuntimeValue::List(weights) = &fields[0] else {
                return Err("generalized-power weights must be List[f64]".to_owned());
            };
            let weights = weights
                .iter()
                .map(|value| float(value, "generalized-power weight"))
                .collect::<Result<Vec<_>, _>>()?;
            Cone::generalized_power(weights, dimension(1)?)
        }
        _ => return Err(format!("invalid Clarabel cone constructor {name}")),
    };
    parsed.map_err(|error| error.to_string())
}

fn settings(value: &RuntimeValue) -> Result<Settings, String> {
    let RuntimeValue::Adt { ctor, fields, .. } = value else {
        return Err("Clarabel.solve settings must be a Settings value".to_owned());
    };
    if ctor != &format!("{CTOR}Settings") || fields.len() != 4 {
        return Err("Clarabel.solve settings have the wrong constructor".to_owned());
    }
    let iterations = u32::try_from(integer(&fields[0], "max_iterations")?)
        .map_err(|_| "max_iterations must fit u32".to_owned())?;
    Settings::new(
        iterations,
        float(&fields[1], "absolute_gap_tolerance")?,
        float(&fields[2], "relative_gap_tolerance")?,
        float(&fields[3], "feasibility_tolerance")?,
    )
    .map_err(|error| error.to_string())
}

fn vector(values: &[f64]) -> Result<RuntimeValue, String> {
    RuntimeTensorValue::from_wide(
        "clarabel_solve",
        Prim::F64,
        vec![values.len()],
        values.to_vec(),
    )
    .map(RuntimeValue::Tensor)
}

fn status_name(status: Status) -> &'static str {
    match status {
        Status::Solved => "SolvedStatus",
        Status::AlmostSolved => "AlmostSolvedStatus",
        Status::PrimalInfeasible => "PrimalInfeasibleStatus",
        Status::DualInfeasible => "DualInfeasibleStatus",
        Status::AlmostPrimalInfeasible => "AlmostPrimalInfeasibleStatus",
        Status::AlmostDualInfeasible => "AlmostDualInfeasibleStatus",
        Status::MaxIterations => "MaxIterationsStatus",
        Status::MaxTime => "MaxTimeStatus",
        Status::NumericalError => "NumericalErrorStatus",
        Status::InsufficientProgress => "InsufficientProgressStatus",
        Status::CallbackTerminated => "CallbackTerminatedStatus",
        Status::Unsolved => "UnsolvedStatus",
    }
}

pub(super) fn invoke(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    if args.len() != 6 {
        return Err(format!(
            "Clarabel.solve expects 6 arguments, got {}",
            args.len()
        ));
    }
    let (p_shape, p_data) = tensor_arg(args, 0, 2)?;
    let (_, q_data) = tensor_arg(args, 1, 1)?;
    let (a_shape, a_data) = tensor_arg(args, 2, 2)?;
    let (_, b_data) = tensor_arg(args, 3, 1)?;
    let RuntimeValue::List(cones) = &args[4] else {
        return Err("Clarabel.solve cones must be List[Cone]".to_owned());
    };
    let cones = cones.iter().map(cone).collect::<Result<Vec<_>, _>>()?;
    let problem = Problem::new(
        DenseMatrix::new(p_shape[0], p_shape[1], p_data).map_err(|error| error.to_string())?,
        DenseVector::new(q_data).map_err(|error| error.to_string())?,
        DenseMatrix::new(a_shape[0], a_shape[1], a_data).map_err(|error| error.to_string())?,
        DenseVector::new(b_data).map_err(|error| error.to_string())?,
        cones,
    )
    .map_err(|error| error.to_string())?;
    let result = solve(&problem, &settings(&args[5])?).map_err(|error| error.to_string())?;
    if result.status() == Status::Solved {
        Ok(constructor(
            "Solved",
            vec![
                vector(result.primal().as_slice())?,
                vector(result.dual().as_slice())?,
                vector(result.slack().as_slice())?,
                RuntimeValue::int64(i64::from(result.iterations())),
                RuntimeValue::float64(result.primal_residual()),
                RuntimeValue::float64(result.dual_residual()),
            ],
            &[
                "primal",
                "dual",
                "slack",
                "iterations",
                "primal_residual",
                "dual_residual",
            ],
        ))
    } else {
        Ok(constructor(
            "Stopped",
            vec![constructor(status_name(result.status()), vec![], &[])],
            &["status"],
        ))
    }
}
