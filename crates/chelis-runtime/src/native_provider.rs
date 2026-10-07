//! Versioned tagged-value host provider entry point. Inputs are borrowed;
//! the result owns its Chelis runtime objects.

use super::*;
use chelis_clarabel_provider::{solve, Cone, DenseMatrix, DenseVector, Problem, Settings, Status};

const QP_SOLVE: &str = "chelis-clarabel/0.1.0/Clarabel.Qp.solve";

fn invalid(message: impl Into<String>) -> String {
    format!("Clarabel native provider: {}", message.into())
}

unsafe fn float(value: chelis_value, field: &str) -> Result<f64, String> {
    if value.tag != CHELIS_VALUE_SCALAR {
        return Err(invalid(format!("{field} must be f64")));
    }
    let scalar = unsafe { value.payload.scalar };
    if scalar.dtype != CHELIS_DTYPE_F64 || scalar.reserved != [0; 7] {
        return Err(invalid(format!("{field} must be f64")));
    }
    Ok(f64::from_bits(scalar.bits))
}

unsafe fn integer(value: chelis_value, field: &str) -> Result<i64, String> {
    if value.tag != CHELIS_VALUE_SCALAR {
        return Err(invalid(format!("{field} must be i64")));
    }
    let scalar = unsafe { value.payload.scalar };
    if scalar.dtype != CHELIS_DTYPE_I64 || scalar.reserved != [0; 7] {
        return Err(invalid(format!("{field} must be i64")));
    }
    Ok(i64::from_ne_bytes(scalar.bits.to_ne_bytes()))
}

unsafe fn adt<'a>(value: chelis_value, field: &str) -> Result<&'a chelis_adt, String> {
    if value.tag != CHELIS_VALUE_ADT {
        return Err(invalid(format!("{field} must be an ADT")));
    }
    let ptr = unsafe { value.payload.adt };
    if ptr.is_null() {
        return Err(invalid(format!("{field} is null")));
    }
    Ok(unsafe { &*ptr })
}

unsafe fn list<'a>(value: chelis_value, field: &str) -> Result<&'a [chelis_value], String> {
    if value.tag != CHELIS_VALUE_LIST {
        return Err(invalid(format!("{field} must be a list")));
    }
    let ptr = unsafe { value.payload.list };
    if ptr.is_null() {
        return Err(invalid(format!("{field} is null")));
    }
    Ok(unsafe { (*ptr).live() })
}

unsafe fn tensor(
    value: chelis_value,
    rank: i32,
    field: &str,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    if value.tag != CHELIS_VALUE_TENSOR {
        return Err(invalid(format!("{field} must be an f64 tensor")));
    }
    let ptr = unsafe { value.payload.tensor };
    if ptr.is_null() || unsafe { chelis_tensor_rank(ptr) } != rank {
        return Err(invalid(format!("{field} must have rank {rank}")));
    }
    let view = unsafe { chelis_tensor_read_view(ptr) };
    if view.dtype != CHELIS_DTYPE_F64 || view.count < 0 {
        return Err(invalid(format!("{field} must have dtype f64")));
    }
    let shape = (0..rank)
        .map(|axis| {
            usize::try_from(unsafe { chelis_tensor_shape(ptr, axis) })
                .map_err(|_| invalid(format!("{field} has an invalid extent")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = usize::try_from(view.count).map_err(|_| invalid("tensor count overflows"))?;
    if shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b)) != Some(count) {
        return Err(invalid(format!("{field} has inconsistent extents")));
    }
    let elements = unsafe { chelis_tensor_elements(ptr) };
    let data = unsafe { (*elements).live() }
        .iter()
        .map(|value| unsafe { float(*value, field) })
        .collect::<Result<Vec<_>, _>>();
    unsafe { chelis_list_release(elements) };
    let data = data?;
    if data.len() != count {
        return Err(invalid(format!("{field} has inconsistent elements")));
    }
    Ok((shape, data))
}

unsafe fn cone(value: chelis_value) -> Result<Cone, String> {
    let value = unsafe { adt(value, "cone") }?;
    let name = unsafe { string_value(value.ctor) }.value.as_str();
    let fields = &value.fields;
    let dimension = |index: usize| -> Result<usize, String> {
        let value = fields
            .get(index)
            .ok_or_else(|| invalid("cone dimension missing"))?;
        usize::try_from(unsafe { integer(*value, "cone dimension") }?)
            .map_err(|_| invalid("cone dimension must be nonnegative"))
    };
    let parsed = match (name, fields.len()) {
        ("ZeroCone", 1) => Cone::zero(dimension(0)?),
        ("NonnegativeCone", 1) => Cone::nonnegative(dimension(0)?),
        ("SecondOrderCone", 1) => Cone::second_order(dimension(0)?),
        ("ExponentialCone", 0) => return Ok(Cone::exponential()),
        ("PowerCone", 1) => Cone::power(unsafe { float(fields[0], "power exponent") }?),
        ("GeneralizedPowerCone", 2) => {
            let weights = unsafe { list(fields[0], "generalized-power weights") }?
                .iter()
                .map(|value| unsafe { float(*value, "generalized-power weight") })
                .collect::<Result<Vec<_>, _>>()?;
            Cone::generalized_power(weights, dimension(1)?)
        }
        _ => return Err(invalid(format!("invalid cone constructor {name}"))),
    };
    parsed.map_err(|error| invalid(error.to_string()))
}

unsafe fn settings(value: chelis_value) -> Result<Settings, String> {
    let value = unsafe { adt(value, "settings") }?;
    if unsafe { string_value(value.ctor) }.value != "Settings" || value.fields.len() != 4 {
        return Err(invalid("invalid Settings constructor"));
    }
    let fields = &value.fields;
    let iterations = u32::try_from(unsafe { integer(fields[0], "max_iterations") }?)
        .map_err(|_| invalid("max_iterations must fit u32"))?;
    Settings::new(
        iterations,
        unsafe { float(fields[1], "absolute_gap_tolerance") }?,
        unsafe { float(fields[2], "relative_gap_tolerance") }?,
        unsafe { float(fields[3], "feasibility_tolerance") }?,
    )
    .map_err(|error| invalid(error.to_string()))
}

unsafe fn output_vector(values: &[f64]) -> Result<chelis_value, String> {
    let extent = i64::try_from(values.len()).map_err(|_| invalid("result extent overflows i64"))?;
    let tensor = unsafe { chelis_alloc(1, &extent, CHELIS_DTYPE_F64) };
    let guard = unsafe { chelis_tensor_begin_write(tensor) };
    let scalars = values
        .iter()
        .map(|value| chelis_scalar_from_bits(CHELIS_DTYPE_F64, value.to_bits()))
        .collect::<Vec<_>>();
    unsafe {
        chelis_tensor_write_literal(
            guard,
            chelis_scalar_from_bits(CHELIS_DTYPE_I64, extent as u64),
            scalars.as_ptr(),
        )
    };
    unsafe { chelis_tensor_end_write(guard) };
    Ok(unsafe { chelis_value_take_tensor(tensor) })
}

unsafe fn construct(name: &str, fields: Vec<chelis_value>) -> chelis_value {
    let ctor = new_runtime_string(name.to_owned());
    let adt = new_adt(ctor, fields, "chelis_native_provider_call_v1");
    unsafe { chelis_value_take_adt(adt) }
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

unsafe fn solve_qp(args: &[chelis_value]) -> Result<chelis_value, String> {
    if args.len() != 6 {
        return Err(invalid(format!("expected 6 arguments, got {}", args.len())));
    }
    let (p_shape, p_data) = unsafe { tensor(args[0], 2, "P") }?;
    let (_, q_data) = unsafe { tensor(args[1], 1, "q") }?;
    let (a_shape, a_data) = unsafe { tensor(args[2], 2, "A") }?;
    let (_, b_data) = unsafe { tensor(args[3], 1, "b") }?;
    let cones = unsafe { list(args[4], "cones") }?
        .iter()
        .map(|value| unsafe { cone(*value) })
        .collect::<Result<Vec<_>, _>>()?;
    let problem = Problem::new(
        DenseMatrix::new(p_shape[0], p_shape[1], p_data).map_err(|e| invalid(e.to_string()))?,
        DenseVector::new(q_data).map_err(|e| invalid(e.to_string()))?,
        DenseMatrix::new(a_shape[0], a_shape[1], a_data).map_err(|e| invalid(e.to_string()))?,
        DenseVector::new(b_data).map_err(|e| invalid(e.to_string()))?,
        cones,
    )
    .map_err(|e| invalid(e.to_string()))?;
    let result =
        solve(&problem, &unsafe { settings(args[5]) }?).map_err(|e| invalid(e.to_string()))?;
    if result.status() == Status::Solved {
        Ok(unsafe {
            construct(
                "Solved",
                vec![
                    output_vector(result.primal().as_slice())?,
                    output_vector(result.dual().as_slice())?,
                    output_vector(result.slack().as_slice())?,
                    internal_value_from_i64(i64::from(result.iterations())),
                    internal_value_from_f64(result.primal_residual()),
                    internal_value_from_f64(result.dual_residual()),
                ],
            )
        })
    } else {
        Ok(unsafe {
            let status = construct(status_name(result.status()), Vec::new());
            construct("Stopped", vec![status])
        })
    }
}

/// ABI v1: one operation identity, borrowed tagged inputs, tagged i64 arity,
/// and one owned tagged result. The operation name is checked before dispatch.
#[no_mangle]
pub unsafe extern "C" fn chelis_native_provider_call_v1(
    operation: chelis_string,
    args: *const chelis_value,
    arity: chelis_scalar,
) -> chelis_value {
    let count = exact_i64_scalar(arity, "native provider arity");
    let identity = unsafe { string_value(operation) }.value.as_str();
    let expected_arity = match identity {
        QP_SOLVE => 6,
        _ => runtime_fail!("Clarabel native provider: unknown operation `{identity}`"),
    };
    if count != expected_arity || args.is_null() {
        runtime_fail!("Domain: native provider received invalid arguments");
    }
    let arguments = unsafe { std::slice::from_raw_parts(args, expected_arity as usize) };
    let result = match identity {
        QP_SOLVE => unsafe { solve_qp(arguments) },
        _ => unreachable!("operation checked above"),
    };
    result.unwrap_or_else(|message| runtime_fail!("{message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn input_tensor(shape: &[i64], values: &[f64]) -> chelis_value {
        let tensor = unsafe { chelis_alloc(shape.len() as i32, shape.as_ptr(), CHELIS_DTYPE_F64) };
        let guard = unsafe { chelis_tensor_begin_write(tensor) };
        let view = unsafe { chelis_tensor_write_view(guard) };
        assert_eq!(view.count, values.len() as i64);
        if !values.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), view.data.cast(), values.len());
            }
        }
        unsafe { chelis_tensor_end_write(guard) };
        unsafe { chelis_value_take_tensor(tensor) }
    }

    unsafe fn arguments(rows: i64) -> [chelis_value; 6] {
        let a = vec![0.0; rows as usize];
        let b = vec![0.0; rows as usize];
        [
            unsafe { input_tensor(&[1, 1], &[2.0]) },
            unsafe { input_tensor(&[1], &[-4.0]) },
            unsafe { input_tensor(&[rows, 1], &a) },
            unsafe { input_tensor(&[rows], &b) },
            unsafe { chelis_value_take_list(chelis_list_empty()) },
            unsafe {
                construct(
                    "Settings",
                    vec![
                        internal_value_from_i64(200),
                        internal_value_from_f64(1e-8),
                        internal_value_from_f64(1e-8),
                        internal_value_from_f64(1e-8),
                    ],
                )
            },
        ]
    }

    #[test]
    fn tagged_provider_call_returns_an_owned_solved_result() {
        unsafe {
            let args = arguments(0);
            let operation = new_runtime_string(QP_SOLVE.to_owned());
            let result = chelis_native_provider_call_v1(
                operation,
                args.as_ptr(),
                chelis_scalar_from_bits(CHELIS_DTYPE_I64, 6),
            );
            assert_eq!(result.tag, CHELIS_VALUE_ADT);
            let result_adt = adt(result, "result").expect("result ADT");
            assert_eq!(string_value(result_adt.ctor).value, "Solved");
            let (shape, primal) = tensor(result_adt.fields[0], 1, "primal").expect("primal");
            assert_eq!(shape, vec![1]);
            assert!((primal[0] - 2.0).abs() < 1e-6);
            chelis_value_release(result);
            chelis_string_release(operation);
            for arg in args {
                chelis_value_release(arg);
            }
        }
    }

    #[test]
    fn malformed_cone_partition_is_rejected_before_solver() {
        unsafe {
            let args = arguments(1);
            let error = match solve_qp(&args) {
                Ok(_) => panic!("missing cone must fail"),
                Err(error) => error,
            };
            assert!(error.contains("cone dimensions sum to 0, but b and A have 1 rows"));
            for arg in args {
                chelis_value_release(arg);
            }
        }
    }
}
