//! The shipped standard-library graphs, evaluated at a declared dtype.
//!
//! The concrete evaluator and the standard-contract fuzz discharges need the
//! value of `Std.Contracts.normal_cdf` at a property's declared width. They do
//! not carry their own approximation of it: this module runs the bundled
//! `normal_cdf` graph through the compiler's evaluation lane, which is the
//! evaluator `chelis eval` uses, so a fuzz verdict is a statement about the
//! function Chelis ships (chelis#2965).
//!
//! Inputs travel as an exact tagged tensor binding and results come back as
//! exact scalars, so no value is rounded on the way in or out. One evaluation
//! handles a whole batch, and each (dtype, input bits) pair is memoized per
//! process, so repeated predicates do not recompile the program.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_types::{ScalarValue, tensor_from_scalars, types::Prim};

type Memo = Mutex<BTreeMap<(&'static str, u64), ScalarValue>>;

fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The bit pattern that identifies a float scalar inside its dtype.
fn float_key(value: ScalarValue) -> Option<(&'static str, u64)> {
    let prim = value.prim();
    if !prim.is_float() {
        return None;
    }
    // Every float dtype embeds exactly in f64, so the f64 bits identify the
    // value within its dtype, signed zero included.
    Some((prim.name(), value.as_f64_lossy().to_bits()))
}

/// `Std.Contracts.normal_cdf` at each input's own dtype. Every input must
/// share one float dtype; the result has that dtype.
pub fn normal_cdf_batch(inputs: &[ScalarValue]) -> Result<Vec<ScalarValue>, String> {
    let Some(first) = inputs.first() else {
        return Ok(Vec::new());
    };
    let prim = first.prim();
    let type_name = match prim {
        Prim::F32 => "f32",
        Prim::F64 => "f64",
        Prim::F16 => "f16",
        Prim::Bf16 => "bf16",
        other => {
            return Err(format!(
                "normal_cdf is defined for float dtypes, not {}",
                other.name()
            ));
        }
    };
    if inputs.iter().any(|value| value.prim() != prim) {
        return Err("normal_cdf batch mixes dtypes".to_string());
    }

    let mut missing = Vec::new();
    {
        let memo = memo().lock().map_err(|_| "normal_cdf memo poisoned")?;
        let mut queued = BTreeSet::new();
        for value in inputs {
            let key = float_key(*value).expect("dtype checked above");
            if !memo.contains_key(&key) && queued.insert(key) {
                missing.push(*value);
            }
        }
    }
    if !missing.is_empty() {
        let computed = evaluate_normal_cdf(type_name, prim, &missing)?;
        let mut memo = memo().lock().map_err(|_| "normal_cdf memo poisoned")?;
        for (input, output) in missing.iter().zip(computed) {
            memo.insert(float_key(*input).expect("dtype checked above"), output);
        }
    }
    let memo = memo().lock().map_err(|_| "normal_cdf memo poisoned")?;
    Ok(inputs
        .iter()
        .map(|value| memo[&float_key(*value).expect("dtype checked above")])
        .collect())
}

/// `Std.Contracts.normal_cdf` at the input's own dtype.
pub fn normal_cdf(input: ScalarValue) -> Result<ScalarValue, String> {
    normal_cdf_batch(std::slice::from_ref(&input)).map(|mut values| values.remove(0))
}

fn evaluate_normal_cdf(
    type_name: &str,
    prim: Prim,
    inputs: &[ScalarValue],
) -> Result<Vec<ScalarValue>, String> {
    let count = inputs.len();
    let source = format!(
        "import Std.Contracts (normal_cdf)\n\
         def prove_ncdf_outputs(prove_ncdf_inputs: tensor[{count}, {type_name}]) -> tensor[{count}, {type_name}] = \
         to_tensor(map(fn (x: {type_name}) -> normal_cdf(x), to_list(prove_ncdf_inputs)))\n"
    );
    // Link the bundled chelis-std exactly as a single-file `chelis eval`
    // program does, so the graph evaluated is the one Chelis ships.
    let entry = chelis_surf::parser::parse_str(&source)
        .map_err(|err| format!("normal_cdf probe did not parse: {err:?}"))?;
    let program = chelis_reef::prepare_single_file_program("prove-normal-cdf", &entry)?
        .ok_or("normal_cdf probe did not link chelis-std")?;
    let bindings = [(
        "prove_ncdf_inputs".to_string(),
        TensorValue {
            shape: vec![i64::try_from(count).map_err(|_| "normal_cdf batch too large")?],
            data: tensor_from_scalars(prim, inputs),
        },
    )]
    .into_iter()
    .collect();
    // The linked decls carry the linker's internal names, which the checker
    // accepts only under the linked-program guard.
    let _linked = chelis_types::install_linked_program_guard();
    let result = compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: chelis_surf::format::format_program(&program.decls),
            bindings,
        },
        &["prove_ncdf_outputs".to_string()],
    )
    .map_err(|err| {
        err.errors
            .iter()
            .map(|diag| diag.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    let [root] = result.roots.as_slice() else {
        return Err("normal_cdf evaluation did not return exactly one root".to_string());
    };
    let ExecutionValue::Tensor { value } = &root.value else {
        return Err(format!(
            "normal_cdf evaluation returned a non-tensor value: {:?}",
            root.value
        ));
    };
    if value.data.prim() != prim || value.data.len() != count {
        return Err(format!(
            "normal_cdf evaluation returned {} values of {}, expected {count} of {}",
            value.data.len(),
            value.data.prim().name(),
            prim.name()
        ));
    }
    Ok((0..count)
        .map(|index| value.data.scalar_at(index))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_types::scalar_from_f64;

    fn f32_scalar(value: f32) -> ScalarValue {
        scalar_from_f64("test", Prim::F32, f64::from(value)).unwrap()
    }

    #[test]
    fn normal_cdf_is_the_eval_lane_result_of_the_shipped_graph() {
        // The same inputs through a plain single-file `normal_cdf` program,
        // spelled as literals rather than a tensor binding: the transport adds
        // no rounding, and the result keeps the operand dtype.
        let source = "import Std.Contracts (normal_cdf)\n\
                      a = normal_cdf(1.0f32)\n\
                      b = normal_cdf(neg(2.5f32))\n\
                      c = normal_cdf(0.75f64)\n";
        let entry = chelis_surf::parser::parse_str(source).unwrap();
        let program = chelis_reef::prepare_single_file_program("t", &entry)
            .unwrap()
            .unwrap();
        let _linked = chelis_types::install_linked_program_guard();
        let roots = compiler::eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: chelis_surf::format::format_program(&program.decls),
                bindings: Default::default(),
            },
            &["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap()
        .roots;
        let expected = |name: &str| {
            let root = roots
                .iter()
                .find(|root| root.name.as_deref() == Some(name))
                .unwrap();
            match &root.value {
                ExecutionValue::Scalar { value } => value.get(),
                other => panic!("{name}: {other:?}"),
            }
        };
        let batch = normal_cdf_batch(&[f32_scalar(1.0), f32_scalar(-2.5)]).unwrap();
        for (got, name) in batch.iter().zip(["a", "b"]) {
            assert_eq!(got.prim(), Prim::F32);
            assert_eq!(
                got.as_f64_lossy().to_bits(),
                expected(name).as_f64_lossy().to_bits()
            );
        }
        let c = normal_cdf(scalar_from_f64("test", Prim::F64, 0.75).unwrap()).unwrap();
        assert_eq!(c.prim(), Prim::F64);
        assert_eq!(
            c.as_f64_lossy().to_bits(),
            expected("c").as_f64_lossy().to_bits()
        );
    }

    #[test]
    fn normal_cdf_rejects_mixed_and_non_float_batches() {
        let f64_value = scalar_from_f64("test", Prim::F64, 0.5).unwrap();
        assert!(normal_cdf_batch(&[f32_scalar(0.5), f64_value]).is_err());
        let int = chelis_types::scalar_from_i64("test", Prim::Int32, 1).unwrap();
        assert!(normal_cdf(int).is_err());
    }
}
