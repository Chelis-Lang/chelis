//! Shape and dtype inference for elementwise scatter.

use super::*;

/// Enforce [05-SPARSE-1]. Unlike `scatter`, indices and updates keep the data
/// rank and share one shape; the result always has the data type.
pub(super) fn check_scatter_elements(
    list: &deep::List,
    kids: &[deep::Expr],
    arg_tys: &[Type],
    pending_ty: Type,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if arg_tys.len() != 4 {
        return report_builtin_arity(errors, list, "scatter_elements", 4, arg_tys.len());
    }
    let data_ty = subst.apply(&arg_tys[0]);
    let indices_ty = subst.apply(&arg_tys[1]);
    let updates_ty = subst.apply(&arg_tys[2]);
    let (data_dims, data_prec) = match &data_ty {
        Type::Tensor(dims, prec) => (dims.clone(), prec.clone()),
        Type::Var(_) | Type::Error(_) => return pending_ty,
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("scatter_elements expects tensor data, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };
    let (index_dims, index_prec) = match &indices_ty {
        Type::Tensor(dims, prec) => (dims.clone(), prec.clone()),
        Type::Var(_) | Type::Error(_) => return pending_ty,
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "scatter_elements expects int32 or int64 tensor indices, got {other}"
                        ),
                    ),
                    vec![],
                ),
            );
        }
    };
    if !matches!(index_prec, TensorPrec::Concrete(Prim::Int32 | Prim::Int64)) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "scatter_elements expects int32 or int64 tensor indices, got tensor[..., {}]",
                        index_prec.name()
                    ),
                ),
                vec![],
            ),
        );
    }
    let (update_dims, update_prec) = match &updates_ty {
        Type::Tensor(dims, prec) => (dims.clone(), prec.clone()),
        Type::Var(_) | Type::Error(_) => return pending_ty,
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("scatter_elements expects tensor updates, got {other}"),
                    ),
                    vec![],
                ),
            );
        }
    };
    if data_dims.len() != index_dims.len() || index_dims.len() != update_dims.len() {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "scatter_elements requires data, indices, and updates to share a rank; got {}, {}, and {}",
                    data_dims.len(),
                    index_dims.len(),
                    update_dims.len()
                ),
                vec![],
            ),
        );
    }
    if let Err(error) = unify_tensor_prec(&data_prec, &update_prec, subst) {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "scatter_elements updates precision must match data precision: {}",
                    error.message
                ),
                vec![],
            ),
        );
    }
    for (index_dim, update_dim) in index_dims.iter().zip(&update_dims) {
        if let Err(error) = unify_dim(index_dim, update_dim, subst) {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "scatter_elements updates shape must equal indices shape: {}",
                        error.message
                    ),
                    vec![],
                ),
            );
        }
    }
    let axis = match resolve_builtin_axis(
        "scatter_elements",
        kids.get(4),
        &subst.apply(&arg_tys[3]),
        &data_ty,
        list,
        errors,
    ) {
        Ok(axis) => axis,
        Err(error) => return error,
    };
    for (dim_index, (index_dim, data_dim)) in index_dims.iter().zip(&data_dims).enumerate() {
        if dim_index == axis {
            continue;
        }
        let index_dim = subst.apply_dim(index_dim);
        let data_dim = subst.apply_dim(data_dim);
        match (&index_dim, &data_dim) {
            (Dim::Lit(index), Dim::Lit(data)) if index <= data => {}
            (Dim::Lit(index), Dim::Lit(data)) => {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "scatter_elements indices extent {index} exceeds data extent {data} on non-axis dimension {dim_index}"
                        ),
                        vec![],
                    ),
                );
            }
            (left, right) if left == right => {}
            _ => {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "scatter_elements cannot prove that the symbolic indices extent fits the symbolic data extent on non-axis dimension {dim_index}; add a concrete shape annotation"
                        ),
                        vec![],
                    ),
                );
            }
        }
    }
    subst.apply(&data_ty)
}
