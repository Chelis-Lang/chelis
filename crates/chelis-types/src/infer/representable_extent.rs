//! Check-time representability of literal tensor sizes.
//!
//! A tensor's element count, suffix strides, and byte size are checked i64
//! products ([04-NUM-11]; the spec/05 descriptor rules), and a run-time size
//! outside them traps `Overflow` under its owning operation before allocation
//! (spec/04-type-system.md section 4.7). When every extent of a tensor type
//! is a literal, that failure is proven before anything runs, and section 4.7
//! makes it a type error: "A violation proven from literals is a type error."
//! The rule reads the final type of every checked node, so a declared
//! parameter, an ascription, and an inferred `expand` result are one case
//! (chelis#3418). A type with a symbolic or runtime extent keeps the run-time
//! trap.

use super::*;
use chelis_abi::metadata::{MetadataError, ShapeMetadata};

pub(super) fn validate_representable_tensor_types(
    exprs: &[deep::Expr],
    product: &InferenceProduct,
    errors: &mut DiagnosticSink<'_>,
) {
    let violations = product
        .owner_types
        .to_sorted()
        .into_iter()
        .filter_map(|(key, owner)| unrepresentable_tensor(owner.ty()).map(|found| (*key, found)))
        .collect::<BTreeMap<_, _>>();
    if violations.is_empty() {
        return;
    }
    // Report in source order, once per distinct type: the first node whose
    // final type carries it, children before parents, so the operation that
    // introduces the type owns the diagnostic rather than its binding.
    let mut reported = BTreeSet::new();
    for expr in exprs {
        report_in_post_order(expr, None, &violations, &mut reported, errors);
    }
}

/// `enclosing_span` is the nearest ancestor's span, which locates a node the
/// desugarer synthesized without one.
fn report_in_post_order(
    expr: &deep::Expr,
    enclosing_span: Option<&str>,
    violations: &BTreeMap<usize, Unrepresentable>,
    reported: &mut BTreeSet<String>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!(
        "validate_representable_tensor_types (report_in_post_order)",
        expr
    );
    let span = match expr.carrier() {
        deep::ExprCarrier::DecodedNode(_, metadata, _) => {
            metadata.span_id().map(|id| id.value()).or(enclosing_span)
        }
        _ => enclosing_span,
    };
    match expr.carrier() {
        deep::ExprCarrier::DecodedNode(_, _, children)
        | deep::ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                report_in_post_order(child, span, violations, reported, errors);
            }
        }
        deep::ExprCarrier::StructuralList(elements) => {
            for element in elements {
                report_in_post_order(element, span, violations, reported, errors);
            }
        }
        deep::ExprCarrier::MetadataExpression(metadata_expr) => {
            report_in_post_order(&metadata_expr.expr, span, violations, reported, errors);
        }
        deep::ExprCarrier::Atom(_) | deep::ExprCarrier::MetadataMap(_) => {}
    }
    let Some(found) = violations.get(&expr_key(expr)) else {
        return;
    };
    if !reported.insert(found.ty.clone()) {
        return;
    }
    let mut error = CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!(
            "tensor type `{}` is not representable: {} (spec/04-type-system.md section 4.7: \
             a violation proven from literals is a type error)",
            found.ty, found.reason
        ),
        vec![
            "Every extent of this type is a literal, so the size can never be allocated or \
             described; reduce an extent, or compute it at run time to get the operation's \
             `Overflow` trap instead"
                .to_string(),
        ],
    );
    match span {
        Some(id) => {
            if let Some(start) = parse_span_offset(id) {
                error = error.at_offset(start);
            }
            error = error.with_span_id(id.to_string());
        }
        None => error = error.at_offset(expr.span().offset),
    }
    errors.push(error);
}

struct Unrepresentable {
    ty: String,
    reason: String,
}

/// The first tensor type within `ty` whose literal extents give an element
/// count, suffix stride, or byte size outside i64.
fn unrepresentable_tensor(ty: &Type) -> Option<Unrepresentable> {
    match ty {
        Type::Tensor(dims, TensorPrec::Concrete(prim)) => {
            let extents = dims
                .iter()
                .map(|dim| match dim {
                    Dim::Lit(extent) => Some(*extent),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let dtype = prim.runtime_dtype().ok()?;
            let Err(MetadataError::Overflow(what)) = ShapeMetadata::contiguous(&extents, dtype)
            else {
                return None;
            };
            Some(Unrepresentable {
                ty: ty.to_string(),
                reason: overflow_reason(what, &extents, *prim, dtype.byte_width()),
            })
        }
        Type::Tensor(_, TensorPrec::Var(_)) => None,
        Type::Fn(args, ret) => args
            .iter()
            .chain(std::iter::once(ret.as_ref()))
            .find_map(unrepresentable_tensor),
        Type::Ref(inner) => unrepresentable_tensor(inner),
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().find_map(unrepresentable_tensor),
        Type::KindedAdt(_, args) => args.iter().find_map(|argument| match argument {
            NominalArg::Type(ty) => unrepresentable_tensor(ty),
            NominalArg::Dimension(_) => None,
        }),
        Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => None,
    }
}

/// Name the overflowing quantity and its exact value.
fn overflow_reason(what: &str, extents: &[i64], prim: Prim, width: usize) -> String {
    let count = extents.iter().try_fold(1_u128, |product, &extent| {
        u128::try_from(extent)
            .ok()
            .and_then(|extent| product.checked_mul(extent))
    });
    let exact = |value: Option<u128>| {
        value.map_or_else(|| "beyond 2^128".to_string(), |value| value.to_string())
    };
    match what {
        "extent product exceeds i64" => {
            format!("its element count {} exceeds i64", exact(count))
        }
        "byte size exceeds i64" => {
            let width = u128::try_from(width).ok();
            let bytes = count
                .zip(width)
                .and_then(|(count, width)| count.checked_mul(width));
            format!(
                "its byte size {} ({} `{}` elements) exceeds i64",
                exact(bytes),
                exact(count),
                prim.name()
            )
        }
        "stride product exceeds i64" => {
            "a suffix stride, the product of its trailing extents, exceeds i64".to_string()
        }
        other => format!("its metadata overflows: {other}"),
    }
}
