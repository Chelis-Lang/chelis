//! Exact compiler capacity identity (chelis#893 C2.1).

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use chelis_unord::UnordSet;
use chelis_vocab::Repr;
use num_bigint::BigUint;

use crate::axis_sources::AxisSource;
use crate::dag::{DimInfo, NodeId, RtAxis};
use crate::ownership::VerifiedDagView;

/// Opaque proof carrier for one tensor storage capacity.
///
/// The fields and constructors are deliberately private. A backend can ask
/// whether two keys are proven equal, but cannot invent an extent identity or
/// normalize a legacy carrier by itself.
///
/// ```compile_fail
/// use chelis_ir::capacity_key::CapacityKey;
/// fn forge() -> CapacityKey {
///     CapacityKey { canonical: todo!(), validity: todo!() }
/// }
/// ```
///
/// ```compile_fail
/// use chelis_ir::capacity_key::CapacityKey;
/// fn destructure(key: CapacityKey) {
///     match key { CapacityKey::Literal(_) => {} }
/// }
/// ```
///
/// ```compile_fail
/// use chelis_ir::capacity_key::CapacityKey;
/// fn narrow(key: CapacityKey) -> usize { key.into() }
/// ```
///
/// Raw equality would bypass validity-domain proof, so it is not implemented:
///
/// ```compile_fail
/// use chelis_ir::capacity_key::CapacityKey;
/// fn require_raw_equality<T: PartialEq>() {}
/// fn bypass() { require_raw_equality::<CapacityKey>(); }
/// ```
///
/// Hashing would provide the same bypass through a map or set key:
///
/// ```compile_fail
/// use chelis_ir::capacity_key::CapacityKey;
/// fn require_hash<T: std::hash::Hash>() {}
/// fn bypass() { require_hash::<CapacityKey>(); }
/// ```
#[derive(Clone, Debug)]
pub struct CapacityKey {
    canonical: CanonicalCapacity,
    validity: ValidityDomain,
}

/// Conservative result when exact capacity equality has not been proved.
///
/// This is not a code-generation failure. Storage planning responds by
/// allocating fresh storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotProvenEqual {
    reason: NotProvenReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NotProvenReason {
    DifferentCanonicalForm,
    PendingValidity,
    UnsupportedExpression,
}

impl fmt::Display for NotProvenEqual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason {
            NotProvenReason::DifferentCanonicalForm => {
                f.write_str("capacity keys have different canonical forms")
            }
            NotProvenReason::PendingValidity => {
                f.write_str("capacity equality has undischarged validity obligations")
            }
            NotProvenReason::UnsupportedExpression => {
                f.write_str("the typed extent expression has no exact capacity-key rule")
            }
        }
    }
}

impl std::error::Error for NotProvenEqual {}

impl CapacityKey {
    /// Prove that two total canonical capacity expressions are identical.
    ///
    /// A failed proof is the conservative allocation path, never permission
    /// to compare a projection or observe one successful runtime value.
    pub fn prove_equal(&self, other: &Self) -> Result<(), NotProvenEqual> {
        if !self.validity.is_total() || !other.validity.is_total() {
            return Err(NotProvenEqual {
                reason: NotProvenReason::PendingValidity,
            });
        }
        if self.canonical == other.canonical {
            Ok(())
        } else {
            Err(NotProvenEqual {
                reason: NotProvenReason::DifferentCanonicalForm,
            })
        }
    }

    /// Borrow the exact element count only for a total literal key.
    pub(crate) fn literal_elements(&self) -> Option<&BigUint> {
        if !self.validity.is_total() {
            return None;
        }
        match &self.canonical {
            CanonicalCapacity::Literal(value) => Some(value),
            CanonicalCapacity::Source(_)
            | CanonicalCapacity::Product(_)
            | CanonicalCapacity::ExactQuotient { .. } => None,
        }
    }

    /// Exact allocation bytes for a literal capacity and representation.
    ///
    /// The multiplication remains inside the private arbitrary-precision
    /// owner; callers receive only the checked final `u64` bound.
    pub(crate) fn literal_allocation_bytes(
        &self,
        repr: Repr,
    ) -> Result<Option<u64>, CapacityKeyBuildError> {
        let Some(elements) = self.literal_elements() else {
            return Ok(None);
        };
        let mut bytes = ExactLiteralProduct::one();
        bytes.include(elements.clone());
        bytes.include(BigUint::from(repr.byte_width()));
        u64::try_from(bytes.into_biguint())
            .map(Some)
            .map_err(|_| CapacityKeyBuildError::LiteralByteCountOverflow)
    }

    fn from_typed_expr(
        expression: CapacityExpr,
    ) -> Result<CapacityKeyBuild, CapacityKeyBuildError> {
        let Some(parts) = canonicalize(expression)? else {
            return Ok(CapacityKeyBuild::NotProven(NotProvenEqual {
                reason: NotProvenReason::UnsupportedExpression,
            }));
        };
        Ok(CapacityKeyBuild::Exact(Self {
            canonical: parts.canonical,
            validity: ValidityDomain::new(parts.obligations),
        }))
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum CanonicalCapacity {
    Literal(BigUint),
    Source(CapacitySource),
    Product(Box<[CanonicalCapacity]>),
    ExactQuotient {
        dividend: Box<CanonicalCapacity>,
        divisor: Box<CanonicalCapacity>,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum CapacitySource {
    ExternalAxis {
        scope: CapacityScope,
        load: NodeId,
        axis: usize,
    },
    ScalarValue {
        scope: CapacityScope,
        node: NodeId,
    },
    OpComputed {
        scope: CapacityScope,
        op: NodeId,
        axis: usize,
    },
}

/// Opaque identity for the one verified program in which a dynamic extent
/// source is meaningful. DAG-local [`NodeId`] values are never compared
/// without this scope.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct CapacityScope(u64);

static NEXT_CAPACITY_SCOPE: AtomicU64 = AtomicU64::new(1);

pub(crate) fn fresh_capacity_scope() -> CapacityScope {
    let scope = NEXT_CAPACITY_SCOPE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("capacity-scope identity space exhausted");
    CapacityScope(scope)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ValidityDomain {
    obligations: Box<[CapacityPredicate]>,
}

impl ValidityDomain {
    fn new(mut obligations: Vec<CapacityPredicate>) -> Self {
        obligations.sort();
        obligations.dedup();
        Self {
            obligations: obligations.into_boxed_slice(),
        }
    }

    fn is_total(&self) -> bool {
        self.obligations.is_empty()
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum CapacityPredicate {
    NonZero(CanonicalCapacity),
    DividesEvenly {
        dividend: CanonicalCapacity,
        divisor: CanonicalCapacity,
    },
}

#[derive(Clone, Debug)]
enum CapacityExpr {
    Literal(BigUint),
    Source(CapacitySource),
    Product(Vec<CapacityExpr>),
    ExactQuotient {
        dividend: Box<CapacityExpr>,
        divisor: Box<CapacityExpr>,
    },
    Unsupported,
}

#[derive(Clone, Debug)]
pub(crate) enum CapacityKeyBuild {
    Exact(CapacityKey),
    NotProven(NotProvenEqual),
}

impl CapacityKeyBuild {
    #[cfg(test)]
    fn into_key_for_test(self) -> CapacityKey {
        match self {
            Self::Exact(key) => key,
            Self::NotProven(reason) => panic!("expected a representable key: {reason}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum CapacityKeyBuildError {
    #[error("exact capacity quotient has a statically zero divisor")]
    StaticZeroDivisor,
    #[error("verified capacity source is malformed: {0}")]
    MalformedSource(String),
    #[error("exact literal allocation byte count does not fit u64")]
    LiteralByteCountOverflow,
}

struct CanonicalParts {
    canonical: CanonicalCapacity,
    obligations: Vec<CapacityPredicate>,
}

/// Exact arbitrary-precision product state.
///
/// Keeping the accumulator behind this type makes the accepted arithmetic
/// owner explicit: capacity canonicalization can contribute only a `BigUint`
/// factor, and only this method performs the product.
struct ExactLiteralProduct(BigUint);

impl ExactLiteralProduct {
    fn one() -> Self {
        Self(BigUint::from(1u8))
    }

    fn include(&mut self, factor: BigUint) {
        self.0 = <BigUint as std::ops::Mul<BigUint>>::mul(self.0.clone(), factor);
    }

    fn is_zero(&self) -> bool {
        self.0 == BigUint::from(0u8)
    }

    fn is_one(&self) -> bool {
        self.0 == BigUint::from(1u8)
    }

    fn into_biguint(self) -> BigUint {
        self.0
    }
}

fn canonicalize(expression: CapacityExpr) -> Result<Option<CanonicalParts>, CapacityKeyBuildError> {
    match expression {
        CapacityExpr::Literal(value) => Ok(Some(CanonicalParts {
            canonical: CanonicalCapacity::Literal(value),
            obligations: Vec::new(),
        })),
        CapacityExpr::Source(source) => Ok(Some(CanonicalParts {
            canonical: CanonicalCapacity::Source(source),
            obligations: Vec::new(),
        })),
        CapacityExpr::Unsupported => Ok(None),
        CapacityExpr::Product(factors) => canonicalize_product(factors),
        CapacityExpr::ExactQuotient { dividend, divisor } => {
            let Some(dividend) = canonicalize(*dividend)? else {
                return Ok(None);
            };
            let Some(divisor) = canonicalize(*divisor)? else {
                return Ok(None);
            };
            if matches!(
                &divisor.canonical,
                CanonicalCapacity::Literal(value) if value == &BigUint::from(0u8)
            ) {
                return Err(CapacityKeyBuildError::StaticZeroDivisor);
            }

            let mut obligations = dividend.obligations;
            obligations.extend(divisor.obligations);
            if let (
                CanonicalCapacity::Literal(dividend_value),
                CanonicalCapacity::Literal(divisor_value),
            ) = (&dividend.canonical, &divisor.canonical)
                && dividend_value % divisor_value == BigUint::from(0u8)
            {
                return Ok(Some(CanonicalParts {
                    canonical: CanonicalCapacity::Literal(dividend_value / divisor_value),
                    obligations,
                }));
            }

            if !matches!(&divisor.canonical, CanonicalCapacity::Literal(_)) {
                obligations.push(CapacityPredicate::NonZero(divisor.canonical.clone()));
            }
            obligations.push(CapacityPredicate::DividesEvenly {
                dividend: dividend.canonical.clone(),
                divisor: divisor.canonical.clone(),
            });
            Ok(Some(CanonicalParts {
                canonical: CanonicalCapacity::ExactQuotient {
                    dividend: Box::new(dividend.canonical),
                    divisor: Box::new(divisor.canonical),
                },
                obligations,
            }))
        }
    }
}

fn canonicalize_product(
    factors: Vec<CapacityExpr>,
) -> Result<Option<CanonicalParts>, CapacityKeyBuildError> {
    let mut literal = ExactLiteralProduct::one();
    let mut canonical_factors = Vec::new();
    let mut obligations = Vec::new();

    for factor in factors {
        let Some(parts) = canonicalize(factor)? else {
            return Ok(None);
        };
        obligations.extend(parts.obligations);
        push_product_factor(parts.canonical, &mut literal, &mut canonical_factors);
    }

    if literal.is_zero() && obligations.is_empty() {
        return Ok(Some(CanonicalParts {
            canonical: CanonicalCapacity::Literal(literal.into_biguint()),
            obligations,
        }));
    }
    if !literal.is_one() || canonical_factors.is_empty() {
        canonical_factors.push(CanonicalCapacity::Literal(literal.into_biguint()));
    }
    canonical_factors.sort();

    let canonical = match canonical_factors.len() {
        0 => CanonicalCapacity::Literal(BigUint::from(1u8)),
        1 => canonical_factors
            .pop()
            .expect("one canonical capacity factor"),
        _ => CanonicalCapacity::Product(canonical_factors.into_boxed_slice()),
    };
    Ok(Some(CanonicalParts {
        canonical,
        obligations,
    }))
}

fn push_product_factor(
    factor: CanonicalCapacity,
    literal: &mut ExactLiteralProduct,
    factors: &mut Vec<CanonicalCapacity>,
) {
    match factor {
        CanonicalCapacity::Literal(value) => {
            literal.include(value);
        }
        CanonicalCapacity::Product(nested) => {
            for factor in nested {
                push_product_factor(factor, literal, factors);
            }
        }
        other => factors.push(other),
    }
}

/// Build the exact storage-capacity key for one verified DAG node.
///
/// This remains crate-private so the shared ownership planner is the only
/// production consumer. Backend crates cannot recover equality from names or
/// from [`crate::dag::DimExpr`].
pub(crate) fn capacity_key_for_node(
    dag: VerifiedDagView<'_>,
    node: NodeId,
) -> Result<CapacityKeyBuild, CapacityKeyBuildError> {
    let output = dag.get(node).ok_or_else(|| {
        CapacityKeyBuildError::MalformedSource(format!("node {} is absent", node.0))
    })?;
    let mut factors = Vec::with_capacity(output.output_type.dims.len());
    let mut visiting = UnordSet::new();
    for axis in 0..output.output_type.dims.len() {
        factors.push(axis_capacity_expr(dag, node, axis, &mut visiting)?);
    }
    CapacityKey::from_typed_expr(CapacityExpr::Product(factors))
}

/// Build one exact proof carrier per output axis for shape equality.
pub(crate) fn shape_capacity_keys_for_node(
    dag: VerifiedDagView<'_>,
    node: NodeId,
) -> Result<Vec<CapacityKeyBuild>, CapacityKeyBuildError> {
    let output = dag.get(node).ok_or_else(|| {
        CapacityKeyBuildError::MalformedSource(format!("node {} is absent", node.0))
    })?;
    let mut keys = Vec::with_capacity(output.output_type.dims.len());
    let mut visiting = UnordSet::new();
    for axis in 0..output.output_type.dims.len() {
        keys.push(CapacityKey::from_typed_expr(axis_capacity_expr(
            dag,
            node,
            axis,
            &mut visiting,
        )?)?);
    }
    Ok(keys)
}

fn axis_capacity_expr(
    dag: VerifiedDagView<'_>,
    node: NodeId,
    axis: usize,
    visiting: &mut UnordSet<(NodeId, usize)>,
) -> Result<CapacityExpr, CapacityKeyBuildError> {
    if !visiting.insert((node, axis)) {
        return Err(CapacityKeyBuildError::MalformedSource(format!(
            "axis-source cycle at node {} axis {axis}",
            node.0
        )));
    }
    let result = axis_capacity_expr_inner(dag, node, axis, visiting);
    visiting.remove(&(node, axis));
    result
}

fn axis_capacity_expr_inner(
    dag: VerifiedDagView<'_>,
    node: NodeId,
    axis: usize,
    visiting: &mut UnordSet<(NodeId, usize)>,
) -> Result<CapacityExpr, CapacityKeyBuildError> {
    let output = dag.get(node).ok_or_else(|| {
        CapacityKeyBuildError::MalformedSource(format!("node {} is absent", node.0))
    })?;
    let dim = output.output_type.dims.get(axis).ok_or_else(|| {
        CapacityKeyBuildError::MalformedSource(format!("node {} has no output axis {axis}", node.0))
    })?;
    if let DimInfo::Lit(value) | DimInfo::Named(_, Some(value)) = dim {
        return Ok(CapacityExpr::Literal(BigUint::from(*value)));
    }

    let sources = dag.output_axis_sources(node);
    let scope = dag.capacity_scope();
    let source = sources.get(axis).ok_or_else(|| {
        CapacityKeyBuildError::MalformedSource(format!(
            "node {} has no checked source for output axis {axis}",
            node.0
        ))
    })?;
    match *source {
        AxisSource::Literal { value } => {
            let value = u64::try_from(value).map_err(|_| {
                CapacityKeyBuildError::MalformedSource(format!(
                    "node {} axis {axis} has negative literal extent {value}",
                    node.0
                ))
            })?;
            Ok(CapacityExpr::Literal(BigUint::from(value)))
        }
        AxisSource::ExternalAxis { load, axis } => {
            Ok(CapacityExpr::Source(CapacitySource::ExternalAxis {
                scope,
                load,
                axis,
            }))
        }
        AxisSource::InputAxis {
            input,
            axis: RtAxis::Lit(input_axis),
        } => {
            let input_node = output.inputs.get(input).copied().ok_or_else(|| {
                CapacityKeyBuildError::MalformedSource(format!(
                    "node {} axis {axis} refers to absent input slot {input}",
                    node.0
                ))
            })?;
            let input_axis = usize::try_from(input_axis).map_err(|_| {
                CapacityKeyBuildError::MalformedSource(format!(
                    "node {} axis {axis} refers to negative input axis {input_axis}",
                    node.0
                ))
            })?;
            axis_capacity_expr(dag, input_node, input_axis, visiting)
        }
        AxisSource::ScalarInput { input } => {
            let source = output.inputs.get(input).copied().ok_or_else(|| {
                CapacityKeyBuildError::MalformedSource(format!(
                    "node {} axis {axis} refers to absent scalar input slot {input}",
                    node.0
                ))
            })?;
            Ok(CapacityExpr::Source(CapacitySource::ScalarValue {
                scope,
                node: source,
            }))
        }
        // Both variants identify this verified program's local output axis.
        // `ClassSupplied` says the operation consumes its stamped extent
        // claim rather than computing a fresh extent, but the runtime class
        // carries no proof object that CapacityKey may use to identify two
        // independently supplied axes. Treating the originating axis as the
        // atom preserves exact identity through pass-through operations and
        // conservatively declines name- or guard-derived equality.
        AxisSource::OpComputed { op, axis } | AxisSource::ClassSupplied { op, axis } => {
            Ok(CapacityExpr::Source(CapacitySource::OpComputed {
                scope,
                op,
                axis,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use super::*;
    use crate::dag::{Dag, DimInfo, RiscOp, TensorType};
    use crate::ownership::{lower_dag_ownership, verify_ownership};
    use chelis_types::types::Prim;

    fn literal(value: u128) -> CapacityExpr {
        CapacityExpr::Literal(BigUint::from(value))
    }

    fn source(node: usize, axis: usize) -> CapacityExpr {
        CapacityExpr::Source(CapacitySource::OpComputed {
            scope: CapacityScope(0),
            op: crate::dag::NodeId(node),
            axis,
        })
    }

    fn product(factors: Vec<CapacityExpr>) -> CapacityExpr {
        CapacityExpr::Product(factors)
    }

    fn quotient(dividend: CapacityExpr, divisor: CapacityExpr) -> CapacityExpr {
        CapacityExpr::ExactQuotient {
            dividend: Box::new(dividend),
            divisor: Box::new(divisor),
        }
    }

    fn exact(expr: CapacityExpr) -> CapacityKey {
        match CapacityKey::from_typed_expr(expr).unwrap() {
            CapacityKeyBuild::Exact(key) => key,
            CapacityKeyBuild::NotProven(reason) => panic!("expected exact key: {reason:?}"),
        }
    }

    fn verified(dag: Dag) -> crate::ownership::VerifiedDagProgram {
        verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap()
    }

    #[test]
    fn capacity_key_products_use_arbitrary_precision_without_collision() {
        let smaller = exact(product(vec![source(0, 0), literal(1u128 << 80)]));
        let larger = exact(product(vec![source(0, 0), literal(1u128 << 81)]));
        assert_eq!(smaller.prove_equal(&smaller), Ok(()));
        assert!(smaller.prove_equal(&larger).is_err());
    }

    #[test]
    fn capacity_key_product_regions_flatten_sort_fold_and_remove_identity() {
        let lhs = exact(product(vec![
            source(2, 0),
            literal(3),
            product(vec![source(1, 0), literal(4), literal(1)]),
        ]));
        let rhs = exact(product(vec![literal(12), source(1, 0), source(2, 0)]));
        assert_eq!(lhs.prove_equal(&rhs), Ok(()));
    }

    #[test]
    fn exact_literal_quotient_folds_but_symbolic_quotient_is_not_a_proof() {
        let folded = exact(quotient(literal(12), literal(3)));
        assert_eq!(folded.literal_elements(), Some(&BigUint::from(4u8)));

        let partial = CapacityKey::from_typed_expr(quotient(source(1, 0), source(2, 0)))
            .unwrap()
            .into_key_for_test();
        assert!(partial.prove_equal(&partial).is_err());
        assert_eq!(partial.literal_elements(), None);
    }

    #[test]
    fn quotient_is_an_ordered_barrier_without_cancellation_or_reassociation() {
        let left = CapacityKey::from_typed_expr(quotient(
            product(vec![source(1, 0), literal(3)]),
            literal(2),
        ))
        .unwrap()
        .into_key_for_test();
        let right = CapacityKey::from_typed_expr(product(vec![
            source(1, 0),
            quotient(literal(3), literal(2)),
        ]))
        .unwrap()
        .into_key_for_test();
        assert!(left.prove_equal(&right).is_err());

        let self_quotient = CapacityKey::from_typed_expr(quotient(source(1, 0), source(1, 0)))
            .unwrap()
            .into_key_for_test();
        assert!(self_quotient.prove_equal(&exact(literal(1))).is_err());
    }

    #[test]
    fn zero_does_not_discard_partial_quotient_validity_in_either_order() {
        let partial = quotient(literal(1), source(1, 0));
        for expr in [
            product(vec![literal(0), partial.clone()]),
            product(vec![partial.clone(), literal(0)]),
        ] {
            let key = CapacityKey::from_typed_expr(expr)
                .unwrap()
                .into_key_for_test();
            assert!(key.prove_equal(&exact(literal(0))).is_err());
            assert!(key.prove_equal(&key).is_err());
        }
    }

    #[test]
    fn static_zero_divisor_is_rejected_and_unsupported_is_conservative() {
        assert!(matches!(
            CapacityKey::from_typed_expr(quotient(literal(1), literal(0))),
            Err(CapacityKeyBuildError::StaticZeroDivisor)
        ));
        assert!(matches!(
            CapacityKey::from_typed_expr(CapacityExpr::Unsupported).unwrap(),
            CapacityKeyBuild::NotProven(_)
        ));
    }

    #[test]
    fn exact_source_identity_ignores_same_spelled_dimension_names() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let left = dag.add_node(
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let right = dag.add_node(
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            ty,
            None,
        );

        let dag = verified(dag);
        let left = capacity_key_for_node(dag.emission(), left)
            .unwrap()
            .into_key_for_test();
        let right = capacity_key_for_node(dag.emission(), right)
            .unwrap()
            .into_key_for_test();
        assert!(left.prove_equal(&right).is_err());
    }

    #[test]
    fn exact_source_identity_is_scoped_to_one_verified_program() {
        fn one_load(name: &str) -> (crate::ownership::VerifiedDagProgram, NodeId) {
            let mut dag = Dag::new();
            let load = dag.add_node(
                RiscOp::Load { name: name.into() },
                vec![],
                TensorType {
                    dims: vec![DimInfo::Named("n".into(), None)],
                    precision: Prim::F32,
                },
                None,
            );
            (verified(dag), load)
        }

        let (left_program, left_node) = one_load("left");
        let (right_program, right_node) = one_load("right");
        let left = capacity_key_for_node(left_program.emission(), left_node)
            .unwrap()
            .into_key_for_test();
        let right = capacity_key_for_node(right_program.emission(), right_node)
            .unwrap()
            .into_key_for_test();

        assert!(left.prove_equal(&right).is_err());
    }

    #[test]
    fn pass_through_axis_retains_the_exact_source_identity() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let input = dag.add_node(
            RiscOp::Load {
                name: "input".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let relu = dag.add_node(RiscOp::Relu, vec![input], ty, None);

        let dag = verified(dag);
        let input = capacity_key_for_node(dag.emission(), input)
            .unwrap()
            .into_key_for_test();
        let relu = capacity_key_for_node(dag.emission(), relu)
            .unwrap()
            .into_key_for_test();
        assert_eq!(input.prove_equal(&relu), Ok(()));
    }

    #[test]
    fn class_supplied_extent_retains_its_origin_through_pass_through() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let supplied = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let relu = dag.add_node(RiscOp::Relu, vec![supplied], ty, None);

        let dag = verified(dag);
        let supplied = capacity_key_for_node(dag.emission(), supplied)
            .unwrap()
            .into_key_for_test();
        let relu = capacity_key_for_node(dag.emission(), relu)
            .unwrap()
            .into_key_for_test();
        assert_eq!(supplied.prove_equal(&relu), Ok(()));
    }

    #[test]
    fn same_spelled_class_supplied_extents_do_not_bypass_source_identity() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let left = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let right = dag.add_node(RiscOp::synth_const(Prim::F32, 1.0), vec![], ty, None);

        let dag = verified(dag);
        let left = capacity_key_for_node(dag.emission(), left)
            .unwrap()
            .into_key_for_test();
        let right = capacity_key_for_node(dag.emission(), right)
            .unwrap()
            .into_key_for_test();
        assert!(left.prove_equal(&right).is_err());
    }

    #[test]
    fn compile_surface_remains_opaque() {
        let key = exact(literal(7));
        assert_eq!(key.literal_elements(), Some(&BigUint::from(7u8)));
    }
}
