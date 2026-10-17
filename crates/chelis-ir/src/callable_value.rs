//! Static builtin values retain identity and every independent contract frame.

use chelis_deep::{DeepTag, Expr};
use std::cell::Cell;
use std::sync::Arc;

thread_local! {
    /// Authored contracts normalized on this thread since the last reset
    /// (chelis#3741). Counted at the normalization, never estimated.
    static CONTRACT_NORMALIZATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Authored contracts [`NormalizedCallableContract::from_authored`] normalized
/// on this thread. A runtime callable carries its contracts normalized, so
/// applying it again must not add to this count.
pub fn contract_normalizations() -> u64 {
    CONTRACT_NORMALIZATIONS.with(Cell::get)
}

/// Reset [`contract_normalizations`] for this thread.
pub fn reset_contract_normalizations() {
    CONTRACT_NORMALIZATIONS.with(|normalizations| normalizations.set(0));
}

/// Semantic invocation frames expose checked syntax, never numeric payloads.
/// Runtime owners may retain additional opaque defining-scope evidence.
pub trait CallableContractFrame: Clone {
    fn signature_syntax(&self) -> &Expr;
    fn authored_syntax(&self) -> &Expr;
}

impl CallableContractFrame for Expr {
    fn signature_syntax(&self) -> &Expr {
        self
    }
    fn authored_syntax(&self) -> &Expr {
        self
    }
}

/// A resolved static operation and its ordered, opaque invocation frames.
/// Only bare producer formation starts an empty chain; aliases transport this
/// value, never reconstruct one from its logical operation descriptor.
/// ```compile_fail
/// use chelis_ir::callable_value::ResolvedBuiltinValue;
/// let _numeric_channel = ResolvedBuiltinValue::<f64>::resolve_bare("neg");
/// ```
#[derive(Clone, Debug)]
pub struct ResolvedBuiltinValue<C: CallableContractFrame> {
    operation: chelis_types::AdmittedBuiltin,
    contracts: Arc<Vec<C>>,
}

impl<C: CallableContractFrame> ResolvedBuiltinValue<C> {
    /// Bare builtin producer formation, after lexical/declaration resolution.
    /// This is not an adapter or a conversion from a callable descriptor.
    pub fn resolve_bare(name: &str) -> Option<Self> {
        chelis_types::AdmittedBuiltin::resolve(name).map(|operation| Self {
            operation,
            contracts: Arc::new(Vec::new()),
        })
    }

    /// A closed join can retain one immutable authority only when both
    /// alternatives already share it. Bare producers have no frame authority.
    pub fn try_join(&self, other: &Self) -> Option<Self> {
        (self.operation == other.operation
            && ((self.contracts.is_empty() && other.contracts.is_empty())
                || Arc::ptr_eq(&self.contracts, &other.contracts)))
        .then(|| self.clone())
    }

    pub fn operation(&self) -> chelis_types::AdmittedBuiltin {
        self.operation
    }
    pub fn contracts(&self) -> &[C] {
        &self.contracts
    }
    pub fn symbol(&self) -> &'static str {
        self.operation.symbol()
    }
    pub fn scheme(&self) -> &chelis_types::types::Scheme {
        self.operation.scheme()
    }
    pub fn declaration(&self) -> &chelis_types::BuiltinDecl {
        self.operation.declaration()
    }
    /// Conversion preserves every frame and its order, or returns a failure.
    pub fn try_map_contracts<D: CallableContractFrame, E>(
        &self,
        convert: impl FnMut(&C) -> Result<D, E>,
    ) -> Result<ResolvedBuiltinValue<D>, E> {
        Ok(ResolvedBuiltinValue {
            operation: self.operation,
            contracts: Arc::new(
                self.contracts
                    .iter()
                    .map(convert)
                    .collect::<Result<_, _>>()?,
            ),
        })
    }
    /// Consume one invocation frame without reconstructing a value from its
    /// operation identity. The remaining opaque authorities keep their order.
    pub(crate) fn split_outer_contract(&self) -> Option<(C, Self)> {
        let (outer, rest) = self.contracts.split_first()?;
        Some((
            outer.clone(),
            Self {
                operation: self.operation,
                contracts: Arc::new(rest.to_vec()),
            },
        ))
    }

    pub(crate) fn checked_identity(&self) -> crate::host_type_state::CheckedBuiltinCallable {
        crate::host_type_state::CheckedBuiltinCallable::from_symbol(self.symbol())
            .expect("admitted operation has a closed checked discriminator")
    }
}

impl<C: CallableContractFrame> ResolvedBuiltinValue<C> {
    pub fn retain_outer_contract(&mut self, contract: C) {
        Arc::make_mut(&mut self.contracts).insert(0, contract);
    }
    pub fn with_outer_contract(mut self, contract: C) -> Self {
        self.retain_outer_contract(contract);
        self
    }
}

/// A whole registry-normalized authored function signature and its origin.
/// Entry and result members are retained even when the entry emits no guard.
#[derive(Clone, Debug)]
pub struct NormalizedCallableContract {
    authored: Expr,
    signature: Expr,
}

impl NormalizedCallableContract {
    pub fn from_authored(
        registry: &chelis_types::adt::AdtRegistry,
        authored: &Expr,
    ) -> Result<Option<Self>, String> {
        CONTRACT_NORMALIZATIONS.with(|normalizations| normalizations.set(normalizations.get() + 1));
        let normalized = crate::host::normalize_authored_entry_type(registry, authored)?;
        let mut signature = &normalized;
        loop {
            match signature {
                Expr::MetaExpr(meta, _) => signature = &meta.expr,
                Expr::Node(node, _) if node.tag() == DeepTag::TRef => {
                    signature = node.children_slice().first().ok_or_else(|| {
                        "callable reference type lacks its checked inner signature".to_owned()
                    })?;
                }
                _ => break,
            }
        }
        if signature.tag() != Some(DeepTag::TFn) {
            return Ok(None);
        }
        if let Expr::Node(node, _) = signature
            && node.children_slice().is_empty()
        {
            return Err("callable signature lacks its result member".into());
        }
        Ok(Some(Self {
            authored: authored.clone(),
            signature: signature.clone(),
        }))
    }
    /// The retained invocation owner actualizes executable binder metadata;
    /// the original authored declaration remains this frame's authority.
    pub(crate) fn with_executable_signature(&self, signature: Expr) -> Result<Self, String> {
        if signature.tag() != Some(DeepTag::TFn)
            || !matches!(&signature, Expr::Node(node, _) if !node.children_slice().is_empty())
        {
            return Err("actualized callable contract lacks its whole function signature".into());
        }
        Ok(Self {
            authored: self.authored.clone(),
            signature,
        })
    }

    pub fn signature(&self) -> &Expr {
        &self.signature
    }
    pub fn authored(&self) -> &Expr {
        &self.authored
    }
}

impl CallableContractFrame for NormalizedCallableContract {
    fn signature_syntax(&self) -> &Expr {
        self.signature()
    }
    fn authored_syntax(&self) -> &Expr {
        self.authored()
    }
}

pub(crate) type StaticBuiltinValue = ResolvedBuiltinValue<NormalizedCallableContract>;

#[cfg(test)]
mod tests {
    use super::*;

    fn function_contract(result: &str) -> Expr {
        chelis_deep::parser::parse_str(&format!(
            "(t-fn {{}} (t-ref {{}} (t-tensor {{}} (d-name {{}} *) (t-prim {{}} f64))) {result})"
        ))
        .unwrap()
        .remove(0)
    }

    #[test]
    fn bare_producer_admission_preserves_canonical_identity_and_whole_scheme() {
        let value = ResolvedBuiltinValue::<Expr>::resolve_bare("abs").unwrap();
        assert_eq!(
            value.operation(),
            chelis_types::AdmittedBuiltin::resolve("abs").unwrap()
        );
        assert!(value.contracts().is_empty());
        assert!(ResolvedBuiltinValue::<Expr>::resolve_bare("print").is_none());
        assert!(ResolvedBuiltinValue::<Expr>::resolve_bare("not_a_builtin").is_none());
    }

    #[test]
    fn closed_join_retains_only_one_shared_contract_authority() {
        let signature = function_contract("(t-tensor {} (d-lit {} 2) (t-prim {} bool))");
        let first = ResolvedBuiltinValue::resolve_bare("is_finite")
            .unwrap()
            .with_outer_contract(signature.clone());
        let copied = first.clone();
        let joined = first
            .try_join(&copied)
            .expect("Copy preserves the whole authority");
        assert_eq!(joined.operation(), first.operation());
        assert_eq!(joined.contracts().len(), 1);
        assert!(Arc::ptr_eq(&joined.contracts, &first.contracts));
        let independently_adapted = ResolvedBuiltinValue::resolve_bare("is_finite")
            .unwrap()
            .with_outer_contract(signature);
        assert!(
            first.try_join(&independently_adapted).is_none(),
            "equal syntax does not select one of two activated authorities"
        );
        assert!(
            first
                .try_join(&ResolvedBuiltinValue::resolve_bare("is_nan").unwrap())
                .is_none()
        );
    }

    #[test]
    fn closed_join_accepts_same_bare_operation_and_rejects_distinct_operations() {
        let first = ResolvedBuiltinValue::<Expr>::resolve_bare("is_finite").unwrap();
        let second = ResolvedBuiltinValue::<Expr>::resolve_bare("is_finite").unwrap();
        assert!(first.try_join(&second).unwrap().contracts().is_empty());
        let different = ResolvedBuiltinValue::<Expr>::resolve_bare("is_nan").unwrap();
        assert!(first.try_join(&different).is_none());
    }

    #[test]
    fn adaptation_copy_and_opaque_conversion_preserve_every_frame_in_order() {
        #[derive(Clone)]
        struct OpaqueFrame {
            label: String,
            signature: Expr,
        }
        impl CallableContractFrame for OpaqueFrame {
            fn signature_syntax(&self) -> &Expr {
                &self.signature
            }
            fn authored_syntax(&self) -> &Expr {
                &self.signature
            }
        }
        let signature = function_contract("(t-tensor {} (d-lit {} 2) (t-prim {} bool))");
        let source = ResolvedBuiltinValue::resolve_bare("neg")
            .unwrap()
            .with_outer_contract(OpaqueFrame {
                label: "inner".into(),
                signature: signature.clone(),
            })
            .with_outer_contract(OpaqueFrame {
                label: "outer".into(),
                signature,
            });
        let (outer, remainder) = source.split_outer_contract().unwrap();
        assert_eq!(outer.label, "outer");
        assert_eq!(remainder.operation(), source.operation());
        let (inner, bare) = remainder.split_outer_contract().unwrap();
        assert_eq!(inner.label, "inner");
        assert_eq!(bare.operation(), source.operation());
        assert!(bare.split_outer_contract().is_none());
        assert_eq!(source.contracts().len(), 2);
        let copied = source.clone();
        assert_eq!(copied.operation(), source.operation());
        assert_eq!(
            copied
                .contracts()
                .iter()
                .map(|frame| frame.label.as_str())
                .collect::<Vec<_>>(),
            ["outer", "inner"]
        );
        let converted = copied
            .try_map_contracts(|frame| {
                Ok::<_, String>(OpaqueFrame {
                    label: format!("mapped_{}", frame.label),
                    signature: frame.signature.clone(),
                })
            })
            .unwrap();
        assert_eq!(converted.operation(), source.operation());
        assert_eq!(
            converted
                .contracts()
                .iter()
                .map(|frame| frame.label.as_str())
                .collect::<Vec<_>>(),
            ["mapped_outer", "mapped_inner"]
        );
        assert_eq!(
            source
                .contracts()
                .iter()
                .map(|frame| frame.label.as_str())
                .collect::<Vec<_>>(),
            ["outer", "inner"]
        );
        assert!(
            source
                .try_map_contracts(|_| Err::<OpaqueFrame, _>("unsupported frame"))
                .is_err()
        );
    }

    #[test]
    fn normalized_result_only_contract_keeps_full_signature_and_checked_origin() {
        let authored = function_contract("(t-tensor {} (d-lit {} 2) (t-prim {} bool))");
        let frame = NormalizedCallableContract::from_authored(
            &chelis_types::adt::AdtRegistry::default(),
            &authored,
        )
        .unwrap()
        .unwrap();
        assert_eq!(frame.signature(), &authored);
        assert_eq!(frame.authored(), &authored);
        let Expr::Node(signature, _) = frame.signature() else {
            panic!("normalized function");
        };
        let parts = signature.children_slice();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].tag(), Some(DeepTag::TRef));
        assert_eq!(parts[1].tag(), Some(DeepTag::TTensor));
        assert!(
            NormalizedCallableContract::from_authored(
                &chelis_types::adt::AdtRegistry::default(),
                &parts[1],
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn producer_cache_inherits_complete_alias_copy_and_generated_label_values() {
        use std::collections::BTreeMap;
        fn var(name: &str) -> Expr {
            chelis_deep::parser::parse_str(&format!("(var {{}} {name})"))
                .unwrap()
                .remove(0)
        }
        let inner = function_contract("(t-tensor {} (d-lit {} 2) (t-prim {} bool))");
        let outer = function_contract("(t-tensor {} (d-name {} *) (t-prim {} bool))");
        let definitions = BTreeMap::from([
            ("f".into(), var("is_finite")),
            ("g".into(), var("f")),
            (
                "h".into(),
                Expr::node(
                    DeepTag::Copy,
                    chelis_deep::Metadata::default(),
                    vec![var("g")],
                    inner.span(),
                ),
            ),
            ("wide".into(), var("h")),
        ]);
        let signatures =
            BTreeMap::from([("f".into(), inner.clone()), ("wide".into(), outer.clone())]);
        let values = crate::lower::resolved_builtin_producer_bindings(
            &definitions,
            &signatures,
            &chelis_types::adt::AdtRegistry::default(),
        )
        .unwrap();
        for name in ["f", "g", "h"] {
            assert_eq!(values[name].symbol(), "is_finite");
            assert_eq!(values[name].contracts().len(), 1);
            assert_eq!(values[name].contracts()[0].signature(), &inner);
            let label = crate::LoadStoreName::top_level(name);
            assert_eq!(values[label.as_str()].contracts()[0].signature(), &inner);
        }
        assert_eq!(values["wide"].contracts().len(), 2);
        assert_eq!(values["wide"].contracts()[0].signature(), &outer);
        assert_eq!(values["wide"].contracts()[1].signature(), &inner);
    }

    #[test]
    fn producer_cache_keeps_shadow_and_non_callable_signature_failures_explicit() {
        use std::collections::BTreeMap;
        let var = |name| {
            chelis_deep::parser::parse_str(&format!("(var {{}} {name})"))
                .unwrap()
                .remove(0)
        };
        let definitions = BTreeMap::from([("neg".into(), var("abs")), ("f".into(), var("neg"))]);
        let registry = chelis_types::adt::AdtRegistry::default();
        let values = crate::lower::resolved_builtin_producer_bindings(
            &definitions,
            &BTreeMap::new(),
            &registry,
        )
        .unwrap();
        assert_eq!(values["f"].symbol(), "abs");
        let malformed = chelis_deep::parser::parse_str("(t-prim {} bool)")
            .unwrap()
            .remove(0);
        let signatures = BTreeMap::from([("f".into(), malformed)]);
        assert!(
            crate::lower::resolved_builtin_producer_bindings(&definitions, &signatures, &registry)
                .is_err()
        );
    }

    #[test]
    fn nominal_callable_normalization_retains_borrowed_inputs_and_full_result() {
        use chelis_types::types::{
            Dim, DimVar, NominalArg, NominalParamKind, Prim, TensorPrec, Type,
        };
        let mut registry = chelis_types::adt::AdtRegistry::default();
        let parameter = DimVar(77);
        registry.aliases.insert(
            "Contract".into(),
            chelis_types::adt::TypeAliasDef {
                params: vec!["n".into()],
                param_kinds: vec![NominalParamKind::Dimension],
                param_vars: Vec::new(),
                param_args: vec![NominalArg::Dimension(Dim::Var(parameter))],
                body: Type::Fn(
                    vec![Type::Ref(Box::new(Type::Tensor(
                        vec![Dim::Wildcard],
                        TensorPrec::Concrete(Prim::F64),
                    )))],
                    Box::new(Type::Tensor(
                        vec![Dim::Var(parameter)],
                        TensorPrec::Concrete(Prim::Bool),
                    )),
                ),
            },
        );
        let authored = chelis_deep::parser::parse_str("(t-adt {} Contract (d-lit {} 2))")
            .unwrap()
            .remove(0);
        let frame = NormalizedCallableContract::from_authored(&registry, &authored)
            .unwrap()
            .unwrap();
        let Expr::Node(signature, _) = frame.signature() else {
            panic!("normalized function");
        };
        assert_eq!(signature.tag(), DeepTag::TFn);
        let parts = signature.children_slice();
        assert_eq!(parts[0].tag(), Some(DeepTag::TRef));
        let Expr::Node(result, _) = &parts[1] else {
            panic!("result tensor");
        };
        assert_eq!(result.tag(), DeepTag::TTensor);
        assert_eq!(result.children_slice()[0].tag(), Some(DeepTag::DLit));
        assert_eq!(frame.authored(), &authored);
        let malformed = chelis_deep::parser::parse_str("(t-adt {} Contract (t-prim {} f64))")
            .unwrap()
            .remove(0);
        assert!(NormalizedCallableContract::from_authored(&registry, &malformed).is_err());
    }
}
