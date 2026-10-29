//! Native context admission and format-14 cache transport.

use chelis_checker_core::ContextSnapshot;
use chelis_checker_core::ErrorWitness;
use std::sync::Arc;

use chelis_deep::ast as deep;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryProofId {
    digest: [u8; 32],
    selector_context_digest: [u8; 32],
    rank_context_digest: [u8; 32],
}

impl LibraryProofId {
    pub(crate) fn for_library(
        annotated_exprs: &[deep::Expr],
        context: Option<Self>,
        selector_context_digest: [u8; 32],
        rank_context_digest: [u8; 32],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"chelis-library-proof-v2");
        if let Some(context) = context {
            hasher.update([1]);
            hasher.update(context.digest);
        } else {
            hasher.update([0]);
        }
        let canonical = chelis_deep::printer::print_canonical_flat(annotated_exprs);
        hasher.update((canonical.len() as u64).to_le_bytes());
        hasher.update(canonical.as_bytes());
        hasher.update(selector_context_digest);
        hasher.update(rank_context_digest);
        Self {
            digest: hasher.finalize().into(),
            selector_context_digest,
            rank_context_digest,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeEnv {
    inner: Arc<ContextSnapshot>,
    library_proof_id: Option<LibraryProofId>,
}

const TYPE_ENV_FORMAT_VERSION: u32 = 14;

#[derive(Serialize)]
struct TypeEnvWireRef<'a> {
    format_version: u32,
    inner: &'a Arc<ContextSnapshot>,
    library_proof_id: Option<LibraryProofId>,
}

#[derive(Deserialize)]
struct TypeEnvWire {
    format_version: u32,
    inner: Arc<ContextSnapshot>,
    library_proof_id: Option<LibraryProofId>,
}

impl Serialize for TypeEnv {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TypeEnvWireRef {
            format_version: TYPE_ENV_FORMAT_VERSION,
            inner: &self.inner,
            library_proof_id: self.library_proof_id,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TypeEnv {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TypeEnvWire::deserialize(deserializer)?;
        if wire.format_version != TYPE_ENV_FORMAT_VERSION {
            return Err(serde::de::Error::custom(
                "obsolete TypeEnv format; regenerate the checker snapshot",
            ));
        }
        wire.inner.validate().map_err(serde::de::Error::custom)?;
        let valid_rank_proof = match wire.library_proof_id {
            Some(proof) => proof.rank_context_digest == wire.inner.rank_context_digest(),
            None => !wire.inner.has_rank_bodies(),
        };
        if !valid_rank_proof {
            return Err(serde::de::Error::custom(
                "rank body evidence does not match its library proof",
            ));
        }
        Ok(Self {
            inner: wire.inner,
            library_proof_id: wire.library_proof_id,
        })
    }
}

impl TypeEnv {
    #[cfg(test)]
    pub(crate) fn with_test_binding(
        &self,
        name: &str,
        scheme: chelis_type_core::Scheme<ErrorWitness>,
    ) -> Self {
        let mut inner = (*self.inner).clone();
        inner.env.bind(name.to_string(), scheme);
        Self {
            inner: Arc::new(inner),
            library_proof_id: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            inner: Arc::new(ContextSnapshot::empty()),
            library_proof_id: None,
        }
    }

    pub(crate) fn from_inner(inner: ContextSnapshot) -> Self {
        Self {
            inner: Arc::new(inner),
            library_proof_id: None,
        }
    }

    pub(crate) fn bind_library_proof(&mut self, proof_id: LibraryProofId) {
        self.library_proof_id = Some(proof_id);
    }

    pub(crate) fn library_proof_id(&self) -> Option<LibraryProofId> {
        self.library_proof_id
    }

    pub(crate) fn inner(&self) -> &ContextSnapshot {
        &self.inner
    }

    pub fn library_def_count(&self) -> usize {
        self.inner.library_def_names.len()
    }

    pub fn has_library_def(&self, name: &str) -> bool {
        self.inner.library_def_names.contains(name)
    }

    pub fn scheme(&self, name: &str) -> Option<&chelis_type_core::Scheme<ErrorWitness>> {
        self.inner.env.lookup(name)
    }

    pub fn matches_checked_program(&self, program: &crate::CheckedProgram) -> bool {
        let selector_context_digest = self.inner.selector_context_digest();
        self.library_proof_id.is_some_and(|proof| {
            proof.selector_context_digest == selector_context_digest
                && proof.rank_context_digest == self.inner.rank_context_digest()
        }) && self.inner.validate_rank_bodies().is_ok()
            && self.library_proof_id == program.library_proof_id()
            && self.inner.ir_types.eq(program.type_env())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_numeric_core::prim::Prim;
    use chelis_type_core::unify;
    use chelis_type_core::{Dim, Scheme, TensorPrec, Type, TypeVarRestriction};

    #[test]
    fn result_origin_is_mandatory_in_schemes_and_contexts() {
        let scheme = Scheme::<ErrorWitness>::mono(Type::Prim(Prim::F32));
        let mut wire = serde_json::to_value(&scheme).unwrap();
        wire.as_object_mut().unwrap().remove("result_origin");
        assert!(serde_json::from_value::<Scheme<ErrorWitness>>(wire).is_err());
        let mut wire = serde_json::to_value(TypeEnv::empty()).unwrap();
        for version in [6, 7, 8, 9, 10, 11, 12, 13] {
            wire["format_version"] = serde_json::json!(version);
            assert!(serde_json::from_value::<TypeEnv>(wire.clone()).is_err());
        }
        let wire = bincode::serialize(&TypeEnv::empty()).unwrap();
        assert!(bincode::deserialize::<TypeEnv>(&wire).is_ok());
    }

    #[test]
    fn dimension_snapshot_rejects_obsolete_direct_encoding_and_missing_summary() {
        let context = TypeEnv::empty();

        let old = bincode::serialize(&(&context.inner, context.library_proof_id)).unwrap();
        assert!(bincode::deserialize::<TypeEnv>(&old).is_err());
        let mut json = serde_json::to_value(&context).unwrap();
        json["format_version"] = serde_json::json!(1);
        assert!(serde_json::from_value::<TypeEnv>(json).is_err());
        let mut json = serde_json::to_value(&context).unwrap();
        json["inner"]["subst"]
            .as_object_mut()
            .unwrap()
            .remove("dimension_labels");
        assert!(serde_json::from_value::<TypeEnv>(json).is_err());
    }

    #[test]
    fn dimension_snapshot_rejects_unallocated_and_invalid_labels() {
        for label in ["fixed", "_", "", "two words"] {
            let mut inner = TypeEnv::empty().inner().clone();
            let v = if label == "fixed" {
                chelis_type_core::DimVar(u32::MAX)
            } else {
                inner.var_gen.fresh_dvar()
            };
            inner.subst.protect_dimensions([v]);
            chelis_type_core::unify_dim(&Dim::Var(v), &Dim::Name(label.into()), &mut inner.subst)
                .unwrap();
            let encoded = bincode::serialize(&TypeEnv::from_inner(inner)).unwrap();
            assert!(
                bincode::deserialize::<TypeEnv>(&encoded).is_err(),
                "{label:?}"
            );
        }
    }

    #[test]
    fn serialized_level_state_resumes_old_ids_at_zero_and_compacts_history() {
        let empty = TypeEnv::empty();
        let mut inner = empty.inner().clone();
        let older = inner.var_gen.fresh_tvar();
        inner
            .env
            .bind("outer".to_string(), Scheme::mono(Type::Var(older)));

        let level = inner.subst.enter_level(&inner.var_gen);
        let lowered_tvar = inner.var_gen.fresh_tvar();
        let lowered_dvar = inner.var_gen.fresh_dvar();
        let lowered_rvar = inner.var_gen.fresh_rvar();
        let child_tvar = inner.var_gen.fresh_tvar();
        let composite = Type::tuple(vec![
            Type::Var(lowered_tvar),
            Type::Tensor(
                vec![Dim::Var(lowered_dvar), Dim::Rank(lowered_rvar)],
                TensorPrec::Var(lowered_tvar),
            ),
        ]);
        unify(&Type::Var(older), &composite, &mut inner.subst).expect("escape binding");
        inner.subst.leave_level(level, &inner.var_gen);
        inner
            .env
            .bind("child".to_string(), Scheme::mono(Type::Var(child_tvar)));

        let context = TypeEnv::from_inner(inner);
        let encoded = bincode::serialize(&context).expect("TypeEnv serializes level state");
        let decoded: TypeEnv =
            bincode::deserialize(&encoded).expect("TypeEnv deserializes level state");
        assert_eq!(decoded.inner().subst.level_of_tvar(lowered_tvar), 0);
        assert_eq!(decoded.inner().subst.level_of_dvar(lowered_dvar), 0);
        assert_eq!(decoded.inner().subst.level_of_rvar(lowered_rvar), 0);
        assert_eq!(decoded.inner().subst.level_of_tvar(child_tvar), 1);
        assert_eq!(decoded.inner().subst.level_metadata_counts(), (2, 1, 1, 1));

        let mut resumed = decoded.inner().resume_for_new_check();
        assert_eq!(resumed.subst.level_of_tvar(child_tvar), 0);
        assert_eq!(resumed.subst.level_metadata_counts(), (1, 0, 0, 0));
        let imported = resumed.env.generalize(
            &Type::Var(child_tvar),
            &resumed.subst,
            &chelis_checker_core::Resources::new(&chelis_syntax_host::NativeStack),
        );
        assert!(imported.tvars.is_empty());

        for cycle in 0..4 {
            let level = resumed.subst.enter_level(&resumed.var_gen);
            let fresh = resumed.var_gen.fresh_tvar();
            resumed.subst.leave_level(level, &resumed.var_gen);
            resumed
                .env
                .bind(format!("cycle_{cycle}"), Scheme::mono(Type::Var(fresh)));

            let context = TypeEnv::from_inner(resumed);
            let encoded = bincode::serialize(&context).expect("cycle serializes");
            let decoded: TypeEnv = bincode::deserialize(&encoded).expect("cycle deserializes");
            resumed = decoded.inner().resume_for_new_check();
            assert_eq!(resumed.subst.level_of_tvar(fresh), 0);
            assert_eq!(resumed.subst.level_metadata_counts(), (1, 0, 0, 0));
            let imported = resumed.env.generalize(
                &Type::Var(fresh),
                &resumed.subst,
                &chelis_checker_core::Resources::new(&chelis_syntax_host::NativeStack),
            );
            assert!(imported.tvars.is_empty());
        }
    }

    #[test]
    fn operation_value_restrictions_survive_context_round_trips() {
        let encoded = bincode::serialize(&TypeEnv::empty()).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&encoded).unwrap();
        for (name, restriction, accepted, rejected) in [
            (
                "mean",
                TypeVarRestriction::FloatValue,
                Prim::F32,
                Prim::Int32,
            ),
            (
                "trunc_div",
                TypeVarRestriction::IntValue,
                Prim::Int32,
                Prim::F32,
            ),
            (
                "add",
                TypeVarRestriction::NumericValue,
                Prim::Int32,
                Prim::Bool,
            ),
        ] {
            let mut resumed = decoded.inner().resume_for_new_check();
            let scheme = resumed.env.lookup(name).unwrap().clone();
            assert!(
                scheme
                    .tvar_restrictions
                    .iter()
                    .any(|(_, bound)| *bound == restriction)
            );
            let Type::Fn(params, _) =
                chelis_type_core::instantiate(&scheme, &mut resumed.var_gen, &resumed.subst)
            else {
                panic!("operation remains callable");
            };
            let precision = resumed.var_gen.fresh_tvar();
            let tensor = Type::Tensor(vec![Dim::Lit(3)], TensorPrec::Var(precision));
            let operand = if matches!(params[0], Type::Ref(_)) {
                Type::reference(tensor)
            } else {
                tensor
            };
            unify(&params[0], &operand, &mut resumed.subst)
                .expect("a value restriction must admit tensor operands");
            let mut valid = resumed.subst.clone();
            unify(&Type::Var(precision), &Type::Prim(accepted), &mut valid).unwrap();
            let error = unify(
                &Type::Var(precision),
                &Type::Prim(rejected),
                &mut resumed.subst,
            )
            .expect_err("late dtype binding must enforce the deserialized family");
            assert!(matches!(
                error.kind,
                chelis_type_core::TypeErrorKind::DtypeFamilyMismatch
            ));
        }
    }

    #[test]
    fn type_variable_restrictions_survive_context_round_trips() {
        let empty = TypeEnv::empty();
        let mut inner = empty.inner().clone();
        let live = inner.var_gen.fresh_tvar();
        inner
            .subst
            .narrow_tvar_restriction(live, TypeVarRestriction::ActiveFloat)
            .expect("an unbounded variable accepts any single dtype family");

        let context = TypeEnv::from_inner(inner);
        let encoded = bincode::serialize(&context).expect("restricted TypeEnv serializes");
        let decoded: TypeEnv =
            bincode::deserialize(&encoded).expect("restricted TypeEnv deserializes");
        assert_eq!(
            decoded.inner().subst.tvar_restriction(live),
            Some(TypeVarRestriction::ActiveFloat)
        );

        let mut resumed = decoded.inner().resume_for_new_check();
        let scheme = resumed
            .env
            .lookup("test_assert_close_tensor")
            .expect("builtin scheme survives round trip")
            .clone();
        assert_eq!(
            scheme.tvar_restrictions,
            vec![(scheme.tvars[0], TypeVarRestriction::ActiveFloat)]
        );
        let instantiated =
            chelis_type_core::instantiate(&scheme, &mut resumed.var_gen, &resumed.subst);
        let Type::Fn(params, _) = instantiated else {
            panic!("assert-close scheme must remain callable");
        };
        let Type::Var(precision) = params[2] else {
            panic!("tolerance must use the quantified precision variable");
        };
        let mut integer_trial = resumed.subst.clone();
        let error = unify(
            &Type::Var(precision),
            &Type::Prim(Prim::Int32),
            &mut integer_trial,
        )
        .expect_err("round-tripped restriction must reject i32");
        assert!(matches!(
            error.kind,
            chelis_type_core::TypeErrorKind::DtypeFamilyMismatch
        ));

        let mut float_trial = resumed.subst.clone();
        unify(
            &Type::Var(precision),
            &Type::Prim(Prim::F32),
            &mut float_trial,
        )
        .expect("round-tripped restriction must accept f32");
    }

    #[test]
    fn type_env_rejects_the_pre_checked_binding_occurrence_encoding() {
        let mut encoded = bincode::serialize(&TypeEnv::empty()).expect("encode current context");
        encoded[..4].copy_from_slice(&11_u32.to_le_bytes());
        assert!(
            bincode::deserialize::<TypeEnv>(&encoded).is_err(),
            "v11 embeds the old Deep Metadata encoding"
        );
    }

    #[test]
    fn type_env_rejects_the_pre_callable_provenance_version() {
        let mut encoded = bincode::serialize(&TypeEnv::empty()).expect("TypeEnv serializes");
        encoded[..4].copy_from_slice(&3_u32.to_le_bytes());
        let error = bincode::deserialize::<TypeEnv>(&encoded)
            .expect_err("TypeEnv v3 must not decode without callable provenance");
        assert!(
            error.to_string().contains("obsolete TypeEnv format"),
            "unexpected predecessor-version diagnostic: {error}"
        );
    }
}
