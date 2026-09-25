//! Stacked type-checker context — Phase C of the compiled-artifact caching
//! plan. A [`TypeEnv`] snapshots the state of the type checker after
//! processing a "library" set of decls (e.g. chelis-std + reef deps +
//! the package's own modules). New code can then be type-checked against
//! that context: name resolution walks the new-code's bindings first and
//! falls through to the library snapshot, but the snapshot itself is
//! never mutated.
//!
//! ## Stacking semantics
//!
//! - **Type-binding env (`Env`)**: cloned per check call. New-code
//!   bindings overwrite library bindings of the same name within the
//!   clone — that is shadowing, since the library snapshot is preserved
//!   and a subsequent check call sees the unmodified library env.
//! - **ADT registry**: cloned per check call. Library `deftype`s remain
//!   visible (so `match libtype { ... }` exhaustivity sees the full
//!   constructor set); new `deftype`s are registered in the clone only.
//!   Naive inner-empty/outer-fallback registry stacking would either
//!   silently accept any match (no constructors found ⇒ "exhaustive") or
//!   reject every match (no constructors found ⇒ "missing all"), so we
//!   use clone-then-extend instead — the cloned registry already
//!   contains the library's constructor sets.
//! - **Type-resolution environment**: rebuilt for each check from the cloned
//!   registry's validated ADT/alias definitions plus the new unit's
//!   precollected declaration headers. The provisional environment is a
//!   serde-skipped runtime field: it permits self/forward references during
//!   one check but cannot serialize a rejected declaration into the library
//!   snapshot.
//! - **`VarGen`**: cloned per check call. Library type-variable IDs are
//!   already taken; new variables come from numbers above the library's
//!   high-water mark.
//! - **`Subst`**: cloned per check call. Library-derived constraints
//!   stay visible to new-code unification.
//! - **Ir type-env**: cloned per check call. Library declared types
//!   remain visible to new-code shape validation; the new-code's own
//!   declared types are added on top.
//! - **Callable provenance**: immutable ordered formal-name identities are
//!   retained for library values so contextual `grad` validation applies the
//!   same metadata/index consistency rule as a monolithic check. New
//!   declarations shadow this snapshot without mutating it.
//!
//! ## No leak invariant
//!
//! The library snapshot is shared via [`Arc`] and is never written to.
//! Each `check_ir_with_context` call clones the inner state at the
//! start, mutates the clone with new-code bindings, and discards the
//! clone when the call returns. A binding declared in one new-code
//! snippet is therefore invisible to a subsequent snippet checked
//! against the same context.
//!
//! ## Position-sensitive errors
//!
//! Errors emitted by the new-code check carry spans from the new-code
//! Deep AST nodes. The library was already checked when the context was
//! built; its errors (if any) are surfaced at build time, not silently
//! re-emitted with library spans during a new-code check.
//!
//! ## Deserialization boundary
//!
//! `TypeEnv` derives serde because compiler-api caches successful checker
//! snapshots. Deserialization restores validated definitions; the transient
//! type-resolution environment is skipped and is reconstructed at the next
//! check. Serde can structurally decode the private `ErrorWitness` carried by
//! `Type::Error`, but successful context construction rejects all checker
//! errors and the totality invariant forbids error types in emitted snapshots.
//! Compiler API cache decoding verifies the envelope and build identity.
//! It requires one opaque identity on both type products.
//! It reruns effect and linearity checks before it creates a library proof.

use chelis_unord::UnordSet;
use std::collections::BTreeMap;
use std::sync::Arc;

use chelis_deep::ast as deep;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::adt::AdtRegistry;
use crate::builtins;
use crate::env::Env;
use crate::types::VarGen;
use crate::unify::Subst;

/// Opaque identity derived from an accepted checked library and its base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryProofId {
    digest: [u8; 32],
    selector_context_digest: [u8; 32],
}

impl LibraryProofId {
    pub(crate) fn for_library(
        annotated_exprs: &[deep::Expr],
        context: Option<Self>,
        selector_context_digest: [u8; 32],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"chelis-library-proof-v1");
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
        Self {
            digest: hasher.finalize().into(),
            selector_context_digest,
        }
    }
}

/// Outer-scope snapshot for stacked IR type checking.
///
/// Cheap to clone — internally `Arc`-shared. Build from a library decl
/// list with [`crate::build_type_env_from_library`], or create an
/// empty one with [`TypeEnv::empty`].
#[derive(Debug, Clone)]
pub struct TypeEnv {
    inner: Arc<TypeEnvInner>,
    library_proof_id: Option<LibraryProofId>,
}

// Source-free snapshots are faithful transports from a trusted checker
// producer, not independently re-proved programs. Structural validation does
// not establish completeness of a maliciously edited label summary.
// v2 added #2071's type-variable restrictions. v3 added checked collection
// relations to `Scheme`; reading an older snapshot as an empty relation list
// would change which indirect calls are admitted. v4 adds immutable callable
// provenance; reading v3 as an empty map would reject valid contextual named
// gradient selectors. v5 adds [04-LIN-10]'s key-free type-variable marks and
// key-carrying data types; reading v4 as empty sets would let a library generic
// be instantiated at a key. v6 records whether each mark's generic is a value
// binding; reading v5 as function generics would change a value binding's
// suggested repair after a round trip.
const TYPE_ENV_FORMAT_VERSION: u32 = 6;

#[derive(Serialize)]
struct TypeEnvWireRef<'a> {
    format_version: u32,
    inner: &'a Arc<TypeEnvInner>,
    library_proof_id: Option<LibraryProofId>,
}

#[derive(Deserialize)]
struct TypeEnvWire {
    format_version: u32,
    inner: Arc<TypeEnvInner>,
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
        wire.inner
            .subst
            .validate_dimension_labels(&wire.inner.var_gen)
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            inner: wire.inner,
            library_proof_id: wire.library_proof_id,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TypeEnvInner {
    /// Type-binding env: every name ever declared in the library, mapped
    /// to its (generalized) scheme. Includes builtins and prelude ADT
    /// constructors.
    pub(crate) env: Env,
    /// VarGen at the library's high-water mark — new fresh tvars/dvars
    /// for new-code checking start above the library's last allocation.
    pub(crate) var_gen: VarGen,
    /// Substitution carrying any library-derived constraints.
    pub(crate) subst: Subst,
    /// ADT registry containing library `deftype`s, type aliases, and
    /// prelude ADTs (Option, List, etc).
    pub(crate) adt_reg: AdtRegistry,
    /// IR declared-type lookup: library def name → declared type
    /// expression. Used by the validate pass (which checks shape
    /// invariants against declared types).
    pub(crate) ir_types: BTreeMap<String, deep::Expr>,
    /// Set of names declared by the library — used by the new-code
    /// cycle / unbound suppression logic to distinguish library
    /// references from new-code references.
    pub(crate) library_def_names: UnordSet<String>,
    /// Module-scoped immutable callable identities and ordered formal names
    /// accepted from the library source. Contextual `grad` validation seeds
    /// its structural resolver from this snapshot.
    pub(crate) selector_callables: crate::infer::SelectorCallableContext,
    /// Checker-enforced opacity metadata (RFC D-CHECK): per-module
    /// export sets, binding -> module attribution, and producer text,
    /// accumulated across the library and new-code phases. Defaults
    /// so older serialized payloads remain decodable in principle;
    /// the cache format version still gates real reuse.
    #[serde(default)]
    pub(crate) opacity: crate::opacity::OpacityModuleMeta,
}

impl TypeEnv {
    /// Empty outer scope — the result of building a context from `&[]`.
    /// `check_ir_with_context(&TypeEnv::empty(), exprs)` is
    /// behaviorally equivalent to [`crate::check_ir_program(exprs)`]
    /// on the same `exprs`.
    pub fn empty() -> Self {
        let (env, var_gen) = builtins::builtin_env();
        let mut adt_reg = AdtRegistry::new();
        let mut env = env;
        let mut var_gen = var_gen;
        builtins::register_prelude_adts(&mut env, &mut var_gen, &mut adt_reg);
        Self {
            inner: Arc::new(TypeEnvInner {
                env,
                var_gen,
                subst: Subst::new(),
                adt_reg,
                ir_types: BTreeMap::new(),
                library_def_names: UnordSet::new(),
                selector_callables: crate::infer::SelectorCallableContext::default(),
                opacity: crate::opacity::OpacityModuleMeta::default(),
            }),
            library_proof_id: None,
        }
    }

    /// Internal constructor — only the `infer` module calls this after
    /// running the library through the full IR pipeline.
    pub(crate) fn from_inner(inner: TypeEnvInner) -> Self {
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

    pub(crate) fn inner(&self) -> &TypeEnvInner {
        &self.inner
    }

    /// Clone this persisted checker snapshot for a new checking unit.
    /// Variables imported from the snapshot are normalized to level zero
    /// before any IDs for the new unit can be minted.
    pub(crate) fn resume_for_new_check(&self) -> TypeEnvInner {
        let mut inner = self.inner().clone();
        inner.subst.resume_for_new_check(&inner.var_gen);
        inner
    }

    /// Number of library defs in scope. Diagnostic helper.
    pub fn library_def_count(&self) -> usize {
        self.inner.library_def_names.len()
    }

    /// Whether a name was declared by the library this context was built
    /// from. Diagnostic helper — does NOT walk builtins or prelude.
    pub fn has_library_def(&self, name: &str) -> bool {
        self.inner.library_def_names.contains(name)
    }

    /// Return the generalized checker scheme for a binding in this accepted
    /// library context.
    ///
    /// Public package metadata uses this immutable view so quantified-domain
    /// restrictions remain attached to function values at serialization
    /// boundaries instead of being reconstructed from a printed type.
    pub fn scheme(&self, name: &str) -> Option<&crate::types::Scheme> {
        self.inner.env.lookup(name)
    }

    /// Confirm that this context and a checked program are one library product pair.
    ///
    /// The library builders derive one opaque identity from accepted checked source
    /// and its exact callable-selector provenance snapshot. Cache parsing requires
    /// that identity, the provenance digest, and the declared-type map to match.
    pub fn matches_checked_program(&self, program: &crate::CheckedProgram) -> bool {
        let selector_context_digest =
            crate::infer::selector_callable_context_digest(&self.inner.selector_callables);
        self.library_proof_id
            .is_some_and(|proof| proof.selector_context_digest == selector_context_digest)
            && self.library_proof_id == program.library_proof_id()
            && self.inner.ir_types.eq(program.type_env())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Dim, Prim, Scheme, TensorPrec, Type, TypeVarRestriction};
    use crate::unify::unify;

    #[test]
    fn dimension_snapshot_rejects_obsolete_direct_encoding_and_missing_summary() {
        let context = TypeEnv::empty();
        // Previous source-free wire had no leading format version.
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
                crate::types::DimVar(u32::MAX)
            } else {
                inner.var_gen.fresh_dvar()
            };
            inner.subst.protect_dimensions([v]);
            crate::unify::unify_dim(&Dim::Var(v), &Dim::Name(label.into()), &mut inner.subst)
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
        let composite = Type::Tuple(vec![
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

        let mut resumed = decoded.resume_for_new_check();
        assert_eq!(resumed.subst.level_of_tvar(child_tvar), 0);
        assert_eq!(resumed.subst.level_metadata_counts(), (1, 0, 0, 0));
        let imported = resumed
            .env
            .generalize(&Type::Var(child_tvar), &resumed.subst);
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
            resumed = decoded.resume_for_new_check();
            assert_eq!(resumed.subst.level_of_tvar(fresh), 0);
            assert_eq!(resumed.subst.level_metadata_counts(), (1, 0, 0, 0));
            let imported = resumed.env.generalize(&Type::Var(fresh), &resumed.subst);
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
            let mut resumed = decoded.resume_for_new_check();
            let scheme = resumed.env.lookup(name).unwrap().clone();
            assert!(
                scheme
                    .tvar_restrictions
                    .iter()
                    .any(|(_, bound)| *bound == restriction)
            );
            let Type::Fn(params, _) =
                resumed
                    .env
                    .instantiate(&scheme, &mut resumed.var_gen, &resumed.subst)
            else {
                panic!("operation remains callable");
            };
            let precision = resumed.var_gen.fresh_tvar();
            let tensor = Type::Tensor(vec![Dim::Lit(3)], TensorPrec::Var(precision));
            let operand = if matches!(params[0], Type::Ref(_)) {
                Type::Ref(Box::new(tensor))
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
                crate::unify::TypeErrorKind::DtypeFamilyMismatch
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

        let mut resumed = decoded.resume_for_new_check();
        let scheme = resumed
            .env
            .lookup("test_assert_close_tensor")
            .expect("builtin scheme survives round trip")
            .clone();
        assert_eq!(
            scheme.tvar_restrictions,
            vec![(scheme.tvars[0], TypeVarRestriction::ActiveFloat)]
        );
        let instantiated = resumed
            .env
            .instantiate(&scheme, &mut resumed.var_gen, &resumed.subst);
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
            crate::unify::TypeErrorKind::DtypeFamilyMismatch
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
