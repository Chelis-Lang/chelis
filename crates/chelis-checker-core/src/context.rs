//! Ordinary semantic continuation data. Host proof admission remains separate.

use crate::adt::AdtRegistry;
use crate::builtins;
use crate::env::Env;
use crate::errors::ErrorWitness;
use crate::prelude::*;
use alloc::collections::BTreeMap;
use chelis_deep::ast as deep;
use chelis_type_core::{Subst, VarGen};
use chelis_unord::UnordSet;

#[derive(Debug, Clone)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(any(test, feature = "serde"), serde(rename = "TypeEnvInner"))]
pub struct ContextSnapshot {
    pub env: Env,
    pub var_gen: VarGen,
    pub subst: Subst<ErrorWitness>,
    pub adt_reg: AdtRegistry,
    pub ir_types: BTreeMap<String, deep::Expr>,
    pub library_def_names: UnordSet<String>,
    pub(crate) selector_callables: crate::infer::SelectorCallableContext,
    pub(crate) rank_bodies: crate::infer::RankBodyContext,
    #[cfg_attr(any(test, feature = "serde"), serde(default))]
    pub(crate) opacity: crate::opacity::OpacityModuleMeta,
}

impl ContextSnapshot {
    pub fn empty() -> Self {
        let (env, var_gen) = builtins::builtin_env();
        let mut adt_reg = AdtRegistry::new();
        let mut env = env;
        let mut var_gen = var_gen;
        builtins::register_prelude_adts(&mut env, &mut var_gen, &mut adt_reg);
        Self {
            env,
            var_gen,
            subst: Subst::<ErrorWitness>::new(),
            adt_reg,
            ir_types: BTreeMap::new(),
            library_def_names: UnordSet::new(),
            selector_callables: crate::infer::SelectorCallableContext::default(),
            rank_bodies: crate::infer::RankBodyContext::default(),
            opacity: crate::opacity::OpacityModuleMeta::default(),
        }
    }

    pub fn resume_for_new_check(&self) -> Self {
        let mut inner = self.clone();
        inner.subst.resume_for_new_check(&inner.var_gen);
        inner
    }
    pub fn library_def_count(&self) -> usize {
        self.library_def_names.len()
    }
    pub fn has_library_def(&self, name: &str) -> bool {
        self.library_def_names.contains(name)
    }
    pub fn scheme(&self, name: &str) -> Option<&chelis_type_core::Scheme<ErrorWitness>> {
        self.env.lookup(name)
    }
    pub fn type_env(&self) -> &BTreeMap<String, deep::Expr> {
        &self.ir_types
    }
    pub fn selector_context_digest(&self) -> [u8; 32] {
        crate::infer::selector_callable_context_digest(&self.selector_callables)
    }
    pub fn rank_context_digest(&self) -> [u8; 32] {
        self.rank_bodies.digest()
    }
    pub fn validate(&self) -> Result<(), String> {
        self.subst.validate_dimension_labels(&self.var_gen)?;
        self.rank_bodies.validate(&self.ir_types)
    }
    pub fn validate_rank_bodies(&self) -> Result<(), String> {
        self.rank_bodies.validate(&self.ir_types)
    }
    #[cfg(test)]
    pub(crate) fn with_test_binding(
        &self,
        name: &str,
        scheme: chelis_type_core::Scheme<ErrorWitness>,
    ) -> Self {
        let mut snapshot = self.clone();
        snapshot.env.bind(name.to_string(), scheme);
        snapshot
    }
    pub fn has_rank_bodies(&self) -> bool {
        !self.rank_bodies.is_empty()
    }
}
