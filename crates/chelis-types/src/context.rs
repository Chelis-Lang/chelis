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
//! - **`VarGen`**: cloned per check call. Library type-variable IDs are
//!   already taken; new variables come from numbers above the library's
//!   high-water mark.
//! - **`Subst`**: cloned per check call. Library-derived constraints
//!   stay visible to new-code unification.
//! - **Phase0e type-env**: cloned per check call. Library declared types
//!   remain visible to new-code shape validation; the new-code's own
//!   declared types are added on top.
//!
//! ## No leak invariant
//!
//! The library snapshot is shared via [`Arc`] and is never written to.
//! Each `check_phase0e_with_context` call clones the inner state at the
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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chelis_deep::ast as deep;
use serde::{Deserialize, Serialize};

use crate::adt::AdtRegistry;
use crate::builtins;
use crate::env::Env;
use crate::types::VarGen;
use crate::unify::Subst;

/// Outer-scope snapshot for stacked Phase 0e type checking.
///
/// Cheap to clone — internally `Arc`-shared. Build from a library decl
/// list with [`crate::build_type_env_from_library`], or create an
/// empty one with [`TypeEnv::empty`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeEnv {
    inner: Arc<TypeEnvInner>,
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
    /// Phase 0e declared-type lookup: library def name → declared type
    /// expression. Used by the validate pass (which checks shape
    /// invariants against declared types).
    pub(crate) phase0e_types: HashMap<String, deep::Expr>,
    /// Set of names declared by the library — used by the new-code
    /// cycle / unbound suppression logic to distinguish library
    /// references from new-code references.
    pub(crate) library_def_names: HashSet<String>,
}

impl TypeEnv {
    /// Empty outer scope — the result of building a context from `&[]`.
    /// `check_phase0e_with_context(&TypeEnv::empty(), exprs)` is
    /// behaviorally equivalent to [`crate::check_phase0e_program(exprs)`]
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
                phase0e_types: HashMap::new(),
                library_def_names: HashSet::new(),
            }),
        }
    }

    /// Internal constructor — only the `infer` module calls this after
    /// running the library through the full Phase 0e pipeline.
    pub(crate) fn from_inner(inner: TypeEnvInner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub(crate) fn inner(&self) -> &TypeEnvInner {
        &self.inner
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
}
