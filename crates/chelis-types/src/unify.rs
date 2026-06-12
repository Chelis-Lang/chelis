//! Unification algorithm for the Chelis type checker.
//!
//! Solves type equations by binding type variables and dimension variables.
//!
//! ## Path compression
//!
//! `Subst` stores type / dim variable bindings as union-find chains. When a
//! caller resolves a variable via [`Subst::apply`] or [`Subst::apply_dim`],
//! the resolver iteratively follows the chain to its terminal value AND
//! writes the resolved value back into the substitution so future lookups
//! land in O(1). The HashMaps live behind a `RefCell` so this in-place
//! compression remains available to callers that hold an immutable
//! borrow of `Subst` (annotation passes, `Env::generalize`, occurs checks).
//!
//! Without path compression, deeply-nested expressions like Cons chains
//! over wildcard tensor literals (e.g.,
//! `pad_sequences_to([[10×4 floats]], ...)`) accumulate dim-variable
//! chains 1000+ links long; resolving every dim through a 1000-link
//! chain on every type walk drives ~2 billion `apply_dim` calls and
//! pushes phase-0e annotation from ~2s into 10+ minutes for chelis-std's
//! `nn/embedding.ch`. See the gdb backtrace recorded in this commit's
//! body for the canonical reproducer.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::types::*;

/// Type error produced during unification.
#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    OccursCheck,
    NotAFunction,
}

/// Substitution: maps type variables to types and dim variables to dims.
///
/// The HashMaps are wrapped in `Mutex` so [`Subst::apply`] and
/// [`Subst::apply_dim`] can perform in-place path compression while
/// keeping the public method receiver `&self`. The Mutex is uncontended
/// in normal use (each `compile_new_source_in_context` call clones the
/// substitution into its own thread-local state), so the locking cost
/// is one atomic compare-exchange per call. Use [`Subst::insert_type`] /
/// [`Subst::insert_dim`] to record new bindings during unification.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Subst {
    types: Mutex<HashMap<TypeVar, Type>>,
    dims: Mutex<HashMap<DimVar, Dim>>,
    /// Rank-variable bindings: a `RankVar` binds to the *entire* shape vector
    /// it stands for (Tier-2 rank polymorphism). A binding to `[Dim::Rank(r2)]`
    /// is a rank-to-rank alias resolved transitively by `resolve_rvar`.
    ranks: Mutex<HashMap<RankVar, Vec<Dim>>>,
    /// Issue #256 soundness ledger. The `borrow` inference arm accepts a
    /// borrow whose inner type is still an unresolved `Type::Var`,
    /// deferring the tensor-or-carrier classification to subsequent
    /// unification (the surrounding `&tensor[..]` parameter pins it).
    /// That deferral is only sound when the variable is *eventually*
    /// pinned to a tensor or tensor-carrying type. When the consumer is
    /// itself fully polymorphic (e.g. `consume_any[a](t: a)`), the
    /// variable is never pinned and a genuinely-non-tensor value would
    /// slip past every gate. Each deferred borrow records the inner
    /// `TypeVar` here; after a def body's inference completes, the
    /// driver resolves each one against the now-complete substitution
    /// and rejects any that did not become a tensor or tensor carrier.
    /// Not serialized: this is transient per-pass bookkeeping, drained
    /// by the inference driver, and never part of a persisted context.
    #[serde(skip)]
    deferred_borrow_vars: Mutex<Vec<TypeVar>>,
}

impl Clone for Subst {
    fn clone(&self) -> Self {
        Subst {
            types: Mutex::new(self.types.lock().expect("subst.types poisoned").clone()),
            dims: Mutex::new(self.dims.lock().expect("subst.dims poisoned").clone()),
            ranks: Mutex::new(self.ranks.lock().expect("subst.ranks poisoned").clone()),
            deferred_borrow_vars: Mutex::new(
                self.deferred_borrow_vars
                    .lock()
                    .expect("subst.deferred_borrow_vars poisoned")
                    .clone(),
            ),
        }
    }
}

impl Subst {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of the type-variable bindings (cloned out of the lock).
    /// Useful for serialization, tests, and read-only inspection.
    pub fn types_snapshot(&self) -> HashMap<TypeVar, Type> {
        self.types.lock().expect("subst.types poisoned").clone()
    }

    /// Snapshot of the dim-variable bindings.
    pub fn dims_snapshot(&self) -> HashMap<DimVar, Dim> {
        self.dims.lock().expect("subst.dims poisoned").clone()
    }

    /// Number of type-variable bindings currently in the substitution.
    pub fn types_len(&self) -> usize {
        self.types.lock().expect("subst.types poisoned").len()
    }

    /// Number of dim-variable bindings currently in the substitution.
    pub fn dims_len(&self) -> usize {
        self.dims.lock().expect("subst.dims poisoned").len()
    }

    /// Record a new type-variable binding.
    pub fn insert_type(&mut self, v: TypeVar, ty: Type) {
        self.types
            .lock()
            .expect("subst.types poisoned")
            .insert(v, ty);
    }

    /// Record a new dim-variable binding.
    pub fn insert_dim(&mut self, v: DimVar, dim: Dim) {
        self.dims
            .lock()
            .expect("subst.dims poisoned")
            .insert(v, dim);
    }

    /// Record a rank-variable binding: `r` stands for the whole shape `dims`.
    pub fn insert_rank(&mut self, r: RankVar, dims: Vec<Dim>) {
        self.ranks
            .lock()
            .expect("subst.ranks poisoned")
            .insert(r, dims);
    }

    /// Snapshot of the rank-variable bindings.
    pub fn ranks_snapshot(&self) -> HashMap<RankVar, Vec<Dim>> {
        self.ranks.lock().expect("subst.ranks poisoned").clone()
    }

    /// Number of rank-variable bindings currently in the substitution.
    pub fn ranks_len(&self) -> usize {
        self.ranks.lock().expect("subst.ranks poisoned").len()
    }

    /// Resolve a rank variable to the shape vector it stands for, chasing
    /// rank-to-rank aliases. Returns `[Dim::Rank(r)]` (the unbound variable)
    /// when `r` has no binding. The returned dims are not themselves
    /// substitution-applied; callers run them through `apply_dim`.
    fn resolve_rvar(&self, start: RankVar) -> Vec<Dim> {
        let map = self.ranks.lock().expect("subst.ranks poisoned");
        let mut current = start;
        // Bound the alias chase. A chain cannot exceed the number of bound
        // ranks; a pathological cyclic alias (e.g. a swapped `..a`/`..b`
        // return sig that aliases a:=b and b:=a) is broken here by returning
        // the current rank rather than looping forever.
        let max_steps = map.len() + 1;
        for _ in 0..max_steps {
            match map.get(&current) {
                None => return vec![Dim::Rank(current)],
                Some(bound) => match bound.as_slice() {
                    [Dim::Rank(next)] if *next != current => current = *next,
                    [Dim::Rank(_)] => return vec![Dim::Rank(current)],
                    _ => return bound.clone(),
                },
            }
        }
        vec![Dim::Rank(current)]
    }

    /// Issue #256: record a borrow site whose inner type was still an
    /// unresolved `Type::Var` when the `borrow` inference arm ran. The
    /// driver drains these after a def body's inference completes and
    /// re-checks each against the final substitution. See the
    /// `deferred_borrow_vars` field doc for the soundness rationale.
    pub fn record_deferred_borrow_var(&self, v: TypeVar) {
        self.deferred_borrow_vars
            .lock()
            .expect("subst.deferred_borrow_vars poisoned")
            .push(v);
    }

    /// Drain the deferred-borrow ledger, returning every recorded
    /// `TypeVar`. Called once per def body's inference by the driver so
    /// the ledger does not leak deferred sites across defs.
    pub fn take_deferred_borrow_vars(&self) -> Vec<TypeVar> {
        std::mem::take(
            &mut *self
                .deferred_borrow_vars
                .lock()
                .expect("subst.deferred_borrow_vars poisoned"),
        )
    }

    /// Resolve a type variable to its terminal binding (a non-Var, or
    /// an unbound Var). Iteratively walks the chain and rewrites every
    /// link directly to the terminal so future lookups are O(1).
    fn resolve_tvar(&self, start: TypeVar) -> Type {
        let mut map = self.types.lock().expect("subst.types poisoned");
        // Phase 1: walk to the terminal, recording the chain.
        let mut chain: Vec<TypeVar> = Vec::new();
        let mut current = start;
        let terminal = loop {
            match map.get(&current) {
                None => break Type::Var(current),
                Some(Type::Var(v)) if *v == current => break Type::Var(current),
                Some(Type::Var(v)) => {
                    chain.push(current);
                    current = *v;
                }
                Some(other) => {
                    chain.push(current);
                    break other.clone();
                }
            }
        };
        // Phase 2: compress — point every link in the chain directly at
        // the terminal value.
        if chain.len() > 1 {
            for v in chain {
                map.insert(v, terminal.clone());
            }
        }
        terminal
    }

    /// Resolve a dim variable to its terminal binding. Same iterative-
    /// walk-then-compress shape as [`Subst::resolve_tvar`].
    fn resolve_dvar(&self, start: DimVar) -> Dim {
        let mut map = self.dims.lock().expect("subst.dims poisoned");
        let mut chain: Vec<DimVar> = Vec::new();
        let mut current = start;
        let terminal = loop {
            match map.get(&current) {
                None => break Dim::Var(current),
                Some(Dim::Var(v)) if *v == current => break Dim::Var(current),
                Some(Dim::Var(v)) => {
                    chain.push(current);
                    current = *v;
                }
                Some(other) => {
                    chain.push(current);
                    break other.clone();
                }
            }
        };
        if chain.len() > 1 {
            for v in chain {
                map.insert(v, terminal.clone());
            }
        }
        terminal
    }

    /// Apply this substitution to a type, resolving all bound variables.
    /// Path-compresses any chains of length ≥ 2 it encounters so future
    /// lookups land in O(1).
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => self.resolve_tvar(*v),
            Type::Fn(args, ret) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                let ret = Box::new(self.apply(ret));
                Type::Fn(args, ret)
            }
            Type::Ref(inner) => Type::Ref(Box::new(self.apply(inner))),
            Type::Tensor(dims, prec) => {
                // A `Dim::Rank` expands to the whole shape vector it is bound
                // to (or stays as the sole `Dim::Rank` while unbound).
                let mut out: Vec<Dim> = Vec::with_capacity(dims.len());
                for d in dims {
                    match d {
                        Dim::Rank(r) => {
                            for rd in self.resolve_rvar(*r) {
                                out.push(self.apply_dim(&rd));
                            }
                        }
                        _ => out.push(self.apply_dim(d)),
                    }
                }
                Type::Tensor(out, self.apply_tensor_prec(prec))
            }
            Type::Adt(name, args) => {
                let args = args.iter().map(|a| self.apply(a)).collect();
                Type::Adt(name.clone(), args)
            }
            Type::Tuple(ts) => {
                let ts = ts.iter().map(|t| self.apply(t)).collect();
                Type::Tuple(ts)
            }
            Type::Prim(_) | Type::Unit | Type::Error => ty.clone(),
        }
    }

    /// Apply this substitution to a dimension. Path-compresses chains.
    pub fn apply_dim(&self, dim: &Dim) -> Dim {
        match dim {
            Dim::Var(v) => self.resolve_dvar(*v),
            _ => dim.clone(),
        }
    }

    /// Apply this substitution to a tensor precision slot.
    ///
    /// Resolves a `TensorPrec::Var` through the type-variable
    /// substitution chain. If the resolved binding is a concrete
    /// primitive, the slot collapses to `TensorPrec::Concrete(_)`. If
    /// the binding is itself another type variable, we keep the slot
    /// as `TensorPrec::Var(target_var)` so the tensor stays
    /// well-formed (the precision slot must be a precision, not an
    /// arbitrary `Type`). Any non-prim, non-var binding (e.g. a
    /// downstream unification error that bound the slot var to
    /// `Type::Fn` or `Type::Tensor`) collapses to `Type::Error` at
    /// the surrounding `Type` level — but here we conservatively
    /// retain the original var; the unifier surfaces the mismatch.
    pub fn apply_tensor_prec(&self, prec: &TensorPrec) -> TensorPrec {
        match prec {
            TensorPrec::Concrete(_) => prec.clone(),
            TensorPrec::Var(v) => match self.resolve_tvar(*v) {
                Type::Prim(p) => TensorPrec::Concrete(p),
                Type::Var(v2) => TensorPrec::Var(v2),
                _ => prec.clone(),
            },
        }
    }

    /// Compose: apply `other` to all bindings in self, then merge.
    pub fn compose(&mut self, other: &Subst) {
        // Note: `other.apply` / `other.apply_dim` lock `other`'s maps;
        // we must not be holding a lock on `other` simultaneously
        // (which we never do — `self` and `other` are distinct).
        {
            let mut self_types = self.types.lock().expect("subst.types poisoned");
            for val in self_types.values_mut() {
                *val = other.apply(val);
            }
        }
        {
            let mut self_dims = self.dims.lock().expect("subst.dims poisoned");
            for val in self_dims.values_mut() {
                *val = other.apply_dim(val);
            }
        }
        {
            // Rank bindings are `Dim::Rank`-free runs (enforced by `bind_rvar`),
            // so applying `other` is a per-dim `apply_dim`. Omitting this would
            // silently drop rank substitutions through a compose — a Tier-3
            // footgun since ranks are now load-bearing.
            let mut self_ranks = self.ranks.lock().expect("subst.ranks poisoned");
            for val in self_ranks.values_mut() {
                *val = val.iter().map(|d| other.apply_dim(d)).collect();
            }
        }
        let other_types = other.types_snapshot();
        let other_dims = other.dims_snapshot();
        let other_ranks = other.ranks_snapshot();
        {
            let mut self_types = self.types.lock().expect("subst.types poisoned");
            for (k, v) in other_types {
                self_types.entry(k).or_insert(v);
            }
        }
        {
            let mut self_dims = self.dims.lock().expect("subst.dims poisoned");
            for (k, v) in other_dims {
                self_dims.entry(k).or_insert(v);
            }
        }
        {
            let mut self_ranks = self.ranks.lock().expect("subst.ranks poisoned");
            for (k, v) in other_ranks {
                self_ranks.entry(k).or_insert(v);
            }
        }
        // `deferred_borrow_vars` is intentionally NOT merged: it is a transient
        // per-def ledger (issue #256), drained after each body's inference, not
        // part of the substitution's logical content.
    }
}

/// Unify two types, producing a substitution or a type error.
pub fn unify(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    let t1 = subst.apply(t1);
    let t2 = subst.apply(t2);

    match (&t1, &t2) {
        // Same type — trivially unified
        (Type::Prim(p1), Type::Prim(p2)) if p1 == p2 => Ok(()),
        (Type::Prim(p1), Type::Prim(p2)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!(
                "precision mismatch: expected {}, got {}",
                p1.name(),
                p2.name()
            ),
        }),

        (Type::Unit, Type::Unit) => Ok(()),

        (Type::Ref(inner1), Type::Ref(inner2)) => unify(inner1, inner2, subst),

        // Type variable binding
        (Type::Var(v), _) => bind_tvar(*v, &t2, subst),
        (_, Type::Var(v)) => bind_tvar(*v, &t1, subst),

        // Function types
        (Type::Fn(args1, ret1), Type::Fn(args2, ret2)) => {
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "function arity mismatch: expected {} args, got {}",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify(a1, a2, subst)?;
            }
            unify(ret1, ret2, subst)?;
            // chelis#339: after the signature's rank spreads are bound, a
            // result-introduced axis name (a named-axis expand insert) must
            // not also be covered by a spread binding — the monomorphized
            // result row would carry the same dim name twice with two
            // different extents, making every later by-name lookup ambiguous.
            check_introduced_name_rank_collision(args1, ret1, subst)?;
            check_introduced_name_rank_collision(args2, ret2, subst)
        }

        // Tensor types
        (Type::Tensor(dims1, p1), Type::Tensor(dims2, p2)) => {
            unify_tensor_prec(p1, p2, subst)?;
            // Rank polymorphism: a `Dim::Rank` is a *name-preserving spread*
            // standing for a run of dims. A shape is `Rank? (Name Rank?)*`
            // (no two spreads adjacent). Resolve both shapes (expanding any
            // bound spread to its run), then dispatch on how many spreads each
            // side carries. With at most one spread per gap and a named anchor
            // locating each interior split, unification stays unitary — there
            // is one most-general binding. See
            // `spec/design/rank_polymorphism.md` §Unification.
            let d1 = resolve_shape(dims1, subst);
            let d2 = resolve_shape(dims2, subst);
            let s1 = d1.iter().filter(|d| matches!(d, Dim::Rank(_))).count();
            let s2 = d2.iter().filter(|d| matches!(d, Dim::Rank(_))).count();
            match (s1, s2) {
                // Both ground (Tier-1 / fully-monomorphic): length + element-wise.
                (0, 0) => {
                    if d1.len() != d2.len() {
                        return Err(TypeError {
                            kind: TypeErrorKind::DimensionMismatch,
                            message: format!(
                                "tensor rank mismatch: {} dims vs {} dims",
                                d1.len(),
                                d2.len()
                            ),
                        });
                    }
                    for (a, b) in d1.iter().zip(d2.iter()) {
                        unify_dim(a, b, subst)?;
                    }
                    Ok(())
                }
                // Exactly one side carries spreads: split the ground side.
                (_, 0) => unify_row_against_ground(&d1, &d2, subst),
                (0, _) => unify_row_against_ground(&d2, &d1, subst),
                // Both carry spreads: only structurally-identical rows unify.
                (_, _) => unify_row_against_row(&d1, &d2, subst),
            }
        }

        // ADT types
        (Type::Adt(n1, args1), Type::Adt(n2, args2)) => {
            if n1 != n2 {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch,
                    message: format!("type mismatch: {n1} vs {n2}"),
                });
            }
            if args1.len() != args2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!(
                        "ADT type argument count mismatch for {n1}: {} vs {}",
                        args1.len(),
                        args2.len()
                    ),
                });
            }
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                unify(a1, a2, subst)?;
            }
            Ok(())
        }

        // Tuple types
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            if ts1.len() != ts2.len() {
                return Err(TypeError {
                    kind: TypeErrorKind::ArityMismatch,
                    message: format!("tuple length mismatch: {} vs {}", ts1.len(), ts2.len()),
                });
            }
            for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                unify(t1, t2, subst)?;
            }
            Ok(())
        }

        // Error propagation: unifying with Error trivially succeeds so
        // downstream call sites do not fan out a cascade of secondary
        // diagnostics from a single upstream error. Call sites that
        // require a precision/shape match against a non-Error declared
        // type must check for the Error sentinel themselves and surface
        // the mismatch explicitly (e.g. the def-body vs declared-sig
        // unify in `infer.rs` does this for WS-A5 RT-3a F1: when the
        // body collapses to Error but the declared type is concrete, we
        // still emit a "body has type `<error>`, declared type is `T`"
        // diagnostic so the user sees the unresolved declared shape).
        (Type::Error, _) | (_, Type::Error) => Ok(()),

        // Everything else is a mismatch
        _ => Err(TypeError {
            kind: TypeErrorKind::TypeMismatch,
            message: format!("type mismatch: {t1} vs {t2}"),
        }),
    }
}

/// Unify two tensor precision slots per `spec/04-type-system.md` §5.8
/// (WS-A5 precision polymorphism).
///
/// - `Concrete(p1)` and `Concrete(p2)` unify only when `p1 == p2`.
/// - `Var(v)` unifies with `Concrete(p)` by binding `v` to `Type::Prim(p)`
///   in the substitution; further references through that var resolve to
///   the concrete prim via `Subst::apply_tensor_prec`.
/// - Two `Var`s unify by linking them at the type-variable level
///   (delegated to `bind_tvar`), keeping the precision slot's monomorphic
///   structure consistent with the rest of the type system.
pub fn unify_tensor_prec(
    p1: &TensorPrec,
    p2: &TensorPrec,
    subst: &mut Subst,
) -> Result<(), TypeError> {
    let p1 = subst.apply_tensor_prec(p1);
    let p2 = subst.apply_tensor_prec(p2);
    match (&p1, &p2) {
        (TensorPrec::Concrete(a), TensorPrec::Concrete(b)) if a == b => Ok(()),
        (TensorPrec::Concrete(a), TensorPrec::Concrete(b)) => Err(TypeError {
            kind: TypeErrorKind::PrecisionMismatch,
            message: format!("tensor precision mismatch: {} vs {}", a.name(), b.name()),
        }),
        (TensorPrec::Var(v), TensorPrec::Concrete(p)) => bind_tvar(*v, &Type::Prim(*p), subst),
        (TensorPrec::Concrete(p), TensorPrec::Var(v)) => bind_tvar(*v, &Type::Prim(*p), subst),
        (TensorPrec::Var(v1), TensorPrec::Var(v2)) => {
            if v1 == v2 {
                Ok(())
            } else {
                bind_tvar(*v1, &Type::Var(*v2), subst)
            }
        }
    }
}

/// Unify two dimensions.
///
/// Wildcard ↔ Var invariant: `unify_dim(Wildcard, Var(v))` succeeds
/// without binding `v`. The `(Wildcard, _) | (_, Wildcard) => Ok(())`
/// arm matches before the `(Var(v), _) => bind_dvar(...)` arm and
/// returns `Ok(())` with no side effects, so the dim var stays free.
/// This is intentional — binding `v := Wildcard` would freeze `v` and
/// make a later concrete arg in the same sig unable to constrain it
/// (`apply_dim` would resolve `Var(v) → Wildcard` and the permissive
/// Wildcard arm would silently accept any value). Leaving `v` free
/// lets a concrete arg in any later position bind it, after which a
/// different concrete value trips the `Lit ↔ Lit` mismatch as the sig
/// demands.
///
/// Name ↔ Lit invariant (issue Chelis-Lang/chelis#219, Option A): a
/// concrete-but-named slot (`Dim::Name("batch")`) accepts a concrete
/// literal (`Dim::Lit(2)`) at the call site without binding any
/// substitution. Names are preserved in diagnostics; they do not
/// impose a distinct-from-literal constraint. This eliminates the
/// asymmetry whereby `Var <-> Lit` was accepted at call sites but
/// `Name <-> Lit` was rejected, which blocked stdlib sigs like
/// `def f(x: tensor[batch, hidden, f32])` from being called with
/// concrete-shaped inputs (e.g. `f(to_tensor([[1.0, 2.0, 3.0]]))`).
/// The relaxation is narrow: distinct `Name <-> Name` and
/// distinct `Lit <-> Lit` continue to be rejected, and the
/// `Var <-> Lit` cross-position contract is unaffected.
pub fn unify_dim(d1: &Dim, d2: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
    let d1 = subst.apply_dim(d1);
    let d2 = subst.apply_dim(d2);

    match (&d1, &d2) {
        (Dim::Name(n1), Dim::Name(n2)) if n1 == n2 => Ok(()),
        (Dim::Lit(l1), Dim::Lit(l2)) if l1 == l2 => Ok(()),
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => Ok(()),
        // Issue #219 Option A: Name and Lit unify without binding any
        // substitution. The Name carries a label for diagnostics, the
        // Lit carries the concrete value; nothing flows into `subst`.
        (Dim::Name(_), Dim::Lit(_)) | (Dim::Lit(_), Dim::Name(_)) => Ok(()),
        (Dim::Var(v), _) => bind_dvar(*v, &d2, subst),
        (_, Dim::Var(v)) => bind_dvar(*v, &d1, subst),
        _ => Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!("dimension mismatch: {d1:?} vs {d2:?}"),
        }),
    }
}

fn bind_tvar(v: TypeVar, ty: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    if let Type::Var(v2) = ty
        && *v2 == v
    {
        return Ok(()); // same variable
    }
    if occurs_in(v, ty, subst) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!("infinite type: ?{} occurs in {ty}", v.0),
        });
    }
    subst.insert_type(v, ty.clone());
    Ok(())
}

fn bind_dvar(v: DimVar, dim: &Dim, subst: &mut Subst) -> Result<(), TypeError> {
    if let Dim::Var(v2) = dim
        && *v2 == v
    {
        return Ok(());
    }
    if occurs_in_dim(v, dim, subst) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!("infinite dimension: d{} occurs in {dim:?}", v.0),
        });
    }
    subst.insert_dim(v, dim.clone());
    Ok(())
}

/// Fully resolve a tensor's dim list under the current substitution: each
/// bound `Dim::Rank` expands to the run it stands for and every other dim is
/// `apply_dim`'d; an unbound (or terminal-rank) spread stays a `Dim::Rank`.
/// This is the per-shape form of `Subst::apply`'s tensor arm, used by the
/// rank-unification dispatch so a row shape arrives with only *unbound*
/// spreads interleaved among concrete dims.
fn resolve_shape(dims: &[Dim], subst: &Subst) -> Vec<Dim> {
    let mut out = Vec::with_capacity(dims.len());
    for d in dims {
        match d {
            Dim::Rank(r) => {
                for rd in subst.resolve_rvar(*r) {
                    out.push(subst.apply_dim(&rd));
                }
            }
            _ => out.push(subst.apply_dim(d)),
        }
    }
    out
}

/// chelis#339 named-axis expand: a signature whose result rows *introduce* an
/// axis name (present in a result tensor row but in no parameter row — the
/// expand-inserted axis) must not have that same name covered by one of its
/// rank-spread bindings at this unification. The symbolic collision check in
/// `check_named_expand_signature` can only see the visible row; a caller
/// whose spread-covered axes include the inserted name would otherwise
/// monomorphize to a result row carrying the same dim name twice with two
/// different extents — every later by-name axis lookup becomes ambiguous and
/// the declared type misstates the runtime shape. Hard error, never a guessed
/// alias (spec/04-type-system.md §4.5.3).
fn check_introduced_name_rank_collision(
    args: &[Type],
    ret: &Type,
    subst: &Subst,
) -> Result<(), TypeError> {
    fn collect(ty: &Type, names: &mut Vec<String>, rvars: &mut Vec<RankVar>) {
        match ty {
            Type::Tensor(dims, _) => {
                for d in dims {
                    match d {
                        Dim::Name(n) => names.push(n.clone()),
                        Dim::Rank(r) => rvars.push(*r),
                        _ => {}
                    }
                }
            }
            Type::Ref(inner) => collect(inner, names, rvars),
            Type::Fn(fn_args, fn_ret) => {
                for a in fn_args {
                    collect(a, names, rvars);
                }
                collect(fn_ret, names, rvars);
            }
            _ => {}
        }
    }

    let mut ret_names = Vec::new();
    let mut ret_rvars = Vec::new();
    collect(ret, &mut ret_names, &mut ret_rvars);
    if ret_names.is_empty() || ret_rvars.is_empty() {
        return Ok(());
    }
    let mut param_names = Vec::new();
    let mut param_rvars = Vec::new();
    for a in args {
        collect(a, &mut param_names, &mut param_rvars);
    }
    let introduced: Vec<&String> = ret_names
        .iter()
        .filter(|n| !param_names.contains(n))
        .collect();
    if introduced.is_empty() {
        return Ok(());
    }
    for rv in ret_rvars {
        for bound in resolve_shape(&[Dim::Rank(rv)], subst) {
            if let Dim::Name(n) = &bound
                && introduced.contains(&n)
            {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: format!(
                        "rank-spread monomorphization collides with an inserted axis name: the \
                         spread binds an operand axis named `{n}`, but the callee's result also \
                         introduces an axis named `{n}` (a named-axis expand insert); the \
                         monomorphized result would carry the same dim name twice with two \
                         different extents, making later by-name axis lookups ambiguous \
                         (spec/04-type-system.md \u{00a7}4.5.3). Rename the inserted axis or \
                         the operand axis."
                    ),
                });
            }
        }
    }
    Ok(())
}

/// The shared diagnostic for an undetermined split between two adjacent
/// rank spreads (the non-unitary case the decidable fragment excludes).
fn adjacent_spread_error() -> TypeError {
    TypeError {
        kind: TypeErrorKind::DimensionMismatch,
        message: "two adjacent rank spreads cannot be split against a concrete shape; the \
                  boundary between them is undetermined (outside the decidable fragment, \
                  spec/04-type-system.md \u{00a7}4.5.3)"
            .to_string(),
    }
}

/// Unify a *row* shape (≥1 spread, no two spreads adjacent) against a *ground*
/// shape (no spreads). Walks the row left to right: a fixed anchor matches the
/// ground positionally; a spread is bound to the run of ground dims up to the
/// next anchor, located by that anchor's name (the §4.2 name-preserving rule).
/// Each interior split anchor must be a `Dim::Name` present **exactly once** in
/// the remaining ground — a missing, ambiguous, or non-named anchor is a hard
/// error, never a guessed split. With one spread per gap and named splits,
/// every split point is forced, so this is unitary.
fn unify_row_against_ground(
    row: &[Dim],
    ground: &[Dim],
    subst: &mut Subst,
) -> Result<(), TypeError> {
    let n = ground.len();
    let mut gi = 0usize; // ground cursor
    let mut ri = 0usize; // row cursor
    while ri < row.len() {
        match &row[ri] {
            Dim::Rank(r) => {
                let rest = &row[ri + 1..];
                match rest.iter().position(|d| !matches!(d, Dim::Rank(_))) {
                    // The spread is immediately followed by a named anchor:
                    // locate that name in the remaining ground to fix the split.
                    Some(0) => {
                        let name = match &rest[0] {
                            Dim::Name(s) => s,
                            other => {
                                return Err(TypeError {
                                    kind: TypeErrorKind::DimensionMismatch,
                                    message: format!(
                                        "rank-spread anchor must be a named dimension to locate the \
                                         split, got {other:?} (spec/04-type-system.md \u{00a7}4.5.3)"
                                    ),
                                });
                            }
                        };
                        let hits: Vec<usize> = (gi..n)
                            .filter(|&j| matches!(&ground[j], Dim::Name(g) if g == name))
                            .collect();
                        match hits.as_slice() {
                            [split] => {
                                bind_rvar(*r, &ground[gi..*split], subst)?;
                                gi = *split;
                            }
                            [] => {
                                return Err(TypeError {
                                    kind: TypeErrorKind::DimensionMismatch,
                                    message: format!(
                                        "rank-spread operand carries no named `{name}` axis; a \
                                         fully-literal or differently-named operand cannot locate \
                                         the axis (spec/04-type-system.md \u{00a7}4.5.3)"
                                    ),
                                });
                            }
                            many => {
                                return Err(TypeError {
                                    kind: TypeErrorKind::DimensionMismatch,
                                    message: format!(
                                        "named axis `{name}` is ambiguous: it appears {} times in \
                                         the operand shape (spec/04-type-system.md \u{00a7}4.5.3)",
                                        many.len()
                                    ),
                                });
                            }
                        }
                        ri += 1;
                    }
                    // A second spread sits between this spread and the next
                    // anchor (or runs to the end): the split between two
                    // adjacent spreads is undetermined — the non-unitary case.
                    Some(_) => {
                        return Err(adjacent_spread_error());
                    }
                    None if rest.is_empty() => {
                        // Trailing spread: absorb the rest of the ground.
                        bind_rvar(*r, &ground[gi..n], subst)?;
                        gi = n;
                        ri += 1;
                    }
                    None => {
                        // `rest` is one or more further spreads with no anchor.
                        return Err(adjacent_spread_error());
                    }
                }
            }
            anchor => {
                if gi >= n {
                    return Err(TypeError {
                        kind: TypeErrorKind::DimensionMismatch,
                        message: format!(
                            "tensor rank too small: the rank-spread shape has more fixed anchor \
                             dimensions than the operand's {n} dimensions"
                        ),
                    });
                }
                unify_dim(anchor, &ground[gi], subst)?;
                gi += 1;
                ri += 1;
            }
        }
    }
    if gi != n {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!(
                "tensor shape mismatch: {} operand dimension(s) left unmatched after the \
                 rank-spread shape was consumed",
                n - gi
            ),
        });
    }
    Ok(())
}

/// Unify two row shapes that *both* carry spreads. Restricted to
/// structurally-identical rows (same length, same per-position kind): spreads
/// alias pairwise, named anchors unify by name. Two rows with a *different*
/// anchor structure are associative/string unification (not unitary) and are
/// rejected — the body of a rank-poly def only ever produces an identical-row
/// self-check, and call sites are row-vs-ground.
fn unify_row_against_row(d1: &[Dim], d2: &[Dim], subst: &mut Subst) -> Result<(), TypeError> {
    if d1.len() != d2.len() {
        return Err(TypeError {
            kind: TypeErrorKind::DimensionMismatch,
            message: format!(
                "cannot unify rank-spread shapes with differing anchor structure ({} vs {} slots); \
                 two spreads with an undetermined split are outside the decidable fragment \
                 (spec/04-type-system.md \u{00a7}4.5.3)",
                d1.len(),
                d2.len()
            ),
        });
    }
    for (a, b) in d1.iter().zip(d2.iter()) {
        match (a, b) {
            (Dim::Rank(r1), Dim::Rank(r2)) => {
                if r1 != r2 {
                    subst.insert_rank(*r1, vec![Dim::Rank(*r2)]);
                }
            }
            (Dim::Rank(_), _) | (_, Dim::Rank(_)) => {
                return Err(TypeError {
                    kind: TypeErrorKind::DimensionMismatch,
                    message: "cannot unify a rank spread against a concrete dimension; the two \
                              rank-spread shapes have differing structure \
                              (spec/04-type-system.md \u{00a7}4.5.3)"
                        .to_string(),
                });
            }
            _ => unify_dim(a, b, subst)?,
        }
    }
    Ok(())
}

/// Bind a rank variable to a concrete shape *run*, with an occurs/nesting
/// check: the bound run must be `Dim::Rank`-free. A run containing any spread
/// would either make the rank infinite (its own occurrence) or smuggle a
/// second spread into a single spread's binding (the non-unitary case).
/// Rank-to-rank aliasing is handled separately by `unify_row_against_row`.
fn bind_rvar(r: RankVar, dims: &[Dim], subst: &mut Subst) -> Result<(), TypeError> {
    if let Some(bad) = dims.iter().find(|d| matches!(d, Dim::Rank(_))) {
        return Err(TypeError {
            kind: TypeErrorKind::OccursCheck,
            message: format!(
                "rank variable r{} cannot bind to a shape containing a rank spread ({bad:?})",
                r.0
            ),
        });
    }
    subst.insert_rank(r, dims.to_vec());
    Ok(())
}

fn occurs_in(v: TypeVar, ty: &Type, subst: &Subst) -> bool {
    let ty = subst.apply(ty);
    match &ty {
        Type::Var(v2) => *v2 == v,
        Type::Fn(args, ret) => {
            args.iter().any(|a| occurs_in(v, a, subst)) || occurs_in(v, ret, subst)
        }
        Type::Ref(inner) => occurs_in(v, inner, subst),
        Type::Tensor(_, prec) => match prec {
            // Dims do not carry type vars (they have their own DimVar lane).
            // Precision slot CAN carry a TypeVar (WS-A5 precision polymorphism);
            // include it in the occurs check so an attempt to unify
            // ?v with `tensor[..., ?v]` is caught as an infinite type.
            TensorPrec::Concrete(_) => false,
            TensorPrec::Var(v2) => *v2 == v,
        },
        Type::Adt(_, args) => args.iter().any(|a| occurs_in(v, a, subst)),
        Type::Tuple(ts) => ts.iter().any(|t| occurs_in(v, t, subst)),
        Type::Prim(_) | Type::Unit | Type::Error => false,
    }
}

fn occurs_in_dim(v: DimVar, dim: &Dim, subst: &Subst) -> bool {
    let dim = subst.apply_dim(dim);
    matches!(dim, Dim::Var(v2) if v2 == v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var_gen() -> VarGen {
        VarGen::default()
    }

    #[test]
    fn unify_same_prim() {
        let mut s = Subst::new();
        assert!(unify(&Type::Prim(Prim::F32), &Type::Prim(Prim::F32), &mut s).is_ok());
    }

    #[test]
    fn unify_different_prim_fails() {
        let mut s = Subst::new();
        let err = unify(&Type::Prim(Prim::F32), &Type::Prim(Prim::Bf16), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_tvar_binds() {
        let mut g = var_gen();
        let v = g.fresh_type();
        let mut s = Subst::new();
        assert!(unify(&v, &Type::Prim(Prim::F32), &mut s).is_ok());
        assert_eq!(s.apply(&v), Type::Prim(Prim::F32));
    }

    #[test]
    fn unify_fn_types() {
        let mut s = Subst::new();
        let f1 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        let f2 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        assert!(unify(&f1, &f2, &mut s).is_ok());
    }

    #[test]
    fn unify_fn_arity_mismatch() {
        let mut s = Subst::new();
        let f1 = Type::Fn(vec![Type::Prim(Prim::F32)], Box::new(Type::Prim(Prim::F32)));
        let f2 = Type::Fn(
            vec![Type::Prim(Prim::F32), Type::Prim(Prim::F32)],
            Box::new(Type::Prim(Prim::F32)),
        );
        let err = unify(&f1, &f2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::ArityMismatch));
    }

    fn tprec(p: Prim) -> TensorPrec {
        TensorPrec::Concrete(p)
    }

    #[test]
    fn unify_tensor_same_dims() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(
            vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
            tprec(Prim::F32),
        );
        let t2 = Type::Tensor(
            vec![Dim::Name("batch".into()), Dim::Name("hidden".into())],
            tprec(Prim::F32),
        );
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn unify_tensor_dim_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("seq".into())], tprec(Prim::F32));
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_tensor_precision_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_dim_var_binds() {
        let mut g = var_gen();
        let dv = g.fresh_dvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Var(dv)], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        assert!(unify(&t1, &t2, &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Name("batch".into()));
    }

    // chelis#258 Tier-2 rank polymorphism: the rank-variable unification arm.

    #[test]
    fn unify_rank_binds_whole_shape() {
        let mut g = var_gen();
        let rv = g.fresh_rvar();
        let mut s = Subst::new();
        let rank_tensor = Type::Tensor(vec![Dim::Rank(rv)], tprec(Prim::F32));
        let concrete = Type::Tensor(
            vec![Dim::Name("a".into()), Dim::Name("b".into())],
            tprec(Prim::F32),
        );
        assert!(unify(&rank_tensor, &concrete, &mut s).is_ok());
        // The rank var now stands for the whole 2-dim shape; applying expands it.
        assert_eq!(s.apply(&rank_tensor), concrete);
    }

    #[test]
    fn unify_rank_to_rank_aliases() {
        let mut g = var_gen();
        let r1 = g.fresh_rvar();
        let r2 = g.fresh_rvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Rank(r1)], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Rank(r2)], tprec(Prim::F32));
        assert!(unify(&t1, &t2, &mut s).is_ok());
        // Binding one to a concrete shape resolves both (alias chased).
        let concrete = Type::Tensor(vec![Dim::Name("n".into())], tprec(Prim::F32));
        assert!(unify(&t2, &concrete, &mut s).is_ok());
        assert_eq!(s.apply(&t1), concrete);
    }

    #[test]
    fn unify_rank_precision_still_checked() {
        // The precision slot unifies independently of the rank binding.
        let mut g = var_gen();
        let rv = g.fresh_rvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Rank(rv)], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("a".into())], tprec(Prim::Int32));
        assert!(
            unify(&t1, &t2, &mut s).is_err(),
            "f32 vs int32 precision must fail even with a rank var"
        );
    }

    // chelis#258 Tier-3 rank polymorphism: name-preserving multi-spread
    // unification (`tensor[..pre, seq, ..post]`).

    fn name(n: &str) -> Dim {
        Dim::Name(n.into())
    }

    #[test]
    fn unify_rank_middle_split_preserves_names() {
        // `tensor[..pre, seq, ..post]` vs `tensor[batch, seq, hidden]` splits
        // at the unique named anchor: pre:=[batch], post:=[hidden]. The output
        // `tensor[..pre, ..post]` then carries the surviving names through.
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(
            vec![name("batch"), name("seq"), name("hidden")],
            tprec(Prim::F32),
        );
        assert!(unify(&row, &ground, &mut s).is_ok());
        let out = Type::Tensor(vec![Dim::Rank(pre), Dim::Rank(post)], tprec(Prim::F32));
        assert_eq!(
            s.apply(&out),
            Type::Tensor(vec![name("batch"), name("hidden")], tprec(Prim::F32)),
            "output drops `seq`, keeps batch/hidden by name"
        );
    }

    #[test]
    fn unify_rank_multi_anchor_split() {
        // Two named anchors, three spreads: `[..a, seq, ..b, head, ..c]`.
        let mut g = var_gen();
        let (a, b, c) = (g.fresh_rvar(), g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![
                Dim::Rank(a),
                name("seq"),
                Dim::Rank(b),
                name("head"),
                Dim::Rank(c),
            ],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(
            vec![
                name("batch"),
                name("seq"),
                name("kv"),
                name("head"),
                name("dim"),
            ],
            tprec(Prim::F32),
        );
        assert!(unify(&row, &ground, &mut s).is_ok());
        let out = Type::Tensor(
            vec![Dim::Rank(a), Dim::Rank(b), Dim::Rank(c)],
            tprec(Prim::F32),
        );
        assert_eq!(
            s.apply(&out),
            Type::Tensor(
                vec![name("batch"), name("kv"), name("dim")],
                tprec(Prim::F32)
            ),
        );
    }

    #[test]
    fn unify_rank_anchor_absent_rejected() {
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(vec![name("batch"), name("hidden")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
        assert!(
            err.message.contains("no named `seq` axis"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unify_rank_anchor_ambiguous_rejected() {
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(vec![name("seq"), name("x"), name("seq")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(err.message.contains("ambiguous"), "{}", err.message);
    }

    #[test]
    fn unify_rank_literal_operand_rejected() {
        // Name↔Lit hazard: a fully-literal operand carries no name to locate
        // the anchor — hard reject, never a guessed reduction.
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        let ground = Type::Tensor(
            vec![Dim::Lit(2), Dim::Lit(768), Dim::Lit(4)],
            tprec(Prim::F32),
        );
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(
            err.message.contains("no named `seq` axis"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unify_rank_too_small_rejected() {
        // Two leading anchors but a rank-1 operand: not enough dims.
        let mut g = var_gen();
        let r = g.fresh_rvar();
        let mut s = Subst::new();
        let row = Type::Tensor(vec![name("a"), name("b"), Dim::Rank(r)], tprec(Prim::F32));
        let ground = Type::Tensor(vec![name("a")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(err.message.contains("rank too small"), "{}", err.message);
    }

    #[test]
    fn unify_two_adjacent_spreads_rejected() {
        // Two adjacent spreads split against a ground is undetermined.
        let mut g = var_gen();
        let (a, b) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(vec![Dim::Rank(a), Dim::Rank(b)], tprec(Prim::F32));
        let ground = Type::Tensor(vec![name("x"), name("y")], tprec(Prim::F32));
        let err = unify(&row, &ground, &mut s).unwrap_err();
        assert!(
            err.message.contains("two adjacent rank spreads"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unify_identical_rows_ok() {
        // The body output `[..pre, ..post]` vs the declared return
        // `[..pre, ..post]` (same vars) is the identical-row self-check.
        let mut g = var_gen();
        let (pre, post) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let row = Type::Tensor(
            vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)],
            tprec(Prim::F32),
        );
        assert!(unify(&row, &row.clone(), &mut s).is_ok());
    }

    #[test]
    fn unify_distinct_anchor_rows_rejected() {
        let mut g = var_gen();
        let (a, b) = (g.fresh_rvar(), g.fresh_rvar());
        let mut s = Subst::new();
        let r1 = Type::Tensor(
            vec![Dim::Rank(a), name("seq"), Dim::Rank(b)],
            tprec(Prim::F32),
        );
        let r2 = Type::Tensor(
            vec![Dim::Rank(a), name("head"), Dim::Rank(b)],
            tprec(Prim::F32),
        );
        assert!(unify(&r1, &r2, &mut s).is_err());
    }

    #[test]
    fn unify_trailing_anchor_by_name() {
        // `[..pre, seq]` vs `[a, b, seq]`: pre:=[a,b], seq at the tail.
        let mut g = var_gen();
        let pre = g.fresh_rvar();
        let mut s = Subst::new();
        let row = Type::Tensor(vec![Dim::Rank(pre), name("seq")], tprec(Prim::F32));
        let ground = Type::Tensor(vec![name("a"), name("b"), name("seq")], tprec(Prim::F32));
        assert!(unify(&row, &ground, &mut s).is_ok());
        let out = Type::Tensor(vec![Dim::Rank(pre)], tprec(Prim::F32));
        assert_eq!(
            s.apply(&out),
            Type::Tensor(vec![name("a"), name("b")], tprec(Prim::F32)),
        );
    }

    #[test]
    fn unify_name_with_lit_accepts() {
        // Issue Chelis-Lang/chelis#219 Option A: a Name slot on the
        // left and a concrete Lit on the right unify without binding
        // any substitution.
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Name("batch".into()), &Dim::Lit(2), &mut s).is_ok());
        // Subst stays untouched — the Name carries diagnostics, the
        // Lit carries the value, nothing flows in.
        assert_eq!(s.dims_len(), 0);
    }

    #[test]
    fn unify_name_with_lit_accepts_symmetric() {
        // Issue #219 Option A: the symmetric direction (Lit on left,
        // Name on right) also unifies.
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Lit(3), &Dim::Name("hidden".into()), &mut s).is_ok());
        assert_eq!(s.dims_len(), 0);
    }

    #[test]
    fn unify_name_distinct_names_still_errors() {
        // Issue #219 regression-lock: the Name <-> Lit relaxation
        // must not bleed into Name <-> Name. Distinct symbolic names
        // continue to surface as DimensionMismatch (this is the
        // dim-polymorphism rigidity rule from §4.4).
        let mut s = Subst::new();
        let err =
            unify_dim(&Dim::Name("batch".into()), &Dim::Name("seq".into()), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_lit_distinct_still_errors() {
        // Issue #219 regression-lock: distinct Lit <-> Lit still
        // errors. The permissive arm is narrowly Name <-> Lit.
        let mut s = Subst::new();
        let err = unify_dim(&Dim::Lit(2), &Dim::Lit(3), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_dim_transitive_binding_resolves() {
        let mut g = var_gen();
        let d0 = g.fresh_dvar();
        let d1 = g.fresh_dvar();
        let mut s = Subst::new();
        assert!(unify_dim(&Dim::Var(d0), &Dim::Var(d1), &mut s).is_ok());
        assert!(unify_dim(&Dim::Var(d1), &Dim::Name("batch".into()), &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(d0)), Dim::Name("batch".into()));
    }

    #[test]
    fn unify_wildcard_matches_anything() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn unify_wildcard_with_var_leaves_var_free() {
        // chelis#143 invariant: `unify_dim(Wildcard, Var(v))` must
        // succeed without binding `v`. Binding `v := Wildcard` would
        // freeze the dim var; a later concrete arg in the same sig
        // couldn't then constrain `v` because `apply_dim` would
        // resolve `Var(v) → Wildcard` and the permissive Wildcard arm
        // would silently accept any concrete value.
        //
        // The current implementation satisfies this because the
        // `(Wildcard, _) | (_, Wildcard) => Ok(())` arm matches before
        // the `(Var(_), _) => bind_dvar(...)` arm (Rust `match` is
        // first-match-wins) and returns `Ok(())` without touching the
        // substitution. This test pins the property in case a future
        // refactor reorders the arms or adds an explicit
        // `(Wildcard, Var)` arm that does bind.
        let mut g = var_gen();
        let dv = g.fresh_dvar();
        let mut s = Subst::new();

        // Wildcard ↔ Var: succeeds, var remains free.
        assert!(unify_dim(&Dim::Wildcard, &Dim::Var(dv), &mut s).is_ok());
        assert_eq!(
            s.apply_dim(&Dim::Var(dv)),
            Dim::Var(dv),
            "Wildcard ↔ Var must not bind the var to Wildcard",
        );

        // Symmetric direction: Var ↔ Wildcard also leaves the var
        // free.
        let dv2 = g.fresh_dvar();
        assert!(unify_dim(&Dim::Var(dv2), &Dim::Wildcard, &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv2)), Dim::Var(dv2));

        // A subsequent concrete unification with the still-free var
        // binds it to the concrete dim.
        assert!(unify_dim(&Dim::Var(dv), &Dim::Lit(2), &mut s).is_ok());
        assert_eq!(s.apply_dim(&Dim::Var(dv)), Dim::Lit(2));

        // And once the var is concretely bound, a conflicting concrete
        // value trips DimensionMismatch — the cross-position contract
        // the sig promised is now enforced end to end.
        let err = unify_dim(&Dim::Var(dv), &Dim::Lit(3), &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_tensor_rank_mismatch() {
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("a".into())], tprec(Prim::F32));
        let t2 = Type::Tensor(
            vec![Dim::Name("a".into()), Dim::Name("b".into())],
            tprec(Prim::F32),
        );
        let err = unify(&t1, &t2, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::DimensionMismatch));
    }

    #[test]
    fn unify_dim_cannot_recover_lit_from_two_var_tensors() {
        // chelis#158 cause partition: when two tensors both arrive at
        // a shared-dim sig with `Var(d)` (or `Wildcard`) dims and
        // never with a concrete `Lit(_)` source, `unify_dim` cannot
        // manufacture a mismatch. This is the unify-layer half of
        // chelis#158's root cause: `to_tensor` and `pad_sequences_to`
        // emit shape-erased tensors that reach the sig with no `Lit`
        // in any position; unify_dim correctly succeeds on every pair
        // and the sig's shared dim var stays free.
        //
        // The fix for chelis#158 must inject a `Lit` somewhere
        // UPSTREAM of unify (most likely in the desugarer when
        // `to_tensor` is applied to a statically-known list literal).
        // This test pins the unify-layer contract so any future
        // "unify_dim should be smarter" proposal can be measured
        // against the constraint that follows: from Var/Wildcard
        // inputs alone, no concrete dim can be inferred.
        let mut g = var_gen();
        let d_sig = g.fresh_dvar();

        // Two sig-instantiated tensor params that share `d_sig`.
        let sig_arg1 = Type::Tensor(vec![Dim::Var(d_sig)], tprec(Prim::F32));
        let sig_arg2 = Type::Tensor(vec![Dim::Var(d_sig)], tprec(Prim::F32));

        // Two caller tensors, both shape-erased to fresh `Var(d_caller_*)`
        // (modeling two independent to_tensor outputs).
        let d_caller_a = g.fresh_dvar();
        let d_caller_b = g.fresh_dvar();
        let caller_a = Type::Tensor(vec![Dim::Var(d_caller_a)], tprec(Prim::F32));
        let caller_b = Type::Tensor(vec![Dim::Var(d_caller_b)], tprec(Prim::F32));

        let mut s = Subst::new();
        // Simulate `pair_id(caller_a, caller_b)` where pair_id's sig
        // is `&tensor[d_sig, f32] -> &tensor[d_sig, f32] -> ...`.
        assert!(unify(&sig_arg1, &caller_a, &mut s).is_ok());
        assert!(unify(&sig_arg2, &caller_b, &mut s).is_ok());

        // After both unifications, `d_sig` is bound to (some chain
        // of) dim vars but never to a concrete `Lit`.
        let resolved = s.apply_dim(&Dim::Var(d_sig));
        assert!(
            matches!(resolved, Dim::Var(_) | Dim::Wildcard),
            "sig dim var resolved to a concrete value despite no \
             Lit source in either caller; this would mean unify_dim \
             manufactured a constraint from nothing, which is the \
             wrong place to fix chelis#158. Got: {resolved:?}"
        );

        // Symmetric: replacing one caller's Var with Wildcard (the
        // actual shape-erased symptom from to_tensor) also leaves
        // d_sig unable to bind to a Lit. No mismatch is producible.
        let mut s2 = Subst::new();
        let caller_a_wild = Type::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
        let caller_b_wild = Type::Tensor(vec![Dim::Wildcard], tprec(Prim::F32));
        assert!(unify(&sig_arg1, &caller_a_wild, &mut s2).is_ok());
        assert!(unify(&sig_arg2, &caller_b_wild, &mut s2).is_ok());
        let resolved2 = s2.apply_dim(&Dim::Var(d_sig));
        assert!(
            matches!(resolved2, Dim::Var(_) | Dim::Wildcard),
            "with Wildcard caller dims, d_sig also cannot resolve to \
             a concrete Lit. Got: {resolved2:?}. The fix for \
             chelis#158 must introduce a Lit at the desugar/builder \
             layer."
        );
    }

    // === WS-A5 precision polymorphism unification ===

    #[test]
    fn unify_tensor_prec_var_binds_to_concrete() {
        // tensor[batch, ?p] vs tensor[batch, f32]
        // ?p must bind to f32 in the substitution.
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        unify(&t1, &t2, &mut s).expect("var precision should bind");
        assert_eq!(s.apply(&Type::Var(pv)), Type::Prim(Prim::F32));
        // After substitution, the var-precision tensor reads as f32 too.
        match s.apply(&t1) {
            Type::Tensor(_, TensorPrec::Concrete(Prim::F32)) => {}
            other => panic!("apply should resolve precision var to f32, got {other}"),
        }
    }

    #[test]
    fn unify_tensor_prec_concrete_binds_var_other_side() {
        // Symmetric: concrete on left, var on right.
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        unify(&t1, &t2, &mut s).expect("var precision should bind");
        assert_eq!(s.apply(&Type::Var(pv)), Type::Prim(Prim::Bf16));
    }

    #[test]
    fn unify_two_tensor_prec_vars_link() {
        // Two precision vars unify by linking; the link survives later
        // binding through either var.
        let mut g = var_gen();
        let p1 = g.fresh_tvar();
        let p2 = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(p1));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(p2));
        unify(&t1, &t2, &mut s).expect("two prec vars should link");
        // Now bind one of them to a concrete prim and confirm the other
        // resolves to the same prim through the link.
        let t3 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F64));
        unify(&t1, &t3, &mut s).expect("link + bind should work");
        assert_eq!(s.apply(&Type::Var(p1)), Type::Prim(Prim::F64));
        assert_eq!(s.apply(&Type::Var(p2)), Type::Prim(Prim::F64));
    }

    #[test]
    fn unify_tensor_prec_var_then_conflicting_concrete_errors() {
        // Once ?p is bound to f32, unifying tensor[..., ?p] with
        // tensor[..., bf16] must error with PrecisionMismatch.
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t_var = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        let t_f32 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::F32));
        let t_bf16 = Type::Tensor(vec![Dim::Name("batch".into())], tprec(Prim::Bf16));
        unify(&t_var, &t_f32, &mut s).expect("first call binds prec var to f32");
        let err = unify(&t_var, &t_bf16, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::PrecisionMismatch));
    }

    #[test]
    fn unify_same_prec_var_with_itself_succeeds() {
        let mut g = var_gen();
        let pv = g.fresh_tvar();
        let mut s = Subst::new();
        let t1 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        let t2 = Type::Tensor(vec![Dim::Name("batch".into())], TensorPrec::Var(pv));
        unify(&t1, &t2, &mut s).expect("self-unify must succeed");
    }

    #[test]
    fn apply_tensor_prec_resolves_through_chain() {
        // Build a substitution that puts ?p1 -> ?p2 -> Prim::F32.
        let mut g = var_gen();
        let p1 = g.fresh_tvar();
        let p2 = g.fresh_tvar();
        let mut s = Subst::new();
        unify(&Type::Var(p1), &Type::Var(p2), &mut s).unwrap();
        unify(&Type::Var(p2), &Type::Prim(Prim::F32), &mut s).unwrap();
        // apply_tensor_prec on either var must collapse to Concrete(F32).
        assert_eq!(
            s.apply_tensor_prec(&TensorPrec::Var(p1)),
            TensorPrec::Concrete(Prim::F32)
        );
        assert_eq!(
            s.apply_tensor_prec(&TensorPrec::Var(p2)),
            TensorPrec::Concrete(Prim::F32)
        );
    }

    #[test]
    fn unify_adt_same() {
        let mut s = Subst::new();
        let a1 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        let a2 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        assert!(unify(&a1, &a2, &mut s).is_ok());
    }

    #[test]
    fn unify_adt_name_mismatch() {
        let mut s = Subst::new();
        let a1 = Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]);
        let a2 = Type::Adt("Result".into(), vec![Type::Prim(Prim::F32)]);
        assert!(unify(&a1, &a2, &mut s).is_err());
    }

    #[test]
    fn occurs_check_prevents_infinite_type() {
        let mut g = var_gen();
        let v = g.fresh_tvar();
        let mut s = Subst::new();
        // Try to unify ?0 with List(?0) — should fail with occurs check
        let list_v = Type::Adt("List".into(), vec![Type::Var(v)]);
        let err = unify(&Type::Var(v), &list_v, &mut s).unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::OccursCheck));
    }

    #[test]
    fn error_type_unifies_with_anything() {
        // Per WS-A5 RT-3a F1 escalation: the permissive rule is retained
        // here so a single upstream error does not fan out a cascade of
        // secondary diagnostics from one root cause; the silent
        // passthrough at the def-body vs declared-sig boundary is
        // closed at the call site in `infer.rs`, not by tightening the
        // unification rule.
        let mut s = Subst::new();
        assert!(unify(&Type::Error, &Type::Prim(Prim::F32), &mut s).is_ok());
        assert!(unify(&Type::Prim(Prim::F32), &Type::Error, &mut s).is_ok());
    }

    #[test]
    fn unify_tuple_types() {
        let mut s = Subst::new();
        let t1 = Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::Int32)]);
        let t2 = Type::Tuple(vec![Type::Prim(Prim::F32), Type::Prim(Prim::Int32)]);
        assert!(unify(&t1, &t2, &mut s).is_ok());
    }

    #[test]
    fn transitive_binding() {
        let mut g = var_gen();
        let a = g.fresh_tvar();
        let b = g.fresh_tvar();
        let mut s = Subst::new();
        // a = b, b = f32 → a = f32
        unify(&Type::Var(a), &Type::Var(b), &mut s).unwrap();
        unify(&Type::Var(b), &Type::Prim(Prim::F32), &mut s).unwrap();
        assert_eq!(s.apply(&Type::Var(a)), Type::Prim(Prim::F32));
    }
}
