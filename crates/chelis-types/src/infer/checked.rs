//! Checked-program construction, metadata ownership, and totality finalization.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use crate::context::LibraryProofId;

#[derive(Clone)]
pub(super) struct DeclaredSigMetadata {
    pub(super) param_types: Vec<deep::Expr>,
    pub(super) binders: UnordSet<String>,
    /// Exact dtype-family capabilities authored on this signature. Keep the
    /// decode result rather than defaulting malformed metadata to no bounds:
    /// declaration collection owns the diagnostic, while structural users
    /// may proceed only through `Ok`.
    pub(super) dtype_bounds:
        Result<UnordMap<String, chelis_deep::DtypeFamily>, chelis_deep::DtypeBoundsError>,
}

/// Explicit annotation-time declaration context. The declared signature map
/// belongs to one annotation unit; `current_type_binders` is narrowed to the
/// `def` whose children are being annotated and is passed through every
/// recursive annotation call.
#[derive(Clone, Copy)]
pub(super) struct AnnotationResolutionContext<'a> {
    declared_signatures: &'a UnordMap<String, DeclaredSigMetadata>,
}

impl<'a> AnnotationResolutionContext<'a> {
    pub(super) fn root(declared_signatures: &'a UnordMap<String, DeclaredSigMetadata>) -> Self {
        Self {
            declared_signatures,
        }
    }

    pub(super) fn declared_signature(self, name: &str) -> Option<&'a DeclaredSigMetadata> {
        self.declared_signatures.get(name)
    }
}

/// Semantic role of one tagged Deep node's child in checker-owned type
/// stamping. This is deliberately distinct from the child's syntactic tag:
/// a `lit` is a runtime expression under `app`, but the same shape is selector
/// syntax in `tuple-get`, `grad`, or `vmap`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ChildStampRole {
    /// Traversed by ordinary expression inference.
    RuntimeExpr,
    /// Compiler/source syntax which is preserved verbatim.
    Syntax,
    /// A field, axis, projection, or transform selector.
    Selector,
    /// Handler payload syntax whose literal-form contract is owned by
    /// `chelis-effects`, not expression inference.
    EffectHandler,
    /// A declaration, parameter, or binding name.
    Binder,
    /// Type/dimension syntax resolved by its owning type consumer.
    Type,
    /// Traversed by a dedicated inference owner rather than `infer_expr` on
    /// the structural parent (module declarations, patterns, helper nodes,
    /// and synthesized pipe stages).
    ExplicitInferenceBypass,
}

/// Exhaustive child-role table for the canonical closed Deep vocabulary.
///
/// Returning `None` is a loud version-skew signal, never permission to treat
/// an unknown child as a runtime expression. The completeness test below
/// iterates `chelis_deep::validate::VALID_TAGS`, the grammar's single source
/// of truth, so adding a tag requires an explicit ownership decision here.
pub(super) fn child_stamp_role(tag: DeepTag, index: usize, _arity: usize) -> ChildStampRole {
    use ChildStampRole::{
        Binder, EffectHandler, ExplicitInferenceBypass, RuntimeExpr, Selector, Syntax, Type,
    };

    match tag {
        // Module wrappers are not inferred as one expression. Their
        // declarations each own a separate inference epoch.
        DeepTag::Module => {
            if index == 0 {
                Binder
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Import | DeepTag::ImportAll | DeepTag::Export => Syntax,

        // Declarations.
        DeepTag::Def => {
            if index == 0 {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Defsig => {
            if index == 0 {
                Binder
            } else {
                Type
            }
        }
        DeepTag::Deftype | DeepTag::Typealias => {
            if index == 0 {
                Binder
            } else if index == 1 {
                Syntax
            } else {
                Type
            }
        }
        DeepTag::Variant | DeepTag::Field => {
            if index == 0 {
                Binder
            } else {
                Type
            }
        }
        DeepTag::Defdim => Binder,

        // Expressions and their structural helper positions.
        DeepTag::Fn => {
            if index == 0 {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::App
        | DeepTag::If
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::Par
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Copy
        | DeepTag::Borrow
        | DeepTag::Unquote
        | DeepTag::Splice => RuntimeExpr,
        DeepTag::HandleEffect => {
            if index == 0 {
                EffectHandler
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Let => {
            if index == 0 {
                ExplicitInferenceBypass
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Match => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Arm => {
            if index == 0 {
                ExplicitInferenceBypass
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Var | DeepTag::Lit => Syntax,
        DeepTag::Record => {
            if index == 0 {
                Type
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::Access => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::Pipe => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::TupleGet => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::RecordUpdate => {
            if index == 0 {
                RuntimeExpr
            } else {
                ExplicitInferenceBypass
            }
        }

        // Pattern nodes are consumed by the primary pattern traversal.
        DeepTag::PatVar => Binder,
        DeepTag::PatLit => Syntax,
        DeepTag::PatCtor | DeepTag::PatRecord => {
            if index == 0 {
                Selector
            } else {
                ExplicitInferenceBypass
            }
        }
        DeepTag::PatTuple => ExplicitInferenceBypass,
        DeepTag::PatWild => Syntax,
        DeepTag::PatAs => {
            if index == 0 {
                Binder
            } else {
                ExplicitInferenceBypass
            }
        }

        // Type and dimension nodes are owned recursively by DeepTypeResolver,
        // never by expression annotation.
        DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TRef
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank => Type,

        // Transform-specific selector/type positions.
        DeepTag::Grad | DeepTag::Vmap => {
            if index == 0 {
                RuntimeExpr
            } else {
                Selector
            }
        }
        DeepTag::Cast => {
            if index == 0 {
                RuntimeExpr
            } else {
                Type
            }
        }

        // Quoted children and effect/resource payloads are syntax data.
        DeepTag::Quote | DeepTag::Effects | DeepTag::Resource => Syntax,

        // Structural helper nodes. `kv` is also used by pattern records, so
        // its value/pattern slot is an explicit owning traversal in both
        // contexts; canonical runtime values still record their normal stamp.
        DeepTag::Params => Binder,
        DeepTag::Bind => {
            if index.is_multiple_of(2) {
                Binder
            } else {
                RuntimeExpr
            }
        }
        DeepTag::Kv => {
            if index == 0 {
                Selector
            } else {
                ExplicitInferenceBypass
            }
        }
    }
}

/// One checker operation's typed inference result. The product is private,
/// session-local, and never serialized: annotation consumes it immediately
/// after the owning inference traversal completes.
#[derive(Default)]
pub(super) struct InferenceProduct {
    pub(super) typed_nodes: usize,
    pub(super) total_nodes: usize,
    pub(super) next_epoch: u64,
    pub(super) active_epoch: Option<TypeStampEpoch>,
    pub(super) owner_types: UnordMap<usize, FinalOwnerType>,
    pub(super) type_headers: TypeResolutionEnv,
    pub(super) adt_registry: AdtRegistry,
    pub(super) function_inference_plan: FunctionInferencePlan,
    /// The canonical declaration-reference graph built for this inference
    /// run. Scheduling and initialization-cycle diagnostics consume this same
    /// instance so the two policies cannot drift or repeat the lexical walk.
    pub(super) top_level_references: TopLevelReferenceGraph,
    shape_lambda_tvars: UnordSet<TypeVar>,
    deferred_type_derivations: Vec<DeferredTypeDerivation>,
    next_deferred_shape_id: u64,
    deferred_shape_checks: Vec<DeferredShapeCheck>,
}

#[derive(Clone)]
pub(super) enum DeferredShapeRule {
    Matmul,
    Reduction {
        name: String,
    },
    Expand {
        /// `expand` or `insert`. The replay must reach the same route arm the
        /// original call did, and the two differ in their result forms.
        builtin: &'static str,
        axis_is_dim_name: bool,
        size_class: SizeClass,
        env: Box<Env>,
    },
    LayerNorm,
    Conv2d,
    ScatterElements {
        list: deep::List,
    },
}

#[derive(Clone)]
pub(super) struct DeferredShapeCheck {
    id: u64,
    rule: DeferredShapeRule,
    arg_exprs: Vec<deep::Expr>,
    arg_tys: Vec<Type>,
    result_ty: Type,
}

#[derive(Clone)]
enum DeferredTypeDerivation {
    TupleProjection {
        source: Type,
        index: usize,
        projected: Type,
    },
}

pub(super) struct TypeStampEpoch {
    id: u64,
    owners: UnordMap<usize, StampRequirement>,
    writes: UnordMap<usize, Vec<OwnerTypeWrite>>,
    /// Transitional per-node bridge identities. A `Node::to_list` reader
    /// clones children, so pointer-keyed owner writes from that temporary
    /// view must resolve back to the original stamped child registered by
    /// `begin_root`. The whole-tree normalization boundary is gone; this map
    /// remains only until the individual inference readers consume Nodes
    /// directly.
    bridge_aliases: UnordMap<usize, usize>,
}

#[derive(Clone, Copy)]
pub(super) struct StampRequirement {
    role: &'static str,
    stamp_required: bool,
}

pub(super) struct OwnerTypeWrite {
    ty: Type,
    source: &'static str,
}

pub(super) struct FinalOwnerType {
    epoch: u64,
    ty: Type,
}

impl InferenceProduct {
    pub(super) fn stats(&self) -> InferStats {
        InferStats {
            typed_nodes: self.typed_nodes,
            total_nodes: self.total_nodes,
        }
    }

    pub(super) fn begin_root(&mut self, root: &deep::Expr) {
        assert!(
            self.active_epoch.is_none(),
            "type-stamp epochs must not overlap"
        );
        let id = self.next_epoch;
        self.next_epoch += 1;
        let mut epoch = TypeStampEpoch {
            id,
            owners: UnordMap::new(),
            writes: UnordMap::new(),
            bridge_aliases: UnordMap::new(),
        };
        register_annotation_owners(root, &mut epoch);
        self.active_epoch = Some(epoch);
    }

    pub(super) fn deferred_shape_checkpoint(&self) -> u64 {
        self.next_deferred_shape_id
    }

    pub(super) fn note_shape_lambda_param(&mut self, ty: &Type) {
        // Record semantic unknown-constructor ownership, not the surface fact
        // that an annotation node was absent. A synthesized `(t-var _ )`, an
        // authored bare type variable, and a variable later exposed by tuple/
        // record projection all remain owned by this lambda parameter. A
        // declared tensor with only symbolic dims/precision has a known outer
        // constructor and therefore never reaches the Type::Var readiness arm.
        self.shape_lambda_tvars.extend(crate::env::free_tvars(ty));
    }

    /// True only when the unresolved outer constructor descends from a lambda
    /// parameter whose constructor was not fixed by its annotation. Ownership
    /// is transitive through unification: if an origin variable has become a
    /// tuple/record/function shape containing `current`, a projection-derived
    /// `current` still belongs to the same parameter. Other unresolved values
    /// retain their existing wildcard/contextual-inference contract.
    pub(super) fn shape_operand_awaits_lambda_binding(&self, ty: &Type, subst: &Subst) -> bool {
        let applied = subst.apply(ty);
        match applied {
            Type::Var(current) => self
                .shape_lambda_tvars
                .to_sorted()
                .into_iter()
                .any(|origin| {
                    crate::env::free_tvars(&subst.apply(&Type::Var(*origin))).contains(&current)
                }),
            Type::Ref(inner) => self.shape_operand_awaits_lambda_binding(&inner, subst),
            _ => false,
        }
    }

    /// Preserve parameter ownership across an inference operation that
    /// deliberately produces a fresh type without unifying it back into the
    /// source type. Tuple projection is the current such operation: an open
    /// tuple has no row/arity type to bind, but its projected element still
    /// semantically descends from the parameter.
    pub(super) fn derive_shape_lambda_type(
        &mut self,
        source: &Type,
        derived: &Type,
        subst: &Subst,
    ) {
        if self.shape_operand_awaits_lambda_binding(source, subst) {
            self.note_shape_lambda_param(derived);
        }
    }

    pub(super) fn defer_tuple_projection(&mut self, source: Type, index: usize, projected: Type) {
        self.deferred_type_derivations
            .push(DeferredTypeDerivation::TupleProjection {
                source,
                index,
                projected,
            });
    }

    fn resolve_deferred_type_derivations(
        &mut self,
        subst: &mut Subst,
        errors: &mut DiagnosticSink<'_>,
    ) {
        let derivations = std::mem::take(&mut self.deferred_type_derivations);
        for derivation in derivations {
            match derivation {
                DeferredTypeDerivation::TupleProjection {
                    source,
                    index,
                    projected,
                } => match subst.apply(&source) {
                    Type::Var(_) => self.deferred_type_derivations.push(
                        DeferredTypeDerivation::TupleProjection {
                            source,
                            index,
                            projected,
                        },
                    ),
                    Type::Tuple(elements) if index < elements.len() => {
                        if let Err(error) = unify(&projected, &elements[index], subst) {
                            errors.push(error.into());
                        }
                    }
                    Type::Tuple(elements) => errors.push(CheckError::new(
                        CheckErrorKind::TupleIndexOutOfBounds,
                        format!(
                            "tuple index {index} out of bounds for tuple of size {}",
                            elements.len()
                        ),
                        vec![],
                    )),
                    Type::Error(_) => {}
                    other => errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!("expected tuple type, got {other}"),
                        vec![],
                    )),
                },
            }
        }
    }

    pub(super) fn has_pending_shape_check_since(&self, checkpoint: u64) -> bool {
        self.deferred_shape_checks
            .iter()
            .any(|check| check.id >= checkpoint)
    }

    pub(super) fn defer_shape_check(
        &mut self,
        rule: DeferredShapeRule,
        arg_exprs: Vec<deep::Expr>,
        arg_tys: Vec<Type>,
        result_ty: Type,
    ) {
        let id = self.next_deferred_shape_id;
        self.next_deferred_shape_id += 1;
        self.deferred_shape_checks.push(DeferredShapeCheck {
            id,
            rule,
            arg_exprs,
            arg_tys,
            result_ty,
        });
    }

    /// Replay shape checks whose previously-free input types have now been
    /// bound by an application. The same checker functions own both the
    /// immediate and deferred paths, so their semantics cannot drift.
    pub(super) fn replay_ready_shape_checks(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        errors: &mut DiagnosticSink<'_>,
    ) {
        self.resolve_deferred_type_derivations(subst, errors);
        let checks = std::mem::take(&mut self.deferred_shape_checks);
        for check in checks {
            if check
                .arg_tys
                .iter()
                .any(|ty| shape_operand_awaits_binding(ty, subst))
            {
                self.deferred_shape_checks.push(check);
                continue;
            }

            let resolved = match &check.rule {
                DeferredShapeRule::Matmul => {
                    check_matmul_signature(&check.arg_tys, &check.result_ty, subst, errors)
                }
                DeferredShapeRule::Reduction { name } => check_reduction_signature(
                    name,
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    subst,
                    errors,
                ),
                DeferredShapeRule::Expand {
                    builtin,
                    axis_is_dim_name,
                    size_class,
                    env,
                } => check_expand_signature(
                    builtin,
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    *axis_is_dim_name,
                    *size_class,
                    env,
                    subst,
                    errors,
                ),
                DeferredShapeRule::LayerNorm => {
                    check_layer_norm_signature(&check.arg_tys, &check.result_ty, vg, subst, errors)
                }
                DeferredShapeRule::Conv2d => check_conv2d_signature(
                    &check.arg_exprs,
                    &check.arg_tys,
                    &check.result_ty,
                    vg,
                    subst,
                    errors,
                ),
                DeferredShapeRule::ScatterElements { list } => {
                    let kids = children(list);
                    check_scatter_elements(
                        list,
                        kids,
                        &check.arg_tys,
                        check.result_ty.clone(),
                        subst,
                        errors,
                    )
                }
            };
            let _ = resolved;
        }
    }

    /// Acceptance boundary for bind-on-first-use shape lambdas. A remaining
    /// obligation means no application supplied enough type information; the
    /// source must state the intended parameter/result shape explicitly.
    pub(super) fn finish_deferred_shape_checks(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        errors: &mut DiagnosticSink<'_>,
    ) {
        self.replay_ready_shape_checks(vg, subst, errors);
        for check in self.deferred_shape_checks.drain(..) {
            let operation = match check.rule {
                DeferredShapeRule::Matmul => "matmul".to_string(),
                DeferredShapeRule::Reduction { name } => name,
                DeferredShapeRule::Expand { builtin, .. } => (*builtin).to_string(),
                DeferredShapeRule::LayerNorm => "layer_norm".to_string(),
                DeferredShapeRule::Conv2d => "conv2d".to_string(),
                DeferredShapeRule::ScatterElements { .. } => "scatter_elements".to_string(),
            };
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "unresolved `{operation}` shape obligation at declaration boundary: \
                     add an outer-constructor parameter annotation or apply the lambda before \
                     the declaration boundary"
                ),
                vec![
                    "A result annotation does not determine an unresolved parameter constructor; top-level declarations do not borrow binding sites from later declarations"
                        .to_string(),
                ],
            ));
        }
    }

    pub(super) fn record_canonical(&mut self, expr: &deep::Expr, ty: Type) {
        self.record(expr, ty, "canonical infer_expr traversal");
    }

    pub(super) fn record_bypass(&mut self, expr: &deep::Expr, ty: Type, source: &'static str) {
        self.record(expr, ty, source);
    }

    pub(super) fn record(&mut self, expr: &deep::Expr, ty: Type, source: &'static str) {
        let Some(epoch) = self.active_epoch.as_mut() else {
            return;
        };
        let key = epoch.canonical_key(expr_key(expr));
        if !epoch.owners.contains_key(&key) {
            return;
        }
        epoch
            .writes
            .entry(key)
            .or_default()
            .push(OwnerTypeWrite { ty, source });
    }

    pub(super) fn register_bridge_children(
        &mut self,
        stamped_children: &[deep::Expr],
        bridged_children: &[deep::Expr],
    ) {
        let Some(epoch) = self.active_epoch.as_mut() else {
            return;
        };
        debug_assert_eq!(stamped_children.len(), bridged_children.len());
        let mut pending = stamped_children
            .iter()
            .zip(bridged_children)
            .collect::<Vec<_>>();
        while let Some((stamped, bridged)) = pending.pop() {
            let stamped_key = epoch.canonical_key(expr_key(stamped));
            epoch.bridge_aliases.insert(expr_key(bridged), stamped_key);
            match (stamped, bridged) {
                (deep::Expr::Node(left, _), deep::Expr::Node(right, _)) => {
                    pending.extend(left.children_slice().iter().zip(right.children_slice()));
                    pending.extend(
                        left.meta()
                            .entries
                            .iter()
                            .zip(&right.meta().entries)
                            .map(|((_, left), (_, right))| (left, right)),
                    );
                }
                (deep::Expr::List(left, _), deep::Expr::List(right, _)) => {
                    pending.extend(left.elements.iter().zip(&right.elements));
                }
                (deep::Expr::BareList(left, _), deep::Expr::BareList(right, _)) => {
                    pending.extend(left.iter().zip(right));
                }
                (deep::Expr::Map(left, _), deep::Expr::Map(right, _)) => {
                    pending.extend(
                        left.entries
                            .iter()
                            .zip(&right.entries)
                            .map(|((_, left), (_, right))| (left, right)),
                    );
                }
                (deep::Expr::MetaExpr(left, _), deep::Expr::MetaExpr(right, _)) => {
                    pending.push((&left.expr, &right.expr));
                    pending.extend(
                        left.entries
                            .iter()
                            .zip(&right.entries)
                            .map(|((_, left), (_, right))| (left, right)),
                    );
                }
                (deep::Expr::UnknownForm(left), deep::Expr::UnknownForm(right)) => {
                    pending.extend(left.children.iter().zip(&right.children));
                    pending.extend(
                        left.meta
                            .entries
                            .iter()
                            .zip(&right.meta.entries)
                            .map(|((_, left), (_, right))| (left, right)),
                    );
                }
                _ => {}
            }
        }
    }

    pub(super) fn finish_root(&mut self, subst: &Subst, errors: &mut DiagnosticSink<'_>) {
        let Some(epoch) = self.active_epoch.take() else {
            errors.push(internal_owner_stamp_error(
                "attempted to finish a type-stamp epoch that was not active".to_string(),
            ));
            return;
        };

        for (key, requirement) in epoch.owners.into_sorted() {
            let Some(writes) = epoch.writes.get(&key) else {
                // Missing owners are diagnosed at the exact annotation lookup,
                // where the original construct and role are still available.
                continue;
            };
            let mut resolved = writes
                .iter()
                .map(|write| (subst.apply(&write.ty), write.source));
            let Some((canonical, canonical_source)) = resolved.next() else {
                continue;
            };
            for (candidate, candidate_source) in resolved {
                if !owner_types_compatible(&canonical, &candidate) {
                    errors.push(internal_owner_stamp_error(format!(
                        "conflicting authoritative type writes for {} in epoch {}: \
                         `{canonical}` from {canonical_source} vs `{candidate}` from \
                         {candidate_source}",
                        requirement.role, epoch.id
                    )));
                }
            }
            if requirement.stamp_required {
                self.owner_types.insert(
                    key,
                    FinalOwnerType {
                        epoch: epoch.id,
                        ty: canonical,
                    },
                );
            }
        }
    }

    /// Apply the complete program substitution to the frozen owner stamps.
    /// Most stamps are already concrete when their root finishes, but a
    /// variable an earlier root left open can be bound by a later one, so
    /// every stamp gets one final resolution before annotation.
    pub(super) fn resolve_owner_types(&mut self, subst: &Subst) {
        let keys = self
            .owner_types
            .to_sorted()
            .into_iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>();
        for key in keys {
            let owner = self
                .owner_types
                .get_mut(&key)
                .expect("collected owner key remains present");
            owner.ty = subst.apply(&owner.ty);
        }
    }

    pub(super) fn owner_type(
        &self,
        expr: &deep::Expr,
        role: &'static str,
        errors: &mut DiagnosticSink<'_>,
    ) -> Option<Type> {
        match self.owner_types.get(&expr_key(expr)) {
            Some(owner) => {
                let _owning_epoch = owner.epoch;
                Some(owner.ty.clone())
            }
            None => {
                let construct = match expr {
                    deep::Expr::List(list, _) => get_tag(list)
                        .map(DeepTag::as_str)
                        .unwrap_or("<untagged-list>"),
                    deep::Expr::Atom(_, _) => "<atom>",
                    deep::Expr::Map(_, _) => "<map>",
                    deep::Expr::MetaExpr(_, _) => "<meta-expr>",
                    deep::Expr::Node(node, _) => node.tag().as_str(),
                    deep::Expr::BareList(_, _) => "<bare-list>",
                    deep::Expr::UnknownForm(data) => &data.head,
                };
                errors.push(internal_owner_stamp_error(format!(
                    "missing authoritative type stamp for {role} `{construct}`"
                )));
                None
            }
        }
    }

    pub(super) fn current_owner_type(
        &self,
        expr: &deep::Expr,
        subst: &Subst,
        errors: &mut DiagnosticSink<'_>,
    ) -> Option<Type> {
        let Some(epoch) = self.active_epoch.as_ref() else {
            errors.push(internal_owner_stamp_error(
                "canonical inference requested a stamp outside an active epoch".to_string(),
            ));
            return None;
        };
        let key = epoch.canonical_key(expr_key(expr));
        let Some(writes) = epoch.writes.get(&key) else {
            errors.push(internal_owner_stamp_error(
                "canonical inference could not find an already-inferred child stamp".to_string(),
            ));
            return None;
        };
        let mut resolved = writes.iter().map(|write| subst.apply(&write.ty));
        let canonical = resolved.next()?;
        if resolved.any(|candidate| !owner_types_compatible(&canonical, &candidate)) {
            errors.push(internal_owner_stamp_error(
                "canonical inference observed conflicting child stamps".to_string(),
            ));
            return None;
        }
        Some(canonical)
    }
}

/// A semantic shape rule can decide symbolic tensor dimensions and precision
/// variables. It must wait only while an operand's *type constructor* is still
/// unknown; treating every free variable as pending would reject legitimate
/// rank/dtype-polymorphic signatures at their declaration boundary.
pub(super) fn shape_operand_awaits_binding(ty: &Type, subst: &Subst) -> bool {
    match subst.apply(ty) {
        Type::Var(_) => true,
        Type::Ref(inner) => shape_operand_awaits_binding(&inner, subst),
        _ => false,
    }
}

impl TypeStampEpoch {
    fn canonical_key(&self, mut key: usize) -> usize {
        while let Some(next) = self.bridge_aliases.get(&key).copied() {
            if next == key {
                break;
            }
            key = next;
        }
        key
    }
}

pub(super) fn expr_key(expr: &deep::Expr) -> usize {
    std::ptr::from_ref(expr).addr()
}

pub(super) fn owner_types_compatible(left: &Type, right: &Type) -> bool {
    let mut compatibility_subst = Subst::new();
    unify(left, right, &mut compatibility_subst).is_ok()
}

pub(super) fn internal_owner_stamp_error(message: String) -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        format!("internal: annotation owner-stamp invariant violated: {message}"),
        vec![],
    )
}

// Decode-once left this walk with no diagnostic of its own to report: the
// only arm that ever wrote to a sink was the version-skew fallback that
// `child_stamp_role`'s totality made unrepresentable, so the walk no longer
// takes a `DiagnosticSink`.
pub(super) fn register_annotation_owners(expr: &deep::Expr, epoch: &mut TypeStampEpoch) {
    stack_guard!("register_annotation_owners", expr);
    let (tag, kids) = match expr {
        deep::Expr::Node(node, _) => (Some(node.tag()), node.children_slice()),
        deep::Expr::List(list, _) => (get_tag(list), children(list)),
        deep::Expr::MetaExpr(meta, _) => {
            register_annotation_owners(&meta.expr, epoch);
            return;
        }
        deep::Expr::BareList(elements, _) => {
            for child in elements {
                register_annotation_owners(child, epoch);
            }
            return;
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                register_annotation_owners(child, epoch);
            }
            return;
        }
        deep::Expr::Atom(_, _) | deep::Expr::Map(_, _) => return,
    };
    if let Some(tag) = tag {
        let (role, stamp_required) = if tag == DeepTag::Fn {
            ("function node", true)
        } else if should_attach_type_metadata(tag) {
            ("metadata-eligible expression", true)
        } else if matches!(tag, DeepTag::PatVar | DeepTag::PatAs) {
            ("pattern binding", true)
        } else {
            ("semantic runtime node", false)
        };
        register_owner(epoch, expr, role, stamp_required);
    }
    for (index, child) in kids.iter().enumerate() {
        // Decode-once: `child_stamp_role` is total over `DeepTag`, so the
        // old "no child ownership classification" version-skew arm is
        // unrepresentable; only genuinely untagged structural lists (empty
        // guards and malformed nested input) take the recursive fallback,
        // and their owning checker rejects the shape before annotation is
        // returned.
        match tag.map(|tag| child_stamp_role(tag, index, kids.len())) {
            Some(ChildStampRole::RuntimeExpr | ChildStampRole::ExplicitInferenceBypass) | None => {
                register_annotation_owners(child, epoch);
            }
            Some(
                ChildStampRole::Syntax
                | ChildStampRole::Selector
                | ChildStampRole::EffectHandler
                | ChildStampRole::Binder
                | ChildStampRole::Type,
            ) => {}
        }
    }
}

pub(super) fn register_owner(
    epoch: &mut TypeStampEpoch,
    expr: &deep::Expr,
    role: &'static str,
    stamp_required: bool,
) {
    epoch
        .owners
        .entry(expr_key(expr))
        .or_insert(StampRequirement {
            role,
            stamp_required,
        });
}

#[cfg(test)]
pub(crate) enum TypeStampMutationCase {
    Missing,
    CompatibleRepeat,
    IncompatibleRepeat,
    UnregisteredSynthesized,
    RuntimeNonStampOwnerLookup,
}

#[cfg(test)]
pub(crate) fn run_type_stamp_mutation_case(
    case: TypeStampMutationCase,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let owner = node_expr(DeepTag::App, vec![]);
    let mut product = InferenceProduct::default();
    product.begin_root(&owner);
    match case {
        TypeStampMutationCase::Missing => {
            product.finish_root(&Subst::new(), errors);
            product.owner_type(&owner, "test owner", errors).is_none()
        }
        TypeStampMutationCase::CompatibleRepeat => {
            let ty = Type::Prim(Prim::Int64);
            product.record_canonical(&owner, ty.clone());
            product.record_bypass(&owner, ty.clone(), "compatible test repeat");
            product.finish_root(&Subst::new(), errors);
            product.owner_type(&owner, "test owner", errors) == Some(ty)
        }
        TypeStampMutationCase::IncompatibleRepeat => {
            product.record_canonical(&owner, Type::Prim(Prim::Int64));
            product.record_bypass(&owner, Type::Prim(Prim::String), "incompatible test repeat");
            product.finish_root(&Subst::new(), errors);
            true
        }
        TypeStampMutationCase::UnregisteredSynthesized => {
            let synthesized = node_expr(DeepTag::Var, vec![symbol_expr("temporary")]);
            product.record_bypass(
                &synthesized,
                Type::Prim(Prim::Int64),
                "unregistered synthesized test node",
            );
            product.record_canonical(&owner, Type::Prim(Prim::Int64));
            product.finish_root(&Subst::new(), errors);
            product
                .owner_type(&synthesized, "synthesized test node", errors)
                .is_none()
        }
        TypeStampMutationCase::RuntimeNonStampOwnerLookup => {
            let runtime_child = node_expr(DeepTag::Var, vec![symbol_expr("x")]);
            let root = node_expr(DeepTag::App, vec![runtime_child]);
            let mut product = InferenceProduct::default();
            product.begin_root(&root);
            // chelis#1107 amendment (justified-safe, not routed): `root` is
            // built two lines above by this file's own `node_expr`, which
            // returns `deep::Expr::List` unconditionally. No stamped `Node`
            // can reach this reader -- it is `#[cfg(test)]` mutation-case
            // scaffolding over a locally constructed value, not program input.
            let deep::Expr::List(root_list, _) = &root else {
                unreachable!("node_expr produces a list")
            };
            let runtime_child = &children(root_list)[0];
            product.record_canonical(runtime_child, Type::Prim(Prim::Int64));
            product.current_owner_type(runtime_child, &Subst::new(), errors)
                == Some(Type::Prim(Prim::Int64))
        }
    }
}

#[cfg(test)]
pub(crate) enum FinalizationMutationCase {
    MissingRuntimeStamp,
    SilentErrorOwner,
    SilentErrorSignature,
}

#[cfg(test)]
pub(crate) fn run_finalization_mutation_case(
    case: FinalizationMutationCase,
    errors: &mut DiagnosticSink<'_>,
) {
    let runtime = node_expr(DeepTag::App, vec![]);
    let mut signature_context = SignatureInferenceMetadata::default();
    let annotated = match case {
        FinalizationMutationCase::MissingRuntimeStamp => vec![runtime],
        FinalizationMutationCase::SilentErrorOwner => {
            let mut product = InferenceProduct::default();
            product.begin_root(&runtime);
            product.record_canonical(&runtime, crate::errors::error_sentinel_for_test());
            product.finish_root(&Subst::new(), errors);
            annotate_ir_program(std::slice::from_ref(&runtime), &product, errors)
        }
        FinalizationMutationCase::SilentErrorSignature => {
            let error_ty = crate::errors::error_sentinel_for_test();
            signature_context.functions.insert(
                "poison".to_string(),
                FunctionSignatureInference {
                    name: "poison".to_string(),
                    authored_signature: false,
                    authored_signature_type: None,
                    recursive_cycle: false,
                    checked_signature: error_ty.clone(),
                    display_signature: error_ty,
                    params: vec![],
                },
            );
            vec![]
        }
    };
    let _ = finalize_checked_program(
        annotated,
        BTreeMap::new(),
        &signature_context,
        &InferenceProduct::default(),
        InferStats::default(),
        errors,
    );
}

/// Collect the declared signature metadata owned by one inference or
/// annotation unit. The returned map is passed explicitly; nested, sequential,
/// and parallel checks cannot observe another unit's declarations.
pub(super) fn collect_declared_sig_metadata<'a>(
    exprs: impl IntoIterator<Item = &'a deep::Expr>,
) -> UnordMap<String, DeclaredSigMetadata> {
    let exprs = exprs.into_iter().collect::<Vec<_>>();
    let mut map: UnordMap<String, DeclaredSigMetadata> = UnordMap::new();
    for expr in &exprs {
        collect_defsig_param_types(expr, &mut map);
    }
    for expr in exprs {
        extend_declared_sig_binders_from_def_params(expr, &mut map);
    }
    map
}

/// Merge declaration-owned binders preserved by an explicit generic `def`
/// into the matching standalone signature's scope. Surf suppresses the
/// synthesized `defsig` when a same-name explicit `sig` exists; in that case
/// the `def f[piece](x: tensor[piece, ...])` parameter syntax is the only Deep
/// node that still distinguishes the bound `d-var piece` from an ordinary
/// closed annotation. Restricting this merge to names that already own a
/// `defsig` keeps an unrelated direct-Deep `(d-var ...)` annotation closed.
pub(super) fn extend_declared_sig_binders_from_def_params(
    expr: &deep::Expr,
    map: &mut UnordMap<String, DeclaredSigMetadata>,
) {
    stack_guard!("extend_declared_sig_binders_from_def_params", expr);
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    if tag == DeepTag::Module {
        for child in kids.iter().skip(1) {
            extend_declared_sig_binders_from_def_params(child, map);
        }
        return;
    }
    if tag != DeepTag::Def {
        return;
    }
    let (Some(name), Some(fn_expr)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
        return;
    };
    let Some((DeepTag::Fn, _, fn_kids)) = stamped_parts(fn_expr) else {
        return;
    };
    let Some(params) = fn_kids.first() else {
        return;
    };
    let Some(metadata) = map.get_mut(name) else {
        return;
    };
    let params = match params {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => node.children_slice(),
        deep::Expr::List(params_list, _) if get_tag(params_list) == Some(DeepTag::Params) => {
            children(params_list)
        }
        deep::Expr::BareList(elements, _) => elements.as_slice(),
        _ => return,
    };
    for param in params {
        let type_expr = match param {
            deep::Expr::MetaExpr(meta, _) => meta
                .entries
                .iter()
                .find(|(key, _)| key == "type")
                .map(|(_, value)| value),
            deep::Expr::List(param_list, _) => param_list.elements.get(1).and_then(|meta| {
                let deep::Expr::Map(meta, _) = meta else {
                    return None;
                };
                meta.entries
                    .iter()
                    .find(|(key, _)| key == "type")
                    .map(|(_, value)| value)
            }),
            deep::Expr::BareList(elements, _) => elements.get(1).and_then(|meta| {
                let deep::Expr::Map(meta, _) = meta else {
                    return None;
                };
                meta.entries
                    .iter()
                    .find(|(key, _)| key == "type")
                    .map(|(_, value)| value)
            }),
            _ => None,
        };
        if let Some(type_expr) = type_expr {
            metadata.binders.merge(deep_type_binder_names(type_expr));
        }
    }
}

/// Recursively collect `(defsig name (t-fn arg-exprs... ret))` entries,
/// descending through `(module ...)` wrappers. Only the leading
/// argument type expressions are stored (the trailing return type is
/// dropped). A re-declared name keeps the first sig seen.
pub(super) fn collect_defsig_param_types(
    expr: &deep::Expr,
    map: &mut UnordMap<String, DeclaredSigMetadata>,
) {
    // Bail before unbounded recursion exhausts the native stack on a
    // deeply-nested input. No error vector here; `stack_guard_tripped`
    // records the bail so the check entry boundary fails hard with a located
    // diagnostic. See `STACK_RED_ZONE_BYTES`. (In practice this walker only
    // descends `module` wrappers, which do not nest deeply, but the guard
    // keeps the "every recursive walker is bounded" invariant uniform and
    // cheap.)
    stack_guard!("collect_defsig_param_types", expr);
    let Some((tag, meta, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            for child in kids.iter().skip(1) {
                collect_defsig_param_types(child, map);
            }
        }
        DeepTag::Defsig => {
            let Some(name) = kids.first().and_then(symbol_name) else {
                return;
            };
            let Some(fn_expr) = kids.get(1) else {
                return;
            };
            let Some((DeepTag::TFn, _, fn_kids)) = stamped_parts(fn_expr) else {
                return;
            };
            if fn_kids.is_empty() {
                return;
            }
            // All but the trailing return type are parameter types.
            let param_type_exprs: Vec<deep::Expr> = fn_kids[..fn_kids.len() - 1].to_vec();
            map.entry(name.to_string())
                .or_insert_with(|| DeclaredSigMetadata {
                    param_types: param_type_exprs,
                    binders: deep_type_binder_names(&kids[1]),
                    dtype_bounds: chelis_deep::decode_dtype_bounds(meta)
                        .map(|bounds| bounds.into_iter().collect()),
                });
        }
        _ => {}
    }
}

pub(super) fn deep_type_binder_names(type_expr: &deep::Expr) -> UnordSet<String> {
    let mut names = UnordSet::new();
    let mut pending = vec![type_expr];
    while let Some(current) = pending.pop() {
        let Some((tag, _, children)) = stamped_parts(current) else {
            continue;
        };
        if matches!(tag, DeepTag::TVar | DeepTag::DVar | DeepTag::DRank)
            && let Some(variable) = children.first().and_then(symbol_name)
            && variable != "_"
        {
            names.insert(variable.to_string());
        }
        pending.extend(children);
    }
    names
}

/// Result of running type inference on a program.
#[derive(Debug)]
pub struct InferResult {
    pub errors: Vec<CheckError>,
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct InferStats {
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckedProgram {
    annotated_exprs: Vec<deep::Expr>,
    type_env: BTreeMap<String, deep::Expr>,
    linearity: LinearityInfo,
    signature_inference: SignatureInferenceMetadata,
    type_headers: TypeResolutionEnv,
    #[serde(default)]
    adt_registry: AdtRegistry,
    /// Honest checker-visit counters for this checked unit. These are
    /// serialized with cached contexts so layered fitness reports can sum the
    /// same inference-product metric as the monolithic path (chelis#973).
    #[serde(default)]
    infer_stats: InferStats,
    #[serde(default)]
    library_proof_id: Option<LibraryProofId>,
    #[serde(default)]
    context_library_proof_id: Option<LibraryProofId>,
}

impl CheckedProgram {
    #[cfg(test)]
    pub(crate) fn unchecked_for_linearity_diagnostic_test(
        annotated_exprs: Vec<deep::Expr>,
        type_env: BTreeMap<String, deep::Expr>,
    ) -> Self {
        Self {
            annotated_exprs,
            type_env,
            linearity: LinearityInfo::default(),
            signature_inference: SignatureInferenceMetadata::default(),
            type_headers: TypeResolutionEnv::default(),
            adt_registry: AdtRegistry::default(),
            infer_stats: InferStats::default(),
            library_proof_id: None,
            context_library_proof_id: None,
        }
    }

    /// Replace only effects-owned metadata on an already-checked program.
    ///
    /// Every span, atom, child, and non-`effects` metadata entry must remain
    /// byte-for-byte identical to `self`. The returned program preserves the
    /// original type/signature/linearity contexts and crosses the same
    /// fallible totality boundary as a fresh checker result.
    pub fn try_with_effect_annotations(
        &self,
        annotated_exprs: Vec<deep::Expr>,
    ) -> Result<Self, InferResult> {
        crate::session::try_checked_program_with_effect_annotations(self, annotated_exprs)
    }

    pub fn exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn annotated_exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn type_env(&self) -> &BTreeMap<String, deep::Expr> {
        &self.type_env
    }

    pub fn linearity(&self) -> &LinearityInfo {
        &self.linearity
    }

    pub fn signature_inference(&self) -> &SignatureInferenceMetadata {
        &self.signature_inference
    }

    pub(crate) fn type_headers(&self) -> &TypeResolutionEnv {
        &self.type_headers
    }

    /// Checker-owned, alias-resolved ADT definitions used by lowering.
    ///
    /// Consumers must use this registry instead of reconstructing constructor
    /// layouts or generic-parameter roles from authored `deftype` syntax.
    pub fn adt_registry(&self) -> &AdtRegistry {
        &self.adt_registry
    }

    pub fn infer_stats(&self) -> InferStats {
        self.infer_stats
    }

    pub(crate) fn bind_library_proof(&mut self, proof_id: LibraryProofId) {
        self.library_proof_id = Some(proof_id);
    }

    pub fn library_proof_id(&self) -> Option<LibraryProofId> {
        self.library_proof_id
    }

    pub(crate) fn bind_context_library_proof(&mut self, proof_id: Option<LibraryProofId>) {
        self.context_library_proof_id = proof_id;
    }

    pub fn with_linearity(mut self, linearity: LinearityInfo) -> Self {
        self.linearity = linearity;
        self
    }

    /// Compose a `library` checked program with a `new_code` checked
    /// program into a single whole-program `CheckedProgram`, equivalent
    /// to what `check_ir_program(library_exprs ++ new_code_exprs)` plus
    /// effects + linearity would produce — provided `new_code` was
    /// produced by the `_with_context` variants stacked on `library`.
    ///
    /// This is the seam the cross-process chelis-std typecheck cache's
    /// `chelis build` path uses: `library` is the cached chelis-std
    /// sub-context's `library_checked` and `new_code` is the
    /// `_with_context`-checked non-chelis-std decls + entry. The result
    /// is the one monolithic `CheckedProgram` the `build` lowering
    /// pipeline consumes, without re-inferring chelis-std.
    ///
    /// Composition rule (mirrors the monolithic `library ++ new` shape):
    /// - `annotated_exprs`: `library` exprs followed by `new_code` exprs,
    ///   in that order. Monolithic `check_ir_program` annotates in
    ///   source order, and the linked program places library decls
    ///   before the entry, so this ordering matches.
    /// - `type_env`: union, `new_code` winning on shadow. `new_code`'s
    ///   `type_env` is already unioned with the library's by the
    ///   `_with_context` builder, so this just back-fills any
    ///   library-only entries.
    /// - `linearity`: the two `reusable_inputs_by_offset` maps merged.
    /// - `signature_inference`: the two `functions` maps merged,
    ///   `new_code` winning on a name clash.
    ///
    /// The monolithic-vs-layered acceptance oracle is what proves this
    /// composition is byte-identical to the monolithic path; a
    /// divergence is a compiler-correctness bug, not a tuning knob.
    ///
    /// Returns `None` unless `new_code` retains the exact library proof.
    pub fn compose(library: &CheckedProgram, new_code: &CheckedProgram) -> Option<Self> {
        if library.library_proof_id.is_none()
            || new_code.context_library_proof_id != library.library_proof_id
        {
            return None;
        }
        let mut annotated_exprs =
            Vec::with_capacity(library.annotated_exprs.len() + new_code.annotated_exprs.len());
        annotated_exprs.extend(library.annotated_exprs.iter().cloned());
        annotated_exprs.extend(new_code.annotated_exprs.iter().cloned());

        let mut type_env = new_code.type_env.clone();
        for (name, ty) in &library.type_env {
            type_env.entry(name.clone()).or_insert_with(|| ty.clone());
        }

        let linearity = library.linearity.merged_with(&new_code.linearity);

        let mut signature_inference = library.signature_inference.clone();
        for (name, sig) in &new_code.signature_inference.functions {
            signature_inference
                .functions
                .insert(name.clone(), sig.clone());
        }

        let mut type_headers = library.type_headers.clone();
        type_headers.extend_from(&new_code.type_headers);

        // A context-checked new-code program normally already carries the
        // library registry. Back-fill defensively so composition remains
        // total for independently deserialized legacy programs as well.
        let mut adt_registry = new_code.adt_registry.clone();
        for (name, definition) in &library.adt_registry.defs {
            adt_registry
                .defs
                .entry(name.clone())
                .or_insert_with(|| definition.clone());
        }
        for (name, alias) in &library.adt_registry.aliases {
            adt_registry
                .aliases
                .entry(name.clone())
                .or_insert_with(|| alias.clone());
        }

        Some(Self {
            annotated_exprs,
            type_env,
            linearity,
            signature_inference,
            type_headers,
            adt_registry,
            infer_stats: InferStats {
                typed_nodes: library.infer_stats.typed_nodes + new_code.infer_stats.typed_nodes,
                total_nodes: library.infer_stats.total_nodes + new_code.infer_stats.total_nodes,
            },
            library_proof_id: new_code.library_proof_id,
            context_library_proof_id: None,
        })
    }
}

/// The sole construction boundary for checked results. It observes the
/// authoritative session sink before adding an invariant diagnostic, so a
/// real root error is never duplicated by the totality backstop.
pub(super) fn finalize_checked_program(
    annotated_exprs: Vec<deep::Expr>,
    type_env: BTreeMap<String, deep::Expr>,
    signature_context: &SignatureInferenceMetadata,
    product: &InferenceProduct,
    infer_stats: InferStats,
    errors: &mut DiagnosticSink<'_>,
) -> CheckedProgram {
    let signature_inference = infer_signature_metadata_with_context_and_headers(
        &annotated_exprs,
        &product.function_inference_plan,
        &type_env,
        signature_context,
        &product.type_headers,
        errors,
    );
    let checked = CheckedProgram {
        annotated_exprs,
        type_env,
        linearity: LinearityInfo::default(),
        signature_inference,
        type_headers: product.type_headers.clone(),
        adt_registry: product.adt_registry.clone(),
        infer_stats,
        library_proof_id: None,
        context_library_proof_id: None,
    };

    validate_checked_program_totality(&checked, signature_context, errors);
    checked
}

pub(super) fn validate_checked_program_totality(
    checked: &CheckedProgram,
    signature_context: &SignatureInferenceMetadata,
    errors: &mut DiagnosticSink<'_>,
) {
    if !errors.is_empty() {
        return;
    }
    let mut traces = totality_invariant_traces(signature_context);
    traces.extend(annotated_totality_invariant_traces(
        checked.annotated_exprs(),
    ));
    traces.extend(totality_invariant_traces(checked.signature_inference()));
    if !traces.is_empty() {
        errors.push(totality_violation_error(&traces));
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignatureInferenceMetadata {
    pub functions: BTreeMap<String, FunctionSignatureInference>,
}

impl SignatureInferenceMetadata {
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignatureInference {
    pub name: String,
    /// True when this function has an authored or Surf-synthesized `defsig`.
    /// Lowering uses this checker-owned fact to distinguish declared
    /// polymorphism from generalized local-callback inference.
    #[serde(default)]
    pub authored_signature: bool,
    /// The checker-decoded authored signature before body inference sharpens
    /// wildcard dimensions or otherwise specializes the checked function.
    ///
    /// Lowering consumes this record instead of reparsing `defsig` syntax.
    #[serde(default)]
    pub authored_signature_type: Option<Type>,
    pub recursive_cycle: bool,
    pub checked_signature: Type,
    pub display_signature: Type,
    pub params: Vec<ParamSignatureInference>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamSignatureInference {
    pub index: usize,
    pub name: String,
    pub written: bool,
    pub inferred_read_only: bool,
    pub checked_type: Type,
    pub display_type: Type,
}

pub(crate) fn checked_program_with_effect_annotations_in_session(
    original: &CheckedProgram,
    annotated_exprs: Vec<deep::Expr>,
    errors: &mut DiagnosticSink<'_>,
) -> CheckedProgram {
    if !effects_only_rewrite_matches(original.annotated_exprs(), &annotated_exprs) {
        errors.push(CheckError::new(
            CheckErrorKind::Other,
            "checked-program effects-only reannotation changed a span, atom, child, or non-`effects` metadata entry"
                .to_string(),
            vec![
                "Run type checking again for structural, body, type, `eff`, or source-span changes"
                    .to_string(),
            ],
        ));
    }

    let checked = CheckedProgram {
        annotated_exprs,
        type_env: original.type_env.clone(),
        linearity: original.linearity.clone(),
        signature_inference: original.signature_inference.clone(),
        type_headers: original.type_headers.clone(),
        adt_registry: original.adt_registry.clone(),
        infer_stats: original.infer_stats,
        library_proof_id: original.library_proof_id,
        context_library_proof_id: original.context_library_proof_id,
    };
    validate_checked_program_totality(&checked, original.signature_inference(), errors);
    checked
}

pub(super) fn effects_only_rewrite_matches(
    original: &[deep::Expr],
    candidate: &[deep::Expr],
) -> bool {
    original.len() == candidate.len()
        && original
            .iter()
            .zip(candidate)
            .all(|(before, after)| effects_only_expr_matches(before, after))
        && candidate.iter().all(effect_metadata_is_singular)
}

pub(super) fn effects_only_expr_matches(before: &deep::Expr, after: &deep::Expr) -> bool {
    stack_guard!("effects_only_expr_matches", before, false);
    match (before, after) {
        (deep::Expr::Atom(before_atom, before_span), deep::Expr::Atom(after_atom, after_span)) => {
            before_atom == after_atom && before_span == after_span
        }
        (deep::Expr::Map(before_map, before_span), deep::Expr::Map(after_map, after_span)) => {
            before_span == after_span
                && metadata_entries_match(&before_map.entries, &after_map.entries, false)
        }
        (
            deep::Expr::MetaExpr(before_meta, before_span),
            deep::Expr::MetaExpr(after_meta, after_span),
        ) => {
            before_span == after_span
                && metadata_entries_match(&before_meta.entries, &after_meta.entries, false)
                && effects_only_expr_matches(&before_meta.expr, &after_meta.expr)
        }
        (deep::Expr::List(before_list, before_span), deep::Expr::List(after_list, after_span)) => {
            before_span == after_span
                && before_list.elements.len() == after_list.elements.len()
                && before_list
                    .elements
                    .iter()
                    .zip(&after_list.elements)
                    .enumerate()
                    .all(|(index, (before_element, after_element))| {
                        if index == 1
                            && let (
                                deep::Expr::Map(before_map, before_map_span),
                                deep::Expr::Map(after_map, after_map_span),
                            ) = (before_element, after_element)
                        {
                            return before_map_span == after_map_span
                                && metadata_entries_match(
                                    &before_map.entries,
                                    &after_map.entries,
                                    true,
                                );
                        }
                        effects_only_expr_matches(before_element, after_element)
                    })
        }
        (deep::Expr::Node(before_node, before_span), deep::Expr::Node(after_node, after_span)) => {
            before_span == after_span
                && before_node.tag() == after_node.tag()
                && metadata_entries_match(
                    &before_node.meta().entries,
                    &after_node.meta().entries,
                    true,
                )
                && before_node.children_slice().len() == after_node.children_slice().len()
                && before_node
                    .children_slice()
                    .iter()
                    .zip(after_node.children_slice())
                    .all(|(before_child, after_child)| {
                        effects_only_expr_matches(before_child, after_child)
                    })
        }
        (
            deep::Expr::BareList(before_elements, before_span),
            deep::Expr::BareList(after_elements, after_span),
        ) => {
            before_span == after_span
                && before_elements.len() == after_elements.len()
                && before_elements
                    .iter()
                    .zip(after_elements)
                    .all(|(before_child, after_child)| {
                        effects_only_expr_matches(before_child, after_child)
                    })
        }
        (deep::Expr::UnknownForm(before_data), deep::Expr::UnknownForm(after_data)) => {
            before_data.head == after_data.head
                && before_data.span == after_data.span
                && metadata_entries_match(&before_data.meta.entries, &after_data.meta.entries, true)
                && before_data.children.len() == after_data.children.len()
                && before_data.children.iter().zip(&after_data.children).all(
                    |(before_child, after_child)| {
                        effects_only_expr_matches(before_child, after_child)
                    },
                )
        }
        _ => false,
    }
}

pub(super) fn metadata_entries_match(
    before: &[(String, deep::Expr)],
    after: &[(String, deep::Expr)],
    ignore_effects: bool,
) -> bool {
    let mut before_entries = before
        .iter()
        .filter(|(key, _)| !ignore_effects || key != "effects");
    let mut after_entries = after
        .iter()
        .filter(|(key, _)| !ignore_effects || key != "effects");
    loop {
        match (before_entries.next(), after_entries.next()) {
            (Some((before_key, before_value)), Some((after_key, after_value))) => {
                if before_key != after_key || !effects_only_expr_matches(before_value, after_value)
                {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

pub(super) fn effect_metadata_is_singular(expr: &deep::Expr) -> bool {
    stack_guard!("effect_metadata_is_singular", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => true,
        deep::Expr::Map(map, _) => map
            .entries
            .iter()
            .all(|(_, value)| effect_metadata_is_singular(value)),
        deep::Expr::MetaExpr(meta, _) => {
            effect_metadata_is_singular(&meta.expr)
                && meta
                    .entries
                    .iter()
                    .all(|(_, value)| effect_metadata_is_singular(value))
        }
        deep::Expr::List(list, _) => {
            let singular_here = match list.elements.get(1) {
                Some(deep::Expr::Map(map, _)) => {
                    map.entries
                        .iter()
                        .filter(|(key, _)| key == "effects")
                        .count()
                        <= 1
                }
                _ => true,
            };
            singular_here && list.elements.iter().all(effect_metadata_is_singular)
        }
        deep::Expr::Node(node, _) => {
            node.meta()
                .entries
                .iter()
                .filter(|(key, _)| key == "effects")
                .count()
                <= 1
                && node
                    .meta()
                    .entries
                    .iter()
                    .all(|(_, value)| effect_metadata_is_singular(value))
                && node
                    .children_slice()
                    .iter()
                    .all(effect_metadata_is_singular)
        }
        deep::Expr::BareList(elems, _) => elems.iter().all(effect_metadata_is_singular),
        deep::Expr::UnknownForm(data) => {
            data.meta
                .entries
                .iter()
                .filter(|(key, _)| key == "effects")
                .count()
                <= 1
                && data
                    .meta
                    .entries
                    .iter()
                    .all(|(_, value)| effect_metadata_is_singular(value))
                && data.children.iter().all(effect_metadata_is_singular)
        }
    }
}
