//! Declaration-owned evidence for spec/04 §4.5.3 body composition.
//! Lexical targets and SCCs come exclusively from TopLevelReferenceGraph.

use super::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
struct RankDependency {
    owner: [u8; 32],
    needs_body: bool,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
struct RankBodyEvidence {
    owner: [u8; 32],
    signature: [u8; 32],
    component_owner: String,
    dependencies: BTreeMap<String, RankDependency>,
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub(crate) struct RankBodyContext {
    owners: BTreeMap<String, [u8; 32]>,
    declarations: BTreeMap<String, RankBodyEvidence>,
}

fn syntax_digest(exprs: &[deep::Expr]) -> [u8; 32] {
    Sha256::digest(chelis_deep::printer::print_canonical_flat(exprs).as_bytes()).into()
}

impl RankBodyContext {
    fn replace_owners(&mut self, exprs: &[deep::Expr]) -> BTreeSet<String> {
        let mut groups = BTreeMap::<String, Vec<deep::Expr>>::new();
        for expr in top_level_decl_items(exprs) {
            if let Some(name) = top_level_decl_name(expr) {
                groups
                    .entry(name.to_string())
                    .or_default()
                    .push(expr.clone());
            }
        }
        for (name, items) in &groups {
            self.owners.insert(name.clone(), syntax_digest(items));
        }
        groups.into_keys().collect()
    }

    fn invalid_dependencies(&self, name: &str, evidence: &RankBodyEvidence) -> bool {
        self.owners.get(name) != Some(&evidence.owner)
            || !self.declarations.contains_key(&evidence.component_owner)
            || evidence.dependencies.iter().any(|(target, dependency)| {
                self.owners.get(target) != Some(&dependency.owner)
                    || (dependency.needs_body && !self.declarations.contains_key(target))
            })
    }

    /// Invalidate requiring callers and whole jointly checked SCCs through a
    /// reverse index, without repeatedly rescanning a dependency chain.
    fn invalidate_changed_dependencies(&mut self) {
        let mut reverse = BTreeMap::<String, BTreeSet<String>>::new();
        let mut groups = BTreeMap::<String, Vec<String>>::new();
        let mut pending = BTreeSet::new();
        for (name, evidence) in &self.declarations {
            groups
                .entry(evidence.component_owner.clone())
                .or_default()
                .push(name.clone());
            if self.invalid_dependencies(name, evidence) {
                pending.insert(name.clone());
            }
            for (target, dependency) in &evidence.dependencies {
                if dependency.needs_body {
                    reverse
                        .entry(target.clone())
                        .or_default()
                        .insert(name.clone());
                }
            }
        }
        while let Some(name) = pending.pop_first() {
            let Some(evidence) = self.declarations.remove(&name) else {
                continue;
            };
            if let Some(members) = groups.remove(&evidence.component_owner) {
                pending.extend(members);
            }
            if let Some(callers) = reverse.remove(&name) {
                pending.extend(callers);
            }
        }
    }

    pub(crate) fn bind_checked_declarations(
        &mut self,
        exprs: &[deep::Expr],
        types: &BTreeMap<String, deep::Expr>,
    ) -> Result<(), String> {
        let replaced = self.replace_owners(exprs);
        for name in &replaced {
            let Some(evidence) = self.declarations.get_mut(name) else {
                continue;
            };
            let ty = types.get(name).ok_or_else(|| {
                format!("rank body evidence for `{name}` has no checked type owner")
            })?;
            evidence.owner = self.owners[name];
            evidence.signature = syntax_digest(core::slice::from_ref(ty));
            for (target, dependency) in &mut evidence.dependencies {
                dependency.owner = *self.owners.get(target).ok_or_else(|| {
                    format!("rank body dependency `{target}` has no declaration owner")
                })?;
            }
        }
        self.invalidate_changed_dependencies();
        self.validate(types)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.owners.is_empty() && self.declarations.is_empty()
    }

    pub(crate) fn validate(&self, types: &BTreeMap<String, deep::Expr>) -> Result<(), String> {
        for (name, evidence) in &self.declarations {
            if self.invalid_dependencies(name, evidence)
                || types
                    .get(name)
                    .map(|ty| syntax_digest(core::slice::from_ref(ty)))
                    != Some(evidence.signature)
            {
                return Err(format!(
                    "rank body evidence for `{name}` disagrees with its declaration, type, component, or dependency owner"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn digest(&self) -> [u8; 32] {
        fn name(hasher: &mut Sha256, name: &str) {
            hasher.update((name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
        }
        let mut hasher = Sha256::new();
        hasher.update(b"chelis-rank-body-evidence-v1");
        hasher.update((self.owners.len() as u64).to_le_bytes());
        for (key, owner) in &self.owners {
            name(&mut hasher, key);
            hasher.update(owner);
        }
        hasher.update((self.declarations.len() as u64).to_le_bytes());
        for (key, evidence) in &self.declarations {
            name(&mut hasher, key);
            hasher.update(evidence.owner);
            hasher.update(evidence.signature);
            name(&mut hasher, &evidence.component_owner);
            hasher.update((evidence.dependencies.len() as u64).to_le_bytes());
            for (target, dependency) in &evidence.dependencies {
                name(&mut hasher, target);
                hasher.update(dependency.owner);
                hasher.update([u8::from(dependency.needs_body)]);
            }
        }
        hasher.finalize().into()
    }
}

fn rank_body_error(def_name: &str, message: String) -> CheckError {
    CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!("rank-polymorphic def `{def_name}` {message} (spec/04-type-system.md §4.5.3)."),
        vec![],
    )
}

fn check_body(
    def_name: &str,
    expr: &deep::Expr,
    graph: &TopLevelReferenceGraph,
    available: &BTreeSet<usize>,
    component: &BTreeSet<usize>,
    context: &RankBodyContext,
    errors: &mut Vec<CheckError>,
    operation: &crate::resources::Operation<'_>,
) {
    stack_guard!(operation, "check_rank_body_discipline", expr);
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        if let deep::Expr::MetaExpr(meta, _) = expr {
            check_body(
                def_name, &meta.expr, graph, available, component, context, errors, operation,
            );
        }
        return;
    };
    match tag {
        DeepTag::Grad | DeepTag::Vmap => errors.push(rank_body_error(
            def_name,
            format!("may not use `{}` in its body: no body evidence transports its function across the opaque rank", tag.as_str()),
        )),
        DeepTag::App => match graph.applications.get(&expr_key(expr)) {
            Some(ResolvedApplication::Definition(vertex))
                if available.contains(vertex) || component.contains(vertex) => {}
            Some(ResolvedApplication::Definition(vertex)) => errors.push(rank_body_error(
                def_name,
                format!("may not call `{}`: checked name-preserving body evidence is unavailable", graph.definition(*vertex).name),
            )),
            Some(ResolvedApplication::Unshadowed(name)) if context.declarations.contains_key(name) => {}
            Some(ResolvedApplication::Unshadowed(name)) if context.owners.contains_key(name) => errors.push(rank_body_error(
                def_name, format!("may not call `{name}`: checked name-preserving body evidence is unavailable"),
            )),
            Some(ResolvedApplication::Unshadowed(name)) => match builtins::shape_class(name) {
                Some(builtins::ShapeClass::Identity | builtins::ShapeClass::NameTracked
                    | builtins::ShapeClass::OrderedPrefix | builtins::ShapeClass::Inert) => {}
                Some(builtins::ShapeClass::Untracked) => errors.push(rank_body_error(
                    def_name,
                    format!("may not call builtin `{name}`: it is not name-trackable; only shape-identity, named-axis, ordered-prefix and inert builtins are admitted"),
                )),
                None => errors.push(rank_body_error(
                    def_name,
                    format!("may not call `{name}`: checked name-preserving body evidence is unavailable"),
                )),
            },
            Some(ResolvedApplication::Local(name)) => errors.push(rank_body_error(
                def_name,
                format!("may not apply local callable `{name}`: its function type supplies no body evidence"),
            )),
            Some(ResolvedApplication::Computed) | None => errors.push(rank_body_error(
                def_name,
                "may not apply a computed callee: resolved declaration-owned body evidence is unavailable".to_string(),
            )),
        },
        _ => {}
    }
    for child in kids {
        check_body(
            def_name, child, graph, available, component, context, errors, operation,
        );
    }
}

/// Check complete bodies against the existing inference/row-unification
/// result, accepting SCCs as one unit. This returns staged evidence; its
/// caller's empty-errors/totality boundary owns publication.
pub(super) fn validate_rank_bodies(
    exprs: &[deep::Expr],
    product: &InferenceProduct,
    graph: &TopLevelReferenceGraph,
    base: &RankBodyContext,
    inference_succeeded: bool,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> RankBodyContext {
    let items = top_level_decl_items(exprs);
    let mut context = base.clone();
    context.replace_owners(exprs);
    context.invalidate_changed_dependencies();
    let components = &product.function_inference_plan;
    if !components.complete {
        return if inference_succeeded {
            context
        } else {
            base.clone()
        };
    }
    let cancel = operation.cancellation;
    let mut accepted = BTreeSet::new();
    for component in &components.components {
        if cancel.is_some_and(|token| token.is_cancelled()) {
            break;
        }
        let mut candidate = BTreeSet::new();
        for member in &component.members {
            let index = member.item_index;
            if let Some(vertex) = graph.definition_vertex_by_item[index]
                && graph.definition(vertex).kind == TopLevelDefinitionKind::Function
            {
                candidate.insert(vertex);
            }
        }
        let mut checked = Vec::new();
        let mut component_valid = true;
        for member in &component.members {
            let index = member.item_index;
            let Some(vertex) = graph.definition_vertex_by_item[index] else {
                continue;
            };
            let definition = graph.definition(vertex);
            let Some((DeepTag::Def, _, def_kids)) = stamped_parts(items[index]) else {
                continue;
            };
            let Some((DeepTag::Fn, _, fn_kids)) = def_kids.get(1).and_then(stamped_parts) else {
                component_valid = false;
                continue;
            };
            let Some(body) = fn_kids.get(1) else {
                component_valid = false;
                continue;
            };
            let mut failures = Vec::new();
            check_body(
                &definition.name,
                body,
                graph,
                &accepted,
                &candidate,
                &context,
                &mut failures,
                operation,
            );
            component_valid &= failures.is_empty();
            checked.push((vertex, index, failures));
        }
        let component_owner = component
            .members
            .iter()
            .map(|member| member.name.as_str())
            .min()
            .expect("component has members");
        for (vertex, index, failures) in checked {
            let name = &graph.definition(vertex).name;
            let (_, _, def_kids) = stamped_parts(items[index]).expect("checked definition");
            let signature = if inference_succeeded {
                product.owner_type(&def_kids[1], "rank-body function", errors)
            } else {
                // A rejected declaration may intentionally have no stamp.
                // Existing stamps still own independent body diagnostics;
                // absence supplies neither a replacement type nor evidence.
                product
                    .owner_types
                    .get(&expr_key(&def_kids[1]))
                    .map(|owner| owner.ty().clone())
            };
            let required =
                signature.as_ref().is_some_and(type_contains_rank) && !is_exact_op35_wrapper(name);
            if component_valid && let Some(signature) = signature {
                accepted.insert(vertex);
                if !inference_succeeded {
                    continue;
                }
                let signature = type_to_deep_expr(&signature);
                let mut dependencies = BTreeMap::new();
                for reference in &graph.item_references[index] {
                    let target = &graph.definition(reference.target).name;
                    dependencies.insert(
                        target.clone(),
                        RankDependency {
                            owner: context.owners[target],
                            needs_body: reference.kind == TopLevelReferenceKind::Apply,
                        },
                    );
                }
                for (target, kind) in &graph.external_references[index] {
                    if let Some(owner) = context.owners.get(target) {
                        dependencies.insert(
                            target.clone(),
                            RankDependency {
                                owner: *owner,
                                needs_body: *kind == TopLevelReferenceKind::Apply,
                            },
                        );
                    }
                }
                context.declarations.insert(
                    name.clone(),
                    RankBodyEvidence {
                        owner: context.owners[name],
                        signature: syntax_digest(core::slice::from_ref(&signature)),
                        component_owner: component_owner.to_string(),
                        dependencies,
                    },
                );
            } else if required {
                if failures.is_empty() {
                    errors.push(rank_body_error(name, "has no body evidence: a member of its recursive component is not name-trackable".to_string()));
                } else {
                    for error in failures {
                        errors.push(error);
                    }
                }
            }
        }
    }
    if inference_succeeded {
        context
    } else {
        base.clone()
    }
}
