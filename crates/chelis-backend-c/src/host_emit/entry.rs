//! Domination proof for signature comparisons delegated out of C helpers.
//!
//! Values have lexical identities. Display names never join independent
//! signatures or shadowed bindings. Each call receives a private helper
//! schedule, removing only comparisons entailed by its dominating entry plan.

use super::*;
use chelis_ir::axis_sources::EntryExtentGuard;
use chelis_ir::host::SignatureEntryPlan;
use chelis_ir::ownership::VerifiedHostTensorHelperView;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Read(usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fact {
    Equal(Read, Read),
    Literal(Read, usize),
}

fn fact(
    guard: &EntryExtentGuard,
    source: impl Fn(chelis_ir::NodeId, usize) -> Option<Read>,
) -> Option<Fact> {
    match guard {
        EntryExtentGuard::Named {
            canonical,
            observed,
            ..
        } => {
            let a = source(canonical.0, canonical.1)?;
            let b = source(observed.0, observed.1)?;
            Some(Fact::Equal(a.min(b), a.max(b)))
        }
        EntryExtentGuard::Literal { required, observed } => {
            Some(Fact::Literal(source(observed.0, observed.1)?, *required))
        }
    }
}

fn implies(facts: &[Fact], obligation: &Fact) -> bool {
    let (start, target, literal) = match obligation {
        Fact::Equal(a, b) => (*a, Some(*b), None),
        Fact::Literal(axis, required) => (*axis, None, Some(*required)),
    };
    let mut equivalent = vec![start];
    let mut cursor = 0;
    while cursor < equivalent.len() {
        let observed = equivalent[cursor];
        if target == Some(observed) {
            return true;
        }
        for fact in facts {
            match fact {
                Fact::Equal(a, b) => {
                    let other = if *a == observed {
                        Some(*b)
                    } else if *b == observed {
                        Some(*a)
                    } else {
                        None
                    };
                    if let Some(other) = other
                        && !equivalent.contains(&other)
                    {
                        equivalent.push(other);
                    }
                }
                Fact::Literal(axis, required)
                    if *axis == observed && literal == Some(*required) =>
                {
                    return true;
                }
                _ => {}
            }
        }
        cursor += 1;
    }
    false
}

fn var_read(expr: &HostExpr, axis: usize, env: &BTreeMap<String, usize>) -> Option<Read> {
    let HostExprKind::Var(name, _) = &expr.kind else {
        return None;
    };
    Some(Read(*env.get(name)?, axis))
}

fn plan_facts(
    plan: &SignatureEntryPlan,
    args: &[HostExpr],
    env: &BTreeMap<String, usize>,
) -> Vec<Fact> {
    plan.guards()
        .iter()
        .filter_map(|guard| fact(guard, |load, axis| var_read(args.get(load.0)?, axis, env)))
        .collect()
}

fn same_claim(left: &EntryExtentGuard, right: &EntryExtentGuard) -> bool {
    match (left, right) {
        (
            EntryExtentGuard::Named { claim: left, .. },
            EntryExtentGuard::Named { claim: right, .. },
        ) => left == right,
        (
            EntryExtentGuard::Literal { required: left, .. },
            EntryExtentGuard::Literal {
                required: right, ..
            },
        ) => left == right,
        _ => false,
    }
}

/// Signature guards that a direct tensor helper must retain to preserve one
/// ordered entry schedule.
///
/// The host adapter normally owns every signature comparison and the helper
/// omits the facts that adapter established. A helper can introduce an
/// earlier result-literal obligation that the authored input signature does
/// not contain, though. If the same helper later carries one of the
/// signature's guards, executing that guard in the adapter would move it
/// ahead of the earlier literal. Delegate only that exact later guard, and
/// only for a direct helper body: callbacks, branches, and preceding host
/// operations do not provide the domination proof needed to move entry work.
pub(super) fn delegated_function_guards(
    function: &HostFunction,
    verified_helpers: &[VerifiedHostTensorHelperView<'_>],
) -> Vec<EntryExtentGuard> {
    let HostExprKind::TensorCall { helper, args, .. } = &function.body.kind else {
        return Vec::new();
    };
    let Some(source) = verified_helpers.get(*helper).copied() else {
        return Vec::new();
    };

    let mut env = BTreeMap::new();
    for (index, param) in function.params.iter().enumerate() {
        env.insert(param.name.clone(), index);
    }
    let plan = function_entry_plan(function);
    let plan_args = function
        .params
        .iter()
        .filter(|param| matches!(param.ty, HostAbiType::Tensor(_)))
        .map(|param| HostExpr::new(HostExprKind::Var(param.name.clone(), param.ty.clone())))
        .collect::<Vec<_>>();
    let plan_guards = plan
        .guards()
        .iter()
        .filter_map(|guard| {
            fact(guard, |load, axis| {
                var_read(plan_args.get(load.0)?, axis, &env)
            })
            .map(|known| (guard, known))
        })
        .collect::<Vec<_>>();
    let read = |load, axis| {
        let RiscOp::Load { name } = &source.dag().get(load)?.op else {
            return None;
        };
        let input = source
            .inputs()
            .iter()
            .position(|candidate| candidate.name == name.as_str())?;
        var_read(args.get(input)?, axis, &env)
    };
    let helper_guards = source
        .dag()
        .entry_extent_guards()
        .into_iter()
        .filter_map(|guard| fact(&guard, read).map(|known| (guard, known)))
        .collect::<Vec<_>>();

    let mut established = Vec::new();
    let mut delegated = Vec::new();
    for (guard, known) in plan_guards {
        let helper_position = helper_guards
            .iter()
            .position(|(candidate, fact)| same_claim(guard, candidate) && fact == &known);
        let must_follow_helper_prefix = helper_position.is_some_and(|position| {
            helper_guards[..position]
                .iter()
                .any(|(_, earlier)| !implies(&established, earlier))
        });
        if must_follow_helper_prefix {
            delegated.push(guard.clone());
        } else {
            established.push(known);
        }
    }
    delegated
}

/// A schedule is tied to the verified payload's structural expression order,
/// so cloning the immutable tree preserves each call's identity.
#[derive(Default)]
pub(super) struct Projection {
    pub variants: Vec<Vec<Vec<EntryExtentGuard>>>,
    calls: BTreeMap<usize, (usize, usize)>,
}

impl Projection {
    pub fn call_variant(&self, expression: usize, helper: usize) -> Option<usize> {
        self.calls
            .get(&expression)
            .and_then(|(expected, variant)| (*expected == helper).then_some(*variant))
    }
}

pub(super) fn variant_name(base: &str, variant: usize) -> String {
    if variant == 0 {
        base.into()
    } else {
        format!("{base}__entry_{variant}")
    }
}

pub(super) fn helper_coverage_with_verified(
    function: &HostFunction,
    verified_helpers: &[VerifiedHostTensorHelperView<'_>],
) -> Projection {
    let mut walker = Walker::new(&function.tensor_helpers);
    let mut env = BTreeMap::new();
    for param in &function.params {
        env.insert(param.name.clone(), walker.fresh());
    }
    let args = function
        .params
        .iter()
        .filter(|p| matches!(p.ty, HostAbiType::Tensor(_)))
        .map(|p| HostExpr::new(HostExprKind::Var(p.name.clone(), p.ty.clone())))
        .collect::<Vec<_>>();
    let plan = function_entry_plan(function);
    let delegated = delegated_function_guards(function, verified_helpers);
    let mut facts = plan
        .guards()
        .iter()
        .filter(|guard| !delegated.contains(guard))
        .filter_map(|guard| fact(guard, |load, axis| var_read(args.get(load.0)?, axis, &env)))
        .collect();
    walker.walk(&function.body, &env, &mut facts);
    walker.finish()
}

#[cfg(test)]
fn helper_coverage(function: &HostFunction) -> Projection {
    helper_coverage_with_verified(function, &[])
}

pub(super) fn global_helper_coverage(program: &crate::host_abi::HostAbiProgram) -> Projection {
    let mut walker = Walker::new(&program.global_tensor_helpers);
    let mut env = BTreeMap::new();
    let mut facts = Vec::new();
    for binding in &program.globals {
        walker.walk(&binding.value, &env, &mut facts);
        let id = var_read(&binding.value, 0, &env)
            .map(|Read(id, _)| id)
            .unwrap_or_else(|| walker.fresh());
        env.insert(binding.name.clone(), id);
    }
    walker.finish()
}

struct Walker<'a> {
    helpers: &'a [HostTensorHelper],
    projection: Projection,
    expression: usize,
    serial: usize,
}

impl<'a> Walker<'a> {
    fn new(helpers: &'a [HostTensorHelper]) -> Self {
        Self {
            helpers,
            projection: Projection {
                variants: vec![Vec::new(); helpers.len()],
                calls: BTreeMap::new(),
            },
            serial: 0,
            expression: 0,
        }
    }

    fn finish(mut self) -> Projection {
        for variants in &mut self.projection.variants {
            if variants.is_empty() {
                variants.push(Vec::new());
            }
        }
        self.projection
    }

    fn fresh(&mut self) -> usize {
        let value = self.serial;
        self.serial += 1;
        value
    }

    fn callback(&mut self, callback: &HostCallback, env: &BTreeMap<String, usize>, facts: &[Fact]) {
        if let HostCallbackKind::Inline { params, body } = &callback.kind {
            let mut local = env.clone();
            for param in params {
                local.insert(param.name.clone(), self.fresh());
            }
            self.walk(body, &local, &mut facts.to_vec());
        }
    }

    fn walk(&mut self, expr: &HostExpr, env: &BTreeMap<String, usize>, facts: &mut Vec<Fact>) {
        let expression = self.expression;
        self.expression += 1;
        match &expr.kind {
            HostExprKind::ResultClaimScope { body, .. } => {
                self.walk(body, env, facts);
            }
            HostExprKind::FormalIngress { value, .. } => {
                self.walk(value, env, facts);
            }
            HostExprKind::SignatureEntry { plan, args } => {
                for arg in args {
                    self.walk(arg, env, facts);
                }
                facts.extend(plan_facts(plan, args, env));
            }
            HostExprKind::Let { bindings, body, .. }
            | HostExprKind::RetainedInvocation { bindings, body, .. } => {
                let mut local = env.clone();
                let mut known = facts.clone();
                for binding in bindings {
                    self.walk(&binding.value, &local, &mut known);
                    let id = var_read(&binding.value, 0, &local)
                        .map(|Read(id, _)| id)
                        .unwrap_or_else(|| self.fresh());
                    local.insert(binding.name.clone(), id);
                }
                self.walk(body, &local, &mut known);
            }
            HostExprKind::TensorCall { helper, args, .. } => {
                for arg in args {
                    self.walk(arg, env, facts);
                }
                let source = &self.helpers[*helper];
                let read = |load, axis| {
                    let RiscOp::Load { name } = &source.dag.get(load)?.op else {
                        return None;
                    };
                    let index = source
                        .inputs
                        .iter()
                        .position(|input| input.name == name.as_str())?;
                    var_read(args.get(index)?, axis, env)
                };
                let mut guards = chelis_ir::axis_sources::entry_extent_guards(&source.dag);
                // Literal ABI comparisons precede the DAG's dynamic guards.
                // Retain them unless entry checked that same input axis/value.
                for node in source.dag.nodes() {
                    if matches!(node.op, RiscOp::Load { .. }) {
                        for (axis, dim) in node.output_type.dims.iter().enumerate() {
                            let required = match dim {
                                DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => Some(*n),
                                _ => None,
                            };
                            if let Some(required) = required {
                                guards.push(EntryExtentGuard::Literal {
                                    required,
                                    observed: (node.id, axis),
                                });
                            }
                        }
                    }
                }
                guards
                    .retain(|guard| fact(guard, read).is_some_and(|check| implies(facts, &check)));
                let variants = &mut self.projection.variants[*helper];
                let variant = variants
                    .iter()
                    .position(|previous| *previous == guards)
                    .unwrap_or_else(|| {
                        variants.push(guards);
                        variants.len() - 1
                    });
                self.projection.calls.insert(expression, (*helper, variant));
            }
            HostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                self.walk(cond, env, facts);
                self.walk(then_expr, env, &mut facts.clone());
                self.walk(else_expr, env, &mut facts.clone());
            }
            HostExprKind::MatchOption {
                scrutinee,
                bind_name,
                some_expr,
                none_expr,
                ..
            } => {
                self.walk(scrutinee, env, facts);
                let mut local = env.clone();
                local.insert(bind_name.clone(), self.fresh());
                self.walk(some_expr, &local, &mut facts.clone());
                self.walk(none_expr, env, &mut facts.clone());
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ..
            } => {
                self.walk(scrutinee, env, facts);
                for arm in arms {
                    let mut local = env.clone();
                    for binding in &arm.bindings {
                        local.insert(binding.name.clone(), self.fresh());
                    }
                    self.walk(&arm.expr, &local, &mut facts.clone());
                }
                if let Some(body) = default_expr {
                    self.walk(body, env, &mut facts.clone());
                }
            }
            HostExprKind::Call { args, .. }
            | HostExprKind::Builtin { args, .. }
            | HostExprKind::List(args, _)
            | HostExprKind::Tuple(args, _)
            | HostExprKind::AdtConstruct { fields: args, .. } => {
                for arg in args {
                    self.walk(arg, env, facts);
                }
            }
            HostExprKind::AdtFieldAccess { base, .. } => self.walk(base, env, facts),
            HostExprKind::Map { callback, list, .. }
            | HostExprKind::Filter { callback, list, .. }
            | HostExprKind::Partition { callback, list, .. }
            | HostExprKind::FlatMap { callback, list, .. } => {
                self.walk(list, env, facts);
                self.callback(callback, env, facts);
            }
            HostExprKind::Fold {
                callback,
                init,
                list,
                ..
            }
            | HostExprKind::Scan {
                callback,
                init,
                list,
                ..
            } => {
                self.walk(init, env, facts);
                self.walk(list, env, facts);
                self.callback(callback, env, facts);
            }
            HostExprKind::WithSeed { seed, body, .. } => {
                self.walk(seed, env, facts);
                self.walk(body, env, facts);
            }
            HostExprKind::Int(_)
            | HostExprKind::Float(_)
            | HostExprKind::Bool(_)
            | HostExprKind::String(_)
            | HostExprKind::Var(_, _)
            | HostExprKind::Unit => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::host::{HostFunctionOrigin, HostParam};
    use chelis_ir::{Dag, TensorType};
    use chelis_types::types::Prim;

    fn verified_host_from_source(source: &str) -> chelis_ir::ownership::VerifiedHostProgram {
        let declarations = chelis_surf::parser::parse_str(source).expect("parse host source");
        let deep = chelis_surf::desugar::desugar_program(&declarations)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_typed_program(&deep)
            .unwrap_or_else(|errors| panic!("check host source: {:?}", errors.errors));
        let checked = chelis_effects::check_program(&checked).expect("effects host source");
        let checked = chelis_types::check_linearity(&checked).expect("linearity host source");
        let realizability = chelis_effects::realizability::infer_realizability(
            &checked,
            crate::TENSOR_CAPABLE_PRIMS,
        );
        let manifest =
            chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
        let lowered =
            chelis_ir::host::try_lower_compiled_program_with_manifest(&checked, &manifest)
                .expect("lower host source");
        let host =
            crate::prepare_host_program_for_codegen(lowered.host.expect("source uses host lane"))
                .expect("select C host payload");
        let manifested = chelis_types::manifest::ManifestedProgram::new(
            checked,
            manifest,
            chelis_types::types::Target::C,
        );
        chelis_ir::ownership::verify_ownership(
            chelis_ir::ownership::lower_host_ownership(&manifested, host)
                .expect("lower host ownership"),
        )
        .expect("verify host ownership")
    }

    fn tensor(name: &str) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Named(name.into(), None)],
            precision: Prim::F32,
        }
    }

    fn var(name: &str) -> HostExpr {
        HostExpr::new(HostExprKind::Var(
            name.into(),
            HostAbiType::Tensor(tensor("seq")),
        ))
    }

    fn function() -> HostFunction {
        let ty = tensor("seq");
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        let sum = dag.add_node(RiscOp::Add, vec![a, b], ty.clone(), None);
        dag.add_root(sum);
        HostFunction {
            helper_result_claim_axes: Vec::new(),
            name: "entry".into(),
            params: ["a", "b"]
                .into_iter()
                .map(|name| HostParam {
                    name: name.into(),
                    ty: HostAbiType::Tensor(ty.clone()),
                })
                .collect(),
            ret_ty: HostAbiType::Tensor(ty.clone()),
            body: HostExpr::new(HostExprKind::TensorCall {
                helper: 0,
                args: vec![var("a"), var("b")],
                ty: HostAbiType::Tensor(ty.clone()),
            }),
            tensor_helpers: vec![HostTensorHelper {
                name: "helper".into(),
                dag,
                inputs: ["a", "b"]
                    .into_iter()
                    .map(|name| chelis_ir::host::HostTensorInput {
                        name: name.into(),
                        ty: ty.clone(),
                    })
                    .collect(),
                output: ty,
                specialization: None,
                summary_rejection: None,
            }],
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
            inherits_random: false,
        }
    }

    #[test]
    fn dominating_signature_discharges_the_exact_helper_comparison() {
        let function = function();
        let required =
            chelis_ir::axis_sources::entry_extent_guards(&function.tensor_helpers[0].dag);
        assert_eq!(required.len(), 1);
        assert!(delegated_function_guards(&function, &[]).is_empty());
        assert_eq!(helper_coverage(&function).variants, vec![vec![required]]);
    }

    #[test]
    fn lowered_mixed_result_claim_delegates_the_following_rows_guard() {
        let verified = verified_host_from_source(
            "def f(z: tensor[rows, f32], a: tensor[cols, f32], q: tensor[rows, f32]) -> tensor[rows, 3, f32] = insert(add(z, q), 1i32, shape(a, 0i32))",
        );
        let projected = crate::host_abi::project_program(verified.emission()).unwrap();
        let function = &projected.program().functions[0];
        let helper = projected.function_tensor_helper(0, 0).unwrap();
        let plan = function_entry_plan(function);
        assert_eq!(plan.guards().len(), 1);
        let helper_guards = helper.dag().entry_extent_guards();
        assert!(
            matches!(helper_guards.as_slice(), [
                EntryExtentGuard::Literal { required: 3, .. },
                EntryExtentGuard::Named { claim, .. }
            ] if claim == "rows"),
            "{helper_guards:#?}"
        );
        assert_eq!(
            delegated_function_guards(function, &[helper]),
            plan.guards()
        );
        assert!(helper_coverage_with_verified(function, &[helper]).variants[0][0].is_empty());
    }

    #[test]
    fn shadowing_a_witness_keeps_the_helper_check() {
        let mut function = function();
        function.params.push(HostParam {
            name: "other".into(),
            ty: HostAbiType::Tensor(tensor("independent")),
        });
        function.body = HostExpr::new(HostExprKind::Let {
            bindings: vec![HostBinding {
                name: "b".into(),
                display_name: None,
                display_roots: Vec::new(),
                ty: HostAbiType::Tensor(tensor("independent")),
                value: var("other"),
            }],
            ty: function.ret_ty.clone(),
            body: Box::new(function.body),
        });
        assert!(helper_coverage(&function).variants[0][0].is_empty());
    }

    #[test]
    fn helper_projection_removes_only_discharged_checks_and_keeps_standalone_entry() {
        let function = function();
        let options = crate::CodegenOptions {
            use_blas: false,
            math_lib_override: Some(crate::MathLib::None),
            static_entry: false,
        };
        let dag = crate::testing::verified_dag(&function.tensor_helpers[0].dag, options).unwrap();
        let projection = helper_coverage(&function);
        let discharged = CEmitter::emit_verified_dag_with_options(
            dag.emission(),
            "dominated",
            options,
            &projection.variants[0][0],
        )
        .unwrap();
        assert!(
            !discharged.contains("numeric trap: domain in load at i64"),
            "{discharged}"
        );
        let full =
            CEmitter::emit_verified_dag_with_options(dag.emission(), "unguarded", options, &[])
                .unwrap();
        assert_eq!(
            full.matches("numeric trap: domain in load at i64").count(),
            1,
            "{full}"
        );
        let standalone = crate::codegen_with_options(dag, "standalone", options).unwrap();
        assert_eq!(
            standalone
                .c_source
                .matches("numeric trap: domain in load at i64")
                .count(),
            1,
            "{}",
            standalone.c_source
        );
    }

    #[test]
    fn unused_canonical_witness_discharges_other_pairs_transitively() {
        let mut function = function();
        function.params.insert(
            0,
            HostParam {
                name: "unused".into(),
                ty: HostAbiType::Tensor(tensor("seq")),
            },
        );
        assert_eq!(helper_coverage(&function).variants[0][0].len(), 1);
    }

    #[test]
    fn cloned_payload_keeps_each_calls_structural_identity() {
        let original = function();
        let copy = original.clone();
        let before = helper_coverage(&original);
        let after = helper_coverage(&copy);
        assert_eq!(before.calls, after.calls);
        assert_eq!(before.variants, after.variants);
        assert_eq!(before.call_variant(0, 0), Some(0));
        assert_eq!(before.call_variant(1, 0), None);
        assert_eq!(before.call_variant(0, 1), None);
    }

    #[test]
    fn every_helper_call_must_be_dominated_before_a_check_is_removed() {
        let mut function = function();
        let plan = function_entry_plan(&function);
        for param in &mut function.params {
            param.ty = HostAbiType::Tensor(tensor("*"));
        }
        let guarded = HostExpr::new(HostExprKind::Let {
            bindings: vec![HostBinding {
                name: "checked".into(),
                display_name: None,
                display_roots: Vec::new(),
                ty: HostAbiType::Unit,
                value: HostExpr::new(HostExprKind::SignatureEntry {
                    plan,
                    args: vec![var("a"), var("b")],
                }),
            }],
            ty: function.ret_ty.clone(),
            body: Box::new(function.body.clone()),
        });
        function.body = guarded.clone();
        assert_eq!(helper_coverage(&function).variants[0][0].len(), 1);
        function.body = HostExpr::new(HostExprKind::Tuple(
            vec![
                guarded,
                HostExpr::new(HostExprKind::TensorCall {
                    helper: 0,
                    args: vec![var("a"), var("b")],
                    ty: function.ret_ty.clone(),
                }),
            ],
            HostAbiType::Tuple(vec![function.ret_ty.clone(), function.ret_ty.clone()]),
        ));
        let projection = helper_coverage(&function);
        assert_eq!(projection.variants[0].len(), 2);
        assert_eq!(projection.variants[0][0].len(), 1);
        assert!(projection.variants[0][1].is_empty());
        assert_eq!(
            projection.calls.values().copied().collect::<Vec<_>>(),
            vec![(0, 0), (0, 1)]
        );
    }
}
