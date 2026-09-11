//! Ordered host scalar sources inside an otherwise ordinary tensor graph.
//!
//! Claim construction sees the complete graph. Only this boundary turns it
//! into independently executable helpers; a placeholder never reaches one.
use super::{HostParam, HostTypeTerm};
use crate::dag::{Dag, NodeId, RiscOp, TensorType};
use chelis_deep::Expr;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HostValueId(pub(crate) usize);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagingStatus {
    Searching,
    HasSources,
    HostControlBoundary,
}

pub(crate) enum StagingAttempt<T> {
    NotApplicable,
    HostControlBoundary,
    Ready(T),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum StageValue {
    Tensor(NodeId),
    Host(HostValueId),
}

#[derive(Debug, Clone)]
pub(crate) struct HostSource {
    pub before: usize,
    pub value: StageValue,
    pub ty: HostTypeTerm,
    pub expression: Expr,
    pub captures: Vec<(String, StageValue, HostTypeTerm)>,
}

#[derive(Debug, Clone)]
pub struct HostStageCapture {
    pub binding: String,
    pub value: String,
    pub ty: HostTypeTerm,
}

#[derive(Debug, Clone)]
pub enum HostStage {
    Source {
        expression: Expr,
        captures: Vec<HostStageCapture>,
        output: String,
        ty: HostTypeTerm,
    },
    Kernel {
        dag: Dag,
        outputs: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct HostStagedPlan {
    stages: Vec<HostStage>,
    output: String,
}

impl HostStagedPlan {
    pub fn stages(&self) -> &[HostStage] {
        &self.stages
    }
    pub fn output(&self) -> &str {
        &self.output
    }
}

pub(crate) fn partition(
    logical: &Dag,
    sources: &[HostSource],
    params: &[HostParam],
    external_inputs: &BTreeMap<NodeId, String>,
    host_parameters: &BTreeMap<HostValueId, (String, HostTypeTerm)>,
    evaluation: Option<&mut crate::evaluation::StagedEvaluationPlan>,
) -> Result<HostStagedPlan, String> {
    let [root] = logical.roots() else {
        return Err("a staged tensor region must have exactly one tensor result".into());
    };
    let mut reserved = params
        .iter()
        .map(|p| p.name.clone())
        .collect::<BTreeSet<_>>();
    for source in sources {
        let mut names = chelis_unord::UnordSet::new();
        super::collect_deep_var_names(&source.expression, &mut names);
        reserved.extend(names.into_sorted());
    }
    let mut names = BTreeMap::new();
    for node in logical.nodes() {
        let mut candidate = format!("__checked_stage_{}", node.id.0);
        while reserved.contains(&candidate) {
            candidate.push('_');
        }
        reserved.insert(candidate.clone());
        names.insert(StageValue::Tensor(node.id), candidate);
    }
    for (id, (name, _)) in host_parameters {
        names.insert(StageValue::Host(*id), name.clone());
    }
    let mut host_types = host_parameters
        .iter()
        .map(|(id, (_, ty))| (*id, ty))
        .collect::<BTreeMap<_, _>>();
    let mut producers = BTreeSet::new();
    for source in sources {
        if !producers.insert(source.value) {
            return Err("a staged value has more than one producer".into());
        }
        match source.value {
            StageValue::Tensor(id) => {
                if source.before != id.0
                    || external_inputs.contains_key(&id)
                    || source.ty
                        != super::HostTypeTerm::Scalar(super::HostPrecisionTerm::Concrete(
                            chelis_types::types::Prim::Int64,
                        ))
                    || logical.get(id).is_none_or(|node| {
                        node.output_type != scalar_type() || !matches!(node.op, RiscOp::Load { .. })
                    })
                {
                    return Err("a staged reshape target must have one exact int64 producer".into());
                }
            }
            StageValue::Host(id) => {
                if host_parameters.contains_key(&id) {
                    return Err("a host parameter cannot be overwritten by a stage".into());
                }
                let mut candidate = format!("__checked_host_{}", id.0);
                while reserved.contains(&candidate) {
                    candidate.push('_');
                }
                reserved.insert(candidate.clone());
                names.insert(source.value, candidate);
                host_types.insert(id, &source.ty);
            }
        }
    }
    for node in logical.nodes() {
        if let RiscOp::Load { name } = &node.op {
            let parameter = external_inputs
                .get(&node.id)
                .is_some_and(|declared| declared == name.as_str());
            let source = producers.contains(&StageValue::Tensor(node.id));
            if parameter == source {
                return Err(
                    "every staged graph input needs exactly one declared or host producer".into(),
                );
            }
        }
    }

    let mut partition = Partition {
        logical,
        sources,
        names,
        reserved,
        available: host_parameters
            .keys()
            .map(|id| StageValue::Host(*id))
            .collect(),
        stages: Vec::new(),
        evaluation,
    };
    let mut start = 0;
    for source in sources {
        if source.before < start || source.before > logical.nodes().len() {
            return Err("a staged source requires one ordered producer".into());
        }
        partition.append_kernel(start, source.before)?;
        let captures = source
            .captures
            .iter()
            .map(|(binding, value, ty)| {
                if !partition.available.contains(value) {
                    return Err(format!("staged capture `{binding}` precedes its producer"));
                }
                let matches = match value {
                    StageValue::Tensor(id) => logical.get(*id).is_some_and(|node| match ty {
                        HostTypeTerm::Tensor(expected) => {
                            expected.precision == node.output_type.precision
                                && expected.dims.len() == node.output_type.dims.len()
                        }
                        HostTypeTerm::Scalar(super::HostPrecisionTerm::Concrete(precision)) => {
                            *precision == node.output_type.precision
                                && node.output_type.dims.is_empty()
                        }
                        _ => false,
                    }),
                    StageValue::Host(id) => host_types.get(id).is_some_and(|actual| *actual == ty),
                };
                if !matches {
                    return Err(format!(
                        "staged capture `{binding}` has a different producer type"
                    ));
                }
                Ok(HostStageCapture {
                    binding: binding.clone(),
                    value: partition.names[value].clone(),
                    ty: ty.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        partition.stages.push(HostStage::Source {
            expression: source.expression.clone(),
            captures,
            output: partition.names[&source.value].clone(),
            ty: source.ty.clone(),
        });
        partition.available.insert(source.value);
        start = source.before + usize::from(matches!(source.value, StageValue::Tensor(_)));
    }
    partition.append_kernel(start, logical.nodes().len())?;
    if !partition.available.contains(&StageValue::Tensor(*root)) {
        return Err("a staged tensor result has no executed producer".into());
    }
    Ok(HostStagedPlan {
        stages: partition.stages,
        output: partition.names[&StageValue::Tensor(*root)].clone(),
    })
}

struct Partition<'a> {
    logical: &'a Dag,
    sources: &'a [HostSource],
    names: BTreeMap<StageValue, String>,
    reserved: BTreeSet<String>,
    available: BTreeSet<StageValue>,
    stages: Vec<HostStage>,
    evaluation: Option<&'a mut crate::evaluation::StagedEvaluationPlan>,
}

impl Partition<'_> {
    fn append_kernel(&mut self, start: usize, end: usize) -> Result<(), String> {
        if start == end {
            return Ok(());
        }
        // Only values read after this cut cross the ABI. In particular, an
        // intermediate consumed inside this segment is not kept as an output.
        let mut exports = self
            .logical
            .roots()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for node in &self.logical.nodes()[end..] {
            exports.extend(node.inputs.iter().chain(&node.shape_deps).copied());
        }
        for source in self.sources.iter().filter(|source| source.before >= end) {
            exports.extend(
                source
                    .captures
                    .iter()
                    .filter_map(|(_, value, _)| match value {
                        StageValue::Tensor(id) => Some(*id),
                        StageValue::Host(_) => None,
                    }),
            );
        }
        let mut dag = Dag::new();
        let mut remap = BTreeMap::new();
        let mut outputs = Vec::new();
        for node in &self.logical.nodes()[start..end] {
            for &dependency in node.inputs.iter().chain(&node.shape_deps) {
                if let std::collections::btree_map::Entry::Vacant(entry) = remap.entry(dependency) {
                    if !self.available.contains(&StageValue::Tensor(dependency)) {
                        return Err(
                            "an executable staged helper contains an unresolved input".into()
                        );
                    }
                    let ty = self
                        .logical
                        .get(dependency)
                        .ok_or("missing staged input")?
                        .output_type
                        .clone();
                    entry.insert(dag.add_node(
                        RiscOp::Load {
                            name: self.names[&StageValue::Tensor(dependency)].as_str().into(),
                        },
                        Vec::new(),
                        ty,
                        None,
                    ));
                }
            }
            let id = dag.add_node(
                node.op.clone(),
                node.inputs.iter().map(|i| remap[i]).collect(),
                node.output_type.clone(),
                node.span_id.clone(),
            );
            let copied = dag.node_mut(id).expect("new staged node");
            copied.merged_spans = node.merged_spans.clone();
            copied.shape_deps = node.shape_deps.iter().map(|i| remap[i]).collect();
            copied.reusable_input = node
                .reusable_input
                .and_then(|input| remap.get(&input).copied());
            remap.insert(node.id, id);
            if exports.contains(&node.id) {
                self.available.insert(StageValue::Tensor(node.id));
                dag.add_root(id);
                outputs.push(self.names[&StageValue::Tensor(node.id)].clone());
            }
        }
        // A host region evaluates each preceding expression even when its
        // value is unused. A completion root retains that execution without
        // exporting every intermediate tensor or replaying a growing prefix.
        let dependencies = dag.nodes().iter().map(|node| node.id).collect();
        let completed = dag.add_node(
            RiscOp::Const {
                value: chelis_types::scalar_from_i64("const", chelis_types::types::Prim::Int64, 0)
                    .expect("exact completion value"),
            },
            Vec::new(),
            scalar_type(),
            None,
        );
        dag.node_mut(completed).expect("completion root").shape_deps = dependencies;
        dag.add_root(completed);
        let mut completion_name = format!("__checked_complete_{end}");
        while self.reserved.contains(&completion_name) {
            completion_name.push('_');
        }
        self.reserved.insert(completion_name.clone());
        outputs.push(completion_name);
        if let Some(evaluation) = &mut self.evaluation {
            // Imported values become Loads, not second executions of their
            // producers. Only this segment's own nodes transport Random sites.
            let local = remap
                .into_iter()
                .filter(|(node, _)| (start..end).contains(&node.0))
                .collect();
            evaluation.append_segment(dag.clone(), &local);
        }
        self.stages.push(HostStage::Kernel { dag, outputs });
        Ok(())
    }
}

pub(crate) fn scalar_type() -> TensorType {
    TensorType {
        dims: Vec::new(),
        precision: chelis_types::types::Prim::Int64,
    }
}

/// Static function references stay direct calls in C. The plan retains the
/// reference at its original binding position; resolving that captured value
/// does not create a first-class function object or guess from a later name.
pub(super) fn resolve_callable_aliases(
    expr: &mut super::HostExpr,
    aliases: &BTreeMap<String, String>,
) {
    use super::{HostCallback, HostCallbackKind, HostExprKind};
    fn callback(callback: &mut HostCallback, aliases: &BTreeMap<String, String>) {
        match &mut callback.kind {
            HostCallbackKind::Named { function, .. } => {
                if let Some(resolved) = aliases.get(function) {
                    *function = resolved.clone();
                }
            }
            HostCallbackKind::Inline { params, body } => {
                let mut inner = aliases.clone();
                for param in params {
                    inner.remove(&param.name);
                }
                resolve_callable_aliases(body, &inner);
            }
        }
    }
    match &mut expr.kind {
        HostExprKind::Var(name, _) => {
            if let Some(resolved) = aliases.get(name) {
                *name = resolved.clone();
            }
        }
        HostExprKind::Call { function, args, .. } => {
            if let Some(resolved) = aliases.get(function) {
                *function = resolved.clone();
            }
            for arg in args {
                resolve_callable_aliases(arg, aliases);
            }
        }
        HostExprKind::Builtin { args, .. }
        | HostExprKind::TensorCall { args, .. }
        | HostExprKind::AdtConstruct { fields: args, .. }
        | HostExprKind::List(args, _)
        | HostExprKind::Tuple(args, _) => {
            for arg in args {
                resolve_callable_aliases(arg, aliases);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => resolve_callable_aliases(base, aliases),
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            resolve_callable_aliases(cond, aliases);
            resolve_callable_aliases(then_expr, aliases);
            resolve_callable_aliases(else_expr, aliases);
        }
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ..
        } => {
            resolve_callable_aliases(scrutinee, aliases);
            resolve_callable_aliases(none_expr, aliases);
            let mut inner = aliases.clone();
            inner.remove(bind_name);
            resolve_callable_aliases(some_expr, &inner);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            resolve_callable_aliases(scrutinee, aliases);
            if let Some(default) = default_expr {
                resolve_callable_aliases(default, aliases);
            }
            for arm in arms {
                let mut inner = aliases.clone();
                for binding in &arm.bindings {
                    inner.remove(&binding.name);
                }
                resolve_callable_aliases(&mut arm.expr, &inner);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            let mut inner = aliases.clone();
            for binding in bindings {
                resolve_callable_aliases(&mut binding.value, &inner);
                inner.remove(&binding.name);
            }
            resolve_callable_aliases(body, &inner);
        }
        HostExprKind::Map {
            callback: cb, list, ..
        }
        | HostExprKind::Filter {
            callback: cb, list, ..
        }
        | HostExprKind::Partition {
            callback: cb, list, ..
        }
        | HostExprKind::FlatMap {
            callback: cb, list, ..
        } => {
            callback(cb, aliases);
            resolve_callable_aliases(list, aliases);
        }
        HostExprKind::Fold {
            callback: cb,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback: cb,
            init,
            list,
            ..
        } => {
            callback(cb, aliases);
            resolve_callable_aliases(init, aliases);
            resolve_callable_aliases(list, aliases);
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            resolve_callable_aliases(seed, aliases);
            resolve_callable_aliases(body, aliases);
        }
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimInfo, RtAxis, RtDim};
    use crate::host_type_state::HostPrecisionTerm;
    use chelis_types::{scalar_from_i64, types::Prim};

    fn fixture() -> (
        Dag,
        Vec<HostSource>,
        Vec<HostParam>,
        BTreeMap<NodeId, String>,
    ) {
        let mut dag = Dag::new();
        let tensor = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor.clone(),
            None,
        );
        let required = dag.add_node(
            RiscOp::Const {
                value: scalar_from_i64("reshape", Prim::Int64, 2).unwrap(),
            },
            vec![],
            scalar_type(),
            None,
        );
        let actual = dag.add_node(
            RiscOp::Load {
                name: "host-source".into(),
            },
            vec![],
            scalar_type(),
            None,
        );
        let checked = dag.add_node(
            RiscOp::CheckedReshapeExtent {
                claims: vec!["2".into()],
                axis: RtAxis::Lit(0),
            },
            vec![actual, required],
            scalar_type(),
            None,
        );
        let result = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1), RtDim::Lit(2)],
            },
            vec![x, checked],
            TensorType {
                dims: vec![DimInfo::Named("actual".into(), None), DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(result);
        let sources = vec![HostSource {
            before: actual.0,
            value: StageValue::Tensor(actual),
            ty: HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int64)),
            expression: chelis_deep::parser::parse_str("(lit {type: (t-prim {} int64)} 2)")
                .unwrap()
                .remove(0),
            captures: vec![(
                "x".into(),
                StageValue::Tensor(x),
                HostTypeTerm::Tensor(tensor.clone()),
            )],
        }];
        (
            dag,
            sources,
            vec![HostParam {
                name: "x".into(),
                ty: HostTypeTerm::Tensor(tensor),
            }],
            BTreeMap::from([(x, "x".into())]),
        )
    }

    #[test]
    fn staged_plan_has_one_source_and_no_unresolved_helper_inputs() {
        let (dag, sources, params, inputs) = fixture();
        let plan = partition(&dag, &sources, &params, &inputs, &BTreeMap::new(), None).unwrap();
        assert_eq!(plan.stages().len(), 3);
        let mut available = BTreeSet::from(["x".to_owned()]);
        for stage in plan.stages() {
            match stage {
                HostStage::Source {
                    captures, output, ..
                } => {
                    assert!(
                        captures
                            .iter()
                            .all(|capture| available.contains(&capture.value))
                    );
                    assert!(available.insert(output.clone()));
                }
                HostStage::Kernel { dag, outputs } => {
                    assert!(crate::verify::verify(dag).is_empty());
                    for node in dag.nodes() {
                        if let RiscOp::Load { name } = &node.op {
                            assert!(available.contains(name.as_str()));
                        }
                    }
                    for output in outputs {
                        assert!(available.insert(output.clone()));
                    }
                }
            }
        }
        assert!(available.contains(plan.output()));
        // Claims and binder spelling do not establish an actual extent. A
        // capture only requires the producer's rank and numeric carrier here;
        // independently executed checked extents enforce shape obligations.
        let mut refined = sources.clone();
        refined[0].captures[0].2 = HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Named("other".into(), Some(2))],
            precision: Prim::F32,
        });
        partition(&dag, &refined, &params, &inputs, &BTreeMap::new(), None).unwrap();
    }

    #[test]
    fn staged_plan_rejects_missing_duplicate_wrong_type_and_forward_producers() {
        let (dag, sources, params, inputs) = fixture();
        for mutation in [
            "missing",
            "duplicate",
            "dtype",
            "forward",
            "parameter_alias",
            "capture_rank",
            "capture_dtype",
            "capture_scalar",
            "capture_host",
        ] {
            let mut sources = sources.clone();
            let mut inputs = inputs.clone();
            let mut host_parameters = BTreeMap::new();
            match mutation {
                "missing" => sources.clear(),
                "duplicate" => sources.push(sources[0].clone()),
                "dtype" => {
                    sources[0].ty = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int32))
                }
                "capture_rank" => {
                    sources[0].captures[0].2 = HostTypeTerm::Tensor(TensorType {
                        dims: vec![],
                        precision: Prim::F32,
                    });
                }
                "capture_dtype" => {
                    sources[0].captures[0].2 = HostTypeTerm::Tensor(TensorType {
                        dims: vec![DimInfo::Named("n".into(), None)],
                        precision: Prim::F64,
                    });
                }
                "capture_scalar" => {
                    sources[0].captures[0].2 =
                        HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::F32));
                }
                "capture_host" => {
                    let host = HostValueId(0);
                    host_parameters.insert(
                        host,
                        (
                            "items".into(),
                            HostTypeTerm::List(Box::new(HostTypeTerm::Scalar(
                                HostPrecisionTerm::Concrete(Prim::Int64),
                            ))),
                        ),
                    );
                    sources[0].captures[0].1 = StageValue::Host(host);
                }
                "forward" => sources[0].captures[0].1 = StageValue::Tensor(dag.roots()[0]),
                "parameter_alias" => {
                    inputs.insert(NodeId(sources[0].before), "host-source".into());
                }
                _ => unreachable!(),
            }
            assert!(
                partition(&dag, &sources, &params, &inputs, &host_parameters, None).is_err(),
                "{mutation}"
            );
        }
    }
}
