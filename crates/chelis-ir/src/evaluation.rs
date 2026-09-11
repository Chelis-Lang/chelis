//! Source-owned evaluation plans for fixed-control Random execution.
//!
//! Ordinary [`Dag`] serialization is deliberately not an execution-plan
//! transport. Forward entry, handler scope, and backward replay identity are
//! recorded by lowering/AD, never recovered from an ordinary graph's shape.

use chelis_types::dtype_semantics::PreparedDropout;
use chelis_types::{RawScalar, cast_raw};
use chelis_unord::UnordMap;

use crate::dag::{Dag, NodeId, RiscOp};
use crate::eval::TensorValue;
use crate::host::RandomLoweringState;

/// Source admission is decided before evaluation. Legacy selection is never
/// an error-recovery path for a damaged or unsupported execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvaluationProfile {
    FixedControl,
    Legacy(LegacyEvaluationReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LegacyEvaluationReason {
    LegacyApi,
    NoDropout,
    RuntimeRate,
    RuntimeSeed,
    ResourceScope,
    RandomVmap,
    HigherOrderAd,
    DynamicControl,
    RecursiveControl,
}

/// Mutable inherited stream state. An evaluator updates this state when a
/// forward primitive enters, including when subsequent evaluation fails.
/// The existing host carrier stores the ordinal modulo 2^64; this is not an
/// unbounded source-event count or an assertion of equality to a proof's Nat.
#[derive(Debug, Clone)]
pub struct RandomExecutionContext {
    state: RandomLoweringState,
}

impl RandomExecutionContext {
    pub fn new(state: RandomLoweringState) -> Self {
        Self { state }
    }

    pub fn state(&self) -> RandomLoweringState {
        self.state
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScopeId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrawId(pub(crate) usize);

#[derive(Debug, Clone)]
pub(crate) struct Scope {
    pub(crate) seed: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RandomSite {
    Forward { draw: DrawId, scope: ScopeId },
    Replay { draw: DrawId },
}

/// Lowering owns these associations. They are not a public authoring API.
#[derive(Debug, Clone)]
pub(crate) struct ExecutionMetadata {
    pub(crate) scopes: Vec<Scope>,
    pub(crate) sites: UnordMap<NodeId, RandomSite>,
    pub(crate) draws: usize,
    pub(crate) order: Vec<NodeId>,
}

impl ExecutionMetadata {
    pub(crate) fn new(seed: Option<u64>) -> Self {
        Self {
            scopes: vec![Scope { seed }],
            sites: UnordMap::new(),
            draws: 0,
            order: Vec::new(),
        }
    }

    pub(crate) fn forward(&mut self, node: NodeId, scope: ScopeId) {
        let draw = DrawId(self.draws);
        self.draws += 1;
        assert!(
            self.sites
                .insert(node, RandomSite::Forward { draw, scope })
                .is_none()
        );
    }

    pub(crate) fn remap(&mut self, remap: &UnordMap<NodeId, NodeId>) -> Result<(), String> {
        self.sites = self
            .sites
            .to_sorted()
            .into_iter()
            .map(|(node, site)| {
                remap
                    .get(node)
                    .copied()
                    .map(|mapped| (mapped, *site))
                    .ok_or_else(|| {
                        format!(
                            "evaluation plan lost random node {} during remapping",
                            node.0
                        )
                    })
            })
            .collect::<Result<_, _>>()?;
        self.order = self
            .order
            .iter()
            .map(|node| {
                remap.get(node).copied().ok_or_else(|| {
                    format!(
                        "evaluation plan lost scheduled node {} during remapping",
                        node.0
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    pub(crate) fn merge_child(
        &mut self,
        mut child: Self,
        inherited: ScopeId,
        remap: &UnordMap<NodeId, NodeId>,
    ) -> Result<(), String> {
        if child.scopes[0].seed != self.scopes[inherited.0].seed {
            return Err("evaluation child scope changed its inherited raw seed".into());
        }
        child.remap(remap)?;
        let scope_offset = self.scopes.len();
        let draw_offset = self.draws;
        self.draws += child.draws;
        self.scopes.extend(child.scopes.into_iter().skip(1));
        for (node, site) in child.sites.into_sorted() {
            let site = match site {
                RandomSite::Forward { draw, scope } => RandomSite::Forward {
                    draw: DrawId(draw.0 + draw_offset),
                    scope: if scope.0 == 0 {
                        inherited
                    } else {
                        ScopeId(scope_offset + scope.0 - 1)
                    },
                },
                RandomSite::Replay { draw } => RandomSite::Replay {
                    draw: DrawId(draw.0 + draw_offset),
                },
            };
            if self.sites.insert(node, site).is_some() {
                return Err("evaluation child overwrote an existing random site".into());
            }
        }
        Ok(())
    }

    pub(crate) fn selected(&self, remap: &UnordMap<NodeId, NodeId>) -> Result<Self, String> {
        let mut selected = Self::new(self.scopes[0].seed);
        selected.scopes = self.scopes.clone();
        let mut draws = UnordMap::new();
        for node in &self.order {
            if !remap.contains_key(node) {
                continue;
            }
            if let Some(RandomSite::Forward { draw, .. }) = self.sites.get(node) {
                draws.insert(draw.0, DrawId(selected.draws));
                selected.draws += 1;
            }
        }
        for (node, site) in self.sites.to_sorted() {
            let Some(&mapped) = remap.get(node) else {
                continue;
            };
            let site = match site {
                RandomSite::Forward { draw, scope } => RandomSite::Forward {
                    draw: *draws
                        .get(&draw.0)
                        .ok_or("selected plan lost its forward draw")?,
                    scope: *scope,
                },
                RandomSite::Replay { draw } => RandomSite::Replay {
                    draw: *draws
                        .get(&draw.0)
                        .ok_or("selected replay has no selected forward draw")?,
                },
            };
            selected.sites.insert(mapped, site);
        }
        selected.order = self
            .order
            .iter()
            .filter_map(|node| remap.get(node).copied())
            .collect();
        Ok(selected)
    }
}

/// An immutable, non-serialized graph together with its source execution
/// order and actual forward/replay associations.
#[derive(Debug, Clone)]
pub struct EvaluationPlan {
    dag: Dag,
    metadata: ExecutionMetadata,
}

impl EvaluationPlan {
    /// Inspect the graph's types and value roots. Cloning/serializing this
    /// view does not preserve the plan and cannot recreate one.
    pub fn dag_for_inspection(&self) -> &Dag {
        &self.dag
    }

    pub(crate) fn new(dag: Dag, metadata: ExecutionMetadata) -> Result<Self, String> {
        let plan = Self { dag, metadata };
        plan.validate()?;
        Ok(plan)
    }

    pub(crate) fn rebind_dimensions(mut self, dag: Dag) -> Result<Self, String> {
        if self.dag.len() != dag.len() || self.dag.roots() != dag.roots() {
            return Err("evaluation dimension rebinding changed graph identities".into());
        }
        self.dag = dag;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), String> {
        if self.metadata.scopes.is_empty() {
            return Err("evaluation plan has no inherited scope".into());
        }
        let mut seen = vec![false; self.dag.len()];
        let mut draws = vec![None; self.metadata.draws];
        for &id in &self.metadata.order {
            let node = self
                .dag
                .get(id)
                .ok_or("evaluation plan schedules a missing node")?;
            if seen[id.0] {
                return Err("evaluation plan schedules a node twice".into());
            }
            if node
                .inputs
                .iter()
                .chain(&node.shape_deps)
                .any(|input| !seen.get(input.0).copied().unwrap_or(false))
            {
                return Err(format!(
                    "evaluation plan schedules node {} before an operand",
                    id.0
                ));
            }
            let random = matches!(node.op, RiscOp::Dropout { .. } | RiscOp::UniformLike { .. });
            match (random, self.metadata.sites.get(&id)) {
                (true, Some(RandomSite::Forward { draw, scope })) => {
                    let entered = draws
                        .get_mut(draw.0)
                        .ok_or("evaluation plan has an invalid DrawId")?;
                    if entered.is_some() || scope.0 >= self.metadata.scopes.len() {
                        return Err("evaluation plan has a duplicate draw or invalid scope".into());
                    }
                    *entered = Some(node);
                }
                (true, Some(RandomSite::Replay { draw }))
                    if matches!(node.op, RiscOp::Dropout { .. }) =>
                {
                    let forward = draws
                        .get(draw.0)
                        .copied()
                        .flatten()
                        .ok_or("evaluation plan replay precedes its forward draw")?;
                    match (&forward.op, &node.op) {
                        (
                            RiscOp::Dropout {
                                rate: forward_rate,
                                seed: forward_seed,
                            },
                            RiscOp::Dropout { rate, seed },
                        ) if rate.to_bits() == forward_rate.to_bits()
                            && seed == forward_seed
                            && node.output_type == forward.output_type => {}
                        _ => {
                            return Err(
                                "evaluation plan replay changes its forward mask contract".into()
                            );
                        }
                    }
                }
                (false, None) => {}
                _ => {
                    return Err(format!(
                        "evaluation plan has missing or incompatible random metadata at node {}",
                        id.0
                    ));
                }
            }
            seen[id.0] = true;
        }
        if self
            .dag
            .roots()
            .iter()
            .any(|root| !seen.get(root.0).copied().unwrap_or(false))
        {
            return Err("evaluation plan omits a value root".into());
        }
        // A runnable plan owns exactly its selected source slice. Census the
        // graph itself as well as the side tables: deleting a dead site's
        // order entry AND metadata must not erase an entered effect.
        if self.dag.nodes().iter().any(|node| {
            matches!(node.op, RiscOp::Dropout { .. } | RiscOp::UniformLike { .. })
                && !seen[node.id.0]
        }) {
            return Err("evaluation plan omits a source random node".into());
        }
        if self
            .metadata
            .sites
            .to_sorted()
            .iter()
            .any(|(node, _)| !seen.get(node.0).copied().unwrap_or(false))
            || draws.iter().any(Option::is_none)
        {
            return Err("evaluation plan omits a recorded random site".into());
        }
        Ok(())
    }

    pub(crate) fn frame<'a>(
        &'a self,
        context: &'a mut RandomExecutionContext,
    ) -> Result<ExecutionFrame<'a>, String> {
        self.validate()?;
        if self.metadata.scopes[0].seed != context.state.seed {
            return Err(
                "evaluation plan inherited seed does not match its execution context".into(),
            );
        }
        Ok(ExecutionFrame {
            metadata: &self.metadata,
            counters: vec![0; self.metadata.scopes.len()],
            keys: vec![None; self.metadata.draws],
            context,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct DrawKey {
    seed: u64,
    ordinal: u64,
}

/// Opaque evaluator companion to the ordinary host stage plan. Partitioning
/// transports source associations directly; serialized DAGs cannot recreate it.
#[derive(Debug, Clone)]
pub struct StagedEvaluationPlan {
    logical: EvaluationPlan,
    segments: Vec<EvaluationSegment>,
}

#[derive(Debug, Clone)]
struct EvaluationSegment {
    dag: Dag,
    metadata: ExecutionMetadata,
}

impl StagedEvaluationPlan {
    pub(crate) fn new(dag: Dag, metadata: ExecutionMetadata) -> Result<Self, String> {
        Ok(Self {
            logical: EvaluationPlan::new(dag, metadata)?,
            segments: Vec::new(),
        })
    }

    pub(crate) fn append_segment(&mut self, dag: Dag, remap: &UnordMap<NodeId, NodeId>) {
        let mut metadata = self.logical.metadata.clone();
        metadata.sites = metadata
            .sites
            .into_sorted()
            .into_iter()
            .filter_map(|(node, site)| remap.get(&node).map(|mapped| (*mapped, site)))
            .collect();
        metadata.order = dag.nodes().iter().map(|node| node.id).collect();
        self.segments.push(EvaluationSegment { dag, metadata });
    }

    /// Start one invocation. Its keys and scope counters survive every cut.
    pub fn frame<'a>(
        &'a self,
        context: &'a mut RandomExecutionContext,
    ) -> Result<StagedEvaluationFrame<'a>, String> {
        Ok(StagedEvaluationFrame {
            frame: self.logical.frame(context)?,
            segments: self.segments.iter(),
        })
    }
}

/// Invocation-local state for the kernels of a checked staged region.
pub struct StagedEvaluationFrame<'a> {
    frame: ExecutionFrame<'a>,
    segments: std::slice::Iter<'a, EvaluationSegment>,
}

impl StagedEvaluationFrame<'_> {
    /// A host source executes between kernels and may advance the inherited
    /// stream. Forward/replay keys and nested-scope counters remain private.
    pub fn with_context<T>(&mut self, run: impl FnOnce(&mut RandomExecutionContext) -> T) -> T {
        run(self.frame.context)
    }

    /// Execute the next partitioned numeric segment in the same invocation.
    pub fn eval_next_kernel<F>(
        &mut self,
        load_input: F,
    ) -> Result<UnordMap<NodeId, TensorValue>, String>
    where
        F: FnMut(&str) -> Option<TensorValue>,
    {
        let segment = self
            .segments
            .next()
            .ok_or("staged evaluation has no next kernel")?;
        self.frame.metadata = &segment.metadata;
        crate::eval::eval_tensor_segment_with_strict(&segment.dag, &mut self.frame, load_input)
    }
}

pub(crate) struct ExecutionFrame<'a> {
    metadata: &'a ExecutionMetadata,
    counters: Vec<u64>,
    keys: Vec<Option<DrawKey>>,
    context: &'a mut RandomExecutionContext,
}

impl ExecutionFrame<'_> {
    pub(crate) fn order(&self) -> &[NodeId] {
        &self.metadata.order
    }

    fn enter(&mut self, node: NodeId, raw_seed: u64) -> Result<DrawKey, String> {
        match self
            .metadata
            .sites
            .get(&node)
            .ok_or("evaluation plan is missing random metadata")?
        {
            RandomSite::Forward { draw, scope } => {
                let seed = self.metadata.scopes[scope.0]
                    .seed
                    .ok_or("dropout requires a handled Random seed")?;
                if seed != raw_seed {
                    return Err("evaluation plan raw seed disagrees with its handler".into());
                }
                let counter = if scope.0 == 0 {
                    &mut self.context.state.counter
                } else {
                    &mut self.counters[scope.0]
                };
                let key = DrawKey {
                    seed,
                    ordinal: *counter,
                };
                *counter = counter.wrapping_add(1);
                if self.keys[draw.0].replace(key).is_some() {
                    return Err("evaluation plan entered the same forward draw twice".into());
                }
                Ok(key)
            }
            RandomSite::Replay { draw } => self.keys[draw.0]
                .ok_or_else(|| "evaluation plan replay has no realized forward key".into()),
        }
    }

    pub(crate) fn dropout(
        &mut self,
        node: NodeId,
        input: &TensorValue,
        rate: f64,
        seed: u64,
    ) -> Result<TensorValue, String> {
        let prim = input.prim();
        if !prim.is_float() {
            return Err("dropout requires a floating tensor".into());
        }
        let rate =
            cast_raw("dropout", RawScalar::Float(rate), prim).map_err(|trap| trap.to_string())?;
        let prepared =
            PreparedDropout::new(input.storage(), rate).map_err(|error| error.to_string())?;
        let key = self.enter(node, seed)?;
        let output = prepared
            .apply(key.seed, key.ordinal)
            .map_err(|error| error.to_string())?;
        Ok(TensorValue::from_storage(input.shape.clone(), output))
    }

    pub(crate) fn uniform_key(
        &mut self,
        node: NodeId,
        seed: u64,
        active: bool,
    ) -> Result<Option<(u64, u64)>, String> {
        if !active {
            return Ok(None);
        }
        let key = self.enter(node, seed)?;
        // Return the actual key; the evaluator's existing legacy UniformLike
        // numerical path owns its seed folding and value kernel.
        Ok(Some((key.seed, key.ordinal)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimInfo, TensorType};
    use crate::eval::eval_tensor_plan_with_strict;
    use chelis_types::types::Prim;

    fn fixture(rates: &[f64], count: usize, replay: bool) -> EvaluationPlan {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(count)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let mut metadata = ExecutionMetadata::new(Some(42));
        let mut root = x;
        for (index, &rate) in rates.iter().enumerate() {
            root = dag.add_node(
                RiscOp::Dropout { rate, seed: 42 },
                vec![x],
                ty.clone(),
                None,
            );
            if replay && index != 0 {
                metadata
                    .sites
                    .insert(root, RandomSite::Replay { draw: DrawId(0) });
            } else {
                metadata.forward(root, ScopeId(0));
            }
        }
        dag.add_root(root);
        metadata.order = dag.nodes().iter().map(|node| node.id).collect();
        EvaluationPlan::new(dag, metadata).unwrap()
    }

    fn context() -> RandomExecutionContext {
        RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 0,
        })
    }

    #[test]
    fn staged_replay_keeps_its_forward_key_across_a_host_draw_and_scope_cuts() {
        use crate::host::HostParam;
        use crate::host::staged::{HostSource, HostStage, HostValueId, StageValue};
        use crate::host_type_state::{HostPrecisionTerm, HostTypeTerm};
        use std::collections::BTreeMap;
        for nested in [false, true] {
            for rate in [0.5, 1.0] {
                let mut logical = fixture(&[rate, rate], 4, true);
                if nested {
                    logical.metadata.scopes.push(Scope { seed: Some(42) });
                    logical.metadata.sites.insert(
                        NodeId(1),
                        RandomSite::Forward {
                            draw: DrawId(0),
                            scope: ScopeId(1),
                        },
                    );
                }
                let ty = logical.dag.get(NodeId(0)).unwrap().output_type.clone();
                let sources = [HostSource {
                    before: 2,
                    value: StageValue::Host(HostValueId(0)),
                    ty: HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int64)),
                    expression: chelis_deep::parser::parse_str("(lit {type: (t-prim {} int64)} 0)")
                        .unwrap()
                        .remove(0),
                    captures: vec![(
                        "forward".into(),
                        StageValue::Tensor(NodeId(1)),
                        HostTypeTerm::Tensor(ty.clone()),
                    )],
                }];
                let mut companion =
                    StagedEvaluationPlan::new(logical.dag.clone(), logical.metadata).unwrap();
                let plan = crate::host::staged::partition(
                    &logical.dag,
                    &sources,
                    &[HostParam {
                        name: "x".into(),
                        ty: HostTypeTerm::Tensor(ty),
                    }],
                    &BTreeMap::from([(NodeId(0), "x".into())]),
                    &BTreeMap::new(),
                    Some(&mut companion),
                )
                .unwrap();
                let mut context = context();
                // Ordinal 1 and ordinal 0 have different independently pinned masks.
                context.state.counter = 1;
                let mut values = UnordMap::new();
                values.insert("x".into(), TensorValue::from_vec(vec![4], vec![1.0; 4]));
                let result = (|| {
                    let mut frame = companion.frame(&mut context)?;
                    for stage in plan.stages() {
                        match stage {
                            HostStage::Source { captures, .. } => {
                                let expected = if nested {
                                    vec![0.0, 2.0, 0.0, 0.0]
                                } else {
                                    vec![2.0, 0.0, 0.0, 0.0]
                                };
                                assert_eq!(values[&captures[0].value].to_f64_lossy_vec(), expected);
                                frame.with_context(|context| context.state.counter += 1);
                            }
                            HostStage::Kernel { dag, outputs } => {
                                let computed =
                                    frame.eval_next_kernel(|name| values.get(name).cloned())?;
                                for (name, root) in outputs.iter().zip(dag.roots()) {
                                    values.insert(name.clone(), computed[root].clone());
                                }
                            }
                        }
                    }
                    Ok::<_, String>(values[plan.output()].to_f64_lossy_vec())
                })();
                if rate == 1.0 {
                    assert_eq!(
                        result.unwrap_err(),
                        "numeric trap: domain in dropout at f32"
                    );
                    assert_eq!(
                        context.state.counter, 1,
                        "failed validation cannot reach the host cut"
                    );
                } else {
                    assert_eq!(
                        result.unwrap(),
                        if nested {
                            vec![0.0, 2.0, 0.0, 0.0]
                        } else {
                            vec![2.0, 0.0, 0.0, 0.0]
                        }
                    );
                    assert_eq!(context.state.counter, if nested { 2 } else { 3 });
                }
            }
        }
    }

    #[test]
    fn kept_value_finalizes_subtraction_then_division_not_reciprocal_multiplication() {
        // Independently calculated f32 discriminator, pinned as stored bits.
        let plan = fixture(&[f32::from_bits(0x3dcccccd) as f64], 1, false);
        let input = f32::from_bits(0x3f800005);
        let values = eval_tensor_plan_with_strict(&plan, &mut context(), |_| {
            Some(TensorValue::from_vec(vec![1], vec![input as f64]))
        })
        .unwrap();
        let actual = values[&plan.dag_for_inspection().roots()[0]].to_f64_lossy_vec()[0] as f32;
        assert_eq!(actual.to_bits(), 0x3f8e38e9);
        assert_ne!(actual.to_bits(), 0x3f8e38ea, "reciprocal-multiply mutant");
    }

    fn run(
        plan: &EvaluationPlan,
        context: &mut RandomExecutionContext,
        count: usize,
    ) -> Result<Vec<f64>, String> {
        let values = eval_tensor_plan_with_strict(plan, context, |name| {
            (name == "x").then(|| TensorValue::from_vec(vec![count], vec![1.0; count]))
        })?;
        Ok(values[&plan.dag.roots()[0]].to_f64_lossy_vec())
    }

    #[test]
    fn rounded_f32_unit_keeps_at_the_exact_rate_boundary() {
        let rate = f32::from_bits(0x3e1c_aae7);
        // The typed numeric owner tests the independent source word/unit;
        // this consumer check additionally proves draw-entry accounting.
        let mut context = context();
        assert_eq!(
            run(&fixture(&[f64::from(rate)], 1, false), &mut context, 1).unwrap(),
            [f64::from(1.0f32 / (1.0f32 - rate))]
        );
        assert_eq!(context.state().counter, 1);
    }

    #[test]
    fn dropped_negative_zero_is_positive_and_kept_negative_zero_stays_negative() {
        for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
            let mut plan = fixture(&[0.5], 2, false);
            for node in plan.dag.nodes().to_vec() {
                plan.dag.node_mut(node.id).unwrap().output_type.precision = prim;
            }
            plan.validate().unwrap();
            let result = eval_tensor_plan_with_strict(&plan, &mut context(), |_| {
                Some(TensorValue::from_vec(vec![2], vec![-0.0; 2]))
            })
            .unwrap();
            let values = result[&plan.dag.roots()[0]].to_f64_lossy_vec();
            assert_eq!(
                values[0].to_bits(),
                0,
                "{prim:?}: dropped value is positive zero"
            );
            assert_eq!(
                values[1].to_bits(),
                (-0.0f64).to_bits(),
                "{prim:?}: kept division preserves sign"
            );
        }
    }

    #[test]
    fn invalid_rate_preserves_only_the_earlier_entered_prefix() {
        for rate in [-0.5, 1.0, 2.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            for count in [0, 1, 32] {
                let mut context = context();
                let error = run(
                    &fixture(&[0.0, rate, 0.0], count, false),
                    &mut context,
                    count,
                )
                .unwrap_err();
                assert_eq!(error, "numeric trap: domain in dropout at f32");
                assert_eq!(context.state().counter, 1, "rate={rate:?}, count={count}");
            }
        }
    }

    #[test]
    fn accepted_empty_and_zero_rate_calls_each_enter_once() {
        for count in [0, 1, 32] {
            let mut context = context();
            assert_eq!(
                run(&fixture(&[0.0, 0.0], count, false), &mut context, count).unwrap(),
                vec![1.0; count]
            );
            assert_eq!(context.state().counter, 2);
        }
    }

    #[test]
    fn replay_uses_the_forward_key_without_an_ambient_draw() {
        let forward = fixture(&[0.5], 32, false);
        let replay = fixture(&[0.5, 0.5], 32, true);
        let expected = run(&forward, &mut context(), 32).unwrap();
        let mut state = context();
        assert_eq!(run(&replay, &mut state, 32).unwrap(), expected);
        assert_eq!(state.state().counter, 1);
        assert_ne!(run(&replay, &mut state, 32).unwrap(), expected);
        assert_eq!(state.state().counter, 2);
    }

    #[test]
    fn missing_or_changed_replay_metadata_is_not_an_ordinal_zero_fallback() {
        let mut plan = fixture(&[0.5, 0.5], 1, true);
        let replay = plan.dag.roots()[0];
        plan.metadata.sites.remove(&replay);
        assert!(
            run(&plan, &mut context(), 1)
                .unwrap_err()
                .contains("metadata")
        );
        let mut plan = fixture(&[0.5, 0.5], 1, true);
        plan.dag.node_mut(replay).unwrap().op = RiscOp::Dropout {
            rate: 0.25,
            seed: 42,
        };
        assert!(
            run(&plan, &mut context(), 1)
                .unwrap_err()
                .contains("mask contract")
        );
    }

    #[test]
    fn joint_dead_draw_omission_is_detected_against_the_source_graph() {
        let mut plan = fixture(&[0.0, 0.0], 0, false);
        let dead = NodeId(1);
        plan.metadata.order.retain(|node| *node != dead);
        plan.metadata.sites.remove(&dead);
        plan.metadata.sites.insert(
            NodeId(2),
            RandomSite::Forward {
                draw: DrawId(0),
                scope: ScopeId(0),
            },
        );
        plan.metadata.draws = 1;
        assert!(
            run(&plan, &mut context(), 0)
                .unwrap_err()
                .contains("source random node")
        );
    }

    #[test]
    fn host_counter_wrap_is_explicitly_modulo_two_to_the_64() {
        let plan = fixture(&[0.0], 0, false);
        let mut context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: u64::MAX,
        });
        run(&plan, &mut context, 0).unwrap();
        assert_eq!(context.state().counter, 0);
    }

    #[test]
    fn source_gradient_spine_orders_replay_before_the_next_forward_entry() {
        let parse = |source: &str| {
            chelis_deep::parser::parse_str(source)
                .unwrap()
                .pop()
                .unwrap()
        };
        let mut definitions = UnordMap::new();
        definitions.insert("loss".into(), parse("(fn {} (params {} (t {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))})) (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum) (app {} (var {} dropout) (var {} t) (lit {type: (t-prim {} f32)} 0.5)) (lit {type: (t-prim {} int32)} 0)))"));
        let expression = parse(
            "(let {} (bind {} gradient (app {} (grad {} (var {} loss)) (var {type: (t-tensor {} (d-lit {} 32) (t-prim {} f32))} x))) (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        );
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".into(),
            TensorType {
                dims: vec![DimInfo::Lit(32)],
                precision: Prim::F32,
            },
        );
        let mut context = context();
        let plan = crate::lower::try_lower_subexpr_evaluation_plan(
            &expression,
            inputs,
            UnordMap::new(),
            definitions,
            &context,
        )
        .unwrap();
        let schedule = plan
            .metadata
            .order
            .iter()
            .filter_map(|node| plan.metadata.sites.get(node).map(|site| (*node, *site)))
            .collect::<Vec<_>>();
        assert!(
            matches!(
                schedule.as_slice(),
                [
                    (
                        _,
                        RandomSite::Forward {
                            draw: DrawId(0),
                            scope: ScopeId(0)
                        }
                    ),
                    (_, RandomSite::Replay { draw: DrawId(0) }),
                    (
                        _,
                        RandomSite::Forward {
                            draw: DrawId(1),
                            scope: ScopeId(0)
                        }
                    ),
                ]
            ),
            "{schedule:?}"
        );
        // These are three real one-result Dropout nodes. The replay key is
        // metadata, not a fabricated mask tensor owner in the source graph.
        assert_eq!(
            plan.dag
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, RiscOp::Dropout { .. }))
                .count(),
            3
        );
        run(&plan, &mut context, 32).unwrap();
        assert_eq!(context.state().counter, 2);
        eprintln!("actual normalized source random schedule: {schedule:?}");
    }
}
