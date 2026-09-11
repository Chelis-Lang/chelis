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
use crate::execution_spine::{Control, OccurrenceId, SourceKind, Spine, Step};
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
pub struct ScopeId(pub(crate) usize);

impl ScopeId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawId(pub(crate) usize);

impl DrawId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Scope {
    pub(crate) seed: Option<u64>,
}

/// An immutable lowering-owned association. Inspecting a site does not permit
/// constructing an execution plan or attaching it to another graph.
#[derive(Debug, Clone, Copy)]
pub enum RandomSite {
    Forward { draw: DrawId, scope: ScopeId },
    Replay { draw: DrawId },
}

/// Lowering owns these associations. They are not a public authoring API.
#[derive(Debug, Clone)]
pub(crate) struct ExecutionMetadata {
    pub(crate) scopes: Vec<Scope>,
    pub(crate) sites: UnordMap<NodeId, RandomSite>,
    pub(crate) draws: usize,
    pub(crate) spine: Spine,
}

impl ExecutionMetadata {
    pub(crate) fn new(seed: Option<u64>) -> Self {
        Self {
            scopes: vec![Scope { seed }],
            sites: UnordMap::new(),
            draws: 0,
            spine: Spine::default(),
        }
    }

    pub(crate) fn validate_for_dag(&self, dag: &Dag) -> Result<(), String> {
        EvaluationPlan {
            dag: dag.clone(),
            metadata: self.clone(),
        }
        .validate()
    }

    pub(crate) fn emission_view(&self) -> EvaluationEmissionView<'_> {
        EvaluationEmissionView { metadata: self }
    }

    pub(crate) fn forward(&mut self, node: NodeId, scope: ScopeId) {
        let draw = DrawId(self.draws);
        self.draws += 1;
        self.spine.forward(node, draw, scope);
        assert!(
            self.sites
                .insert(node, RandomSite::Forward { draw, scope })
                .is_none()
        );
    }

    pub(crate) fn remap(&mut self, remap: &UnordMap<NodeId, NodeId>) -> Result<(), String> {
        self.spine.remap(remap)?;
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
        Ok(())
    }

    pub(crate) fn merge_child(
        &mut self,
        mut child: Self,
        inherited: ScopeId,
        remap: &UnordMap<NodeId, NodeId>,
        dag: &Dag,
    ) -> Result<(), String> {
        if child.scopes[0].seed != self.scopes[inherited.0].seed {
            return Err("evaluation child scope changed its inherited raw seed".into());
        }
        child.remap(remap)?;
        let scope_offset = self.scopes.len();
        let draw_offset = self.draws;
        child.spine.rebase(
            |draw| Ok(DrawId(draw.0 + draw_offset)),
            |scope| {
                if scope.0 == 0 {
                    inherited
                } else {
                    ScopeId(scope_offset + scope.0 - 1)
                }
            },
        )?;
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
        self.spine.append_child(child.spine, dag)?;
        Ok(())
    }

    pub(crate) fn selected(
        &self,
        remap: &UnordMap<NodeId, NodeId>,
        occurrences: &std::collections::BTreeSet<OccurrenceId>,
    ) -> Result<Self, String> {
        let mut selected = Self::new(self.scopes[0].seed);
        selected.scopes = self.scopes.clone();
        selected.spine = self.spine.selected(remap, occurrences)?;
        let mut draws = UnordMap::new();
        for event in &selected.spine.source {
            if let SourceKind::Forward { draw, .. } = event.kind {
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
        selected.spine.rebase(
            |draw| {
                draws
                    .get(&draw.0)
                    .copied()
                    .ok_or_else(|| "selected source lost its draw".into())
            },
            |scope| scope,
        )?;
        Ok(selected)
    }

    pub(crate) fn complete(&mut self, dag: &Dag) -> Result<(), String> {
        self.spine.complete(dag)?;
        Ok(())
    }
}

/// An immutable, non-serialized graph together with its source execution
/// order and actual forward/replay associations.
#[derive(Debug, Clone)]
pub struct EvaluationPlan {
    dag: Dag,
    metadata: ExecutionMetadata,
}

/// One unfused execution schedule sealed with ownership of that exact graph.
/// Neither graph inspection nor a separately verified graph can construct this
/// carrier. Backend payload rewrites must happen before this boundary.
#[derive(Debug)]
pub struct VerifiedEvaluationPlan {
    ownership: crate::ownership::VerifiedDagProgram,
    metadata: ExecutionMetadata,
}

/// Borrowed execution information from a consumed, ownership-sealed plan.
/// This is not a serialized certificate or a plan-construction API.
#[derive(Clone, Copy)]
pub struct EvaluationEmissionView<'a> {
    metadata: &'a ExecutionMetadata,
}

impl<'a> EvaluationEmissionView<'a> {
    pub fn steps(self) -> &'a [Step] {
        &self.metadata.spine.steps
    }

    pub fn source(self) -> &'a [crate::execution_spine::Occurrence] {
        &self.metadata.spine.source
    }

    pub fn site(self, node: NodeId) -> Option<RandomSite> {
        self.metadata.sites.get(&node).copied()
    }

    pub fn draw_count(self) -> usize {
        self.metadata.draws
    }

    pub fn inherited_seed(self) -> Option<u64> {
        self.metadata.scopes[0].seed
    }
}

impl VerifiedEvaluationPlan {
    /// Inspect the exact ownership-sealed graph without detaching it from its
    /// execution metadata. This is for ABI/observer projection before the
    /// consuming emission boundary below.
    pub fn emission(&self) -> crate::ownership::VerifiedDagView<'_> {
        self.ownership.emission()
    }
    /// Consume the sealed graph and borrow its own execution information for
    /// one emission. The view cannot outlive this callback; no constructor
    /// permits pairing a view with a different graph for this entry point.
    pub fn with_emission<R>(
        self,
        emit: impl FnOnce(crate::ownership::VerifiedDagProgram, EvaluationEmissionView<'_>) -> R,
    ) -> R {
        let Self {
            ownership,
            metadata,
        } = self;
        emit(
            ownership,
            EvaluationEmissionView {
                metadata: &metadata,
            },
        )
    }
}

impl EvaluationPlan {
    pub(crate) fn into_parts(self) -> (Dag, ExecutionMetadata) {
        (self.dag, self.metadata)
    }
    /// Seal ownership in the order the source plan actually executes.
    ///
    /// Current lowering preserves graph-node order while inserting controls.
    /// Reject a different schedule rather than applying a DAG-order lifetime
    /// plan to it, or silently reordering effects to fit that lifetime plan.
    pub fn verify_ownership(self) -> Result<VerifiedEvaluationPlan, String> {
        self.validate()?;
        let scheduled = self
            .metadata
            .spine
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::Node(node) => Some(*node),
                Step::Control { .. } => None,
            });
        if !scheduled.eq(self.dag.nodes().iter().map(|node| node.id)) {
            return Err("native execution order differs from ownership graph order".into());
        }
        let ownership = crate::ownership::lower_dag_ownership(self.dag)
            .and_then(crate::ownership::verify_ownership)
            .map_err(|error| error.to_string())?;
        Ok(VerifiedEvaluationPlan {
            ownership,
            metadata: self.metadata,
        })
    }

    /// Inspect the graph's types and value roots. Cloning/serializing this
    /// view does not preserve the plan and cannot recreate one.
    pub fn dag_for_inspection(&self) -> &Dag {
        &self.dag
    }

    /// Borrow the actual executable sequence, including value-free controls.
    /// These views do not authorize construction of a runnable plan.
    pub fn steps_for_inspection(&self) -> &[Step] {
        &self.metadata.spine.steps
    }

    /// Independently recorded source occurrences, mapped through the owning
    /// passes. Backward replay is not a second source occurrence.
    pub fn source_for_inspection(&self) -> &[crate::execution_spine::Occurrence] {
        &self.metadata.spine.source
    }

    #[cfg(feature = "lowering-trace")]
    pub(crate) fn snapshot(dag: &Dag, metadata: &ExecutionMetadata) -> Self {
        let mut metadata = metadata.clone();
        metadata
            .complete(dag)
            .expect("observed source region preserves node order");
        Self::new(dag.clone(), metadata)
            .expect("observed source region retains its execution census")
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
        let mut source = self.metadata.spine.source.iter();
        let mut scopes = vec![ScopeId(0)];
        let mut entered_scopes = std::collections::BTreeSet::new();
        let mut occurrences = std::collections::BTreeSet::new();
        for step in &self.metadata.spine.steps {
            let id = match *step {
                Step::Node(id) => id,
                Step::Control {
                    occurrence,
                    control,
                } => {
                    let expected = source
                        .next()
                        .ok_or("execution has an extra source control")?;
                    if expected.id != occurrence
                        || expected.kind != SourceKind::Control(control)
                        || !occurrences.insert(occurrence)
                    {
                        return Err(
                            "execution control disagrees with the independent source census".into(),
                        );
                    }
                    match control {
                        Control::Enter { scope, seed } => {
                            if scope.0 == 0
                                || !entered_scopes.insert(scope.0)
                                || self
                                    .metadata
                                    .scopes
                                    .get(scope.0)
                                    .and_then(|scope| scope.seed)
                                    != Some(seed)
                            {
                                return Err(
                                    "execution enters an invalid or repeated seed scope".into()
                                );
                            }
                            scopes.push(scope);
                        }
                        Control::Leave { scope } => {
                            if scopes.len() <= 1 || scopes.pop() != Some(scope) {
                                return Err("execution leaves an unmatched seed scope".into());
                            }
                        }
                    }
                    continue;
                }
            };
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
                    let expected = source.next().ok_or("execution has an extra source draw")?;
                    if expected.kind
                        != (SourceKind::Forward {
                            node: id,
                            draw: *draw,
                            scope: *scope,
                        })
                        || !occurrences.insert(expected.id)
                        || scopes.last() != Some(scope)
                    {
                        return Err(
                            "execution draw disagrees with the independent source census or scope"
                                .into(),
                        );
                    }
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
        if source.next().is_some() || scopes != [ScopeId(0)] {
            return Err("execution omits a source occurrence or pending seed exit".into());
        }
        if seen.iter().any(|seen| !seen) {
            return Err("execution spine omits a selected graph node".into());
        }
        if self
            .dag
            .roots()
            .iter()
            .any(|root| !seen.get(root.0).copied().unwrap_or(false))
        {
            return Err("evaluation plan omits a value root".into());
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
            scopes: vec![ScopeId(0)],
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
    source_position: usize,
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
            source_position: 0,
        })
    }

    pub(crate) fn source_position(&self) -> usize {
        self.source_position
    }

    pub(crate) fn source_len(&self) -> usize {
        self.logical.metadata.spine.source.len()
    }

    pub(crate) fn append_segment(
        &mut self,
        dag: Dag,
        remap: &UnordMap<NodeId, NodeId>,
        source_end: usize,
    ) -> Result<(), String> {
        let occurrences = self
            .logical
            .metadata
            .spine
            .source
            .get(self.source_position..source_end)
            .ok_or("staged execution has an invalid source occurrence cut")?
            .iter()
            .map(|event| event.id)
            .collect();
        let mut metadata = self.logical.metadata.clone();
        metadata.spine = metadata.spine.selected(remap, &occurrences)?;
        metadata.sites = metadata
            .sites
            .into_sorted()
            .into_iter()
            .filter_map(|(node, site)| remap.get(&node).map(|mapped| (*mapped, site)))
            .collect();
        metadata.complete(&dag)?;
        self.segments.push(EvaluationSegment { dag, metadata });
        self.source_position = source_end;
        Ok(())
    }

    /// Start one invocation. Its keys and scope counters survive every cut.
    pub fn frame<'a>(
        &'a self,
        context: &'a mut RandomExecutionContext,
    ) -> Result<StagedEvaluationFrame<'a>, String> {
        if self.source_position != self.source_len() {
            return Err("staged execution omits source occurrences".into());
        }
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
    scopes: Vec<ScopeId>,
}

impl ExecutionFrame<'_> {
    pub(crate) fn steps(&self) -> &[Step] {
        &self.metadata.spine.steps
    }

    pub(crate) fn control(&mut self, control: Control) -> Result<(), String> {
        match control {
            Control::Enter { scope, .. } => {
                self.counters[scope.0] = 0;
                self.scopes.push(scope);
            }
            Control::Leave { scope } => {
                if self.scopes.len() <= 1 || self.scopes.pop() != Some(scope) {
                    return Err("execution left an unmatched seed scope".into());
                }
            }
        }
        Ok(())
    }

    fn enter(&mut self, node: NodeId, raw_seed: u64) -> Result<DrawKey, String> {
        match self
            .metadata
            .sites
            .get(&node)
            .ok_or("evaluation plan is missing random metadata")?
        {
            RandomSite::Forward { draw, scope } => {
                if self.scopes.last() != Some(scope) {
                    return Err("forward draw is outside its entered source scope".into());
                }
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
        metadata.spine.record_nodes(&dag);
        metadata.complete(&dag).unwrap();
        EvaluationPlan::new(dag, metadata).unwrap()
    }

    fn context() -> RandomExecutionContext {
        RandomExecutionContext::new(RandomLoweringState {
            seed: Some(42),
            counter: 0,
        })
    }

    #[test]
    fn ownership_seal_retains_actual_forward_replay_and_payload() {
        let plan = fixture(&[0.5, 0.5], 4, true);
        let expected_steps = plan.steps_for_inspection().to_vec();
        let expected_source = plan.source_for_inspection().to_vec();
        plan.verify_ownership()
            .unwrap()
            .with_emission(|owned, execution| {
                assert_eq!(owned.emission().roots(), &[NodeId(2)]);
                assert_eq!(owned.emission().len(), 3);
                assert_eq!(execution.steps(), expected_steps);
                assert_eq!(execution.source(), expected_source);
                assert_eq!(execution.draw_count(), 1);
                assert_eq!(execution.inherited_seed(), Some(42));
                assert!(matches!(
                    execution.site(NodeId(1)),
                    Some(RandomSite::Forward {
                        draw: DrawId(0),
                        scope: ScopeId(0)
                    })
                ));
                assert!(matches!(
                    execution.site(NodeId(2)),
                    Some(RandomSite::Replay { draw: DrawId(0) })
                ));
            });
    }

    #[test]
    fn ownership_seal_rejects_a_different_valid_topological_schedule() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        for name in ["x", "y"] {
            let node = dag.add_node(RiscOp::Load { name: name.into() }, vec![], ty.clone(), None);
            dag.add_root(node);
        }
        let mut metadata = ExecutionMetadata::new(None);
        metadata.spine.record_nodes(&dag);
        metadata.complete(&dag).unwrap();
        metadata.spine.steps.swap(0, 1);
        // Evaluator ordering is valid, but a DAG-order storage plan cannot be
        // applied to a different schedule merely because both are topological.
        let plan = EvaluationPlan::new(dag, metadata).unwrap();
        assert_eq!(
            plan.verify_ownership().unwrap_err(),
            "native execution order differs from ownership graph order"
        );
    }

    #[test]
    fn ownership_seal_checks_ownership_not_only_execution_metadata() {
        let mut plan = fixture(&[0.5], 4, false);
        let source = plan.dag.roots()[0];
        let ty = plan.dag.get(source).unwrap().output_type.clone();
        for _ in 0..2 {
            plan.dag
                .add_node(RiscOp::Drop, vec![source], ty.clone(), None);
        }
        plan.metadata.spine.record_nodes(&plan.dag);
        plan.metadata.complete(&plan.dag).unwrap();
        plan.validate().unwrap();
        assert!(
            plan.verify_ownership().is_err(),
            "duplicate logical termination must reject"
        );
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
                    let enter = Control::Enter {
                        scope: ScopeId(1),
                        seed: 42,
                    };
                    let leave = Control::Leave { scope: ScopeId(1) };
                    logical.metadata.spine.source[0].kind = SourceKind::Forward {
                        node: NodeId(1),
                        draw: DrawId(0),
                        scope: ScopeId(1),
                    };
                    logical.metadata.spine.source.insert(
                        0,
                        crate::execution_spine::Occurrence {
                            id: OccurrenceId(1),
                            kind: SourceKind::Control(enter),
                        },
                    );
                    logical
                        .metadata
                        .spine
                        .source
                        .push(crate::execution_spine::Occurrence {
                            id: OccurrenceId(2),
                            kind: SourceKind::Control(leave),
                        });
                    logical.metadata.spine.steps.insert(
                        1,
                        Step::Control {
                            occurrence: OccurrenceId(1),
                            control: enter,
                        },
                    );
                    logical.metadata.spine.steps.insert(
                        3,
                        Step::Control {
                            occurrence: OccurrenceId(2),
                            control: leave,
                        },
                    );
                }
                let ty = logical.dag.get(NodeId(0)).unwrap().output_type.clone();
                let sources = [HostSource {
                    before: 2,
                    occurrences_before: Some(if nested { 3 } else { 1 }),
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
        plan.metadata.spine.retain_nodes(|node| node != dead);
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
                .contains("independent source census")
        );
    }

    fn lower_source(source: &str) -> EvaluationPlan {
        let expr = chelis_deep::parser::parse_str(source)
            .unwrap()
            .pop()
            .unwrap();
        let inputs = [(
            "x".into(),
            TensorType {
                dims: vec![DimInfo::Lit(32)],
                precision: Prim::F32,
            },
        )]
        .into_iter()
        .collect();
        crate::lower::try_lower_subexpr_evaluation_plan(
            &expr,
            inputs,
            UnordMap::new(),
            UnordMap::new(),
            &context(),
        )
        .unwrap()
    }

    #[test]
    fn value_free_equal_seed_scopes_survive_and_do_not_enter_draws() {
        let plan = lower_source(
            "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (var {} x)))",
        );
        assert_eq!(plan.source_for_inspection().len(), 4);
        assert!(matches!(
            plan.source_for_inspection(),
            [
                crate::execution_spine::Occurrence {
                    kind: SourceKind::Control(Control::Enter {
                        scope: ScopeId(1),
                        ..
                    }),
                    ..
                },
                crate::execution_spine::Occurrence {
                    kind: SourceKind::Control(Control::Enter {
                        scope: ScopeId(2),
                        ..
                    }),
                    ..
                },
                crate::execution_spine::Occurrence {
                    kind: SourceKind::Control(Control::Leave { scope: ScopeId(2) }),
                    ..
                },
                crate::execution_spine::Occurrence {
                    kind: SourceKind::Control(Control::Leave { scope: ScopeId(1) }),
                    ..
                },
            ]
        ));
        let mut state = context();
        state.state.counter = 17;
        assert_eq!(run(&plan, &mut state, 32).unwrap(), vec![1.0; 32]);
        assert_eq!(state.state().counter, 17);
    }

    #[test]
    fn joint_control_and_runtime_scope_deletion_does_not_delete_source_evidence() {
        let mut plan = lower_source(
            "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (var {} x))",
        );
        plan.metadata
            .spine
            .steps
            .retain(|step| matches!(step, Step::Node(_)));
        plan.metadata.scopes.truncate(1);
        assert!(plan.validate().unwrap_err().contains("source occurrence"));
    }

    #[test]
    fn staged_control_only_segment_requires_the_actual_source_cut() {
        use crate::host::staged::{HostSource, HostStage, HostValueId, StageValue};
        use crate::host_type_state::{HostPrecisionTerm, HostTypeTerm};
        use std::collections::BTreeMap;
        let logical = lower_source(
            "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (var {} x))",
        );
        for cut in [Some(2), None, Some(3)] {
            let mut companion =
                StagedEvaluationPlan::new(logical.dag.clone(), logical.metadata.clone()).unwrap();
            let sources = [HostSource {
                before: 0,
                occurrences_before: cut,
                value: StageValue::Host(HostValueId(0)),
                ty: HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(Prim::Int64)),
                expression: chelis_deep::parser::parse_str("(lit {type: (t-prim {} int64)} 0)")
                    .unwrap()
                    .remove(0),
                captures: Vec::new(),
            }];
            let partition = crate::host::staged::partition(
                &logical.dag,
                &sources,
                &[],
                &BTreeMap::from([(NodeId(0), "x".into())]),
                &BTreeMap::new(),
                Some(&mut companion),
            );
            if cut != Some(2) {
                assert!(partition.unwrap_err().contains("source"));
                continue;
            }
            let partition = partition.unwrap();
            assert!(matches!(
                partition.stages(),
                [
                    HostStage::Kernel { .. },
                    HostStage::Source { .. },
                    HostStage::Kernel { .. }
                ]
            ));
            assert_eq!(companion.segments[0].metadata.spine.source.len(), 2);
            assert!(companion.segments[1].metadata.spine.source.is_empty());
            let mut context = context();
            context.state.counter = 17;
            let mut frame = companion.frame(&mut context).unwrap();
            frame.eval_next_kernel(|_| None).unwrap();
            assert_eq!(frame.frame.scopes, [ScopeId(0)]);
            frame.with_context(|context| {
                assert_eq!(context.state.counter, 17);
                context.state.counter += 1;
            });
            frame
                .eval_next_kernel(|name| {
                    (name == "x").then(|| TensorValue::from_vec(vec![32], vec![1.0; 32]))
                })
                .unwrap();
            frame.with_context(|context| assert_eq!(context.state.counter, 18));
        }
    }

    #[test]
    fn control_corruption_is_rejected_against_the_source_and_stack() {
        let source =
            "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (var {} x))";
        let original = lower_source(source);
        let controls = original
            .metadata
            .spine
            .steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| matches!(step, Step::Control { .. }).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(controls.len(), 2);
        for corruption in 0..5 {
            let mut plan = original.clone();
            let steps = &mut plan.metadata.spine.steps;
            match corruption {
                0 => {
                    steps.remove(controls[0]);
                }
                1 => {
                    steps.insert(controls[0], steps[controls[0]]);
                }
                2 => {
                    steps.swap(controls[0], controls[1]);
                }
                3 => {
                    let Step::Control {
                        control: Control::Enter { seed, .. },
                        ..
                    } = &mut steps[controls[0]]
                    else {
                        unreachable!()
                    };
                    *seed = 43;
                }
                _ => {
                    // Even jointly deleting a leave from execution and census
                    // cannot convert a pending scope into a finished plan.
                    steps.remove(controls[1]);
                    plan.metadata.spine.source.pop();
                }
            }
            let error = plan.validate().unwrap_err();
            assert!(
                error.contains("source") || error.contains("scope"),
                "{corruption}: {error}"
            );
        }
    }

    #[test]
    fn same_seed_balanced_scope_identity_substitution_is_rejected() {
        let mut plan = lower_source(
            "(handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (handle-effect {effect: random} (lit {type: (t-prim {} int64)} 42) (var {} x)))",
        );
        for step in &mut plan.metadata.spine.steps {
            if let Step::Control { control, .. } = step {
                let scope = match control {
                    Control::Enter { scope, .. } | Control::Leave { scope } => scope,
                };
                scope.0 = 3 - scope.0;
            }
        }
        assert!(
            plan.validate()
                .unwrap_err()
                .contains("independent source census")
        );
    }

    #[test]
    fn deleting_the_graph_and_runtime_site_still_cannot_erase_a_source_draw() {
        let plan = fixture(&[0.0, 0.0], 0, false);
        let roots = plan.dag.roots().to_vec();
        let (_, remap) = crate::optimize::project_execution_slice(&plan.dag, &roots, &roots);
        assert!(!remap.contains_key(&NodeId(1)));
        let occurrences = plan
            .metadata
            .spine
            .source
            .iter()
            .map(|event| event.id)
            .collect();
        assert!(
            plan.metadata
                .selected(&remap, &occurrences)
                .unwrap_err()
                .contains("lost source node")
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
            .spine
            .nodes()
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
