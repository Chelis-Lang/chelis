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
#[cfg(any(test, feature = "lowering-trace"))]
use crate::execution_spine::OccurrenceId;
use crate::execution_spine::{Control, FullOccurrenceId, SourceKind, Spine, Step};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResourcePolicy {
    Legacy,
    RecordRequirements,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScopeId(pub(crate) usize);

impl ScopeId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        child: Self,
        inherited: ScopeId,
        remap: &UnordMap<NodeId, NodeId>,
        dag: &Dag,
    ) -> Result<(), String> {
        self.merge_child_inner(child, inherited, remap, dag)
    }

    #[cfg(feature = "lowering-trace")]
    pub(crate) fn merge_child_observed(
        &mut self,
        child: Self,
        inherited: ScopeId,
        remap: &UnordMap<NodeId, NodeId>,
        dag: &Dag,
    ) -> Result<crate::lowering_trace::EffectRemap, String> {
        let occurrence_offset = self.spine.occurrence_count();
        let draw_offset = self.draws;
        let scope_offset = self.scopes.len();
        let occurrences = (0..child.spine.occurrence_count())
            .map(|index| (OccurrenceId(index), OccurrenceId(index + occurrence_offset)))
            .collect();
        let draws = (0..child.draws)
            .map(|index| (DrawId(index), DrawId(index + draw_offset)))
            .collect();
        let scopes = (0..child.scopes.len())
            .map(|index| {
                let source = ScopeId(index);
                let target = if index == 0 {
                    inherited
                } else {
                    ScopeId(scope_offset + index - 1)
                };
                (source, target)
            })
            .collect();
        self.merge_child_inner(child, inherited, remap, dag)?;
        Ok(crate::lowering_trace::EffectRemap {
            occurrences,
            draws,
            scopes,
        })
    }

    fn merge_child_inner(
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
        occurrences: &std::collections::BTreeSet<FullOccurrenceId>,
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
    /// Complete helper-local source census, including retained requirements.
    #[cfg(feature = "lowering-trace")]
    pub fn full_source(self) -> crate::lowering_trace::FullSpineObservation {
        self.metadata.spine.full_observation()
    }

    /// The scope allocated by this execution-metadata owner for caller state.
    #[cfg(feature = "lowering-trace")]
    pub fn inherited_scope(self) -> ScopeId {
        ScopeId(0)
    }

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

    /// Opt-in snapshot of the complete source order, including value-free
    /// Resource requirements. The established random-only views remain
    /// unchanged for existing consumers.
    #[cfg(feature = "lowering-trace")]
    pub fn full_spine_for_inspection(&self) -> crate::lowering_trace::FullSpineObservation {
        self.metadata.spine.full_observation()
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
        self.metadata.spine.validate_full(|node| {
            matches!(
                self.metadata.sites.get(&node),
                Some(RandomSite::Forward { .. })
            )
        })?;
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
        if source.next().is_some() {
            return Err("execution omits a source occurrence or pending seed exit".into());
        }
        if scopes != [ScopeId(0)] {
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

    pub(crate) fn validate_for_context(
        &self,
        context: &RandomExecutionContext,
    ) -> Result<(), String> {
        self.validate()?;
        if self.metadata.scopes[0].seed != context.state.seed {
            return Err(
                "evaluation plan inherited seed does not match its execution context".into(),
            );
        }
        Ok(())
    }

    pub(crate) fn frame<'a>(
        &'a self,
        context: &'a mut RandomExecutionContext,
    ) -> Result<ExecutionFrame<'a>, String> {
        self.validate_for_context(context)?;
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
        metadata.spine = metadata.spine.selected_legacy(remap, &occurrences)?;
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
mod input_preparation_tests {
    //! Test-first contract for additive load preparation.
    use super::*;
    use crate::dag::{DimInfo, TensorType};
    use crate::eval::{
        eval_tensor_plan_with_strict, eval_tensor_roots_with_strict, prepare_tensor_plan_inputs,
        prepare_tensor_roots_inputs,
    };
    use chelis_types::dtype_semantics::{RawTensor, finalize_tensor};
    use chelis_types::types::Prim;

    fn value() -> TensorValue {
        TensorValue::from_storage(
            vec![2],
            finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0, 5.0])).unwrap(),
        )
    }

    fn ty() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        }
    }

    fn loads(names: &[&str]) -> Dag {
        let mut dag = Dag::new();
        for name in names {
            dag.add_node(
                RiscOp::Load {
                    name: (*name).into(),
                },
                vec![],
                ty(),
                None,
            );
        }
        dag
    }

    fn context(seed: u64) -> RandomExecutionContext {
        RandomExecutionContext::new(RandomLoweringState {
            seed: Some(seed),
            counter: 5,
        })
    }

    fn plan(mut dag: Dag) -> EvaluationPlan {
        let mut metadata = ExecutionMetadata::new(Some(42));
        let x = NodeId(0);
        let out = dag.add_node(
            RiscOp::Dropout {
                rate: 0.5,
                seed: 42,
            },
            vec![x],
            ty(),
            None,
        );
        metadata.forward(out, ScopeId(0));
        dag.add_root(out);
        metadata.spine.record_nodes(&dag);
        metadata.complete(&dag).unwrap();
        EvaluationPlan::new(dag, metadata).unwrap()
    }

    #[test]
    fn input_preparation_roots_preserve_values_order_dedup_and_empty_selection() {
        let mut dag = loads(&["x", "unused", "x"]);
        let root = dag.add_node(RiscOp::Add, vec![NodeId(0), NodeId(2)], ty(), None);
        for roots in [vec![root], vec![]] {
            let mut old_calls = Vec::new();
            let old = eval_tensor_roots_with_strict(&dag, &roots, |name| {
                old_calls.push(name.to_owned());
                Some(value())
            })
            .unwrap();
            let mut calls = Vec::new();
            let prepared = prepare_tensor_roots_inputs(&dag, &roots, |name| {
                calls.push(name.to_owned());
                Ok(Some(value()))
            })
            .unwrap();
            assert_eq!(calls, old_calls);
            assert_eq!(
                calls,
                if roots.is_empty() {
                    vec!["x", "unused"]
                } else {
                    vec!["x"]
                }
            );
            let actual =
                eval_tensor_roots_with_strict(&dag, &roots, |name| prepared.get(name).cloned())
                    .unwrap();
            assert_eq!(actual[&root], old[&root]);
            assert_eq!(actual[&root].storage(), old[&root].storage());
            assert_eq!(
                calls, old_calls,
                "executing the map cannot re-enter the provider"
            );
        }
    }

    #[test]
    fn input_preparation_shape_dependencies_remain_required() {
        let mut dag = loads(&["shape", "unused"]);
        let root = dag.add_node(
            RiscOp::synth_const(Prim::F32, 7.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_shape_dep(root, NodeId(0));
        let mut calls = Vec::new();
        let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
            calls.push(name.to_owned());
            Ok(Some(value()))
        })
        .unwrap();
        assert_eq!(calls, ["shape"]);
        assert!(prepared.contains_key("shape"));
        let old = eval_tensor_roots_with_strict(&dag, &[root], |_| None).unwrap_err();
        assert_eq!(
            prepare_tensor_roots_inputs(&dag, &[root], |_| Ok(None)).unwrap_err(),
            old
        );
        assert_eq!(old, "missing required input `shape`");
    }

    #[test]
    fn input_preparation_optional_absence_is_not_a_strict_load_error() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        for _ in 0..2 {
            dag.add_node(
                RiscOp::Load {
                    name: "optional".into(),
                },
                vec![],
                symbolic.clone(),
                None,
            );
        }
        let live = dag.add_node(
            RiscOp::Load {
                name: "live".into(),
            },
            vec![],
            symbolic,
            None,
        );
        for supplied in [false, true] {
            let mut old_calls = Vec::new();
            let old = eval_tensor_roots_with_strict(&dag, &[live], |name| {
                old_calls.push(name.to_owned());
                (name == "live" || supplied).then(value)
            });
            let mut calls = Vec::new();
            let prepared = prepare_tensor_roots_inputs(&dag, &[live], |name| {
                calls.push(name.to_owned());
                Ok((name == "live" || supplied).then(value))
            })
            .unwrap();
            assert_eq!(calls, old_calls);
            assert_eq!(
                calls,
                if supplied {
                    vec!["optional", "live"]
                } else {
                    vec!["optional", "optional", "live"]
                }
            );
            let actual =
                eval_tensor_roots_with_strict(&dag, &[live], |name| prepared.get(name).cloned());
            assert_eq!(actual, old);
        }
    }

    #[test]
    fn input_preparation_defers_missing_optional_dimension_to_inference() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        dag.add_node(
            RiscOp::Load {
                name: "shape".into(),
            },
            vec![],
            symbolic.clone(),
            None,
        );
        let one = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            TensorType::scalar_f32(),
            None,
        );
        let root = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::RtDim::Sym("n".into()),
            },
            vec![one],
            symbolic,
            None,
        );
        let mut calls = Vec::new();
        let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
            calls.push(name.to_owned());
            Ok(None)
        })
        .unwrap();
        assert_eq!(calls, ["shape"]);
        assert!(prepared.is_empty());
        let old = eval_tensor_roots_with_strict(&dag, &[root], |_| None).unwrap_err();
        let actual =
            eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned())
                .unwrap_err();
        assert_eq!(actual, old);
        assert!(actual.contains("symbolic dimension"), "{actual}");
    }

    #[test]
    fn input_preparation_preserves_to_end_binding_without_symbolic_dimensions() {
        let mut dag = loads(&["x"]);
        let root = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(crate::dag::RtDim::Lit(0), crate::dag::RtDim::ToEnd)],
            },
            vec![NodeId(0)],
            ty(),
            None,
        );
        let old = eval_tensor_roots_with_strict(&dag, &[root], |_| Some(value())).unwrap();
        let prepared = prepare_tensor_roots_inputs(&dag, &[root], |_| Ok(Some(value()))).unwrap();
        let actual =
            eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned())
                .unwrap();
        assert_eq!(actual, old);
        assert_eq!(actual[&root], value());
    }

    // #1956 demand-role stubs will reuse these graphs, not another selector.
    fn shape_only_fixture(names: &[&str]) -> (Dag, NodeId) {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        for name in names {
            dag.add_node(
                RiscOp::Load {
                    name: (*name).into(),
                },
                vec![],
                symbolic.clone(),
                None,
            );
        }
        let one = dag.add_node(
            RiscOp::synth_const_tensor(Prim::F32, vec![1.0, 1.0]),
            vec![],
            ty(),
            None,
        );
        let root = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![crate::dag::RtDim::Sym("n".into())],
            },
            vec![one],
            symbolic,
            None,
        );
        (dag, root)
    }

    #[test]
    fn input_preparation_shape_witness_is_required_but_its_missing_error_is_deferred() {
        let (dag, root) = shape_only_fixture(&["shape"]);
        for supplied in [false, true] {
            let mut calls = Vec::new();
            let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
                calls.push(name.to_owned());
                Ok(supplied.then(value))
            })
            .unwrap();
            assert_eq!(calls, ["shape"]);
            let actual =
                eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned());
            let mut old_calls = Vec::new();
            let old = eval_tensor_roots_with_strict(&dag, &[root], |name| {
                old_calls.push(name.to_owned());
                supplied.then(value)
            });
            assert_eq!(calls, old_calls);
            assert_eq!(actual, old);
            if supplied {
                assert_eq!(actual.unwrap()[&root].shape, [2]);
            } else {
                assert_eq!(
                    actual.unwrap_err(),
                    "missing required input `shape` for symbolic dimension `n`"
                );
            }
        }
    }

    #[test]
    fn input_preparation_ambiguous_dead_sources_preserve_provider_and_error_order() {
        let (dag, root) = shape_only_fixture(&["a", "b"]);
        for supplied in [false, true] {
            let mut calls = Vec::new();
            let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
                calls.push(name.to_owned());
                Ok(supplied.then(value))
            })
            .unwrap();
            assert_eq!(calls, ["a", "b"]);
            let old =
                eval_tensor_roots_with_strict(&dag, &[root], |_| supplied.then(value)).unwrap_err();
            assert_eq!(
                eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned())
                    .unwrap_err(),
                old
            );
            assert_eq!(
                old,
                "ambiguous dead-load sources [\"a\", \"b\"] for live symbolic dimension `n`"
            );
        }
        let mut calls = Vec::new();
        let error = prepare_tensor_roots_inputs(&dag, &[root], |name| {
            calls.push(name.to_owned());
            Err("provider failed before inference".into())
        })
        .unwrap_err();
        assert_eq!(calls, ["a"]);
        assert_eq!(error, "provider failed before inference");
    }

    fn computed_shape_fixture() -> (Dag, NodeId) {
        let (mut dag, _) = shape_only_fixture(&["shape"]);
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty(), None);
        let root = dag.add_node(
            RiscOp::Stride {
                strides: vec![crate::dag::RtDim::Lit(2)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            },
            None,
        );
        (dag, root)
    }

    #[test]
    fn input_preparation_computed_extent_does_not_need_surplus_declarer() {
        let (dag, root) = computed_shape_fixture();
        assert!(crate::dag::op_declared_dim_names(&dag).contains("n"));
        let mut calls = Vec::new();
        let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
            calls.push(name.to_owned());
            Ok((name == "x").then(value))
        })
        .unwrap();
        assert_eq!(calls, ["shape", "x"]);
        assert!(!prepared.contains_key("shape"));
        let actual =
            eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned())
                .unwrap();
        assert_eq!(actual[&root].shape, [1]);
        assert_eq!(
            actual[&root].storage().scalar_at(0),
            value().storage().scalar_at(0)
        );
        assert_eq!(
            actual,
            eval_tensor_roots_with_strict(&dag, &[root], |name| (name == "x").then(value)).unwrap()
        );
    }

    fn internal_shape_fixture() -> (Dag, NodeId) {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load {
                name: "shape".into(),
            },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), Some(2))],
                precision: Prim::F32,
            },
            None,
        );
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty(), None);
        let root = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![crate::dag::RtDim::Sym("n".into())],
            },
            vec![x],
            ty(),
            None,
        );
        (dag, root)
    }

    #[test]
    fn input_preparation_internal_symbol_requires_exact_fallback_source() {
        let (dag, root) = internal_shape_fixture();
        for supplied in [false, true] {
            let mut calls = Vec::new();
            let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
                calls.push(name.to_owned());
                Ok((supplied || name == "x").then(value))
            })
            .unwrap();
            assert_eq!(calls, ["shape", "x"]);
            let actual =
                eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned());
            assert_eq!(
                actual,
                eval_tensor_roots_with_strict(&dag, &[root], |name| (supplied || name == "x")
                    .then(value))
            );
            if supplied {
                assert_eq!(actual.unwrap()[&root], value());
            } else {
                assert_eq!(
                    actual.unwrap_err(),
                    "missing symbolic dimension binding `n`"
                );
            }
        }
    }

    #[test]
    fn input_preparation_optional_none_may_precede_same_named_required_load() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        };
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            symbolic.clone(),
            None,
        );
        let root = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], symbolic, None);
        let mut calls = Vec::new();
        let prepared = prepare_tensor_roots_inputs(&dag, &[root], |name| {
            calls.push(name.to_owned());
            Ok((calls.len() == 2).then(value))
        })
        .unwrap();
        assert_eq!(calls, ["x", "x"]);
        let actual =
            eval_tensor_roots_with_strict(&dag, &[root], |name| prepared.get(name).cloned())
                .unwrap();
        assert_eq!(actual[&root], value());
    }

    #[test]
    fn input_preparation_provider_errors_and_missing_live_inputs_stop_in_order() {
        let dag = loads(&["first", "broken", "later"]);
        let mut calls = Vec::new();
        let error = prepare_tensor_roots_inputs(&dag, &[], |name| {
            calls.push(name.to_owned());
            if name == "broken" {
                Err("initializer failed exactly".into())
            } else {
                Ok(Some(value()))
            }
        })
        .unwrap_err();
        assert_eq!(error, "initializer failed exactly");
        assert_eq!(calls, ["first", "broken"]);
        let mut old_calls = Vec::new();
        let old = eval_tensor_roots_with_strict(&dag, &[], |name| {
            old_calls.push(name.to_owned());
            (name == "first").then(value)
        })
        .unwrap_err();
        calls.clear();
        let missing = prepare_tensor_roots_inputs(&dag, &[], |name| {
            calls.push(name.to_owned());
            Ok((name == "first").then(value))
        })
        .unwrap_err();
        assert_eq!(missing, old);
        assert_eq!(calls, old_calls);
        let plan = plan(dag);
        calls.clear();
        let error = prepare_tensor_plan_inputs(&plan, &context(42), |name| {
            calls.push(name.to_owned());
            if name == "broken" {
                Err("initializer failed exactly".into())
            } else {
                Ok(Some(value()))
            }
        })
        .unwrap_err();
        assert_eq!(error, "initializer failed exactly");
        assert_eq!(calls, ["first", "broken"]);
    }

    #[test]
    fn input_preparation_leaves_rank_and_numeric_errors_to_evaluation() {
        let dag = loads(&["x"]);
        for input in [
            TensorValue::from_storage(
                vec![],
                finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0])).unwrap(),
            ),
            TensorValue::from_storage(
                vec![1],
                finalize_tensor("test", Prim::F32, RawTensor::Float(vec![3.0])).unwrap(),
            ),
            TensorValue::from_storage(
                vec![2],
                finalize_tensor("test", Prim::F16, RawTensor::Float(vec![3.0, 5.0])).unwrap(),
            ),
        ] {
            let old =
                eval_tensor_roots_with_strict(&dag, &[], |_| Some(input.clone())).unwrap_err();
            let prepared =
                prepare_tensor_roots_inputs(&dag, &[], |_| Ok(Some(input.clone()))).unwrap();
            assert_eq!(
                prepared["x"].storage(),
                input.storage(),
                "preparation preserves the tagged input"
            );
            let actual =
                eval_tensor_roots_with_strict(&dag, &[], |name| prepared.get(name).cloned())
                    .unwrap_err();
            assert_eq!(actual, old);
        }
    }

    #[test]
    fn input_preparation_drop_and_bad_axis_fail_before_provider_effects() {
        let mut dropped = loads(&["x"]);
        let root = dropped.add_node(RiscOp::Drop, vec![NodeId(0)], ty(), None);
        let mut bad_axis = loads(&["x"]);
        let bad = bad_axis.add_node(
            RiscOp::Reshape { new_shape: vec![] },
            vec![NodeId(0)],
            ty(),
            None,
        );
        for (dag, root) in [(dropped, root), (bad_axis, bad)] {
            let old = eval_tensor_roots_with_strict(&dag, &[root], |_| {
                panic!("old validation precedes inputs")
            })
            .unwrap_err();
            let actual = prepare_tensor_roots_inputs(&dag, &[root], |_| {
                panic!("preparation validation precedes inputs")
            })
            .unwrap_err();
            assert_eq!(actual, old);
        }
    }

    #[test]
    fn input_preparation_plan_retains_dead_executed_loads_and_random_state() {
        let plan = plan(loads(&["x", "executed_but_value_dead"]));
        let mut old_context = context(42);
        let mut old_calls = Vec::new();
        let old = eval_tensor_plan_with_strict(&plan, &mut old_context, |name| {
            old_calls.push(name.to_owned());
            Some(value())
        })
        .unwrap();
        let mut new_context = context(42);
        let mut calls = Vec::new();
        let prepared = prepare_tensor_plan_inputs(&plan, &new_context, |name| {
            calls.push(name.to_owned());
            Ok(Some(value()))
        })
        .unwrap();
        assert_eq!(
            new_context.state().counter,
            5,
            "no execution frame enters during preparation"
        );
        assert_eq!(calls, ["x", "executed_but_value_dead"]);
        assert_eq!(calls, old_calls);
        let actual = eval_tensor_plan_with_strict(&plan, &mut new_context, |name| {
            prepared.get(name).cloned()
        })
        .unwrap();
        assert_eq!(actual, old);
        assert_eq!(new_context.state().counter, old_context.state().counter);
        assert_eq!(new_context.state().counter, 6);
        assert_eq!(calls, old_calls);
    }

    #[test]
    fn input_preparation_plan_seed_and_integrity_precede_provider_effects() {
        let valid = plan(loads(&["x"]));
        let mut invalid = valid.clone();
        invalid.metadata.spine.steps.pop();
        for (plan, seed) in [(valid, 43), (invalid, 42)] {
            let mut context = context(seed);
            let old = eval_tensor_plan_with_strict(&plan, &mut context, |_| {
                panic!("old validation precedes inputs")
            })
            .unwrap_err();
            let actual = prepare_tensor_plan_inputs(&plan, &context, |_| {
                panic!("preparation validation precedes inputs")
            })
            .unwrap_err();
            assert_eq!(actual, old);
            assert_eq!(context.state().counter, 5);
        }
    }

    #[test]
    fn input_preparation_provider_draw_precedes_original_plan_without_relowering() {
        let initializer = plan(loads(&["seed_input"]));
        let consumer = plan(loads(&["weights"]));
        let initializer_root = initializer.dag_for_inspection().roots()[0];
        let mut reference_context = context(42);
        let initialized =
            eval_tensor_plan_with_strict(&initializer, &mut reference_context, |_| Some(value()))
                .unwrap();
        let expected = eval_tensor_plan_with_strict(&consumer, &mut reference_context, |_| {
            Some(initialized[&initializer_root].clone())
        })
        .unwrap();
        let wrong_ordinal = eval_tensor_plan_with_strict(&consumer, &mut context(42), |_| {
            Some(initialized[&initializer_root].clone())
        })
        .unwrap();
        assert_ne!(
            expected, wrong_ordinal,
            "fixture distinguishes consumer ordinals 5 and 6"
        );

        let mut execution_context = context(42);
        let before_provider = execution_context.clone();
        let mut calls = 0;
        let prepared = prepare_tensor_plan_inputs(&consumer, &before_provider, |name| {
            assert_eq!(name, "weights");
            calls += 1;
            eval_tensor_plan_with_strict(&initializer, &mut execution_context, |_| Some(value()))
                .map(|values| Some(values[&initializer_root].clone()))
        })
        .unwrap();
        assert_eq!(before_provider.state().counter, 5);
        assert_eq!(execution_context.state().counter, 6);
        let actual = eval_tensor_plan_with_strict(&consumer, &mut execution_context, |name| {
            prepared.get(name).cloned()
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(actual, expected);
        for root in consumer.dag_for_inspection().roots() {
            assert_eq!(actual[root].storage(), expected[root].storage());
        }
        assert_eq!(
            execution_context.state().seed,
            reference_context.state().seed
        );
        assert_eq!(
            execution_context.state().counter,
            reference_context.state().counter
        );
        assert_eq!(execution_context.state().counter, 7);
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

    fn resource_fixture(device: &str, draw: bool) -> EvaluationPlan {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let mut metadata = ExecutionMetadata::new(Some(42));
        let before = (dag.len(), dag.roots().to_vec());
        metadata.spine.require(&dag, device.to_owned());
        assert_eq!(before, (dag.len(), dag.roots().to_vec()));
        let root = if draw {
            let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
            let root = dag.add_node(
                RiscOp::Dropout {
                    rate: 0.5,
                    seed: 42,
                },
                vec![x],
                ty,
                None,
            );
            metadata.forward(root, ScopeId(0));
            root
        } else {
            dag.add_node(RiscOp::synth_const(Prim::F32, 1.0), vec![], ty, None)
        };
        dag.add_root(root);
        metadata.spine.record_nodes(&dag);
        metadata.complete(&dag).unwrap();
        EvaluationPlan::new(dag, metadata).unwrap()
    }

    #[test]
    fn resource_cut_is_value_free_and_exactly_validated() {
        let plan = resource_fixture("cpu:author-device", false);
        assert!(plan.metadata.spine.has_full_sidecar_for_test());
        assert!(plan.source_for_inspection().is_empty());
        assert_eq!(plan.steps_for_inspection(), [Step::Node(NodeId(0))]);
        #[cfg(feature = "lowering-trace")]
        {
            let observed = plan.full_spine_for_inspection();
            assert!(matches!(
                observed.source.as_slice(),
                [crate::lowering_trace::FullSourceOccurrence {
                    id: crate::lowering_trace::SourceEventId(0),
                    kind: crate::lowering_trace::FullSourceKind::Requirement(device),
                }] if device == "cpu:author-device"
            ));
            assert!(matches!(
                observed.steps.as_slice(),
                [
                    crate::lowering_trace::FullStep::Requirement {
                        occurrence: crate::lowering_trace::SourceEventId(0)
                    },
                    crate::lowering_trace::FullStep::Node(NodeId(0))
                ]
            ));
        }

        for corruption in 0..3 {
            let mut damaged = plan.clone();
            damaged
                .metadata
                .spine
                .corrupt_requirement_for_test(corruption);
            assert!(
                damaged.validate().unwrap_err().contains("source"),
                "corruption {corruption}"
            );
        }

        let mut reordered = resource_fixture("cpu:author-device", true);
        reordered.metadata.spine.corrupt_requirement_for_test(3);
        assert!(reordered.validate().is_err());
    }

    #[test]
    fn full_source_ids_do_not_alias_random_occurrence_ids() {
        let plan = resource_fixture("cpu:author-device", true);
        assert_eq!(
            plan.metadata
                .spine
                .full_source_ids_for_test()
                .map(|id| id.0)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert!(matches!(
            plan.source_for_inspection(),
            [crate::execution_spine::Occurrence {
                id: OccurrenceId(0),
                kind: SourceKind::Forward { .. },
            }]
        ));
        #[cfg(feature = "lowering-trace")]
        {
            let observed = plan.full_spine_for_inspection();
            assert!(matches!(
                observed.source.as_slice(),
                [
                    crate::lowering_trace::FullSourceOccurrence {
                        id: crate::lowering_trace::SourceEventId(0),
                        kind: crate::lowering_trace::FullSourceKind::Requirement(_),
                    },
                    crate::lowering_trace::FullSourceOccurrence {
                        id: crate::lowering_trace::SourceEventId(1),
                        kind: crate::lowering_trace::FullSourceKind::Forward { .. },
                    }
                ]
            ));
        }
    }

    #[test]
    fn late_resource_promotion_preserves_prior_random_ids_and_mixed_order() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let mut metadata = ExecutionMetadata::new(None);
        metadata.scopes.push(Scope { seed: Some(42) });
        metadata.spine.control(
            &dag,
            Control::Enter {
                scope: ScopeId(1),
                seed: 42,
            },
        );
        let input = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let draw = dag.add_node(
            RiscOp::Dropout {
                rate: 0.5,
                seed: 42,
            },
            vec![input],
            ty,
            None,
        );
        metadata.forward(draw, ScopeId(1));
        metadata.spine.require(&dag, "cpu:late-promotion".into());
        metadata
            .spine
            .control(&dag, Control::Leave { scope: ScopeId(1) });
        dag.add_root(draw);
        metadata.spine.record_nodes(&dag);
        metadata.complete(&dag).unwrap();
        let plan = EvaluationPlan::new(dag, metadata).unwrap();

        assert_eq!(
            plan.source_for_inspection()
                .iter()
                .map(|event| event.id.index())
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            plan.metadata
                .spine
                .full_source_ids_for_test()
                .map(|id| id.0)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        #[cfg(feature = "lowering-trace")]
        {
            let full = plan.full_spine_for_inspection();
            assert!(matches!(
                full.source[2].kind,
                crate::lowering_trace::FullSourceKind::Requirement(ref device)
                    if device == "cpu:late-promotion"
            ));
            let position = |wanted| {
                full.steps
                    .iter()
                    .position(|step| match (wanted, step) {
                        (0, crate::lowering_trace::FullStep::Control { occurrence, .. }) => {
                            occurrence.0 == 0
                        }
                        (1, crate::lowering_trace::FullStep::Node(node)) => *node == draw,
                        (2, crate::lowering_trace::FullStep::Requirement { occurrence }) => {
                            occurrence.0 == 2
                        }
                        (3, crate::lowering_trace::FullStep::Control { occurrence, .. }) => {
                            occurrence.0 == 3
                        }
                        _ => false,
                    })
                    .unwrap()
            };
            assert!(position(0) < position(1));
            assert!(position(1) < position(2));
            assert!(position(2) < position(3));
        }
    }

    #[test]
    fn established_random_spine_carriers_remain_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<crate::execution_spine::Step>();
        assert_copy::<crate::execution_spine::SourceKind>();
        assert_copy::<crate::execution_spine::Occurrence>();
    }

    #[test]
    fn growing_source_events_do_not_rebuild_the_legacy_projection_per_event() {
        let mut metadata = ExecutionMetadata::new(Some(42));
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        for _ in 0..1024 {
            let input = dag.add_node(
                RiscOp::synth_const(Prim::F32, 1.0),
                vec![],
                ty.clone(),
                None,
            );
            let draw = dag.add_node(
                RiscOp::Dropout {
                    rate: 0.5,
                    seed: 42,
                },
                vec![input],
                ty.clone(),
                None,
            );
            metadata.forward(draw, ScopeId(0));
        }
        assert!(!metadata.spine.has_full_sidecar_for_test());
        assert_eq!(metadata.spine.legacy_projection_rebuilds_for_test(), 0);
        metadata.spine.record_nodes(&dag);
        assert_eq!(metadata.spine.legacy_projection_rebuilds_for_test(), 0);
        metadata.spine.complete(&dag).unwrap();
        assert_eq!(metadata.spine.legacy_projection_rebuilds_for_test(), 0);
    }

    #[test]
    fn evaluator_profile_still_declines_resource_scopes() {
        let source = "(handle-effect {effect: resource} (lit {} \"cpu:author-device\") \
                      (handle-effect {effect: random} \
                      (lit {type: (t-prim {} int64)} 42) \
                      (app {} (var {} dropout) (var {} x) \
                      (lit {type: (t-prim {} f32)} 0.5))))";
        let expression = chelis_deep::parser::parse_str(source)
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(
            crate::lower::evaluation_profile(&expression, &UnordMap::new()),
            EvaluationProfile::Legacy(LegacyEvaluationReason::ResourceScope)
        );
        let error = crate::lower::try_lower_subexpr_evaluation_plan(
            &expression,
            [(
                "x".into(),
                TensorType {
                    dims: vec![DimInfo::Lit(4)],
                    precision: Prim::F32,
                },
            )]
            .into_iter()
            .collect(),
            UnordMap::new(),
            UnordMap::new(),
            &context(),
        )
        .unwrap_err();
        assert!(error.message.contains("ResourceScope"), "{error}");
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
    fn source_ad_replays_signed_and_nonfinite_cotangents_with_the_saved_key() {
        use chelis_types::dtype_semantics::StorageView;

        let bits = |value: &TensorValue| -> Vec<u64> {
            match value.storage().view() {
                StorageView::F16(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
                StorageView::Bf16(values) => {
                    values.iter().map(|v| u64::from(v.to_bits())).collect()
                }
                StorageView::F32(values) => values.iter().map(|v| u64::from(v.to_bits())).collect(),
                StorageView::F64(values) => values.iter().map(|v| v.to_bits()).collect(),
                _ => panic!("expected a floating tensor"),
            }
        };
        // [04-NUM-2]/[05-OP-37]: independently encoded results of g / 0.5.
        // These are IEEE conformance cases, not derivatives over nonfinite reals.
        let cotangents = [-0.0, -3.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        for (prim, kept_bits, two_bits, minus_three_bits) in [
            (
                Prim::F16,
                [0x8000, 0xc600, 0, 0x7c00, 0xfc00, 0x7e00],
                0x4000,
                0xc200,
            ),
            (
                Prim::Bf16,
                [0x8000, 0xc0c0, 0, 0x7f80, 0xff80, 0x7fc0],
                0x4000,
                0xc040,
            ),
            (
                Prim::F32,
                [
                    0x80000000, 0xc0c00000, 0, 0x7f800000, 0xff800000, 0x7fc00000,
                ],
                0x40000000,
                0xc0400000,
            ),
            (
                Prim::F64,
                [
                    0x8000000000000000,
                    0xc018000000000000,
                    0,
                    0x7ff0000000000000,
                    0xfff0000000000000,
                    0x7ff8000000000000,
                ],
                0x4000000000000000,
                0xc008000000000000,
            ),
        ] {
            // Only v is differentiated; weights is a captured tensor input.
            let source = format!(
                "def sample(x: tensor[4, {prim}], weights: tensor[4, {prim}]) -> tensor[4, {prim}] = {{\n loss = fn (v: tensor[4, {prim}]) -> tensor_to_scalar(sum(mul(dropout(v, 0.5{prim}), weights), 0i32))\n grad(loss)(x)\n }}\n def next(x: tensor[4, {prim}]) -> tensor[4, {prim}] = dropout(x, 0.5{prim})",
                prim = prim.name(),
            );
            let declarations = chelis_surf::parser::parse_str(&source).unwrap();
            let checked = chelis_types::check_ir_program(&chelis_surf::desugar::desugar_program(
                &declarations,
            ))
            .unwrap();
            let session = crate::host::HostLoweringSession::new(&checked);
            let plan = |name| {
                crate::host::host_def_evaluation_plan(&session, name, &context())
                    .unwrap()
                    .unwrap()
                    .plan()
                    .unwrap()
                    .clone()
            };
            let forward_and_ad = plan("sample");
            let next = plan("next");
            let sites = forward_and_ad
                .metadata
                .spine
                .nodes()
                .iter()
                .filter_map(|node| {
                    forward_and_ad
                        .metadata
                        .sites
                        .get(node)
                        .map(|site| (*node, *site))
                })
                .collect::<Vec<_>>();
            assert!(
                matches!(
                    sites.as_slice(),
                    [
                        (
                            _,
                            RandomSite::Forward {
                                draw: DrawId(0),
                                scope: ScopeId(0)
                            }
                        ),
                        (_, RandomSite::Replay { draw: DrawId(0) }),
                    ]
                ),
                "{sites:?}"
            );
            let input =
                TensorValue::finalize_from_wide("load", prim, vec![4], vec![1.0; 4]).unwrap();
            // Each value reaches every coordinate, both a kept and dropped site.
            for offset in 0..cotangents.len() {
                let weights = TensorValue::finalize_from_wide(
                    "load",
                    prim,
                    vec![4],
                    (0..4)
                        .map(|i| cotangents[(offset + i) % cotangents.len()])
                        .collect(),
                )
                .unwrap();
                let mut state = context();
                {
                    let mut frame = forward_and_ad.frame(&mut state).unwrap();
                    let values = crate::eval::eval_tensor_segment_with_strict(
                        &forward_and_ad.dag,
                        &mut frame,
                        |name| match name {
                            "x" => Some(input.clone()),
                            "weights" => Some(weights.clone()),
                            _ => None,
                        },
                    )
                    .unwrap();
                    // Independent [05-RNG-1] seed42 ordinal0 mask is [drop, keep, drop, drop].
                    let forward = &values[&sites[0].0];
                    assert_eq!(forward.prim(), prim);
                    assert_eq!(forward.shape, [4]);
                    assert_eq!(bits(forward), [0, two_bits, 0, 0]);
                    // spec/06 §2.4 adds an exact +0 base before replay and
                    // again at the gradient root: +0 + (-0) is +0 under RNE.
                    let replay_input = forward_and_ad.dag.get(sites[1].0).unwrap().inputs[0];
                    assert_eq!(values[&replay_input].prim(), prim);
                    assert_eq!(values[&replay_input].shape, [4]);
                    let expected_input = (0..4)
                        .map(|i| match (offset + i) % cotangents.len() {
                            0 => 0,
                            1 => minus_three_bits,
                            other => kept_bits[other],
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        bits(&values[&replay_input]),
                        expected_input,
                        "replay cotangent"
                    );
                    let kept = (offset + 1) % cotangents.len();
                    let expected = [0, if kept == 0 { 0 } else { kept_bits[kept] }, 0, 0];
                    for node in [sites[1].0, forward_and_ad.dag.roots()[0]] {
                        let gradient = &values[&node];
                        assert_eq!(gradient.prim(), prim);
                        assert_eq!(gradient.shape, [4]);
                        assert_eq!(
                            bits(gradient),
                            expected,
                            "{prim:?}, offset={offset}, node={node:?}"
                        );
                    }
                    if offset == 0 {
                        // A direct -0 cotangent at this already-realized replay
                        // boundary precedes accumulation and must keep its sign.
                        let negative_zero =
                            TensorValue::finalize_from_wide("load", prim, vec![4], vec![-0.0; 4])
                                .unwrap();
                        let replay = frame.dropout(sites[1].0, &negative_zero, 0.5, 42).unwrap();
                        assert_eq!(replay.prim(), prim);
                        assert_eq!(replay.shape, [4]);
                        assert_eq!(bits(&replay), [0, kept_bits[0], 0, 0]);
                    }
                    assert_eq!(
                        frame
                            .keys
                            .iter()
                            .map(|key| key.map(|key| (key.seed, key.ordinal)))
                            .collect::<Vec<_>>(),
                        [Some((42, 0))]
                    );
                    assert_eq!(frame.counters, [0]);
                    assert_eq!(frame.scopes, [ScopeId(0)]);
                }
                assert_eq!(state.state().seed, Some(42));
                assert_eq!(state.state().counter, 1);
                let values = eval_tensor_plan_with_strict(&next, &mut state, |name| {
                    (name == "x").then(|| input.clone())
                })
                .unwrap();
                let result = &values[&next.dag.roots()[0]];
                assert_eq!(result.prim(), prim);
                assert_eq!(result.shape, [4]);
                assert_eq!(bits(result), [two_bits, 0, 0, 0], "ordinal1 continuation");
                assert_eq!(state.state().seed, Some(42));
                assert_eq!(state.state().counter, 2);
            }
        }
    }

    #[test]
    fn invalid_dtype_at_dropout_kernel_preserves_frames_keys_and_next_draw() {
        // [05-OP-37]/[05-RNG-1], E1 kernel-boundary evidence. Strict evaluator
        // Load rejects these tagged inputs earlier; this deliberately calls
        // the real dropout frame, not a well-typed source evaluation.
        for prim in [
            Prim::Bool,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
        ] {
            for count in [0, 4] {
                for nested in [false, true] {
                    let body = format!(
                        "(let {{}} (bind {{}} gradient (app {{}} (grad {{}} (fn {{}} (params {{}} (t {{type: (t-tensor {{}} (d-lit {{}} {count}) (t-prim {{}} f32))}})) (app {{type: (t-tensor {{}} (t-prim {{}} f32))}} (var {{}} sum) (app {{}} (var {{}} dropout) (var {{}} t) (lit {{type: (t-prim {{}} f32)}} 0.5)) (lit {{type: (t-prim {{}} int32)}} 0)))) (var {{}} x))) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} f32)}} 0.5)))"
                    );
                    let source = if nested {
                        format!(
                            "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} int64)}} 42) {body})"
                        )
                    } else {
                        body
                    };
                    let plan = lower_source_with_count(&source, count);
                    let bad = TensorValue::finalize_from_wide_int(
                        "load",
                        prim,
                        vec![count],
                        vec![1; count],
                    )
                    .unwrap();
                    let valid = TensorValue::finalize_from_wide(
                        "load",
                        Prim::F32,
                        vec![count],
                        vec![1.0; count],
                    )
                    .unwrap();
                    let mut state = context();
                    // A nested seed must preserve an already-advanced parent.
                    state.state.counter = u64::from(nested);
                    let snapshot = |frame: &ExecutionFrame<'_>| {
                        (
                            frame.context.state.seed,
                            frame.context.state.counter,
                            frame.counters.clone(),
                            frame.scopes.clone(),
                            frame
                                .keys
                                .iter()
                                .map(|key| key.map(|key| (key.seed, key.ordinal)))
                                .collect::<Vec<_>>(),
                        )
                    };
                    {
                        let mut frame = plan.frame(&mut state).unwrap();
                        let mut calls = 0;
                        for step in plan.steps_for_inspection() {
                            match *step {
                                Step::Control { control, .. } => frame.control(control).unwrap(),
                                Step::Node(node) => {
                                    let RiscOp::Dropout { rate, seed } =
                                        plan.dag.get(node).unwrap().op
                                    else {
                                        continue;
                                    };
                                    let before = snapshot(&frame);
                                    assert_eq!(
                                        frame.dropout(node, &bad, rate, seed).unwrap_err(),
                                        "dropout requires a floating tensor"
                                    );
                                    assert_eq!(
                                        snapshot(&frame),
                                        before,
                                        "{prim:?}, count={count}, nested={nested}, call={calls}"
                                    );
                                    let result = frame.dropout(node, &valid, rate, seed).unwrap();
                                    let expected = if calls < 2 {
                                        [0.0f64, 2.0, 0.0, 0.0]
                                    } else {
                                        [2.0f64, 0.0, 0.0, 0.0]
                                    };
                                    assert_eq!(result.prim(), Prim::F32);
                                    assert_eq!(result.shape, [count]);
                                    assert_eq!(
                                        result
                                            .to_f64_lossy_vec()
                                            .iter()
                                            .map(|value| value.to_bits())
                                            .collect::<Vec<_>>(),
                                        expected[..count]
                                            .iter()
                                            .map(|value| value.to_bits())
                                            .collect::<Vec<_>>()
                                    );
                                    calls += 1;
                                }
                            }
                        }
                        assert_eq!(calls, 3, "forward, replay, then next forward");
                        assert_eq!(
                            frame
                                .keys
                                .iter()
                                .map(|key| key.map(|key| (key.seed, key.ordinal)))
                                .collect::<Vec<_>>(),
                            [Some((42, 0)), Some((42, 1))]
                        );
                        assert_eq!(frame.scopes, [ScopeId(0)]);
                        assert_eq!(frame.counters, if nested { vec![0, 2] } else { vec![0] });
                    }
                    let preserved = if nested { 1 } else { 2 };
                    assert_eq!(state.state().seed, Some(42));
                    assert_eq!(state.state().counter, preserved);
                    let expected = if nested {
                        vec![2.0, 0.0, 0.0, 0.0]
                    } else {
                        vec![2.0, 2.0, 0.0, 0.0]
                    };
                    assert_eq!(
                        run(&fixture(&[0.5], 4, false), &mut state, 4).unwrap(),
                        expected
                    );
                    assert_eq!(state.state().counter, preserved + 1);
                }
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
        lower_source_with_count(source, 32)
    }

    fn lower_source_with_count(source: &str, count: usize) -> EvaluationPlan {
        let expr = chelis_deep::parser::parse_str(source)
            .unwrap()
            .pop()
            .unwrap();
        let inputs = [(
            "x".into(),
            TensorType {
                dims: vec![DimInfo::Lit(count)],
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
        let occurrences = plan.metadata.spine.full_source_ids_for_test().collect();
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
