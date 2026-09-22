//! Feature-only C vocabulary for observing actual invocation-local Random state.
//!
//! Everything emitted here has translation-unit-local linkage. The callback is
//! synchronous: frame pointers are valid only for the duration of the call, so
//! the observer cannot retain an unbounded event history in generated code.

pub(crate) const PRIVATE_PARAM: &str =
    "chelis_rng_state *__chelis_rng, __chelis_random_observer *__chelis_observer";
pub(crate) const PRIVATE_ARGS: &[&str] = &["__chelis_rng", "__chelis_observer"];

#[derive(Clone)]
pub(crate) struct SourceSite<'a> {
    pub(crate) verified: chelis_ir::ownership::VerifiedHostSourceSite<'a>,
    supported: bool,
    kind: usize,
    target: usize,
    seed: u64,
    inherited_scope: usize,
    events: Vec<String>,
}

impl SourceSite<'_> {
    pub(crate) fn supported(&self) -> bool {
        self.supported
    }

    pub(crate) fn matches(
        &self,
        site: &crate::host_abi::ProjectedHostSite<'_>,
        expr: &crate::host_abi::HostAbiExpr,
    ) -> bool {
        use crate::host_abi::HostAbiExprKind as E;
        use chelis_ir::host::ConcreteHostExprKind as S;
        if self.verified.site().id() != site.id {
            return false;
        }
        match (&self.verified.expression().kind, &expr.kind) {
            (S::WithSeed { seed: source, .. }, E::WithSeed { seed, .. }) => {
                match (&source.kind, &seed.kind) {
                    (S::Int(a), E::Int(b)) => self.kind == 1 && a == b && self.seed == *a as u64,
                    _ => !self.supported && self.kind == 1,
                }
            }
            (S::TensorCall { helper: a, .. }, E::TensorCall { helper: b, .. }) => {
                self.kind == 3 && a == b && self.target == *a
            }
            (S::Call { function: a, .. }, E::Call { function: b, .. }) => {
                let projected = site
                    .directives
                    .iter()
                    .filter_map(|action| match action {
                        chelis_ir::ownership::VerifiedHostAction::Operation(
                            chelis_ir::ownership::VerifiedHostOperation::Apply {
                                kind:
                                    chelis_ir::ownership::VerifiedApplyKind::DirectCall {
                                        callee, ..
                                    },
                                ..
                            },
                        ) => Some(*callee),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                self.kind == 2
                    && a == b
                    && projected
                        == self
                            .verified
                            .direct_callee()
                            .into_iter()
                            .collect::<Vec<_>>()
                    && self
                        .verified
                        .direct_callee_word()
                        .is_none_or(|callee| callee == self.target)
            }
            (_, E::WithSeed { .. } | E::TensorCall { .. } | E::Call { .. }) => false,
            _ => self.kind == 0,
        }
    }
    pub(crate) fn name(&self) -> String {
        format!("__chelis_random_source_{}", self.verified.site_word())
    }

    pub(crate) fn pointer(&self) -> String {
        format!("&{}", self.name())
    }

    fn declaration(&self) -> Vec<String> {
        let name = self.name();
        let mut lines = Vec::new();
        let events = if self.events.is_empty() {
            "NULL".to_string()
        } else {
            lines.push(format!(
                "static const __chelis_random_source_event {name}_events[] = {{{}}};",
                self.events.join(", ")
            ));
            format!("{name}_events")
        };
        let (has_parent, parent) = optional_u64(self.verified.seed_parent_word());
        lines.push(format!(
            "static const __chelis_random_source_site {name} = {{{}, {{{}ULL}}, {{{}ULL}}, {}, {{{}ULL}}, {{{}ULL}}, {has_parent}, {{{parent}}}, {{{}ULL}}, {events}, {}ULL}};",
            usize::from(self.supported), self.verified.unit_word(), self.verified.site_word(), self.kind,
            self.target, self.seed, self.inherited_scope, self.events.len()
        ));
        lines
    }
}

fn value_only(expr: &chelis_ir::host::ConcreteHostExpr) -> bool {
    use chelis_ir::host::ConcreteHostExprKind as E;
    match &expr.kind {
        E::ResultClaimScope { body, .. } => value_only(body),
        E::FormalIngress { value, .. } => value_only(value),
        E::Int(_) | E::Float(_) | E::Bool(_) | E::String(_) | E::Var(..) | E::Unit => true,
        E::Tuple(items, _) | E::List(items, _) => items.iter().all(value_only),
        E::Builtin { name, args, .. } if (name == "cast" || name == "copy") && args.len() == 1 => {
            args.iter().all(value_only)
        }
        _ => false,
    }
}

fn straight_line(expr: &chelis_ir::host::ConcreteHostExpr) -> bool {
    use chelis_ir::host::ConcreteHostExprKind as E;
    if value_only(expr) {
        return true;
    }
    match &expr.kind {
        E::Let { bindings, body, .. } | E::RetainedInvocation { bindings, body, .. } => {
            bindings.iter().all(|b| straight_line(&b.value)) && straight_line(body)
        }
        E::ResultClaimScope { body, .. } => straight_line(body),
        E::FormalIngress { value, .. } => straight_line(value),
        E::Tuple(items, _) => items.iter().all(straight_line),
        E::WithSeed { seed, body, .. } => matches!(seed.kind, E::Int(_)) && straight_line(body),
        E::TensorCall { args, .. } | E::Call { args, .. } | E::SignatureEntry { args, .. } => {
            args.iter().all(value_only)
        }
        _ => false,
    }
}

/// The only constructor uses sealed source cursors and their own helper plans.
/// Local/full joins compare complete kinds, not arithmetic offsets or C labels.
pub(crate) fn source_sites(
    emission: chelis_ir::ownership::VerifiedHostEmission<'_>,
) -> Vec<SourceSite<'_>> {
    use chelis_ir::execution_spine::{Control, SourceKind};
    use chelis_ir::host::ConcreteHostExprKind as E;
    use chelis_ir::lowering_trace::FullSourceKind as F;
    emission
        .source_expressions()
        .into_iter()
        .map(|verified| {
            let function = verified
                .unit_word()
                .checked_sub(1)
                .and_then(|i| emission.function(i));
            let mut site = SourceSite {
                verified,
                supported: function
                    .is_some_and(|f| f.specialization().is_none() && straight_line(f.body())),
                kind: 0,
                target: 0,
                seed: 0,
                inherited_scope: 0,
                events: Vec::new(),
            };
            match &verified.expression().kind {
                E::WithSeed { seed, .. } => {
                    site.kind = 1;
                    if let E::Int(seed) = seed.kind {
                        site.seed = seed as u64;
                    } else {
                        site.supported = false;
                    }
                }
                E::Call { function, args, .. } => {
                    site.kind = 2;
                    let declared = (0..emission.function_count())
                        .find(|&i| emission.function(i).is_some_and(|f| f.name() == function));
                    let callee = verified.direct_callee_word();
                    site.supported &= args.iter().all(value_only)
                        && callee.is_some()
                        && callee == declared.map(|i| i + 1);
                    if let Some(callee) = callee {
                        site.target = callee;
                    }
                }
                E::TensorCall { helper, args, .. } => {
                    site.kind = 3;
                    site.target = *helper;
                    site.supported &= args.iter().all(value_only);
                    if let Some(execution) = function
                        .and_then(|f| f.tensor_helper(*helper))
                        .and_then(|h| h.execution())
                    {
                        site.inherited_scope = execution.inherited_scope().index();
                        let full = execution.full_source();
                        let random: Vec<_> = full
                            .source
                            .iter()
                            .filter(|e| !matches!(e.kind, F::Requirement(_)))
                            .collect();
                        site.supported &= random.len() == execution.source().len();
                        for (legacy, full) in execution.source().iter().zip(random) {
                            let (kind, expected, draw, scope) = match legacy.kind {
                                SourceKind::Forward { node, draw, scope } => (
                                    "__CHELIS_RANDOM_OBSERVER_FORWARD",
                                    F::Forward { node, draw, scope },
                                    Some(draw.index()),
                                    scope.index(),
                                ),
                                SourceKind::Control(control) => match control {
                                    Control::Enter { scope, .. } => (
                                        "__CHELIS_RANDOM_OBSERVER_FIXED_ENTER",
                                        F::Control(control),
                                        None,
                                        scope.index(),
                                    ),
                                    Control::Leave { scope } => (
                                        "__CHELIS_RANDOM_OBSERVER_FIXED_LEAVE",
                                        F::Control(control),
                                        None,
                                        scope.index(),
                                    ),
                                },
                            };
                            site.supported &= expected == full.kind;
                            let (has_draw, draw) = optional_u64(draw);
                            site.events.push(format!(
                                "{{{kind}, {{{}ULL}}, {has_draw}, {{{draw}}}, {{{scope}ULL}}, {{{}ULL}}}}",
                                legacy.id.index(),
                                full.id.0
                            ));
                        }
                    }
                }
                _ => {}
            }
            site
        })
        .collect()
}

pub(crate) fn append_source_sites(out: &mut Vec<String>, sites: &[SourceSite<'_>]) {
    for site in sites.iter().filter(|s| s.kind != 0) {
        out.extend(site.declaration());
    }
}

pub(crate) fn source_bijection(
    sources: &[SourceSite<'_>],
    sites: &[crate::host_abi::ProjectedHostSite<'_>],
) -> bool {
    sources.len() == sites.len()
        && sites.iter().all(|site| {
            sources
                .iter()
                .filter(|source| source.verified.site().id() == site.id)
                .count()
                == 1
        })
}

pub(crate) fn push_source_call(indent: &str, name: &str, source: &str) -> Vec<String> {
    vec![
        format!(
            "{indent}__chelis_random_call_frame {name} = __chelis_random_push_call(__chelis_observer, {source});"
        ),
        format!("{indent}if (__chelis_observer != NULL) __chelis_observer->calls = &{name};"),
    ]
}

pub(crate) fn pop_source_call(indent: &str, name: &str) -> String {
    format!("{indent}if (__chelis_observer != NULL) __chelis_observer->calls = {name}.previous;")
}

pub(crate) fn append_support(out: &mut Vec<String>) {
    out.extend(SUPPORT.lines().map(ToOwned::to_owned));
}

pub(crate) fn append_inactive_context(out: &mut Vec<String>, indent: &str) {
    out.push(format!(
        "{indent}__chelis_random_observer *__chelis_observer = NULL;"
    ));
}

pub(crate) fn append_observed_context(out: &mut Vec<String>, indent: &str) {
    out.push(format!("{indent}if (__chelis_sink == NULL) {{ abort(); }}"));
    out.push(format!(
        "{indent}__chelis_random_observer __chelis_observer_local = {{__chelis_sink, __chelis_sink_context, {{__chelis_invocation}}, {{0ULL}}, NULL, NULL, NULL, 0ULL, 1}};"
    ));
    out.push(format!(
        "{indent}__chelis_random_observer *__chelis_observer = &__chelis_observer_local;"
    ));
    out.push(record(
        indent,
        "__CHELIS_RANDOM_OBSERVER_INVOCATION_INIT",
        "__CHELIS_RANDOM_OBSERVER_INVOCATION_IDENTITY",
        "NULL",
        None,
        None,
        None,
        "*__chelis_rng",
        None,
    ));
}

pub(crate) fn push_frame(
    indent: &str,
    frame_var: &str,
    saved_state_expression: &str,
) -> Vec<String> {
    vec![
        format!(
            "{indent}__chelis_random_observer_frame {frame_var} = {{{saved_state_expression}, __chelis_observer != NULL ? __chelis_observer->saved : NULL}};"
        ),
        format!(
            "{indent}if (__chelis_observer != NULL) {{ __chelis_observer->saved = &{frame_var}; }}"
        ),
    ]
}

pub(crate) fn pop_frame(indent: &str, frame_var: &str) -> String {
    format!(
        "{indent}if (__chelis_observer != NULL) {{ __chelis_observer->saved = {frame_var}.previous; }}"
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    indent: &str,
    event: &str,
    identity: &str,
    producer: &str,
    occurrence: Option<usize>,
    draw: Option<usize>,
    scope: Option<usize>,
    state_expression: &str,
    used: Option<(&str, &str)>,
) -> String {
    let (has_occurrence, occurrence) = optional_u64(occurrence);
    let (has_draw, draw) = optional_u64(draw);
    let (has_scope, scope) = optional_u64(scope);
    let (has_used, used_seed, used_counter) = match used {
        Some((seed, counter)) => ("1", seed, counter),
        None => ("0", "0ULL", "0ULL"),
    };
    format!(
        "{indent}__chelis_random_observer_record(__chelis_observer, {event}, {identity}, {producer}, {has_occurrence}, {occurrence}, {has_draw}, {draw}, {has_scope}, {scope}, {state_expression}, {has_used}, {used_seed}, {used_counter});"
    )
}

fn optional_u64(value: Option<usize>) -> (&'static str, String) {
    value.map_or_else(
        || ("0", "0ULL".to_string()),
        |value| ("1", format!("{value}ULL")),
    )
}

const SUPPORT: &str = r#"/* CHELIS_NATIVE_RANDOM_OBSERVER_BEGIN */
typedef struct { uint64_t value; } __chelis_random_observer_u64;
typedef struct {
    int active;
    __chelis_random_observer_u64 seed;
    __chelis_random_observer_u64 counter;
} __chelis_random_observer_state;
typedef struct __chelis_random_observer_frame {
    chelis_rng_state state;
    const struct __chelis_random_observer_frame *previous;
} __chelis_random_observer_frame;
typedef enum {
    __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT,
    __CHELIS_RANDOM_OBSERVER_HOST_INSTALL,
    __CHELIS_RANDOM_OBSERVER_HOST_RESTORE,
    __CHELIS_RANDOM_OBSERVER_FIXED_ENTER,
    __CHELIS_RANDOM_OBSERVER_FIXED_LEAVE,
    __CHELIS_RANDOM_OBSERVER_FORWARD,
    __CHELIS_RANDOM_OBSERVER_REPLAY
} __chelis_random_observer_event_kind;
typedef enum {
    __CHELIS_RANDOM_OBSERVER_INVOCATION_IDENTITY,
    __CHELIS_RANDOM_OBSERVER_HOST_IDENTITY_UNSUPPORTED,
    __CHELIS_RANDOM_OBSERVER_FIXED_IDENTITY
} __chelis_random_observer_identity_kind;
/* Separate namespaces: host sites, ownership units, helper-local source IDs.
 * Linked frames qualify repeated calls; no flattened occurrence counter. */
typedef struct {
    __chelis_random_observer_event_kind kind;
    __chelis_random_observer_u64 occurrence;
    int has_draw;
    __chelis_random_observer_u64 draw, scope, full;
} __chelis_random_source_event;
typedef struct {
    int supported;
    __chelis_random_observer_u64 unit, site;
    int kind; /* 0 value, 1 seed, 2 direct call, 3 helper call */
    __chelis_random_observer_u64 target, seed;
    int has_seed_parent;
    __chelis_random_observer_u64 seed_parent, inherited_scope;
    const __chelis_random_source_event *events;
    uint64_t event_count;
} __chelis_random_source_site;
typedef struct __chelis_random_call_frame {
    const __chelis_random_source_site *source;
    const struct __chelis_random_call_frame *previous;
    int certified;
} __chelis_random_call_frame;
typedef struct __chelis_random_host_frame {
    const __chelis_random_source_site *source;
    const struct __chelis_random_host_frame *previous;
    const __chelis_random_call_frame *calls;
    int certified;
} __chelis_random_host_frame;
typedef struct {
    __chelis_random_observer_event_kind kind;
    __chelis_random_observer_identity_kind identity;
    const char *producer;
    __chelis_random_observer_u64 invocation;
    __chelis_random_observer_u64 sequence;
    int has_occurrence;
    __chelis_random_observer_u64 occurrence;
    int has_draw;
    __chelis_random_observer_u64 draw;
    int has_scope;
    __chelis_random_observer_u64 scope;
    __chelis_random_observer_state state;
    const __chelis_random_observer_frame *saved;
    int has_used;
    __chelis_random_observer_u64 used_seed;
    __chelis_random_observer_u64 used_counter;
    int has_continuation;
    __chelis_random_observer_u64 continuation_seed;
    __chelis_random_observer_u64 continuation_counter;
    __chelis_random_observer_u64 continuation_successor;
    const __chelis_random_source_site *source;
    const __chelis_random_call_frame *calls;
    const __chelis_random_host_frame *host_scope;
    int source_certified;
    int has_full_source;
    __chelis_random_observer_u64 full_source;
    int scope_is_inherited;
} __chelis_random_observer_event;
typedef int (*__chelis_random_observer_sink)(void *, const __chelis_random_observer_event *);
typedef struct {
    __chelis_random_observer_sink sink;
    void *sink_context;
    __chelis_random_observer_u64 invocation;
    __chelis_random_observer_u64 sequence;
    const __chelis_random_observer_frame *saved;
    const __chelis_random_call_frame *calls;
    const __chelis_random_host_frame *host;
    uint64_t unit;
    int admitted;
} __chelis_random_observer;
typedef struct { uint64_t unit; int admitted; } __chelis_random_function_frame;
static inline __chelis_random_function_frame __chelis_random_push_function(__chelis_random_observer *observer, uint64_t unit, int supported) {
    __chelis_random_function_frame frame = {observer != NULL ? observer->unit : 0ULL, observer != NULL ? observer->admitted : 0};
    if (observer != NULL) {
        observer->unit = unit;
        observer->admitted = frame.admitted && supported && (observer->calls == NULL ||
            (observer->calls->certified && observer->calls->source != NULL &&
             observer->calls->source->kind == 2 && observer->calls->source->target.value == unit));
    }
    return frame;
}
static inline void __chelis_random_pop_function(__chelis_random_observer *observer, __chelis_random_function_frame frame) {
    if (observer != NULL) { observer->unit = frame.unit; observer->admitted = frame.admitted; }
}
static inline int __chelis_random_source_admitted(const __chelis_random_observer *observer, const __chelis_random_source_site *source) {
    if (observer == NULL || source == NULL || !observer->admitted || !source->supported || source->unit.value != observer->unit) return 0;
    if (observer->calls != NULL && !observer->calls->certified) return 0;
    if (observer->host != NULL && !observer->host->certified) return 0;
    if (source->has_seed_parent && (observer->host == NULL || observer->host->source == NULL ||
        observer->host->source->unit.value != source->unit.value ||
        observer->host->source->site.value != source->seed_parent.value)) return 0;
    return 1;
}
static inline __chelis_random_call_frame __chelis_random_push_call(const __chelis_random_observer *observer, const __chelis_random_source_site *source) {
    __chelis_random_call_frame frame = {source, observer != NULL ? observer->calls : NULL,
        __chelis_random_source_admitted(observer, source) && (source->kind == 2 || source->kind == 3)};
    return frame;
}
static inline __chelis_random_host_frame __chelis_random_push_host(const __chelis_random_observer *observer, const __chelis_random_source_site *source, uint64_t seed) {
    __chelis_random_host_frame frame = {source, observer != NULL ? observer->host : NULL,
        observer != NULL ? observer->calls : NULL,
        __chelis_random_source_admitted(observer, source) && source->kind == 1 && source->seed.value == seed};
    return frame;
}
static inline __chelis_random_observer_state __chelis_random_observer_state_of(chelis_rng_state state) {
    __chelis_random_observer_state observed = {state.active, {state.seed}, {state.counter}};
    return observed;
}
static inline void __chelis_random_observer_record(
    __chelis_random_observer *observer,
    __chelis_random_observer_event_kind kind,
    __chelis_random_observer_identity_kind identity,
    const char *producer,
    int has_occurrence,
    uint64_t occurrence,
    int has_draw,
    uint64_t draw,
    int has_scope,
    uint64_t scope,
    chelis_rng_state state,
    int has_used,
    uint64_t used_seed,
    uint64_t used_counter
) {
    if (observer == NULL) return;
    chelis_rng_state continuation = state;
    __chelis_random_observer_event event = {
        kind, identity, producer, observer->invocation, observer->sequence,
        has_occurrence, {occurrence}, has_draw, {draw}, has_scope, {scope},
        __chelis_random_observer_state_of(state), observer->saved,
        has_used, {used_seed}, {used_counter},
        continuation.active, {continuation.seed}, {continuation.counter}, {0ULL},
        NULL, observer->calls, observer->host, 0, 0, {0ULL}, 0
    };
    if (identity == __CHELIS_RANDOM_OBSERVER_FIXED_IDENTITY && observer->calls != NULL) {
        event.source = observer->calls->source;
        event.source_certified = observer->admitted && observer->calls->certified && event.source != NULL && event.source->kind == 3;
        if (event.source_certified) {
            unsigned matches = 0;
            for (uint64_t i = 0; i < event.source->event_count; ++i) {
                const __chelis_random_source_event *entry = &event.source->events[i];
                int occurrence_matches = kind == __CHELIS_RANDOM_OBSERVER_REPLAY
                    ? entry->kind == __CHELIS_RANDOM_OBSERVER_FORWARD && !has_occurrence
                    : entry->kind == kind && has_occurrence && entry->occurrence.value == occurrence;
                if (occurrence_matches && entry->has_draw == has_draw && (!has_draw || entry->draw.value == draw) && has_scope && entry->scope.value == scope) {
                    event.full_source = entry->full;
                    matches++;
                }
            }
            event.has_full_source = matches == 1;
            event.source_certified = event.has_full_source;
            event.scope_is_inherited = scope == event.source->inherited_scope.value;
        }
    } else if ((kind == __CHELIS_RANDOM_OBSERVER_HOST_INSTALL || kind == __CHELIS_RANDOM_OBSERVER_HOST_RESTORE) && observer->host != NULL) {
        event.source = observer->host->source;
        event.source_certified = observer->admitted && observer->host->certified;
        if (kind == __CHELIS_RANDOM_OBSERVER_HOST_RESTORE) event.host_scope = observer->host->previous;
    }
    if (continuation.active) continuation.counter++;
    event.continuation_successor.value = continuation.counter;
    observer->sequence.value++;
    if (observer->sink(observer->sink_context, &event) != 0) abort();
}
/* CHELIS_NATIVE_RANDOM_OBSERVER_END */"#;

#[cfg(test)]
mod source_tests {
    use super::*;

    #[test]
    fn retained_signature_entry_preserves_straight_line_source_admission() {
        use chelis_ir::host::{
            ConcreteHostExpr as Expr, ConcreteHostExprKind as Kind, HostTensorInput,
            SignatureEntryPlan,
        };
        let tensor = chelis_ir::TensorType {
            dims: vec![chelis_ir::DimInfo::Lit(2)],
            precision: chelis_types::types::Prim::F32,
        };
        let ty = chelis_ir::ConcreteHostType::Tensor(tensor.clone());
        let plan = SignatureEntryPlan::new([HostTensorInput {
            name: "x".into(),
            ty: tensor,
        }]);
        let mut entry = Expr::new(Kind::SignatureEntry {
            plan,
            args: vec![Expr::new(Kind::Var("x".into(), ty.clone()))],
        });
        assert!(straight_line(&entry));
        let Kind::SignatureEntry { args, .. } = &mut entry.kind else {
            unreachable!()
        };
        args[0] = Expr::new(Kind::Call {
            function: "effectful_producer".into(),
            args: Vec::new(),
            arg_tys: Vec::new(),
            ty,
        });
        assert!(!straight_line(&entry));
    }

    #[test]
    fn source_identity_rejects_duplicate_orphan_kind_helper_and_seed_sidecars() {
        let verified = crate::host_abi_tests::verified_execution_host_from_source(
            r#"
def loss(x: tensor[4, f32]) -> f32 = with seed(7i64) { tensor_to_scalar(sum(dropout(x, 0.5f32), 0)) }
def run(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = with seed(42i64) {
 a = grad(loss)(x)
 b = with seed(99i64) { dropout(x, 0.5f32) }
 (a, b)
}
"#,
        );
        let projected = crate::host_abi::project_program(verified.emission()).unwrap();
        let all = source_sites(projected.source_emission());
        let unit = all
            .iter()
            .find(|site| site.seed == 42)
            .unwrap()
            .verified
            .unit_word();
        let sources: Vec<_> = all
            .iter()
            .filter(|site| site.verified.unit_word() == unit)
            .cloned()
            .collect();
        let sites: Vec<_> = projected
            .function_sites(unit - 1)
            .unwrap()
            .iter()
            .filter(|site| site.kind == chelis_ir::ownership::HostSiteKind::Expression)
            .cloned()
            .collect();
        assert!(source_bijection(&sources, &sites));
        let mut duplicate = sources.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(!source_bijection(&duplicate, &sites));
        let mut orphan = sources.clone();
        orphan.push(
            all.iter()
                .find(|site| site.verified.unit_word() != unit)
                .unwrap()
                .clone(),
        );
        assert!(!source_bijection(&orphan, &sites));
        assert!(!source_bijection(&sources[1..], &sites));
        for source in sources
            .iter()
            .filter(|site| site.kind == 1 || site.kind == 3)
        {
            let site = sites
                .iter()
                .find(|site| site.id == source.verified.site().id())
                .unwrap();
            // The ABI projection can only change types, not the source kind.
            let expr = if source.kind == 1 {
                crate::host_abi::HostAbiExpr::new(crate::host_abi::HostAbiExprKind::WithSeed {
                    seed: Box::new(crate::host_abi::HostAbiExpr::new(
                        crate::host_abi::HostAbiExprKind::Int(source.seed as i64),
                    )),
                    body: Box::new(crate::host_abi::HostAbiExpr::new(
                        crate::host_abi::HostAbiExprKind::Unit,
                    )),
                    ty: crate::host_abi::HostAbiType::Unit,
                })
            } else {
                crate::host_abi::HostAbiExpr::new(crate::host_abi::HostAbiExprKind::TensorCall {
                    helper: source.target,
                    args: Vec::new(),
                    ty: crate::host_abi::HostAbiType::Unit,
                })
            };
            assert!(source.matches(site, &expr));
            let mut wrong = source.clone();
            wrong.kind = 0;
            assert!(!wrong.matches(site, &expr));
            let mut wrong = source.clone();
            wrong.target += 1;
            wrong.seed = wrong.seed.wrapping_add(1);
            assert!(!wrong.matches(site, &expr));
            let other = sites.iter().find(|other| other.id != site.id).unwrap();
            assert!(!source.matches(other, &expr));
        }
    }
}
