//! Feature-only C vocabulary for observing actual invocation-local Random state.
//!
//! Everything emitted here has translation-unit-local linkage. The callback is
//! synchronous: frame pointers are valid only for the duration of the call, so
//! the observer cannot retain an unbounded event history in generated code.

pub(crate) const PRIVATE_PARAM: &str =
    "chelis_rng_state *__chelis_rng, __chelis_random_observer *__chelis_observer";
pub(crate) const PRIVATE_ARGS: &[&str] = &["__chelis_rng", "__chelis_observer"];

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
        "{indent}__chelis_random_observer __chelis_observer_local = {{__chelis_sink, __chelis_sink_context, {{__chelis_invocation}}, {{0ULL}}, NULL}};"
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
} __chelis_random_observer_event;
typedef int (*__chelis_random_observer_sink)(void *, const __chelis_random_observer_event *);
typedef struct {
    __chelis_random_observer_sink sink;
    void *sink_context;
    __chelis_random_observer_u64 invocation;
    __chelis_random_observer_u64 sequence;
    const __chelis_random_observer_frame *saved;
} __chelis_random_observer;
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
        continuation.active, {continuation.seed}, {continuation.counter}, {0ULL}
    };
    if (continuation.active) continuation.counter++;
    event.continuation_successor.value = continuation.counter;
    observer->sequence.value++;
    if (observer->sink(observer->sink_context, &event) != 0) abort();
}
/* CHELIS_NATIVE_RANDOM_OBSERVER_END */"#;
