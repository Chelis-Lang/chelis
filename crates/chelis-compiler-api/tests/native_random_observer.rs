//! Feature-only compile/run observations of actual generated-C Random locals.
#![cfg(feature = "native-random-observer")]

#[allow(dead_code)]
mod ownership_support;

use serde_json::{Value, json};

fn observed_source() -> String {
    ownership_support::emit_selected(
        r#"
def fixed_loss(x: tensor[4, f32]) -> f32 = with seed(7i64) {
  tensor_to_scalar(sum(dropout(x, 0.5f32), 0))
}

def run(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32], tensor[4, f32]) = with seed(42i64) {
  replayed = grad(fixed_loss)(x)
  nested = with seed(99i64) { dropout(x, 0.5f32) }
  following = dropout(x, 0.5f32)
  (replayed, nested, following)
}
"#,
        "run",
    )
}

#[test]
fn authored_observer_spellings_do_not_collide_with_private_support_or_wrappers() {
    let source = ownership_support::emit_selected(
        r#"
def victim(x: f32) -> f32 = x
def victim__chelis_observed(x: f32) -> f32 = x
def chelis_random_observer(x: f32) -> f32 = x
def chelis_random_observer_record(x: f32) -> f32 = x
def chelis_random_observer_state_of(x: f32) -> f32 = x
def run(x: f32) -> f32 = add(add(victim(x), victim__chelis_observed(x)), add(chelis_random_observer(x), add(chelis_random_observer_record(x), chelis_random_observer_state_of(x))))
"#,
        "run",
    );
    let driver = r#"
static int count_event(void *context, const __chelis_random_observer_event *event) {
    assert(event->kind == __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT);
    ++*(int *)context;
    return 0;
}
int main(void) {
    chelis_tensor *receipt = input(1);
    int count = 0;
    assert(run(2.0f) == 10.0f);
    assert(__chelis_observed_run(2.0f, count_event, &count, 11ULL) == 10.0f);
    assert(count == 1);
    chelis_tensor_release(receipt);
    return 0;
}
"#;
    ownership_support::balanced(&ownership_support::run(&source, driver));
}

const JSON_SINK: &str = r#"
static const char *event_name(__chelis_random_observer_event_kind kind) {
    switch (kind) {
        case __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT: return "invocation_init";
        case __CHELIS_RANDOM_OBSERVER_HOST_INSTALL: return "host_install";
        case __CHELIS_RANDOM_OBSERVER_HOST_RESTORE: return "host_restore";
        case __CHELIS_RANDOM_OBSERVER_FIXED_ENTER: return "fixed_enter";
        case __CHELIS_RANDOM_OBSERVER_FIXED_LEAVE: return "fixed_leave";
        case __CHELIS_RANDOM_OBSERVER_FORWARD: return "forward";
        case __CHELIS_RANDOM_OBSERVER_REPLAY: return "replay";
    }
    return "invalid";
}
static int put_u64(FILE *out, __chelis_random_observer_u64 value) {
    return fprintf(out, "{\"u64\":\"%" PRIu64 "\"}", value.value) < 0 ? -1 : 0;
}
static int put_state(FILE *out, __chelis_random_observer_state state) {
    if (!state.active) return fputs("{\"active\":false,\"seed\":null,\"counter\":null}", out) < 0 ? -1 : 0;
    if (fputs("{\"active\":true,\"seed\":", out) < 0 || put_u64(out, state.seed) ||
        fputs(",\"counter\":", out) < 0 || put_u64(out, state.counter) || fputc('}', out) == EOF) return -1;
    return 0;
}
static int json_sink(void *context, const __chelis_random_observer_event *event) {
    FILE *out = (FILE *)context;
    const __chelis_random_observer_frame *frames[16];
    size_t depth = 0;
    for (const __chelis_random_observer_frame *frame = event->saved; frame != NULL; frame = frame->previous) {
        if (depth == 16) return -1;
        frames[depth++] = frame;
    }
    if (fprintf(out, "{\"schema\":\"chelis-native-random-observation-v1\",\"invocation\":") < 0 ||
        put_u64(out, event->invocation) || fputs(",\"sequence\":", out) < 0 ||
        put_u64(out, event->sequence) || fprintf(out, ",\"event\":\"%s\",\"identity\":\"%s\",\"producer\":",
            event_name(event->kind), event->identity == __CHELIS_RANDOM_OBSERVER_FIXED_IDENTITY ? "fixed" :
            event->identity == __CHELIS_RANDOM_OBSERVER_HOST_IDENTITY_UNSUPPORTED ? "unsupported_host" : "invocation") < 0) return -1;
    if (event->producer != NULL) {
        if (fprintf(out, "\"%s\"", event->producer) < 0) return -1;
    } else if (fputs("null", out) < 0) return -1;
    if (fputs(",\"occurrence\":", out) < 0) return -1;
    if (event->has_occurrence ? put_u64(out, event->occurrence) : fputs("null", out) < 0) return -1;
    if (fputs(",\"draw\":", out) < 0) return -1;
    if (event->has_draw ? put_u64(out, event->draw) : fputs("null", out) < 0) return -1;
    if (fputs(",\"scope\":", out) < 0) return -1;
    if (event->has_scope ? put_u64(out, event->scope) : fputs("null", out) < 0) return -1;
    if (fputs(",\"state\":", out) < 0 || put_state(out, event->state) || fputs(",\"saved\":[", out) < 0) return -1;
    for (size_t i = depth; i > 0; --i) {
        if (i != depth && fputc(',', out) == EOF) return -1;
        if (put_state(out, __chelis_random_observer_state_of(frames[i - 1]->state))) return -1;
    }
    if (fputs("],\"used\":", out) < 0) return -1;
    if (event->has_used) {
        if (fputs("{\"seed\":", out) < 0 || put_u64(out, event->used_seed) ||
            fputs(",\"counter\":", out) < 0 || put_u64(out, event->used_counter) || fputc('}', out) == EOF) return -1;
    } else if (fputs("null", out) < 0) return -1;
    if (fputs(",\"continuation\":", out) < 0) return -1;
    if (event->has_continuation) {
        if (fputs("{\"seed\":", out) < 0 || put_u64(out, event->continuation_seed) ||
            fputs(",\"counter\":", out) < 0 || put_u64(out, event->continuation_counter) ||
            fputs(",\"successor\":", out) < 0 || put_u64(out, event->continuation_successor) || fputc('}', out) == EOF) return -1;
    } else if (fputs("null", out) < 0) return -1;
    return fputs("}\n", out) < 0 ? -1 : 0;
}
"#;

fn driver(invocation: u64, sink: &str) -> String {
    format!(
        r#"
{JSON_SINK}
static int selected_sink(void *context, const __chelis_random_observer_event *event) {{
    {sink}
}}
int main(void) {{
    chelis_tensor *x = input(4);
    chelis_tuple *result = __chelis_observed_run(x, selected_sink, stdout, {invocation}ULL);
    chelis_tuple_release(result);
    chelis_tensor_release(x);
    return 0;
}}
"#
    )
}

fn tagged(value: u64) -> Value {
    json!({"u64": value.to_string()})
}

fn state(seed: Option<(u64, u64)>) -> Value {
    match seed {
        Some((seed, counter)) => {
            json!({"active": true, "seed": tagged(seed), "counter": tagged(counter)})
        }
        None => json!({"active": false, "seed": null, "counter": null}),
    }
}

fn row(sequence: u64, event: &str, identity: &str, state_value: Value, saved: Vec<Value>) -> Value {
    json!({
        "schema": "chelis-native-random-observation-v1",
        "invocation": tagged(17),
        "sequence": tagged(sequence),
        "event": event,
        "identity": identity,
        "producer": null,
        "occurrence": null,
        "draw": null,
        "scope": null,
        "state": state_value,
        "saved": saved,
        "used": null,
        "continuation": null
    })
}

fn expected_rows() -> Vec<Value> {
    let inactive = state(None);
    let outer_saved = vec![inactive.clone()];
    let mut rows = vec![
        row(0, "invocation_init", "invocation", inactive.clone(), vec![]),
        row(
            1,
            "host_install",
            "unsupported_host",
            state(Some((42, 0))),
            outer_saved.clone(),
        ),
    ];
    let mut enter = row(
        2,
        "fixed_enter",
        "fixed",
        state(Some((7, 0))),
        vec![inactive.clone(), state(Some((42, 0)))],
    );
    enter["occurrence"] = tagged(0);
    enter["scope"] = tagged(1);
    rows.push(enter);
    let mut forward = row(
        3,
        "forward",
        "fixed",
        state(Some((7, 1))),
        vec![inactive.clone(), state(Some((42, 0)))],
    );
    forward["occurrence"] = tagged(1);
    forward["draw"] = tagged(0);
    forward["scope"] = tagged(1);
    forward["used"] = json!({"seed": tagged(7), "counter": tagged(0)});
    rows.push(forward);
    let mut leave = row(
        4,
        "fixed_leave",
        "fixed",
        state(Some((42, 0))),
        outer_saved.clone(),
    );
    leave["occurrence"] = tagged(2);
    leave["scope"] = tagged(1);
    rows.push(leave);
    let mut replay = row(
        5,
        "replay",
        "fixed",
        state(Some((42, 0))),
        outer_saved.clone(),
    );
    replay["draw"] = tagged(0);
    replay["scope"] = tagged(1);
    replay["used"] = json!({"seed": tagged(7), "counter": tagged(0)});
    rows.push(replay);
    rows.push(row(
        6,
        "host_install",
        "unsupported_host",
        state(Some((99, 0))),
        vec![inactive.clone(), state(Some((42, 0)))],
    ));
    let mut nested = row(
        7,
        "forward",
        "fixed",
        state(Some((99, 1))),
        vec![inactive.clone(), state(Some((42, 0)))],
    );
    nested["occurrence"] = tagged(0);
    nested["draw"] = tagged(0);
    nested["scope"] = tagged(0);
    nested["used"] = json!({"seed": tagged(99), "counter": tagged(0)});
    rows.push(nested);
    rows.push(row(
        8,
        "host_restore",
        "unsupported_host",
        state(Some((42, 0))),
        outer_saved.clone(),
    ));
    let mut following = row(
        9,
        "forward",
        "fixed",
        state(Some((42, 1))),
        outer_saved.clone(),
    );
    following["occurrence"] = tagged(0);
    following["draw"] = tagged(0);
    following["scope"] = tagged(0);
    following["used"] = json!({"seed": tagged(42), "counter": tagged(0)});
    rows.push(following);
    rows.push(row(
        10,
        "host_restore",
        "unsupported_host",
        inactive,
        vec![],
    ));
    for row in &mut rows {
        if row["state"]["active"] == true {
            let seed = row["state"]["seed"].clone();
            let counter = row["state"]["counter"].clone();
            let n = counter["u64"].as_str().unwrap().parse::<u64>().unwrap();
            row["continuation"] =
                json!({"seed": seed, "counter": counter, "successor": tagged(n.wrapping_add(1))});
        }
    }
    for row in &mut rows[2..6] {
        row["producer"] = json!("run__tensor_0__with_rng");
    }
    rows[7]["producer"] = json!("run__tensor_1__with_rng");
    rows[9]["producer"] = json!("run__tensor_2__with_rng");
    rows
}

#[test]
fn nested_host_and_fixed_frames_record_actual_forward_replay_and_restoration() {
    let c = observed_source();
    assert!(c.contains("static chelis_tuple* __chelis_observed_run("));
    let (summary, stdout) =
        ownership_support::run_with_stdout(&c, &driver(17, "return json_sink(context, event);"));
    ownership_support::balanced(&summary);
    let actual = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect::<Vec<Value>>();
    let fixed_producers = actual
        .iter()
        .filter(|row| row["identity"] == "fixed")
        .map(|row| row["producer"].as_str().expect("fixed producer identity"))
        .collect::<Vec<_>>();
    assert_eq!(
        fixed_producers,
        [
            "run__tensor_0__with_rng",
            "run__tensor_0__with_rng",
            "run__tensor_0__with_rng",
            "run__tensor_0__with_rng",
            "run__tensor_1__with_rng",
            "run__tensor_2__with_rng",
        ]
    );
    assert!(
        actual
            .iter()
            .filter(|row| row["identity"] != "fixed")
            .all(|row| row["producer"].is_null())
    );
    assert_eq!(actual, expected_rows());
}

#[test]
fn repeated_observed_calls_restart_sequence_and_keep_invocation_identity() {
    let c = observed_source();
    let driver = format!(
        r#"
#include <pthread.h>
{JSON_SINK}
typedef struct {{ FILE *stream; uint64_t invocation; }} observer_task;
static void *run_observed(void *raw) {{
    observer_task *task = (observer_task *)raw;
    chelis_tensor *x = input(4);
    chelis_tuple *result = __chelis_observed_run(x, json_sink, task->stream, task->invocation);
    chelis_tuple_release(result);
    chelis_tensor_release(x);
    return NULL;
}}
int main(void) {{
    observer_task tasks[2] = {{{{tmpfile(), 20ULL}}, {{tmpfile(), 21ULL}}}};
    pthread_t threads[2];
    for (int i = 0; i < 2; ++i) {{
        assert(tasks[i].stream != NULL);
        assert(pthread_create(&threads[i], NULL, run_observed, &tasks[i]) == 0);
    }}
    for (int i = 0; i < 2; ++i) {{
        assert(pthread_join(threads[i], NULL) == 0);
        rewind(tasks[i].stream);
        for (int byte = fgetc(tasks[i].stream); byte != EOF; byte = fgetc(tasks[i].stream)) {{
            assert(fputc(byte, stdout) != EOF);
        }}
        assert(fclose(tasks[i].stream) == 0);
    }}
    return 0;
}}
"#
    );
    let (summary, stdout) = ownership_support::run_with_stdout(&c, &driver);
    ownership_support::balanced(&summary);
    let rows = stdout
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 22);
    assert!(rows[..11].iter().all(|row| row["invocation"] == tagged(20)));
    assert!(rows[11..].iter().all(|row| row["invocation"] == tagged(21)));
    assert_eq!(rows[0]["sequence"], tagged(0));
    assert_eq!(rows[11]["sequence"], tagged(0));
}

#[test]
fn feature_on_ordinary_public_call_emits_no_observation() {
    let c = observed_source();
    assert!(c.contains("chelis_tuple* run(chelis_tensor* x)"));
    assert!(c.contains("static chelis_tuple* __chelis_observed_run("));
    let driver = r#"
int main(void) {
    chelis_tensor *x = input(4);
    chelis_tuple *result = run(x);
    chelis_tuple_release(result);
    chelis_tensor_release(x);
    return 0;
}
"#;
    let (summary, stdout) = ownership_support::run_with_stdout(&c, driver);
    ownership_support::balanced(&summary);
    assert!(
        stdout.is_empty(),
        "ordinary public call emitted observer data"
    );
}

#[test]
fn independent_expectation_rejects_missing_saved_preincrement_and_replay_advance_mutants() {
    let expected = expected_rows();
    let mut missing_saved = expected.clone();
    missing_saved[3]["saved"].as_array_mut().unwrap().pop();
    assert_ne!(missing_saved, expected);
    let mut preincrement = expected.clone();
    preincrement[3]["state"]["counter"] = tagged(0);
    assert_ne!(preincrement, expected);
    let mut replay_advance = expected.clone();
    replay_advance[5]["state"]["counter"] = tagged(1);
    assert_ne!(replay_advance, expected);
}

#[test]
fn sink_error_is_not_silent_success() {
    let c = observed_source();
    ownership_support::run_expect_failure(
        &c,
        &driver(17, "(void)context; (void)event; return -1;"),
    );
}

#[test]
fn rejection_only_max_counter_mutant_wraps_native_copy_and_fails_nonwrapping_check() {
    let c = observed_source();
    let driver = format!(
        r#"
{JSON_SINK}
int main(void) {{
    chelis_tensor *x = input(1);
    __chelis_random_observer observer = {{json_sink, stdout, {{31ULL}}, {{0ULL}}, NULL}};
    chelis_rng_state rejection_only = {{5ULL, UINT64_MAX, 1}};
    __chelis_random_observer_record(
        &observer,
        __CHELIS_RANDOM_OBSERVER_FORWARD,
        __CHELIS_RANDOM_OBSERVER_FIXED_IDENTITY,
        "rejection-only-mutant",
        1, 0ULL, 1, 0ULL, 1, 0ULL,
        rejection_only,
        1, rejection_only.seed, rejection_only.counter
    );
    chelis_tensor_release(x);
    return 0;
}}
"#
    );
    let (summary, stdout) = ownership_support::run_with_stdout(&c, &driver);
    ownership_support::balanced(&summary);
    let row: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(row["continuation"]["counter"], tagged(u64::MAX));
    assert_eq!(row["continuation"]["successor"], tagged(0));
    let counter = row["continuation"]["counter"]["u64"]
        .as_str()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    assert!(
        counter.checked_add(1).is_none(),
        "the independent nonwrapping admission must reject this driver-only mutant"
    );
}
