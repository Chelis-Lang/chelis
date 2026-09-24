//! Feature-only compile/run observations of actual generated-C Random locals.
#![cfg(feature = "native-random-observer")]

mod ownership_support;

use serde_json::{Value, json};

const OBSERVED_SOURCE: &str = r#"
def fixed_loss(x: tensor[4, f32]) -> f32 = with seed(7i64) {
  tensor_to_scalar(sum(dropout(x, 0.5f32), 0))
}

def run(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32], tensor[4, f32]) = with seed(42i64) {
  replayed = grad(fixed_loss)(x)
  nested = with seed(99i64) { dropout(x, 0.5f32) }
  following = dropout(x, 0.5f32)
  (replayed, nested, following)
}
"#;

fn observed_source() -> ownership_support::GeneratedProgram {
    ownership_support::emit_selected(OBSERVED_SOURCE, "run")
}

#[test]
fn host_source_identity_qualifies_nested_and_following_helper_occurrences() {
    let source = observed_source();
    let driver = driver(
        &source,
        17,
        r#"
    if (event->kind != __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT) {
        assert(event->source != NULL);
        assert(event->source_certified);
        if (event->identity == __CHELIS_RANDOM_OBSERVER_FIXED_IDENTITY) {
            assert(event->calls != NULL);
            assert(event->has_full_source);
            assert(event->calls->source == event->source);
        }
    }
    return json_sink(context, event);
"#,
    );
    let (ledger, stdout) = ownership_support::run_with_stdout(&source, &driver);
    ownership_support::balanced(&ledger);
    let rows: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows, expected_rows(source.symbol("run")));
}

// Retained compiler metadata is the association authority, never a C label or
// a guessed Resource offset. This is deliberately not a second RNG machine.
fn with_source_metadata(source: &str) -> (ownership_support::GeneratedProgram, Value) {
    use chelis_compiler_api::compiler::compile_for_execution_with_observer;
    use chelis_compiler_api::emission_observer::SelectedEmission;
    use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
    use chelis_ir::host::ConcreteHostExprKind as E;
    let mut metadata = serde_json::Map::new();
    let artifact = compile_for_execution_with_observer(CompileRequest {
        source_kind: SourceKind::Surf, source: source.into(), target: CompileTarget::C, entry_name: Some("run".into()),
    }, &mut |observation| {
        let SelectedEmission::Host(host) = observation.selected else { panic!("host fixture"); };
        for site in host.source_expressions() {
            let (kind, target, seed) = match &site.expression().kind {
                E::WithSeed { seed, .. } => { let E::Int(seed) = seed.kind else { continue }; (1, 0, seed as u64) },
                E::Call { .. } => (2, site.direct_callee_word().unwrap_or(0), 0),
                E::TensorCall { helper, .. } => (3, *helper, 0),
                _ => continue,
            };
            // A helper's retained events are its draw keys in node order: the
            // draw index is the occurrence and full-source word, and the
            // scope is 0 for the inherited stream or `instance + 1`.
            let mut full = Vec::new();
            if kind == 3 {
                let dag = host
                    .function(site.unit_word() - 1)
                    .unwrap()
                    .tensor_helper(target)
                    .unwrap()
                    .dag();
                let keys = dag.nodes().iter().filter_map(|node| match node.op {
                    chelis_ir::dag::RiscOp::DrawKey { handler, .. } => Some(handler),
                    _ => None,
                });
                for (draw, handler) in keys.enumerate() {
                    let scope = match handler {
                        chelis_ir::dag::RandomHandler::Inherited => 0,
                        chelis_ir::dag::RandomHandler::Scoped { instance } => instance as usize + 1,
                    };
                    full.push(json!([
                        "forward",
                        draw.to_string(),
                        scope.to_string(),
                        draw.to_string(),
                        draw.to_string()
                    ]));
                }
            }
            metadata.insert(site.site_word().to_string(), json!({
                "descriptor": [site.unit_word().to_string(), site.site_word().to_string(), kind.to_string(), target.to_string()],
                "seed": seed.to_string(), "parent": site.seed_parent_word().map(|s| s.to_string()), "full": full,
            }));
        }
    }).unwrap();
    let c = artifact
        .compile_result
        .files
        .iter()
        .find(|f| f.path.ends_with(".c"))
        .unwrap()
        .contents
        .clone();
    let header = artifact
        .compile_result
        .files
        .iter()
        .find(|f| f.path.ends_with(".h"))
        .unwrap()
        .contents
        .clone();
    (
        ownership_support::GeneratedProgram::new(c, header),
        Value::Object(metadata),
    )
}

const SOURCE_SINK: &str = r#"
static int put_source(FILE *out, const __chelis_random_source_site *source) {
    if (source == NULL) return fputs("null", out) < 0 ? -1 : 0;
    return fprintf(out, "[\"%" PRIu64 "\",\"%" PRIu64 "\",\"%d\",\"%" PRIu64 "\"]", source->unit.value, source->site.value, source->kind, source->target.value) < 0 ? -1 : 0;
}
static int source_sink(void *context, const __chelis_random_observer_event *event) {
    FILE *out = context;
    if (json_sink(context, event)) return -1;
    if (fprintf(out, "{\"certified\":%s,\"source\":", event->source_certified ? "true" : "false") < 0 || put_source(out, event->source) || fputs(",\"calls\":[", out) < 0) return -1;
    int first = 1;
    for (const __chelis_random_call_frame *frame = event->calls; frame != NULL; frame = frame->previous) {
        if (!first && fputc(',', out) == EOF) return -1;
        first = 0;
        if (put_source(out, frame->source)) return -1;
    }
    if (fputs("],\"host\":", out) < 0 || put_source(out, event->host_scope == NULL ? NULL : event->host_scope->source) || fputs(",\"full\":", out) < 0) return -1;
    if (event->has_full_source ? put_u64(out, event->full_source) : fputs("null", out) < 0) return -1;
    return fprintf(out, ",\"inherited\":%s}\n", event->scope_is_inherited ? "true" : "false") < 0 ? -1 : 0;
}
"#;

fn source_rows(c: &ownership_support::GeneratedProgram, repeat: bool) -> Vec<(Value, Value)> {
    let observed_entry = selected_observed_entry(c, "run");
    source_rows_with_entry(c, &observed_entry, repeat)
}

fn source_rows_with_entry(
    c: &ownership_support::GeneratedProgram,
    observed_entry: &str,
    repeat: bool,
) -> Vec<(Value, Value)> {
    assert!(
        c.contains(&format!("static chelis_tuple* {observed_entry}(")),
        "missing selected observed entry {observed_entry}"
    );
    let driver = format!(
        r#"
{JSON_SINK}
{SOURCE_SINK}
int main(void) {{
    chelis_tensor *x = input(4);
    for (int i = 0; i < {}; ++i) {{
        chelis_tuple *result = {observed_entry}(x, source_sink, stdout, 17ULL + (uint64_t)i);
        chelis_tuple_release(result);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#,
        if repeat { 2 } else { 1 }
    );
    let (ledger, stdout) = ownership_support::run_with_stdout(c, &driver);
    ownership_support::balanced(&ledger);
    let values: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(values.len() % 2, 0);
    values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| (v[0].clone(), v[1].clone()))
        .collect()
}

fn selected_observed_entry(c: &ownership_support::GeneratedProgram, entry: &str) -> String {
    let observed_entry = format!("__chelis_observed_{}", c.symbol(entry));
    assert!(
        c.contains(&format!("static chelis_tuple* {observed_entry}(")),
        "missing selected observed entry {observed_entry}"
    );
    observed_entry
}

fn associations_match(rows: &[(Value, Value)], metadata: &Value) -> bool {
    rows.iter().all(|(row, identity)| {
        if row["event"] == "invocation_init" {
            return identity["source"].is_null() && !identity["certified"].as_bool().unwrap();
        }
        if identity["certified"] != true {
            return false;
        }
        let Some(id) = identity["source"][1].as_str() else {
            return false;
        };
        let Some(site) = metadata.get(id) else {
            return false;
        };
        if site["descriptor"] != identity["source"] {
            return false;
        }
        let calls = identity["calls"].as_array().unwrap();
        if calls.iter().any(|call| {
            call[1]
                .as_str()
                .and_then(|id| metadata.get(id))
                .is_none_or(|s| s["descriptor"] != *call)
        }) {
            return false;
        }
        if calls
            .windows(2)
            .any(|pair| pair[1][2] != "2" || pair[1][3] != pair[0][0])
        {
            return false;
        }
        if row["identity"] == "fixed" {
            if calls.first() != Some(&identity["source"]) || identity["source"][2] != "3" {
                return false;
            }
            let event = if row["event"] == "replay" {
                "forward"
            } else {
                row["event"].as_str().unwrap()
            };
            let matching: Vec<_> = site["full"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|entry| {
                    entry[0] == event
                        && entry[1] == row["draw"]["u64"]
                        && entry[2] == row["scope"]["u64"]
                })
                .collect();
            if matching.len() != 1 || matching[0][3] != identity["full"]["u64"] {
                return false;
            }
            if row["event"] == "replay" {
                if !row["occurrence"].is_null() {
                    return false;
                }
            } else if matching[0][4] != row["occurrence"]["u64"] {
                return false;
            }
            if !site["parent"].is_null() && identity["host"][1] != site["parent"] {
                return false;
            }
        } else {
            if identity["source"][2] != "1" {
                return false;
            }
            if row["event"] == "host_install"
                && (row["state"]["seed"]["u64"] != site["seed"]
                    || identity["host"] != identity["source"])
            {
                return false;
            }
            if row["event"] == "host_restore"
                && !site["parent"].is_null()
                && identity["host"][1] != site["parent"]
            {
                return false;
            }
        }
        true
    })
}

#[test]
fn retained_full_source_ids_survive_resource_offsets_and_reject_association_mutants() {
    let input = OBSERVED_SOURCE
        .replace("with seed(7i64)", "with device(\"cpu\") { with seed(7i64)")
        .replace("\n}\n\ndef run", "\n}}\n\ndef run");
    let (c, metadata) = with_source_metadata(&input);
    let observed_entry = selected_observed_entry(&c, "run");
    let rows = source_rows_with_entry(&c, &observed_entry, true);
    assert_eq!(rows.len(), 18);
    assert!(associations_match(&rows, &metadata));
    // A Resource handler around the helper's draw leaves its draw word alone.
    assert_eq!(rows[2].1["full"], tagged(0));
    assert_eq!(rows[2].1["inherited"], false);
    assert_eq!(rows[5].1["inherited"], true);
    for (label, index, field, replacement) in [
        ("wrong full source", 2, "full", tagged(1)),
        ("missing call", 2, "calls", json!([])),
        (
            "wrong host occurrence",
            1,
            "source",
            rows[4].1["source"].clone(),
        ),
        (
            "wrong helper with equal local draw",
            5,
            "source",
            rows[7].1["source"].clone(),
        ),
        (
            "wrong inherited host scope",
            5,
            "host",
            rows[7].1["host"].clone(),
        ),
    ] {
        let mut mutant = rows.clone();
        mutant[index].1[field] = replacement;
        assert!(!associations_match(&mutant, &metadata), "{label}");
    }
    assert_eq!(rows[0].0["invocation"], tagged(17));
    assert_eq!(rows[9].0["invocation"], tagged(18));
    assert_eq!(rows[9].0["sequence"], tagged(0));
    for (field, replacement) in [
        ("occurrence", tagged(99)),
        ("draw", Value::Null),
        ("scope", tagged(99)),
    ] {
        let mut mutant = rows.clone();
        mutant[2].0[field] = replacement;
        assert!(
            !associations_match(&mutant, &metadata),
            "wrong/missing legacy {field}"
        );
    }

    // Both following helpers have local occurrence/draw 0. The actual callee
    // must still reject a different helper slot as its source authority.
    let helper = &rows[5].1["source"];
    let guard = format!(
        "call->source->unit.value == {}ULL && call->source->target.value == {}ULL",
        helper[0].as_str().unwrap(),
        helper[3].as_str().unwrap()
    );
    assert_eq!(c.matches(&guard).count(), 1);
    let mutant = c.replace(
        &guard,
        &guard.replace(
            &format!("target.value == {}ULL", helper[3].as_str().unwrap()),
            "target.value == 999ULL",
        ),
    );
    ownership_support::run_expect_failure(
        &c.with_source(mutant),
        &driver_for(
            &observed_entry,
            17,
            "assert(event->kind == __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT || event->source_certified); return json_sink(context, event);",
        ),
    );
}

#[test]
fn repeated_direct_callees_keep_linked_identity_and_restore_the_call_stack() {
    let input = r#"
def fixed_loss(x: tensor[4, f32]) -> f32 = with seed(7i64) { tensor_to_scalar(sum(dropout(x, 0.5f32), 0)) }
def leaf(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = {
 replayed = grad(fixed_loss)(x)
 next = dropout(x, 0.5f32)
 (replayed, next)
}
def middle(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = leaf(x)
def run(x: tensor[4, f32]) -> ((tensor[4, f32], tensor[4, f32]), (tensor[4, f32], tensor[4, f32]), tensor[4, f32]) = with seed(42i64) {
 ax = copy(x)
 bx = copy(x)
 a = middle(ax)
 b = with seed(99i64) { middle(bx) }
 c = dropout(x, 0.5f32)
 (a, b, c)
}
"#;
    let (c, metadata) = with_source_metadata(input);
    let rows = source_rows(&c, true);
    assert!(associations_match(&rows, &metadata), "{rows:#?}");
    let forwards: Vec<_> = rows
        .iter()
        .filter(|(r, _)| r["event"] == "forward" && r["invocation"] == tagged(17))
        .collect();
    assert_eq!(forwards.len(), 5);
    for ((row, _), (seed, counter)) in
        forwards
            .iter()
            .zip([(7, 0), (42, 0), (7, 0), (99, 0), (42, 1)])
    {
        assert_eq!(
            row["used"],
            json!({"seed": tagged(seed), "counter": tagged(counter)})
        );
    }
    assert_eq!(
        forwards[0].1["source"], forwards[2].1["source"],
        "same actual helper"
    );
    assert_eq!(forwards[0].1["calls"].as_array().unwrap().len(), 3);
    assert_ne!(
        forwards[0].1["calls"][2], forwards[2].1["calls"][2],
        "distinct source call sites"
    );
    assert_eq!(
        forwards[4].1["calls"].as_array().unwrap().len(),
        1,
        "both direct-call frames restored"
    );
    assert_eq!(
        forwards[4].0["used"],
        json!({"seed": tagged(42), "counter": tagged(1)})
    );
    assert_eq!(forwards[4].0["state"], state(Some((42, 2))));
    assert_eq!(rows.last().unwrap().0["state"], state(None));
    let callee = forwards[0].1["calls"][1][3].as_str().unwrap();
    let guard = format!("__chelis_random_push_function(__chelis_observer, {callee}ULL,");
    assert_eq!(c.matches(&guard).count(), 1);
    let wrong_callee = c.replace(
        &guard,
        "__chelis_random_push_function(__chelis_observer, 999ULL,",
    );
    ownership_support::run_expect_failure(
        &c.with_source(wrong_callee.clone()),
        &driver(
            &c.with_source(wrong_callee),
            17,
            "assert(event->kind == __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT || event->source_certified); return json_sink(context, event);",
        ),
    );
}

#[test]
fn unsupported_argument_effects_cannot_certify_descendant_calls() {
    let input = r#"
def loss(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(x, 0))
def leaf(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = (dropout(x, 0.5f32), grad(loss)(x))
def run(x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = with seed(42i64) {
 leaf(with seed(99i64) { dropout(x, 0.5f32) })
}
"#;
    let (c, _) = with_source_metadata(input);
    let rows = source_rows(&c, false);
    assert!(
        rows.iter()
            .filter(|(row, _)| row["event"] == "forward")
            .count()
            >= 2
    );
    assert!(
        rows.iter()
            .all(|(_, identity)| identity["certified"] == false)
    );
    assert_eq!(rows.last().unwrap().0["state"], state(None));
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
    let public_entry = source.symbol("run");
    let observed_entry = format!("__chelis_observed_{public_entry}");
    assert!(source.contains(&format!("float {public_entry}(float x)")));
    assert!(source.contains(&format!("static float {observed_entry}(")));
    assert!(!source.contains("float run(float x)"));
    assert!(!source.contains("static float __chelis_observed_run("));
    let driver = r#"
static int count_event(void *context, const __chelis_random_observer_event *event) {
    assert(event->kind == __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT);
    ++*(int *)context;
    return 0;
}
int main(void) {
    chelis_tensor *receipt = input(1);
    int count = 0;
    assert(PUBLIC_ENTRY(2.0f) == 10.0f);
    assert(OBSERVED_ENTRY(2.0f, count_event, &count, 11ULL) == 10.0f);
    assert(count == 1);
    chelis_tensor_release(receipt);
    return 0;
}
"#
    .replace("PUBLIC_ENTRY", public_entry)
    .replace("OBSERVED_ENTRY", &observed_entry);
    ownership_support::balanced(&ownership_support::run(&source, &driver));
}

const JSON_SINK: &str = r#"
static const char *event_name(__chelis_random_observer_event_kind kind) {
    switch (kind) {
        case __CHELIS_RANDOM_OBSERVER_INVOCATION_INIT: return "invocation_init";
        case __CHELIS_RANDOM_OBSERVER_HOST_INSTALL: return "host_install";
        case __CHELIS_RANDOM_OBSERVER_HOST_RESTORE: return "host_restore";
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

fn driver(c: &ownership_support::GeneratedProgram, invocation: u64, sink: &str) -> String {
    let observed_entry = selected_observed_entry(c, "run");
    driver_for(&observed_entry, invocation, sink)
}

fn driver_for(observed_entry: &str, invocation: u64, sink: &str) -> String {
    format!(
        r#"
{JSON_SINK}
static int selected_sink(void *context, const __chelis_random_observer_event *event) {{
    {sink}
}}
int main(void) {{
    chelis_tensor *x = input(4);
    chelis_tuple *result = {observed_entry}(x, selected_sink, stdout, {invocation}ULL);
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

fn expected_rows(run_symbol: &str) -> Vec<Value> {
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
    let mut forward = row(
        2,
        "forward",
        "fixed",
        state(Some((7, 1))),
        outer_saved.clone(),
    );
    forward["occurrence"] = tagged(0);
    forward["draw"] = tagged(0);
    forward["scope"] = tagged(1);
    forward["used"] = json!({"seed": tagged(7), "counter": tagged(0)});
    rows.push(forward);
    let mut replay = row(
        3,
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
        4,
        "host_install",
        "unsupported_host",
        state(Some((99, 0))),
        vec![inactive.clone(), state(Some((42, 0)))],
    ));
    let mut nested = row(
        5,
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
        6,
        "host_restore",
        "unsupported_host",
        state(Some((42, 0))),
        outer_saved.clone(),
    ));
    let mut following = row(
        7,
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
    rows.push(row(8, "host_restore", "unsupported_host", inactive, vec![]));
    for row in &mut rows {
        if row["state"]["active"] == true {
            let seed = row["state"]["seed"].clone();
            let counter = row["state"]["counter"].clone();
            let n = counter["u64"].as_str().unwrap().parse::<u64>().unwrap();
            row["continuation"] =
                json!({"seed": seed, "counter": counter, "successor": tagged(n.wrapping_add(1))});
        }
    }
    for row in &mut rows[2..4] {
        row["producer"] = json!(format!("{run_symbol}__tensor_0__with_rng"));
    }
    rows[5]["producer"] = json!(format!("{run_symbol}__tensor_1__with_rng"));
    rows[7]["producer"] = json!(format!("{run_symbol}__tensor_2__with_rng"));
    rows
}

#[test]
fn nested_host_and_fixed_frames_record_actual_forward_replay_and_restoration() {
    let c = observed_source();
    let observed_entry = selected_observed_entry(&c, "run");
    assert!(c.contains(&format!("static chelis_tuple* {observed_entry}(")));
    assert!(!c.contains("static chelis_tuple* __chelis_observed_run("));
    let (summary, stdout) = ownership_support::run_with_stdout(
        &c,
        &driver(&c, 17, "return json_sink(context, event);"),
    );
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
    let run_symbol = c.symbol("run");
    assert_eq!(
        fixed_producers,
        [
            format!("{run_symbol}__tensor_0__with_rng"),
            format!("{run_symbol}__tensor_0__with_rng"),
            format!("{run_symbol}__tensor_1__with_rng"),
            format!("{run_symbol}__tensor_2__with_rng"),
        ]
    );
    assert!(
        fixed_producers
            .iter()
            .all(|producer| !producer.starts_with("run__tensor_"))
    );
    assert!(
        actual
            .iter()
            .filter(|row| row["identity"] != "fixed")
            .all(|row| row["producer"].is_null())
    );
    assert_eq!(actual, expected_rows(run_symbol));
}

#[test]
fn repeated_observed_calls_restart_sequence_and_keep_invocation_identity() {
    let c = observed_source();
    let observed_entry = selected_observed_entry(&c, "run");
    let driver = format!(
        r#"
#include <pthread.h>
{JSON_SINK}
typedef struct {{ FILE *stream; uint64_t invocation; }} observer_task;
static void *run_observed(void *raw) {{
    observer_task *task = (observer_task *)raw;
    chelis_tensor *x = input(4);
    chelis_tuple *result = {observed_entry}(x, json_sink, task->stream, task->invocation);
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
    assert_eq!(rows.len(), 18);
    assert!(rows[..9].iter().all(|row| row["invocation"] == tagged(20)));
    assert!(rows[9..].iter().all(|row| row["invocation"] == tagged(21)));
    assert_eq!(rows[0]["sequence"], tagged(0));
    assert_eq!(rows[9]["sequence"], tagged(0));
}

#[test]
fn feature_on_ordinary_public_call_emits_no_observation() {
    let c = observed_source();
    let public_entry = c.symbol("run");
    let observed_entry = selected_observed_entry(&c, "run");
    assert!(c.contains(&format!("chelis_tuple* {public_entry}(chelis_tensor* x)")));
    assert!(c.contains(&format!("static chelis_tuple* {observed_entry}(")));
    assert!(!c.contains("chelis_tuple* run(chelis_tensor* x)"));
    assert!(!c.contains("static chelis_tuple* __chelis_observed_run("));
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input(4);
    chelis_tuple *result = {public_entry}(x);
    chelis_tuple_release(result);
    chelis_tensor_release(x);
    return 0;
}}
"#
    );
    let (summary, stdout) = ownership_support::run_with_stdout(&c, &driver);
    ownership_support::balanced(&summary);
    assert!(
        stdout.is_empty(),
        "ordinary public call emitted observer data"
    );
}

#[test]
fn independent_expectation_rejects_missing_saved_preincrement_and_replay_advance_mutants() {
    let c = observed_source();
    let expected = expected_rows(c.symbol("run"));
    let mut missing_saved = expected.clone();
    missing_saved[2]["saved"].as_array_mut().unwrap().pop();
    assert_ne!(missing_saved, expected);
    let mut preincrement = expected.clone();
    preincrement[2]["state"]["counter"] = tagged(0);
    assert_ne!(preincrement, expected);
    let mut replay_advance = expected.clone();
    replay_advance[3]["state"]["counter"] = tagged(1);
    assert_ne!(replay_advance, expected);
}

#[test]
fn sink_error_is_not_silent_success() {
    let c = observed_source();
    ownership_support::run_expect_failure(
        &c,
        &driver(&c, 17, "(void)context; (void)event; return -1;"),
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
