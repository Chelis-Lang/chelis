//! chelis#2781: a loop's result stays live for the enclosing scope when a
//! later loop at the same depth passes it to a named definition.
//!
//! A `map`, `filter`, `scan`, `fold` or `flat_map` mints its exit-block
//! parameter while the loop *body* scope is open, so the lowering recorded the
//! body's depth as that owner's home scope. The owner's home scope is the
//! enclosing one - that is where `bind` registers it - and a later loop body at
//! the same depth then read the recorded depth as "local to this body" and
//! *moved* the value on a by-value call instead of copying it. [04-LIN-5]
//! requires the copy: "a path that preserves another live use first creates a
//! copy", and the root manifest is another live use. Without it the owner
//! reached the consuming loop's header live on the entry path and dead on the
//! back edge, and host ownership verification rejected the program with
//! `block bN in \`roots\` is reached with inconsistent live owners`, while
//! `chelis check` scored it 1.0 and `chelis eval` ran it correctly.
//!
//! The corpus crosses every list-producing loop form with every way a later
//! loop reaches a named definition, plus the controls that already built. The
//! oracle: the compiled program runs against the `ownership-ledger` runtime,
//! every allocation must be finalized with no live owner left, and stdout must
//! equal the in-process evaluator's rendering of the same roots. Every case
//! runs before the test reports, so one failing shape cannot hide another.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The two named definitions every consumer reaches, one consuming the list
/// through `index` and one through `len`.
const PRELUDE: &str = "def at2(xs: List[f64], j: i64) -> f64 = index(xs, j)\n\
                       def n2(xs: List[f64]) -> i64 = len(xs)\n";

/// One way to bind the top-level list `hs`. Every form but `literal` is a loop
/// whose exit-block parameter is minted inside the body scope.
struct Producer {
    name: &'static str,
    binding: &'static str,
    /// Whether this form mints its result as a loop exit-block parameter. The
    /// `literal` control does not, and never failed.
    loop_result: bool,
}

const PRODUCERS: &[Producer] = &[
    Producer {
        name: "map",
        binding: "hs = map(fn (x) -> x, [1.0f64, 2.0f64])\n",
        loop_result: true,
    },
    Producer {
        name: "filter",
        binding: "hs = filter(fn (x) -> gt(x, 0.5f64), [1.0f64, 2.0f64])\n",
        loop_result: true,
    },
    Producer {
        name: "scan",
        binding: "hs = scan(fn (a, x) -> add(a, x), 0.0f64, [1.0f64, 2.0f64])\n",
        loop_result: true,
    },
    Producer {
        name: "fold",
        binding: "hs = fold(fn (a, x) -> append(a, x), [], [1.0f64, 2.0f64])\n",
        loop_result: true,
    },
    Producer {
        name: "flat_map",
        binding: "hs = flat_map(fn (x) -> [x], [1.0f64, 2.0f64])\n",
        loop_result: true,
    },
    Producer {
        name: "literal",
        binding: "hs = [1.0f64, 2.0f64]\n",
        loop_result: false,
    },
];

/// One way the program reads `hs` again after the producer.
struct Consumer {
    name: &'static str,
    body: &'static str,
    /// Whether the call to the named definition stands in the callback's tail
    /// position at the loop body's own scope depth. Only those cases could
    /// reach the defect; the rest are locks on the neighbouring lowering.
    tail_at_body_depth: bool,
}

const CONSUMERS: &[Consumer] = &[
    Consumer {
        name: "map_tail",
        body: "p = map(fn (j) -> at2(hs, j), range(0i64, 2i64))\n",
        tail_at_body_depth: true,
    },
    Consumer {
        name: "map_len_tail",
        body: "p = map(fn (j) -> n2(hs), range(0i64, 2i64))\n",
        tail_at_body_depth: true,
    },
    Consumer {
        name: "fold_tail",
        body: "p = fold(fn (a, j) -> at2(hs, j), 0.0f64, range(0i64, 2i64))\n",
        tail_at_body_depth: true,
    },
    Consumer {
        name: "scan_tail",
        body: "p = scan(fn (a, j) -> at2(hs, j), 0.0f64, range(0i64, 2i64))\n",
        tail_at_body_depth: true,
    },
    // `filter` runs its predicate one scope deeper than the item so the step
    // can keep the item (chelis#2577), so the call is not at the body's depth.
    Consumer {
        name: "filter_tail",
        body: "p = filter(fn (j) -> gt(at2(hs, j), 0.5f64), range(0i64, 2i64))\n",
        tail_at_body_depth: false,
    },
    // A list-literal element is not a tail use.
    Consumer {
        name: "flat_map_tail",
        body: "p = flat_map(fn (j) -> [at2(hs, j)], range(0i64, 2i64))\n",
        tail_at_body_depth: false,
    },
    // An operand of an enclosing builtin is not a tail use.
    Consumer {
        name: "map_nested",
        body: "p = map(fn (j) -> add(at2(hs, j), 1.0f64), range(0i64, 2i64))\n",
        tail_at_body_depth: false,
    },
    // No second loop at all: the issue's `direct call` variant.
    Consumer {
        name: "direct",
        body: "p = at2(hs, 1i64)\n",
        tail_at_body_depth: false,
    },
];

fn instantiate(producer: &Producer, consumer: &Consumer) -> String {
    format!("{}{PRELUDE}{}", producer.binding, consumer.body)
}

/// The evaluator's rendering of every root, in the compiled driver's format.
fn evaluated(source: &str) -> String {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn check(name: &str, source: &str) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(source);
        let generated = ownership_support::emit(source, name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let head: String = message.chars().take(600).collect();
        format!("{name}:\n{source}  -> {head}")
    })
}

/// Run every consumer over one producer and report every failing case.
fn every_consumer_of(producer_name: &str) {
    let producer = PRODUCERS
        .iter()
        .find(|producer| producer.name == producer_name)
        .expect("declared producer");
    let failures: Vec<String> = CONSUMERS
        .iter()
        .filter_map(|consumer| {
            let name = format!("{}_producer_{}", consumer.name, producer.name);
            check(&name, &instantiate(producer, consumer)).err()
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} `{}` cases failed:\n\n{}",
        failures.len(),
        CONSUMERS.len(),
        producer.name,
        failures.join("\n\n")
    );
}

// REGRESSION TESTS, one per producer so the producers run in parallel. On
// `35a2b7101`, the base of the fix, every case pairing a loop producer with a
// consumer whose call sits at the body's depth failed to build with `block b4
// in `roots` is reached with inconsistent live owners`: 20 of the 48 cases.
// The other 28 built and balanced there and lock the neighbouring lowering.

#[test]
fn a_mapped_list_survives_a_later_loop() {
    every_consumer_of("map");
}

#[test]
fn a_filtered_list_survives_a_later_loop() {
    every_consumer_of("filter");
}

#[test]
fn a_scanned_list_survives_a_later_loop() {
    every_consumer_of("scan");
}

#[test]
fn a_folded_list_survives_a_later_loop() {
    every_consumer_of("fold");
}

#[test]
fn a_flat_mapped_list_survives_a_later_loop() {
    every_consumer_of("flat_map");
}

#[test]
fn a_literal_list_survives_a_later_loop() {
    every_consumer_of("literal");
}

/// Every declared producer has a test above, so a producer added to the table
/// cannot go unrun, and the table keeps at least one case that reproduced.
#[test]
fn every_producer_has_a_test_and_the_table_keeps_its_regression_cases() {
    let source = include_str!("issue_2781_loop_result_captured_by_a_loop.rs");
    for producer in PRODUCERS {
        assert!(
            source.contains(&format!("every_consumer_of(\"{}\");", producer.name)),
            "producer `{}` has no test",
            producer.name
        );
    }
    let reproducing = PRODUCERS.iter().filter(|p| p.loop_result).count()
        * CONSUMERS.iter().filter(|c| c.tail_at_body_depth).count();
    assert_eq!(
        reproducing, 20,
        "the corpus no longer holds the 20 cases that failed on the base commit"
    );
}
