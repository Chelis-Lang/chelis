//! chelis#3031: compiled C sorts exactly as `chelis eval` does.
//!
//! The runtime's lane sort is a stable sort under the order
//! `should_swap_for_stable_sort` defines: NaNs after every other value, the
//! IEEE order otherwise with signed zeros equal, and exact order on integers.
//! That order is a strict weak order over every input, so the sort never
//! sees an inconsistent comparator. These programs mix NaNs, signed zeros,
//! infinities, duplicates and integer extremes, and print the sorted values
//! and source indices, which must equal what `chelis eval` prints.
//!
//! Oracle: each program compiles, runs against the `ownership-ledger`
//! runtime, and prints what `chelis eval` prints. The ledger's balance is not
//! asserted: every sort result leaks in compiled C (chelis#3032).

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

const CASES: &[(&str, &str)] = &[
    (
        "floats_with_nans_signed_zeros_and_duplicates",
        "module Demo.Main\ndef nan(seed: i64) -> f64 = div(0.0f64, 0.0f64)\ndef infinity(seed: i64) -> f64 = div(1.0f64, 0.0f64)\ndef run(seed: i64) -> (List[f64], List[i64]) = {\n  (values, order) = sort(to_tensor([nan(0i64), -0.0f64, 0.0f64, 1.0f64, neg(nan(0i64)), neg(infinity(0i64)), 1.0f64, -0.0f64, infinity(0i64), nan(0i64)]), 0i32)\n  (to_list(values), to_list(order))\n}\nresult = run(0i64)\n",
    ),
    (
        "integer_extremes_and_duplicates",
        "module Demo.Main\ndef run(seed: i64) -> (List[i64], List[i64]) = {\n  smallest = sub(-9223372036854775807i64, 1i64)\n  (values, order) = sort(to_tensor([9223372036854775807i64, smallest, 0i64, 9223372036854775807i64, -1i64, smallest, 0i64]), 0i32)\n  (to_list(values), to_list(order))\n}\nresult = run(0i64)\n",
    ),
    // Integers above 2^53 compare exactly, not through a lossy f64 reading.
    (
        "integers_beyond_two_to_the_53",
        "module Demo.Main\ndef run(seed: i64) -> (List[i64], List[i64]) = {\n  (values, order) = sort(to_tensor([9223372036854775807i64, 9223372036854775806i64, 9007199254740993i64, 9007199254740992i64, 9223372036854775807i64]), 0i32)\n  (to_list(values), to_list(order))\n}\nresult = run(0i64)\n",
    ),
    (
        "narrow_integers",
        "module Demo.Main\ndef run(seed: i64) -> (List[i8], List[i64]) = {\n  (values, order) = sort(to_tensor([127i8, -127i8, 0i8, 127i8, -1i8]), 0i32)\n  (to_list(values), to_list(order))\n}\nresult = run(0i64)\n",
    ),
];

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

fn evaluated(source: &str) -> String {
    let result = eval(request(source))
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

#[test]
fn compiled_sort_orders_mixed_inputs_as_eval_does() {
    let mut failures = Vec::new();
    for (name, source) in CASES {
        let expected = evaluated(source);
        let generated = ownership_support::emit(source, name);
        let (_summary, stdout) = ownership_support::run_program(&generated);
        if stdout != expected {
            failures.push(format!(
                "{name}: eval printed\n{expected}compiled C printed\n{stdout}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
