//! chelis#2907: an element-wise scatter whose index and update extents are
//! known only at run time compiles, and its runtime plan guards them.
//!
//! A tensor helper loads each runtime-length input under its own anonymous
//! dimension name, so updates computed from the indices (`sub(positions,
//! positions)`) or read from an equal-length list carry a different name
//! from the indices. The verifier required the two names to be equal and the
//! C build stopped at an internal invariant, while `chelis eval` ran the
//! program. The runtime plan compares the observed update shape with the
//! index shape before reading any element ([05-OP-66]), which is the guard
//! runtime_extents.md C2.3 requires for an extent the graph cannot prove, so
//! only a static contradiction is refused before execution.
//!
//! Oracle: the positive programs compile, run against the `ownership-ledger`
//! runtime with every allocation finalized, and print what `chelis eval`
//! prints. The negative twin passes updates of a different length; both
//! lanes must fail at the scatter.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Case {
    name: &'static str,
    source: &'static str,
}

const POSITIVE: &[Case] = &[
    // The issue's program: the updates are elementwise in the indices.
    Case {
        name: "updates_derived_from_the_indices",
        source: "module Demo.Main\ndef clear[h](base: &tensor[h, i64], holes: List[i64]) -> List[i64] = {\n  positions = to_tensor(holes)\n  cleared = sub(positions, positions)\n  to_list(scatter_elements(copy(base), positions, cleared, 0i32))\n}\ndef run(seed: i64) -> List[i64] = {\n  base = to_tensor([5i64, 6i64, 7i64, 8i64])\n  clear(base, [1i64, 3i64])\n}\nresult = run(0i64)\n",
    },
    // The issue's helper variant: the updates are shaped like the indices.
    Case {
        name: "updates_from_a_helper_shaped_like_the_indices",
        source: "module Demo.Main\ndef zeros_like[n](like: &tensor[n, i64]) -> tensor[n, i64] = 0i64 |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))\ndef clear[h](base: &tensor[h, i64], holes: List[i64]) -> List[i64] = {\n  positions = to_tensor(holes)\n  cleared = zeros_like(positions)\n  to_list(scatter_elements(copy(base), positions, cleared, 0i32))\n}\nresult = clear(to_tensor([5i64, 6i64, 7i64, 8i64]), [1i64, 3i64])\n",
    },
    // Independent lists of equal length.
    Case {
        name: "updates_from_an_equal_length_list",
        source: "module Demo.Main\ndef fill[h](base: &tensor[h, i64], holes: List[i64], values: List[i64]) -> List[i64] = to_list(scatter_elements(copy(base), to_tensor(holes), to_tensor(values), 0i32))\nresult = fill(to_tensor([5i64, 6i64, 7i64, 8i64]), [1i64, 3i64], [0i64, 9i64])\n",
    },
];

const LENGTHS_DIFFER: Case = Case {
    name: "updates_longer_than_the_indices",
    source: "module Demo.Main\ndef fill[h](base: &tensor[h, i64], holes: List[i64], values: List[i64]) -> List[i64] = to_list(scatter_elements(copy(base), to_tensor(holes), to_tensor(values), 0i32))\nresult = fill(to_tensor([5i64, 6i64, 7i64, 8i64]), [1i64, 3i64], [0i64, 9i64, 4i64])\n",
};

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

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn check_positive(case: &Case) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(case.source);
        let generated = ownership_support::emit(case.source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{}\n  -> {head}", case.name, case.source)
    })
}

// REGRESSION TEST. On `4a53cc7bd` the C build stopped at `scatter_elements
// requires indices.dims == updates.dims` for every positive program.
#[test]
fn scatter_over_run_time_extents_compiles_as_eval_runs_it() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn updates_of_another_length_fail_in_both_lanes() {
    let refused = eval(request(LENGTHS_DIFFER.source)).expect_err("the evaluator must refuse");
    let refused = refused
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    // spec/04 section 4.7: both lanes trap `Domain` in `scatter_elements` with the
    // same context line (chelis#3107).
    let failure = "scatter_elements operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\n\
                   numeric trap: domain in scatter_elements at i64";
    assert!(refused.contains(failure), "{refused}");
    let generated = ownership_support::emit(LENGTHS_DIFFER.source, LENGTHS_DIFFER.name);
    let stderr = ownership_support::run_failure_stderr(&generated, "");
    assert!(stderr.contains(failure), "{stderr}");
}
