//! chelis#3403: `len` and `index` read a borrowed container on both lanes.
//!
//! `spec/05-risc-primitives.md` section 1.3.1 makes the container queries
//! auto-borrow their argument, and `spec/04-type-system.md` section 8.2 makes
//! `&T` a borrow type a parameter can declare. A `&List[T]` or `&Dict[K, V]`
//! parameter is therefore queryable, and the caller still owns the container
//! after lending it. The checker refused every such query as if the source had
//! written the explicit `len(&xs)`, which stays refused.
//!
//! Oracle: each program prints what `chelis eval` prints, an exact rendering
//! pinned here, when compiled to C and run against the `ownership-ledger`
//! runtime with every allocation finalized. The negative twin spells the
//! explicit borrow, and both lanes refuse it with the same diagnostic.

mod ownership_support;

use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "module Demo.Main\n\
def rows(seed: i64) -> List[tensor[2, f32]] = [to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])]\n\
def count_l(xs: &List[tensor[2, f32]]) -> i64 = len(xs)\n\
def first_l(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = index(xs, 0i64)\n\
def last_l(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = index(xs, sub(len(xs), 1i64))\n\
def count_d(d: &Dict[string, tensor[2, f32]]) -> i64 = len(d)\n";

struct Case {
    name: &'static str,
    body: &'static str,
    eval: &'static str,
}

const POSITIVE: &[Case] = &[
    Case {
        name: "len_of_a_borrowed_list",
        body: "n = count_l(rows(0i64))\n",
        eval: "n = 2\n",
    },
    Case {
        name: "index_of_a_borrowed_list",
        body: "first = to_list(first_l(rows(0i64)))\nlast = to_list(last_l(rows(0i64)))\n",
        eval: "first = [1.0, 2.0]\nlast = [3.0, 4.0]\n",
    },
    Case {
        name: "len_of_a_borrowed_dict",
        body: "n = count_d(dict_of([(\"a\", to_tensor([1.0f32, 2.0f32])), (\"b\", to_tensor([3.0f32, 4.0f32])), (\"c\", to_tensor([5.0f32, 6.0f32]))]))\n",
        eval: "n = 3\n",
    },
    // The caller lends one list to every query and then reads it itself.
    Case {
        name: "caller_keeps_the_lent_list",
        body: "def total(seed: i64) -> tensor[2, f32] = {\n  xs = rows(seed)\n  a = first_l(xs)\n  b = last_l(xs)\n  n = count_l(xs)\n  add(add(a, b), index(xs, sub(n, 2i64)))\n}\nresult = to_list(total(0i64))\n",
        eval: "result = [5.0, 8.0]\n",
    },
];

/// The explicit borrow expression, on a borrowed and on an owned container,
/// with the one refusal both lanes report.
const EXPLICIT_BORROW: &[(&str, &str)] = &[
    (
        "def count_e(xs: &List[tensor[2, f32]]) -> i64 = len(&xs)\nn = count_e(rows(0i64))\n",
        "len auto-borrows its List/Dict argument, so an explicit `&` is not a supported surface \
         form: write `len(xs)`, not `len(&xs)` (got &List tensor[2, f32])",
    ),
    (
        "def count_e(d: Dict[string, tensor[2, f32]]) -> i64 = len(&d)\nn = count_e(dict_of([(\"a\", to_tensor([1.0f32, 2.0f32]))]))\n",
        "len auto-borrows its List/Dict argument, so an explicit `&` is not a supported surface \
         form: write `len(xs)`, not `len(&xs)` (got &Dict string tensor[2, f32])",
    ),
    (
        "def first_e(xs: &List[tensor[2, f32]]) -> tensor[2, f32] = index(&xs, 0i64)\nfirst = to_list(first_e(rows(0i64)))\n",
        "index auto-borrows its List argument, so an explicit `&` is not a supported surface \
         form: write `index(xs, i)`, not `index(&xs, i)` (got &List tensor[2, f32])",
    ),
];

fn source(body: &str) -> String {
    format!("{PRELUDE}{body}")
}

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
        let source = source(case.body);
        let expected = evaluated(&source);
        assert_eq!(expected, case.eval, "eval output");
        let generated = ownership_support::emit(&source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{}\n  -> {head}", case.name, source(case.body))
    })
}

// REGRESSION TEST. On `ad87a3616` the checker refused every positive program:
// "len auto-borrows its List/Dict argument, so an explicit `&` is not a
// supported surface form" (and the `index` twin) for a borrowed parameter.
#[test]
fn borrowed_container_queries_compile_as_eval_runs_them() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn explicit_borrow_of_a_container_is_refused_on_both_lanes() {
    for &(body, message) in EXPLICIT_BORROW {
        let source = source(body);
        let refused = eval(request(&source)).expect_err("the evaluator must refuse");
        let refused = format!("{refused:?}");
        assert!(refused.contains(message), "eval of {body}: {refused}");
        let refused = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source,
            target: CompileTarget::C,
            entry_name: Some("fixture".into()),
        })
        .expect_err("the C build must refuse");
        let refused = format!("{refused:?}");
        assert!(refused.contains(message), "C build of {body}: {refused}");
    }
}
