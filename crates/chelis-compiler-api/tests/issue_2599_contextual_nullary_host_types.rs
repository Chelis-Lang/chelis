//! chelis#2599, chelis#2470, chelis#2417, chelis#2878: a bare `None` or `[]`
//! whose type only its context fixes compiles to C exactly as `chelis eval`
//! runs it.
//!
//! The checker types a bare nullary constructor or empty list from its
//! context: a `scan` or `fold` seed, a callback's accumulator or result, a
//! sibling list element, a call's parameter, or the other operand of `eq`,
//! `concat` or `dict_merge`. Host lowering gives such a
//! value an inference hole, and the pass that pushes checked context down the
//! host tree must fill it before the code-generation boundary. That pass did
//! not enter a higher-order operation's seed or callback body, or the
//! arguments of a builtin that has no expected-type rule (such as `len`), so a
//! hole below one of them reached the boundary and `chelis build` rejected a
//! program that `chelis check` accepted and `chelis eval` ran.
//!
//! Oracle for the positive corpus: the compiled program runs against the
//! `ownership-ledger` runtime, every allocation must be finalized with no live
//! owner left, and stdout must equal the in-process evaluator's rendering of
//! the same roots. Every case runs before the test reports, so one failing
//! shape cannot hide another.
//!
//! The negative corpus mirrors each positive shape with a bare `None` whose
//! type nothing fixes. The checker accepts those programs and the evaluator
//! runs them, but no host type exists, so compilation must still fail with the
//! typed host-resolution diagnostic rather than choose a type.

mod ownership_support;

use chelis_compiler_api::compiler::{CompilerError, compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use chelis_vocab::DiagnosticKind;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// One program: its name and complete Surf source.
struct Case {
    name: &'static str,
    source: &'static str,
}

const POSITIVE: &[Case] = &[
    // chelis#2599: the seed is the only thing that fixes `None`'s type.
    Case {
        name: "scan_seed_none",
        source: "def case(flag: bool) -> List[Option[string]] =\n  scan(fn (acc: Option[string], x: string) -> Some(x), None, [string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\na = case(true)\n",
    },
    // chelis#2599: a sibling element in the callback's result list.
    Case {
        name: "flat_map_sibling_none",
        source: "def case(flag: bool) -> List[Option[string]] =\n  flat_map(fn (x: string) -> [Some(x), None], [string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\na = case(true)\n",
    },
    // An `Option[i64]` fold accumulator whose callback returns bare `None`.
    Case {
        name: "fold_callback_returns_none",
        source: "def case(flag: bool) -> List[Option[i64]] =\n  [fold(fn (acc: Option[i64], x: i64) -> if gt(x, 2i64) then None else Some(x), Some(0i64), [1i64, 2i64, 3i64])]\na = case(true)\n",
    },
    Case {
        name: "fold_callback_matches_to_none",
        source: "def case(flag: bool) -> List[Option[i64]] =\n  [fold(fn (acc: Option[i64], k: i64) -> match acc with {\n    | Some(v) => if gt(v, 5i64) then None else Some(add(v, k))\n    | None => None\n  }, Some(0i64), range(0i64, 5i64))]\na = case(true)\n",
    },
    Case {
        name: "fold_seed_none",
        source: "def case(flag: bool) -> List[Option[i64]] =\n  [fold(fn (acc: Option[i64], k: i64) -> Some(k), None, range(0i64, 3i64))]\na = case(true)\n",
    },
    Case {
        name: "map_callback_returns_none",
        source: "def case(flag: bool) -> List[Option[string]] =\n  map(fn (x: string) -> if gt(string_len(x), 2i64) then Some(x) else None, [string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")])\na = case(true)\n",
    },
    Case {
        name: "partition_list_sibling_none",
        source: "def case(flag: bool) -> (List[Option[i64]], List[Option[i64]]) =\n  partition(fn (o: Option[i64]) -> flag, [Some(1i64), None])\na = case(true)\nb = case(false)\n",
    },
    Case {
        name: "scan_callback_returns_none",
        source: "def case(flag: bool) -> List[Option[i64]] =\n  scan(fn (acc: Option[i64], x: i64) -> if gt(x, 1i64) then None else Some(x), Some(0i64), [1i64, 2i64])\na = case(true)\n",
    },
    // A predicate's result is `bool`, so only the predicate body's own
    // subexpressions can type the `None`.
    Case {
        name: "filter_predicate_sibling_none",
        source: "def case(flag: bool) -> List[i64] =\n  filter(fn (x: i64) -> gt(len([Some(x), None]), x), [0i64, 1i64, 5i64])\na = case(true)\n",
    },
    Case {
        name: "partition_predicate_sibling_none",
        source: "def case(flag: bool) -> (List[i64], List[i64]) =\n  partition(fn (x: i64) -> gt(len([Some(x), None]), x), [0i64, 1i64, 5i64])\na = case(true)\n",
    },
    // The parent's type is known and the child's own type is resolved and
    // shaped differently: `len` returns `i64` over a list of options, and
    // each higher-order operation below returns something other than its
    // list operand. The child keeps its own type, which is what reaches the
    // `None` inside it.
    Case {
        name: "len_of_sibling_typed_list",
        source: "def case(flag: bool) -> i64 = len([Some(1i64), None])\na = case(true)\n",
    },
    Case {
        name: "fold_scalar_accumulator_over_option_list",
        source: "def case(flag: bool) -> i64 =\n  fold(fn (acc: i64, o: Option[i64]) -> add(acc, 1i64), 0i64, [Some(1i64), None])\na = case(true)\n",
    },
    Case {
        name: "scan_scalar_accumulator_over_option_list",
        source: "def case(flag: bool) -> List[i64] =\n  scan(fn (acc: i64, o: Option[i64]) -> add(acc, 1i64), 0i64, [Some(1i64), None])\na = case(true)\n",
    },
    Case {
        name: "map_over_option_list",
        source: "def case(flag: bool) -> List[i64] =\n  map(fn (o: Option[i64]) -> 1i64, [Some(1i64), None])\na = case(true)\n",
    },
    Case {
        name: "flat_map_over_option_list",
        source: "def case(flag: bool) -> List[Option[i64]] =\n  flat_map(fn (o: Option[i64]) -> [o, o], [Some(1i64), None])\na = case(true)\n",
    },
    Case {
        name: "filter_over_option_list",
        source: "def case(flag: bool) -> List[Option[i64]] =\n  filter(fn (o: Option[i64]) -> flag, [Some(1i64), None])\na = case(true)\nb = case(false)\n",
    },
    // chelis#2470: the parameter type fixes `None` inside `len(...)`.
    Case {
        name: "call_argument_none_under_len",
        source: "def pick(o: Option[List[i64]]) -> List[i64] =\n  match o with {\n    | Some(xs) => xs\n    | None => []\n  }\nb = len(pick(Some([1i64, 2i64, 3i64, 4i64])))\nd = len(pick(None))\n",
    },
    // chelis#2417: the parameter type fixes a recursive accumulator's `[]`.
    Case {
        name: "recursive_accumulator_empty_list_under_len",
        source: "def build(n: i64, acc: List[f32]) -> List[f32] = if lte(n, 0i64) then acc else build(sub(n, 1i64), append(acc, 1.0f32))\nout = len(build(3i64, []))\n",
    },
    // chelis#2878: the other operand fixes a nested empty literal's type.
    Case {
        name: "equality_operand_nested_empty_literals",
        source: "def g(xs: List[List[i32]]) -> bool = eq(xs, [[]])\ndef k(a: Option[List[i64]]) -> bool = eq(a, Some([]))\ndef f(d: Dict[string, i64]) -> bool = eq(d, dict_of([]))\ndef h(flag: bool) -> bool = eq(Some(None), Some(Some(1i32)))\ndef m(flag: bool) -> bool = eq([[]], [[1i32]])\na = g([[1i32]])\nb = g([[]])\nc = k(None)\nd = k(Some([]))\ne = f(dict_of([(\"x\", 1i64)]))\nf2 = h(true)\ng2 = m(true)\n",
    },
    Case {
        name: "concat_and_dict_merge_nested_empty_literals",
        source: "def c(xs: List[List[i32]]) -> List[List[i32]] = concat(xs, [[]])\ndef d(m: Dict[string, List[i64]]) -> Dict[string, List[i64]] = dict_merge(m, dict_of([(\"k\", [])]))\na = c([[1i32]])\nb = d(dict_of([(\"x\", [1i64])]))\n",
    },
    // chelis#3153: the empty accumulator's element type is a DIMENSION-generic
    // ADT, so only the callee's parameter type carries the dimension into it.
    // A type-generic ADT (`Box[a]`), a non-generic ADT, and a bare
    // `tensor[n, f32]` element each already compiled, so the dimension beneath
    // the ADT is the whole trigger.
    Case {
        name: "empty_accumulator_dimension_generic_adt",
        source: "type Col[n] =\n  | FCol(tensor[n, f32])\n  | BCol(tensor[n, bool])\ndef repl[n](xs: List[Col[n]], acc: List[Col[n]]) -> List[Col[n]] =\n  if eq(len(xs), 0i64) then acc\n  else {\n    hp = index(xs, 0i64)\n    repl(skip(xs, 1i64), append(acc, hp))\n  }\na = len(repl([FCol(to_tensor([cast(1.0, f32), cast(2.0, f32)]))], []))\n",
    },
    // chelis#3153, Coral's actual shape: the element is a tuple whose second
    // component is the dimension-generic ADT, threaded through a named
    // accumulator binding rather than an inline argument.
    Case {
        name: "empty_accumulator_tuple_of_dimension_generic_adt",
        source: "type Col[n] =\n  | FCol(tensor[n, f32])\n  | BCol(tensor[n, bool])\ndef repl[n](xs: List[(string, Col[n])], acc: List[(string, Col[n])]) -> List[(string, Col[n])] =\n  if eq(len(xs), 0i64) then acc\n  else {\n    hp = index(xs, 0i64)\n    next = append(acc, hp)\n    repl(skip(xs, 1i64), next)\n  }\na = len(repl([(\"a\", FCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))], []))\n",
    },
];

/// Each shape above with a `None` whose type nothing determines.
const AMBIGUOUS: &[Case] = &[
    Case {
        name: "root_none",
        source: "a = None\n",
    },
    Case {
        name: "len_of_none_list",
        source: "def case(flag: bool) -> i64 = len([None, None])\na = case(true)\n",
    },
    Case {
        name: "scan_callback_len_of_none",
        source: "def case(flag: bool) -> List[i64] =\n  scan(fn (acc: i64, x: i64) -> len([None, None]), 0i64, [1i64, 2i64])\na = case(true)\n",
    },
    Case {
        name: "flat_map_callback_len_of_none",
        source: "def case(flag: bool) -> List[i64] =\n  flat_map(fn (x: i64) -> [len([None])], [1i64, 2i64])\na = case(true)\n",
    },
    Case {
        name: "fold_callback_len_of_none",
        source: "def case(flag: bool) -> i64 =\n  fold(fn (acc: i64, x: i64) -> add(acc, len([None])), 0i64, [1i64, 2i64])\na = case(true)\n",
    },
    // chelis#3153 negative parity: both arguments are empty, so nothing fixes
    // the dimension of `Col[n]`. The checker accepts it and the evaluator runs
    // it, but no host type exists, so the fix must NOT invent one.
    Case {
        name: "empty_accumulator_dimension_generic_adt_unfixed",
        source: "type Col[n] =\n  | FCol(tensor[n, f32])\n  | BCol(tensor[n, bool])\ndef repl[n](xs: List[Col[n]], acc: List[Col[n]]) -> List[Col[n]] =\n  if eq(len(xs), 0i64) then acc\n  else {\n    hp = index(xs, 0i64)\n    repl(skip(xs, 1i64), append(acc, hp))\n  }\na = len(repl([], []))\n",
    },
];

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
        let head: String = panic_message(payload).chars().take(600).collect();
        format!("{}:\n{}\n  -> {head}", case.name, case.source)
    })
}

fn unresolved_host_type_rejection(error: &CompilerError) -> Result<(), String> {
    if error.stage != "lower" || error.errors.len() != 1 {
        return Err(format!("unexpected error envelope: {error:?}"));
    }
    let diagnostic = &error.errors[0];
    let message = &diagnostic.message;
    if diagnostic.kind() == DiagnosticKind::LowerError
        && message.contains("host type did not resolve before the code-generation boundary")
        && message.contains("unresolved host inference variable")
        && message.contains("[05-UNS-1]")
        && message.contains("chelis#730")
    {
        Ok(())
    } else {
        Err(format!(
            "not the typed host-resolution rejection: {diagnostic:?}"
        ))
    }
}

fn check_ambiguous(case: &Case) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        // The checker accepts the program and the evaluator runs it, so the
        // rejection below belongs to host lowering alone.
        evaluated(case.source);
        let result = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: case.source.to_string(),
            target: CompileTarget::C,
            entry_name: None,
        });
        match result {
            Ok(_) => Err("compiled although no host type exists".to_string()),
            Err(error) => unresolved_host_type_rejection(&error),
        }
    }))
    .map_err(panic_message)
    .and_then(|outcome| outcome)
    .map_err(|message| {
        let head: String = message.chars().take(600).collect();
        format!("{}:\n{}\n  -> {head}", case.name, case.source)
    })
}

// REGRESSION TEST. On `5008a9310` every case failed to compile with the
// unresolved host inference variable diagnostic.
#[test]
fn a_context_fixed_nullary_value_compiles_as_eval_runs_it() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        POSITIVE.len(),
        failures.join("\n\n")
    );
}

#[test]
fn a_nullary_value_no_context_fixes_is_still_rejected() {
    let failures: Vec<String> = AMBIGUOUS
        .iter()
        .filter_map(|case| check_ambiguous(case).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        AMBIGUOUS.len(),
        failures.join("\n\n")
    );
}
