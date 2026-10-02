//! chelis#2576: compiled C renders an `Option` nested inside a list or a
//! data-type value, directly or through a nested tuple, exactly as `chelis
//! eval` does. A root whose value is an option, and an option component of a
//! tuple or record root, render the same way (chelis#2597).
//!
//! The runtime's recursive value renderer had no arm for the `Option` value
//! tag, so a root such as `map(fn (x: string) -> Some(x), xs)` built, printed
//! its label and then aborted in `chelis_print_list` with
//! `validate_value rejects unknown tags`, although the tag is valid and the
//! list was well formed. The values themselves were always boxed correctly.
//!
//! The corpus puts an option inside a list, a data-type value, a tuple held
//! by a list, and another option, with scalar, string, list and tensor
//! payloads. Oracle: the compiled program runs
//! against the `ownership-ledger` runtime, every allocation must be finalized
//! with no live owner left, and stdout must equal the in-process evaluator's
//! rendering of the same roots. Every case runs before the test reports, so
//! one failing shape cannot hide another.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const STRINGS: &str = "[string_concat(\"ab\", \"c\"), string_concat(\"x\", \"\")]";

/// One case: declarations, the function's result type and its body.
struct Case {
    name: &'static str,
    prelude: &'static str,
    result: &'static str,
    body: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "map_some_string",
        prelude: "",
        result: "List[Option[string]]",
        body: "map(fn (x: string) -> Some(x), {strings})",
    },
    Case {
        name: "map_some_or_none",
        prelude: "def pick(s: string) -> Option[string] = if gt(string_len(s), 2i64) then Some(s) else None\n",
        result: "List[Option[string]]",
        body: "map(fn (x: string) -> pick(x), {strings})",
    },
    Case {
        name: "map_some_scalar",
        prelude: "",
        result: "List[Option[i64]]",
        body: "map(fn (x: string) -> Some(string_len(x)), {strings})",
    },
    Case {
        name: "map_some_list",
        prelude: "",
        result: "List[Option[List[string]]]",
        body: "map(fn (x: string) -> Some([x, string_concat(x, \"!\")]), {strings})",
    },
    Case {
        name: "map_some_tensor",
        prelude: "",
        result: "List[Option[tensor[3, f32]]]",
        body: "map(fn (x: tensor[3, f32]) -> Some((x + x)), [to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])])",
    },
    Case {
        name: "map_nested_option",
        prelude: "",
        result: "List[Option[Option[string]]]",
        body: "map(fn (x: string) -> Some(Some(x)), {strings})",
    },
    Case {
        name: "map_tuple_holding_option",
        prelude: "",
        result: "List[(Option[string], i64)]",
        body: "map(fn (x: string) -> (Some(x), string_len(x)), {strings})",
    },
    Case {
        name: "map_data_type_holding_option",
        prelude: "type Boxed =\n  | Boxed(Option[string])\n",
        result: "List[Boxed]",
        body: "map(fn (x: string) -> Boxed(Some(x)), {strings})",
    },
    Case {
        name: "filter_over_options",
        prelude: "",
        result: "List[Option[string]]",
        body: "filter(fn (o: Option[string]) -> flag, map(fn (x: string) -> Some(x), {strings}))",
    },
    Case {
        name: "flat_map_some",
        prelude: "",
        result: "List[Option[string]]",
        body: "flat_map(fn (x: string) -> [Some(x), Some(string_concat(x, \"!\"))], {strings})",
    },
    Case {
        name: "scan_some",
        prelude: "",
        result: "List[Option[string]]",
        body: "scan(fn (acc: Option[string], x: string) -> Some(x), Some(string_concat(\"in\", \"it\")), {strings})",
    },
    Case {
        name: "list_literal_of_options",
        prelude: "",
        result: "List[Option[string]]",
        body: "[Some(string_concat(\"ab\", \"c\")), None]",
    },
];

/// chelis#2597: the option is the root itself or one of the root's
/// manifested components, not an item of a container.
const ROOT_CASES: &[Case] = &[
    Case {
        name: "root_some_string",
        prelude: "",
        result: "Option[string]",
        body: "if flag then Some(string_concat(\"ab\", \"c\")) else None",
    },
    Case {
        name: "root_nested_option",
        prelude: "",
        result: "Option[Option[i64]]",
        body: "if flag then Some(Some(2i64)) else Some(None)",
    },
    Case {
        name: "root_some_tensor",
        prelude: "",
        result: "Option[tensor[3, f32]]",
        body: "if flag then Some(to_tensor([1.0, 2.0, 3.0])) else None",
    },
    Case {
        name: "tuple_component",
        prelude: "",
        result: "(Option[string], i64)",
        body: "(if flag then Some(string_concat(\"ab\", \"c\")) else None, 1i64)",
    },
    Case {
        name: "nested_tuple_component",
        prelude: "",
        result: "((Option[f32], i64), i64)",
        body: "((if flag then Some(1.5f32) else None, 2i64), 3i64)",
    },
    // A call result's constructor is not statically fixed, so this record
    // stays one bare root ([05-OBS-8]); it controls the expanded cases above.
    Case {
        name: "bare_record_root_with_an_option_field",
        prelude: "type Zone =\n  | Zone { name: string, next: Option[(i64, Option[i64])] }\n",
        result: "Zone",
        body: "Zone { name: string_concat(\"UT\", \"C\"), next: if flag then Some((0i64, None)) else None }",
    },
];

fn instantiate(case: &Case) -> String {
    format!(
        "{}def case(flag: bool) -> {} =\n  {}\na = case(true)\nb = case(false)\n",
        case.prelude,
        case.result,
        case.body.replace("{strings}", STRINGS),
    )
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
        format!("{name}:\n{source}\n  -> {head}")
    })
}

// REGRESSION TEST. On `d029224fd` every case aborted after printing `a =`.
#[test]
fn an_option_inside_a_compiled_root_renders_as_eval_renders_it() {
    let failures: Vec<String> = CASES
        .iter()
        .filter_map(|case| check(case.name, &instantiate(case)).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n\n")
    );
}

// REGRESSION TEST. On `5008a9310` a root of option type failed the build and
// an option component of a tuple or record root aborted after its label.
#[test]
fn an_option_root_or_root_component_renders_as_eval_renders_it() {
    let failures: Vec<String> = ROOT_CASES
        .iter()
        .filter_map(|case| check(case.name, &instantiate(case)).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        ROOT_CASES.len(),
        failures.join("\n\n")
    );
}
