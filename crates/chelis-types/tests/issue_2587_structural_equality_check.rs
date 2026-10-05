//! chelis#2587: the checker's half of [05-OP-36]'s recursive equality.
//!
//! `eq` and `neq` admit unit and two `List`, tuple, `Dict`, `Option`, or ADT
//! values of one static type whose reachable fields are recursively
//! equality-comparable, and return one scalar `bool`. A reachable function or
//! resource handle is rejected with a diagnostic naming [05-OP-36]; a
//! reachable key is refused by the key rule of spec/04 section 1.1, which
//! names no comparison for `key`. Mismatched static types and ordered
//! comparison of a structured value stay type errors.
//!
//! Every positive below has a negative beside it.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_linearity, check_typed_program, infer_ir_program};

fn deep(sources: &[&str]) -> Vec<chelis_deep::Expr> {
    let mut decls = Vec::new();
    for source in sources {
        decls.extend(parse_str(source).unwrap_or_else(|error| panic!("{source}\n{error:?}")));
    }
    let deep = desugar_program(&decls).expect("fixture desugars");
    chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .expect("macro expansion")
        .into_exprs()
}

/// Both checker ingresses, then linearity: the verdict a front end gives.
fn verdict(sources: &[&str]) -> Result<(), Vec<CheckError>> {
    let exprs = deep(sources);
    let ir_errors = infer_ir_program(&exprs).errors;
    let checked = match check_typed_program(&exprs) {
        Ok(checked) => {
            assert!(
                ir_errors.is_empty(),
                "the IR ingress rejects what the typed ingress accepts: {ir_errors:?}\n{sources:?}"
            );
            checked
        }
        Err(result) => {
            assert!(
                !ir_errors.is_empty(),
                "the IR ingress accepts what the typed ingress rejects: {:?}\n{sources:?}",
                result.errors
            );
            return Err(result.errors);
        }
    };
    check_linearity(&checked).map(|_| ())
}

fn accepts(sources: &[&str]) {
    if let Err(errors) = verdict(sources) {
        panic!(
            "expected acceptance of {sources:?}, got {:?}",
            errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }
}

fn rejects(sources: &[&str]) -> Vec<CheckError> {
    let errors = verdict(sources).expect_err(&format!("expected a rejection of {sources:?}"));
    assert!(!errors.is_empty(), "{sources:?}");
    errors
}

/// A rejection carrying [05-OP-36]'s equality-domain diagnostic, which names
/// the operation, the operand type, and the incomparable type it reaches.
fn rejects_with_equality_domain(source: &str, operation: &str, reached: &str) {
    let errors = rejects(&[source]);
    let error = errors
        .iter()
        .find(|error| error.message.contains("[05-OP-36]"))
        .unwrap_or_else(|| {
            panic!(
                "{source}: no [05-OP-36] diagnostic in {:?}",
                errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            )
        });
    assert!(
        matches!(error.kind, CheckErrorKind::TypeMismatch),
        "{source}: {:?}",
        error.kind
    );
    assert!(
        error.message.contains(&format!("`{operation}`")) && error.message.contains(reached),
        "{source}: {}",
        error.message
    );
}

const POINT: &str = "type Point =\n  | Point { x: f64, y: f64 }\n";
const HOLDER: &str = "type Holder[T] =\n  | Holder(T)\n";
/// The data types the admitted-kind table draws on.
const DATA_TYPES: &str = "type Point =\n  | Point { x: f64, y: f64 }\ntype Chain =\n  | Link(i32, \
                          Chain)\n  | End\ntype Holder[T] =\n  | Holder(T)\ntype Tag[T] =\n  | \
                          Tag\n";
const CALLBACK: &str = "type Callback =\n  | Callback { run: (i32) -> i32 }\n";
const KEYED: &str = "type Keyed =\n  | Keyed { k: key }\n";

#[test]
fn every_admitted_structured_kind_checks_and_returns_one_bool() {
    for operation in ["eq", "neq"] {
        for parameters in [
            "unit",
            "(i32, string)",
            "(f32,)",
            "List[f32]",
            "List[List[i64]]",
            "Dict[string, i64]",
            "Dict[i64, List[f64]]",
            "Option[i32]",
            "Option[Option[string]]",
            "List[tensor[2, f32]]",
            "(tensor[3, bool], List[bf16])",
            "Point",
            "List[Point]",
            "Chain",
            "Holder[List[i32]]",
            "Holder[Dict[bool, Point]]",
            // A type argument no field reaches is not part of the value.
            "Tag[(i32) -> i32]",
        ] {
            let prelude = DATA_TYPES;
            let source = format!(
                "{prelude}def f(a: {parameters}, b: {parameters}) -> bool = not({operation}(a, \
                 b))\n"
            );
            accepts(&[&source]);
            // NEGATIVE PARITY: the result is a scalar bool, never the
            // operand type or a free variable another use could bind.
            let wrong = format!(
                "{prelude}def f(a: {parameters}, b: {parameters}) -> i32 = {operation}(a, b)\n"
            );
            rejects(&[&wrong]);
        }
    }
}

#[test]
fn unit_values_compare_without_annotations() {
    accepts(&["def f() -> bool = eq((), ())\n"]);
    accepts(&["def f() -> bool = neq((), ())\n"]);
    accepts(&["def f() -> bool = eq([(1i32, \"a\")], [(1i32, \"a\")])\n"]);
    accepts(&["def f() -> bool = eq(Some([1.5f32]), None)\n"]);
}

#[test]
fn a_reachable_function_is_rejected_by_the_equality_domain() {
    for operation in ["eq", "neq"] {
        for (prelude, parameters, reached) in [
            ("", "(i32) -> i32", "(i32) -> i32"),
            ("", "List[(i32) -> i32]", "(i32) -> i32"),
            ("", "(i32, (f32) -> f32)", "(f32) -> f32"),
            ("", "Option[(i64) -> bool]", "(i64) -> bool"),
            ("", "Dict[string, (i32) -> i32]", "(i32) -> i32"),
            (CALLBACK, "Callback", "(i32) -> i32"),
            (CALLBACK, "List[Option[Callback]]", "(i32) -> i32"),
            (HOLDER, "Holder[(string) -> unit]", "(string) -> ()"),
        ] {
            let source = format!(
                "{prelude}def f(a: {parameters}, b: {parameters}) -> bool = {operation}(a, b)\n"
            );
            rejects_with_equality_domain(&source, operation, reached);
        }
    }
}

#[test]
fn a_reachable_resource_handle_is_rejected_by_the_equality_domain() {
    for operation in ["eq", "neq"] {
        for parameters in ["MappedFile", "List[MappedFile]", "(i32, MappedFile)"] {
            let source =
                format!("def f(a: {parameters}, b: {parameters}) -> bool = {operation}(a, b)\n");
            rejects_with_equality_domain(&source, operation, "MappedFile");
        }
    }
}

/// spec/04 section 1.1: `key` has no comparison, so a key anywhere in the
/// compared structure is refused by the key rule, whose diagnostic names the
/// key and the section.
#[test]
fn a_reachable_key_is_refused() {
    for operation in ["eq", "neq"] {
        for (prelude, parameters) in [
            ("", "key"),
            ("", "List[key]"),
            ("", "(i32, key)"),
            ("", "Option[key]"),
            ("", "Dict[string, key]"),
            ("", "List[tensor[2, key]]"),
            (KEYED, "Keyed"),
            (HOLDER, "Holder[key]"),
        ] {
            let source = format!(
                "{prelude}def f(a: {parameters}, b: {parameters}) -> bool = {operation}(a, b)\n"
            );
            let errors = rejects(&[&source]);
            assert!(
                errors
                    .iter()
                    .any(|error| error.message.contains("key")
                        && error.message.contains("section 1.1")),
                "{source}: {:?}",
                errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn mismatched_static_types_stay_rejected() {
    for operation in ["eq", "neq"] {
        for (prelude, lhs, rhs) in [
            ("", "List[i32]", "List[i64]"),
            ("", "(i32, string)", "(i32, bool)"),
            ("", "(i32, string)", "(i32, string, i32)"),
            ("", "Option[f32]", "Option[f64]"),
            ("", "Dict[string, i64]", "Dict[i64, i64]"),
            ("", "List[i32]", "Option[i32]"),
            ("", "unit", "(i32,)"),
            (POINT, "Point", "Option[Point]"),
            (HOLDER, "Holder[i32]", "Holder[string]"),
        ] {
            let source =
                format!("{prelude}def f(a: {lhs}, b: {rhs}) -> bool = {operation}(a, b)\n");
            rejects(&[&source]);
        }
    }
}

#[test]
fn ordered_comparison_of_a_structured_value_stays_rejected() {
    for operation in ["lt", "gt", "lte", "gte", "cmplt"] {
        for (prelude, parameters) in [
            ("", "unit"),
            ("", "List[i32]"),
            ("", "(i32, i32)"),
            ("", "Option[i32]"),
            ("", "Dict[string, i32]"),
            (POINT, "Point"),
        ] {
            let source = format!(
                "{prelude}def f(a: {parameters}, b: {parameters}) -> bool = {operation}(a, b)\n"
            );
            rejects(&[&source]);
        }
    }
}

const PROB_MODULE: &str = "module Stats.Prob
export (probability)
@opaque
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";

const SEALED_MODULE: &str = "module Stats.Sealed
export (sealed)
@opaque
type Sealed =
  | Sealed { run: (f32) -> f32 }
def sealed(x: f32) -> Sealed = Sealed { run: fn (y: f32) -> y }
";

/// spec/04 section 2.5: equality inspects no field through the surface, so it
/// is not in the opaque rejection set.
#[test]
fn opaque_values_compare_outside_their_defining_module() {
    for operation in ["eq", "neq"] {
        let outside = format!(
            "module Agent.Strategy
def same(a: Probability, b: Probability) -> bool = {operation}(a, b)
def fresh() -> bool = {operation}(probability(0.5f32), probability(0.25f32))
"
        );
        accepts(&[PROB_MODULE, &outside]);
        // NEGATIVE PARITY: opacity does not hide a reachable function from
        // the equality domain.
        let sealed = format!(
            "module Agent.Strategy
def same(a: Sealed, b: Sealed) -> bool = {operation}(a, b)
"
        );
        let errors = rejects(&[SEALED_MODULE, &sealed]);
        assert!(
            errors
                .iter()
                .any(|error| error.message.contains("[05-OP-36]")),
            "{:?}",
            errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }
}

/// A field type that is still a variable is decided once it binds, and at the
/// declaration boundary when it never does: an unbounded binder or an
/// undetermined element type could be a function.
#[test]
fn a_variable_field_type_is_decided_when_it_binds_or_at_the_boundary() {
    for operation in ["eq", "neq"] {
        for source in [
            format!("def ok() -> bool = (fn (x) -> {operation}([x], [x]))(1i32)\n"),
            format!("def ok() -> bool = (fn (x) -> {operation}((x, 1i64), (x, 2i64)))(\"a\")\n"),
            format!("def ok[p: Float](x: List[p], y: List[p]) -> bool = {operation}(x, y)\n"),
            format!("def ok[p: Int](x: Option[(p, p)]) -> bool = {operation}(x, None)\n"),
            format!("{HOLDER}def ok[p: Numeric](x: Holder[p]) -> bool = {operation}(x, x)\n"),
        ] {
            accepts(&[&source]);
        }
        for source in [
            format!("def bad[a](x: List[a]) -> bool = {operation}(x, x)\n"),
            format!("def bad[a](x: (i32, a)) -> bool = {operation}(x, x)\n"),
            format!("{HOLDER}def bad[a](x: Holder[a]) -> bool = {operation}(x, x)\n"),
            format!("def bad() -> bool = {operation}([], [])\n"),
        ] {
            let errors = rejects(&[&source]);
            assert!(
                errors.iter().any(|error| error
                    .message
                    .contains(&format!("`{operation}` admits only some operand types"))),
                "{source}: {:?}",
                errors.iter().map(|e| &e.message).collect::<Vec<_>>()
            );
        }
    }
}
