//! chelis#2216 and chelis#2158: a builtin's operand-kind or dtype-family
//! constraint is decided when the operand's type is a rigid authored binder.
//!
//! `spec/04-type-system.md` [04-INF-6] quantifies an authored binder over every
//! instantiation its declaration admits, and [04-DTYPE-2] makes a bounded
//! binder a scalar primitive of its family at each of them. Before the fix two
//! checker paths skipped the operation's own admission rule for such an
//! operand, and each program below checked at score 1 before trapping in `eval`
//! or emitting C that does not compile:
//!
//! - a call suspended on an operand whose type was still a variable
//!   (chelis#1512's ledger) waited for a binding that a binder never receives,
//!   and the declaration boundary dropped it (`numel(k)` with `k: p`);
//! - `cast_trunc` admitted a source whose dtype was a variable as "re-checked
//!   once unification binds it" (chelis#2158), and the variable-target cast
//!   admitted a variable source and a concrete integer source unchecked.
//!
//! Every fixture is asserted on both checker ingresses (chelis#1107). The
//! suspended call is now replayed at each instantiation of its binders, so a
//! rejection carries the route's own text and names the instantiation where it
//! fails, and an operation that accepts every member of the bound stays
//! accepted. An unbounded binder is replayed at an arbitrary type, left
//! unbound, rather than at any one type. A `cast_trunc` source requirement is
//! recorded on the variable that stands for the source dtype, like every other
//! dtype-family policy; a `cast` source may also be `bool` ([05-OP-63]), so
//! only an authored binder is held to a family there.
//!
//! What is claimed is the forms below. A flexible inference variable that
//! nothing binds is decided the same way, at an arbitrary type, since
//! chelis#2518; `issue_731_declaration_close_obligations.rs` owns those forms.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("surf must parse: {source}\n{e:?}"));
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn expanded(source: &str) -> Vec<Expr> {
    expand_program(&desugared(source), &ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs()
}

/// One diagnostic as `[kind] message` plus its suggestions.
fn rendered(errors: &[CheckError]) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = errors
        .iter()
        .map(|e| {
            (
                format!("[{:?}] {}", e.kind, e.message),
                e.suggestions.clone(),
            )
        })
        .collect();
    out.sort();
    out
}

fn agreed_diagnostics(source: &str) -> Vec<(String, Vec<String>)> {
    let typed = match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    assert_eq!(
        typed, ir,
        "both checker ingresses must return the same diagnostics for:\n{source}"
    );
    typed
}

fn accepts(source: &str) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics.is_empty(),
        "must type-check:\n{source}\ngot {diagnostics:?}"
    );
}

/// The diagnostic whose message contains `fragment`, among a rejection's.
fn rejection_containing(source: &str, fragment: &str) -> (String, Vec<String>) {
    let diagnostics = agreed_diagnostics(source);
    diagnostics
        .iter()
        .find(|(message, _)| message.contains(fragment))
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "expected a diagnostic containing {fragment:?} for:\n{source}\ngot {diagnostics:?}"
            )
        })
}

/// The instantiation a rejection on an unbounded binder `a` names: an
/// arbitrary type, not any one type.
const ARBITRARY_A: &str = "`a` := an arbitrary type";

/// `(program, the route's own diagnostic, the instantiation it names)`: a call
/// suspended on an operand that an authored binder denotes.
const SUSPENDED_CALLS: &[(&str, &str, &str)] = &[
    // chelis#2216's program.
    (
        "def bad[p: Float](k: p) -> p = cast(numel(k), p)",
        "[TypeMismatch] numel expects tensor input, got f32",
        "`p := f32`",
    ),
    // Tensor-only operands, under each kind of bound. An unbounded binder
    // admits every type, and the call is decided at an arbitrary one.
    (
        "def bad[p: Int](k: p) -> i64 = numel(k)",
        "numel expects tensor input, got i8",
        "`p := i8`",
    ),
    (
        "def bad[a](k: a) -> i64 = numel(k)",
        "`numel` admits only some operand types, so it cannot be applied to an operand of an arbitrary type",
        ARBITRARY_A,
    ),
    (
        "def bad[p: Float](k: p) -> i32 = rank(k)",
        "rank expects tensor input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> i64 = shape(k, 0)",
        "shape expects tensor input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> p = tensor_to_scalar(k)",
        "tensor_to_scalar",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> p = cumsum(k, 0)",
        "cumsum expects tensor input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> p = sort(k, 0)",
        "sort expects tensor input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> List[p] = to_list(k)",
        "to_list",
        "`p := f32`",
    ),
    // A tensor-only check that lives in the dtype-admissibility replay.
    (
        "def bad[p: Float](k: p) -> p = softmax(k, 0)",
        "softmax expects tensor input, got f32",
        "`p := f32`",
    ),
    // Collection and string operands.
    (
        "def bad[p: Float](k: p) -> p = take(k, 1)",
        "take expects List input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> tensor[3, p] = to_tensor(k)",
        "to_tensor expects List input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> i64 = string_len(k)",
        "string_len expects string input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[a](k: a) -> i64 = string_len(k)",
        "`string_len` admits only some operand types, so it cannot be applied to an operand of an arbitrary type",
        ARBITRARY_A,
    ),
    (
        "def bad[p: Float](k: p) -> i32 = fail(k)",
        "fail expects string input, got f32",
        "`p := f32`",
    ),
    // A position that admits one dtype, or one family, rejects a bound that
    // admits more.
    (
        "def bad[p: Float](xs: List[i32], n: p) -> List[i32] = take(xs, n)",
        "take expects integer count, got f32",
        "`p := f32`",
    ),
    (
        "def bad[a](xs: List[i32], n: a) -> List[i32] = take(xs, n)",
        "`take` admits only some operand types, so it cannot be applied to an operand of an arbitrary type",
        ARBITRARY_A,
    ),
    (
        "def bad[p: Float](a: p, b: p) -> List[i64] = range(a, b)",
        "range expects integer arguments, got f32",
        "`p := f32`",
    ),
    (
        "def bad[a](k: a) -> tensor[a] = scalar_to_tensor(k)",
        "`scalar_to_tensor` admits only some operand types, so it cannot be applied to an operand of an arbitrary type",
        ARBITRARY_A,
    ),
    // A lambda parameter identified with the binder only by its application.
    (
        "def bad[p: Float](k: p) -> i64 = (fn (y) -> numel(y))(k)",
        "numel expects tensor input, got f32",
        "`p := f32`",
    ),
    (
        "def bad[p: Float](k: p) -> i64 = (fn (y) -> string_len(y))(k)",
        "string_len expects string input, got f32",
        "`p := f32`",
    ),
];

/// REGRESSION TEST. Every row scored 1 before the fix. Each is now rejected
/// with the route's own diagnostic kind and text, naming the binder and the
/// instantiation where the call fails, with a repair.
#[test]
fn a_call_suspended_on_a_binder_is_decided_at_its_instantiations() {
    for (program, route_diagnostic, instantiation) in SUSPENDED_CALLS {
        let (message, suggestions) = rejection_containing(program, route_diagnostic);
        assert!(
            message.contains("authored type binder")
                && message.contains(&format!("does not at {instantiation}"))
                && message.contains("[04-INF-6]"),
            "the rejection must name the binder and {instantiation}, got {message}"
        );
        assert!(
            suggestions
                .first()
                .is_some_and(|repair| repair.contains("Declare the operand with the type")),
            "the rejection must lead with the binder repair, got {suggestions:?}"
        );
    }
}

/// NEGATIVE PARITY for the test above: an operation that every instantiation
/// admits stays accepted, including one that accepts each member of an `Int`
/// bound in a position no single dtype owns, and a lambda whose operand the
/// application makes a tensor.
#[test]
fn a_call_every_instantiation_admits_stays_accepted() {
    for program in [
        "def ok[p: Int](xs: List[i32], n: p) -> List[i32] = take(xs, n)",
        "def ok[p: Int](a: p, b: p) -> List[i64] = range(a, b)",
        "def ok[p: Float](k: p) -> p = mul(k, k)",
        "def ok[p: Int](k: p) -> p = add(k, k)",
        "def ok[p: Numeric](k: p) -> bool = lt(k, k)",
        "def ok[p: Float](k: p) -> p = div(exp(k), k)",
        "def ok[p: Float](k: key, x: tensor[4, p], r: p) -> tensor[4, p] = dropout(k, x, r)",
        "def ok[p: Float](k: tensor[3, p]) -> i64 = numel(k)",
        "def ok[p: Float](k: tensor[3, p]) -> i64 = (fn (y) -> numel(y))(k)",
    ] {
        accepts(program);
    }
}

/// REGRESSION TEST for round 1's representative instantiation. An unbounded
/// binder was replayed at `()` alone, a type [05-OP-36] admits, so that one
/// witness could not decide `eq` and `neq` for every instantiation. The call is
/// now decided at an arbitrary type: the equality operations admit only some
/// types (not a function), so they are rejected, naming no particular type.
#[test]
fn equality_on_an_unbounded_binder_is_rejected_at_an_arbitrary_type() {
    for operation in ["eq", "neq"] {
        let program = format!("def bad[a](x: a) -> bool = {operation}(x, x)");
        let (message, suggestions) = rejection_containing(
            &program,
            &format!(
                "`{operation}` admits only some operand types, so it cannot be applied to an \
                 operand of an arbitrary type"
            ),
        );
        assert!(
            message.contains(&format!("does not at {ARBITRARY_A}")) && !message.contains("()"),
            "{program}: {message}"
        );
        assert!(
            suggestions
                .first()
                .is_some_and(|repair| repair.contains("declares no dtype-family bound")),
            "{program}: {suggestions:?}"
        );
    }
}

/// NEGATIVE PARITY for the test above, passing before the fix too: an
/// operation that accepts an operand of every type stays accepted on an
/// unbounded binder.
#[test]
fn an_operation_that_accepts_every_type_stays_accepted_on_an_unbounded_binder() {
    for program in [
        "def ok[a](x: a) -> a = debug(x)",
        "def ok[a](x: a) -> unit ! {IO} = print(x)",
        "def ok[a](x: a, xs: List[i32]) -> a = fold(fn (acc: a, y: i32) -> acc, x, xs)",
        "def ok[a](xs: List[a], x: a) -> List[a] = append(xs, x)",
    ] {
        accepts(program);
    }
}

/// NON-REGRESSION LOCK, passing before the fix too: a call with a shape rule of
/// its own reports the operand once. Its dtype replay waits on the same binder
/// and is left to the shape rule's boundary rejection, so the new decision adds
/// no second diagnostic.
#[test]
fn a_call_with_its_own_shape_rule_reports_the_binder_once() {
    for (program, operation) in [
        ("def bad[p: Float](k: p) -> p = sum(k, 0)", "sum"),
        ("def bad[p: Float](k: p) -> p = mean(k, 0)", "mean"),
        ("def bad[p: Float](k: p) -> i64 = count(k, 0)", "count"),
    ] {
        let diagnostics = agreed_diagnostics(program);
        let [(message, _)] = diagnostics.as_slice() else {
            panic!("expected exactly one diagnostic for:\n{program}\ngot {diagnostics:?}");
        };
        assert!(
            message.contains(&format!("unresolved `{operation}` shape obligation")),
            "{program}: {message}"
        );
    }
}

/// REGRESSION TEST for chelis#2158. A tensor `cast_trunc` whose source
/// precision is an `Int` or `Numeric` binder checked at score 1. It is now
/// rejected at the cast, naming the binder and the bound to declare; an
/// unbounded binder is reported against its declaration.
#[test]
fn cast_trunc_rejects_a_binder_source_that_admits_an_integer() {
    for (binders, target) in [
        ("[p: Numeric]", "i64"),
        ("[p: Int]", "i64"),
        ("[p: Numeric, q: Int]", "q"),
    ] {
        let result = if target == "q" { "q" } else { "i64" };
        let program = format!(
            "def tn{binders}(x: tensor[2, p]) -> tensor[2, {result}] = cast_trunc(x, {target})"
        );
        let (message, suggestions) = rejection_containing(
            &program,
            "its source dtype is the declared type parameter `p`",
        );
        assert!(
            message.starts_with(
                "[PrecisionMismatch] `cast_trunc` requires a source of dtype family `Float`"
            ),
            "{program}: {message}"
        );
        assert!(
            suggestions
                .iter()
                .any(|hint| hint.contains("Declare `p: Float`")),
            "{program}: {suggestions:?}"
        );
    }
    // The scalar source under an authored bound is named the same way; before
    // the fix it was rejected only as "cast requires tensor or prim type".
    rejection_containing(
        "def bad[p: Int](k: p) -> i64 = cast_trunc(k, i64)",
        "its source dtype is the declared type parameter `p`",
    );
    let (message, suggestions) = rejection_containing(
        "def tn[p](x: tensor[2, p]) -> tensor[2, i64] = cast_trunc(x, i64)",
        "requires dtype family `Float` in its body",
    );
    assert!(message.contains("unbounded authored variable"), "{message}");
    assert!(
        suggestions.iter().any(|hint| hint.contains("`p: Float`")),
        "{suggestions:?}"
    );
}

/// REGRESSION TEST for round 1's over-rejection. [04-NUM-14] and [05-OP-63]
/// admit a `bool` source for `cast`. The variable-target arm required a
/// `Numeric` source of a lambda parameter later bound to `bool`, and rejected
/// a `bool` scalar outright; both are accepted now, at either target family.
#[test]
fn a_bool_source_casts_to_a_binder_target() {
    for family in ["Int", "Float"] {
        accepts(&format!(
            "def f[q: {family}](b: bool) -> q = (fn (y) -> cast(y, q))(b)"
        ));
        accepts(&format!("def f[q: {family}](b: bool) -> q = cast(b, q)"));
    }
}

/// NEGATIVE PARITY for the test above: [05-OP-6] still refuses a `bool`
/// `cast_trunc` source, and an unbounded authored source, which also denotes
/// types no cast admits, is still held to `Numeric`. Both were rejected before
/// the repair too, but the `bool` source was refused as a non-numeric scalar;
/// it now reaches [05-OP-6]'s own rule.
#[test]
fn a_bool_trunc_source_and_an_unbounded_cast_source_stay_rejected() {
    rejection_containing(
        "def bad[q: Int](b: bool) -> q = cast_trunc(b, q)",
        "`cast_trunc` requires a float source and an integer target ([05-OP-6])",
    );
    rejection_containing(
        "def bad[a, q: Int](x: a) -> q = cast(x, q)",
        "declared type parameter `a` of `bad` requires dtype family `Numeric`",
    );
}

/// REGRESSION TEST for round 1's duplicate report. A lambda parameter that the
/// application identifies with a `Numeric` binder carried `cast_trunc`'s float
/// requirement, and the defect was reported against the binder and again
/// against the parameter. It is reported once, against the binder.
#[test]
fn a_binder_reached_through_a_lambda_is_reported_once() {
    let program = "def bad[p: Numeric](k: tensor[2, p]) -> tensor[2, i64] = \
                   (fn (y) -> cast_trunc(y, i64))(k)";
    let diagnostics = agreed_diagnostics(program);
    let [(message, _)] = diagnostics.as_slice() else {
        panic!("expected exactly one diagnostic for:\n{program}\ngot {diagnostics:?}");
    };
    assert!(
        message.contains("declared type parameter `p` of `bad` requires dtype family `Float`"),
        "{message}"
    );
}

/// NEGATIVE PARITY for chelis#2158: the `Float` control, scalar and tensor,
/// concrete and variable target, stays accepted.
#[test]
fn cast_trunc_from_a_float_binder_stays_accepted() {
    for program in [
        "def tn[p: Float](x: tensor[2, p]) -> tensor[2, i64] = cast_trunc(x, i64)",
        "def tn[p: Float, q: Int](x: tensor[2, p]) -> tensor[2, q] = cast_trunc(x, q)",
        "def tn[p: Float](k: p) -> i64 = cast_trunc(k, i64)",
        "def tn[p: Float, q: Int](k: p) -> q = cast_trunc(k, q)",
        "def tn[p: Numeric, q: Float](k: p) -> q = cast(k, q)",
    ] {
        accepts(program);
    }
}

const NUMERIC_ID: &str = "def id2[q: Numeric](x: tensor[2, q]) -> tensor[2, q] = x\n";

/// REGRESSION TEST. The same `None` admitted an inference variable too: a
/// precision that a later binding made `i32` checked at score 1 and panicked in
/// `eval`. The float requirement now travels with the variable.
#[test]
fn a_precision_variable_later_bound_to_an_integer_is_rejected() {
    let program = format!(
        "{NUMERIC_ID}def tn(x: tensor[2, i32]) -> tensor[2, i64] = \
         (fn (y) -> cast_trunc(id2(y), i64))(x)"
    );
    rejection_containing(
        &program,
        "type variable bounded by dtype family `Float` (the active float dtypes) cannot be \
         instantiated at `i32`",
    );
    accepts(&format!(
        "{NUMERIC_ID}def tn(x: tensor[2, f32]) -> tensor[2, i64] = \
         (fn (y) -> cast_trunc(id2(y), i64))(x)"
    ));
}

/// REGRESSION TEST for an over-rejection with the same cause. The
/// variable-target arm read the source precision's restriction at the cast,
/// so a precision a later binding makes `f32` was rejected. It is accepted now,
/// and the same program over `i32` is still rejected.
#[test]
fn a_variable_target_trunc_decides_a_late_bound_precision_either_way() {
    accepts(&format!(
        "{NUMERIC_ID}def tn[r: Int](x: tensor[2, f32]) -> tensor[2, r] = \
         (fn (y) -> cast_trunc(id2(y), r))(x)"
    ));
    rejection_containing(
        &format!(
            "{NUMERIC_ID}def tn[r: Int](x: tensor[2, i32]) -> tensor[2, r] = \
             (fn (y) -> cast_trunc(id2(y), r))(x)"
        ),
        "cannot be instantiated at `i32`",
    );
}

/// REGRESSION TEST. The scalar variable-target arm applied no [05-OP-6] rule:
/// an integer source, a float target, and a source binder that admits an
/// integer or a non-numeric type all checked at score 1.
#[test]
fn a_scalar_variable_target_cast_applies_the_trunc_pair_rule() {
    for program in [
        "def bad[q: Int](k: i32) -> q = cast_trunc(k, q)",
        "def bad[q: Float](k: f32) -> q = cast_trunc(k, q)",
        "def bad[p: Float, q: Numeric](k: p) -> q = cast_trunc(k, q)",
    ] {
        rejection_containing(
            program,
            "`cast_trunc` requires a float source and an integer target ([05-OP-6])",
        );
    }
    for (program, fragment) in [
        (
            "def bad[p: Int, q: Int](k: p) -> q = cast_trunc(k, q)",
            "its source dtype is the declared type parameter `p`",
        ),
        (
            "def bad[p: Numeric, q: Int](k: p) -> q = cast_trunc(k, q)",
            "its source dtype is the declared type parameter `p`",
        ),
        (
            "def bad[a, q: Int](k: a) -> q = cast_trunc(k, q)",
            "declared type parameter `a` of `bad` requires dtype family `Float`",
        ),
        (
            "def bad[a, q: Float](k: a) -> q = cast(k, q)",
            "declared type parameter `a` of `bad` requires dtype family `Numeric`",
        ),
    ] {
        rejection_containing(program, fragment);
    }
}
