//! chelis#2219: `copy` is generic in its operand's type.
//!
//! `spec/04-type-system.md` section 8.2: "`copy(x)` accepts either owned `T`
//! or borrowed `&T` and yields a fresh owned value". Section 8.3 prescribes an
//! authored `copy(x)` for a destructured component's fan-out, [04-LIN-4]
//! makes a copy the only way a borrowed argument becomes an owned result, and
//! [04-LIN-9] refuses `copy` for a key-carrying value, the one exception.
//! Before the fix the checker admitted only a tensor or a borrowed tensor and
//! refused every other operand ("copy argument 1: expected tensor, got ...").

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_linearity, check_typed_program};

/// Type errors, then linearity errors, of a Surf program.
fn check_errors(source: &str) -> Vec<CheckError> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(checked) => match check_linearity(&checked) {
            Ok(_) => Vec::new(),
            Err(errors) => errors,
        },
        Err(result) => result.errors,
    }
}

#[track_caller]
fn assert_checks(source: &str, what: &str) {
    let errors = check_errors(source);
    assert!(
        errors.is_empty(),
        "{what}: expected no errors; got {errors:?}"
    );
}

/// The one error a program reports, which must have exactly `kind` and
/// `message`.
#[track_caller]
fn assert_single_error(source: &str, kind: CheckErrorKind, message: &str, what: &str) {
    let errors = check_errors(source);
    assert!(
        matches!(
            errors.as_slice(),
            [error] if std::mem::discriminant(&error.kind) == std::mem::discriminant(&kind)
                && error.message == message
        ),
        "{what}: expected exactly one {kind:?} `{message}`; got {errors:?}"
    );
}

const LIN: &str = "type Lin[i] =\n  | Lin { w: tensor[i, f32] }\n\
def weight(p: Lin[2]) -> tensor[2, f32] = match p with {\n  | Lin { w } => w\n}\n";

#[test]
fn copy_of_an_owned_value_of_any_kind_yields_that_type() {
    for (ty, value) in [
        (
            "List[tensor[2, f32]]",
            "[to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])]",
        ),
        (
            "Dict[string, tensor[2, f32]]",
            "dict_of([(\"a\", to_tensor([1.0f32, 2.0f32]))])",
        ),
        ("Lin[2]", "Lin { w: to_tensor([1.0f32, 2.0f32]) }"),
        (
            "(tensor[1, f32], tensor[1, f32])",
            "(to_tensor([1.0f32]), to_tensor([2.0f32]))",
        ),
        ("Option[tensor[1, f32]]", "Some(to_tensor([1.0f32]))"),
        ("string", "\"ab\""),
        ("i64", "3i64"),
        ("unit", "()"),
    ] {
        assert_checks(
            &format!("{LIN}def probe() -> {ty} = {{\n  v: {ty} = {value}\n  copy(v)\n}}\n"),
            &format!("copy of a {ty}"),
        );
    }
}

#[test]
fn copy_result_is_the_operand_type_not_a_fresh_one() {
    // A declared result that disagrees with the operand is a signature
    // mismatch naming the copied type, so copy decided it, rather than
    // leaving a variable any declaration would satisfy.
    assert_single_error(
        &format!("{LIN}def probe(p: Lin[2]) -> Lin[3] = copy(p)\n"),
        CheckErrorKind::DimensionMismatch,
        "def 'probe' body doesn't match declared signature: expected `(Lin 2) -> Lin 3`, \
         got `(Lin 2) -> Lin 2`",
        "copy of a Lin[2] declared Lin[3]",
    );
}

#[test]
fn copy_of_a_borrow_yields_the_owned_referent() {
    assert_checks(
        "def owned(xs: &List[tensor[2, f32]]) -> List[tensor[2, f32]] = copy(xs)\n",
        "copy of a borrowed List parameter",
    );
    assert_checks(
        &format!("{LIN}def owned(p: Lin[2]) -> Lin[2] = copy(&p)\n"),
        "copy of an explicit borrow of an ADT",
    );
    assert_checks(
        &format!(
            "{LIN}def own(p: &Lin[2]) -> Lin[2] = copy(p)\n\
             def probe() -> tensor[2, f32] = {{\n\
             \x20 p = Lin {{ w: to_tensor([1.0f32, 2.0f32]) }}\n\
             \x20 q = own(p)\n\
             \x20 add(weight(q), weight(p))\n\
             }}\n"
        ),
        "copy of a borrowed ADT parameter, the caller keeping the original",
    );
    // The result is the owned referent, never the borrow.
    assert_single_error(
        "def owned(xs: &List[tensor[2, f32]]) -> &List[tensor[2, f32]] = copy(xs)\n",
        CheckErrorKind::TypeMismatch,
        "def 'owned' body doesn't match declared signature: expected \
         `(&List tensor[2, f32]) -> &List tensor[2, f32]`, got \
         `(&List tensor[2, f32]) -> List tensor[2, f32]`",
        "copy of a borrow declared as a borrow",
    );
}

#[test]
fn a_copied_borrow_reaches_an_owned_parameter() {
    // spec/04 section 8.2: `&T` where owned `T` is expected is a type error
    // unless the program writes `copy(x)`.
    let grow = "def grow(xs: List[tensor[2, f32]]) -> List[tensor[2, f32]] = \
                append(xs, to_tensor([5.0f32, 6.0f32]))\n";
    assert_checks(
        &format!(
            "{grow}def regrow(xs: &List[tensor[2, f32]]) -> List[tensor[2, f32]] = grow(copy(xs))\n"
        ),
        "an owned parameter fed a copied borrow",
    );
    // NEGATIVE TWIN: without the copy, the borrow is refused.
    let errors = check_errors(&format!(
        "{grow}def regrow(xs: &List[tensor[2, f32]]) -> List[tensor[2, f32]] = grow(xs)\n"
    ));
    assert!(
        !errors.is_empty(),
        "a borrowed List passed to an owned parameter must be refused"
    );
}

#[test]
fn copy_does_not_consume_its_operand() {
    // Linearity: the original stays usable after its copy is consumed, for an
    // ADT, a List and a Dict alike.
    assert_checks(
        &format!(
            "{LIN}def grow(xs: List[tensor[2, f32]]) -> List[tensor[2, f32]] = \
             append(xs, to_tensor([5.0f32, 6.0f32]))\n\
             def probe() -> i64 = {{\n\
             \x20 p = Lin {{ w: to_tensor([1.0f32, 2.0f32]) }}\n\
             \x20 a = weight(copy(p))\n\
             \x20 b = weight(p)\n\
             \x20 xs = [to_tensor([1.0f32, 2.0f32])]\n\
             \x20 ys = grow(copy(xs))\n\
             \x20 zs = grow(xs)\n\
             \x20 d = dict_of([(\"a\", to_tensor([1.0f32, 2.0f32]))])\n\
             \x20 e = dict_insert(copy(d), \"b\", to_tensor([3.0f32, 4.0f32]))\n\
             \x20 f = dict_insert(d, \"c\", add(a, b))\n\
             \x20 add(add(len(ys), len(zs)), add(len(e), len(f)))\n\
             }}\n"
        ),
        "each copy leaves its operand live",
    );
}

#[test]
fn a_copied_component_fans_out() {
    // spec/04 section 8.3: a destructured component is not copy-repaired;
    // the authored `copy(x)` on the earlier use is the repair, whatever the
    // component's type.
    let program = |first: &str| {
        format!(
            "def grow(xs: List[tensor[2, f32]]) -> List[tensor[2, f32]] = \
             append(xs, to_tensor([5.0f32, 6.0f32]))\n\
             def probe() -> i64 = {{\n\
             \x20 (xs, n) = ([to_tensor([1.0f32, 2.0f32])], 1i64)\n\
             \x20 a = grow({first})\n\
             \x20 b = grow(xs)\n\
             \x20 add(add(len(a), len(b)), n)\n\
             }}\n"
        )
    };
    assert_checks(&program("copy(xs)"), "a copied List component");
    let errors = check_errors(&program("xs"));
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UseAfterConsume)),
        "an uncopied component fan-out stays a hard error; got {errors:?}"
    );
}

#[test]
fn a_deferred_copy_decides_as_the_eager_one() {
    // The lambda parameter is unresolved when `copy(v)` is inferred, so the
    // decision is suspended and replayed once the application binds it.
    for (ty, ret) in [
        ("List[tensor[2, f32]]", "List[tensor[2, f32]]"),
        ("&List[tensor[2, f32]]", "List[tensor[2, f32]]"),
        ("Lin[2]", "Lin[2]"),
        ("&Lin[2]", "Lin[2]"),
        ("string", "string"),
    ] {
        let eager = check_errors(&format!("{LIN}def probe(v: {ty}) -> {ret} = copy(v)\n"));
        let deferred = check_errors(&format!(
            "{LIN}def apply_it(f: ({ty}) -> {ret}, t: {ty}) -> {ret} = f(t)\n\
             def probe(t: {ty}) -> {ret} = apply_it(fn (v) -> copy(v), t)\n"
        ));
        assert!(
            eager.is_empty() && deferred.is_empty(),
            "copy of {ty} -> {ret}: eager {eager:?} / deferred {deferred:?}"
        );
    }
}

#[test]
fn copy_of_a_key_carrying_value_is_refused() {
    // [04-LIN-9]: no copy applies to a key-carrying value.
    assert_single_error(
        "def probe() -> i64 = {\n  k = key_from_seed(1i64)\n  k2 = copy(k)\n  1i64\n}\n",
        CheckErrorKind::KeyReuse,
        "key-carrying variable `k` cannot be copied at surf:61..62: a key is used at most \
         once and has no read that leaves it live ([04-LIN-9])",
        "copy of a key",
    );
    assert_single_error(
        "type Keyed =\n  | Keyed { k: key, n: i64 }\n\
         def probe() -> i64 = {\n  s = Keyed { k: key_from_seed(1i64), n: 1i64 }\n  t = copy(s)\n  1i64\n}\n",
        CheckErrorKind::KeyReuse,
        "key-carrying variable `s` cannot be copied at surf:124..125: a key is used at most \
         once and has no read that leaves it live ([04-LIN-9])",
        "copy of a key-carrying ADT",
    );
    // An operand that is not a variable is named as a value.
    assert_single_error(
        "def probe() -> i64 = {\n  k2 = copy(key_from_seed(1i64))\n  1i64\n}\n",
        CheckErrorKind::KeyReuse,
        "a key-carrying value cannot be copied at surf:35..54: a key is used at most once and \
         has no read that leaves it live ([04-LIN-9])",
        "copy of a key-carrying call",
    );
    let errors = check_errors(
        "def probe() -> i64 = {\n  k = (key_from_seed(1i64), 2i64)\n  k2 = copy(k)\n  1i64\n}\n",
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("cannot be copied")
                && error.message.contains("[04-LIN-9]")),
        "copy of a key-carrying tuple must be refused; got {errors:?}"
    );
}

#[test]
fn copy_of_a_never_resolved_operand_is_refused() {
    // A declared parameter may be instantiated at a borrow, which changes
    // what the copy yields, so an operand whose type never settles is
    // refused, naming the parameter.
    assert_single_error(
        "def go[t](x: t) -> i32 = {\n  g = copy(x)\n  1i32\n}\n",
        CheckErrorKind::TypeMismatch,
        "copy requires its operand's type to be determined at the call, because whether \
         it is a borrow decides the result (spec/04-type-system.md section 8.2), got `t`; \
         argument 1: expected an operand of determined type, got `t`",
        "copy of a never-resolved parameter",
    );
    // NEGATIVE TWIN: declared as a borrow, the copied type is decided.
    assert_checks(
        "def dup[t](x: &t) -> t = copy(x)\n",
        "copy of a borrowed type parameter",
    );
}

#[test]
fn a_copied_function_is_fenced_as_its_operand() {
    // The copy of a function value is that value, so the core-transform
    // fence (chelis#1952, #1954) classifies a copied target exactly as its
    // operand: a copy of a top-level function or of a shadowing local is
    // refused where the operand is, and a copy of a direct, unshadowed
    // declaration is admitted where the declaration is.
    let fence = "the core transform fragment rejects this `grad` target";
    let sq = "def sq(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0i32))\n";
    for (body, what) in [
        (
            "def gq() -> tensor[2, f32] = {\n  sq = fn (x: tensor[2, f32]) -> \
             tensor_to_scalar(sum(x, 0i32))\n  grad(copy(sq))(to_tensor([1.0f32, 2.0f32]))\n}\n",
            "grad(copy(sq)) of a shadowing local",
        ),
        (
            "def gq() -> tensor[2, f32] = {\n  h = copy(sq)\n  \
             grad(h)(to_tensor([1.0f32, 2.0f32]))\n}\n",
            "a copied alias of a top-level function",
        ),
    ] {
        let errors = check_errors(&format!("{sq}{body}"));
        assert!(
            errors.iter().any(|error| error.message.contains(fence)),
            "{what}: expected the core-transform fence; got {errors:?}"
        );
    }
    assert_checks(
        &format!("{sq}def gq() -> tensor[2, f32] = grad(copy(sq))(to_tensor([1.0f32, 2.0f32]))\n"),
        "grad(copy(sq)) of an unshadowed declaration",
    );
}
