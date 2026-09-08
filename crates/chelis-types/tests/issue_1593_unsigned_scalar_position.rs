//! chelis#1593 at the checker: an unsigned dtype spelling in a SCALAR type
//! position is rejected with the `spec/04-type-system.md` §1.1.1 / §1.1.2
//! diagnostic, the same one the cast target and the `tensor[...]` precision
//! slot already produce.
//!
//! Before this, the scalar `TypeExpr::Named` desugar arm made any lowercase
//! non-primitive name a `(t-var {} <name>)`, so `def f(x: u8) -> u8 = x` typed
//! as `forall u8. u8 -> u8`, checked clean, and scored 1.0: a program written
//! against a reserved and rejected dtype was accepted with an empty error
//! vector. In the positions that did reject (`deftype` field, `typealias`
//! body, value annotation) the message was "undeclared type variable `u8`",
//! which names the wrong defect and offers no remedy.
//!
//! The repair also covers two neighbours of that arm. An explicit `[..]`
//! quantifier list does not rebind a reserved spelling, in a scalar position
//! or in a tensor precision slot, because §5.8.1 states the rule on the
//! category. And the arm consults the tensor precision slot's predicate pair,
//! which rejects the other §1.1.1 reserved-but-deferred names as well as the
//! unsigned ones, so `complex64` in a scalar position reaches its own §1.1.1
//! diagnostic instead of scoring 1.0.
//!
//! Test labels are recorded in each function's doc comment. The suite claims
//! the eight §1.1.2 spellings in the six scalar positions it enumerates below,
//! those eight again under an explicit binder list in a scalar and a tensor
//! precision position, and the deferred family at `complex64` and `int4`
//! only; it locks the two `t-prim` positions that already worked. No other
//! type position and no other reserved spelling is claimed.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;

/// The eight §1.1.2 spellings, exactly as `is_unsigned_dtype_name` lists them.
const UNSIGNED: [&str; 8] = [
    "u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64",
];

/// Two of the other §1.1.1 reserved-but-deferred spellings, which ride the
/// same repair because the scalar arm consults the tensor precision slot's
/// predicate pair. Claimed at these two names only.
const DEFERRED: [&str; 2] = ["complex64", "int4"];

/// The deferred family's own §1.1.1 diagnostic, whose wording differs from the
/// unsigned one.
fn assert_deferred_diagnostic(source: &str, name: &str, position: &str) {
    let messages = messages(source);
    assert!(
        messages
            .iter()
            .any(|message| message.contains(&format!("`{name}`"))
                && message.contains("is reserved but deferred")
                && message.contains("spec/04-type-system.md §1.1.1")),
        "the diagnostic for `{name}` in {position} must name the spelling and \
         cite §1.1.1; got: {messages:?}"
    );
}

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(source: &str) -> Vec<String> {
    let deep = surf_to_deep(source);
    match check_ir_program(&deep) {
        Ok(_) => Vec::new(),
        Err(report) => report
            .errors
            .iter()
            .map(|error| error.message.clone())
            .collect(),
    }
}

/// The three things the diagnostic owes a reader per §1.1.1 / §1.1.2: the
/// offending spelling, the governing sections, and a remedy.
fn assert_unsigned_diagnostic(source: &str, name: &str, position: &str) {
    let messages = messages(source);
    assert!(
        !messages.is_empty(),
        "`{name}` in {position} must be rejected, not accepted with an empty \
         error vector"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains(&format!("`{name}`"))
                && message.contains("unsigned integer types are deferred")),
        "the diagnostic for `{name}` in {position} must name the spelling and \
         say why it is rejected; got: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("spec/04-type-system.md §1.1.1")
                && message.contains("§1.1.2")),
        "the diagnostic for `{name}` in {position} must cite §1.1.1 and \
         §1.1.2; got: {messages:?}"
    );
    assert!(
        messages.iter().any(|message| message
            .contains("active set: f32, f64, bf16, f16, bool, int8, int16, int32, int64")),
        "the diagnostic for `{name}` in {position} must offer the active \
         primitive set as the remedy; got: {messages:?}"
    );
}

/// REGRESSION test. The named instance of chelis#1593: `def f(x: u8) -> u8 = x`
/// scored 1.0 with no error at all.
#[test]
fn a_scalar_parameter_and_return_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"),
            name,
            "a scalar parameter and return",
        );
    }
}

/// REGRESSION test. A standalone `sig` also scored 1.0.
#[test]
fn a_sig_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (f)\nsig f: {name} -> {name}\ndef f(x) = x\n"),
            name,
            "a sig",
        );
    }
}

/// REGRESSION test. A record field rejected before, with the wrong message
/// ("undeclared type variable"). The claim here is the message, not the
/// rejection.
#[test]
fn a_record_field_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (Box)\ntype Box = | Box {{ v: {name} }}\n"),
            name,
            "a record field",
        );
    }
}

/// REGRESSION test. Same shape as the record field, positional variant form.
#[test]
fn a_positional_adt_field_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (Wrap)\ntype Wrap = | W({name})\n"),
            name,
            "a positional ADT field",
        );
    }
}

/// REGRESSION test. A `typealias` body.
#[test]
fn a_typealias_body_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (S)\ntype S = {name}\n"),
            name,
            "a typealias body",
        );
    }
}

/// REGRESSION test. A lambda parameter annotation: a value-position annotation
/// with no enclosing quantifier scope, which §5.8.1's last sentence sends to
/// "the existing rule".
#[test]
fn a_lambda_annotation_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (f)\ndef f() -> int32 = (fn (x: {name}) -> 1i32)(1i32)\n"),
            name,
            "a lambda parameter annotation",
        );
    }
}

/// DISPOSITION LOCK. Green in both states. The cast target is the position that
/// already produced this diagnostic; it must keep producing exactly it.
#[test]
fn a_cast_target_still_carries_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (f)\ndef f() -> int32 = cast(1i32, {name})\n"),
            name,
            "a cast target",
        );
    }
}

/// DISPOSITION LOCK. Green in both states. The `tensor[...]` precision slot is
/// the positive precedent this change copies.
#[test]
fn a_tensor_precision_slot_still_carries_the_spec_diagnostic() {
    for name in UNSIGNED {
        let messages = messages(&format!(
            "module P.M\nexport (f)\nsig f: tensor[d, {name}] -> tensor[d, {name}]\n\
             def f(x) = x\n"
        ));
        assert!(
            messages.iter().any(
                |message| message.contains("unsigned integer types are deferred")
                    && message.contains("tensor element")
            ),
            "the tensor precision slot must keep its own tensor-flavoured \
             §1.1.1 diagnostic for `{name}`; got: {messages:?}"
        );
    }
}

/// DISPOSITION LOCK. Green in both states, and the neighbour this change must
/// not disturb: a lowercase name that is not a reserved dtype spelling is still
/// an implicitly quantified type variable and the program still checks clean.
#[test]
fn a_genuine_lowercase_name_still_quantifies_and_checks_clean() {
    for source in [
        "module P.M\nexport (f)\ndef f(x: a) -> a = x\n",
        "module P.M\nexport (f)\nsig f: a -> a\ndef f(x) = x\n",
    ] {
        let messages = messages(source);
        assert!(
            messages.is_empty(),
            "a genuine lowercase type variable must still check clean: \
             {source}\n{messages:?}"
        );
    }
}

/// DISPOSITION LOCK. Green in both states. Negative parity for the predicate:
/// the SIGNED spellings, including chelis#1587's short aliases, must not match
/// the unsigned family. `int8`..`int64` share the suffix digits and `int8`
/// contains no `uint8` only because the check is exact.
#[test]
fn signed_spellings_do_not_match_the_unsigned_family() {
    for name in [
        "int8", "int16", "int32", "int64", "i8", "i16", "i32", "i64", "f32", "f64", "bool",
    ] {
        let messages = messages(&format!(
            "module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"
        ));
        assert!(
            messages.is_empty(),
            "`{name}` is an active primitive (or an accepted input spelling \
             for one) and must still be accepted: {messages:?}"
        );
    }
}

/// REGRESSION test. An explicit `[..]` quantifier list does not rebind a
/// reserved spelling, per §5.8.1's category rule. Both the scalar and the
/// tensor precision position, and both reserved families.
#[test]
fn an_explicit_binder_does_not_rebind_a_reserved_name() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!("module P.M\nexport (f)\ndef f[{name}](x: {name}) -> {name} = x\n"),
            name,
            "an explicit binder in a scalar position",
        );
        let tensor = messages(&format!(
            "module P.M\nexport (f)\n\
             def f[{name}](x: tensor[3, {name}]) -> tensor[3, {name}] = x\n"
        ));
        assert!(
            tensor
                .iter()
                .any(|message| message.contains("unsigned integer types are deferred")),
            "an explicit binder must not rebind `{name}` in a tensor precision \
             slot; got: {tensor:?}"
        );
    }
    for name in DEFERRED {
        assert_deferred_diagnostic(
            &format!("module P.M\nexport (f)\ndef f[{name}](x: {name}) -> {name} = x\n"),
            name,
            "an explicit binder in a scalar position",
        );
        assert_deferred_diagnostic(
            &format!(
                "module P.M\nexport (f)\n\
                 def f[{name}](x: tensor[3, {name}]) -> tensor[3, {name}] = x\n"
            ),
            name,
            "an explicit binder in a tensor precision slot",
        );
    }
}

/// DISPOSITION LOCK. Green in both states, and the positive control for the
/// row above: an explicit binder with an ordinary lowercase name still binds
/// and the program still checks clean, in both positions.
#[test]
fn an_explicit_binder_with_an_ordinary_name_still_checks_clean() {
    for source in [
        "module P.M\nexport (f)\ndef f[p](x: p) -> p = x\n",
        "module P.M\nexport (f)\ndef f[p](x: tensor[3, p]) -> tensor[3, p] = x\n",
    ] {
        let messages = messages(source);
        assert!(
            messages.is_empty(),
            "an ordinary explicit binder must still bind: {source}\n{messages:?}"
        );
    }
}

/// REGRESSION test. The reserved-but-deferred family in a scalar position.
/// Before, `def f(x: complex64) -> complex64 = x` scored 1.0 with an empty
/// error vector, exactly as the unsigned family did.
#[test]
fn the_deferred_family_is_rejected_in_a_scalar_position() {
    for name in DEFERRED {
        assert_deferred_diagnostic(
            &format!("module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"),
            name,
            "a scalar parameter and return",
        );
    }
}
