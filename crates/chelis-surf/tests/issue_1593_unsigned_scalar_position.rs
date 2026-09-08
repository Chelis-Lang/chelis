//! chelis#1593: an unsigned dtype spelling in a SCALAR type position must
//! desugar to `(t-prim {} <name>)` so the type checker's
//! `spec/04-type-system.md` §1.1.1 / §1.1.2 rejection fires, never to an
//! implicitly quantified `(t-var {} <name>)`.
//!
//! `spec/04-type-system.md` §5.8.1 states the rule: the §1.1.2 unsigned
//! spellings "are explicitly excluded so they reach the type-checker's §1.1.2
//! rejection path with a precise diagnostic, not silently absorbed as
//! quantifiers". The exclusion reached the `tensor[...]` precision slot, whose
//! non-primitive fall-through is `t-prim`, but not the scalar `TypeExpr::Named`
//! arm, whose fall-through is `t-var`. `def f(x: u8) -> u8 = x` therefore typed
//! as `forall u8. u8 -> u8` and scored 1.0.
//!
//! Same class as chelis#1587 (a lowercase non-type name silently quantified),
//! repaired the opposite way: #1587's short signed spellings MAP to their
//! primitives, and these MAP TO NOTHING - they reach the rejection.
//!
//! Two neighbours belong to the same arm. An explicit `[..]` quantifier list
//! must not rebind a reserved spelling either, in the scalar position or in
//! the tensor precision slot, so the reserved row sits ABOVE the quantifier
//! check in both; §5.8.1 states the rule on the category, and the primitive
//! row already outranks a binder for the same reason. And the arm consults one
//! predicate over BOTH §1.1.1 reserved families, matching the tensor precision
//! slot, which rejects the deferred names too, so the two positions recognise
//! exactly the same names.
//!
//! Test labels are recorded in each function's doc comment.

use chelis_deep::parser::parse_and_stamp_file;
use chelis_deep::printer::print_canonical_flat;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_source;
use chelis_surf::parser::parse_str;

/// The eight §1.1.2 spellings, exactly as `UNSIGNED_DTYPE_NAMES` in
/// `crates/chelis-surf/src/desugar.rs` and `is_unsigned_dtype_name` in
/// `crates/chelis-types/src/infer/expr_record.rs` list them.
const UNSIGNED: [&str; 8] = [
    "u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64",
];

/// Two of the other §1.1.1 reserved-but-deferred spellings. The scalar arm
/// consults one predicate over both families because the tensor precision slot
/// does, so these ride the same repair and are claimed at these two names only.
const DEFERRED: [&str; 2] = ["complex64", "int4"];

fn deep_of(source: &str) -> String {
    let decls = parse_str(source).expect("parse");
    print_canonical_flat(&desugar_program(&decls))
}

/// `(t-prim {} name)` and `(t-var {} name)` renderings. The `{} ` prefix makes
/// the match exact: `u8` never matches inside `uint8`.
fn prim_of(name: &str) -> String {
    format!("(t-prim {{}} {name})")
}

fn tvar_of(name: &str) -> String {
    format!("(t-var {{}} {name})")
}

fn assert_reaches_rejection(deep: &str, name: &str, position: &str) {
    assert!(
        deep.contains(&prim_of(name)),
        "`{name}` in {position} must desugar to a `t-prim` so the §1.1.2 \
         rejection can fire: {deep}"
    );
    assert!(
        !deep.contains(&tvar_of(name)),
        "`{name}` in {position} must not be absorbed as a quantified type \
         variable: {deep}"
    );
}

/// REGRESSION test. The named instance of chelis#1593: a scalar parameter and
/// return annotation.
#[test]
fn every_unsigned_name_in_a_scalar_parameter_and_return_reaches_the_rejection() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"
        ));
        assert_reaches_rejection(&deep, name, "a scalar parameter and return");
    }
}

/// REGRESSION test. A standalone `sig` is the implicit-quantifier context
/// §5.8.1's sentence names directly.
#[test]
fn every_unsigned_name_in_a_sig_reaches_the_rejection() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\nsig f: {name} -> {name}\ndef f(x) = x\n"
        ));
        assert_reaches_rejection(&deep, name, "a sig");
    }
}

/// REGRESSION test. A record field of a `deftype` variant.
#[test]
fn every_unsigned_name_in_a_record_field_reaches_the_rejection() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (Box)\ntype Box = | Box {{ v: {name} }}\n"
        ));
        assert_reaches_rejection(&deep, name, "a record field");
    }
}

/// REGRESSION test. A positional constructor field of a `deftype` variant.
#[test]
fn every_unsigned_name_in_a_positional_adt_field_reaches_the_rejection() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (Wrap)\ntype Wrap = | W({name})\n"
        ));
        assert_reaches_rejection(&deep, name, "a positional ADT field");
    }
}

/// REGRESSION test. A `typealias` body.
#[test]
fn every_unsigned_name_in_a_typealias_body_reaches_the_rejection() {
    for name in UNSIGNED {
        let deep = deep_of(&format!("module P.M\nexport (S)\ntype S = {name}\n"));
        assert_reaches_rejection(&deep, name, "a typealias body");
    }
}

/// REGRESSION test. A lambda parameter annotation, a value-position annotation
/// with no enclosing quantifier scope.
#[test]
fn every_unsigned_name_in_a_lambda_annotation_reaches_the_rejection() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f() -> int32 = (fn (x: {name}) -> 1i32)(1i32)\n"
        ));
        assert_reaches_rejection(&deep, name, "a lambda parameter annotation");
    }
}

/// DISPOSITION LOCK. Green in both states. The tensor precision slot is the
/// positive precedent this change copies: its non-primitive fall-through was
/// already `t-prim`, and must stay that way.
#[test]
fn the_tensor_precision_slot_still_emits_t_prim() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\nsig f: tensor[d, {name}] -> tensor[d, {name}]\n\
             def f(x) = x\n"
        ));
        assert_reaches_rejection(&deep, name, "a tensor precision slot");
    }
}

/// DISPOSITION LOCK. Green in both states. The cast target already reached the
/// §1.1.1 diagnostic through its own `t-prim` fall-through.
#[test]
fn a_cast_target_still_emits_t_prim() {
    for name in UNSIGNED {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f() -> int32 = cast(1i32, {name})\n"
        ));
        assert_reaches_rejection(&deep, name, "a cast target");
    }
}

/// DISPOSITION LOCK. Green in both states, and the neighbour this change must
/// not disturb: a lowercase name that is NOT a reserved dtype spelling is still
/// an implicitly quantified type variable per §5.8.1.
#[test]
fn a_genuine_lowercase_name_still_quantifies() {
    for source in [
        "module P.M\nexport (f)\ndef f(x: a) -> a = x\n",
        "module P.M\nexport (f)\nsig f: a -> a\ndef f(x) = x\n",
    ] {
        let deep = deep_of(source);
        assert!(
            deep.contains(&tvar_of("a")),
            "`a` must stay a quantified type variable: {deep}"
        );
        assert!(
            !deep.contains(&prim_of("a")),
            "`a` must not become a primitive: {deep}"
        );
    }
}

/// DISPOSITION LOCK. Green in both states. Unlike chelis#1587's signed
/// spellings, an unsigned spelling maps to no canonical primitive, so the
/// canonical formatter must leave the user's text alone rather than rewrite it
/// to a name the language does not have.
#[test]
fn the_formatter_leaves_an_unsigned_spelling_unchanged() {
    for name in UNSIGNED {
        let source = format!("module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n");
        let once = format_source(&source).expect("format once");
        assert!(
            once.contains(name),
            "the formatter must preserve `{name}`: {once}"
        );
        let twice = format_source(&once).expect("format twice");
        assert_eq!(once, twice, "formatter idempotence on `{name}`");
    }
}

/// REGRESSION test. §5.8.1's rule is stated on the category, so an explicit
/// `[..]` quantifier list does not rebind a reserved spelling. Below the
/// quantifier check this still produced `(t-var {} u8)`.
#[test]
fn an_explicit_binder_does_not_rebind_a_reserved_scalar_name() {
    for name in UNSIGNED.iter().chain(DEFERRED.iter()) {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f[{name}](x: {name}) -> {name} = x\n"
        ));
        assert_reaches_rejection(&deep, name, "an explicit binder in a scalar position");
    }
}

/// REGRESSION test. The tensor form of the same hole. The precision slot's
/// quantifier check sat above its reserved-name row, so an explicit binder
/// defeated the very fall-through the scalar arm was copying.
#[test]
fn an_explicit_binder_does_not_rebind_a_reserved_tensor_precision() {
    for name in UNSIGNED.iter().chain(DEFERRED.iter()) {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\n\
             def f[{name}](x: tensor[3, {name}]) -> tensor[3, {name}] = x\n"
        ));
        assert_reaches_rejection(&deep, name, "an explicit binder in a tensor precision slot");
    }
}

/// DISPOSITION LOCK. Green in both states, and the positive control for the
/// two rows above: an explicit binder list with an ordinary lowercase name
/// still binds, in both the scalar and the tensor precision position.
#[test]
fn an_explicit_binder_with_an_ordinary_name_still_binds() {
    let scalar = deep_of("module P.M\nexport (f)\ndef f[p](x: p) -> p = x\n");
    assert!(
        scalar.contains(&tvar_of("p")) && !scalar.contains(&prim_of("p")),
        "`p` must stay an explicitly bound type variable: {scalar}"
    );
    let tensor = deep_of("module P.M\nexport (f)\ndef f[p](x: tensor[3, p]) -> tensor[3, p] = x\n");
    assert!(
        tensor.contains(&tvar_of("p")) && !tensor.contains(&prim_of("p")),
        "`p` must stay a bound precision variable: {tensor}"
    );
}

/// REGRESSION test. The reserved-but-deferred family reaches the same
/// rejection in a scalar position, because the scalar arm consults the same
/// predicate pair the tensor precision slot does.
#[test]
fn the_deferred_family_reaches_the_rejection_in_a_scalar_position() {
    for name in DEFERRED {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"
        ));
        assert_reaches_rejection(&deep, name, "a scalar parameter and return");
    }
}

// ---------------------------------------------------------------------------
// Round 1 P1: the Deep carrier. The Surf repair alone left a hand-written
// `(t-var {} u8)` naming a type variable, which broke
// `spec/02-surf-syntax.md` §0.1's first law: the decompiler printed
// `def f[u8](x: u8) -> u8 = x`, and desugaring that gives `t-prim`, not the
// `t-var` it started from. §0.1 says a well-formed public Deep node HAS a
// Surf representation, so a node with none is not well-formed public Deep and
// the decompiler fails closed rather than leaving the law false.
// ---------------------------------------------------------------------------

fn deep_of_source(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp_file(source).expect("deep parse")
}

const RESERVED_TVAR_SCALAR: &str = "(module {surf_path: \"P.M\"}\n\
     p.m\n\
     (export {} f)\n\
     (defsig {} f (t-fn {} (t-var {} NAME) (t-var {} NAME)))\n\
     (def {} f (fn {} (params {} (x {type: (t-var {} NAME)})) (var {} x))))";

const RESERVED_TVAR_TENSOR: &str = "(module {surf_path: \"P.M\"}\n\
     p.m\n\
     (export {} f)\n\
     (defsig {} f (t-fn {} (t-tensor {} (d-lit {} 3) (t-var {} NAME)) \
     (t-tensor {} (d-lit {} 3) (t-var {} NAME))))\n\
     (def {} f (fn {} (params {} x) (var {} x))))";

/// REGRESSION test. The decompiler must refuse a Deep type variable whose name
/// is a spelling [04-DTYPE-1] rejects, in a scalar position and in a tensor
/// precision slot. Before, it printed Surf that this same build rejects.
#[test]
fn the_decompiler_refuses_a_reserved_deep_type_variable() {
    for name in UNSIGNED.iter().chain(DEFERRED.iter()) {
        for template in [RESERVED_TVAR_SCALAR, RESERVED_TVAR_TENSOR] {
            let deep = deep_of_source(&template.replace("NAME", name));
            let error = chelis_surf::decompile::try_decompile_program(&deep).expect_err(&format!(
                "`(t-var {{}} {name})` has no Surf representation and must fail closed"
            ));
            let rendered = error.to_string();
            assert!(
                rendered.contains(name) && rendered.contains("type variable"),
                "the refusal must name `{name}` and the role: {rendered}"
            );
        }
    }
}

/// DISPOSITION LOCK. Green at the parent commit and green here, and the label
/// matters: `spec/02-surf-syntax.md` §0.1's first law never broke for an
/// ORDINARY type variable, only for the reserved spellings, which the row above
/// removes from the law's domain by failing closed. What this row locks is that
/// removing them did not narrow the domain any further: every Deep the
/// decompiler still accepts must still satisfy `desugar(resugar(deep))`. A
/// too-eager predicate would show up here and nowhere else.
#[test]
fn the_surf_deep_round_trip_law_holds_for_every_resugarable_type_variable() {
    for name in ["a", "p", "elem"] {
        for template in [RESERVED_TVAR_SCALAR, RESERVED_TVAR_TENSOR] {
            let source = template.replace("NAME", name);
            let deep = deep_of_source(&source);
            let surf = chelis_surf::decompile::try_decompile_program(&deep)
                .unwrap_or_else(|error| panic!("`{name}` must still resugar: {error}"));
            let redesugared = desugar_program(&parse_str(&surf).expect("reparse"));
            assert_eq!(
                print_canonical_flat(&chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
                    &deep
                )),
                print_canonical_flat(&chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
                    &redesugared
                )),
                "desugar(resugar(deep)) must reproduce the Deep for `{name}`:\n{surf}"
            );
        }
    }
}
