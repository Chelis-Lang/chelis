//! chelis#3398: a borrow after an ordinary consume is consuming fan-out.
//!
//! spec/04 section 8.3: "Ordinary consuming fan-out (a later use after an
//! earlier ordinary consume) is handled by inserted copies; a use after any
//! other consume is rejected." The later use may be a borrow: an argument in
//! a `&T` position, an auto-borrowed operand of a read-only primitive
//! (spec/05 section 1.3.1), a borrowing closure capture, or an argument of a
//! `grad(f)(..)` or `vmap(f)(..)` call. The earlier ordinary consume then
//! receives the copy, exactly as it does when the later use is a consume.
//!
//! The negative twins pin what stays rejected: a use after a `drop`
//! ([04-LIN-11]), after a match scrutinee, after a consuming closure capture,
//! and after the consume of a destructured component.
//!
//! The second half pins the copy-repair report: each repair names the
//! binding, the consume that receives the copy, and every later use that
//! forced it, and the report is a function of the program text.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{CopyRepair, CopyRepairUseKind, check_typed_program, copy_repairs};

const PRELUDE: &str = "\
def eats(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))
def look(x: &tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))
def s(t: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(t, t), 0i32))
def sq(r: tensor[2, f32]) -> tensor[2, f32] = mul(r, r)
";

fn repairs(source: &str) -> Result<Vec<CopyRepair>, Vec<CheckError>> {
    let decls = parse_str(&format!("{PRELUDE}{source}")).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep)
        .unwrap_or_else(|error| panic!("type check should succeed: {:?}", error.errors));
    copy_repairs(&checked)
}

#[track_caller]
fn accepted(source: &str) -> Vec<CopyRepair> {
    repairs(source)
        .unwrap_or_else(|errors| panic!("expected the fan-out to be copy-repaired; got {errors:?}"))
}

#[track_caller]
fn rejected(source: &str, consumed_by: &str) {
    let errors = repairs(source).expect_err("expected a use-after-consume");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("variable `x`")
                && error.message.contains(consumed_by)
        }),
        "expected UseAfterConsume on `x` naming {consumed_by}; got {errors:?}"
    );
}

/// The one repair a single-copy fixture records in `f`, checked field by
/// field. Prelude declarations may hold repairs of their own (`loss` reads
/// `p.w` twice, which is component fan-out), so only `f`'s are counted.
#[track_caller]
fn assert_single_repair(
    repairs: &[CopyRepair],
    consumed_by: &str,
    later: &[CopyRepairUseKind],
) -> CopyRepair {
    let in_f = repairs
        .iter()
        .filter(|repair| repair.declaration.as_deref() == Some("f"))
        .collect::<Vec<_>>();
    assert_eq!(
        in_f.len(),
        1,
        "expected exactly one copy in `f`: {repairs:#?}"
    );
    let repair = in_f[0].clone();
    assert_eq!(repair.binding, "x", "{repair:#?}");
    assert_eq!(repair.consumed_by, consumed_by, "{repair:#?}");
    assert_eq!(
        repair
            .forced_by
            .iter()
            .map(|use_| use_.kind)
            .collect::<Vec<_>>(),
        later,
        "{repair:#?}"
    );
    repair
}

#[test]
fn borrowed_parameter_after_an_owned_call_is_copy_repaired() {
    let repairs =
        accepted("def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  add(u, look(x))\n}\n");
    assert_single_repair(&repairs, "call to `eats`", &[CopyRepairUseKind::Borrow]);
}

#[test]
fn grad_argument_after_an_owned_call_is_copy_repaired() {
    let repairs = accepted(
        "def f(x: tensor[2, f32]) -> (f32, tensor[2, f32]) = {\n  v = s(x)\n  g = grad(s)(x)\n  (v, g)\n}\n",
    );
    assert_single_repair(&repairs, "call to `s`", &[CopyRepairUseKind::Borrow]);
}

#[test]
fn vmap_argument_after_an_owned_call_is_copy_repaired() {
    let repairs = accepted(
        "def eat_rows(m: tensor[3, 2, f32]) -> f32 = tensor_to_scalar(sum(sum(m, 0i32), 0i32))\n\
         def f(x: tensor[3, 2, f32]) -> (f32, tensor[3, 2, f32]) = {\n  t = eat_rows(x)\n  (t, vmap(sq)(x))\n}\n",
    );
    assert_single_repair(&repairs, "call to `eat_rows`", &[CopyRepairUseKind::Borrow]);
}

#[test]
fn adt_grad_argument_after_an_owned_call_is_copy_repaired() {
    let decls = "type Lin[i] =\n  | Lin { w: tensor[i, f32] }\n\
                 def loss(p: Lin[2]) -> f32 = tensor_to_scalar(sum(mul(p.w, p.w), 0i32))\n";
    let repairs = accepted(&format!(
        "{decls}def f(x: Lin[2]) -> (f32, Lin[2]) = {{\n  v = loss(x)\n  g = grad(loss)(x)\n  (v, g)\n}}\n"
    ));
    assert_single_repair(&repairs, "call to `loss`", &[CopyRepairUseKind::Borrow]);
}

#[test]
fn read_only_primitive_after_realize_is_copy_repaired() {
    let repairs = accepted(
        "def f(x: tensor[2, f32]) -> tensor[2, f32] = {\n  y = realize(x)\n  add(y, sigmoid(x))\n}\n",
    );
    assert_single_repair(&repairs, "realize", &[CopyRepairUseKind::Borrow]);
}

#[test]
fn container_query_after_an_owned_parameter_is_copy_repaired() {
    let repairs = accepted(
        "def keep(xs: List[tensor[2, f32]]) -> List[tensor[2, f32]] = xs\n\
         def f(x: List[tensor[2, f32]]) -> (List[tensor[2, f32]], i64) = {\n  k = keep(x)\n  (k, len(x))\n}\n",
    );
    assert_single_repair(&repairs, "call to `keep`", &[CopyRepairUseKind::Borrow]);
}

#[test]
fn closure_captures_after_an_owned_call_are_copy_repaired() {
    let borrowing = accepted(
        "def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  g = fn (k: f32) -> add(k, look(x))\n  g(u)\n}\n",
    );
    assert_single_repair(&borrowing, "call to `eats`", &[CopyRepairUseKind::Capture]);
    let consuming = accepted(
        "def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  g = fn (k: f32) -> add(k, eats(x))\n  g(u)\n}\n",
    );
    assert_single_repair(&consuming, "call to `eats`", &[CopyRepairUseKind::Capture]);
}

#[test]
fn a_borrow_before_the_only_consume_needs_no_copy() {
    let repairs =
        accepted("def f(x: tensor[2, f32]) -> f32 = {\n  u = look(x)\n  add(u, eats(x))\n}\n");
    assert!(repairs.is_empty(), "{repairs:#?}");
}

#[test]
fn a_borrow_after_drop_is_rejected() {
    rejected(
        "def f(x: tensor[2, f32]) -> f32 = {\n  u = drop(x)\n  look(x)\n}\n",
        "call to `drop`",
    );
}

#[test]
fn a_grad_argument_after_drop_is_rejected() {
    rejected(
        "def f(x: tensor[2, f32]) -> tensor[2, f32] = {\n  u = drop(x)\n  grad(s)(x)\n}\n",
        "call to `drop`",
    );
}

#[test]
fn a_borrow_after_an_owned_call_then_drop_is_rejected() {
    rejected(
        "def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  v = drop(x)\n  add(u, look(x))\n}\n",
        "call to `drop`",
    );
}

#[test]
fn a_borrow_after_a_match_scrutinee_is_rejected() {
    rejected(
        "def f(x: tensor[2, f32]) -> f32 = {\n  u = match x with {\n    | y => eats(y)\n  }\n  add(u, look(x))\n}\n",
        "match scrutinee",
    );
}

#[test]
fn a_borrow_after_a_consuming_capture_is_rejected() {
    rejected(
        "def f(x: tensor[2, f32]) -> f32 = {\n  g = fn (k: f32) -> add(k, eats(x))\n  add(g(1.0f32), look(x))\n}\n",
        "closure capture",
    );
}

#[test]
fn a_borrow_after_a_destructured_component_consume_is_rejected() {
    let errors = repairs(
        "def f(p: (tensor[2, f32], tensor[2, f32])) -> f32 = {\n  (x, b) = p\n  u = eats(x)\n  add(add(u, look(x)), eats(b))\n}\n",
    )
    .expect_err("a component is excepted from copy insertion");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::UseAfterConsume)
                && error.message.contains("variable `x`")
        }),
        "{errors:?}"
    );
}

/// Each later consume becomes the latest one, so the next later use copies
/// at it: three consumes and a borrow place three copies, one per earlier
/// site, each forced by the use that follows it.
#[test]
fn a_fan_out_chain_copies_at_every_earlier_consume() {
    let repairs = accepted(
        "def f(x: tensor[2, f32]) -> f32 = {\n  a = eats(x)\n  b = eats(x)\n  c = eats(x)\n  add(add(a, b), add(c, look(x)))\n}\n",
    );
    let kinds = repairs
        .iter()
        .map(|repair| {
            assert_eq!(repair.binding, "x");
            assert_eq!(repair.consumed_by, "call to `eats`");
            assert_eq!(repair.forced_by.len(), 1, "{repair:#?}");
            repair.forced_by[0].kind
        })
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            CopyRepairUseKind::Consume,
            CopyRepairUseKind::Consume,
            CopyRepairUseKind::Borrow
        ]
    );
    let sites = repairs
        .iter()
        .map(|repair| repair.copy_at.clone())
        .collect::<Vec<_>>();
    let mut distinct = sites.clone();
    distinct.dedup();
    assert_eq!(distinct, sites, "one repair per copy site: {repairs:#?}");
}

/// Several later borrows after one consume share the one copy at it.
#[test]
fn later_uses_after_one_consume_share_its_copy() {
    let repairs = accepted(
        "def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  add(add(u, look(x)), look(x))\n}\n",
    );
    assert_single_repair(
        &repairs,
        "call to `eats`",
        &[CopyRepairUseKind::Borrow, CopyRepairUseKind::Borrow],
    );
}

/// A `drop` after an ordinary consume takes the original; the earlier
/// consume receives the copy.
#[test]
fn a_drop_after_an_owned_call_copies_at_the_call() {
    let repairs =
        accepted("def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  v = drop(x)\n  u\n}\n");
    assert_single_repair(&repairs, "call to `eats`", &[CopyRepairUseKind::Drop]);
}

/// The copy belongs to the declaration that holds the consume, even when the
/// later use sits in a later top-level initializer.
#[test]
fn a_top_level_repair_names_the_declaration_holding_the_copy() {
    let repairs = accepted("x = to_tensor([1.0f32, 2.0f32])\nu = eats(x)\nv = look(x)\n");
    assert_eq!(repairs.len(), 1, "{repairs:#?}");
    assert_eq!(repairs[0].declaration.as_deref(), Some("u"));
    assert_eq!(repairs[0].binding, "x");
}

/// The location strings are the span identities the checker's diagnostics
/// print, so a repair points at the same source text an error would.
#[test]
fn repair_sites_are_source_span_identities() {
    let source = "def f(x: tensor[2, f32]) -> f32 = {\n  u = eats(x)\n  add(u, look(x))\n}\n";
    let repairs = accepted(source);
    let repair = assert_single_repair(&repairs, "call to `eats`", &[CopyRepairUseKind::Borrow]);
    let text = format!("{PRELUDE}{source}");
    let slice = |id: &str| {
        let (start, end) = id
            .strip_prefix("surf:")
            .and_then(|range| range.split_once(".."))
            .unwrap_or_else(|| panic!("`{id}` is not a Surf span identity"));
        text[start.parse::<usize>().unwrap()..end.parse::<usize>().unwrap()].to_string()
    };
    assert_eq!(slice(&repair.copy_at), "eats(x)");
    assert_eq!(slice(&repair.forced_by[0].at), "x");
}

/// The report is a function of the program text: repeated runs agree
/// exactly, including order.
#[test]
fn the_report_is_deterministic() {
    let source = "def f(x: tensor[2, f32]) -> f32 = {\n  a = eats(x)\n  b = eats(x)\n  g = fn (k: f32) -> add(k, look(x))\n  add(add(a, b), g(look(x)))\n}\n";
    let first = serde_json::to_string(&accepted(source)).expect("serialize");
    for _ in 0..8 {
        let again = serde_json::to_string(&accepted(source)).expect("serialize");
        assert_eq!(first, again);
    }
}
