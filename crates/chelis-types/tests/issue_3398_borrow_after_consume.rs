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
    copy_repairs(&checked, None)
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

fn repairs_with_roots(source: &str, roots: &std::collections::BTreeSet<String>) -> Vec<CopyRepair> {
    let decls = parse_str(&format!("{PRELUDE}{source}")).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep)
        .unwrap_or_else(|error| panic!("type check should succeed: {:?}", error.errors));
    copy_repairs(&checked, Some(roots)).expect("linearity should pass")
}

/// The source text a `surf:a..b` span identity names, in a fixture that
/// follows the prelude.
fn span_text(source: &str, id: &str) -> String {
    let text = format!("{PRELUDE}{source}");
    let (start, end) = id
        .strip_prefix("surf:")
        .and_then(|range| range.split_once(".."))
        .unwrap_or_else(|| panic!("`{id}` is not a Surf span identity"));
    text[start.parse::<usize>().unwrap()..end.parse::<usize>().unwrap()].to_string()
}

/// A match scrutinee after an ordinary consume takes the original, so every
/// still later use is rejected, borrow or consume (spec/04 section 8.3).
#[test]
fn a_use_after_an_owned_call_then_a_match_scrutinee_is_rejected() {
    for later in ["look(x)", "eats(x)"] {
        rejected(
            &format!(
                "def f(x: tensor[2, f32]) -> f32 = {{\n  u = eats(x)\n  v = match x with {{\n    | y => eats(y)\n  }}\n  add(add(u, v), {later})\n}}\n"
            ),
            "match scrutinee",
        );
    }
}

/// A join keeps the strongest consume of any branch, so a branch that hands
/// the value to a match scrutinee or a consuming capture rejects a later use
/// whichever side of the `if` it is on.
#[test]
fn a_join_rejects_after_a_non_ordinary_branch_in_either_order() {
    let scrutinee = "match x with {\n    | y => eats(y)\n  }";
    let capture = "{\n    g = fn (k: f32) -> add(k, eats(x))\n    g(1.0f32)\n  }";
    for (other, consumed_by) in [(scrutinee, "match scrutinee"), (capture, "closure capture")] {
        for (then_branch, else_branch) in [("eats(x)", other), (other, "eats(x)")] {
            rejected(
                &format!(
                    "def f(x: tensor[2, f32], c: bool) -> f32 = {{\n  u = if c then {then_branch} else {else_branch}\n  add(u, look(x))\n}}\n"
                ),
                consumed_by,
            );
        }
    }
}

/// Every path's latest ordinary consume receives a copy when a use follows
/// the join ([04-LIN-5]): both branches are listed, each forced by the use.
#[test]
fn a_join_copies_at_the_consume_of_every_branch() {
    let if_source = "def f(x: tensor[2, f32], c: bool) -> f32 = {\n  u = if c then eats(x) else add(eats(x), 1.0f32)\n  add(u, look(x))\n}\n";
    let match_source = "def f(x: tensor[2, f32], c: bool) -> f32 = {\n  u = match c with {\n    | true => eats(x)\n    | false => add(eats(x), 1.0f32)\n  }\n  add(u, eats(x))\n}\n";
    for (source, later) in [
        (if_source, CopyRepairUseKind::Borrow),
        (match_source, CopyRepairUseKind::Consume),
    ] {
        let repairs = accepted(source);
        assert_eq!(repairs.len(), 2, "{repairs:#?}");
        for repair in &repairs {
            assert_eq!(span_text(source, &repair.copy_at), "eats(x)");
            assert_eq!(repair.forced_by.len(), 1, "{repair:#?}");
            assert_eq!(repair.forced_by[0].kind, later);
            assert_eq!(repair.forced_by[0].at, repairs[0].forced_by[0].at);
        }
        assert_ne!(repairs[0].copy_at, repairs[1].copy_at);
    }
}

/// A branch that does not touch the value leaves the earlier consume as the
/// latest on its path, so that consume and the other branch's both copy.
#[test]
fn a_join_keeps_the_earlier_consume_of_an_untouched_branch() {
    let source = "def f(x: tensor[2, f32], c: bool) -> f32 = {\n  a = eats(x)\n  u = if c then eats(x) else 1.0f32\n  add(add(a, u), look(x))\n}\n";
    let repairs = accepted(source);
    assert_eq!(repairs.len(), 2, "{repairs:#?}");
    let kinds = repairs
        .iter()
        .map(|repair| {
            repair
                .forced_by
                .iter()
                .map(|later| later.kind)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            vec![CopyRepairUseKind::Consume, CopyRepairUseKind::Borrow],
            vec![CopyRepairUseKind::Borrow]
        ]
    );
}

/// The reads a destructuring `let` desugars to are not source uses: the
/// destructure is listed once, at the value it destructures.
#[test]
fn a_destructure_after_a_consume_lists_only_source_uses() {
    let source = "def eatp(p: (tensor[2, f32], tensor[2, f32])) -> f32 = tensor_to_scalar(sum(p.0, 0i32))\n\
                  def f(p: (tensor[2, f32], tensor[2, f32])) -> f32 = {\n  u = eatp(p)\n  (a, b) = p\n  add(u, add(eats(a), look(b)))\n}\n";
    let repairs = accepted(source)
        .into_iter()
        .filter(|repair| repair.declaration.as_deref() == Some("f"))
        .collect::<Vec<_>>();
    assert_eq!(repairs.len(), 1, "{repairs:#?}");
    assert_eq!(span_text(source, &repairs[0].copy_at), "eatp(p)");
    assert_eq!(repairs[0].forced_by.len(), 1, "{repairs:#?}");
    assert_eq!(span_text(source, &repairs[0].forced_by[0].at), "p");
}

/// A projection moves its component out ([04-LIN-11]), so its fan-out copies
/// the component, named by its path.
#[test]
fn a_projection_repair_names_the_component() {
    let source = "type Lin[i] =\n  | Lin { w: tensor[i, f32] }\n\
                  def loss(p: Lin[2]) -> f32 = tensor_to_scalar(sum(mul(p.w, p.w), 0i32))\n";
    let repairs = accepted(source);
    assert_eq!(repairs.len(), 1, "{repairs:#?}");
    assert_eq!(repairs[0].binding, "p.w");
    assert_eq!(span_text(source, &repairs[0].copy_at), "p.w");
    assert_eq!(span_text(source, &repairs[0].forced_by[0].at), "p.w");
}

/// [04-LIN-6]: a root observation is a terminal consuming use in the same
/// copy insertion, so the last initializer that consumed a root copies, and
/// of two roots on one value the first copies. A binding that is not a root
/// forces nothing.
#[test]
fn root_observation_forces_copies() {
    let source = "def bump(x: tensor[2, f32]) -> tensor[2, f32] = realize(x)\n\
                  x = to_tensor([1.0f32, 2.0f32])\nu = eats(x)\ny = bump(x)\n";
    let all = accepted(source);
    let root_forced = all
        .iter()
        .filter(|repair| {
            repair
                .forced_by
                .iter()
                .any(|later| later.kind == CopyRepairUseKind::Root)
        })
        .collect::<Vec<_>>();
    assert_eq!(root_forced.len(), 1, "{all:#?}");
    assert_eq!(root_forced[0].declaration.as_deref(), Some("y"));
    assert_eq!(span_text(source, &root_forced[0].copy_at), "bump(x)");

    let roots = ["u", "y"].map(String::from).into();
    let without_x = repairs_with_roots(source, &roots);
    assert!(
        without_x.iter().all(|repair| repair
            .forced_by
            .iter()
            .all(|later| later.kind != CopyRepairUseKind::Root)),
        "{without_x:#?}"
    );

    let aliased = accepted("x = to_tensor([1.0f32, 2.0f32])\ny = x\n");
    assert_eq!(aliased.len(), 1, "{aliased:#?}");
    assert_eq!(aliased[0].declaration.as_deref(), Some("x"));
    assert_eq!(aliased[0].consumed_by, "the root observation of `x`");
    assert_eq!(aliased[0].forced_by[0].kind, CopyRepairUseKind::Root);
}

/// One use in the generated programs below: a call that consumes or borrows
/// the parameter whole, or projects one component path out of it (a
/// projection moves its component out, [04-LIN-11]).
struct GenUse {
    call: &'static str,
    /// The component path the use touches; empty for the parameter whole.
    path: &'static [&'static str],
    consumes: bool,
}

/// A parameter type for the generated programs: its declarations, its
/// spelling, its leaves (every path to a non-product component), and the
/// uses the generator draws from.
struct Carrier {
    decls: &'static str,
    ty: &'static str,
    leaves: &'static [&'static [&'static str]],
    uses: &'static [GenUse],
}

/// The copies the language places, modelled per leaf and independently of
/// the checker's representation ([04-LIN-11], spec/04 section 8.3). Each leaf
/// remembers the latest consume that touched it. A use of path `u` needs the
/// value of every leaf under `u`, so it forces a copy at the latest consume of
/// each of those leaves; a consume of path `q` then becomes the latest consume
/// of every leaf under `q`. Returns, per copy site, the uses that forced it,
/// as indices into `uses`.
fn per_leaf_copies(
    carrier: &Carrier,
    uses: &[usize],
) -> std::collections::BTreeMap<usize, Vec<usize>> {
    let under = |leaf: &[&str], path: &[&str]| leaf.starts_with(path);
    let mut latest: Vec<Option<usize>> = vec![None; carrier.leaves.len()];
    let mut copies = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for (index, use_) in uses.iter().map(|use_| &carrier.uses[*use_]).enumerate() {
        let touched = carrier
            .leaves
            .iter()
            .enumerate()
            .filter(|(_, leaf)| under(leaf, use_.path))
            .map(|(leaf, _)| leaf)
            .collect::<Vec<_>>();
        let forced = touched
            .iter()
            .filter_map(|leaf| latest[*leaf])
            .collect::<std::collections::BTreeSet<_>>();
        for site in forced {
            copies.entry(site).or_default().push(index);
        }
        if use_.consumes {
            for leaf in touched {
                latest[leaf] = Some(index);
            }
        }
    }
    copies
}

const CARRIERS: &[Carrier] = &[
    Carrier {
        decls: "type R2 =\n  | R2 { w: tensor[2, f32], b: tensor[2, f32] }\n\
                def eatp(p: R2) -> f32 = tensor_to_scalar(sum(p.w, 0i32))\n\
                def lookp(p: &R2) -> f32 = 0.0f32\n",
        ty: "R2",
        leaves: &[&["w"], &["b"]],
        uses: &[
            GenUse {
                call: "eatp(p)",
                path: &[],
                consumes: true,
            },
            GenUse {
                call: "lookp(p)",
                path: &[],
                consumes: false,
            },
            GenUse {
                call: "eats(p.w)",
                path: &["w"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.b)",
                path: &["b"],
                consumes: true,
            },
        ],
    },
    Carrier {
        decls: "def eatp(p: (tensor[2, f32], tensor[2, f32])) -> f32 = tensor_to_scalar(sum(p.0, 0i32))\n\
                def lookp(p: &(tensor[2, f32], tensor[2, f32])) -> f32 = 0.0f32\n",
        ty: "(tensor[2, f32], tensor[2, f32])",
        leaves: &[&["0"], &["1"]],
        uses: &[
            GenUse {
                call: "eatp(p)",
                path: &[],
                consumes: true,
            },
            GenUse {
                call: "lookp(p)",
                path: &[],
                consumes: false,
            },
            GenUse {
                call: "eats(p.0)",
                path: &["0"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.1)",
                path: &["1"],
                consumes: true,
            },
        ],
    },
    Carrier {
        decls: "type R3 =\n  | R3 { w: tensor[2, f32], b: tensor[2, f32], c: tensor[2, f32] }\n\
                def eatp(p: R3) -> f32 = tensor_to_scalar(sum(p.w, 0i32))\n\
                def lookp(p: &R3) -> f32 = 0.0f32\n",
        ty: "R3",
        leaves: &[&["w"], &["b"], &["c"]],
        uses: &[
            GenUse {
                call: "eatp(p)",
                path: &[],
                consumes: true,
            },
            GenUse {
                call: "lookp(p)",
                path: &[],
                consumes: false,
            },
            GenUse {
                call: "eats(p.w)",
                path: &["w"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.b)",
                path: &["b"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.c)",
                path: &["c"],
                consumes: true,
            },
        ],
    },
    Carrier {
        decls: "def eatp(p: (tensor[2, f32], tensor[2, f32], tensor[2, f32])) -> f32 = tensor_to_scalar(sum(p.0, 0i32))\n\
                def lookp(p: &(tensor[2, f32], tensor[2, f32], tensor[2, f32])) -> f32 = 0.0f32\n",
        ty: "(tensor[2, f32], tensor[2, f32], tensor[2, f32])",
        leaves: &[&["0"], &["1"], &["2"]],
        uses: &[
            GenUse {
                call: "eatp(p)",
                path: &[],
                consumes: true,
            },
            GenUse {
                call: "lookp(p)",
                path: &[],
                consumes: false,
            },
            GenUse {
                call: "eats(p.0)",
                path: &["0"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.1)",
                path: &["1"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.2)",
                path: &["2"],
                consumes: true,
            },
        ],
    },
    Carrier {
        decls: "type Inner =\n  | Inner { w: tensor[2, f32], b: tensor[2, f32] }\n\
                type Outer =\n  | Outer { a: Inner, c: tensor[2, f32] }\n\
                def eatp(p: Outer) -> f32 = tensor_to_scalar(sum(p.c, 0i32))\n\
                def lookp(p: &Outer) -> f32 = 0.0f32\n\
                def eati(i: Inner) -> f32 = tensor_to_scalar(sum(i.w, 0i32))\n",
        ty: "Outer",
        leaves: &[&["a", "w"], &["a", "b"], &["c"]],
        uses: &[
            GenUse {
                call: "eatp(p)",
                path: &[],
                consumes: true,
            },
            GenUse {
                call: "lookp(p)",
                path: &[],
                consumes: false,
            },
            GenUse {
                call: "eati(p.a)",
                path: &["a"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.a.w)",
                path: &["a", "w"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.a.b)",
                path: &["a", "b"],
                consumes: true,
            },
            GenUse {
                call: "eats(p.c)",
                path: &["c"],
                consumes: true,
            },
        ],
    },
];

/// Every sequence of one to three uses over records and tuples of two and
/// three components and a nested record: the report lists exactly the copies
/// the per-leaf model places, each at the consume that receives it and forced
/// by every use that needs it. Disjoint components never copy for each
/// other; a use of the whole, or of an enclosing component, needs every leaf
/// under it, so sibling components that jointly cover it each force the
/// earlier consume that last took them.
#[test]
fn projection_copies_match_the_per_leaf_model_for_every_short_use_sequence() {
    for carrier in CARRIERS {
        let alphabet = 0..carrier.uses.len();
        let mut sequences: Vec<Vec<usize>> = Vec::new();
        for a in alphabet.clone() {
            sequences.push(vec![a]);
            for b in alphabet.clone() {
                sequences.push(vec![a, b]);
                for c in alphabet.clone() {
                    sequences.push(vec![a, b, c]);
                }
            }
        }
        for uses in &sequences {
            let mut body = format!("{}def f(p: {}) -> f32 = {{\n", carrier.decls, carrier.ty);
            let mut line_starts = Vec::new();
            for (index, use_) in uses.iter().enumerate() {
                line_starts.push(PRELUDE.len() + body.len());
                body.push_str(&format!("  u{index} = {}\n", carrier.uses[*use_].call));
            }
            let total = (1..uses.len()).fold("u0".to_string(), |acc, index| {
                format!("add({acc}, u{index})")
            });
            body.push_str(&format!("  {total}\n}}\n"));
            let line_of = |id: &str| {
                let start: usize = id
                    .strip_prefix("surf:")
                    .and_then(|range| range.split_once(".."))
                    .and_then(|(start, _)| start.parse().ok())
                    .unwrap_or_else(|| panic!("`{id}` is not a Surf span identity"));
                line_starts
                    .iter()
                    .rposition(|line| *line <= start)
                    .unwrap_or_else(|| panic!("`{id}` precedes every use"))
            };
            let actual = accepted(&body)
                .into_iter()
                .filter(|repair| repair.declaration.as_deref() == Some("f"))
                .map(|repair| {
                    (
                        line_of(&repair.copy_at),
                        repair
                            .forced_by
                            .iter()
                            .map(|later| line_of(&later.at))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            let calls = uses
                .iter()
                .map(|use_| carrier.uses[*use_].call)
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                per_leaf_copies(carrier, uses),
                "{} {calls:?}\n{body}",
                carrier.ty
            );
        }
    }
}

/// A closure an eager top-level initializer creates captures at its creation
/// ([04-LIN-2]), so a capture after an earlier initializer's consume is a
/// later use, and the consume receives the copy (spec/04 section 8.3). Unlike
/// a top-level function declaration, the closure has a place in initializer
/// order.
#[test]
fn a_top_level_closure_initializer_after_a_top_level_consume_is_copy_repaired() {
    let source = "x = to_tensor([1.0f32, 2.0f32])\nu = eats(x)\nv = {\n  g = fn (k: f32) -> add(k, look(x))\n  g(u)\n}\n";
    let repairs = accepted(source);
    assert_eq!(repairs.len(), 1, "{repairs:#?}");
    assert_eq!(repairs[0].declaration.as_deref(), Some("u"));
    assert_eq!(span_text(source, &repairs[0].copy_at), "eats(x)");
    assert_eq!(
        repairs[0]
            .forced_by
            .iter()
            .map(|later| later.kind)
            .collect::<Vec<_>>(),
        // `x` is also a root, so its observation needs the value too.
        [CopyRepairUseKind::Capture, CopyRepairUseKind::Root]
    );
}

/// A consuming capture takes the original, so a use after it is rejected even
/// when an ordinary consume came first and the capture itself was repaired.
#[test]
fn a_use_after_an_owned_call_then_a_consuming_capture_is_rejected() {
    for later in ["look(x)", "eats(x)"] {
        rejected(
            &format!(
                "def f(x: tensor[2, f32]) -> f32 = {{\n  u = eats(x)\n  g = fn (k: f32) -> add(k, eats(x))\n  add(g(u), {later})\n}}\n"
            ),
            "closure capture",
        );
    }
}

/// A destructured component is excepted from copy insertion, so capturing one
/// after its consume is rejected, whether the capture borrows or consumes.
#[test]
fn a_capture_of_a_consumed_component_is_rejected() {
    for body in ["look(a)", "eats(a)"] {
        let errors = repairs(&format!(
            "def f(p: (tensor[2, f32], tensor[2, f32])) -> f32 = {{\n  (a, b) = p\n  u = eats(a)\n  g = fn (k: f32) -> add(k, {body})\n  add(g(u), eats(b))\n}}\n"
        ))
        .expect_err("a component's fan-out is not copied");
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::UseAfterConsume)
                    && error.message.contains("variable `a`")
            }),
            "{body}: {errors:?}"
        );
    }
}

/// The witness the per-leaf model exists for: after `eatp(p)` takes the whole
/// value, each later component projection needs the original it left, so the
/// copy at `eatp(p)` is forced by both projections, not only the first.
#[test]
fn a_whole_consume_is_forced_by_every_later_component() {
    for carrier in &CARRIERS[..2] {
        let source = format!(
            "{}def f(p: {}) -> f32 = {{\n  u = {}\n  v = {}\n  w = {}\n  add(add(u, v), w)\n}}\n",
            carrier.decls,
            carrier.ty,
            carrier.uses[0].call,
            carrier.uses[2].call,
            carrier.uses[3].call
        );
        let repairs = accepted(&source)
            .into_iter()
            .filter(|repair| repair.declaration.as_deref() == Some("f"))
            .collect::<Vec<_>>();
        assert_eq!(repairs.len(), 1, "{}: {repairs:#?}", carrier.ty);
        assert_eq!(span_text(&source, &repairs[0].copy_at), "eatp(p)");
        let forced = repairs[0]
            .forced_by
            .iter()
            .map(|later| span_text(&source, &later.at))
            .collect::<Vec<_>>();
        assert_eq!(
            forced,
            [
                carrier.uses[2].call[5..8].to_string(),
                carrier.uses[3].call[5..8].to_string()
            ],
            "{}",
            carrier.ty
        );
    }
}

/// Joins union the latest consumes per leaf ([04-LIN-5]): after a branch that
/// projects `p.w` and one that consumes `p` whole, a later `p.b` needs only the
/// whole consume, and a later `p.w` needs both branches' consumes.
#[test]
fn a_join_keeps_each_branch_consume_per_component() {
    let source = "type R2 =\n  | R2 { w: tensor[2, f32], b: tensor[2, f32] }\n\
                  def eatp(p: R2) -> f32 = tensor_to_scalar(sum(p.w, 0i32))\n\
                  def f(p: R2, c: bool) -> f32 = {\n  u = if c then eats(p.w) else eatp(p)\n  v = eats(p.b)\n  w = eats(p.w)\n  add(add(u, v), w)\n}\n";
    let repairs = accepted(source)
        .into_iter()
        .filter(|repair| repair.declaration.as_deref() == Some("f"))
        .map(|repair| {
            (
                span_text(source, &repair.copy_at),
                repair
                    .forced_by
                    .iter()
                    .map(|later| span_text(source, &later.at))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        repairs,
        [
            ("p.w".to_string(), vec!["p.w".to_string()]),
            (
                "eatp(p)".to_string(),
                vec!["p.b".to_string(), "p.w".to_string()]
            ),
        ]
    );
}

/// A component the binding's type does not resolve, a field whose type is a
/// type parameter, still gets a leaf of its own when a projection reaches it.
#[test]
fn components_under_a_type_parameter_are_tracked_apart() {
    let decls = "type Boxed[a] =\n  | Boxed { v: a }\n";
    let ty = "Boxed[(tensor[2, f32], tensor[2, f32])]";
    let disjoint = accepted(&format!(
        "{decls}def f(p: {ty}) -> f32 = add(eats(p.v.0), eats(p.v.1))\n"
    ));
    assert!(
        disjoint
            .iter()
            .all(|repair| repair.declaration.as_deref() != Some("f")),
        "{disjoint:#?}"
    );
    let repeated = accepted(&format!(
        "{decls}def f(p: {ty}) -> f32 = add(eats(p.v.0), eats(p.v.0))\n"
    ));
    assert_eq!(
        repeated
            .iter()
            .filter(|repair| repair.declaration.as_deref() == Some("f"))
            .count(),
        1,
        "{repeated:#?}"
    );
}
