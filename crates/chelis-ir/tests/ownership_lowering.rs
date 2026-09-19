//! Verified ownership-lowering acceptance tests (chelis#1286).
//!
//! The fixtures and inline programs pass through Surf parsing, type/effect/
//! linearity checking, the real root manifest, and concrete host lowering
//! before the ownership boundary. Positive and negative cases therefore do
//! not self-certify against a synthetic ownership graph.

use std::collections::BTreeSet;

use chelis_effects::realizability::{compute_root_manifest, infer_realizability};
use chelis_ir::host::{
    ConcreteHostProgram, HostExpr, HostExprKind, HostFunctionOrigin,
    try_lower_compiled_program_with_manifest,
};
use chelis_ir::ownership::{
    HostOwnershipProgram, OwnershipError, VerifiedDagAction, VerifiedHostAction,
    VerifiedHostOperation, VerifiedHostProgram, VerifiedHostTerminator, VerifiedOwnershipUse,
    lower_dag_ownership, lower_host_ownership, verify_ownership,
};
use chelis_ir::{ConcreteHostType, Dag, DimInfo, RiscOp, TensorType};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::manifest::{ManifestedProgram, RootManifest};
use chelis_types::types::{Prim, Target};
use chelis_types::{CheckedProgram, check_linearity, check_typed_program};

const C_TENSOR_PRIMS: &[Prim] = &[
    Prim::F32,
    Prim::Bool,
    Prim::Bf16,
    Prim::F16,
    Prim::Int32,
    Prim::Int64,
];

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-cli/tests/fixtures/compiled_value_ownership/"
);

#[test]
fn verified_dag_exposes_the_exact_drop_source_without_raw_plan_access() {
    let ty = TensorType::scalar_f32();
    let mut dag = Dag::new();
    let source = dag.add_node(
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        ty.clone(),
        None,
    );
    let drop = dag.add_node(RiscOp::Drop, vec![source], ty, None);
    let verified = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();

    assert_eq!(
        verified.emission().action_for_node(drop),
        Some(VerifiedDagAction::OwnedDrop { node: drop, source })
    );
}

#[test]
fn verified_dag_distinguishes_a_borrowed_logical_drop_from_an_owned_terminal() {
    let ty = TensorType::scalar_f32();
    let mut dag = Dag::new();
    let borrowed = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let borrowed_drop = dag.add_node(RiscOp::Drop, vec![borrowed], ty.clone(), None);
    let owned = dag.add_node(
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        ty.clone(),
        None,
    );
    let owned_drop = dag.add_node(RiscOp::Drop, vec![owned], ty, None);
    let verified = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();

    assert_eq!(
        verified.emission().action_for_node(borrowed_drop),
        Some(VerifiedDagAction::BorrowedDrop {
            node: borrowed_drop,
            source: borrowed,
        })
    );
    assert_eq!(
        verified.emission().action_for_node(owned_drop),
        Some(VerifiedDagAction::OwnedDrop {
            node: owned_drop,
            source: owned,
        })
    );
}

struct Front {
    checked: CheckedProgram,
    manifest: RootManifest,
    manifested: ManifestedProgram,
    host: ConcreteHostProgram,
}

fn fixture_source(name: &str) -> String {
    let path = format!("{FIXTURE_DIR}{name}.ch");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

fn front(source: &str) -> Front {
    let declarations = surf_parse(source).unwrap_or_else(|error| panic!("parse: {error:?}"));
    let deep = desugar_program(&declarations);
    let checked = check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("type check: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked)
        .unwrap_or_else(|error| panic!("effects: {error:?}"));
    let checked = check_linearity(&checked).unwrap_or_else(|error| panic!("linearity: {error:?}"));
    let realizability = infer_realizability(&checked, C_TENSOR_PRIMS);
    let manifest = compute_root_manifest(&checked, &realizability);
    let compiled = try_lower_compiled_program_with_manifest(&checked, &manifest)
        .unwrap_or_else(|diagnostic| panic!("host lowering: {diagnostic:?}"));
    let host = compiled.host.expect("test source must have a host lane");
    let manifested = ManifestedProgram::new(checked.clone(), manifest.clone(), Target::C);
    Front {
        checked,
        manifest,
        manifested,
        host,
    }
}

fn lower_source(source: &str) -> Result<HostOwnershipProgram, OwnershipError> {
    let front = front(source);
    lower_host_ownership(&front.manifested, front.host)
}

fn lower_fixture(name: &str) -> Result<HostOwnershipProgram, OwnershipError> {
    lower_source(&fixture_source(name))
}

fn verified_source(source: &str) -> VerifiedHostProgram {
    let lowered = lower_source(source).unwrap_or_else(|error| panic!("lowering: {error}"));
    verify_ownership(lowered).unwrap_or_else(|error| panic!("verification: {error}"))
}

fn verified_fixture(name: &str) -> VerifiedHostProgram {
    let lowered = lower_fixture(name).unwrap_or_else(|error| panic!("{name}: {error}"));
    verify_ownership(lowered).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn unit_text(program: &VerifiedHostProgram, name: &str) -> String {
    let rendered = program.render();
    let header = if name == "roots" {
        "unit roots Roots\n".to_string()
    } else {
        format!("unit {name} Function\n")
    };
    let start = rendered
        .find(&header)
        .unwrap_or_else(|| panic!("missing {header:?} in:\n{rendered}"));
    let rest = &rendered[start..];
    let end = rest[header.len()..]
        .find("\nunit ")
        .map(|offset| header.len() + offset)
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

fn line_index(text: &str, needle: &str) -> usize {
    text.lines()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("missing {needle:?} in:\n{text}"))
}

fn projected_owner_terminal(text: &str) -> (usize, usize) {
    let (project_index, project_line) = text
        .lines()
        .enumerate()
        .find(|(_, line)| line.contains("project borrow"))
        .unwrap_or_else(|| panic!("missing projected owner in:\n{text}"));
    let owner = project_line
        .split_whitespace()
        .next_back()
        .expect("project operation carries an owner identity");
    let terminal = format!("drop move {owner}");
    let terminal_index = text
        .lines()
        .position(|line| line.trim() == terminal)
        .unwrap_or_else(|| panic!("missing terminal {terminal:?} in:\n{text}"));
    (project_index, terminal_index)
}

#[test]
fn real_phase2_corpus_lowers_and_verifies() {
    for name in [
        "issue_1222_distinct_roots",
        "issue_1222_root_alias",
        "issue_1344_captured_copy",
        "issue_1344_fresh_binding",
        "issue_1346_fold_alias",
        "issue_1346_fold_fresh",
        "issue_1352_if_alias",
        "issue_1352_if_fresh",
        "issue_1352_match_adt_fresh",
        "issue_1352_match_option_fresh",
        "issue_1356_fresh_argument",
        "issue_1356_nested_fresh_argument",
        "issue_1356_variable_argument",
        "contextual_callback",
        "control_recursive_scalar",
        "control_scalar_aggregate",
        "option_string",
        "issue_543_adt_tensor",
        "issue_543_tuple_tensor",
        "issue_544_dict_string_1",
    ] {
        let lowered = lower_fixture(name).unwrap_or_else(|error| panic!("{name}: {error}"));
        verify_ownership(lowered).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

#[test]
fn target_rejected_function_containers_reach_backend_after_verification() {
    for name in [
        "reject_option_function",
        "reject_list_function",
        "reject_tuple_function",
        "reject_dict_function",
        "reject_adt_function",
    ] {
        let lowered = lower_fixture(name).unwrap_or_else(|error| panic!("{name}: {error}"));
        verify_ownership(lowered).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

#[test]
fn aliased_roots_copy_the_earlier_sink_in_manifest_order() {
    let roots = unit_text(&verified_fixture("issue_1222_root_alias"), "roots");
    // Each artifact sink receives its own owner. The shared materialized value
    // remains live until both sinks have been emitted, then is dropped once.
    assert_eq!(count(&roots, "= copy clone"), 2, "{roots}");
    assert_eq!(count(&roots, "root original move"), 1, "{roots}");
    assert_eq!(count(&roots, "root alias move"), 1, "{roots}");
    assert_eq!(count(&roots, "drop move"), 1, "{roots}");
    assert!(
        line_index(&roots, "root original move") < line_index(&roots, "root alias move"),
        "{roots}"
    );

    let mut front = front(&fixture_source("issue_1222_root_alias"));
    front.manifest.entries.reverse();
    let manifested = ManifestedProgram::new(front.checked, front.manifest.clone(), Target::C);
    let lowered = lower_host_ownership(&manifested, front.host).unwrap();
    let roots = unit_text(&verify_ownership(lowered).unwrap(), "roots");
    assert!(
        line_index(&roots, "root alias move") < line_index(&roots, "root original move"),
        "{roots}"
    );
}

#[test]
fn non_root_heap_value_drops_at_its_verified_last_use() {
    let mut front = front("kept = [1i64]\nout = len(kept)\n");
    front
        .manifest
        .entries
        .retain(|entry| entry.def_name == "out");
    let manifested = ManifestedProgram::new(front.checked, front.manifest, Target::C);
    let lowered = lower_host_ownership(&manifested, front.host).unwrap();
    let roots = unit_text(&verify_ownership(lowered).unwrap(), "roots");
    assert_eq!(count(&roots, "root out move"), 1, "{roots}");
    assert_eq!(count(&roots, "drop move"), 1, "{roots}");
    assert!(
        line_index(&roots, "drop move") < line_index(&roots, "root out move"),
        "the unused container must be released before the unrelated root sink:\n{roots}"
    );
}

#[test]
fn recursive_tail_frame_drops_precede_the_direct_call() {
    let step = unit_text(&verified_fixture("issue_1206_depth_1"), "step");
    let recursive_call = line_index(&step, "call:step");
    let drops = step
        .lines()
        .enumerate()
        .filter_map(|(index, line)| line.contains("drop move").then_some(index))
        .collect::<Vec<_>>();
    assert!(
        !drops.is_empty(),
        "the recursive frame must own temporaries:\n{step}"
    );
    assert!(
        drops.iter().all(|drop| *drop < recursive_call),
        "every non-carried frame owner must die before the typed tail call:\n{step}"
    );
    assert!(
        step.lines().any(|line| line.contains("return move")),
        "{step}"
    );
}

#[test]
fn nested_scope_escape_projects_the_same_owner_into_the_outer_payload_slot() {
    let verified = verified_source("b = {\n  q = to_tensor([1.0f32, 2.0f32])\n  q\n}\n");
    let body = unit_text(&verified, "roots");
    assert_eq!(count(&body, "project borrow"), 1, "{body}");
    let (project, projected_owner_drop) = projected_owner_terminal(&body);
    assert!(
        project < projected_owner_drop,
        "the outer payload slot must be bound before the projected owner reaches its terminal:\n{body}"
    );

    let flat = unit_text(
        &verified_source("b = to_tensor([1.0f32, 2.0f32])\n"),
        "roots",
    );
    assert_eq!(count(&flat, "project borrow"), 0, "{flat}");

    let doubly_nested = unit_text(
        &verified_source(
            "b = {\n  middle = {\n    q = to_tensor([1.0f32, 2.0f32])\n    q\n  }\n  middle\n}\n",
        ),
        "roots",
    );
    assert_eq!(
        count(&doubly_nested, "project borrow"),
        2,
        "each lexical boundary must project the owner into the next outer payload slot:\n{doubly_nested}"
    );
    let (first_project, projected_owner_drop) = projected_owner_terminal(&doubly_nested);
    assert!(
        first_project < projected_owner_drop,
        "both payload projections must precede the owner's terminal:\n{doubly_nested}"
    );
    assert_eq!(
        doubly_nested
            .lines()
            .take(projected_owner_drop)
            .filter(|line| line.contains("project borrow"))
            .count(),
        2,
        "both projections of the same owner must precede its typed terminal:\n{doubly_nested}"
    );
}

#[test]
fn source_copy_mints_a_fresh_owned_identity_for_named_fresh_and_tail_values() {
    let named_program = verified_source(
        "a = to_tensor([1.0f32, 2.0f32])\n\
         b = copy(a)\n",
    );
    let named = unit_text(&named_program, "roots");
    assert!(named.contains("= tensor:"), "{named}");
    assert!(
        (0..named_program.nested_dag_count()).any(|index| named_program
            .nested_dag_render(index)
            .is_some_and(|dag| dag.contains("clone n"))),
        "source copy must be preserved in a verified nested DAG: {:?}",
        (0..named_program.nested_dag_count())
            .filter_map(|index| named_program.nested_dag_render(index))
            .collect::<Vec<_>>()
    );
    assert!(named.contains("root a move"), "{named}");
    assert!(named.contains("root b move"), "{named}");

    let fresh_program = verified_source("b = copy(to_tensor([3.0f32, 4.0f32]))\n");
    let fresh = unit_text(&fresh_program, "roots");
    assert!(
        fresh.contains("= tensor:"),
        "the fresh operand must still produce a distinct host owner:\n{fresh}"
    );
    assert!(
        (0..fresh_program.nested_dag_count()).any(|index| fresh_program
            .nested_dag_render(index)
            .is_some_and(|dag| dag.contains("clone n"))),
        "fresh source copy must be preserved in a verified nested DAG"
    );

    let tail_program = verified_source(
        "def duplicate(x: tensor[2, f32]) -> tensor[2, f32] = copy(x)\n\
         source = to_tensor([5.0f32, 6.0f32])\n\
         out = duplicate(source)\n",
    );
    let tail = unit_text(&tail_program, "duplicate");
    assert_eq!(
        count(&tail, "= copy clone %"),
        1,
        "entry adaptation must mint one owner before the helper consumes it:\n{tail}"
    );
    assert!(
        tail.lines().any(|line| line.contains("= tensor:")),
        "{tail}"
    );
    assert!(
        (0..tail_program.nested_dag_count()).any(|index| tail_program
            .nested_dag_render(index)
            .is_some_and(|dag| dag.contains("clone n"))),
        "tail source copy must be preserved in a verified nested DAG"
    );
    assert!(tail.contains("return move"), "{tail}");
}

/// chelis#2068: a by-value Copy scalar named once but used in several argument
/// slots of a tail-position user call (`f3(x, x)`) must duplicate per slot, not
/// move the single owner on the first slot and then read a dead owner on the
/// second. Regressed in 0.18.7; before the fix this failed with
/// `owner %1 in `g` b1 is not live`.
#[test]
fn scalar_reused_in_tail_call_slots_duplicates_and_verifies() {
    let source = "def f3(a: f32, b: f32) -> f32 = add(a, b)\n\
                  def g(x: f32) -> f32 = f3(x, x)\n\
                  def main() -> f32 = g(cast(0.5, f32))\n";
    // The core oracle: the whole program lowers AND verifies. Before the fix
    // the verifier rejected `g` because the first slot moved `x` and the second
    // read it dead.
    let program = verified_source(source);
    let g = unit_text(&program, "g");
    // Each occurrence of the Copy scalar is served by its own minted copy
    // rather than moving the single owner, so both call slots see a live owner.
    // The original scalar owner is then discarded, never moved into a slot.
    assert_eq!(
        count(&g, "= copy clone %"),
        2,
        "both reused scalar slots must duplicate the owner so neither reads it dead:\n{g}"
    );
    assert!(
        g.contains("discard %1"),
        "the original scalar owner must be discarded, not moved into a call slot:\n{g}"
    );
}

/// chelis#2068 soundness guard: the scalar carve-out is scoped strictly to Copy
/// scalars. A heap value (here a `string`, whose owned parameter mode moves it)
/// used in two tail-call slots is a genuine use-after-move and MUST still be
/// rejected, since moving one owned string into two owned slots would need an
/// explicit copy the user did not write.
#[test]
fn heap_value_reused_in_tail_call_slots_is_still_rejected() {
    let source = "def joins(a: string, b: string) -> string = string_concat(a, b)\n\
                  def dupstr(s: string) -> string = joins(s, s)\n\
                  def run_main() -> string = dupstr(\"hi\")\n";
    let error = match lower_source(source) {
        Ok(lowered) => verify_ownership(lowered).expect_err("heap double-move must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, OwnershipError::OwnerNotLive { .. }),
        "a genuine use-after-move of an owned heap value must stay rejected, got: {error}"
    );
}

#[test]
fn debug_observes_the_existing_owner_and_is_not_source_copy() {
    let roots = unit_text(
        &verified_source(
            "a = to_tensor([1.0f32, 2.0f32])\n\
             b = debug(a)\n",
        ),
        "roots",
    );
    assert!(roots.contains("builtin:debug(borrow"), "{roots}");
    assert_eq!(
        count(&roots, "= copy clone"),
        count(&roots, "root "),
        "debug must not mint an incidental owner; only artifact-root clones remain:\n{roots}"
    );
}

#[test]
fn missing_manifest_binding_is_a_typed_lowering_error() {
    let mut front = front("out = [1i64]\n");
    let mut ghost = front.manifest.entries[0].clone();
    ghost.name = "ghost".to_string();
    ghost.def_name = "ghost".to_string();
    front.manifest.entries.push(ghost);
    let manifested = ManifestedProgram::new(front.checked, front.manifest, Target::C);
    assert_eq!(
        lower_host_ownership(&manifested, front.host).unwrap_err(),
        OwnershipError::ManifestRootWithoutBinding {
            root: "ghost".to_string(),
            def_name: "ghost".to_string(),
        }
    );
}

#[test]
fn fresh_and_variable_arguments_have_distinct_single_transfer_chains() {
    let fresh = verified_fixture("issue_1356_fresh_argument");
    let roots = unit_text(&fresh, "roots");
    assert!(roots.contains("call:identity(move"), "{roots}");
    // Every clone belongs to an artifact root; the fresh call argument moves.
    assert_eq!(
        count(&roots, "= copy clone"),
        count(&roots, "root "),
        "{roots}"
    );

    let variable = verified_fixture("issue_1356_variable_argument");
    let roots = unit_text(&variable, "roots");
    assert!(roots.contains("call:identity(move"), "{roots}");
    // The named argument needs one additional clone before the consuming call.
    assert_eq!(
        count(&roots, "= copy clone"),
        count(&roots, "root ") + 1,
        "{roots}"
    );
}

#[test]
fn artifact_entry_borrow_is_copied_before_an_owned_formal() {
    let identity = unit_text(&verified_fixture("issue_1356_fresh_argument"), "identity");
    assert!(identity.contains("b0 (EntryBorrow %0)"), "{identity}");
    assert!(identity.contains("= copy clone %0"), "{identity}");
    assert!(identity.contains("jump b1 [move"), "{identity}");
    assert!(identity.contains("b1 (Owned"), "{identity}");
    assert!(identity.contains("return move"), "{identity}");
}

#[test]
fn artifact_entry_borrow_crosses_a_borrowed_formal_without_a_copy() {
    let verified = verified_source(
        "def peek(x: &tensor[2, f32]) -> i64 = 1i64\n\
         input = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n\
         out = peek(&input)\n",
    );
    let peek = unit_text(&verified, "peek");
    assert!(peek.contains("b0 (EntryBorrow %0)"), "{peek}");
    assert!(peek.contains("jump b1 [borrow %0]"), "{peek}");
    assert!(peek.contains("b1 (Borrowed"), "{peek}");
    assert_eq!(count(&peek, "= copy clone"), 0, "{peek}");
}

#[test]
fn capture_is_borrowed_and_an_owned_return_copies_it() {
    let verified =
        verified_source("global = \"abc\"\ndef read() -> string = global\nout = read()\n");
    let function = unit_text(&verified, "read");
    assert!(function.contains("EntryBorrow"), "{function}");
    assert!(function.contains("Borrowed"), "{function}");
    assert!(function.contains("= copy clone"), "{function}");
    assert!(function.contains("return move"), "{function}");
    assert_eq!(count(&function, "drop move"), 0, "{function}");
}

#[test]
fn mixed_if_arms_join_one_owned_result_and_copy_only_the_alias_arm() {
    for fixture in ["issue_1352_if_fresh", "issue_1352_if_alias"] {
        let roots = unit_text(&verified_fixture(fixture), "roots");
        assert!(roots.contains("branch borrow"), "{fixture}:\n{roots}");
        // Two copies mint the artifact root owners. Exactly one earlier copy
        // adapts the borrowed arm into the join's owned block parameter.
        assert_eq!(count(&roots, "= copy clone"), 3, "{fixture}:\n{roots}");
        let join = line_index(&roots, "b3 (Owned %");
        assert_eq!(
            roots
                .lines()
                .take(join)
                .filter(|line| line.contains("= copy clone"))
                .count(),
            1,
            "{fixture}:\n{roots}"
        );
        assert!(
            roots.lines().any(|line| line.contains("(Owned %")),
            "{fixture}:\n{roots}"
        );
        assert_eq!(
            count(&roots, "root selected move"),
            1,
            "{fixture}:\n{roots}"
        );
    }
}

#[test]
fn match_adt_and_match_option_use_the_same_owned_join_rule() {
    let adt = verified_fixture("issue_1352_match_adt_fresh");
    let choose = unit_text(&adt, "choose");
    assert!(choose.contains("match borrow"), "{choose}");
    assert!(
        choose.lines().any(|line| line.contains("(Owned %")),
        "{choose}"
    );
    assert!(choose.contains("= copy clone"), "{choose}");

    let option = verified_fixture("issue_1352_match_option_fresh");
    let choose = unit_text(&option, "choose");
    assert!(choose.contains("match borrow"), "{choose}");
    assert!(choose.contains("option_payload"), "{choose}");
    assert!(
        choose.lines().any(|line| line.contains("(Owned %")),
        "{choose}"
    );
    assert!(choose.contains("= copy clone"), "{choose}");
}

#[test]
fn option_some_moves_fresh_payloads_and_clones_named_payloads_before_the_move() {
    let fresh = unit_text(&verified_fixture("option_string"), "option_length");
    assert!(fresh.contains("builtin:Some(move"), "{fresh}");

    let named = unit_text(
        &verified_source(
            r#"
def length_after_wrap() -> i64 = {
  text = "abc"
  wrapped: Option[string] = Some(text)
  string_len(text)
}
length = length_after_wrap()
"#,
        ),
        "length_after_wrap",
    );
    assert!(named.contains("= copy clone"), "{named}");
    assert!(named.contains("builtin:Some(move"), "{named}");
}

/// chelis#2205: a list whose scheduled last use is `append` moves into the
/// builtin; one that is read again afterwards stays borrowed.
///
/// Counted receipt, asserted as a ratio: for an N-step let-bound append chain
/// the number of `builtin:append` applications that still BORROW their
/// container must not grow with N. Evidentiary status: REGRESSION TEST,
/// proven failing first: on `main` (`1b7e9fcd7`) the 8-step chain rendered 8
/// borrowing appends and the 16-step chain 16 (no step moved); after the
/// last-use upgrade both render 0, and every step moves.
#[test]
fn append_at_a_lists_last_use_moves_and_the_chain_stops_borrowing() {
    fn chain(steps: usize) -> String {
        let mut source = String::from("def build() -> i64 = {\n  x0: List[i64] = []\n");
        for step in 1..=steps {
            source.push_str(&format!(
                "  x{step} = append(x{}, cast({step}, i64))\n",
                step - 1
            ));
        }
        source.push_str(&format!("  len(x{steps})\n}}\nbuilt = build()\n"));
        source
    }
    fn borrowing_appends(steps: usize) -> (usize, usize) {
        let text = unit_text(&verified_source(&chain(steps)), "build");
        (
            count(&text, "builtin:append(borrow"),
            count(&text, "builtin:append(move"),
        )
    }
    let (small_borrow, small_move) = borrowing_appends(8);
    let (large_borrow, large_move) = borrowing_appends(16);
    eprintln!(
        "#2205 receipt: 8-step chain borrows {small_borrow} / moves {small_move}; \
         16-step chain borrows {large_borrow} / moves {large_move}"
    );
    assert_eq!(
        small_borrow + small_move,
        8,
        "every append in the 8-step chain is rendered exactly once"
    );
    assert!(
        large_borrow <= small_borrow,
        "#2205: borrowing appends must not grow with the chain; 8 steps borrowed \
         {small_borrow}, 16 steps borrowed {large_borrow}"
    );
}

/// chelis#2205 negative half: a list read after the append is not moved into
/// it, and a list read twice moves only at the second, final append.
#[test]
fn append_before_a_later_read_keeps_borrowing() {
    let text = unit_text(
        &verified_source(
            "def twice() -> i64 = {\n  a = [cast(1, i64)]\n  b = append(a, cast(4, i64))\n  c = append(a, cast(5, i64))\n  add(len(b), len(c))\n}\nresult = twice()\n",
        ),
        "twice",
    );
    let first = line_index(&text, "builtin:append(");
    let second = text
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("builtin:append("))
        .nth(1)
        .map(|(index, _)| index)
        .expect("two appends");
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[first].contains("builtin:append(borrow"),
        "the first append still borrows `a`, which is read again: {text}"
    );
    assert!(
        lines[second].contains("builtin:append(move"),
        "the second append is `a`'s last use and moves it: {text}"
    );
}

/// chelis#2205: a tuple-held alias retains the list, so the append at the
/// binding's last use is still a verified Move (the strong-owner count, not
/// the IR, decides whether the runtime pushes in place).
#[test]
fn append_at_last_use_moves_even_when_an_aggregate_holds_the_list() {
    let text = unit_text(
        &verified_source(
            "def held() -> i64 = {\n  xs = [cast(1, i64), cast(2, i64)]\n  held = (xs, cast(9, i64))\n  zs = append(xs, cast(3, i64))\n  add(len(zs), len(held.0))\n}\nresult = held()\n",
        ),
        "held",
    );
    assert!(text.contains("= copy clone"), "the tuple retains `xs`: {text}");
    assert!(
        text.contains("builtin:append(move"),
        "the append is `xs`'s last use and moves it: {text}"
    );
}

#[test]
fn fold_accumulator_is_one_owned_block_parameter_on_both_paths() {
    let alias = unit_text(&verified_fixture("issue_1346_fold_alias"), "roots");
    assert!(alias.contains("loop borrow"), "{alias}");
    let first_loop_entry = alias.find("jump b1 [move").unwrap();
    assert_eq!(
        count(&alias[..first_loop_entry], "= copy clone"),
        1,
        "{alias}"
    );
    assert!(
        alias
            .lines()
            .filter(|line| line.contains("(Owned %"))
            .count()
            >= 3
    );
    assert!(alias.contains("jump b1 [move"), "{alias}");

    let fresh = unit_text(&verified_fixture("issue_1346_fold_fresh"), "roots");
    assert!(fresh.contains("loop borrow"), "{fresh}");
    assert!(fresh.contains("drop move"), "{fresh}");
    assert!(fresh.contains("jump b1 [move"), "{fresh}");
}

#[test]
fn all_previously_supported_host_combinators_reach_the_verified_boundary() {
    let verified = verified_source("xs = [[1i64], [2i64]]\nys = map(fn (v: List[i64]) -> v, xs)\n");
    let roots = unit_text(&verified, "roots");
    assert!(roots.contains("empty_list"), "{roots}");
    assert!(roots.contains("list_push"), "{roots}");
    assert!(roots.contains("loop borrow"), "{roots}");

    for source in [
        "xs = [1i64, 2i64]\nys = filter(fn (v: i64) -> gte(v, 2i64), xs)\n",
        "xs = [1i64, 2i64]\nys = scan(fn (acc: i64, v: i64) -> add(acc, v), 0i64, xs)\n",
        "xs = [1i64, 2i64]\nys = partition(fn (v: i64) -> gt(v, 1i64), xs)\n",
        "xs = [1i64, 2i64]\nys = flat_map(fn (v: i64) -> [v, v], xs)\n",
        "sampled = with seed(7i64) { 1i64 }\n",
    ] {
        verify_ownership(lower_source(source).unwrap()).unwrap();
    }
}

fn assert_front_rejects(source: &str) {
    let declarations = surf_parse(source).expect("negative twin still parses");
    let deep = desugar_program(&declarations);
    assert!(
        check_typed_program(&deep).is_err(),
        "negative twin unexpectedly checked"
    );
}

#[test]
fn supported_combinators_keep_their_preexisting_typed_failure_twins() {
    assert_front_rejects("xs = [1i64]\nys = filter(fn (v: i64) -> missing(v), xs)\n");
    assert_front_rejects(
        "xs = [1i64]\nys = scan(fn (acc: i64, v: i64) -> missing(acc, v), 0i64, xs)\n",
    );
    assert_front_rejects("xs = [1i64]\nys = partition(fn (v: i64) -> missing(v), xs)\n");
    assert_front_rejects("xs = [1i64]\nys = flat_map(fn (v: i64) -> missing(v), xs)\n");
    assert_front_rejects("sampled = with seed(7i64) { missing }\n");
}

#[test]
fn malformed_real_host_programs_fail_at_typed_boundaries() {
    let front = front("def id(p: i64) -> i64 = p\nout = id(1i64)\n");
    let mut wrong_arity = front.host.clone();
    wrong_arity.globals[0].value = HostExpr::new(HostExprKind::Call {
        function: "id".to_string(),
        args: Vec::new(),
        arg_tys: Vec::new(),
        ty: ConcreteHostType::Int64,
    });
    assert!(matches!(
        lower_host_ownership(&front.manifested, wrong_arity),
        Err(OwnershipError::CallArityMismatch { .. })
    ));

    let mut unbound = front.host.clone();
    unbound.globals[0].value = HostExpr::new(HostExprKind::Var(
        "nobody".to_string(),
        ConcreteHostType::Int64,
    ));
    assert_eq!(
        lower_host_ownership(&front.manifested, unbound).unwrap_err(),
        OwnershipError::UnboundName {
            unit: "roots".to_string(),
            name: "nobody".to_string(),
        }
    );
}

#[test]
fn checked_scalar_call_slots_restore_the_exact_literal_type() {
    let verified = verified_source(
        "def keep_i32(x: i32) -> i32 = x\n\
         def keep_i64(x: i64) -> i64 = x\n\
         def keep_f32(x: f32) -> f32 = x\n\
         def keep_f64(x: f64) -> f64 = x\n\
         a = keep_i32(7)\n\
         b = keep_i64(7i64)\n\
         c = keep_f32(1.5)\n\
         d = keep_f64(1.5f64)\n\
         e = keep_f32(cast(42, f32))\n",
    );
    let roots = unit_text(&verified, "roots");
    for callee in ["keep_i32", "keep_i64", "keep_f32", "keep_f64"] {
        assert!(roots.contains(&format!("call:{callee}")), "{roots}");
    }
}

#[test]
fn checked_tensor_call_slots_preserve_dimension_instantiation_and_reject_forgery() {
    let source = "sig guarded[n]: tensor[n, f32] -> tensor[n, f32]\n\
                  def guarded(x) = {\n\
                    n = cast(shape(x, cast(0, i32)), i64)\n\
                    if gt(cast(1, i64), n) then fail(\"empty\") else x\n\
                  }\n\
                  def call(x: tensor[4, f32]) -> tensor[4, f32] = guarded(x)\n\
                  out = call(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";
    let front = front(source);
    let verified = verify_ownership(
        lower_host_ownership(&front.manifested, front.host.clone()).expect("valid instantiation"),
    )
    .expect("valid ownership");
    assert!(unit_text(&verified, "call").contains("call:guarded"));

    let mut forged = front.host;
    let HostExprKind::Call { arg_tys, .. } = &mut forged
        .functions
        .iter_mut()
        .find(|function| function.name == "call")
        .expect("call function")
        .body
        .kind
    else {
        panic!("guarded call must remain a direct host call")
    };
    arg_tys[0] = ConcreteHostType::Tensor(TensorType {
        dims: vec![DimInfo::Lit(5)],
        precision: Prim::F32,
    });
    let error = lower_host_ownership(&front.manifested, forged).unwrap_err();
    assert!(
        matches!(error, OwnershipError::CallArgumentType { argument: 0, .. }),
        "{error:?}"
    );
}

#[test]
fn checked_nominal_dimension_provenance_rejects_a_different_name() {
    let mut front = front(
        "def nominal(x: tensor[batch, f32]) -> tensor[batch, f32] = {\n\
           size = cast(shape(x, cast(0, i32)), i64)\n\
           if gt(cast(1, i64), size) then fail(\"empty\") else x\n\
         }\n\
         def caller(x: tensor[batch, f32]) -> tensor[batch, f32] = nominal(x)\n\
         out = 0\n",
    );
    let caller = front
        .host
        .functions
        .iter_mut()
        .find(|function| function.name == "caller")
        .expect("caller function");
    caller.params[0].ty = ConcreteHostType::Tensor(TensorType {
        dims: vec![DimInfo::Named("seq".into(), None)],
        precision: Prim::F32,
    });
    let error = lower_host_ownership(&front.manifested, front.host).unwrap_err();
    assert!(
        matches!(
            error,
            OwnershipError::CallArgumentType {
                ref callee,
                argument: 0,
                ..
            } if callee == "nominal"
        ),
        "{error:?}"
    );
}

#[test]
fn checked_repeated_dimension_variable_rejects_inconsistent_actuals() {
    let mut front = front(
        "sig paired[n]: tensor[n, f32] -> tensor[n, f32] -> tensor[n, f32]\n\
         def paired(x, y) = {\n\
           size = cast(shape(y, cast(0, i32)), i64)\n\
           if gt(cast(1, i64), size) then fail(\"empty\") else x\n\
         }\n\
         def caller(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = paired(x, y)\n\
         out = 0\n",
    );
    let caller = front
        .host
        .functions
        .iter_mut()
        .find(|function| function.name == "caller")
        .expect("caller function");
    caller.params[1].ty = ConcreteHostType::Tensor(TensorType {
        dims: vec![DimInfo::Lit(5)],
        precision: Prim::F32,
    });
    let error = lower_host_ownership(&front.manifested, front.host).unwrap_err();
    assert!(
        matches!(
            error,
            OwnershipError::CallArgumentType {
                ref callee,
                argument: 1,
                ..
            } if callee == "paired"
        ),
        "{error:?}"
    );
}

#[test]
fn forged_call_argument_types_cannot_retag_an_actual_expression() {
    let front = front("def keep(x: i32) -> i32 = x\nout = keep(1)\n");

    let mut forged_slot = front.host.clone();
    let HostExprKind::Call { arg_tys, .. } = &mut forged_slot
        .globals
        .iter_mut()
        .find(|binding| binding.name == "out")
        .expect("out binding")
        .value
        .kind
    else {
        panic!("out must remain a direct call")
    };
    arg_tys[0] = ConcreteHostType::Int64;
    assert!(matches!(
        lower_host_ownership(&front.manifested, forged_slot),
        Err(OwnershipError::CallArgumentType { argument: 0, .. })
    ));

    let mut forged_value = front.host;
    let HostExprKind::Call { args, .. } = &mut forged_value
        .globals
        .iter_mut()
        .find(|binding| binding.name == "out")
        .expect("out binding")
        .value
        .kind
    else {
        panic!("out must remain a direct call")
    };
    args[0] = HostExpr::new(HostExprKind::Bool(true));
    assert!(matches!(
        lower_host_ownership(&front.manifested, forged_value),
        Err(OwnershipError::CallArgumentType { argument: 0, .. })
    ));
}

#[test]
fn every_current_concrete_host_expr_kind_has_a_closed_disposition() {
    // The lowering match is wildcard-free, so a new enum variant also fails
    // compilation until this executable census and the lowering both change.
    let lowered: BTreeSet<&str> = [
        "Int",
        "Float",
        "Bool",
        "String",
        "List",
        "Tuple",
        "Var",
        "Call",
        "Builtin",
        "AdtConstruct",
        "AdtFieldAccess",
        "If",
        "MatchOption",
        "MatchAdt",
        "Let",
        "Map",
        "Fold",
        "TensorCall",
        "Unit",
    ]
    .into_iter()
    .collect();
    let successor: BTreeSet<&str> = ["Filter", "Scan", "Partition", "FlatMap", "WithSeed"]
        .into_iter()
        .collect();
    assert!(lowered.is_disjoint(&successor));
    assert_eq!(lowered.len() + successor.len(), 24);
}

#[test]
fn callable_manifest_root_consumes_the_materialized_observation_binding() {
    let verified = verified_source("def answer() -> string = \"owned\"\n");
    let roots = unit_text(&verified, "roots");
    assert!(roots.contains("call:answer"), "{roots}");
    assert!(roots.contains("root answer move"), "{roots}");

    let front = front("def answer() -> string = \"owned\"\n");
    let mut missing = front.host;
    missing.functions.clear();
    assert!(matches!(
        lower_host_ownership(&front.manifested, missing),
        Err(OwnershipError::ManifestRootWithoutBinding { .. })
    ));
}

#[test]
fn authored_entries_are_borrow_adapters_but_internal_specializations_are_not() {
    let authored = front("def id(x: string) -> string = x\nout = id(\"x\")\n");
    let authored_verified =
        verify_ownership(lower_host_ownership(&authored.manifested, authored.host).unwrap())
            .unwrap();
    assert!(unit_text(&authored_verified, "id").contains("EntryBorrow"));

    let internal = front("def id(x: string) -> string = x\nout = id(\"x\")\n");
    let mut host = internal.host;
    host.functions[0].origin = HostFunctionOrigin::Monomorphized;
    let internal_verified =
        verify_ownership(lower_host_ownership(&internal.manifested, host).unwrap()).unwrap();
    assert!(!unit_text(&internal_verified, "id").contains("EntryBorrow"));
}

#[test]
fn internal_borrowed_formal_remains_live_through_both_branch_arms() {
    let internal = front(
        "def choose(x: &tensor[1, f32], flag: bool) -> tensor[1, f32] =\n\
         if flag then x else x\n\
         input = to_tensor([cast(1.0, f32)])\n\
         out = choose(&input, true)\n",
    );
    let mut host = internal.host;
    let function = host
        .functions
        .iter_mut()
        .find(|function| function.name == "choose")
        .expect("choose host function");
    function.origin = HostFunctionOrigin::Monomorphized;
    let verified = verify_ownership(lower_host_ownership(&internal.manifested, host).unwrap())
        .expect("an internal borrowed formal spans its complete function body");
    let function = unit_text(&verified, "choose");
    assert!(!function.contains("EntryBorrow"), "{function}");
    assert!(function.contains("b0 (Borrowed %0"), "{function}");
    assert_eq!(count(&function, "copy clone %0"), 2, "{function}");
}

#[test]
fn host_payload_crosses_the_independently_verified_site_boundary() {
    let verified = verified_fixture("issue_1352_if_fresh");
    assert!(verified.render().contains("root "));
}

#[test]
fn verified_sites_export_closed_typed_clone_drop_and_root_actions() {
    let verified = verified_fixture("issue_1222_root_alias");
    let emission = verified.emission();
    let actions = emission
        .root_sites()
        .flat_map(|site| site.actions())
        .collect::<Vec<_>>();
    assert!(actions.iter().any(|action| matches!(
        action,
        VerifiedHostAction::Operation(VerifiedHostOperation::Clone { source, .. })
            if source.use_() == VerifiedOwnershipUse::Clone && source.owner().is_heap()
    )));
    assert!(actions.iter().any(|action| matches!(
        action,
        VerifiedHostAction::Operation(VerifiedHostOperation::RootConsume { .. })
    )));
    assert!(actions.iter().any(|action| matches!(
        action,
        VerifiedHostAction::Terminator(VerifiedHostTerminator::Exit { .. })
    )));
}

#[test]
fn verified_control_edges_carry_exact_target_parameters_and_operands() {
    let verified = verified_fixture("issue_1352_if_fresh");
    let edges = verified
        .emission()
        .root_sites()
        .flat_map(|site| site.actions())
        .filter_map(|action| match action {
            VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump { edge, .. }) => Some(edge),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(edges.iter().any(|edge| {
        edge.params().len() == 1
            && edge.args().len() == 1
            && edge.args()[0].use_() == VerifiedOwnershipUse::Move
    }));
}

fn scalar_tensor() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    }
}

#[test]
fn standalone_and_nested_dags_cross_verified_payload_boundaries() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_tensor(),
        None,
    );
    let neg = dag.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    dag.add_root(neg);
    let dag = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();
    assert!(dag.render().contains("borrow load n0"));
    assert!(dag.render().contains("root move n1"));

    let mut terminal_dag = Dag::new();
    let load = terminal_dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = terminal_dag.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    let copied = terminal_dag.add_node(RiscOp::Copy, vec![produced], scalar_tensor(), None);
    terminal_dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![produced],
        scalar_tensor(),
        None,
    );
    terminal_dag.add_node(RiscOp::Drop, vec![copied], scalar_tensor(), None);
    let terminal = verify_ownership(lower_dag_ownership(terminal_dag).unwrap()).unwrap();
    let terminal = terminal.render();
    assert!(terminal.contains("clone n2 from n1"), "{terminal}");
    assert!(terminal.contains("store n3 move n1"), "{terminal}");
    assert!(terminal.contains("drop n4 move n2"), "{terminal}");

    let host = verified_source(
        "def peek(x: &tensor[2, f32]) -> i64 = 1i64\n\
         input = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n\
         out = peek(&input)\n",
    );
    assert!(host.nested_dag_count() > 0);
    assert!(host.nested_dag_render(0).is_some());
}

#[test]
fn a_dag_owner_cannot_have_two_terminal_directives() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = dag.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![produced],
        scalar_tensor(),
        None,
    );
    dag.add_node(RiscOp::Drop, vec![produced], scalar_tensor(), None);
    assert!(matches!(
        lower_dag_ownership(dag),
        Err(OwnershipError::DagDuplicateTerminal { owner: 1 })
    ));
}

#[test]
fn a_borrowed_dag_drop_is_a_non_consuming_logical_discard() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let discarded = dag.add_node(RiscOp::Drop, vec![load], scalar_tensor(), None);
    let copied_after_discard = dag.add_node(RiscOp::Copy, vec![load], scalar_tensor(), None);
    dag.add_root(copied_after_discard);
    let verified = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();
    assert_eq!(
        verified.emission().action_for_node(discarded),
        Some(VerifiedDagAction::BorrowedDrop {
            node: discarded,
            source: load,
        })
    );

    let mut twin = Dag::new();
    let load = twin.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let copied = twin.add_node(RiscOp::Copy, vec![load], scalar_tensor(), None);
    twin.add_node(RiscOp::Drop, vec![copied], scalar_tensor(), None);
    verify_ownership(lower_dag_ownership(twin).unwrap()).unwrap();
}

#[test]
fn realize_clones_a_borrowed_or_fanned_out_source_and_moves_a_last_owned_source() {
    let mut borrowed = Dag::new();
    let load = borrowed.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let realized = borrowed.add_node(RiscOp::Realize, vec![load], scalar_tensor(), None);
    let later = borrowed.add_node(RiscOp::Copy, vec![load], scalar_tensor(), None);
    borrowed.add_root(realized);
    borrowed.add_root(later);
    let verified = verify_ownership(lower_dag_ownership(borrowed).unwrap()).unwrap();
    assert_eq!(
        verified.emission().action_for_node(realized),
        Some(VerifiedDagAction::CloneProduce {
            node: realized,
            source: load,
        })
    );

    let mut fanned = Dag::new();
    let load = fanned.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = fanned.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    let realized = fanned.add_node(RiscOp::Realize, vec![produced], scalar_tensor(), None);
    let later = fanned.add_node(RiscOp::Copy, vec![produced], scalar_tensor(), None);
    fanned.add_root(realized);
    fanned.add_root(later);
    let verified = verify_ownership(lower_dag_ownership(fanned).unwrap()).unwrap();
    assert_eq!(
        verified.emission().action_for_node(realized),
        Some(VerifiedDagAction::CloneProduce {
            node: realized,
            source: produced,
        })
    );

    let mut last = Dag::new();
    let load = last.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = last.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    let realized = last.add_node(RiscOp::Realize, vec![produced], scalar_tensor(), None);
    last.add_root(realized);
    let verified = verify_ownership(lower_dag_ownership(last).unwrap()).unwrap();
    assert_eq!(
        verified.emission().action_for_node(realized),
        Some(VerifiedDagAction::MoveProduce {
            node: realized,
            source: produced,
        })
    );
}

#[test]
fn store_clones_a_borrowed_source_but_moves_an_owned_source() {
    let mut borrowed = Dag::new();
    let load = borrowed.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let stored = borrowed.add_node(
        RiscOp::Store { name: "out".into() },
        vec![load],
        scalar_tensor(),
        None,
    );
    borrowed.add_root(stored);
    let verified = verify_ownership(lower_dag_ownership(borrowed).unwrap()).unwrap();
    assert_eq!(
        verified.emission().action_for_node(stored),
        Some(VerifiedDagAction::CloneStore {
            node: stored,
            source: load,
        })
    );

    let mut owned = Dag::new();
    let load = owned.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = owned.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    let stored = owned.add_node(
        RiscOp::Store { name: "out".into() },
        vec![produced],
        scalar_tensor(),
        None,
    );
    owned.add_root(stored);
    let verified = verify_ownership(lower_dag_ownership(owned).unwrap()).unwrap();
    assert_eq!(
        verified.emission().action_for_node(stored),
        Some(VerifiedDagAction::MoveStore {
            node: stored,
            source: produced,
        })
    );
}

#[test]
fn dangling_owned_dag_producer_receives_a_verified_scope_drop() {
    let mut dag = Dag::new();
    let unused = dag.add_node(
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        scalar_tensor(),
        None,
    );
    let output = dag.add_node(
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        scalar_tensor(),
        None,
    );
    dag.add_root(output);
    let verified = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();
    assert!(
        verified
            .emission()
            .actions()
            .any(|action| action == VerifiedDagAction::ScopeDrop { source: unused })
    );
}

#[test]
fn a_dag_copy_after_store_move_is_rejected() {
    let mut dag = Dag::new();
    let load = dag.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = dag.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![produced],
        scalar_tensor(),
        None,
    );
    let copied = dag.add_node(RiscOp::Copy, vec![produced], scalar_tensor(), None);
    dag.add_root(copied);
    assert!(matches!(
        lower_dag_ownership(dag),
        Err(OwnershipError::DagUseAfterTerminal {
            owner: 1,
            consumer: 3,
        })
    ));

    let mut twin = Dag::new();
    let load = twin.add_node(
        RiscOp::Load {
            name: "entry".into(),
        },
        vec![],
        scalar_tensor(),
        None,
    );
    let produced = twin.add_node(RiscOp::Neg, vec![load], scalar_tensor(), None);
    let copied = twin.add_node(RiscOp::Copy, vec![produced], scalar_tensor(), None);
    twin.add_node(
        RiscOp::Store { name: "out".into() },
        vec![produced],
        scalar_tensor(),
        None,
    );
    twin.add_root(copied);
    verify_ownership(lower_dag_ownership(twin).unwrap()).unwrap();
}

#[test]
fn signature_entry_requires_tensor_observations_and_preserves_borrows() {
    let front = front(
        "def guarded[n](x: tensor[n, f32]) -> tensor[n, f32] ! { IO } = { _ = print(\"entered\")\n x }\nout = guarded(to_tensor([1.0f32, 2.0f32]))\n",
    );
    let mut host = front.host.clone();
    let function = host
        .functions
        .iter_mut()
        .find(|f| f.name == "guarded")
        .unwrap();
    let ty = match &function.params[0].ty {
        ConcreteHostType::Tensor(ty) => ty.clone(),
        other => panic!("{other:?}"),
    };
    let plan = chelis_ir::host::SignatureEntryPlan::new([chelis_ir::host::HostTensorInput {
        name: "x".into(),
        ty,
    }]);
    function.body = HostExpr::new(HostExprKind::Let {
        bindings: vec![chelis_ir::host::HostBinding {
            name: "checked".into(),
            display_name: None,
            display_roots: Vec::new(),
            ty: ConcreteHostType::Unit,
            value: HostExpr::new(HostExprKind::SignatureEntry {
                plan,
                args: vec![HostExpr::new(HostExprKind::Var(
                    "x".into(),
                    function.params[0].ty.clone(),
                ))],
            }),
        }],
        body: Box::new(function.body.clone()),
        ty: function.ret_ty.clone(),
    });
    verify_ownership(lower_host_ownership(&front.manifested, host.clone()).unwrap()).unwrap();
    for missing in [false, true] {
        let mut forged = host.clone();
        let function = forged
            .functions
            .iter_mut()
            .find(|f| f.name == "guarded")
            .unwrap();
        let HostExprKind::Let { bindings, .. } = &mut function.body.kind else {
            panic!("let")
        };
        let HostExprKind::SignatureEntry { args, .. } = &mut bindings[0].value.kind else {
            panic!("entry")
        };
        if missing {
            args.clear();
        } else {
            args[0] = HostExpr::new(HostExprKind::String("not a tensor".into()));
        }
        let error = lower_host_ownership(&front.manifested, forged).unwrap_err();
        if missing {
            assert!(
                matches!(
                    error,
                    OwnershipError::CallArityMismatch {
                        supplied: 0,
                        declared: 1,
                        ..
                    }
                ),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(error, OwnershipError::CallArgumentType { argument: 0, .. }),
                "{error:?}"
            );
        }
    }
}
