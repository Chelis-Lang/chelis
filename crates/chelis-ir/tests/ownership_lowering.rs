//! Phase 2 corpus for the `chelis-ir` ownership module (chelis#1286).
//!
//! Every test lowers a small Surf program through the existing checker and
//! host-lowering path, then through `lower_ownership` / `verify_ownership`,
//! and asserts on the rendered ownership form or on the exact typed error.
//! Positive and negative cases are paired per rule of
//! `spec/design/compiled_value_ownership.md` C1-C3 and [04-LIN-3..7].
//!
//! Fixture sources are read by path from
//! `crates/chelis-cli/tests/fixtures/compiled_value_ownership/`; nothing is
//! added, renamed, or removed there.

use std::collections::BTreeSet;

use chelis_effects::realizability::{compute_root_manifest, infer_realizability};
use chelis_ir::ConcreteHostType;
use chelis_ir::host::{
    ConcreteHostProgram, HostBinding, HostCallback, HostCallbackKind, HostExpr, HostExprKind,
    HostFunction, HostFunctionOrigin, HostParam, try_lower_compiled_program_with_manifest,
};
use chelis_ir::ownership::{
    AdversarialMutation, OwnershipError, OwnershipProgram, VerifiedOwnershipProgram,
    lower_ownership, verify_ownership,
};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::manifest::RootManifest;
use chelis_types::types::{Lane, Prim};
use chelis_types::{CheckedProgram, check_linearity, check_typed_program};

/// The C target's tensor-capable primitives, mirrored from
/// `chelis_backend_c::TENSOR_CAPABLE_PRIMS` so lane assignment matches
/// `chelis build --target c` without a backend dependency.
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

fn fixture_source(name: &str) -> String {
    let path = format!("{FIXTURE_DIR}{name}.ch");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

struct Front {
    checked: CheckedProgram,
    manifest: RootManifest,
    host: ConcreteHostProgram,
}

fn front(source: &str) -> Front {
    let decls = surf_parse(source).unwrap_or_else(|error| panic!("surf parse: {error:?}"));
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("type check: {:?}", errors.errors));
    let checked =
        chelis_effects::check_program(&checked).unwrap_or_else(|e| panic!("effects: {e:?}"));
    let checked = check_linearity(&checked).unwrap_or_else(|e| panic!("linearity: {e:?}"));
    let realizability = infer_realizability(&checked, C_TENSOR_PRIMS);
    let manifest = compute_root_manifest(&checked, &realizability);
    let compiled = try_lower_compiled_program_with_manifest(&checked, &manifest)
        .unwrap_or_else(|diagnostic| panic!("host lowering: {diagnostic:?}"));
    let host = compiled
        .host
        .expect("every corpus program has a host program");
    Front {
        checked,
        manifest,
        host,
    }
}

fn lower_source(source: &str) -> Result<OwnershipProgram, OwnershipError> {
    let f = front(source);
    lower_ownership(&f.checked, &f.host, &f.manifest)
}

fn lower_fixture(name: &str) -> Result<OwnershipProgram, OwnershipError> {
    lower_source(&fixture_source(name))
}

fn verified_source(source: &str) -> VerifiedOwnershipProgram {
    let program = lower_source(source).unwrap_or_else(|e| panic!("lowering: {e}"));
    verify_ownership(program).unwrap_or_else(|e| panic!("verification: {e}"))
}

fn verified_fixture(name: &str) -> VerifiedOwnershipProgram {
    verified_source(&fixture_source(name))
}

/// The rendered text of one unit: the `unit roots` block or one
/// `function <name>` block, up to the next unit header.
fn unit_text(rendered: &str, header: &str) -> String {
    let start = rendered
        .find(header)
        .unwrap_or_else(|| panic!("rendering lacks `{header}`:\n{rendered}"));
    let rest = &rendered[start + header.len()..];
    let end = rest
        .find("\nfunction ")
        .or_else(|| rest.find("\nunit "))
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

fn roots_text(program: &VerifiedOwnershipProgram) -> String {
    unit_text(&program.render(), "unit roots")
}

fn function_text(program: &VerifiedOwnershipProgram, name: &str) -> String {
    unit_text(&program.render(), &format!("function {name} "))
}

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

fn line_index(text: &str, needle: &str) -> usize {
    text.lines()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no line contains `{needle}` in:\n{text}"))
}

/// The owner id (`%N`) defined by the first line containing `needle`.
fn owner_defined_on(text: &str, needle: &str) -> String {
    let line = text
        .lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no line contains `{needle}` in:\n{text}"));
    let trimmed = line.trim_start();
    let end = trimmed[1..]
        .find(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(trimmed.len());
    assert!(
        trimmed.starts_with('%'),
        "line `{line}` does not define an owner"
    );
    trimmed[..end].to_string()
}

fn mutated(program: OwnershipProgram, mutation: AdversarialMutation) -> OwnershipError {
    let mutated = program
        .apply_adversarial_mutation(mutation)
        .unwrap_or_else(|e| panic!("mutation must find a site: {e}"));
    verify_ownership(mutated).expect_err("the verifier must reject the mutated program")
}

// ── Boundary ────────────────────────────────────────────────────────────

#[test]
fn lowered_corpus_verifies_and_the_verified_type_is_only_minted_by_verify() {
    // The compile-time half of this rule is the `compile_fail` doctest on
    // `VerifiedOwnershipProgram`; this is the runtime half.
    let program = lower_fixture("issue_1222_root_alias").expect("lowering");
    let verified = verify_ownership(program).expect("verification");
    assert!(verified.render().starts_with("unit roots"));
}

#[test]
fn corpus_fixtures_lower_and_verify() {
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
        let program = lower_fixture(name).unwrap_or_else(|e| panic!("{name}: lowering: {e}"));
        verify_ownership(program).unwrap_or_else(|e| panic!("{name}: verification: {e}"));
    }
}

// ── Manifested roots ([04-LIN-6], chelis#1222) ──────────────────────────

#[test]
fn root_alias_copies_for_the_earlier_root_and_consumes_at_the_final_root() {
    let roots = roots_text(&verified_fixture("issue_1222_root_alias"));
    let list = owner_defined_on(&roots, "= list [");
    assert_eq!(count(&roots, "= copy clone"), 1, "{roots}");
    let copy = owner_defined_on(&roots, "= copy clone");
    assert!(
        roots.contains(&format!("root_consume original move {copy}")),
        "{roots}"
    );
    assert!(
        roots.contains(&format!("root_consume alias move {list}")),
        "{roots}"
    );
    assert_eq!(count(&roots, "drop move"), 0, "{roots}");
    assert!(
        roots.contains(&format!("{list} \"original\",\"alias\":")),
        "the alias is a display name on the one owner:\n{roots}"
    );
}

#[test]
fn distinct_roots_consume_without_copies() {
    let roots = roots_text(&verified_fixture("issue_1222_distinct_roots"));
    assert_eq!(count(&roots, "= copy clone"), 0, "{roots}");
    assert_eq!(count(&roots, "root_consume"), 2, "{roots}");
    assert_eq!(count(&roots, "drop move"), 0, "{roots}");
    assert!(
        line_index(&roots, "root_consume first") < line_index(&roots, "root_consume second")
    );
}

#[test]
fn root_sinks_follow_manifest_order() {
    let mut f = front(&fixture_source("issue_1222_distinct_roots"));
    f.manifest.entries.reverse();
    let program = lower_ownership(&f.checked, &f.host, &f.manifest).expect("lowering");
    let roots = roots_text(&verify_ownership(program).expect("verification"));
    assert!(
        line_index(&roots, "root_consume second") < line_index(&roots, "root_consume first"),
        "{roots}"
    );
}

#[test]
fn non_root_top_level_owner_receives_a_drop() {
    let mut f = front("kept = [1i64]\nout = len(kept)\n");
    f.manifest.entries.retain(|entry| entry.def_name == "out");
    let program = lower_ownership(&f.checked, &f.host, &f.manifest).expect("lowering");
    let roots = roots_text(&verify_ownership(program).expect("verification"));
    let kept = owner_defined_on(&roots, "= list [");
    assert!(roots.contains(&format!("drop move {kept}")), "{roots}");
    assert_eq!(count(&roots, "root_consume"), 1, "{roots}");
    assert!(
        line_index(&roots, "root_consume out") < line_index(&roots, "drop move"),
        "roots consume before the non-root drop:\n{roots}"
    );
}

#[test]
fn manifest_root_naming_no_host_binding_is_rejected() {
    let mut f = front("out = [1i64]\n");
    let mut ghost = f.manifest.entries[0].clone();
    ghost.name = "ghost".to_string();
    ghost.def_name = "ghost".to_string();
    f.manifest.entries.push(ghost);
    let error = lower_ownership(&f.checked, &f.host, &f.manifest).expect_err("rejected");
    assert_eq!(
        error,
        OwnershipError::ManifestRootWithoutBinding {
            root: "ghost".to_string(),
            def_name: "ghost".to_string(),
        }
    );
}

#[test]
fn tensor_lane_manifest_entries_do_not_become_host_sinks() {
    let verified = verified_fixture("issue_1346_fold_alias");
    let roots = roots_text(&verified);
    assert!(!roots.contains("root_consume base"), "{roots}");
    assert!(roots.contains("root_consume picked.0"), "{roots}");
    assert!(roots.contains("root_consume picked.1"), "{roots}");
}

#[test]
fn root_sink_borrowing_its_root_is_rejected() {
    let program = lower_fixture("issue_1222_distinct_roots").expect("lowering");
    let error = mutated(program, AdversarialMutation::RootSinkBorrows);
    assert!(
        matches!(error, OwnershipError::OperandModeMismatch { .. }),
        "{error}"
    );
}

// ── Calls and returns ([04-LIN-4], chelis#1356, chelis#1344) ────────────

#[test]
fn fresh_argument_has_one_transfer_chain() {
    let verified = verified_fixture("issue_1356_fresh_argument");
    let roots = roots_text(&verified);
    let literal = owner_defined_on(&roots, "= list [");
    assert!(
        roots.contains(&format!("= call identity(move {literal})")),
        "{roots}"
    );
    assert_eq!(count(&roots, "= copy clone"), 0, "{roots}");
    let identity = function_text(&verified, "identity");
    let param = owner_defined_on(&identity, "\"values\"");
    assert!(identity.contains(&format!("return move {param}")), "{identity}");
    assert_eq!(count(&identity, "= copy clone"), 1, "{identity}");
    assert!(
        line_index(&identity, "= copy clone") < line_index(&identity, "body b"),
        "the only copy is the artifact-boundary adapter:\n{identity}"
    );
    assert_eq!(count(&identity, "drop move"), 0, "{identity}");
}

#[test]
fn nested_fresh_argument_has_one_transfer_chain() {
    let verified = verified_fixture("issue_1356_nested_fresh_argument");
    let roots = roots_text(&verified);
    assert_eq!(count(&roots, "= copy clone"), 0, "{roots}");
    assert_eq!(count(&roots, "drop move"), 0, "{roots}");
    let identity = function_text(&verified, "identity");
    assert_eq!(count(&identity, "drop move"), 0, "{identity}");
}

#[test]
fn variable_argument_copies_before_the_owned_formal() {
    let verified = verified_fixture("issue_1356_variable_argument");
    let roots = roots_text(&verified);
    let source = owner_defined_on(&roots, "= list [");
    let copy = owner_defined_on(&roots, "= copy clone");
    assert!(
        roots.contains(&format!("= copy clone {source}")),
        "{roots}"
    );
    assert!(
        roots.contains(&format!("= call identity(move {copy})")),
        "{roots}"
    );
    assert!(
        roots.contains(&format!("root_consume source move {source}")),
        "{roots}"
    );
    assert_eq!(count(&roots, "drop move"), 0, "{roots}");
}

#[test]
fn captured_owner_is_borrowed_through_a_local_alias() {
    let verified = verified_fixture("issue_1344_captured_copy");
    let take_length = function_text(&verified, "take_length");
    assert!(
        take_length.contains("captures: %0 \"global_values\",\"local_alias\": List[int64]"),
        "{take_length}"
    );
    assert!(take_length.contains("= builtin len(borrow %0)"), "{take_length}");
    assert_eq!(count(&take_length, "= copy clone"), 0, "{take_length}");
    assert_eq!(count(&take_length, "drop move"), 0, "{take_length}");
}

#[test]
fn fresh_binding_is_dropped_at_scope_exit() {
    let verified = verified_fixture("issue_1344_fresh_binding");
    let make_length = function_text(&verified, "make_length");
    let values = owner_defined_on(&make_length, "\"values\"");
    assert_eq!(count(&make_length, "drop move"), 1, "{make_length}");
    assert!(
        line_index(&make_length, "= builtin len(")
            < line_index(&make_length, &format!("drop move {values}")),
        "{make_length}"
    );
    assert!(
        line_index(&make_length, &format!("drop move {values}"))
            < line_index(&make_length, "return move"),
        "{make_length}"
    );
}

#[test]
fn returned_capture_is_copied_before_the_result() {
    let verified = verified_source("g = \"abc\"\ndef f() -> string = g\nout = f()\n");
    let f = function_text(&verified, "f");
    assert!(f.contains("captures: %0 \"g\": string"), "{f}");
    let copy = owner_defined_on(&f, "= copy clone %0");
    assert!(f.contains(&format!("return move {copy}")), "{f}");
    assert_eq!(count(&f, "drop move"), 0, "{f}");
}

#[test]
fn returned_let_alias_of_an_owned_parameter_moves_the_parameter() {
    let verified = verified_source(
        "def f(p: List[int64]) -> List[int64] = {\n  d = p\n  d\n}\nout = f([1i64])\n",
    );
    let f = function_text(&verified, "f");
    let param = owner_defined_on(&f, "\"p\",\"d\"");
    assert!(f.contains(&format!("return move {param}")), "{f}");
    assert_eq!(count(&f, "drop move"), 0, "{f}");
    assert_eq!(
        count(&f, "= copy clone"),
        1,
        "only the adapter copies:\n{f}"
    );
}

#[test]
fn owned_parameter_not_returned_is_dropped_before_the_result() {
    let verified = verified_source(
        "def f(p: List[int64]) -> List[int64] = [len(p)]\nout = f([1i64])\n",
    );
    let f = function_text(&verified, "f");
    let param = owner_defined_on(&f, "\"p\"");
    assert!(f.contains(&format!("= builtin len(borrow {param})")), "{f}");
    assert!(f.contains(&format!("drop move {param}")), "{f}");
    assert!(
        line_index(&f, &format!("drop move {param}")) < line_index(&f, "return move"),
        "{f}"
    );
}

#[test]
fn owned_formal_receives_a_copy_of_the_entry_borrow() {
    let verified = verified_fixture("issue_1356_fresh_argument");
    let identity = function_text(&verified, "identity");
    assert!(
        identity.contains("entry b0 (entry_borrow %0: List[int64])"),
        "{identity}"
    );
    let copy = owner_defined_on(&identity, "= copy clone %0");
    assert!(
        identity.contains(&format!("jump b1 [move {copy}]")),
        "{identity}"
    );
    assert!(identity.contains("body b1 (owned %2 \"values\": List[int64])"), "{identity}");
}

#[test]
fn borrowed_formal_receives_the_entry_borrow_directly() {
    let verified = verified_source(
        "def peek(x: &tensor[2, f32]) -> List[int64] = [1i64]\nt = to_tensor([1.0, 2.0])\nout = peek(&t)\n",
    );
    let peek = function_text(&verified, "peek");
    assert!(
        peek.contains("entry b0 (entry_borrow %0: tensor[2, f32])"),
        "{peek}"
    );
    assert!(peek.contains("jump b1 [borrow %0]"), "{peek}");
    assert!(peek.contains("body b1 (borrowed %1 \"x\": tensor[2, f32])"), "{peek}");
    assert_eq!(count(&peek, "= copy clone"), 0, "{peek}");
    let roots = roots_text(&verified);
    let t = owner_defined_on(&roots, "= builtin to_tensor(");
    assert!(roots.contains(&format!("= call peek(borrow {t})")), "{roots}");
    assert_eq!(count(&roots, "= copy clone"), 0, "{roots}");
}

#[test]
fn borrowed_and_owned_formals_take_their_own_adapter_edges() {
    let verified = verified_source(
        "def peek(x: &tensor[2, f32], y: tensor[2, f32]) -> List[int64] = [1i64]\nt = to_tensor([1.0, 2.0])\nu = to_tensor([3.0, 4.0])\nout = peek(&t, u)\n",
    );
    let peek = function_text(&verified, "peek");
    let copy = owner_defined_on(&peek, "= copy clone %1");
    assert!(
        peek.contains(&format!("jump b1 [borrow %0, move {copy}]")),
        "{peek}"
    );
    let y = owner_defined_on(&peek, "\"y\"");
    assert!(peek.contains(&format!("drop move {y}")), "{peek}");
    let roots = roots_text(&verified);
    assert_eq!(count(&roots, "= copy clone"), 1, "u is copied for the owned formal:\n{roots}");
}

#[test]
fn entry_borrow_moved_into_an_owned_formal_is_rejected() {
    let program = lower_fixture("issue_1356_fresh_argument").expect("lowering");
    let error = mutated(program, AdversarialMutation::MoveEntryBorrowIntoOwnedFormal);
    assert!(
        matches!(error, OwnershipError::BorrowEscapes { .. }),
        "{error}"
    );
}

#[test]
fn call_to_an_unknown_function_is_rejected() {
    let f = front("out = 1i64\n");
    let mut host = f.host.clone();
    host.globals[0].value = HostExpr::new(HostExprKind::Call {
        function: "ghost".to_string(),
        args: Vec::new(),
        arg_tys: Vec::new(),
        ty: ConcreteHostType::Int64,
    });
    let error = lower_ownership(&f.checked, &host, &f.manifest).expect_err("rejected");
    assert_eq!(
        error,
        OwnershipError::UnknownCallee {
            unit: "roots".to_string(),
            callee: "ghost".to_string(),
        }
    );
}

#[test]
fn call_with_the_wrong_arity_is_rejected() {
    let f = front("def id(p: int64) -> int64 = p\nout = id(1i64)\n");
    let mut host = f.host.clone();
    host.globals[0].value = HostExpr::new(HostExprKind::Call {
        function: "id".to_string(),
        args: Vec::new(),
        arg_tys: Vec::new(),
        ty: ConcreteHostType::Int64,
    });
    let error = lower_ownership(&f.checked, &host, &f.manifest).expect_err("rejected");
    assert_eq!(
        error,
        OwnershipError::CallArityMismatch {
            unit: "roots".to_string(),
            callee: "id".to_string(),
            supplied: 0,
            declared: 1,
        }
    );
}

#[test]
fn unbound_name_is_rejected() {
    let f = front("out = 1i64\n");
    let mut host = f.host.clone();
    host.globals[0].value = HostExpr::new(HostExprKind::Var(
        "nobody".to_string(),
        ConcreteHostType::Int64,
    ));
    let error = lower_ownership(&f.checked, &host, &f.manifest).expect_err("rejected");
    assert_eq!(
        error,
        OwnershipError::UnboundName {
            unit: "roots".to_string(),
            name: "nobody".to_string(),
        }
    );
}

// ── Branches and matches ([04-LIN-5], chelis#1352) ──────────────────────

#[test]
fn mixed_if_arms_copy_the_alias_and_move_the_fresh_value() {
    for name in ["issue_1352_if_fresh", "issue_1352_if_alias"] {
        let roots = roots_text(&verified_fixture(name));
        let original = owner_defined_on(&roots, "\"original\"");
        assert_eq!(count(&roots, "= copy clone"), 1, "{name}:\n{roots}");
        let copy = owner_defined_on(&roots, &format!("= copy clone {original}"));
        assert!(roots.contains("branch borrow"), "{name}:\n{roots}");
        assert!(
            roots.contains(&format!("[move {copy}]")),
            "the alias arm moves its copy into the join:\n{roots}"
        );
        let join = owner_defined_on(&roots, "\"selected\"");
        assert!(
            roots.contains(&format!("join\" (owned {join} \"selected\": List[int64])")),
            "{name}:\n{roots}"
        );
        assert!(
            roots.contains(&format!("root_consume selected move {join}")),
            "{name}:\n{roots}"
        );
        assert!(
            roots.contains(&format!("root_consume original move {original}")),
            "{name}:\n{roots}"
        );
        assert_eq!(count(&roots, "drop move"), 0, "{name}:\n{roots}");
    }
}

#[test]
fn match_adt_arms_join_through_one_owned_parameter() {
    let verified = verified_fixture("issue_1352_match_adt_fresh");
    let choose = function_text(&verified, "choose");
    let original = owner_defined_on(&choose, "\"original\"");
    let choice = owner_defined_on(&choose, "\"choice\"");
    assert!(
        choose.contains(&format!("match borrow {choice} [Existing => b")),
        "{choose}"
    );
    assert!(choose.contains("Fresh => b"), "{choose}");
    assert_eq!(
        count(&choose, &format!("= copy clone {original}")),
        1,
        "{choose}"
    );
    assert_eq!(count(&choose, "= list ["), 1, "{choose}");
    assert!(choose.contains("\"join\" (owned"), "{choose}");
    assert!(choose.contains(&format!("drop move {original}")), "{choose}");
    assert!(choose.contains(&format!("drop move {choice}")), "{choose}");
}

#[test]
fn match_option_control_joins_the_same_way() {
    let verified = verified_fixture("issue_1352_match_option_fresh");
    let choose = function_text(&verified, "choose");
    let original = owner_defined_on(&choose, "\"original\"");
    let choice = owner_defined_on(&choose, "\"choice\"");
    assert!(
        choose.contains(&format!("match borrow {choice} [Some => b")),
        "{choose}"
    );
    assert!(choose.contains("None => b"), "{choose}");
    assert!(choose.contains("\"some\" (owned"), "{choose}");
    assert!(choose.contains("\"flag\": bool"), "{choose}");
    assert_eq!(
        count(&choose, &format!("= copy clone {original}")),
        2,
        "the None arm and the inner then arm each copy:\n{choose}"
    );
    assert_eq!(count(&choose, "\"join\" (owned"), 2, "{choose}");
    assert!(choose.contains(&format!("drop move {original}")), "{choose}");
}

#[test]
fn forwarding_an_alias_into_a_join_without_a_copy_is_rejected() {
    let program = lower_fixture("issue_1352_if_fresh").expect("lowering");
    let error = mutated(program, AdversarialMutation::ForwardAliasWithoutCopy);
    assert!(
        matches!(error, OwnershipError::InconsistentLiveness { .. }),
        "{error}"
    );
}

#[test]
fn match_binder_is_a_fresh_owner_dropped_in_its_arm() {
    let verified = verified_fixture("option_string");
    let f = function_text(&verified, "option_length");
    let inner = owner_defined_on(&f, "\"inner\"");
    assert!(f.contains(&format!("= builtin string_len(borrow {inner})")), "{f}");
    assert!(f.contains(&format!("drop move {inner}")), "{f}");
    let value = owner_defined_on(&f, "\"value\"");
    assert!(f.contains(&format!("drop move {value}")), "{f}");
}

// ── Loops and folds ([04-LIN-5], chelis#1346) ───────────────────────────

#[test]
fn fold_alias_body_moves_the_accumulator_to_the_back_edge() {
    let verified = verified_fixture("issue_1346_fold_alias");
    let roots = roots_text(&verified);
    let base = owner_defined_on(&roots, "\"base\"");
    let init_copy = owner_defined_on(&roots, &format!("= copy clone {base}"));
    let values = owner_defined_on(&roots, "\"values\"");
    assert!(
        roots.contains(&format!("jump b1 [move {init_copy}]")),
        "the initializer moves into iteration zero:\n{roots}"
    );
    assert!(
        roots.contains(&format!("loop over borrow {values} carrying move")),
        "{roots}"
    );
    let acc = owner_defined_on(&roots, "\"acc\"");
    assert!(
        roots.contains(&format!("\"loop_body\" (owned {acc} \"acc\": (f32, f32), owned")),
        "{roots}"
    );
    assert!(
        roots.contains(&format!("jump b1 [move {acc}]")),
        "the body consumes the previous owner once and moves it to the back-edge:\n{roots}"
    );
    assert!(!roots.contains(&format!("drop move {acc}")), "{roots}");
    assert!(roots.contains("\"loop_exit\" (owned"), "{roots}");
    assert!(roots.contains("root_consume picked.0 move"), "{roots}");
    assert!(roots.contains("root_consume picked.1 move"), "{roots}");
}

#[test]
fn fold_fresh_body_drops_the_previous_accumulator() {
    let verified = verified_fixture("issue_1346_fold_fresh");
    let roots = roots_text(&verified);
    let acc = owner_defined_on(&roots, "\"acc\"");
    let body = unit_text(&roots, "\"loop_body\"");
    let tuple = owner_defined_on(&body, "= tuple [");
    assert!(body.contains(&format!("drop move {acc}")), "{body}");
    assert!(body.contains(&format!("jump b1 [move {tuple}]")), "{body}");
    assert!(
        line_index(&body, &format!("drop move {acc}")) < line_index(&body, "jump b1"),
        "{body}"
    );
}

#[test]
fn named_callback_fold_calls_the_step_function_with_moved_parameters() {
    let verified = verified_source(
        "def step(acc: int64, v: int64) -> int64 = add(acc, v)\nxs = [1i64, 2i64]\ns = fold(step, 0i64, xs)\n",
    );
    let roots = roots_text(&verified);
    let body = unit_text(&roots, "\"loop_body\"");
    assert!(body.contains("= call step(move %"), "{body}");
    let result = owner_defined_on(&body, "= call step(");
    assert!(body.contains(&format!("jump b1 [move {result}]")), "{body}");
}

#[test]
fn map_pushes_each_callback_result_into_the_carried_accumulator() {
    let verified = verified_source("xs = [[1i64], [2i64]]\nys = map(fn (v: List[int64]) -> v, xs)\n");
    let roots = roots_text(&verified);
    assert!(roots.contains("= literal empty_list"), "{roots}");
    let body = unit_text(&roots, "\"loop_body\"");
    let acc = owner_defined_on(&body, "(owned");
    let element = owner_defined_on(&body, "\"v\"");
    assert!(
        body.contains(&format!("= list_push move {acc}, move {element}")),
        "{body}"
    );
    assert_eq!(count(&body, "= copy clone"), 0, "{body}");
    assert_eq!(count(&body, "drop move"), 0, "{body}");
}

#[test]
fn map_body_drops_an_unconsumed_element() {
    let verified = verified_source("xs = [[1i64], [2i64]]\nys = map(fn (v: List[int64]) -> len(v), xs)\n");
    let roots = roots_text(&verified);
    let body = unit_text(&roots, "\"loop_body\"");
    let element = owner_defined_on(&body, "\"v\"");
    assert!(body.contains(&format!("drop move {element}")), "{body}");
}

#[test]
fn a_second_consume_of_a_terminated_owner_is_rejected() {
    let program = lower_fixture("issue_1346_fold_fresh").expect("lowering");
    let error = mutated(program, AdversarialMutation::DuplicateFirstDrop);
    assert!(matches!(error, OwnershipError::UseAfterMove { .. }), "{error}");
}

// ── Verifier rules (C1) ─────────────────────────────────────────────────

#[test]
fn use_of_an_undefined_owner_is_rejected() {
    let program = lower_fixture("issue_1222_distinct_roots").expect("lowering");
    let error = mutated(program, AdversarialMutation::ReferenceUndefinedOwner);
    assert!(
        matches!(error, OwnershipError::UndefinedOwner { .. }),
        "{error}"
    );
}

#[test]
fn owner_without_a_terminal_operation_is_rejected() {
    let program = lower_fixture("issue_1344_fresh_binding").expect("lowering");
    let error = mutated(program, AdversarialMutation::RemoveFirstDrop);
    assert!(
        matches!(error, OwnershipError::OwnerNotTerminated { .. }),
        "{error}"
    );
}

#[test]
fn borrow_crossing_a_consuming_edge_is_rejected() {
    let program = lower_fixture("control_scalar_aggregate").expect("lowering");
    let error = mutated(program, AdversarialMutation::BorrowAndMoveInOneAggregate);
    assert!(
        matches!(error, OwnershipError::BorrowCrossesConsume { .. }),
        "{error}"
    );
}

#[test]
fn clone_outside_a_copy_is_rejected() {
    let program = lower_fixture("control_scalar_aggregate").expect("lowering");
    let error = mutated(program, AdversarialMutation::CloneInsideAggregate);
    assert!(
        matches!(error, OwnershipError::CloneOutsideCopy { .. }),
        "{error}"
    );
}

#[test]
fn copy_reusing_its_source_identity_is_rejected() {
    let program = lower_fixture("issue_1222_root_alias").expect("lowering");
    let error = mutated(program, AdversarialMutation::CopyReusesSourceIdentity);
    assert!(
        matches!(error, OwnershipError::CopyReusesIdentity { .. }),
        "{error}"
    );
}

#[test]
fn misclassified_owner_is_rejected() {
    let program = lower_fixture("issue_1222_distinct_roots").expect("lowering");
    let error = mutated(program, AdversarialMutation::FlipFirstHeapOwnerClass);
    assert!(
        matches!(error, OwnershipError::ClassificationMismatch { .. }),
        "{error}"
    );
}

#[test]
fn unreachable_block_is_rejected() {
    let program = lower_fixture("issue_1222_distinct_roots").expect("lowering");
    let error = mutated(program, AdversarialMutation::AppendUnreachableBlock);
    assert!(
        matches!(error, OwnershipError::UnreachableBlock { .. }),
        "{error}"
    );
}

// ── Classification and function values (C4, chelis#879) ────────────────

#[test]
fn contextual_callback_parameter_is_nonheap_and_callable() {
    let verified = verified_fixture("contextual_callback");
    let apply = function_text(&verified, "apply");
    assert!(
        apply.contains("entry b0 (owned %0: fn(int8) -> int8, owned %1: int8)"),
        "a callback and a scalar are values, not entry borrows:\n{apply}"
    );
    let callback = owner_defined_on(&apply, "\"callback\"");
    let value = owner_defined_on(&apply, "\"value\"");
    assert!(
        apply.contains(&format!("= call_callback {callback}(move {value})")),
        "{apply}"
    );
    assert_eq!(count(&apply, "= copy clone"), 0, "{apply}");
    assert_eq!(count(&apply, "drop move"), 0, "{apply}");
}

#[test]
fn function_container_is_rejected() {
    let error = lower_fixture("reject_list_function").expect_err("rejected");
    assert_eq!(
        error,
        OwnershipError::FunctionContainer {
            unit: "roots".to_string(),
            ty: "List[fn(int8) -> int8]".to_string(),
        }
    );
}

#[test]
fn first_class_function_value_is_rejected() {
    let f = front("def id(p: int64) -> int64 = p\nout = id(1i64)\n");
    let mut host = f.host.clone();
    let fn_ty = ConcreteHostType::Function(vec![ConcreteHostType::Int64], Box::new(ConcreteHostType::Int64));
    host.globals[0].value = HostExpr::new(HostExprKind::Var("id".to_string(), fn_ty.clone()));
    host.globals[0].ty = fn_ty;
    let error = lower_ownership(&f.checked, &host, &f.manifest).expect_err("rejected");
    assert_eq!(
        error,
        OwnershipError::FirstClassFunctionValue {
            unit: "roots".to_string(),
            name: "id".to_string(),
        }
    );
}

#[test]
fn scalar_aggregates_and_recursion_carry_no_heap_owner_for_scalars() {
    let verified = verified_fixture("control_recursive_scalar");
    let step = function_text(&verified, "step");
    assert_eq!(count(&step, "drop move"), 0, "{step}");
    assert_eq!(count(&step, "= copy clone"), 0, "{step}");
    let roots = roots_text(&verified);
    assert!(roots.contains("= list [move %"), "{roots}");
    assert_eq!(count(&roots, "= copy clone"), 0, "scalars are never copied:\n{roots}");
}

#[test]
fn heap_children_of_an_aggregate_are_moved_in() {
    let verified = verified_fixture("issue_543_tuple_tensor");
    let make = function_text(&verified, "make");
    let first = owner_defined_on(&make, "\"first\"");
    let second = owner_defined_on(&make, "\"second\"");
    assert_eq!(count(&make, "= copy clone"), 2, "{make}");
    assert!(make.contains(&format!("drop move {first}")), "{make}");
    assert!(make.contains(&format!("drop move {second}")), "{make}");
    let roots = roots_text(&verified);
    assert!(roots.contains("root_consume out move"), "{roots}");
}

// ── Total coverage of `HostExprKind` ────────────────────────────────────

fn hand_built_global(value: HostExprKind<ConcreteHostType>, ty: ConcreteHostType) -> Front {
    let mut f = front("out = 1i64\n");
    f.host.globals[0].value = HostExpr::new(value);
    f.host.globals[0].ty = ty;
    f
}

fn int_callback(params: Vec<(&str, ConcreteHostType)>, ret: ConcreteHostType) -> HostCallback<ConcreteHostType> {
    HostCallback {
        kind: HostCallbackKind::Inline {
            params: params
                .into_iter()
                .map(|(name, ty)| HostParam {
                    name: name.to_string(),
                    ty,
                })
                .collect(),
            body: Box::new(HostExpr::new(HostExprKind::Int(1))),
        },
        ret_ty: ret,
    }
}

fn expect_unlowered(value: HostExprKind<ConcreteHostType>, ty: ConcreteHostType, variant: &str) {
    let f = hand_built_global(value, ty);
    let error = lower_ownership(&f.checked, &f.host, &f.manifest).expect_err("typed rejection");
    assert_eq!(
        error,
        OwnershipError::UnloweredExprKind {
            unit: "roots".to_string(),
            variant: variant.to_string(),
        }
    );
}

#[test]
fn filter_is_typed_rejected_by_variant() {
    let error = lower_source("xs = [1i64, 2i64]\nys = filter(fn (v: int64) -> gte(v, 2i64), xs)\n")
        .expect_err("typed rejection");
    assert_eq!(
        error,
        OwnershipError::UnloweredExprKind {
            unit: "roots".to_string(),
            variant: "Filter".to_string(),
        }
    );
}

#[test]
fn scan_is_typed_rejected_by_variant() {
    let error = lower_source(
        "xs = [1i64, 2i64]\nys = scan(fn (acc: int64, v: int64) -> add(acc, v), 0i64, xs)\n",
    )
    .expect_err("typed rejection");
    assert_eq!(
        error,
        OwnershipError::UnloweredExprKind {
            unit: "roots".to_string(),
            variant: "Scan".to_string(),
        }
    );
}

#[test]
fn partition_flat_map_and_with_seed_are_typed_rejected_by_variant() {
    let list_ty = ConcreteHostType::List(Box::new(ConcreteHostType::Int64));
    let xs = || HostExpr::new(HostExprKind::List(vec![HostExpr::new(HostExprKind::Int(1))], list_ty.clone()));
    expect_unlowered(
        HostExprKind::Partition {
            callback: int_callback(vec![("v", ConcreteHostType::Int64)], ConcreteHostType::Bool),
            list: Box::new(xs()),
            ty: ConcreteHostType::Tuple(vec![list_ty.clone(), list_ty.clone()]),
        },
        ConcreteHostType::Tuple(vec![list_ty.clone(), list_ty.clone()]),
        "Partition",
    );
    expect_unlowered(
        HostExprKind::FlatMap {
            callback: int_callback(vec![("v", ConcreteHostType::Int64)], list_ty.clone()),
            list: Box::new(xs()),
            ty: list_ty.clone(),
        },
        list_ty.clone(),
        "FlatMap",
    );
    expect_unlowered(
        HostExprKind::WithSeed {
            seed: Box::new(HostExpr::new(HostExprKind::Int(42))),
            body: Box::new(HostExpr::new(HostExprKind::Int(1))),
            ty: ConcreteHostType::Int64,
        },
        ConcreteHostType::Int64,
        "WithSeed",
    );
}

#[test]
fn literal_unit_and_none_and_nil_lower_as_fresh_values() {
    let verified = verified_source(
        "def nothing() -> Option[int64] = None\ndef empty() -> List[int64] = Nil\nu = ()\na = nothing()\nb = empty()\n",
    );
    let nothing = function_text(&verified, "nothing");
    assert!(nothing.contains("= literal none"), "{nothing}");
    let empty = function_text(&verified, "empty");
    assert!(empty.contains("= literal empty_list"), "{empty}");
    let roots = roots_text(&verified);
    assert!(roots.contains("= literal unit"), "{roots}");
}

#[test]
fn field_access_borrows_its_base_and_mints_a_fresh_owner() {
    let verified = verified_source(
        "type Box =\n  | Box { items: List[int64] }\ndef f() -> List[int64] = {\n  b = Box { items: [1i64] }\n  match b with {\n    | Box { items } => items\n  }\n}\nout = f()\n",
    );
    let f = function_text(&verified, "f");
    let b = owner_defined_on(&f, "\"b\"");
    assert!(f.contains(&format!("match borrow {b} [Box => b")), "{f}");
    let items = owner_defined_on(&f, "\"items\"");
    assert!(f.contains(&format!("[move {items}]")), "the arm moves its binder into the join:\n{f}");
    assert!(f.contains(&format!("drop move {b}")), "{f}");
}

#[test]
fn every_host_expr_kind_has_a_recorded_disposition() {
    // The lowering's `match` is wildcard-free, so this list is the human
    // record of the variants it covers today; the compiler proves totality.
    let lowered: BTreeSet<&str> = [
        "Int", "Float", "Bool", "String", "List", "Tuple", "Var", "Call", "Builtin",
        "AdtConstruct", "AdtFieldAccess", "If", "MatchOption", "MatchAdt", "Let", "Map",
        "Fold", "TensorCall", "Unit",
    ]
    .into_iter()
    .collect();
    let rejected: BTreeSet<&str> = ["Filter", "Scan", "Partition", "FlatMap", "WithSeed"]
        .into_iter()
        .collect();
    assert!(lowered.is_disjoint(&rejected));
    assert_eq!(lowered.len() + rejected.len(), 24, "24 `HostExprKind` variants");
    let mut program = ConcreteHostProgram::default();
    program.functions.push(HostFunction {
        name: "noop".to_string(),
        params: Vec::new(),
        ret_ty: ConcreteHostType::Unit,
        body: HostExpr::new(HostExprKind::Unit),
        tensor_helpers: Vec::new(),
        origin: HostFunctionOrigin::Authored,
        specialization: None,
        summary_rejections: Vec::new(),
    });
    assert_eq!(program.functions.len(), 1);
    assert_eq!(Lane::Host, Lane::Host);
}
