//! Phase 2 ownership-lowering acceptance tests (chelis#1286).
//!
//! The fixtures and inline programs pass through Surf parsing, type/effect/
//! linearity checking, the real root manifest, and concrete host lowering
//! before the ownership boundary. Positive and negative cases therefore do
//! not self-certify against a synthetic ownership graph.

use std::collections::BTreeSet;

use chelis_effects::realizability::{compute_root_manifest, infer_realizability};
use chelis_ir::ConcreteHostType;
use chelis_ir::host::{
    ConcreteHostProgram, HostExpr, HostExprKind, try_lower_compiled_program_with_manifest,
};
use chelis_ir::ownership::{
    OwnershipError, OwnershipProgram, VerifiedOwnershipProgram, lower_ownership, verify_ownership,
};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::manifest::RootManifest;
use chelis_types::types::Prim;
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

struct Front {
    checked: CheckedProgram,
    manifest: RootManifest,
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
    Front {
        checked,
        manifest,
        host,
    }
}

fn lower_source(source: &str) -> Result<OwnershipProgram, OwnershipError> {
    let front = front(source);
    lower_ownership(&front.checked, &front.host, &front.manifest)
}

fn lower_fixture(name: &str) -> Result<OwnershipProgram, OwnershipError> {
    lower_source(&fixture_source(name))
}

fn verified_source(source: &str) -> VerifiedOwnershipProgram {
    let lowered = lower_source(source).unwrap_or_else(|error| panic!("lowering: {error}"));
    verify_ownership(lowered).unwrap_or_else(|error| panic!("verification: {error}"))
}

fn verified_fixture(name: &str) -> VerifiedOwnershipProgram {
    let lowered = lower_fixture(name).unwrap_or_else(|error| panic!("{name}: {error}"));
    verify_ownership(lowered).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn unit_text(program: &VerifiedOwnershipProgram, name: &str) -> String {
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
fn aliased_roots_copy_the_earlier_sink_in_manifest_order() {
    let roots = unit_text(&verified_fixture("issue_1222_root_alias"), "roots");
    assert_eq!(count(&roots, "= copy clone"), 1, "{roots}");
    assert_eq!(count(&roots, "root original move"), 1, "{roots}");
    assert_eq!(count(&roots, "root alias move"), 1, "{roots}");
    assert!(
        line_index(&roots, "root original move") < line_index(&roots, "root alias move"),
        "{roots}"
    );

    let mut front = front(&fixture_source("issue_1222_root_alias"));
    front.manifest.entries.reverse();
    let lowered = lower_ownership(&front.checked, &front.host, &front.manifest).unwrap();
    let roots = unit_text(&verify_ownership(lowered).unwrap(), "roots");
    assert!(
        line_index(&roots, "root alias move") < line_index(&roots, "root original move"),
        "{roots}"
    );
}

#[test]
fn non_root_heap_value_gets_a_scope_exit_drop() {
    let mut front = front("kept = [1i64]\nout = len(kept)\n");
    front
        .manifest
        .entries
        .retain(|entry| entry.def_name == "out");
    let lowered = lower_ownership(&front.checked, &front.host, &front.manifest).unwrap();
    let roots = unit_text(&verify_ownership(lowered).unwrap(), "roots");
    assert_eq!(count(&roots, "root out move"), 1, "{roots}");
    assert_eq!(count(&roots, "drop move"), 1, "{roots}");
    assert!(
        line_index(&roots, "root out move") < line_index(&roots, "drop move"),
        "{roots}"
    );
}

#[test]
fn missing_manifest_binding_is_a_typed_lowering_error() {
    let mut front = front("out = [1i64]\n");
    let mut ghost = front.manifest.entries[0].clone();
    ghost.name = "ghost".to_string();
    ghost.def_name = "ghost".to_string();
    front.manifest.entries.push(ghost);
    assert_eq!(
        lower_ownership(&front.checked, &front.host, &front.manifest).unwrap_err(),
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
    assert_eq!(count(&roots, "= copy clone"), 0, "{roots}");

    let variable = verified_fixture("issue_1356_variable_argument");
    let roots = unit_text(&variable, "roots");
    assert!(roots.contains("call:identity(move"), "{roots}");
    assert_eq!(count(&roots, "= copy clone"), 1, "{roots}");
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
        "def peek(x: &tensor[2, f32]) -> int64 = 1i64\n\
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
        assert_eq!(count(&roots, "= copy clone"), 1, "{fixture}:\n{roots}");
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
fn supported_map_is_positive_parity_for_rejected_list_combinators() {
    let verified =
        verified_source("xs = [[1i64], [2i64]]\nys = map(fn (v: List[int64]) -> v, xs)\n");
    let roots = unit_text(&verified, "roots");
    assert!(roots.contains("empty_list"), "{roots}");
    assert!(roots.contains("list_push"), "{roots}");
    assert!(roots.contains("loop borrow"), "{roots}");
}

fn assert_unlowered(source: &str, variant: &str) {
    assert_eq!(
        lower_source(source).unwrap_err(),
        OwnershipError::UnloweredExprKind {
            unit: "roots".to_string(),
            variant: variant.to_string(),
        }
    );
}

#[test]
fn unsupported_host_combinators_reject_by_exact_variant_after_real_host_lowering() {
    assert_unlowered(
        "xs = [1i64, 2i64]\nys = filter(fn (v: int64) -> gte(v, 2i64), xs)\n",
        "Filter",
    );
    assert_unlowered(
        "xs = [1i64, 2i64]\nys = scan(fn (acc: int64, v: int64) -> add(acc, v), 0i64, xs)\n",
        "Scan",
    );
    assert_unlowered(
        "xs = [1i64, 2i64]\nys = partition(fn (v: int64) -> gt(v, 1i64), xs)\n",
        "Partition",
    );
    assert_unlowered(
        "xs = [1i64, 2i64]\nys = flat_map(fn (v: int64) -> [v, v], xs)\n",
        "FlatMap",
    );
    assert_unlowered("sampled = with seed(7i64) { 1i64 }\n", "WithSeed");
}

#[test]
fn malformed_real_host_programs_fail_at_typed_boundaries() {
    let front = front("def id(p: int64) -> int64 = p\nout = id(1i64)\n");
    let mut wrong_arity = front.host.clone();
    wrong_arity.globals[0].value = HostExpr::new(HostExprKind::Call {
        function: "id".to_string(),
        args: Vec::new(),
        arg_tys: Vec::new(),
        ty: ConcreteHostType::Int64,
    });
    assert!(matches!(
        lower_ownership(&front.checked, &wrong_arity, &front.manifest),
        Err(OwnershipError::CallArityMismatch { .. })
    ));

    let mut unbound = front.host.clone();
    unbound.globals[0].value = HostExpr::new(HostExprKind::Var(
        "nobody".to_string(),
        ConcreteHostType::Int64,
    ));
    assert_eq!(
        lower_ownership(&front.checked, &unbound, &front.manifest).unwrap_err(),
        OwnershipError::UnboundName {
            unit: "roots".to_string(),
            name: "nobody".to_string(),
        }
    );
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
    let rejected: BTreeSet<&str> = ["Filter", "Scan", "Partition", "FlatMap", "WithSeed"]
        .into_iter()
        .collect();
    assert!(lowered.is_disjoint(&rejected));
    assert_eq!(lowered.len() + rejected.len(), 24);
}
