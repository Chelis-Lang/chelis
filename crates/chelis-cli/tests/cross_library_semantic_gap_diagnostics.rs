//! Wave 4 / W4-A — M5(c) acceptance oracle.
//!
//! Locks the public `SummaryRejection` diagnostic shape emitted when
//! a callsite is summary-eligible (helper body contains a sparse
//! `RiscOp::Gather`, `RiscOp::ScatterAdd`, or `RiscOp::Scatter`) but
//! the summarizer rejects.
//!
//! ## Diagnostic shape contract
//!
//! Every test in this file asserts the **exact** value of the emitted
//! [`SummaryRejection`] via pattern matching on:
//!
//!   * the [`SummaryRejectionClass`] variant,
//!   * the [`SummaryRejectionDetail`] variant and its structured fields,
//!   * the [`HelperPath::def_name`] field,
//!   * the presence / absence of span IDs on `callsite_span` and
//!     `helper_body_span`.
//!
//! `contains()` on the rendered `Display` string is **explicitly
//! rejected** — the W5 red-team thoroughness contract depends on the
//! enum being the matchable surface, not free text.
//!
//! ## Categories covered
//!
//! Each test name names the W3-B-enumerated category it locks:
//!
//!   1. `multiple_roots`                       — `SummaryRejectionClass::MultipleRoots`
//!   2. `multiple_return_paths`                — `SummaryRejectionClass::MultipleReturnPaths`
//!   3. `non_load_operand`                     — `SummaryRejectionClass::NonLoadOperand`
//!   4. `post_processing_after_sparse_op`      — `SummaryRejectionClass::PostProcessingAfterSparseOp`
//!   5. `indices_dtype_mismatch`               — `SummaryRejectionClass::IndicesDTypeMismatch`
//!   6. `payload_dtype_mismatch`               — `SummaryRejectionClass::PayloadDTypeMismatch`
//!   7. `wildcard_dim`                         — `SummaryRejectionClass::WildcardDim`
//!
//! The seven categories mirror the W3-B negative recognition tests in
//! `crates/chelis-ir/tests/host_sparse_summary.rs` and
//! `crates/chelis-cli/tests/cross_library_sparse_summaries.rs`, which
//! lock the *absence-of-summary* contract. This file's tests
//! complement those: same rejection categories, now also locking
//! *structured rejection emission* per W4-A.
//!
//! ## No false-positive section
//!
//! The "no false positives on summarized callsites" invariant is
//! locked at the end of this file: an accepted callsite from the
//! `cross_library_sparse_summaries.rs` positive surface must produce
//! an empty `summary_rejections` vector. If a rejection fires for a
//! summarized callsite, this section flags it before any downstream
//! consumer trusts the structured surface.

use chelis_ir::host::{
    HostFunctionSpecialization, SparseOpKind, SummaryRejection, SummaryRejectionClass,
    SummaryRejectionDetail, host_program_summary_rejections, lower_compiled_program,
};
use chelis_ir::{PayloadRole, WildcardLocation};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::types::Prim;
use chelis_types::{check_ir_program, check_linearity};

/// Drive the full parse → desugar → typecheck → effect → linearity →
/// host-lower chain on a flat Surf source. Returns the rejections
/// collected by `lower_compiled_program`. Panics on any front-end
/// error; the tests in this file expect well-typed input even when the
/// callsite is summary-rejected.
///
/// The seam mirrors the chain `cmd_build` in `crates/chelis-cli/src/main.rs`
/// runs before invoking the backend.
fn rejections_for_source(source: &str) -> Vec<SummaryRejection> {
    let decls = parse_str(source).expect("parse_str");
    let deep = desugar_program(&decls);
    let checked = check_ir_program(&deep).expect("check_ir_program");
    let checked = chelis_effects::check_program(&checked).expect("effect check");
    let checked = check_linearity(&checked).expect("linearity check");
    let compiled = lower_compiled_program(&checked);
    let host = compiled
        .host
        .expect("host lowering must produce a HostProgram");
    host_program_summary_rejections(&host).to_vec()
}

/// Find the (single) rejection in `rejections` whose helper_path's
/// `def_name` ends with `def_name_suffix` (Reef bindings come out with
/// module-prefix mangling, so an `endswith` check is the canonical
/// stable surface). Panics if there is no such rejection or if there
/// is more than one.
fn find_rejection<'a>(
    rejections: &'a [SummaryRejection],
    def_name_suffix: &str,
) -> &'a SummaryRejection {
    let matches: Vec<&SummaryRejection> = rejections
        .iter()
        .filter(|r| r.helper_path.def_name.ends_with(def_name_suffix))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one rejection for def `{def_name_suffix}`, got {}: {:#?}",
        matches.len(),
        rejections,
    );
    matches[0]
}

// =========================================================================
// Category 4: PostProcessingAfterSparseOp
//
// Helper `bad` post-processes `gather(...)` with an elementwise `add`.
// The helper DAG root is `add`, not `gather`, so the summarizer
// rejects with PostProcessingAfterSparseOp and names the tail op.
// =========================================================================

#[test]
fn post_processing_after_gather_emits_structured_rejection() {
    let source = "def bad(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  bad(table, indices, zero)\n";
    let rejections = rejections_for_source(source);
    let rejection = find_rejection(&rejections, "bad");

    // Exact pattern-match on the rejection class.
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::PostProcessingAfterSparseOp,
    );

    // Exact pattern-match on the structured detail.
    let SummaryRejectionDetail::PostProcessingAfterSparseOp { op, tail_op } = &rejection.detail
    else {
        panic!(
            "expected SummaryRejectionDetail::PostProcessingAfterSparseOp, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(*op, SparseOpKind::Gather);
    assert_eq!(tail_op, "add");
}

// =========================================================================
// Category 3: NonLoadOperand
//
// Helper inserts an `add` on the values operand before the gather.
// The Gather node's first input is `add(table, zero)`, not a direct
// Load, so the recognizer rejects with NonLoadOperand at operand_index = 0.
// =========================================================================

#[test]
fn non_load_operand_on_gather_emits_structured_rejection() {
    let source = "def bad(table: tensor[1000, 128, f32], zero: tensor[1000, 128, f32], \
                  indices: tensor[64, int64]) -> tensor[64, 128, f32] = \
                  gather(add(table, zero), indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], zero: tensor[1000, 128, f32], \
                  indices: tensor[64, int64]) -> tensor[64, 128, f32] = \
                  bad(table, zero, indices)\n";
    let rejections = rejections_for_source(source);
    let rejection = find_rejection(&rejections, "bad");

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::NonLoadOperand,
    );
    let SummaryRejectionDetail::NonLoadOperand { op, operand_index } = &rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::NonLoadOperand, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(*op, SparseOpKind::Gather);
    assert_eq!(*operand_index, 0);
}

// =========================================================================
// Category 2: MultipleReturnPaths
//
// Helper body branches via if/then/else; the function body lowers to
// `HostExprKind::If` and never reaches the tensor-helper summarizer.
// The function-level pass detects the multi-return-path case and
// emits the structured rejection at the outer If's span.
// =========================================================================

#[test]
fn multiple_return_paths_emits_structured_rejection() {
    let source = "def bad(flag: bool, table: tensor[1000, 128, f32], \
                  ind_a: tensor[64, int64], ind_b: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = \
                  if flag then gather(table, ind_a, 0) \
                  else gather(table, ind_b, 0)\n\
                  def f(flag: bool, table: tensor[1000, 128, f32], \
                  ind_a: tensor[64, int64], ind_b: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = bad(flag, table, ind_a, ind_b)\n";
    let rejections = rejections_for_source(source);
    let rejection = find_rejection(&rejections, "bad");

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::MultipleReturnPaths,
    );
    let SummaryRejectionDetail::MultipleReturnPaths { branch_count } = &rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::MultipleReturnPaths, got {:?}",
            rejection.detail,
        );
    };
    assert!(
        *branch_count >= 2,
        "two-arm if/then/else must report >= 2 branches; got {branch_count}",
    );
}

// =========================================================================
// Category 7: WildcardDim
//
// Top-level expression body whose tensor types resolve to synthetic
// `Named("*", None)` wildcard dims (pad_sequences + under-constrained
// gather). The recognizer rejects.
// =========================================================================

#[test]
fn wildcard_dim_top_level_gather_emits_structured_rejection() {
    let source = "lhs = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)\n\
                  ids_list: List[int64] = [cast(0, int64), cast(1, int64)]\n\
                  token_ids = to_tensor(ids_list)\n\
                  result = gather(lhs, token_ids, 0)\n";
    let rejections = rejections_for_source(source);

    // Wildcard rejections fire on the top-level expression's tensor
    // helper, not on a user-named def. The rejection's def_name will
    // therefore be the synthetic global helper name. Find any
    // rejection of class WildcardDim and assert its detail.
    let wildcard: Vec<&SummaryRejection> = rejections
        .iter()
        .filter(|r| r.rejection_class == SummaryRejectionClass::WildcardDim)
        .collect();
    assert_eq!(
        wildcard.len(),
        1,
        "expected exactly one WildcardDim rejection; got {wildcard:?}",
    );
    let rejection = wildcard[0];
    let SummaryRejectionDetail::WildcardDim { location } = &rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::WildcardDim, got {:?}",
            rejection.detail,
        );
    };
    // The location must be either Output or Input(i) — both are valid
    // surface shapes for this fixture. We pattern-match on the enum
    // variant rather than on a string.
    match location {
        WildcardLocation::Output | WildcardLocation::Input(_) => {}
    }
}

// =========================================================================
// Category 1: MultipleRoots
//
// The Surf surface today does not have a path that lowers to a
// multi-root helper DAG (every Surf `def` body is a single expression
// whose tensor lane produces a single root). The IR-level companion
// test `multiple_roots_synthetic_helper_emits_structured_rejection`
// in `crates/chelis-ir/tests/host_sparse_summary_diagnostics.rs`
// drives the synthetic case mechanically and locks the structured
// rejection there.
//
// We assert the negative companion here: no Surf source in this file
// produces a `MultipleRoots` rejection (would indicate the
// host-lowering pipeline started emitting multi-root DAGs from valid
// surface, which would itself be a bug).
#[test]
fn surface_sources_do_not_emit_multiple_roots_rejection() {
    // Reuse one of the surface sources that *does* produce a
    // structured rejection to confirm `MultipleRoots` isn't fired in
    // the same compilation.
    let source = "def bad(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  bad(table, indices, zero)\n";
    let rejections = rejections_for_source(source);
    assert!(
        !rejections
            .iter()
            .any(|r| r.rejection_class == SummaryRejectionClass::MultipleRoots),
        "no Surf surface should emit MultipleRoots today; rejections were {rejections:#?}",
    );
}

// =========================================================================
// Category 5: IndicesDTypeMismatch
//
// Indices precision is f32 instead of int32 / int64. Today the
// chelis type system rejects this at the parse/check stage (the Surf
// signature `tensor[64, int64]` is required for the indices operand);
// the IR-level companion test drives a synthetic helper with bogus
// indices precision and locks the structured rejection mechanically
// (`crates/chelis-ir/tests/host_sparse_summary_diagnostics.rs`).
//
// At the CLI level we assert the negative companion: a well-typed
// Surf source must NOT produce an `IndicesDTypeMismatch` rejection
// (any int32/int64 indices is accepted).
#[test]
fn well_typed_surface_does_not_emit_indices_dtype_mismatch() {
    let source = "def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n";
    let rejections = rejections_for_source(source);
    assert!(
        !rejections
            .iter()
            .any(|r| r.rejection_class == SummaryRejectionClass::IndicesDTypeMismatch),
        "well-typed gather over int64 indices must not emit IndicesDTypeMismatch; rejections: {rejections:#?}",
    );
}

// =========================================================================
// Category 6: PayloadDTypeMismatch
//
// Payload precision (values/target/updates) disagrees with output.
// As with IndicesDTypeMismatch, the Surf type checker enforces
// payload precision agreement, so this case is exercised at the IR
// level via a synthetic helper. The IR companion test
// `payload_dtype_mismatch_scatter_add_emits_structured_rejection`
// locks the structured rejection mechanically.
//
// CLI-level negative companion: well-typed payload precision must
// not emit PayloadDTypeMismatch.
#[test]
fn well_typed_surface_does_not_emit_payload_dtype_mismatch() {
    let source = "def f(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n";
    let rejections = rejections_for_source(source);
    assert!(
        !rejections
            .iter()
            .any(|r| r.rejection_class == SummaryRejectionClass::PayloadDTypeMismatch),
        "well-typed scatter_replace must not emit PayloadDTypeMismatch; rejections: {rejections:#?}",
    );
}

// =========================================================================
// HelperPath, callsite_span, helper_body_span — joint contract
//
// The diagnostic must name the helper, the callsite span, and the
// helper body span. These tests lock that the structured fields are
// populated meaningfully, not pattern-match on the rendered string.
// =========================================================================

#[test]
fn rejection_carries_helper_def_name() {
    let source = "def bad(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  bad(table, indices, zero)\n";
    let rejections = rejections_for_source(source);
    let rejection = find_rejection(&rejections, "bad");
    assert_eq!(rejection.helper_path.module, None);
    assert!(
        rejection.helper_path.def_name.ends_with("bad"),
        "helper_path.def_name must name the owning def `bad`; got {:?}",
        rejection.helper_path.def_name,
    );
}

#[test]
fn rejection_carries_surf_spans_when_available() {
    // Surf source goes through the Surf desugarer (which threads
    // `surf:<start>..<end>` IDs into Deep `meta["span"]`), so the
    // rejected callsite + helper body must carry surf-prefixed span
    // IDs. We assert presence + prefix, not pattern-match on byte
    // offsets (those are fragile under source-formatting changes).
    let source = "def bad(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  bad(table, indices, zero)\n";
    let rejections = rejections_for_source(source);
    let rejection = find_rejection(&rejections, "bad");
    // At least one of callsite_span / helper_body_span must be a
    // surf-prefixed span ID for surface-driven inputs. (Both being
    // None is the synthetic-fixture case, which is locked separately
    // at the IR level.)
    let has_surf_span = rejection
        .callsite_span
        .as_deref()
        .is_some_and(|s| s.starts_with("surf:"))
        || rejection
            .helper_body_span
            .as_deref()
            .is_some_and(|s| s.starts_with("surf:"));
    assert!(
        has_surf_span,
        "surface-driven rejection must carry at least one surf-prefixed span; got callsite={:?}, body={:?}",
        rejection.callsite_span, rejection.helper_body_span,
    );
}

// =========================================================================
// No false positives on summarized callsites
//
// The full positive surface from `cross_library_sparse_summaries.rs`:
// every accepted callsite (gather, scatter_replace, nested wrappers)
// MUST produce zero rejections. If a summarized callsite is also
// reported as rejected, downstream consumers can't trust the
// structured surface.
// =========================================================================

#[test]
fn summarized_gather_emits_no_rejections() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n";
    let rejections = rejections_for_source(source);
    assert!(
        rejections.is_empty(),
        "summarized gather callsite must not emit any rejection; got {rejections:#?}",
    );
}

#[test]
fn summarized_scatter_replace_emits_no_rejections() {
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n";
    let rejections = rejections_for_source(source);
    assert!(
        rejections.is_empty(),
        "summarized scatter_replace callsite must not emit any rejection; got {rejections:#?}",
    );
}

#[test]
fn summarized_nested_gather_wrapper_emits_no_rejections() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def wrap_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = wrap_g(table, indices)\n";
    let rejections = rejections_for_source(source);
    assert!(
        rejections.is_empty(),
        "nested-wrapper summarized gather must not emit any rejection; got {rejections:#?}",
    );
}

// =========================================================================
// Cross-cutting invariant: the same source produces the same
// summarized-vs-rejected outcome that `cross_library_sparse_summaries.rs`
// asserts at the C-emission level. Locking this guards against the
// W4-A path silently introducing different rejection decisions from
// the W3-B summarization path.
// =========================================================================

#[test]
fn summarized_gather_helper_has_specialization() {
    // Cross-check: the same source whose `f` body should hit the
    // inline sparse-gather loop in C (per `cross_library_sparse_summaries.rs`'s
    // `user_def_gather_helper_emits_inline_sparse_gather_loop`) must
    // have a `SparseGather` specialization on both `my_g` and `f`
    // *and* no rejection.
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n";
    let decls = parse_str(source).expect("parse_str");
    let deep = desugar_program(&decls);
    let checked = check_ir_program(&deep).expect("check_ir_program");
    let checked = chelis_effects::check_program(&checked).expect("effect check");
    let checked = check_linearity(&checked).expect("linearity check");
    let compiled = lower_compiled_program(&checked);
    let host = compiled
        .host
        .expect("host lowering must produce a HostProgram");

    assert!(
        host.summary_rejections.is_empty(),
        "summarized callsite must not register any rejection; got {:#?}",
        host.summary_rejections,
    );

    let my_g_spec = host
        .functions
        .iter()
        .find(|f| f.name.ends_with("my_g"))
        .expect("my_g must lower to a host function")
        .specialization
        .as_ref()
        .expect("my_g must have a function specialization");
    assert!(
        matches!(my_g_spec, HostFunctionSpecialization::SparseGather(_)),
        "my_g must be specialized as SparseGather; got {my_g_spec:?}",
    );
}

// =========================================================================
// Display surface sanity (NOT the contract — supplemental coverage)
//
// `Display` is provided for human-readable diagnostic emission. We
// lock that the rendered string mentions the helper name and the
// rejection class, but the rendering itself is NOT the matchable
// contract — downstream consumers must pattern-match on the enum.
// =========================================================================

#[test]
fn rejection_display_mentions_helper_and_class() {
    let source = "def bad(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  bad(table, indices, zero)\n";
    let rejections = rejections_for_source(source);
    let rejection = find_rejection(&rejections, "bad");
    let rendered = rejection.to_string();
    // Supplemental — NOT a contract test. The matchable contract is
    // the enum variant + struct fields, locked above. This assertion
    // exists so a regression that emits an empty / unhelpful Display
    // surfaces immediately, but is intentionally weak vs. the
    // pattern-match assertions.
    assert!(
        rendered.contains("post-processing-after-sparse-op"),
        "Display must include the rejection-class kebab-case name; got {rendered}",
    );
}

// =========================================================================
// Negative parity with the IR-level synthetic tests
//
// The IR-level companion (`crates/chelis-ir/tests/host_sparse_summary_diagnostics.rs`)
// covers the categories the Surf type system blocks at the surface
// (categories 1, 5, 6 — MultipleRoots, IndicesDTypeMismatch,
// PayloadDTypeMismatch). Those tests drive synthetic helper DAGs
// through `try_summarize_sparse_helper_for_test` and pattern-match on
// the returned `SparseSummaryAttempt::Rejected` variant.
//
// The CLI-level test surface (this file) covers the categories that
// reach the surface chain: PostProcessingAfterSparseOp, NonLoadOperand,
// MultipleReturnPaths, WildcardDim. All seven categories from the
// W3-B enumeration are jointly locked across the two files.
// =========================================================================

#[test]
fn all_seven_rejection_classes_have_distinct_variants() {
    // Compile-time-checked: all seven Class variants are distinct.
    // (PartialEq on the enum is locked by the derive; this test
    // ensures no two variant-pairs were accidentally introduced as
    // alias spellings.)
    let classes = [
        SummaryRejectionClass::MultipleRoots,
        SummaryRejectionClass::MultipleReturnPaths,
        SummaryRejectionClass::NonLoadOperand,
        SummaryRejectionClass::PostProcessingAfterSparseOp,
        SummaryRejectionClass::IndicesDTypeMismatch,
        SummaryRejectionClass::PayloadDTypeMismatch,
        SummaryRejectionClass::WildcardDim,
    ];
    for i in 0..classes.len() {
        for j in (i + 1)..classes.len() {
            assert_ne!(
                classes[i], classes[j],
                "variants {i} and {j} compared equal -- must be distinct",
            );
        }
    }
}

// =========================================================================
// Per-W4-A-brief, mirror the W3-B `host_sparse_summary.rs` rejection
// categories that drive synthetic helper DAGs. Items below
// cross-reference the synthetic-helper IR tests but assert the
// rejections are observable through the surface chain.
// =========================================================================

#[test]
fn payload_role_variants_are_distinct() {
    let roles = [
        PayloadRole::Values,
        PayloadRole::Target,
        PayloadRole::Updates,
        PayloadRole::Output,
    ];
    for i in 0..roles.len() {
        for j in (i + 1)..roles.len() {
            assert_ne!(
                roles[i], roles[j],
                "PayloadRole variants {i} and {j} must be distinct",
            );
        }
    }
}

#[test]
fn sparse_op_kind_variants_match_risc_ops() {
    // Each variant must round-trip through Display to its
    // canonical Surf builtin name. This is a sanity check that
    // downstream consumers comparing on names get stable strings.
    assert_eq!(SparseOpKind::Gather.to_string(), "gather");
    assert_eq!(SparseOpKind::ScatterAdd.to_string(), "scatter_add");
    assert_eq!(SparseOpKind::ScatterReplace.to_string(), "scatter_replace");
    assert_eq!(SparseOpKind::Unknown.to_string(), "<unknown>");
}

// =========================================================================
// Future-extension reserved classes (UnrecognizedShape,
// NonContiguousLayout, RankMismatch) — these are in the public enum
// surface but the current recognizer does not emit them. We assert
// they are reachable through Display (no panic) and that no
// surface fixture in this file unexpectedly emits them.
// =========================================================================

#[test]
fn reserved_rejection_classes_have_display() {
    // Just confirm the Display impl handles every reserved variant
    // without panicking and yields the canonical kebab-case name.
    assert_eq!(
        SummaryRejectionClass::UnrecognizedShape.to_string(),
        "unrecognized-shape",
    );
    assert_eq!(
        SummaryRejectionClass::NonContiguousLayout.to_string(),
        "non-contiguous-layout",
    );
    assert_eq!(
        SummaryRejectionClass::RankMismatch.to_string(),
        "rank-mismatch",
    );
}

#[test]
fn well_typed_surface_does_not_emit_reserved_classes() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n";
    let rejections = rejections_for_source(source);
    for r in &rejections {
        assert!(
            !matches!(
                r.rejection_class,
                SummaryRejectionClass::UnrecognizedShape
                    | SummaryRejectionClass::NonContiguousLayout
                    | SummaryRejectionClass::RankMismatch,
            ),
            "well-typed surface must not emit reserved-class rejection; got {r:?}",
        );
    }
}

// =========================================================================
// Wave 6 Task A — BLAS rejection diagnostics through the Surf surface.
//
// Mirrors the sparse-side rejection tests above. Six BLAS-prefixed
// `SummaryRejectionClass` variants — only the ones reachable through
// the Surf user-def matmul-helper path are exercised here; the IR-
// level companion `crates/chelis-ir/tests/host_blas_summary_diagnostics.rs`
// drives the synthetic-DAG-only cases (`BlasMultipleRoots`,
// `BlasNotMatmulPattern` subcase B, `BlasNonLoadOperand` via Const,
// `BlasInputPrecisionMismatch` via mismatched Load precision,
// `BlasDimensionBindingFailure` via unbound symbolic dims).
//
// Surface-reachable here:
//   * `BlasOutputPrecisionMismatch` — `def my_mm(a: ..f64, b: ..f64)
//     -> ..f64 = matmul(a, b)` — the W5 P0 silent rejection is now
//     diagnosed.
//
// Surf-blocked at typecheck (and therefore tested only at the IR
// level):
//   * F32 lhs × Int32 rhs matmul → blocked by
//     `check_matmul_signature` (precision mismatch error).
//   * Hand-built multi-root BLAS DAG → no Surf source produces it.
//
// No false positive on accepted F32 matmul: covered below.
// =========================================================================

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is now admitted (cblas_dgemm via the cycle's F64-completeness lift); the W5 P0 BlasOutputPrecisionMismatch path no longer fires for F64 helpers. Bf16/F16 (WS-A3) and the integer family (rejected upstream at the type checker, spec §5.7.2) cover the remaining rejection paths."]
fn surface_f64_matmul_helper_emits_blas_output_precision_mismatch_rejection() {
    // F64 user-`def` matmul helper. The W5 P0 fix kept this off the
    // `RiscOp::BlasMatmul` path silently; W6 Task A upgrades that
    // silence to a structured rejection.
    let source = "def my_mm(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);

    // At least one BLAS-precision rejection MUST fire for the F64
    // helper. (Multiple helpers may register rejections — the inner
    // `my_mm` and the outer `f` both lower through helper paths.)
    let blas_rejections: Vec<&SummaryRejection> = rejections
        .iter()
        .filter(|r| r.rejection_class == SummaryRejectionClass::BlasOutputPrecisionMismatch)
        .collect();
    assert!(
        !blas_rejections.is_empty(),
        "F64 matmul helper MUST emit at least one BlasOutputPrecisionMismatch \
         rejection -- W5 P0 silent rejection is now diagnosed. \
         Got rejections: {rejections:#?}",
    );
    let rejection = blas_rejections[0];

    let SummaryRejectionDetail::BlasOutputPrecisionMismatch { observed } = &rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::BlasOutputPrecisionMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(*observed, Prim::F64);
}

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted; the rejection this test asserts no longer fires."]
fn surface_f64_matmul_rejection_carries_surf_span_when_available() {
    // The surface-driven F64 helper's rejection must carry a
    // surf-prefixed span on at least one of callsite_span /
    // helper_body_span. Locks the same contract the sparse-side
    // tests assert in this file (W4-A) — the BLAS path also
    // propagates Surf spans through the rejection.
    let source = "def my_mm(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    let rejection = rejections
        .iter()
        .find(|r| r.rejection_class == SummaryRejectionClass::BlasOutputPrecisionMismatch)
        .expect("must emit BlasOutputPrecisionMismatch for F64 helper");
    let has_surf_span = rejection
        .callsite_span
        .as_deref()
        .is_some_and(|s| s.starts_with("surf:"))
        || rejection
            .helper_body_span
            .as_deref()
            .is_some_and(|s| s.starts_with("surf:"));
    assert!(
        has_surf_span,
        "surface-driven BLAS rejection must carry at least one surf-prefixed span; \
         got callsite={:?}, body={:?}",
        rejection.callsite_span, rejection.helper_body_span,
    );
}

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted; the rejection this test asserts no longer fires."]
fn surface_f64_matmul_rejection_carries_helper_def_name() {
    // Same source as above; locks that the helper_path.def_name
    // names the F64 def (`my_mm`), mirroring the sparse-side
    // `rejection_carries_helper_def_name` test above.
    let source = "def my_mm(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    // Find the rejection on def `my_mm` (the F64 helper). Some
    // surfaces also register a rejection on the caller `f` — we
    // care that `my_mm` itself is named.
    let my_mm_rejection = rejections
        .iter()
        .find(|r| {
            r.rejection_class == SummaryRejectionClass::BlasOutputPrecisionMismatch
                && r.helper_path.def_name.ends_with("my_mm")
        })
        .unwrap_or_else(|| {
            panic!(
                "expected a BlasOutputPrecisionMismatch rejection on def `my_mm`; \
             got rejections: {rejections:#?}",
            )
        });
    assert_eq!(my_mm_rejection.helper_path.module, None);
    assert!(
        my_mm_rejection.helper_path.def_name.ends_with("my_mm"),
        "helper_path.def_name must name `my_mm`; got {:?}",
        my_mm_rejection.helper_path.def_name,
    );
}

#[test]
fn surface_f32_matmul_helper_emits_no_blas_rejection() {
    // No false positive: a well-typed F32 matmul helper must NOT
    // emit a BLAS rejection (it summarizes via the BlasMatmul path).
    let source = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    for r in &rejections {
        assert!(
            !matches!(
                r.rejection_class,
                SummaryRejectionClass::BlasMultipleRoots
                    | SummaryRejectionClass::BlasOutputPrecisionMismatch
                    | SummaryRejectionClass::BlasNotMatmulPattern
                    | SummaryRejectionClass::BlasNonLoadOperand
                    | SummaryRejectionClass::BlasInputPrecisionMismatch
                    | SummaryRejectionClass::BlasDimensionBindingFailure
            ),
            "F32 matmul helper must NOT emit any Blas* rejection; got {r:?}",
        );
    }
}

#[test]
fn surface_elementwise_helper_emits_no_blas_rejection() {
    // Negative: a non-matmul elementwise helper must NOT trigger
    // any BLAS rejection (no false positive on the
    // not-matmul-near path).
    let source = "def my_add(a: tensor[4, 4, f64], b: tensor[4, 4, f64]) \
                  -> tensor[4, 4, f64] = add(a, b)\n\
                  def f(a: tensor[4, 4, f64], b: tensor[4, 4, f64]) \
                  -> tensor[4, 4, f64] = my_add(a, b)\n";
    let rejections = rejections_for_source(source);
    for r in &rejections {
        assert!(
            !matches!(
                r.rejection_class,
                SummaryRejectionClass::BlasMultipleRoots
                    | SummaryRejectionClass::BlasOutputPrecisionMismatch
                    | SummaryRejectionClass::BlasNotMatmulPattern
                    | SummaryRejectionClass::BlasNonLoadOperand
                    | SummaryRejectionClass::BlasInputPrecisionMismatch
                    | SummaryRejectionClass::BlasDimensionBindingFailure
            ),
            "F64 elementwise helper (not matmul-near) must NOT emit any Blas* rejection; \
             got {r:?}",
        );
    }
}

#[test]
#[ignore = "WS-A2: F64 BLAS matmul is admitted; the rejection this test asserts no longer fires."]
fn surface_f64_matmul_rejection_display_mentions_blas_precision() {
    // Supplemental Display surface check — not the matchable
    // contract. Mirrors `rejection_display_mentions_helper_and_class`
    // for the BLAS path.
    let source = "def my_mm(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = matmul(a, b)\n\
                  def f(a: tensor[8, 16, f64], b: tensor[16, 4, f64]) \
                  -> tensor[8, 4, f64] = my_mm(a, b)\n";
    let rejections = rejections_for_source(source);
    let rejection = rejections
        .iter()
        .find(|r| r.rejection_class == SummaryRejectionClass::BlasOutputPrecisionMismatch)
        .expect("must emit BlasOutputPrecisionMismatch for F64 helper");
    let rendered = rejection.to_string();
    assert!(
        rendered.contains("blas-output-precision-mismatch"),
        "Display must include the BLAS rejection-class kebab-case name; got {rendered}",
    );
}

#[test]
fn six_blas_rejection_classes_have_distinct_variants() {
    // Compile-time-checked: all six BLAS variants are distinct from
    // each other AND distinct from the seven sparse variants.
    // Mirrors `all_seven_rejection_classes_have_distinct_variants`.
    let blas = [
        SummaryRejectionClass::BlasMultipleRoots,
        SummaryRejectionClass::BlasOutputPrecisionMismatch,
        SummaryRejectionClass::BlasNotMatmulPattern,
        SummaryRejectionClass::BlasNonLoadOperand,
        SummaryRejectionClass::BlasInputPrecisionMismatch,
        SummaryRejectionClass::BlasDimensionBindingFailure,
    ];
    for i in 0..blas.len() {
        for j in (i + 1)..blas.len() {
            assert_ne!(
                blas[i], blas[j],
                "BLAS variants {i} and {j} compared equal -- must be distinct",
            );
        }
    }
    let sparse = [
        SummaryRejectionClass::MultipleRoots,
        SummaryRejectionClass::MultipleReturnPaths,
        SummaryRejectionClass::NonLoadOperand,
        SummaryRejectionClass::PostProcessingAfterSparseOp,
        SummaryRejectionClass::IndicesDTypeMismatch,
        SummaryRejectionClass::PayloadDTypeMismatch,
        SummaryRejectionClass::WildcardDim,
    ];
    for b in &blas {
        for s in &sparse {
            assert_ne!(
                b, s,
                "BLAS variant {b:?} must NOT collapse onto sparse {s:?}"
            );
        }
    }
}

#[test]
fn six_blas_rejection_classes_have_distinct_display() {
    // Locks the Display strings; downstream tooling that filters
    // diagnostics by string class name (a non-contract surface)
    // gets stable values.
    let pairs = [
        (
            SummaryRejectionClass::BlasMultipleRoots,
            "blas-multiple-roots",
        ),
        (
            SummaryRejectionClass::BlasOutputPrecisionMismatch,
            "blas-output-precision-mismatch",
        ),
        (
            SummaryRejectionClass::BlasNotMatmulPattern,
            "blas-not-matmul-pattern",
        ),
        (
            SummaryRejectionClass::BlasNonLoadOperand,
            "blas-non-load-operand",
        ),
        (
            SummaryRejectionClass::BlasInputPrecisionMismatch,
            "blas-input-precision-mismatch",
        ),
        (
            SummaryRejectionClass::BlasDimensionBindingFailure,
            "blas-dimension-binding-failure",
        ),
    ];
    for (class, expected) in &pairs {
        assert_eq!(class.to_string(), *expected);
    }
}

// =========================================================================
// Dead-code reachability — keep `Prim` used so the import does not warn.
// =========================================================================

#[test]
fn prim_import_used_in_payload_dtype_mismatch_pattern() {
    // This is a sanity hook for the unused-import warning. The Prim
    // type appears in `SummaryRejectionDetail::PayloadDTypeMismatch`
    // and `IndicesDTypeMismatch`; binding a value of each variant
    // exercises the import path.
    let _ = SummaryRejectionDetail::IndicesDTypeMismatch {
        op: SparseOpKind::Gather,
        observed: Prim::F32,
    };
    let _ = SummaryRejectionDetail::PayloadDTypeMismatch {
        op: SparseOpKind::ScatterAdd,
        which: PayloadRole::Updates,
        expected: Prim::F32,
        observed: Prim::F64,
    };
}
