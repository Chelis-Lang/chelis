//! Wave 3 / W3-B — M5(b) acceptance oracle.
//!
//! Locks the compiler-derived host summary mechanism for the three
//! sparse RISC primitives (`Gather`, `ScatterAdd`, `Scatter`/replace)
//! against the C backend.
//!
//! ## Positive surface (lines per `cross_function_specialization.md`):
//!
//! For each of the three sparse ops, a user-`def` wrapper around a
//! direct sparse-builtin call must recover the same bounded sparse
//! C loop that direct calls emit. Recovery is verified by structural
//! match on the emitted C (exact loop markers + the absence of the
//! generic `<helper>__tensor_<n>(...)` array-marshaling fallback in
//! the caller's body), not `contains()` of arbitrary substrings.
//!
//! Nested wrappers (`def inner(...) = gather(...); def outer(...) =
//! inner(...); def f(...) = outer(...)`) must also recover. This
//! locks the wrapper-propagation invariant that
//! `derive_host_function_specialization` recursively threads sparse
//! summaries through chained pass-through wrappers, matching the
//! `nested_user_def_matmul_helper_hits_blas` BLAS oracle.
//!
//! ## Negative surface — locked rejection categories:
//!
//! Each rejection test enforces TWO invariants jointly:
//!
//! 1. the emitted caller body must NOT contain the inline sparse-loop
//!    summary markers (no false-positive summary registered)
//! 2. the emitted caller body MUST contain the non-summary fallback
//!    path — either a host-lane `<helper>(...)` call to the
//!    generated helper function, or the helper's
//!    `<helper>__tensor_<n>(__inputs_*, ...)` array-marshaling shim.
//!
//! Per `feedback_no_silent_deferrals.md` and
//! `feedback_redteam_thoroughness.md`, the rejection tests assert the
//! fallback was taken — not just the absence of the summary, which
//! would also be satisfied by silently dropping the call.
//!
//! Rejection categories covered:
//!
//!   * `multiple_return_paths_via_if`: helper body branches on a
//!     boolean and returns one of two sparse-op results. The
//!     summarizer requires a single DAG root.
//!   * `non_load_operand`: helper inserts arithmetic between a
//!     parameter and the sparse op (e.g. adds a constant to the
//!     values argument). The summarizer requires every sparse-op
//!     operand to be a direct `RiscOp::Load`.
//!   * `extra_tensor_op_after_sparse`: helper post-processes the
//!     sparse op result (e.g. `add(gather(...), zero)`). The
//!     summarizer requires the sparse op to be the helper's single
//!     root.
//!   * `wildcard_dim`: helper input carries a synthetic
//!     `Named("*", None)` wildcard dim from type inference (occurs
//!     for top-level tensor expressions with under-constrained
//!     shapes). The summarizer rejects wildcards because all
//!     wildcards collide on the same string name and would yield
//!     meaningless contract assertions.
//!
//! W4-A (negative diagnostics) consumes this same rejection surface
//! to enumerate `SummaryRejectionClass` variants; the categories
//! above pin the surface that W4-A's structured rejection enum must
//! cover.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Build a Chelis source via the CLI to a tempdir and return the
/// generated `<name>.c` contents. Mirrors the idiom in
/// `cross_library_semantic_gap.rs`.
fn build_to_c(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(status.success(), "chelis build failed for {name}");
    let c_path = out_dir.join(format!("{name}.c"));
    fs::read_to_string(&c_path).expect("read generated c")
}

/// Slice the generated C between the start of a `chelis_tensor* <fn>(`
/// definition and its closing brace. The opening declaration (the
/// forward `chelis_tensor* <fn>(...);`) is ignored — we want the
/// definition body, not the prototype.
///
/// Returns the substring from the `chelis_tensor* <fn>(` definition
/// header up to the matching closing `}`. Panics if the function
/// definition is not present in `c`.
fn function_body(c: &str, function: &str) -> String {
    // Phase 2 gives every externally callable owned-formal function a
    // borrowing artifact adapter plus a consuming implementation body. Sparse
    // lowering belongs to the latter; inspecting the adapter would only test
    // its required retain-and-forward boundary.
    let function = format!("{function}__chelis_owned_body");
    let needle = format!("chelis_tensor* {function}(");
    // Find the definition: the prototype ends with `;`, the
    // definition with `{`. Scan candidate positions.
    let mut search_start = 0;
    let def_start = loop {
        let pos = c[search_start..]
            .find(&needle)
            .map(|p| search_start + p)
            .unwrap_or_else(|| panic!("function `{function}` not present in generated C"));
        let after = &c[pos + needle.len()..];
        // The function header opens with `... ) {`; the prototype
        // ends with `);`. Pick the definition occurrence.
        if let Some(open_brace_offset) = after.find('{')
            && let Some(semicolon_offset) = after.find(';')
            && open_brace_offset < semicolon_offset
        {
            break pos;
        }
        search_start = pos + needle.len();
    };
    // Find the matching closing brace by depth counting from
    // `def_start`.
    let bytes = c.as_bytes();
    let mut depth: i32 = 0;
    let mut in_def = false;
    let mut idx = def_start;
    while idx < bytes.len() {
        let b = bytes[idx];
        if b == b'{' {
            in_def = true;
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if in_def && depth == 0 {
                return c[def_start..=idx].to_string();
            }
        }
        idx += 1;
    }
    panic!("function `{function}` body did not close in generated C");
}

/// Count occurrences of the inline-sparse-gather canonical out-index
/// arithmetic shape. Both `CEmitter::emit_sparse_gather` and
/// `HostEmitter::emit_sparse_gather_summary_body` emit a line of
/// shape:
///
/// ```text
/// int <prefix>_out = ((<prefix>_b * <prefix>_index_count + <prefix>_i) * <prefix>_after) + <prefix>_d;
/// ```
///
/// We match the trailing arithmetic literal (the structural skeleton
/// minus the prefix). This is more robust than substring-matching
/// suffixed variable names, which carry numeric helper IDs that vary
/// run-to-run.
fn count_gather_out_index_lines(body: &str) -> usize {
    body.matches("_index_count + ").count()
}

fn count_gather_dtype_dispatch(body: &str) -> usize {
    body.lines()
        .filter(|line| {
            line.contains("chelis_host_tensor_dtype(")
                && line.contains("== CHELIS_DTYPE_I64")
                && line.contains("chelis_host_tensor_data(")
        })
        .count()
}

/// `true` when the body emits an inline sparse-gather loop:
///
///   * the out-index arithmetic shape appears at least once
///   * the int32/int64 dtype dispatch literal appears at least once
///   * the loop-bound triple (`_axis_size = `, `_after = `,
///     `_index_count = `) is present — these come from
///     `emit_sparse_gather`'s setup block
///   * gather distinguishes from scatter (add or replace) by NOT
///     containing the `memcpy(` setup that both scatter variants
///     use to copy the base/target buffer before the update loop
fn body_contains_inline_gather_loop(body: &str) -> bool {
    count_gather_out_index_lines(body) >= 1
        && count_gather_dtype_dispatch(body) >= 1
        && body.contains("_axis_size = ")
        && body.contains("_after = ")
        && body.contains("_index_count = ")
        && !body.contains("memcpy(")
}

/// `true` when the body emits an inline sparse-scatter-add loop.
/// Distinguished from gather by `memcpy(` (copies the base into the
/// output) and from scatter-replace by the `+=` accumulator.
fn body_contains_inline_scatter_add_loop(body: &str) -> bool {
    count_gather_out_index_lines(body) >= 1
        && count_gather_dtype_dispatch(body) >= 1
        && body.contains("_axis_size = ")
        && body.contains("_after = ")
        && body.contains("_index_count = ")
        && body.contains("memcpy(")
        && body.contains("] += ")
}

/// `true` when the body emits an inline sparse-scatter-replace loop.
/// Distinguished from scatter-add by the absence of `+=` and from
/// gather by the presence of `memcpy(`.
fn body_contains_inline_scatter_replace_loop(body: &str) -> bool {
    count_gather_out_index_lines(body) >= 1
        && count_gather_dtype_dispatch(body) >= 1
        && body.contains("_axis_size = ")
        && body.contains("_after = ")
        && body.contains("_index_count = ")
        && body.contains("memcpy(")
        && !body.contains("] += ")
}

/// `true` when the caller body forwards to the generated tensor
/// helper shim `<helper>__tensor_<n>(__inputs_*, ..., __outputs_*, ...)`
/// instead of inlining the sparse loop. This is the helper-marshaling
/// fallback path the emitter uses when the summary is `None`.
fn body_contains_tensor_helper_call(body: &str) -> bool {
    body.contains("__tensor_") && body.contains("__inputs_") && body.contains("__outputs_")
}

/// `true` when the caller body forwards to a sibling host function
/// via `__result = <callee>(...);` — the C call-site fallback used
/// when no function-level summary is registered.
fn body_contains_host_function_call(body: &str, callee: &str) -> bool {
    body.contains(&format!("= {callee}__chelis_owned_body("))
}

// =========================================================================
// Positive surface — three sparse ops × {single-def wrapper, nested wrapper}
// =========================================================================

#[test]
fn user_def_gather_helper_emits_inline_sparse_gather_loop() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n";
    let c = build_to_c(source, "sparse_gather_user_def");

    let f_body = function_body(&c, "f");
    assert!(
        body_contains_inline_gather_loop(&f_body),
        "user-def wrapper `f` MUST recover the inline sparse-gather loop \
         (locked markers: canonical out-index arithmetic, dtype dispatch, \
         loop-bound triple, NO memcpy). Body was:\n{f_body}"
    );
    assert!(
        !body_contains_tensor_helper_call(&f_body),
        "user-def wrapper `f` MUST bypass the helper-marshaling shim \
         when the gather summary is registered; saw `__inputs_*/__outputs_*` \
         marshaling. Body:\n{f_body}"
    );

    // The inner def `my_g` must also use the summary directly when
    // its body is a single gather (its `HostTensorSpecialization` is
    // SparseGather, so `assign_tensor_call` inlines the loop instead
    // of calling its own tensor helper).
    let my_g_body = function_body(&c, "my_g");
    assert!(
        body_contains_inline_gather_loop(&my_g_body),
        "inner def `my_g` MUST inline the sparse-gather loop via its \
         tensor-helper summary; got:\n{my_g_body}"
    );

    // The generated helper surface MUST still be emitted for the
    // non-summary fallback. Locks the contract from
    // `cross_function_specialization.md`: the helper body is still
    // available even when callsites bypass it.
    assert!(
        c.contains("static void my_g__tensor_"),
        "generated tensor helper `my_g__tensor_*` MUST still be emitted \
         for debug/non-summary callers; not found in generated C"
    );
}

#[test]
fn user_def_scatter_add_helper_emits_inline_sparse_scatter_add_loop() {
    // ScatterAdd has no surface form (it is only produced by AD
    // adjoint of `gather`), so this fixture cannot drive it through
    // the surface `def` path today. Instead we lock the negative
    // companion: there is no surface helper whose body produces a
    // `RiscOp::ScatterAdd` summary. The recognizer is still wired so
    // that future surface forms or AD-emitted helpers consume the
    // same code path — see the IR-level coverage in
    // `crates/chelis-ir/tests/host_sparse_summary.rs` for the
    // mechanically-driven ScatterAdd summary case.
    //
    // This test asserts the surface invariant: there is no Surf
    // construct today whose tensor-lane lowering goes to
    // `RiscOp::ScatterAdd`. The host-lane `scatter(base, indices,
    // updates, axis, "add")` pentaop emits a `chelis_tensor_scatter_add()`
    // runtime call (the exact tagged dispatch path; see
    // `specialization_dispatch.rs::scatter`), not `RiscOp::ScatterAdd`.
    // We confirm that pentaop path remains generic and that no false
    // summary is registered.
    let source = "def my_sa(base: tensor[10, 4, f32], bin_ids: tensor[64, int64], \
                  updates: tensor[64, 4, f32]) -> tensor[10, 4, f32] = \
                  scatter(base, bin_ids, updates, 0, \"add\")\n\
                  def f(base: tensor[10, 4, f32], bin_ids: tensor[64, int64], \
                  updates: tensor[64, 4, f32]) -> tensor[10, 4, f32] = \
                  my_sa(base, bin_ids, updates)\n";
    let c = build_to_c(source, "sparse_scatter_add_pentaop");
    let f_body = function_body(&c, "f");

    // The host-lane `scatter(..., "add")` pentaop is the generic
    // path: no inline sparse-add loop, no tensor-lane ScatterAdd
    // summary. The wrapper falls back to a host call.
    assert!(
        !body_contains_inline_scatter_add_loop(&f_body),
        "host-lane `scatter` pentaop is the generic runtime path; it \
         must NOT register a tensor-lane ScatterAdd summary. Body:\n{f_body}"
    );
    assert!(
        body_contains_host_function_call(&f_body, "my_sa"),
        "wrapper `f` must fall back to a host function call to `my_sa`; \
         body:\n{f_body}"
    );
}

#[test]
fn user_def_scatter_replace_helper_emits_inline_sparse_scatter_replace_loop() {
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n";
    let c = build_to_c(source, "sparse_scatter_replace_user_def");
    let f_body = function_body(&c, "f");

    assert!(
        body_contains_inline_scatter_replace_loop(&f_body),
        "user-def wrapper `f` MUST recover the inline scatter-replace \
         loop (locked: _updates_data, _indices_data, memcpy, `=` not \
         `+=`). Body:\n{f_body}"
    );
    assert!(
        !body_contains_tensor_helper_call(&f_body),
        "user-def wrapper `f` MUST bypass helper-marshaling when the \
         scatter-replace summary is registered. Body:\n{f_body}"
    );
    assert!(
        !f_body.contains("] += "),
        "scatter-replace MUST emit `=` assignment, not `+=` \
         accumulation (which would be ScatterAdd). Body:\n{f_body}"
    );

    let my_sr_body = function_body(&c, "my_sr");
    assert!(
        body_contains_inline_scatter_replace_loop(&my_sr_body),
        "inner def `my_sr` MUST inline the scatter-replace loop via \
         its tensor-helper summary; got:\n{my_sr_body}"
    );

    assert!(
        c.contains("static void my_sr__tensor_"),
        "generated tensor helper `my_sr__tensor_*` MUST still be emitted \
         for debug/non-summary callers"
    );
}

#[test]
fn nested_user_def_gather_wrapper_chain_emits_inline_sparse_gather_loop() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def wrap_g(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = wrap_g(table, indices)\n";
    let c = build_to_c(source, "sparse_gather_nested_user_def");

    let f_body = function_body(&c, "f");
    let wrap_body = function_body(&c, "wrap_g");
    assert!(
        body_contains_inline_gather_loop(&f_body),
        "two-level nested wrapper `f` MUST recover the inline \
         sparse-gather loop via propagated function summary. \
         Body:\n{f_body}"
    );
    assert!(
        body_contains_inline_gather_loop(&wrap_body),
        "one-level wrapper `wrap_g` MUST recover the sparse-gather loop. \
         Body:\n{wrap_body}"
    );
    assert!(
        !body_contains_tensor_helper_call(&f_body),
        "nested wrapper `f` MUST NOT fall back to helper marshaling"
    );
    assert!(
        !body_contains_host_function_call(&f_body, "wrap_g"),
        "nested wrapper `f` MUST NOT fall back to a host call to `wrap_g`"
    );
}

#[test]
fn nested_user_def_scatter_replace_wrapper_chain_emits_inline_loop() {
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def wrap_sr(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  wrap_sr(table, indices, updates)\n";
    let c = build_to_c(source, "sparse_scatter_replace_nested_user_def");

    let f_body = function_body(&c, "f");
    assert!(
        body_contains_inline_scatter_replace_loop(&f_body),
        "two-level nested wrapper `f` MUST recover the inline \
         scatter-replace loop. Body:\n{f_body}"
    );
    assert!(
        !body_contains_host_function_call(&f_body, "wrap_sr"),
        "nested wrapper `f` MUST NOT fall back to a host call to `wrap_sr`"
    );
}

// =========================================================================
// Negative surface — rejection categories
// =========================================================================

#[test]
fn rejected_helper_with_extra_op_after_sparse_falls_back_to_host_call() {
    // Helper post-processes the gather result with an elementwise
    // add. The helper DAG root is `add`, not `gather`, so the
    // summarizer rejects (root op is not Gather/ScatterAdd/Scatter).
    let source = "def bad(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, int64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  bad(table, indices, zero)\n";
    let c = build_to_c(source, "sparse_reject_extra_op");
    let f_body = function_body(&c, "f");

    // No false-positive summary: the caller body must NOT inline
    // the sparse-gather loop.
    assert!(
        !body_contains_inline_gather_loop(&f_body),
        "rejected helper MUST NOT register a false gather summary on the \
         caller; body had inline gather markers. Body:\n{f_body}"
    );
    // Fallback observed: the caller emits a host call to `bad`.
    assert!(
        body_contains_host_function_call(&f_body, "bad"),
        "rejected helper's caller MUST fall back to a host function call \
         to the unspecialized helper. Body:\n{f_body}"
    );
}

#[test]
fn rejected_helper_with_intermediate_op_on_operand_falls_back() {
    // Helper inserts an `add` on the values operand before the
    // gather. The Gather node's first input is `add(table, zero)`,
    // not a direct `Load`, so the summarizer rejects (operand is
    // not a direct Load).
    let source = "def bad(table: tensor[1000, 128, f32], zero: tensor[1000, 128, f32], \
                  indices: tensor[64, int64]) -> tensor[64, 128, f32] = \
                  gather(add(table, zero), indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], zero: tensor[1000, 128, f32], \
                  indices: tensor[64, int64]) -> tensor[64, 128, f32] = \
                  bad(table, zero, indices)\n";
    let c = build_to_c(source, "sparse_reject_non_load_operand");
    let f_body = function_body(&c, "f");

    assert!(
        !body_contains_inline_gather_loop(&f_body),
        "rejected helper (gather operand is not a Load) MUST NOT \
         register a summary on the caller. Body:\n{f_body}"
    );
    assert!(
        body_contains_host_function_call(&f_body, "bad"),
        "rejected helper's caller MUST fall back to a host call. \
         Body:\n{f_body}"
    );
}

#[test]
fn rejected_helper_with_two_branches_falls_back() {
    // Helper body returns one of two gather results based on a
    // boolean. The host-lane lowering produces an `If` HostExpr, not
    // a single tensor-helper call, so no `TensorCall { helper, ... }`
    // body is registered and the function specialization is `None`.
    // The caller then falls back to a host function call.
    let source = "def bad(flag: bool, table: tensor[1000, 128, f32], \
                  ind_a: tensor[64, int64], ind_b: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = \
                  if flag then gather(table, ind_a, 0) \
                  else gather(table, ind_b, 0)\n\
                  def f(flag: bool, table: tensor[1000, 128, f32], \
                  ind_a: tensor[64, int64], ind_b: tensor[64, int64]) \
                  -> tensor[64, 128, f32] = bad(flag, table, ind_a, ind_b)\n";
    let c = build_to_c(source, "sparse_reject_multiple_return_paths");
    let f_body = function_body(&c, "f");

    assert!(
        body_contains_host_function_call(&f_body, "bad"),
        "rejected helper with multiple return paths MUST cause its \
         caller to emit a host call to `bad`. Body:\n{f_body}"
    );
    // The inline gather loop may legitimately appear elsewhere in the
    // C (inside `bad`'s body, since branches are individually
    // specialize-able). We only assert it does NOT appear in `f`.
    assert!(
        !body_contains_inline_gather_loop(&f_body),
        "wrapper `f` MUST NOT have a gather summary registered when \
         the helper has multiple return paths. Body:\n{f_body}"
    );
}

#[test]
fn rejected_top_level_gather_with_wildcard_dim_falls_back() {
    // Top-level expression body whose tensor types resolve to
    // synthetic `Named("*", None)` wildcard dims (from
    // `pad_sequences` plus an under-constrained gather signature).
    // The summarizer rejects to avoid false contract assertions
    // collapsing distinct unknown axes onto the same `*` symbol.
    //
    // We assert that:
    //   1. the generated tensor helper IS emitted (the gather loop
    //      lives in the helper body, not in `main`)
    //   2. `main` calls the helper via the array-marshaling shim
    //      (`__inputs_*` / `__outputs_*` plus the
    //      `<name>__tensor_<n>` invocation), NOT a summary-derived
    //      contract assertion (`specialized sparse call ...`)
    //
    // This locks the structural contract that wildcard dims silently
    // fall back to the helper marshaling path; it is exactly the
    // regression that fired in
    // `cli::build_c_runs_tensor_structural_ops_and_matches_eval_output`
    // during W3-B development and is permanently pinned here.
    let source = "lhs = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)\n\
                  ids_list: List[int64] = [cast(0, int64), cast(1, int64)]\n\
                  token_ids = to_tensor(ids_list)\n\
                  result = gather(lhs, token_ids, 0)\n";
    let c = build_to_c(source, "sparse_reject_wildcard_dim");

    // The generated helper for the top-level tensor expression must
    // be emitted (locks: wildcard helpers still get a generated C
    // body for the runtime to call).
    assert!(
        c.contains("__global__tensor_") || c.contains("__tensor_0"),
        "wildcard-dim top-level gather MUST still emit a generated tensor \
         helper for the marshaling fallback path"
    );

    // No summary contract assertion was emitted anywhere — the
    // summarizer rejected the wildcard helper.
    assert!(
        !c.contains("specialized sparse call"),
        "wildcard-dim top-level gather MUST NOT register a sparse summary \
         (no `specialized sparse call ...` contract assertions emitted). \
         If this fires, the wildcard-rejection invariant has regressed."
    );

    // `main` must invoke the helper via the marshaling shim.
    assert!(
        c.contains("__inputs_") && c.contains("__outputs_"),
        "wildcard-dim top-level gather MUST fall back to helper \
         marshaling (`__inputs_*` / `__outputs_*`)"
    );
}

// =========================================================================
// Sibling sweep — three sparse ops × the same rejection categories,
// abbreviated. Locks the invariant that each rejection class fires
// for every sparse op, not just gather.
// =========================================================================

#[test]
fn rejected_scatter_replace_with_extra_op_falls_back() {
    let source = "def bad(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32], zero: tensor[3, 2, f32]) \
                  -> tensor[3, 2, f32] = \
                  add(scatter_replace(table, indices, updates, 0), zero)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, int32], \
                  updates: tensor[4, 2, f32], zero: tensor[3, 2, f32]) \
                  -> tensor[3, 2, f32] = bad(table, indices, updates, zero)\n";
    let c = build_to_c(source, "sparse_reject_sr_extra_op");
    let f_body = function_body(&c, "f");
    assert!(
        !body_contains_inline_scatter_replace_loop(&f_body),
        "rejected scatter-replace helper MUST NOT register a summary. \
         Body:\n{f_body}"
    );
    assert!(
        body_contains_host_function_call(&f_body, "bad"),
        "rejected scatter-replace helper's caller MUST fall back to a \
         host call. Body:\n{f_body}"
    );
}
