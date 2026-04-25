module Std.Tests.Nn.Attention
import Std.Nn.Attention (scaled_dot_product_attention, multi_head_attention, grouped_query_attention, gqa_broadcast_kv)
import Std.Test (assert_true)
-- Acceptance surface for `Std.Nn.Attention` at Phase 3t A1 Wave 2.
--
-- Numeric end-to-end evaluation through `chelis test` is NOT reachable: the
-- host runtime lowering does not implement `matmul`, `softmax`, `permute`, or
-- `expand`, all of which `scaled_dot_product_attention` calls. See
-- `spec/design/chelis_phase3_plan.md` §3j-pre and the Batch 3b acceptance
-- surface (publish + import-touch + check-level shape negatives) in
-- `crates/chelis-cli/tests/phase3j_pre_std_batch3b.rs`.
--
-- What this test file pins:
--   1. `chelis test` resolves the `Std.Nn.Attention` import surface (every
--      exported symbol is bound by name in this module).
--   2. The file-level compile pass (run by `chelis test` before any test
--      executes — see `compile_with_reef_graph` in the test runner) accepts
--      a typed wrapper that invokes `scaled_dot_product_attention`,
--      `multi_head_attention`, `grouped_query_attention`, and
--      `gqa_broadcast_kv` with the shipped concrete shapes, exercising the
--      type-checker over the attention call sites and proving the
--      published shapes line up with the documented contract.
--
-- What this test file does NOT pin (deferred to the Phase 3j-pre oracle once
-- host-runtime `matmul`/`softmax`/`permute`/`expand` land):
--   - numeric correctness of the attention forward pass;
--   - shape-negative diagnostics (those are pinned at the `chelis check`
--     level in `phase3j_pre_std_batch3b`, which can assert on stderr;
--     `chelis test` runs eval, not check, so it cannot consume an
--     expected-error contract).

-- Typed wrapper around `scaled_dot_product_attention`. Type-checked at
-- file-compile time even though no test invokes it; if the published
-- signature drifts away from `tensor[4, 4, f32]` this def stops compiling
-- and the whole file fails with a single `<file>` row.
def expected_sdpa_shape(q: tensor[4, 4, f32], k: tensor[4, 4, f32], v: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = scaled_dot_product_attention(q, k, v, scale)

-- MHA per-head wrapper: same shape contract; type-checked at file-compile.
def expected_mha_shape(q_head: tensor[4, 4, f32], k_head: tensor[4, 4, f32], v_head: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = multi_head_attention(q_head, k_head, v_head, scale)

-- GQA per-head wrapper: same shape contract.
def expected_gqa_shape(q_head: tensor[4, 4, f32], k_head: tensor[4, 4, f32], v_head: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = grouped_query_attention(q_head, k_head, v_head, scale)

-- GQA KV broadcast wrapper: pins the rank-3 in / rank-3 out shape contract
-- (1 KV head -> 2 query-group heads via gather on axis 0).
def expected_gqa_broadcast_shape(kv_pool: tensor[1, 4, 4, f32], group_map: tensor[2, int64]) -> tensor[2, 4, 4, f32] = gqa_broadcast_kv(kv_pool, group_map)

def test_attention_module_imports() -> unit ! { Test } = {
  -- Importability: if the `import` at the top of this file fails to bind
  -- any exported attention symbol, the file never reaches this test and
  -- `chelis test` reports a single `<file>` failure instead. Reaching
  -- this body means all four exports were resolved.
  assert_true(true, "Std.Nn.Attention exports resolve")
}

def test_attention_typechecks_with_concrete_shapes() -> unit ! { Test } = {
  -- The file-level compile pass type-checks the four `expected_*_shape`
  -- defs above. If any attention signature drifts, `chelis test` aborts
  -- with a `<file>` compile failure before this test runs. Reaching this
  -- body therefore witnesses that the shipped wrappers still match the
  -- documented `tensor[4, 4, f32]` contract.
  assert_true(true, "Std.Nn.Attention concrete-shape signatures type-check")
}
