module Std.Tests.Nn.Conv
import Std.Nn.Conv (conv1d, conv2d_small)
import Std.Test (assert_true)
-- Acceptance surface for `Std.Nn.Conv` at Phase 3t A1 Wave 2.
--
-- Numeric end-to-end evaluation through `chelis test` is NOT reachable: the
-- shipped wrappers `conv1d` and `conv2d_small` are concrete-shape (rank-4)
-- requirements imposed by the Phase 0e `lower_conv2d` pass — see
-- `spec/design/chelis_phase3_plan.md` §3j-pre, where Batch 3b ships these
-- as type-check-only wrappers and the executable numeric oracle is owed to
-- Phase 3j. The eval path used by `chelis test` cannot synthesise a
-- rank-4 `tensor[1, 4, 1, 16, f32]` (or `tensor[1, 3, 8, 8, f32]`) input
-- in source: list literals plus `pad_sequences_to` produce rank-2,
-- `reshape` to a rank-4 shape is rejected at type-check, and there is no
-- evaluator-level rank-4 constructor today.
--
-- What this test file pins:
--   1. `chelis test` resolves the `Std.Nn.Conv` import surface (every
--      exported symbol is bound by name in this module).
--   2. The file-level compile pass (run by `chelis test` before any test
--      executes — see `compile_with_reef_graph` in the test runner)
--      accepts typed wrappers that invoke `conv1d` and `conv2d_small`
--      with the shipped concrete shapes, exercising the type-checker
--      over the conv call sites and proving the published shapes line
--      up with the documented contract.
--
-- What this test file does NOT pin (deferred to the Phase 3j-pre oracle
-- once the host-runtime `conv2d` lowering supports the eval path):
--   - numeric correctness of either conv forward pass;
--   - shape-negative diagnostics (those are pinned at the `chelis check`
--     level in `phase3j_pre_std_batch3b`, which can assert on stderr;
--     `chelis test` runs eval, not check, and a wrong-shape conv call
--     surfaces as a single file-level compile FAIL row, which cannot be
--     turned into a passing test-level negative assertion).

-- Typed wrapper around `conv1d`. Type-checked at file-compile time even
-- though no test invokes it; if the published signature drifts away from
-- input `tensor[1, 4, 1, 16, f32]` / kernel `tensor[8, 4, 1, 3, f32]` /
-- output `tensor[1, 8, 1, 14, f32]`, this def stops compiling and the
-- whole file fails with a single `<file>` row.
def expected_conv1d_shape(x: tensor[1, 4, 1, 16, f32], k: tensor[8, 4, 1, 3, f32]) -> tensor[1, 8, 1, 14, f32] = conv1d(x, k)

-- Typed wrapper around `conv2d_small`. Same role: pins
-- input `tensor[1, 3, 8, 8, f32]` / kernel `tensor[8, 3, 3, 3, f32]` /
-- output `tensor[1, 8, 6, 6, f32]`.
def expected_conv2d_small_shape(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv2d_small(x, k)

def test_conv_module_imports() -> unit ! { Test } = {
  -- Importability: if the `import` at the top of this file fails to bind
  -- either exported conv symbol, the file never reaches this test and
  -- `chelis test` reports a single `<file>` failure instead. Reaching
  -- this body means both `conv1d` and `conv2d_small` were resolved.
  assert_true(true, "Std.Nn.Conv exports resolve")
}

def test_conv_typechecks_with_concrete_shapes() -> unit ! { Test } = {
  -- The file-level compile pass type-checks the two `expected_*_shape`
  -- defs above. If either conv signature drifts, `chelis test` aborts
  -- with a `<file>` compile failure before this test runs. Reaching this
  -- body therefore witnesses that the shipped wrappers still match the
  -- documented concrete-shape contract.
  assert_true(true, "Std.Nn.Conv concrete-shape signatures type-check")
}
