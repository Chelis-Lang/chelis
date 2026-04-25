module Std.Tests.Loss.CrossEntropy
import Std.Loss.CrossEntropy (loss)
import Std.Test (assert_true)
-- Acceptance surface for `Std.Loss.CrossEntropy` at Phase 3t A1 Wave 2.
--
-- Numeric end-to-end evaluation through `chelis test` is NOT reachable: the
-- shipped `loss` body composes `softmax(_, 1) |> log |> mul(labels) |> sum(1)
-- |> neg`, and the host runtime used by `chelis test` does not implement the
-- `softmax` builtin (it is only lowered through `chelis build` + gcc/hipcc).
-- A direct probe (`logits = [[0.0, 0.0]]`, `labels = [[1.0, 0.0]]`,
-- `loss(logits, labels)`) fails with
-- `unsupported builtin `softmax` in host runtime`. See
-- `spec/design/chelis_phase3_plan.md` §3j-pre and the sibling
-- `tests/nn/attention.ch` / `tests/nn/conv.ch` files, which document the same
-- "type-check-only at `chelis test`, numeric oracle deferred to `chelis
-- build`" pattern for the other Wave 2 modules that depend on `softmax`,
-- `matmul`, `permute`, or `expand`.
--
-- What this test file pins:
--   1. `chelis test` resolves the `Std.Loss.CrossEntropy` import surface
--      (the exported `loss` symbol is bound by name in this module — if the
--      module name or the `loss` export drifts, the file never reaches the
--      tests below and `chelis test` reports a single `<file>` failure).
--   2. The file-level compile pass (run by `chelis test` before any test
--      executes — see `compile_with_reef_graph` in the test runner) accepts
--      a typed wrapper that invokes `loss` with concrete shapes, exercising
--      the type-checker over the call site and proving the published
--      signature still lines up with the documented contract:
--      `loss(logits: tensor[batch, classes, f32],
--            labels: tensor[batch, classes, f32]) -> tensor[batch, f32]`.
--
-- What this test file does NOT pin (deferred to the Phase 3j-pre oracle once
-- host-runtime `softmax` lands, or to a `chelis build`-driven numeric oracle
-- in the meantime):
--   - numeric correctness of the cross-entropy forward pass — for example,
--     the identities `loss([[LARGE, -LARGE]], [[1.0, 0.0]]) ≈ 0` and
--     `loss([[0.0, 0.0]], any_one_hot) = log(2)`, which are derivable from
--     the formula but cannot be checked here because `softmax` is unsupported
--     in the host runtime;
--   - shape-negative diagnostics (those are pinned at the `chelis check`
--     level; `chelis test` runs eval, and a wrong-shape `loss` call would
--     surface as a single file-level compile FAIL row, which cannot be
--     turned into a passing test-level negative assertion).

-- Typed wrapper around `loss`. Type-checked at file-compile time even though
-- no test invokes it; if the published signature drifts away from
-- `tensor[batch, classes, f32] -> tensor[batch, classes, f32] ->
-- tensor[batch, f32]`, this def stops compiling and the whole file fails
-- with a single `<file>` row. The two wrappers below pin both the
-- two-class shape (binary-style cross-entropy with one-hot labels) and a
-- multi-class shape (3 classes, batch of 4), so a regression in either
-- direction trips this file.
def expected_loss_shape_2class(logits: tensor[1, 2, f32], labels: tensor[1, 2, f32]) -> tensor[1, f32] = loss(logits, labels)

def expected_loss_shape_multiclass(logits: tensor[4, 3, f32], labels: tensor[4, 3, f32]) -> tensor[4, f32] = loss(logits, labels)

def test_crossentropy_module_imports() -> unit ! { Test } = {
  -- Importability: if the `import Std.Loss.CrossEntropy (loss)` at the top
  -- of this file fails to bind the exported `loss` symbol, the file never
  -- reaches this test and `chelis test` reports a single `<file>` failure
  -- instead. Reaching this body means the export was resolved.
  assert_true(true, "Std.Loss.CrossEntropy.loss export resolves")
}

def test_crossentropy_typechecks_with_concrete_shapes() -> unit ! { Test } = {
  -- The file-level compile pass type-checks the `expected_loss_shape_2class`
  -- and `expected_loss_shape_multiclass` defs above. If the `loss`
  -- signature drifts away from
  -- `(tensor[batch, classes, f32], tensor[batch, classes, f32])
  --     -> tensor[batch, f32]`,
  -- `chelis test` aborts with a `<file>` compile failure before this test
  -- runs. Reaching this body therefore witnesses that the shipped wrapper
  -- still matches the documented batched-cross-entropy shape contract on
  -- both 2-class and multi-class shapes.
  assert_true(true, "Std.Loss.CrossEntropy.loss concrete-shape signature type-checks")
}
