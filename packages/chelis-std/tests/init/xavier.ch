module Std.Tests.Init.Xavier
import Std.Init.Xavier (sample)
import Std.Test (assert_true)
-- Acceptance surface for `Std.Init.Xavier` at Phase 3t A1 Wave 2.
--
-- Numeric end-to-end evaluation through `chelis test` is NOT reachable: the
-- shipped signature
--
--   sig sample: tensor[32, 128, f32] -> f32 -> tensor[32, 128, f32] ! { Random }
--
-- pins a concrete rank-2 shape `tensor[32, 128, f32]`, and the eval path
-- used by `chelis test` cannot synthesise such a value in source. List
-- literals plus `pad_sequences_to` produce rank-2 tensors with symbolic
-- (not concrete) dimensions, and `reshape` to a concrete-dim rank-2 shape
-- is rejected at type-check (the `to_tensor` + `reshape` route mirrors the
-- conv2d gap documented in `tests/nn/conv.ch`). On top of that, `sample`
-- in this build is sig-only — the body is owed to the Phase 3j-pre native
-- numeric oracle in `crates/chelis-cli/tests/phase3j_pre_std_batch4.rs`,
-- where `xavier_uniform` / `xavier_normal` are exercised under
-- `with seed(...)` against a rank-1 statistical-sanity template.
--
-- What this test file pins:
--   1. `chelis test` resolves the `Std.Init.Xavier` import surface
--      (`sample` is bound by name in this module).
--   2. The file-level compile pass (run by `chelis test` before any test
--      executes — see `compile_with_reef_graph` in the test runner)
--      accepts typed wrappers that invoke `sample` with the shipped
--      concrete shape `tensor[32, 128, f32] -> f32 -> tensor[32, 128, f32]`,
--      exercising the type-checker over the call site and proving the
--      published shape and arity line up with the documented contract.
--   3. The documented `with seed(...) { sample(...) }` usage pattern
--      parses and is accepted by the file-level compile pass at this
--      concrete rank-2 shape. NOTE: the current type-checker does not
--      enforce effect rows on return-type annotations, AND it does not
--      enforce the inner call's shape contract under `with seed(...)`
--      (both verified by mutation: dropping `! { Random }` and changing
--      the wrapper template from `tensor[32, 128, f32]` to
--      `tensor[16, 128, f32]` both compile clean while the same shape
--      mutation on the unseeded `expected_sample_shape` wrapper is
--      caught with "dimension mismatch: Lit(32) vs Lit(16)"). This is
--      therefore a SYNTAX-and-arity pin on the canonical-reference
--      usage pattern, not a semantic shape-or-effect check. Effect-row
--      and through-handler shape enforcement are tracked outside this
--      file as language-level gaps.
--
-- What this test file does NOT pin (deferred to the Phase 3j-pre oracle
-- for `xavier_uniform` / `xavier_normal`, which use a rank-1 template and
-- are reachable from the eval path):
--   - determinism under same seed (two `with seed(42) { sample(...) }`
--     calls produce the same tensor);
--   - seed sensitivity (different seeds produce different tensors);
--   - output shape contract at runtime (sampled tensor has the same
--     rank-2 shape as the template);
--   - statistical sanity (sampled values bounded, std deviation roughly
--     `sqrt(2 / (fan_in + fan_out))`).
--   These four behaviours cannot be exercised here because there is no
--   source-level path to a `tensor[32, 128, f32]` value. They are tracked
--   as a gap on the `Std.Init.Xavier` acceptance surface; the rank-1
--   xavier_* variants in `Std.Init.XavierExt` carry the live numeric
--   coverage in the meantime.

-- Typed wrapper around `sample`. Type-checked at file-compile time even
-- though no test invokes it; if the published signature drifts away from
-- input `tensor[32, 128, f32]` and `f32` to output `tensor[32, 128, f32]`
-- with a `Random` effect, this def stops compiling and the whole file
-- fails with a single `<file>` row.
def expected_sample_shape(template: tensor[32, 128, f32], gain: f32) -> tensor[32, 128, f32] ! { Random } = sample(template, gain)

-- Seeded wrapper: pins that `with seed(...)` parses and is accepted by
-- the file-level compile pass around a call to `sample` at this
-- concrete shape. NOTE: under `with seed(...)`, the current type-
-- checker does NOT enforce either (a) the effect row on the return-
-- type annotation or (b) the inner call's shape contract (verified by
-- mutation: changing this wrapper's template to `tensor[16, 128, f32]`
-- compiles clean, even though the inner `sample` is published as
-- `tensor[32, 128, f32] -> ...`). This pin is therefore a SYNTAX-and-
-- arity pin on the documented `with seed(...) { sample(...) }` usage
-- pattern from the canonical reference, not a semantic shape-or-effect
-- check. Both gaps are language-level and are tracked outside this
-- file.
def expected_sample_seeded(template: tensor[32, 128, f32], gain: f32) -> tensor[32, 128, f32] = with seed(42) { sample(template, gain) }

-- Distinct-seed wrapper: pins that the same shape contract holds under a
-- different seed value. Same caveat as `expected_sample_seeded` — this
-- is a syntax-and-shape pin on the seeded call site, not a semantic
-- determinism check (determinism cannot be exercised here because
-- `sample` is sig-only at this concrete rank-2 shape).
def expected_sample_seeded_distinct(template: tensor[32, 128, f32], gain: f32) -> tensor[32, 128, f32] = with seed(7) { sample(template, gain) }

def test_xavier_module_imports() -> unit ! { Test } = {
  -- Importability: if the `import` at the top of this file fails to bind
  -- `sample`, the file never reaches this test and `chelis test` reports
  -- a single `<file>` failure instead. Reaching this body means the
  -- single export was resolved.
  assert_true(true, "Std.Init.Xavier exports resolve")
}

def test_xavier_typechecks_with_concrete_shape() -> unit ! { Test } = {
  -- The file-level compile pass type-checks `expected_sample_shape`
  -- above. If the `sample` signature drifts, `chelis test` aborts with a
  -- `<file>` compile failure before this test runs. Reaching this body
  -- therefore witnesses that the shipped wrapper still matches the
  -- documented `tensor[32, 128, f32] -> f32 -> tensor[32, 128, f32] ! { Random }`
  -- contract.
  assert_true(true, "Std.Init.Xavier.sample concrete-shape signature type-checks")
}

def test_xavier_with_seed_call_site_compiles() -> unit ! { Test } = {
  -- The file-level compile pass accepts `expected_sample_seeded` and
  -- `expected_sample_seeded_distinct` above. Both wrappers wrap a call
  -- to `sample` inside `with seed(N) { ... }` at the rank-2 shape. If
  -- either the surface syntax for `with seed(...)` regresses (e.g.
  -- `seed` stops being a recognised handler keyword) or the call-site
  -- arity changes, `chelis test` aborts with a `<file>` compile
  -- failure before this test runs. Reaching this body witnesses that
  -- the canonical-reference usage pattern
  -- `with seed(N) { sample(template, gain) }` is still surface-
  -- syntactically valid. (See the seeded-wrapper comment above for the
  -- caveats: under `with seed(...)`, neither the effect row on the
  -- return-type annotation nor the inner call's shape contract is
  -- enforced by today's type-checker, so this is NOT a semantic
  -- shape-or-effect check.)
  assert_true(true, "Std.Init.Xavier.sample with seed(...) call site compiles at rank-2 shape")
}
