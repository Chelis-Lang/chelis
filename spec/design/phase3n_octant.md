# Phase 3n / 3o — Octant (LaTeX ↔ Deep Bridge)

Status: **planned**, not yet started.

This document is the executable sub-phase plan for Octant. The architectural
design lives in `chelis_octant_design.md`; this file defines the concrete
deliverables, prerequisites, test plans, acceptance oracles, non-silent
deferrals, and open infrastructure decisions for sub-phases `3n` and `3o`.

## 0. Summary

Octant is a notation bridge that parses a bounded LaTeX subset used in
quantitative finance, lowers it to typed Deep, round-trips it back to LaTeX
with Chelis type overlays, and carries provenance spans on every emitted Deep
node so each compiled fragment points back to its source math.

The work is split across two Chelis sub-phases:

- **`3n` (Part A)** — Parser, symbolic AST, deterministic lowering for
  arithmetic / unary / derivatives / special functions / matrix ops / integrals
  through Nautilus, `Octant.Render` with type overlays, and `Octant.Provenance`
  on every Deep node produced by Octant lowering.
- **`3o` (Part B)** — LLM-assisted lowering of SDE / Monte Carlo / calibration
  notation through Shoals, Greek rendering pattern matches, and
  `Octant.Notebook` cell runtime. Provenance extends to cover the new
  finance-notation node kinds using the 3n contract.

Document ingestion (full LaTeX paper parsing, `\begin{equation}` extraction,
prose association) is parked as a **post-Phase-3 stub**.

```text
                3j: Nautilus  ∥  3k: Coral
                         ↓
          3l: Shoals   ∥   3n: Octant (Part A)
                         ↓
                3o: Octant (Part B)
                         ↓
                3f: SKILL.md v2 redo
```

`3n` has the same prerequisite as `3l` (namely `3j` green) and therefore runs
in parallel with `3l`. `3o` requires both `3l` and `3n` green.

## 1. Sub-phase 3n — Octant (Part A)

### 1.1 Goal

Ship a reef package `octant` that can:

1. Parse the quant-finance LaTeX subset defined in `chelis_octant_design.md §4.1`.
2. Lower it deterministically to Deep for every expression form that maps
   mechanically — arithmetic, unary functions, powers, roots, derivatives
   (`\partial` → `grad`), special functions (`erf`, `normal_cdf`, `log_gamma`,
   `beta`, `digamma`), summation / product folds, piecewise / conditional,
   integrals through `Nautilus.Integrate`, matrix ops through `Nautilus.LinAlg`.
3. Render typed Deep back to LaTeX with type overlays (named tensor dims as
   subscripts, effect markers, linearity markers, `grad` rendered as
   `\frac{\partial}{\partial x}`).
4. Emit provenance spans on every Deep node produced by Octant lowering, linking
   it back to the originating LaTeX substring (source line, column, and the
   raw LaTeX fragment).

### 1.2 Prerequisite

- `3j: Nautilus` green — Octant needs `Nautilus.Special`, `Nautilus.Distributions`,
  `Nautilus.LinAlg`, `Nautilus.Integrate`.
- Does **not** depend on `3k: Coral` or `3l: Shoals`.

### 1.3 Modules shipped in 3n

| Module | Contents |
|---|---|
| `Octant.Parse` | LaTeX subset parser over the Section 4.1 in-scope grammar. Out-of-scope LaTeX produces clean diagnostic errors, not silent drops. |
| `Octant.Symbolic` | The ~30-node `SymExpr` AST (see `chelis_octant_design.md §4.2`). |
| `Octant.Lower` (deterministic path only) | SymExpr → Deep for every node where the LaTeX uniquely determines the computation. Special functions route through `Nautilus.Special` / `Nautilus.Distributions`; integrals route through `Nautilus.Integrate`; matrix ops route through `Nautilus.LinAlg`. |
| `Octant.Render` | Deep → LaTeX with type overlays (named dims → subscripts, effects → markers, `grad` → partial-derivative notation). Excludes the Greek pattern matches that need Shoals context — those land in 3o. |
| `Octant.Provenance` | Source-span annotations on every Deep node produced by Octant lowering. Contract: the metadata map on the Deep node carries `provenance` (raw LaTeX fragment) and `source_span` (line, column, length) keys. |

### 1.4 Test plan

- **Parser corpus:** at least one positive test per in-scope LaTeX construct
  from `chelis_octant_design.md §4.1`, plus negative tests for out-of-scope
  constructs (`\begin{theorem}`, TikZ blocks, symbolic integration requests,
  paragraph text) that must produce diagnostic errors naming the offending
  token.
- **Black-Scholes `d_1` round-trip (headline acceptance test):** parse the
  LaTeX `d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) T}{\sigma \sqrt{T}}` →
  lower to Deep → compile through `chelis check` / `chelis eval` → render
  back to LaTeX with type overlays → **strip type overlays** (named-dim
  subscripts, effect markers, linearity markers are rendering decoration,
  not part of the `§4.1` in-scope grammar `Octant.Parse` accepts) →
  re-parse the stripped output and
  assert the re-parsed `SymExpr` is structurally equal to the first-parse
  `SymExpr` modulo whitespace, bracket normalization, and floating-point
  formatting (the Cross-Sub-Phase Invariant §3.4 determinism contract, **not**
  byte-exact LaTeX matching) → walk the lowered Deep and assert **every**
  node carries a `provenance` span whose fragment is a substring of the
  original LaTeX input.
- **Provenance error-localization (audit-trail semantics, not just presence):**
  parse a deliberately ill-typed fragment (for example
  `\sigma \sqrt{T}` where `sigma` is declared as a named tensor and `T` as
  a scalar so the multiply is a shape mismatch) and assert the `chelis check`
  error message **surfaces the LaTeX source span** — the failing fragment
  string, line, and column from the original LaTeX, not just the Deep node
  identifier. This pins the error-localization workflow from
  `chelis_octant_design.md §4.5`, which is the actual audit-trail semantics
  — provenance presence alone is insufficient.
- **Type overlay rendering:** tests that named tensor dimensions appear as
  subscripts, that `Random`/`IO` effects produce the correct markers, and
  that `grad(f, wrt=x)` renders as `\frac{\partial f}{\partial x}`.
- **Provenance completeness:** a fuzz-style test that generates ten varied
  expressions from the in-scope grammar, lowers each, and asserts that no
  emitted Deep node has a missing or empty `provenance` / `source_span`
  metadata entry. This pins the "core value proposition" invariant.
- **Special function lowering:** positive tests for `\text{erf}(x)`,
  `\Phi(x)` (normal CDF), `\Gamma(x)`, `\log\Gamma(x)`, `B(a, b)` each
  lowering to the correct `Nautilus` call, plus a negative test that a
  special function not in the closed vocabulary produces a clean
  "unsupported special function" diagnostic.
- **Integral lowering:** `\int_0^T f(t)\,dt` lowers to
  `Nautilus.Integrate.adaptive_simpson(f, 0.0, T)` and round-trips.
- **Matrix lowering:** `\mathbf{A}^{-1}\mathbf{b}` lowers to
  `Nautilus.LinAlg.solve(A, b)` (inverse-times-vector recognized as solve).
- **Reef package gate:** `chelis reef build` produces a valid `.chb`, a
  consumer crate imports `octant` and type-checks.

### 1.5 Acceptance oracle

`cargo test -p chelis-cli phase3n_octant_oracle -- --exact`

This is the owning executable oracle for sub-phase `3n`. It is named here but
**deliberately not implemented by this planning change** — its creation is
owned by whoever picks up the 3n coding work (same pattern as the
`phase3j_nautilus_oracle` and `phase3l_shoals_oracle` entries already named
in `chelis_phase3_plan.md §3j` and `§3l` before those phases shipped). The
oracle must exercise, in a single test file:

1. The Black-Scholes `d_1` round-trip.
2. The provenance-completeness invariant (every emitted Deep node carries a
   source span).
3. The provenance error-localization test (a deliberately ill-typed fragment
   produces a `chelis check` error whose message contains the originating
   LaTeX source span).
4. The out-of-scope LaTeX diagnostic invariant (Cross-Sub-Phase Invariant
   §3.2) — fed `\begin{theorem}`, a TikZ block, and "please integrate
   `\int e^{-x^2}`" the parser must return diagnostics naming the offending
   token, not silently drop to an empty `SymExpr`.
5. At least one special-function lowering through Nautilus (for example
   `\Phi(d_1)` → `Nautilus.Distributions.normal_cdf`).
6. The reef-package gate (`chelis reef build` → consumer imports the
   resulting `.chb`).

A 3n completion claim additionally requires a fresh-context red team as per
the Chelis agent contract.

### 1.6 Non-silent deferrals (3n)

Any Octant capability blocked by an upstream Chelis feature gap is recorded
here, not silently dropped:

- **SDE lowering** — deferred to 3o (depends on `Shoals.Stochastic`).
- **Monte Carlo expectation lowering** — deferred to 3o (depends on
  `Shoals.Pricing`).
- **Yield curve / day count lowering** — deferred to 3o (depends on
  `Shoals.Curves` and `Std.Time`).
- **Greek pattern rendering** (`grad(price, wrt=spot) → Δ`) — deferred to 3o
  because it needs the finance variable-name conventions that land with
  Shoals context.
- **Signal processing lowering** (FFT, STFT) — blocked indefinitely by
  complex number support (Phase 5f). Stubbed in Octant with a clean
  "complex numbers not yet supported" diagnostic.
- **Notebook cell runtime** — deferred to 3o (depends on the full
  parse / lower / render / provenance surface plus finance notation).
- **Full-document LaTeX ingestion** — parked as a post-Phase-3 stub.

### 1.7 Open infrastructure decision (must confirm before 3n coding starts)

**LaTeX parser implementation choice.** The `chelis_octant_design.md §9`
open question 1 lists three options:

1. Pure Chelis implementation inside the octant reef package.
2. Rust crate linked via runtime FFI, similar to the potential future
   nalgebra bridge.
3. Existing Rust LaTeX parser crate (for example `pulldown-latex`,
   `latex2mathml` internals) consumed through the runtime FFI.

**Recommended default for 3n:** option (3) — lean on an existing Rust crate
and expose it through the runtime FFI, same pattern as `memmap2` in 3g and
`parquet2` in 3k. Option (1) is the long-term target once Chelis string
processing matures. The user must explicitly sign off on this choice before
3n implementation work begins.

## 2. Sub-phase 3o — Octant (Part B)

### 2.1 Goal

Extend Octant with finance-notation lowering and the interactive notebook:

1. LLM-assisted lowering of SDE notation (`dS = \mu S\,dt + \sigma S\,dW_t`)
   through `Shoals.Stochastic`, including the discretization choice
   (Euler-Maruyama vs Milstein), time grid, and noise generation strategy.
2. LLM-assisted lowering of Monte Carlo expectations
   (`\mathbb{E}[f(S_T)]`) through `Shoals.Pricing`, including variance
   reduction choice and `Random` effect handling.
3. LLM-assisted lowering of calibration objectives through `Nautilus.Optim`.
4. Yield curve / day count lowering through `Shoals.Curves` and `Std.Time`.
5. Greek rendering pattern matches in `Octant.Render`:
   `grad(price, wrt=spot) → \Delta`, `grad(price, wrt=vol) → \mathcal{V}`,
   `grad(price, wrt=rate) → \rho`, `grad(price, wrt=T) → \Theta`, configurable.
6. `Octant.Notebook` cell kinds: formula, parameter, execution, Greek.
7. Provenance annotations extend to cover the new node kinds using the 3n
   contract (no Deep node emitted by Octant lowering is missing a source
   span).

### 2.2 Prerequisite

- `3l: Shoals` green — needed for `Shoals.Stochastic`, `Shoals.Pricing`,
  `Shoals.Curves`.
- `3i` green — `Std.Time` is a direct dependency of the yield curve / day
  count lowering path, not only a transitive dependency through `Shoals`.
  Called out explicitly so the 3o start gate is unambiguous.
- `3n: Octant (Part A)` green.

### 2.3 Modules shipped in 3o

| Module | Contents |
|---|---|
| `Octant.Lower` (LLM-assisted path) | SDE, Monte Carlo expectation, calibration, yield curve lowering via the coding model, routing to `Shoals.Stochastic`, `Shoals.Pricing`, `Nautilus.Optim`, `Shoals.Curves`. |
| `Octant.Render` (finance additions) | Greek pattern matches, configurable variable-name conventions (`spot/S → Delta`, `vol/sigma → Vega`, `rate/r → Rho`, `T/time/tau → Theta`). |
| `Octant.Notebook` | Cell runtime — formula cells (live LaTeX → compiled Deep), parameter cells, execution cells, Greek cells. Not a Jupyter kernel. |
| `Octant.Provenance` (extension) | Same contract as 3n, extended to the new SDE / MC / calibration / curve node kinds. |

### 2.4 Test plan

- **Black-Scholes full pricer round-trip:** parse the full Black-Scholes
  call formula → lower → compile → evaluate → assert numerical match
  against analytical Black-Scholes within `1e-10`.
- **Greek round-trip:** `grad(price, wrt=spot)` lowers and renders as
  `\Delta`, and the rendered formula matches the analytical delta within
  `1e-6`.
- **GBM SDE lowering:** `dS = \mu S\,dt + \sigma S\,dW_t` lowers through
  `Shoals.Stochastic`, the resulting path has correct statistics
  (mean = `S_0 \exp(\mu T)`, variance within tolerance).
- **Monte Carlo expectation:** `\mathbb{E}[\max(S_T - K, 0)]` lowers
  through `Shoals.Pricing` and converges to the analytical Black-Scholes
  price on a vanilla European call within `1%` at 100k paths.
- **Notebook cell contracts:** a formula cell edit triggers parse → lower →
  compile → render in one transaction; a parameter cell's binding is
  visible to downstream execution cells; a Greek cell shows the formula,
  any symbolic simplification, and the numerical value.
- **Provenance completeness (3o extension):** every Deep node produced by
  the new SDE / MC / calibration lowering carries a valid `provenance`
  span. This is a regression of the 3n invariant applied to the new node
  kinds.
- **LLM-assisted lowering correctness contract.** The coding model is not
  trusted to produce correct Deep on its own. Every LLM-assisted lowering
  result must be (a) type-checked by `chelis check` before it is accepted,
  (b) rejected with a non-silent diagnostic if it fails to type-check or
  fails the round-trip property — the fallback behavior is `Octant lowering
  failed: coding model produced Deep that did not match the declared
  SymExpr shape`, NOT a silent approximation or retry. Tests must cover:
  (i) a known-good SDE lowers correctly and matches the analytical GBM
  statistics; (ii) a deliberately mangled coding-model output (injected by
  stubbing the lowering callback to return ill-typed Deep) produces the
  diagnostic described above and does not poison downstream state;
  (iii) determinism — lowering the same SymExpr twice in the same process
  produces the same Deep (or the same diagnostic), and divergence is a
  test failure.

### 2.5 Acceptance oracle

`cargo test -p chelis-cli phase3o_octant_oracle -- --exact`

Exercises the Black-Scholes full pricer round-trip (including Greeks via
`grad` and `Shoals.Pricing` Monte Carlo), the provenance-completeness
invariant on the extended node kinds, a notebook cell-kind contract test,
and the reef-package gate. Fresh-context red team still required before any
3o completion claim.

### 2.6 Non-silent deferrals and open risks (3o)

- **Open risk: LLM-assisted lowering correctness.** The SDE / MC /
  calibration / yield curve lowering depends on a coding model producing
  typecheck-clean Deep for notation the deterministic path cannot handle.
  The open question on how this integrates (SKILL prompt vs fine-tuned
  model vs MCP server — `chelis_octant_design.md §9` open question 2) is
  **not resolved** at 3o start. The correctness contract in §2.4 bullet
  "LLM-assisted lowering correctness contract" is the minimum gate: any
  integration option must satisfy type-check + non-silent diagnostic +
  determinism before it is allowed to ship. If at 3o start no integration
  option can satisfy that gate, 3o is blocked and the blockage is tracked
  here rather than worked around.
- **Octant Phase 4 — full-document LaTeX ingestion** (parsing full LaTeX
  papers, extracting `\begin{equation}` blocks, associating formulas with
  surrounding prose) remains parked as a post-Phase-3 stub. Tracked
  alongside `school` and `darwin` in `chelis_project_plan.md`.
- **`Octant.Signal` signal processing lowering** remains stubbed, blocked
  by complex numbers (Phase 5f).

## 3. Cross-sub-phase invariants

These invariants span both 3n and 3o and must hold on every release:

1. **Provenance completeness.** No Deep node emitted by any Octant lowering
   path may be missing a `provenance` span. This invariant is tested in 3n
   against the deterministic path and extended in 3o to cover the
   LLM-assisted path.
2. **Out-of-scope LaTeX produces diagnostics, never silent drops.** Any
   LaTeX construct outside the documented grammar (e.g. `\begin{theorem}`,
   TikZ, paragraph text, symbolic integration requests) must produce a
   diagnostic error naming the offending token, not silently be elided or
   approximated. This is a direct application of the Chelis agent
   contract's "no silent fallback" rule.
3. **Octant adds no numerical capabilities of its own.** If a lowering
   target does not exist in Nautilus / Shoals / `chelis-std`, Octant must
   emit a "not yet supported" diagnostic, not inline a local replacement.
4. **Round-trip determinism.** Parse → lower → render must be deterministic
   modulo floating-point formatting. Re-parsing the rendered output must
   produce the same SymExpr modulo whitespace and bracket normalization.

## 3a. Toolchain dependencies

**`properties/`/`references/` layout downstream of Octant.** Octant emits `.dp`
files from customer LaTeX inputs. Where the customer integrates those `.dp`
files into their reef package layout — under `src/properties/`, `src/references/`,
or top-level `properties/`/`references/` once chelis-reef supports it — is the
customer's choice, not Octant's. Until chelis-reef gains multi-source-root
support (tracked at `spec/upstream-bugs/reef-multi-source-roots.md`), customers
who follow the Shoals v0.1.0 pattern will place Octant outputs under `src/`.
Customers on a future chelis-reef will place them at the canonical root. Octant's
contract ends at emitting the `.dp`; the integration shape is downstream.

## 4. Post-Phase-3 stub: Octant document ingestion

Octant Phase 4 (from the design doc) is parked outside Phase 3. It covers
parsing full LaTeX model documents, extracting `\begin{equation}`
environments, and associating extracted formulas with surrounding prose so
model-validation teams can ingest an entire model document as a single unit.

This is tracked alongside `school` and `darwin` in
`chelis_project_plan.md` as a post-Phase-3 shell stub. No Phase 3
sub-phase implements it.
