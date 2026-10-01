# Octant — LaTeX ↔ Deep Bridge for Quantitative Finance

**Shell name:** Octant (`chelis-lang/octant`)

**Marine rationale:** An octant is the navigational instrument that preceded the
sextant — measuring angles between celestial objects and the horizon to determine
position at sea. It bridges observation (math) and computation (code). Compact,
precise, and the original tool of the trade.

**Depends on:** `chelis-std`, `nautilus` (required); `shoals` (required only for
SDE / Monte Carlo / yield curve lowering).

**Status:** Planned for Phase 3 as sub-phases `3n` (Part A) and `3o` (Part B), with
document ingestion parked as a post-Phase-3 stub. See `phase3n_octant.md` for the
executable sub-phase plan.

---

## 1. Problem

Quants work in five disconnected modes: whiteboard math → LaTeX documentation →
optional SymPy verification → Python/C++ implementation → Jupyter notebooks mixing
LaTeX and code. The implementation step is where bugs are born — a sign error, a
transposed index, a missing term. The LaTeX says one thing; the code says something
slightly different. Nobody catches it until P&L does not match.

The gap: LaTeX and code are separate representations of the same mathematical object
with no verified bridge between them.

## 2. What Octant Is

A thin notation layer between human mathematical conventions and Chelis's Deep
representation. **Not** a computer algebra system. **Not** SymPy. The symbolic
layer is a notation adapter — the intelligence is in the coding model (for complex
translations) and the compiler (for verification).

Three representations of every mathematical object, maintained in sync:

1. **Rendered math** (what the quant sees and edits) — standard mathematical
   notation, rendered from LaTeX.
2. **Deep** (what the compiler and agents operate on) — verified, typed, compilable.
3. **Surf** (available for inspection, not the default view) — readable code for
   power users who want to see programmatic structure.

The quant writes or pastes LaTeX. The system generates Deep. The compiler verifies
it. The rendered output shows the formula back in mathematical notation with Chelis
type annotations overlaid — dimension names on tensors, effect annotations on
stochastic terms, linearity markers on consumed values.

## 3. Architecture

```text
                    ┌──────────────┐
                    │  LaTeX input │  ← quant writes / pastes math
                    └──────┬───────┘
                           │ parse
                    ┌──────▼───────┐
                    │ Symbolic AST │  ← thin, ~30 node types matching
                    └──────┬───────┘    RISC primitives + Nautilus/Shoals
                           │ lower (deterministic for simple expressions,
                           │        LLM-assisted for complex structures)
                    ┌──────▼───────┐
                    │     Deep     │  ← typed, compilable, carries provenance
                    └──────┬───────┘
                           │ compile (standard Chelis pipeline)
                    ┌──────▼───────┐
                    │   C / HIP    │  ← binary with Greeks, Monte Carlo, etc.
                    └──────────────┘

Round trip: Deep → Symbolic AST → rendered LaTeX (with type overlays)
```

## 4. Modules

### 4.1 `Octant.Parse` — LaTeX subset parser

Parses the subset of LaTeX used in quantitative finance into a symbolic AST. This
is a bounded problem — not all of mathematics, just numerical expressions, operator
notation, and the conventions quants actually use.

**In scope:**

- Arithmetic: `+`, `-`, `\times`, `\cdot`, `/`, `\frac{}{}`
- Powers and roots: `x^2`, `\sqrt{}`, `\sqrt[n]{}`
- Transcendentals: `\ln`, `\log`, `\exp`, `e^x`, `\sin`, `\cos`
- Special functions: `\Gamma`, `\Phi` (normal CDF), `\Phi^{-1}`, `\text{erf}`,
  `B(a,b)` (beta)
- Summation and products: `\sum_{i=0}^{N}`, `\prod_{i=1}^{n}`
- Expectations and probabilities: `\mathbb{E}[X]`, `\mathbb{P}(A)`
- Derivatives: `\frac{\partial V}{\partial S}` → `grad(V, wrt=S)`, `\nabla`
- Integrals: `\int_a^b f(x)\,dx` → `Nautilus.Integrate.adaptive_simpson(f, a, b)`
- Conditional / piecewise: `\begin{cases} ... \end{cases}`
- Standard finance notation: `N(d_1)` (normal CDF), `\sigma\sqrt{T}`, `dW_t`
  (Wiener increment)
- Subscript / superscript conventions: `S_0` (initial price), `r_f` (risk-free
  rate), `\sigma_{imp}` (implied vol)
- Matrix notation: `\mathbf{A}^{-1}`, `\mathbf{A}^\top`, `\det(\mathbf{A})`,
  `\text{tr}(\mathbf{A})`

**Out of scope:**

- Symbolic integration / simplification / equation solving (that is CAS territory)
- Proof construction
- Text-mode LaTeX (paragraphs, sections, bibliographies)
- TikZ / diagrams

**Output:** a `SymExpr` AST — a tree of ~30 node types that map directly to Chelis
RISC primitives and Nautilus / Shoals library calls.

### 4.2 `Octant.Symbolic` — The thin AST

```text
SymExpr ::=
  | Literal(f64)
  | Variable(name: String, subscripts: Vec<String>)
  | BinOp(op: {Add, Sub, Mul, Div, Pow}, lhs: SymExpr, rhs: SymExpr)
  | UnaryOp(op: {Neg, Sqrt, Exp, Log, Sin, Cos, Abs}, arg: SymExpr)
  | SpecialFn(name: {Erf, ErfInv, Gamma, LogGamma, Digamma, Beta,
  |            NormalCDF, NormalPDF, NormalInvCDF}, args: Vec<SymExpr>)
  | Derivative(expr: SymExpr, wrt: Variable)        -- maps to grad
  | Integral(integrand: SymExpr, var: Variable,
  |          lower: SymExpr, upper: SymExpr)         -- maps to Nautilus.Integrate
  | Sum(body: SymExpr, index: Variable,
  |     lower: SymExpr, upper: SymExpr)              -- maps to fold
  | Product(body: SymExpr, index: Variable,
  |         lower: SymExpr, upper: SymExpr)          -- maps to fold with mul
  | Expectation(expr: SymExpr, over: Variable)       -- maps to Monte Carlo or analytic
  | Conditional(branches: Vec<(SymExpr, SymExpr)>,
  |             otherwise: SymExpr)                  -- maps to if/else chain
  | MatrixOp(op: {Inverse, Transpose, Det, Trace,
  |           Solve, Cholesky}, args: Vec<SymExpr>)  -- maps to Nautilus.LinAlg
  | SDEDrift(expr: SymExpr)                          -- marks SDE drift term
  | SDEDiffusion(expr: SymExpr)                      -- marks SDE diffusion term
  | Annotation(expr: SymExpr, metadata: Map)         -- provenance, dimension hints
```

This is intentionally small. No algebraic simplification, no symbolic integration,
no pattern matching beyond what is needed to recognize standard forms. The AST is
a notation bridge, not a CAS.

### 4.3 `Octant.Lower` — Symbolic AST → Deep

Two paths:

**Deterministic lowering** for expressions that map mechanically to Chelis:

- `\frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}` →
  `div(add(log(div(S, K)), mul(add(r, div(mul(sigma, sigma), 2.0)), T)),
   mul(sigma, sqrt(T)))`
- `\frac{\partial V}{\partial S}` → `grad(V, wrt=S)`
- `N(d_1)` → `Nautilus.Distributions.normal_cdf(d1)`
- `\int_0^T f(t)\,dt` → `Nautilus.Integrate.adaptive_simpson(f, 0.0, T)`
- `\mathbf{A}^{-1}\mathbf{b}` → `Nautilus.LinAlg.solve(A, b)` (recognize
  inverse-times-vector as solve)

**LLM-assisted lowering** for structures that require discretization choices,
numerical method selection, or control flow design:

- An SDE like `dS = \mu S\,dt + \sigma S\,dW_t` needs: choice of discretization
  (Euler-Maruyama vs Milstein), time grid, noise generation strategy, path
  storage. The coding model generates the `Shoals.Stochastic` call structure.
- A calibration objective like `\min_\theta \sum_i (V_{model}(\theta) -
  V_{market})^2` needs: choice of optimizer (L-BFGS vs Nelder-Mead), convergence
  criteria, initial guess strategy. The coding model generates the
  `Nautilus.Optim` call structure.
- A Monte Carlo expectation `\mathbb{E}[f(S_T)]` needs: number of paths, variance
  reduction choice (antithetic, control variate), random key threading. The
  coding model generates the `Shoals.Pricing` Monte Carlo engine call.

The boundary between deterministic and LLM-assisted is: if the LaTeX uniquely
determines the computation (arithmetic, derivatives, function application), lower
deterministically. If the LaTeX specifies the *what* but not the *how* (discretize
this SDE, optimize this objective, estimate this expectation), the coding model
fills in the *how*.

### 4.4 `Octant.Render` — Deep → LaTeX (round-trip)

Pretty-prints the compiler's typed AST back to mathematical notation. This is the
inspection / review view.

**Type overlay rendering:**

- Named tensor dimensions → subscripts: `tensor[instrument, scenario, f32]`
  renders as `T_{instrument \times scenario}`
- Effects → color-coded markers: a `key` parameter renders as a blue die marker,
  `! { IO }` renders as a yellow lightning marker. Pure expressions have no
  marker.
- `grad(f, wrt=x)` → `\frac{\partial f}{\partial x}` or `\nabla_x f`
- `vmap(f)` → renders with a batch subscript: `f_{\text{batch}}`
- Linearity annotations → consumed variables get a single-use marker (for
  example, overline)

**Finance-specific rendering:**

- `grad(price, wrt=spot)` → `\Delta` (delta)
- `grad(price, wrt=vol)` → `\mathcal{V}` (vega)
- `grad(price, wrt=rate)` → `\rho`
- `grad(price, wrt=T)` → `\Theta`
- These are pattern-matched from the `wrt` variable name against known
  conventions — `spot/S → Delta`, `vol/sigma → Vega`, `rate/r → Rho`,
  `T/time/tau → Theta`. Configurable.

### 4.5 `Octant.Provenance` — Source linking

Every Deep node carries an optional provenance annotation linking it back to the
LaTeX expression it was generated from:

```text
(app {provenance "\\frac{\\ln(S/K)}{\\sigma\\sqrt{T}}", source_line 3, source_col 12}
  (var {} div)
  (app {provenance "\\ln(S/K)"} (var {} log) (app {} (var {} div) (var {} S) (var {} K)))
  (app {provenance "\\sigma\\sqrt{T}"} (var {} mul) (var {} sigma) (app {} (var {} sqrt) (var {} T))))
```

This enables:

- **Audit trail:** "This compiled code implements this formula. Here is the link
  between each code fragment and the mathematical expression it came from."
- **Error localization:** when the compiler reports a dimension mismatch, the
  error message can point to the LaTeX expression that caused it, not just the
  Deep node.
- **Model validation workflow:** a reviewer sees the rendered formula, clicks on
  a sub-expression, and sees the compiled Deep with type annotations for exactly
  that piece. "Does `\sigma\sqrt{T}` have the right dimensions?" — the overlay
  shows `f32 × f32 → f32`, confirming it is a scalar-scalar multiply.

**This is the core value proposition of Octant and therefore lands in sub-phase
`3n` (Part A), not `3o`.** A Black-Scholes `d_1` round-trip without source spans
on every emitted Deep node would ship a parser, not a product.

### 4.6 `Octant.Notebook` — Interactive environment

A cell-based environment where each cell is a mathematical expression that is
simultaneously rendered LaTeX and live compiled Deep.

**Cell types:**

- **Formula cell:** LaTeX input → compiled Deep → rendered math with type
  overlays. Edit the formula, the code regenerates, the compiler re-verifies,
  the type annotations update.
- **Parameter cell:** declares named parameters with types and dimensions.
  `S_0 = 100.0 : f32` (initial spot price), `\sigma = 0.2 : f32` (volatility).
- **Execution cell:** evaluates a compiled expression with the current parameter
  bindings. Shows the numerical result alongside the formula.
- **Greek cell:** a derivative expression that renders as
  `\Delta = N(d_1) = 0.6327` — the formula, its symbolic simplification (if the
  system can produce one), and its numerical value, all in one display.

**This is NOT a Jupyter kernel.** It is a Chelis-native environment. The cells
produce Deep, not Python. The execution runs compiled C, not interpreted code.
The rendering is mathematical notation, not code blocks.

Whether this is implemented as a web app, a VS Code extension, a terminal UI
(extending Cove), or a standalone desktop app is an implementation decision, not
a design decision. The shell provides the core capabilities (parse, lower,
render, provenance); the UI layer consumes them.

Notebook lands in sub-phase `3o` (Part B) because it needs the full parse / lower
/ render / provenance surface **plus** finance notation to be useful.

---

## 5. What Octant Does NOT Do

- **Symbolic algebra.** No simplification, no equation solving, no symbolic
  integration. If you need `\int e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}
  \text{erf}(x)`, you already know the answer — you write `erf(x)` directly.
  Octant does not derive it for you.
- **Theorem proving.** Octant does not prove that your pricing formula is
  arbitrage-free or that your risk measure is coherent. It verifies that your
  code implements your formula (via provenance + type checking), not that your
  formula is mathematically correct.
- **Replace SymPy.** SymPy is 750k+ lines covering symbolic integration, group
  theory, polynomial algebra, etc. Octant is a notation bridge — maybe 5-10k
  lines of parser + lowering + renderer.
- **Parse arbitrary LaTeX documents.** Octant parses mathematical expressions,
  not LaTeX documents. No `\section{}`, no `\begin{theorem}`, no BibTeX. Feed
  it the formula, not the paper. Full-document ingestion is parked as a
  post-Phase-3 stub.

---

## 6. User Workflows

### Quant researcher at a fund

1. Derives a new signal model on a whiteboard.
2. Types the formulas into Octant cells as LaTeX.
3. Octant compiles each formula to Deep, type-checks it, renders it back with
   dimension annotations.
4. The quant sees: "This formula operates on `tensor[instrument, f32]` and
   produces `tensor[instrument, f32]` drawing from a `key`" — confirming the
   dimensions and stochastic structure match their intent.
5. Runs a backtest on compiled code directly from the notebook. No Python
   translation step.

### Model validation team at a bank

1. Receives a model document in LaTeX (standard practice).
2. Pastes the pricing formulas into Octant.
3. Octant compiles them and shows rendered math with type annotations.
4. The validator verifies: "The compiled representation matches the documented
   model" — by comparing rendered math to the source document, not by reading
   Python / C++ code.
5. Provenance links let the validator click any sub-expression and see: the
   original LaTeX, the compiled Deep, the type information, and the numerical
   result for a test case.

### Structuring desk creating a new product

1. The structurer defines the payoff formula in LaTeX:
   `\text{payoff} = \max(S_T - K, 0)` or something exotic.
2. Octant compiles it, the compiler verifies dimensions and effects.
3. `grad(payoff, wrt=S)` gives delta automatically — no quant developer writes
   Greek code.
4. Monte Carlo pricing via `\mathbb{E}[\text{payoff}]` generates a
   `Shoals.Pricing` call with the compiled payoff as the inner function.
5. The desk has a compiled pricer with automatic Greeks for a new product, same
   day.

---

## 7. Dependencies and Ecosystem Position

```text
chelis-std (core)
    ├── nautilus (numerical methods)
    │       ↑ Octant.Lower maps integrals → Nautilus.Integrate,
    │         special functions → Nautilus.Special,
    │         linear algebra → Nautilus.LinAlg,
    │         distributions → Nautilus.Distributions,
    │         optimization → Nautilus.Optim
    │
    ├── coral (dataframes)
    │       ↑ Octant could lower tabular operations but this is stretch scope
    │
    ├── shoals (finance) — required for SDE / MC / curves
    │       ↑ Octant.Lower maps SDE notation → Shoals.Stochastic,
    │         Monte Carlo expectations → Shoals.Pricing,
    │         yield curve operations → Shoals.Curves
    │
    └── octant (LaTeX ↔ Deep bridge)
            ├── Octant.Parse (LaTeX subset → SymExpr)
            ├── Octant.Symbolic (SymExpr AST)
            ├── Octant.Lower (SymExpr → Deep, deterministic + LLM-assisted)
            ├── Octant.Render (Deep → LaTeX with type overlays)
            ├── Octant.Provenance (source linking between LaTeX and Deep)
            └── Octant.Notebook (interactive cell-based environment)
```

Octant is a consumer of Nautilus, Coral, and Shoals — it maps notation to their
APIs. It does not provide numerical capabilities of its own. If Nautilus does
not have a function, Octant cannot lower to it.

---

## 8. Implementation Phases (mapped to Chelis sub-phases)

The original design doc proposed four Octant-internal phases. Mapped against the
Chelis Phase 3 dependency graph:

| Octant phase | Chelis sub-phase | Contents |
|---|---|---|
| 1. Parser + deterministic arithmetic lowering | `3n` (Part A) | `Octant.Parse`, `Octant.Symbolic`, `Octant.Lower` (deterministic path), `Octant.Render`, **`Octant.Provenance`** |
| 2. Finance-specific notation + Nautilus / Shoals integration | `3o` (Part B) | `Octant.Lower` LLM-assisted path for SDEs / MC / calibration, matrix and integral lowering through Nautilus, Greek rendering pattern matches |
| 3. Notebook | `3o` (Part B) | `Octant.Notebook` cell kinds and runtime |
| 4. Model document ingestion | post-Phase-3 stub | parses full LaTeX papers, extracts `\begin{equation}` blocks, associates formulas with surrounding prose |

Rationale for the split:

- `3n` has the same prerequisite as `3l: Shoals` — both need `3j: Nautilus`
  green. `3n` therefore runs **in parallel with `3l`**, giving the user's
  requested "around the same time frame as Shoals."
- `3o` needs `3l` (Shoals) green for SDE / MC / curve lowering, so it is
  sequential after `3l`.
- Provenance is moved into `3n` (originally Octant Phase 3) because the audit
  trail is Octant's core value proposition. Without it, the first round-trip
  test is meaningless.
- Document ingestion is parked as a post-Phase-3 stub alongside `school` and
  `darwin`: its parser scope is significantly larger than the quant-finance
  expression subset, and no Phase 3 user workflow depends on it.

The executable sub-phase contracts (acceptance oracles, test plans, non-silent
deferrals, infrastructure decisions) live in `phase3n_octant.md`.

---

## 9. Open Questions

1. **Where does the LaTeX parser live?** Options: (a) pure Chelis
   implementation in the octant reef package, (b) Rust crate linked via FFI
   (like potential future nalgebra), (c) existing Rust LaTeX parser crate (for
   example `pulldown-latex`, `latex2mathml` internals). Option (c) is pragmatic
   for `3n`; option (a) is the long-term if Chelis's string processing matures
   enough. This decision is owned by `phase3n_octant.md` and must be confirmed
   with the user before `3n` coding starts.

2. **How does LLM-assisted lowering integrate?** The coding model needs to see
   the SymExpr AST and produce Deep. This could be: (a) a `SKILL.md`-style
   prompt with the SymExpr as context, (b) a fine-tuned model that maps
   SymExpr → Deep directly, (c) an MCP server that the notebook calls.

3. **Is Octant a reef package or a separate tool?** The parser and AST are
   library code (reef package). The notebook is an application. These might be
   separate deliverables — `chelis-lang/octant` (the library) and
   `chelis-lang/octant-notebook` (the UI), or the notebook could be a feature
   of Cove (the existing TUI environment).

4. **What about MathML / Presentation MathML as an alternative input format?**
   Some systems (Word equation editor, web-based tools) produce MathML rather
   than LaTeX. Supporting both input formats doubles the parser work but
   increases the addressable workflow. Defer unless a specific customer
   workflow requires it.

5. **Interaction with the Lean mechanization.** The provenance annotations and type
   overlays apply the type system's guarantees to a formula. Octant cites those
   guarantees; the mechanization does not depend on Octant.
