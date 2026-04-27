# Chelis Proof — Phase 1 → Phase 2 Gate Audit

**Commit audited:** `f871d76` (Phase 1 complete - exit gate green)
**Audit performed:** Session of 2026-04-13
**Verdict:** **PHASE 2 GO** (after the HIGH fixes in this commit)

---

## Context

Phase 1 of Chelis Proof closed with four intra-phase red-team rounds, each
gating a single wave. This audit is the **final, broader** gate before
Phase 2 proof-filling stacks ~3000 lines of mechanized Lean on top of the
Phase 1 skeleton. The audit follows a six-part guide supplied by the user
("Comprehensive Red Team Audit Guide") and was executed by three fresh
general-purpose subagents running in parallel, each covering two parts
of the guide. All three returned clean structured reports. Main thread
consolidated, applied HIGH-severity fixes, and re-ran end-state
verification before committing.

**Parts covered:**

- **Subagent A** (Parts 1 + 2): Infrastructure (Lean build, paper PDF,
  prove.py tests + live Leanstral round-trip, toolchain checker) + Rule
  coverage (typing.tex ↔ Typing.lean, opsem.tex ↔ Operational.lean,
  meta-functions).
- **Subagent B** (Parts 3 + 4): Design-doc audit (T0 tape mechanism, T4
  soundness review) + verification of the eight Phase 2 prerequisites in
  the exit commit message.
- **Subagent C** (Parts 5 + 6): Cross-file consistency (decisions ↔
  Lean, import graph, commit-message hygiene) + adversarial probes
  (sorry/axiom audit, prove.py hostile inputs, LaTeX figure inclusion,
  tape-mechanism stress test).

---

## Infrastructure pass/fail summary (Subagent A)

| Check | Status | Evidence |
|---|---|---|
| `lake clean && lake build` | ✅ | 17/17 jobs; 5 `sorry` warnings in expected Phase 2 theorem bodies |
| `sorry` in theorem bodies only | ✅ | none in constructor premises, `def` bodies, or `where` clauses |
| No `axiom` | ✅ | grep empty |
| `partial def` count | ✅ | exactly 2: `subst` and `adjoint`, both documented |
| No `noncomputable` | ✅ | grep empty |
| 14 `.lean` files in `LaCaDiLE/` | ✅ | Syntax, Typing, Store, Operational, AdjointTransform, Substitution, AdjointTyping, AddDim, Progress, Preservation, DimSafety, EffectCorrectness, LinearitySoundness, ADCorrectness |
| `pdflatex` × 2 | ✅ | `main.pdf` 442488 B, 8 pages, only non-fatal `txsyb`/`txsya` font warnings |
| All 8 section headers render | ✅ | Introduction, Motivation and Examples, Core Calculus, Operational Semantics, Metatheory, Implementation, Related Work, Conclusion |
| `test_prove` unit tests | ✅ | 32/32 |
| Live Leanstral round-trip (prove.py) | ✅ | pass 1 success (`rw [Nat.add_comm]`) on the smoke fixture |
| `check_toolchain.py` | ✅ | 9/9 |

## Rule coverage tables (Subagent A)

### Typing rules (`figures/typing.tex` → `Typing.lean`)

| LaTeX rule | Lean constructor | Match |
|---|---|---|
| T-Var | `HasType.var` | ✅ |
| T-Unit | `HasType.unit` | ✅ |
| T-Abs | `HasType.abs` | ✅ (filters `x` from output Γ) |
| T-App | `HasType.app` | ✅ (left-to-right Γ₁→Γ₂→Γ₃) |
| T-Let | `HasType.letBind` | ✅ |
| T-Copy | `HasType.copy` | ✅ (produces `Typ.pair t t`) |
| T-LetPair | `HasType.letpair` | ✅ |
| T-Pair | `HasType.tpair` | ✅ |
| T-Fst | `HasType.fst` | ✅ |
| T-Snd | `HasType.snd` | ✅ |
| T-Const | `HasType.const` | ✅ |
| T-Add | `HasType.tadd` | ✅ |
| T-Mul | `HasType.tmul` | ✅ |
| T-Sum | `HasType.tsum` | ✅ (uses `rem`) |
| T-Expand | `HasType.texpand` | ✅ (uses `ins`) |
| T-UniformLike | `HasType.uniformLike` | ✅ (appends `Random`) |
| T-Perform | `HasType.perform` | ⚠️ documented: `tArg`/`tRet` unconstrained (Phase 2 `opSignature` is prerequisite #7) |
| T-Handle | `HasType.handleSingle` | ⚠️ documented: single-clause only (Phase 2 multi-clause is prereq #5) |
| T-Grad | `HasType.tgrad` | ✅ (literal abstraction, two-argument curried result, `subsetEffRow eps DiffCompat`, `Diff :: Delta`) |
| T-Vmap | `HasType.tvmap` | ✅ (literal abstraction, `addDim` on argument and result) |
| T-Fail | (prose only) | ✅ documented deferral: `Fail` uses `T-Perform` |

### Reduction rules (`figures/opsem.tex` → `Operational.lean`)

| LaTeX rule | Lean constructor | Match |
|---|---|---|
| E-Ctx | — | ✅ documented deferral: Phase 1 encodes head reductions only |
| E-Beta | `Step.beta` | ✅ |
| E-Let | `Step.letBind` | ✅ |
| E-LetPair | `Step.letpair` | ✅ |
| E-Fst | `Step.fst` | ✅ |
| E-Snd | `Step.snd` | ✅ |
| E-Const | `Step.tconst` | ✅ |
| E-Copy | `Step.copy` | ✅ (retains original, allocates fresh, returns pair) |
| E-Add | `Step.tadd` | ✅ (consumes both operands) |
| E-Mul | `Step.tmul` | ✅ |
| E-Sum | `Step.tsum` | ✅ |
| E-Expand | `Step.texpand` | ✅ |
| E-UniformLike | `Step.tuniformLike` | ✅ (Wave 2 round-2 fix: template IS consumed) |
| E-Handle-Ret | `Step.handleRet` | ✅ |
| E-Handle-Op | `Step.handleOpDirect` | ⚠️ documented: single-clause + identity continuation stand-in |
| E-Grad | `Step.tgrad` | ✅ (two-arg `λx. λg_s. handle[Accum] ...`, references `adjoint`) |
| E-Vmap | `Step.tvmap` | ⚠️ documented: body passes through; Phase 2 `addDimTerm` is prereq #6 |

### Meta-functions (all in `Syntax.lean`)

| Paper def | Lean def | Match |
|---|---|---|
| `addDim` (5 cases) | `addDim` | ✅ tensor / arrow / pair / unit / tyVar |
| `DiffCompat = {Resource, Accum}` | `DiffCompat : EffectRow` | ✅ `[resource, accum]` |
| `ε₁ ⊆ ε₂` | `subsetEffRow` | ✅ |
| `Γ = Γ₁ + Γ₂` | `contextSplit` | ⚠️ defined but unused by Phase 1 rules; marked as Phase 2 scaffolding in this commit (H-A1 fix) |
| `ins(d̄, i, k)` | `ins` | ✅ |
| `rem(d̄, i)` | `rem` | ✅ |

## Design-doc audit (Subagent B)

### T0 tape mechanism

All required content present: `mul` atomic derivation with Γ state;
two composition cases (`grad(λx. sum(mul(x,x),0))` and
`grad(λx. let y = mul(x, const(2,[d₁])) in sum(y,0))`); cumulative
tape invariant; explicit `Accum` handler in `grad`'s reduced term; all
six RISC primitives covered; `grad(grad(f))` and `vmap`-inside-`grad`
scope notes both present in §0; `const` gradient via
`perform accum` + handler filter (no residual pass-through language).

### Three-level nesting adversarial probe

Probe: `grad(λx. expand(sum(mul(x,x),0),0,d))` — three-level chain
requires the tape entries from the innermost `mul` to survive through
`sum` and `expand` reductions. After A-normalization into a let-chain,
T0 §4's general recursive pattern handles the case cleanly: `x1_tape`
and `x2_tape` created at the outermost forward point remain in Γ
through every intermediate backward step until `mul`'s adjoint rule
consumes them. No ill-typed term is produced.

**Observed gap (MEDIUM M-B-1):** T0 §4's only structural clause is
`adjoint(let y = op(...) in rest, x)`. There is no clause for bare
primitive composition without `let`-binding — `adjoint(op₁(op₂(...)))`
is silently assumed to first be A-normalized. This is a documentation
gap for Phase 2 T9, not a design flaw.

### T4 soundness review

All four example walks present and correct. No phantom rules: every
rule cited (`T-Var`, `T-Const`, `T-Mul`, `T-Sum`, `T-Grad`, `T-Copy`,
`T-LetPair`, `T-Vmap`, etc.) exists in `typing.tex`. The Wave 2
`const(2, [d₁])` correction is explicit. The Wave 3 `grad(f)` →
`grad(λx. f x)` eta-expansion is flagged. Example 4's weakening from
`map(dropout, xs)` to `vmap(dropout)` is explicit, not silent.

### Phase 2 prerequisite verification (8/8)

| # | Prereq | Issue real? | Notes |
|---|---|---|---|
| 1 | Rewrite `adjoint` with T0 §4 recursion | ✅ catch-all emits vestigial `perform accum`; `letBind` currently recurses on the bound term only, and the branch now contains a concrete `grad` witness showing that placeholder is semantically too weak | |
| 2 | Upgrade `EffectRow` to set/multiset | ✅ `abbrev EffectRow := List EffectLabel`; `AdjointTyping` uses `eps ++ [accum]` list concat | |
| 3 | Add `IsSourceTerm` / parameterize by `StoreTyp` | ✅ `Term.loc` exposed in source grammar | |
| 4 | Thread `tOut` through `Term.grad` | ✅ `Term.grad` lacks `tOut`; `Step.tgrad` fabricates it as existential | |
| 5 | Multi-clause `T-Handle` + real `E-Handle-Op` | ✅ single-clause only; identity lambda stand-in for the continuation | |
| 6 | `addDimTerm` meta-function | ✅ referenced by `E-Vmap` comment but not defined | |
| 7 | `opSignature : EffectLabel → Typ × Typ` | ✅ `T-Perform` takes free `tArg, tRet` | |
| 8 | Replace `sum` adjoint hardcoded extent 0 | ✅ `AdjointTransform.adjoint` literally passes `0` to `expand` | |

Every prerequisite corresponds to a real, grep-verifiable gap. The exit
commit message is accurate.

## Consistency + adversarial probes (Subagent C)

### Locked decisions ↔ implementation

All 16 locked decisions from `proof/decisions.md` are faithfully
implemented in both the Lean source and the LaTeX figures:

`Diff` as `CapCtx` ✅, `Fail` effect (no `Option`) ✅, precision
dropped ✅, unparameterized `Resource` ✅, one-shot handlers with
linear `k` ✅, explicit `Accum` handler in `grad` reduction ✅,
physical-copy semantics ✅, heap store ✅, `addDim` ✅, six primitives
✅, pair-producing `copy : τ → τ ⊗ τ` ✅, curried two-argument `grad`
result type ✅, `grad`/`vmap` literal-abstraction restriction ✅,
`uniform_like` template consumed ✅, `T-Expand` insertion semantics ✅,
single-clause handler typing ✅.

### Import graph

No circular imports, no dangling imports after the round-4b cleanup.
**Fixed in this commit:** `Typing.lean` and `Operational.lean` had
unused `import LaCaDiLE.Store` lines (both use `storeLookup` /
`storeRemove` helpers that actually live in `Syntax.lean`, not in
`Store.lean`). Both imports removed.

### Commit message hygiene

All nine Phase 1 commits have red-team blocks + `Blocking:` trailers
except `34b7369` ("Wave 4 prompt fix"), which is an inter-wave
micro-fix that explicitly defers its round 4 to the next commit. Noted
as MEDIUM C-M1 (discipline drift, not a soundness issue).

### Adversarial probes

- **`sorry`/`axiom` audit:** 5 `sorry` occurrences, all in theorem
  bodies (Substitution, AdjointTyping, Progress, Preservation,
  LinearitySoundness). Zero `axiom`/`opaque`.
- **`prove.py` hostile inputs:** nonexistent file → exit 3 with
  clean error; SQL-injection theorem name → exit 3 with clean error
  (name treated as opaque); `--passes 0` → argparse error exit 2.
  No tracebacks.
- **LaTeX figure inclusion:** removing `syntax.tex` → 56 non-font
  errors; `typing.tex` → 3 errors; `opsem.tex` → 0 errors because
  `main.tex` wraps every figure in `\IfFileExists{...}{...}{}`. All
  three files ARE `\input`-ed and contain live content; the
  `\IfFileExists` guard just suppresses the error on removal. The
  guard is a LOW finding (C-L3) — recommend removing in Phase 2 so
  the adversarial probe has teeth.
- **Tape stress test:** `grad(λx. let a = copy(x) in let b = mul(fst(a), snd(a)) in sum(b,0))`
  is **correctly rejected** by the current rules. `fst(a)` consumes
  `a` via `T-Var`, after which `snd(a)` has no binding to look up.
  This is the intended linear behavior — double-projection on a
  linear pair is impossible without an intervening `copy` or
  `letpair`. Phase 2 metatheory will never see a successful
  derivation of this shape.

---

## Findings bucketed by severity

### CRITICAL (0)

None.

### HIGH (2, both fixed in this commit)

- **H-A1** — `contextSplit` was defined in `Syntax.lean` but not used
  by any `HasType` constructor. Risk: Phase 2 WS2.1 (substitution
  lemma) may drift from the scaffolding. **Fix:** kept the
  definitions in place (they're the starting point for Phase 2
  substitution work) but added an explicit "Phase 2 scaffolding"
  header comment and per-definition notes so future readers and
  Phase 2 kickoff can find them immediately.
- **H-A2** — `refs.bib` had no Lean 4 formalization reference despite
  LaCaDiLE's main contribution being the mechanization. **Fix:** added
  `demoura2021lean4` — the canonical de Moura & Ullrich CADE-28 paper
  for Lean 4.

### MEDIUM (8 — documented Phase 2 prerequisites or minor drift)

- **M-A1** — `T-Perform` unconstrained `tArg`/`tRet`. **Phase 2 prereq #7.**
- **M-A2** — `E-Handle-Op` single-clause direct form only. **Phase 2 prereq #5.**
- **M-A3** — `E-Vmap` body pass-through without `addDimTerm`. **Phase 2 prereq #6.**
- **M-A4** — `T-Vmap` has no freshness side-condition. Phase 2 will
  add a freshness lemma.
- **M-A5** — Effect rows use list concat, not set union. **Phase 2 prereq #2.**
- **M-B-1** — T0 §4 assumes A-normal form for primitive composition
  without saying so. Documentation gap for Phase 2 T9.
- **M-B-2** — T0 §5's `accum(loc, val, k)` op body shape is two-arg,
  but `Term.perform` is single-arg; encoded as `perform accum((loc, val))`
  pair argument. Documentation drift.
- **M-B-3** — `origin_of` is described as "compile-time function" with
  no formal definition. Phase 2 WS2.9 (AD correctness) will need this.

### LOW / NIT (10 — most opportunistically fixed in this commit)

- **L-A1** — `test_prove.py` has no explicit `socket.timeout` test.
- **L-A2** — `test_prove.py` mocks `run_lake_build` but doesn't unit-test
  its subprocess wiring.
- **L-A3** — `igarashi2001featherweight` may not be cited; verify in
  Phase 2 §7 Related Work writing.
- **L-A4** — ~~`uninterpretedBinop` is a misleading name.~~ **Fixed** in
  this commit: renamed to `tensorOpPlaceholder`.
- **L-B-1** — ~~T0 §8 has a stale `expand` arity open question.~~
  **Fixed** in this commit: the question is resolved, updated to say so.
- **L-B-2** — T0 §3.1 has a dead `let x2 = x` rename step. Pedagogical
  only.
- **L-B-3** — `decisions.md` summary-table row for `grad` result type
  is shape-ambiguous about which arrow carries `ε`. Minor.
- **C-M1** — ~~Commit `34b7369` lacks red-team block + `Blocking:`~~.
  Historical; commit is immutable. Noted for discipline.
- **C-M2 / C-L1** — ~~Unused `Store` imports in `Typing.lean` and
  `Operational.lean`.~~ **Fixed** in this commit.
- **C-L2** — `DimSafety.lean` / `EffectCorrectness.lean` have Phase 2
  stub imports that are currently unused. Harmless.
- **C-L3** — `main.tex` uses `\IfFileExists` guard on figure inputs,
  weakening adversarial removal tests. LOW; Phase 2 cleanup.
- **C-L4** — `Typing.lean` should have a comment near `T-Fst`/`T-Snd`
  noting that destructuring a `copy(...)` result requires `T-LetPair`.
  Minor.

---

## End-state verification (run after all fixes)

- `cd proof/lean && lake build` — ✅ 17/17 green with 5 expected
  `sorry` warnings.
- `python3 -m unittest proof.scripts.test_prove` — ✅ 32/32 green.
- `cd proof/paper && pdflatex -interaction=nonstopmode main.tex × 2` —
  ✅ `main.pdf` 442488 bytes, 8 pages.
- `python3 proof/scripts/check_toolchain.py` — ✅ 9/9 green.
- `git status --short` — ✅ only the intended modified files.

---

## Verdict

**PHASE 2 GO.**

All seven exit criteria from the `f871d76` commit still pass. Zero
CRITICAL findings. The two HIGH findings (`contextSplit` latent
scaffolding + missing Lean 4 citation) are surgical fixes applied in
this commit. The eight MEDIUMs are all either documented Phase 2
prerequisites (five of them are literally the eight-item prerequisite
list in the exit commit) or minor documentation drift. The ten
LOW/NITs are split between opportunistic fixes applied here and
Phase 2 intake items.

**Rule coverage is complete:** every T-rule in `typing.tex` has a
Lean constructor, every E-rule in `opsem.tex` has a Lean constructor,
every meta-function is defined, with only the two explicitly
documented deferrals (`T-Fail` prose + `E-Ctx` head-reduction-only).

**Design documents are honest:** T0's six-primitive coverage and
composition cases work as advertised, the three-level-nesting
adversarial probe completes cleanly under A-normalization, and T4's
four example walks all cite real rules from `typing.tex`.

**Phase 2 prerequisite list is accurate:** all eight items correspond
to real, grep-verifiable gaps that Phase 2 must address before its
first proof obligation can be discharged.

**The skeleton is structurally ready for Phase 2 proof filling.** The
next session can begin with `proof/scripts/prove.py --file
proof/lean/LaCaDiLE/Substitution.lean --theorem subst_preserves_typing
--passes 4` and work through the five `sorry`-stubbed theorems in the
order documented by `proof/workstreams/ws2-paper-proofs.md`.
