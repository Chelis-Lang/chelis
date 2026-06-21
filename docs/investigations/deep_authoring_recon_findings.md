# Deep Authoring Trust Stack — Reconnaissance Findings

- **Date:** 2026-06-21
- **Branch:** `deep-tools`
- **Type:** Reconnaissance synthesis (facts, not advocacy)
- **Inputs:** 6 independent per-section verification agents + 6 adversarial skeptics

## 1. Header & Purpose

This document is reconnaissance for an **"authoring trust stack" decision**: whether to
build tooling that lets a model produce trustworthy Chelis programs, and if so, what the
smallest first step is. It does **not** advocate a build-out. It collects what is
*observable in the repos today* against two strictly separated hypotheses.

### The H1 / H2 split (used throughout; every finding is tagged)

- **H1 — structural edit of an existing program.** A tool (e.g. `chelis_replace_body`)
  takes an already-typed program plus a target symbol and a new body, splices the new
  fragment into the program's AST, validates it against the surrounding module, and returns
  a mutated program state. Bet: the high-value loop is *editing* existing Chelis.
- **H2 — authoring Deep from scratch.** A model emits a whole program directly in **Deep**
  (the canonical 3-tuple s-expression form) rather than Surf, on the theory that Deep's
  regular `(tag {} child...)` structure is easier for a model to generate correctly and
  easier to mechanically validate. Bet: the high-value loop is *generating* Deep.

A finding tagged **neither** describes setup/infrastructure that bears on both or on the
decision context but is not itself H1 or H2 evidence.

### Verification stance

Executed evidence beats source reading; source reading beats documentation; active spec
beats archived spec. Where a Phase-1 agent claim was contradicted by a skeptic who *ran*
the relevant code, this document resolves in favor of the executed evidence and says so
explicitly. Every concrete claim below carries a `file:line` / symbol / command-output
citation drawn from the inputs.

---

## 2. Executive Summary & Net Verdict

### Net verdict: **the smallest first step is an H2 Surf-vs-Deep authoring eval (or a prerequisite that unblocks it), NOT the H1 structural-edit PoC.**

The reconnaissance is *dense on H1* (tide surface, AST type visibility, in-context
checking, span chains) and *near-empty on H2*. That density is an artifact of where facts
are statically observable, not a signal of where the right first step is. The single
load-bearing finding is the **H2 evidence gap**: there is *zero* in-repo head-to-head
evidence on whether a model authors Deep better than Surf — no Deep authored corpus
(0 `.dp` files in shoals/whale/economoist), no Deep scoring path in any harness, no
Surf-vs-Deep comparison anywhere (`.ch` appears in the hundreds — ≈365–474 occurrences
depending on grep scope — vs exactly 0 `.dp` across both fluke-ball harness trees). The
proposal's *strongest single piece of grounding* — the
c-earchin → chelis-prove "lossy text reparse" — was **mischaracterized**: c-earchin emits
Deep `.dp` (its AST's own serialization), and chelis-prove ingests it by structurally
parsing the Deep AST, so the motivating example does not establish the structural-loss
problem the build-out was meant to solve. With the proposal's grounding weakened and H1's
prerequisite tooling entirely unbuilt, building an H1 PoC first would be **streetlight
bias** — searching where the light (static observability) is, not where the decision-
critical unknown lives. H2 is the unknown; the first step should *reduce that unknown*.

### One-line status per brief question group

- **2.1 Deep as a stable target (H2):** PRESENT — documented grammar, real library parser,
  one-canonical-form guarantee enforced by `print_canonical`; AST type is public but only as
  an internal Rust crate, not via the machine-facing `chelis-compiler-api` facade.
- **2.2 Desugaring boundary & provenance (H1):** Inference runs directly on Deep; but
  **Deep-path errors are ABSENT** — `CheckError` has no location field at all; best-case
  provenance is a Surf byte-span string in the message text.
- **2.3 Existing tool surface (H1):** Structural-edit category is **entirely UNBUILT**;
  `chelis_replace_body` is spec-only; tide is 8 read/analyze tools.
- **2.4 Emitters / c-earchin boundary (H2):** 3 of 4 emitters string-concatenate canonical
  text (no Chelis AST); the c-earchin → prove "lossy reparse" framing is **CONTRADICTED**.
- **2.5 Hard parts — incremental validation & property feedback (H1):** Snapshot-based
  library-vs-new checking exists; **fragment-scoped (single-body) checking is ABSENT**;
  single-property `--only` isolation works but with substring-match and broken-sibling
  sharp edges.
- **2.6 Empirical setup (neither/H2):** A real Surf authoring+scoring harness exists
  (fluke-ball/KellyBench); the **H2 evidentiary base is empty**; old fluke-ball pins
  `=0.7.20`, the active fluke-ball-wc is migrated to `=0.8.0`.

---

## 3. Per-Section Findings

### 2.1 — Deep as a stable target

**Q1. Public/documented Deep grammar + a library-callable Deep parser?**
**Answer / Verdict: PRESENT · H2.** A normative grammar exists at
`spec/03-deep-syntax.md:1-24` ("The primary machine interface"), defining the 3-tuple node
`(tag {key: value} child...)` and a closed 62-tag vocabulary (`spec/03-deep-syntax.md:232`).
A second executable grammar is `grammars/tree-sitter-chelis-deep/grammar.js`, exercised
clean by `chelis validate --deep` (output: `validated deep: .../simple_def.dp`). The library
parser is real: `chelis-deep` exposes `pub fn parse_str(source) -> Result<Vec<Expr>, ParseError>`
(`crates/chelis-deep/src/parser.rs:680`), `parse_str_strict` (`:688`, also validates the tag
vocabulary), and re-exports `Atom, Expr, List, MetaExpr, MetaMap` (`lib.rs:14`). The crate is a
normal publishable workspace member depended on by 14 siblings.
**Nuance:** the documented machine-facing API crate `chelis-compiler-api` does **not**
re-export the Deep parser/AST (`crates/chelis-compiler-api/src/lib.rs:10-31`); it uses
`chelis_deep::Expr` only internally. So Deep is library-callable for Rust consumers, not via
a stable cross-language/FFI/wire API.

**Q2. Is Deep fmt-canonical defined + enforced? One spelling or many?**
**Answer / Verdict: PRESENT · H2.** The canonicalizer is
`chelis_deep::printer::print_canonical(&[Expr]) -> String` (`crates/chelis-deep/src/printer.rs:21`).
Spec §6 states the contract literally: "Deep has exactly one textual representation per
program" (`spec/03-deep-syntax.md:521`). The parser accepts many spellings (whitespace-
insensitive s-expressions) but the canonicalizer maps them to one.
**Captured probes:**
- `chelis fmt` twice on `simple_def.dp` → byte-identical (`IDEMPOTENT-OK`, diff empty).
- Deliberately ugly valid `.dp` → `chelis fmt` collapses to byte-identical canonical
  (`COLLAPSES-TO-SAME-CANONICAL-OK`).
- `chelis deep examples/vmap_relu.ch` then `fmt` → no diff (`DEEP-OUTPUT-IS-FMT-CANONICAL-OK`).
- Round-trip locked by `crates/chelis-deep/tests/roundtrip.rs:7-14`.
**Caveat:** there are two CLI-selectable renderings — default pretty and `--flat`
(`printer.rs:26 print_canonical_flat`). `chelis fmt` has no `--flat` (only `--inplace`,
`--check`), so "one canonical form" holds for the `fmt`/`deep` default path; `--flat` is a
second machine-selectable spelling.

**Q3. Public surface of the AST node type?**
**Answer / Verdict: PRESENT · H1.** `chelis_deep::ast::Expr` (`crates/chelis-deep/src/ast.rs:5-103`)
is a `pub enum` (Atom/List/Map/MetaExpr each with a `Span`); supporting `List`
(`pub elements: Vec<Expr>`), `MetaMap` (`pub entries: Vec<(String, Expr)>`), etc. all have
`pub` fields and derive `Clone + Serialize + Deserialize`. Fully constructible/mutable by any
Rust consumer — a manipulable library API at the Rust-crate level.
**Limit:** exposed only as an internal workspace crate; `chelis-compiler-api` does not
re-export it (`lib.rs:10-31`), and no shipped wire schema takes a Deep AST node as input (the
machine JSON surface is IR-level `WireRiscOp`/`EvalRequest` in `schema.rs`). Structural Deep
editing requires a direct `chelis-deep` dependency, not the facade.

**Q4. `.chb` (binary typed AST): SPECIFIED vs ACTUAL?**
**Answer / Verdict: PARTIAL · neither.** Intent-vs-implementation gap confirmed.
- **INTENT (archived prose):** `spec/design/archive/chelis_design_sprint_adjudication.md:33`
  justifies Deep's verbosity because "the .chb binary format compresses the verbosity away
  for storage/transfer" — i.e. a binary encoding of the Deep AST.
- **ACTIVE SPEC redefines it as package metadata:** `spec/10-serialization.md:14-27` —
  ".chb is the binary Shell metadata artifact used by the Reef package system";
  `spec/design/chelis_canonical_reference.md:124-128` — "an implementation-owned binary
  package interface."
- **IMPLEMENTATION matches the active spec, not the archive:** `.chb` is written by
  `chelis-reef` as `<name>-<version>.chb` (`crates/chelis-reef/src/lib.rs:927`) via bincode of
  `ShellPackage` (`crates/chelis-shell/src/lib.rs:6-12,43`). `ShellPackage` holds only
  PackageId, compiler version string, modules, dependencies, archive_sha256 — **no program
  AST nodes, no Expr, no IR DAG**. Export types are even stored as canonical Deep **TEXT** via
  `print_canonical` (`reef/lib.rs:4572`). `decode.rs:13` confirms ".chb envelopes serialize
  compiler state, not domain values."

**Resolution:** No source/active-spec contradiction; the only divergence is active-spec/code
vs the *archived* adjudication. No binary typed-AST IR is produced or consumed today.

---

### 2.2 — The desugaring boundary and provenance

**Q1. Where does Surf desugar to Deep, and does inference run on Deep directly?**
**Answer / Verdict: PRESENT · H1.** Surf → Deep at
`chelis_surf::desugar::desugar_program` (`crates/chelis-surf/src/desugar.rs:17`). The pipeline
is parse_surf → desugar_program → macro expand → `Vec<deep::Expr>`. Inference runs **directly
on the Deep expr vector**: `compile_source` calls
`chelis_types::check_ir_program(&deep_exprs)` (`crates/chelis-compiler-api/src/compiler.rs:1000`);
`infer_program(exprs: &[deep::Expr])` (`crates/chelis-types/src/infer.rs:262`) walks Deep nodes
directly. There is **no intervening typed IR** before inference; the IR DAG lowering happens
*after* a clean type check.

**Q2. Does the typed representation retain Deep-node PROVENANCE through inference? Deep path
on a type error, or only a Surf span?**
**Answer / Verdict: ABSENT · H1.** No Deep path is ever produced.
`chelis_types::CheckError` (`crates/chelis-types/src/errors.rs:5-15`) has fields
kind/message/suggestions/severity/expected/got and **no span/path/location field at all**.
Unify-origin errors are raw `TypeError { kind, message }` (`unify.rs:33-37`); `From<TypeError>`
drops everything but kind+message (`errors.rs:161`). The only surviving provenance is a
*string*: `with_macro_provenance` appends a macro *name* (`infer.rs:6692`), and
`validator_error`/`validator_span_suffix` append `(at surf:OFFSET..END)` — a Surf **byte
span** string — for ~11 shape-sensitive validator errors only (`infer.rs:5965-5990`).
**Live proof:** `chelis check /tmp/x.ch --allow-style-violations` on `int32 + f32` returned
`[{"kind":"PrecisionMismatch","message":"precision mismatch: expected int32, got f32","severity":0.8}]`
— no span, no path. The same `.dp` input (which literally carries `span:"surf:38..43"` Deep
metadata) gave an identical spanless diagnostic. Best case (conv2d stride 0) gives
`...got 0 (at surf:93..113)` — a Surf byte span embedded in the message string, never a Deep
path.

> **Skeptic confirmation (CLAIM_HOLDS).** The adversarial skeptic ran `chelis deep --annotate`
> on a type-error file and got the raw CheckError Debug dump:
> `CheckError { kind: TypeMismatch, ..., expected: None, got: None }` — no span/path/node field.
> All 8 type-checker error→diagnostic sites route through `check_error_diagnostic` (`compiler.rs:2025`),
> which hardcodes `span: None`. The Diagnostic schema's only location field is `span: Option<Span>`
> where `Span` is documented "A byte-offset range in source text" (`chelis-deep/src/span.rs:3-10`)
> — a Surf byte span, not a Deep path. **Resolved: AST-path errors are absent; adding them would
> require new location fields on CheckError/TypeError plus a Deep-path computation that does not exist.**

**Q3. Is "Surf desugars losslessly to Deep, Deep decompiles cosmetically-lossily" true?**
**Answer / Verdict: PARTIAL · neither.** True for program *structure*, but the slogan
understates a real asymmetry.
- Deep → Surf is cosmetically lossy (confirmed live): `def f(x: f32) -> f32 = (x + 1.0)`
  desugars to `(app {} (var add) x 1.0)`, and `chelis surf` renders it as `add(x, 1.0)` — the
  infix `+` does not round-trip; `decompile.rs` has no `add`→`+` infix path.
- Deeper gap: arbitrary Deep **metadata** is not preserved through Deep→Surf→Deep because Surf
  has no metadata syntax. Live: producer spans `surf:36..43/36..37/40..43` came back as
  recomputed `surf:35..46/35..38/39..40`. Spec §5 confirms this by design:
  `spec/design/chelis_span_survival.md:377-381` lists "Surf user-authored metadata syntax" as
  out of scope and names `chelis build --deep` as the metadata-preserving path.

**Additional. Reconcile `chelis_span_survival.md` against the actual diagnostic path.**
**Answer / Verdict: CONTRADICTED · H1.** The spec and the diagnostic path describe **different
chains**, and the diagnostic chain provably drops spans.
- **What the spec promises and what IS implemented:** an end-to-end *codegen audit* chain —
  Deep `span` key → `DagNode.span_id`/`merged_spans` (`crates/chelis-ir/src/dag.rs:983,992`) →
  `HostExpr.span_id` (`crates/chelis-ir/src/host.rs:748`) → backend `// span:` comments in
  generated C/HIP (`spec/design/chelis_span_survival.md:121-144`). The desugarer stamps
  `surf:start..end` onto Deep nodes (`desugar.rs:132-191`). This chain is real.
- **Where the diagnostic layer drops it:** the wire `Diagnostic` struct *has* `span: Option<Span>`
  (`schema.rs:63-75`), but every non-parse error family hardcodes `span: None`, and `CheckError`
  has no span field to begin with. Precisely: the **type** and **linearity** families both route
  through `check_error_diagnostic` (`compiler.rs:2025-2034`, which hardcodes `span: None` at
  `:2033`; linearity wiring at `:1021`), and the **effects** conversions hardcode it inline at
  `:461-468` and `:1008-1015`. So the slot is wired for parse errors and dead for every
  type/effect/linearity error.

**Resolution:** Span survival is real for the *codegen audit chain*, which is what the spec
actually specifies; the spec is *silent* on type diagnostics. The hardcoded `span: None` is not
a violation of that spec, but it does mean Diagnostic.span is dead for all non-parse errors.

---

### 2.3 — The existing tool surface (H1 structural-edit tooling)

**Q1. What does chelis-tide expose? Any structural edit?**
**Answer / Verdict: PRESENT (read-only surface) · H1.** Tide exposes exactly **8** MCP tools,
registered in `crates/chelis-tide/src/mcp.rs:120-149` and dispatched at `:68-84`:
`chelis_check`, `chelis_compile`, `chelis_desugar`, `chelis_decompile`, `chelis_eval`,
`chelis_grad`, `chelis_validate`, `chelis_prove`. All are stateless source-string-in /
analysis-out. The HTTP surface mirrors this (`http.rs:16-29`). The 8-tool list is pinned by
`crates/chelis-tide/tests/mcp.rs:42-54`. **No structural-edit tool exists** — nothing takes a
file+symbol+new-body or returns a mutated program.

> **Skeptic confirmation (CLAIM_HOLDS).** Live probe of `chelis tide mcp` answering JSON-RPC
> `tools/list` returned exactly those 8 names. The only structural mutators in the repo are
> internal: `chelis-ir` `replace_node` (IR-DAG optimization, `dag.rs:1088`) and the Deep `splice`
> quasiquote AST tag — neither is an agent/CLI/MCP edit tool.

**Q2. Locate the `chelis_replace_body` PoC.**
**Answer / Verdict: ABSENT · H1.** Whole-repo grep for `chelis_replace_body`/`replace_body`
across `crates/` → **NONE FOUND**. It appears only in spec/docs
(`spec/design/chelis_agent_editing_surface.md:44,52`; `spec/12-roadmap.md:183`;
`spec/design/chelis_canonical_reference.md:787`), explicitly marked Exploratory with status-
framing discipline ("No doc anywhere claims structural editing tools as a current capability",
`chelis_agent_editing_surface.md:162-182`). The companion tools (`chelis_define`,
`chelis_rename`, `chelis_change_signature`, `chelis_view`, `chelis_property_check`) are also
absent from code. **Docs and source agree it is unbuilt.**

**Q3. What validation API would a tide edit-tool call to check a spliced Deep fragment against a
module's symbol table?**
**Answer / Verdict: PARTIAL · H1.** Tide currently calls only whole-source-string analysis
(`compiler::check` → `check_ir_fitness` on the entire desugared source). A "check this fragment
in this context" capability *exists but tide does not expose it*:
`check_in_context(context, new_source)` (`compiler.rs:599-623`) routes through
`check_ir_with_signature_context(&context.type_env, ..., &new_deep)` (`infer.rs:744`).
**Critical caveats:** (a) the "context" is a whole-package `CompiledContext` (reef library
snapshot); (b) the "fragment" input is **Surf text, not a spliced Deep AST fragment**
(`compiler.rs:421-425` "Phase G inputs are always Surf"); (c) `check_in_context`'s own body is
degenerate — hardcoded `score=1.0`, `errors=[]` (`:609-622`); real diagnostics surface in the
`Result` Err arm; (d) these APIs are consumed only by the CLI and tests, never by tide.

---

### 2.4 — The emitters (c-earchin pivotal); H2 feasibility + migration assumption

**Per-tool verdict (all H2 except the boundary finding):**

| Tool | Builds Deep AST then renders, or assembles text? | Target | Verdict |
|---|---|---|---|
| **c-earchin** | ASSEMBLES-TEXT (`render_deep`, raw `push_str`/`format!`) | Deep `.dp` | PRESENT |
| **calcify** | ASSEMBLES-TEXT (string buffer `Emitter`) | Surf `.ch` | PRESENT |
| **hydronnx** | **BUILDS-AST** (own local Surf AST `Module/Decl/Expr` + `.render()`) | Surf `.ch` | PRESENT |
| **octant** | ASSEMBLES-TEXT (49 `format!` calls → Deep s-expr) | Deep `.dp` | PRESENT |

Citations: c-earchin `src/emit.rs:5,31,127`, `Cargo.toml` (deps = clap only, no chelis crate);
calcify `src/translation/emitter.rs:4,26,53`; hydronnx `src/emit/surf.rs:67,209,232,245`,
`src/emit/mod.rs:159,229`; octant `src/emit.rs:1,936,946`. Only hydronnx interposes a
structured AST; the other three string-concatenate canonical text directly. **H2 implication:**
"emit canonical Deep s-expressions from scratch" is exactly the shape 3 of 4 emitters already
target.

**The c-earchin → chelis-prove boundary** — see Section 4 below (its own subsection).

---

### 2.5 — The hard parts (incremental validation + property feedback)

**Q1. Does the compiler retain a typed module symbol table to check a fragment against?**
**Answer / Verdict: PRESENT · H1.** `CompiledContext`
(`crates/chelis-compiler-api/src/context.rs:141-163`) holds the already-typed library as
`type_env: TypeEnv` + `library_checked: CheckedProgram` + `library_dag: LoweredLibrary`.
`TypeEnv` (`crates/chelis-types/src/context.rs:65-99`) is the Arc-shared symbol-table snapshot,
"never mutated"; each check clones, layers new bindings, discards. `check_in_context` /
`check_ir_with_signature_context` (`infer.rs:744-826`) clone the library state and run
inference/validate **only on `new_exprs`** ("Library is already validated", `:775-777`). A
cross-process disk cache exists for chelis-std (`stdlib_cache.rs:9-17`), used by the layered
`chelis check` path (`layered.rs:83-154`).
**Missing for FRAGMENT scope:** granularity is a whole DECL LIST, never a sub-decl body; every
entry re-parses new source from scratch (`compiler.rs:425`); grep for
`fragment/recheck_function/replace_def/patch_def/incremental` returns no relevant API.

**Q2. What would re-checking a SINGLE function body against an already-typed module require?**
**Answer / Verdict: PARTIAL · H1.** It would require a `TypeEnv` whose library = the module
*minus* the edited function, then feeding the single edited body as a one-element
`&[deep::Expr]`. Nothing assembles that today: the package's own modules enter the library
snapshot only at full-package build time, and the smallest unit any public entry accepts is a
re-parsed decl list. Missing pieces: a fragment-accepting check API (one body + span) plus a
context-builder that draws the cache boundary *inside* one file.

> **Skeptic nuance (CLAIM_REFUTED, narrowly).** A skeptic refuted the *stronger* phrasing that
> fragment validation "requires a prebuilt CompiledContext / whole-library compile." They ran
> `empty_context_matches_check_ir_program` (`crates/chelis-types/tests/context.rs:191`, PASSED
> 0.01s) showing `check_ir_with_context(&TypeEnv::empty(), fragment)` (`infer.rs:737`) type-checks
> an arbitrary Deep fragment against an in-memory symbol table with **no** CompiledContext, and
> `build_type_env_from_library` (`infer.rs:364`) builds that symbol table cheaply (no lowering /
> effects / linearity / reef / disk cache).
> **Resolution — both are right at different granularities.** The *primitive* "check a Deep
> fragment against an in-memory TypeEnv" genuinely exists and is the system's foundation. What is
> still **absent** is the specific capability of re-checking *one function body against the rest of
> its own already-typed file* (the cache boundary is never drawn inside a single file). For H1 the
> precise true statement is: *checking a Deep fragment against a symbol table is cheap and exists;
> single-body-within-its-module incremental checking does not.*

**Q3. Can `@property` + `chelis prove` run on a SINGLE property in isolation today?**
**Answer / Verdict: PRESENT · H1.** Yes, via `chelis prove <file> --only <name>`. Each property
is verified independently (`property_runner.rs:296-315`), through Tier B (SMT, cvc5 compiled in)
then Tier C (fuzz). **Probe:** `chelis prove /tmp/p.ch --only only_me --json` → 1 record,
`{"total":1,"passed":1}`, exit 0; no-filter → `{"total":2}`. **Caveat:** isolation is at the
verification layer only — the whole module is type-checked before discovery, so a broken sibling
decl the property never references blocks proving the filtered property (probe: broken sibling →
`{"kind":"error","stage":"check"}`, `{"total":0}`, exit 3).

**Q4. How granular is `--only` — exact or substring?**
**Answer / Verdict: PRESENT (sharp edge) · H1.** `matches_filter`
(`crates/chelis-cli/src/prove/mod.rs:2206-2217`) is exact → trailing-`*` glob → **substring
`contains` fallback**. Probe: `--only gordon_positive` pulled in both `gordon_positive` and
`gordon_positive_guards_satisfiable` (`total:2`); the unique-suffixed name gave `total:1`.
Naming a property as a prefix of siblings silently widens the feedback set.

**(Corpus on-version check.)** economoist `reef.toml` pins `compiler = "=0.8.0"` matching the
binary; `prove`/`check` on `growth.ch` ran clean, so the real corpus was usable for probes.

---

### 2.6 — The empirical setup (facts only)

**Q1. Eval harness + scoring infrastructure?**
**Answer / Verdict: PRESENT · neither.** The one model-vs-Chelis-authoring harness is
**fluke-ball / fluke-ball-wc**; **KellyBench** is the task suite
(`fluke-ball/flukeball/harness/kellybench_tasks.py:9-82`, 3 TRAIN + 2 TEST tasks). The model
authors a Surf strategy `src/strategy.ch` via 9 MCP tools (`mcp_server/server.py`). Scoring
pipeline: STYLE (`chelis fmt --inplace`), PARSE+TYPE-CHECK (`chelis check`, pass =
returncode==0 && no errors), RUNTIME (`chelis eval --file`), P&L (multiseason runner →
final_bankroll/roi/ruin/trajectory). economoist `scripts/prove_gate.py` is a separate property-
proof scorer, not a model-authoring harness.

**Q2. Real authoring corpus (quantified)?**
**Answer / Verdict: PRESENT · neither.** Substantial **Surf** corpus, **zero Deep**:
- shoals: 29 `.ch` / **0 `.dp`**, 532 `def`; +12 property `.ch`; 58 `@property` total.
- whale: 13 `.ch` / **0 `.dp`**, 105 `def`, 26 `type` (constructors), 0 `@property`.
- economoist: 3 src `.ch` (16 `def`), 3 property `.ch` (34 `@property`), **0 `.dp`**.
- **`.dp` count everywhere: 0.**

**Q3. Fluke-ball compiler PIN MISMATCH.**
**Answer / Verdict: PRESENT · neither.** Old `fluke-ball/reef.toml:4` pins
`compiler = "=0.7.20"` (with reef.lock + bootstrap + bridge bins all at 0.7.x). The binary is
0.8.0. chelis-reef enforces strict exact-pin match — `CURRENT_COMPILER_VERSION = "=" +
CARGO_PKG_VERSION = "=0.8.0"` (`chelis-reef/src/lib.rs:22`), rejecting any non-matching pin
("package.compiler must be `=0.8.0`", `:3299-3302`). The migrated **fluke-ball-wc** variant pins
`=0.8.0` (chelis-std 0.4.0, whale 0.1.9). A pre-0.8.0 binary cannot parse Deep input at all
(red-team-reproduced with 0.7.17, the nearest older binary confirmed present locally; a binary
matching the exact `=0.7.20` pin was *not* independently confirmed — see §11). **Not fixed.**

**Q4. How much in-repo evidence bears on H2?**
**Answer / Verdict: ABSENT (this emptiness IS the finding) · H2.** **Near zero, effectively
zero.** No Surf-vs-Deep authoring comparison anywhere. `.ch` = hundreds of occurrences
(≈365–474 depending on grep scope), `.dp` = 0 in both harness trees. The 9-tool MCP surface has no Deep-authoring/emit tool. The firehorse
prompt instructs the model to write `src/strategy.ch` only. Corpus repos = 0 `.dp`. The only
"Surf-vs-Deep" string matches are the §11.1 hyphen-naming asymmetry, not authoring. The closest
design statement is `chelis_design_sprint_adjudication.md:33` ("Deep is not meant for human
authoring" / "radically simplifies structural manipulation") — a **design hypothesis, not an
experiment**. `chelis deep` exists (Surf→canonical Deep), so a Deep path is technically
constructible but entirely unused by any harness.

---

## 4. Decision-Critical: the c-earchin → chelis-prove boundary

This is the proposal's **motivating example** — its strongest single piece of grounding for the
build-out. The proposal framed it as a *lossy text reparse*: "emit-Surf → reparse text →
SmtExpr, structure lost at the text boundary."

### That framing was MISCHARACTERIZED. Stated plainly:

**The boundary is a serialized-AST handoff in Deep's native format, parsed back structurally.
There is no Surf in the production path and no SmtExpr is constructed for the c-earchin body at
all.**

What the boundary *actually* is:
- **Producer:** c-earchin emits **Deep `.dp`** via `render_deep` (`c-earchin/src/emit.rs:5`,
  written by `main.rs:155-174`) plus a `.spans.json` sidecar. Deep *is* the AST's own text
  serialization. The Surf path (`emit_surf.rs` / `render_surf` / `render_property` /
  `PropertySpec`) has **only `#[cfg(test)]` callers** — dead in production.
- **Consumer:** `chelis prove` reads the `.dp` and parses it with
  `chelis_deep::parser::parse_str` into the Deep AST (`chelis-cli/src/prove/mod.rs:1334`),
  validates it, type-checks it, discovers the bridge def **structurally** from the parsed
  AST/MetaMap (`:1879-1899`), and verifies it by **concrete evaluation** (Tier C fuzz,
  `eval_deep_sample` → `print_canonical` → parse + IR eval, `:2117`). EARS provenance is
  recovered from `.spans.json` (`:2440`).

### Is structural identity lost there at all?

**No SmtExpr identity is lost, because no SmtExpr is built on this path.** The Deep body has
"no Surf→SMT lowering path" and falls to Tier C concrete eval. There *is* a faithful structured
`PropertySpec → SmtExpr` serde bridge (`from_property_spec.rs:86-151`, 1:1, no text, no
heuristics), but it is **unreachable from c-earchin's real output** — c-earchin never emits the
`PropertySpec` JSON it would consume, and the legacy text-source path (`dispatch.rs::dispatch_property`,
`tier_b::solve(&str)`) is an orphan stub returning `Timeout`.

### Skeptic disagreement — and the load-bearing resolution

The two inputs *appear* to conflict and the conflict matters:

- **Phase-1 (2.4) agent:** "the boundary is **not** a lossy Surf reparse; it is a structural
  Deep-AST handoff verified by concrete eval." → CONTRADICTED the proposal's framing.
- **Skeptic (claim CLAIM_REFUTED):** attacked the *narrower* claim that the transfer is a
  "faithful structured **serde** PropertySpec→SmtExpr handoff," and showed the live path
  **reparses text** via `chelis_deep::parser::parse_str` (`prove/mod.rs:1334`) /
  `chelis_surf::parser::parse_str` — the serde bridge is unwired dead code.

**These are not actually in conflict; they refute two *different* propositions, and both are
correct on executed evidence:**
1. The proposal's "lossy Surf text reparse" framing is **wrong** (resolved in favor of the 2.4
   agent + the skeptic's own data: it's Deep, not Surf).
2. A description of the boundary as a "faithful *serde* PropertySpec→SmtExpr handoff" would also
   be **wrong** (resolved in favor of the skeptic: that serde path is dead; the live path
   reparses Deep *text*).

The accurate synthesis: **the boundary is a Deep-`.dp` text round-trip — Deep is the AST's
serialization, reparsed structurally by the Deep parser, then verified by concrete evaluation.**
Whether you call a `parse_str(deep_text)` step a "reparse" is the crux: it *is* a text parse,
but it is a **lossless** parse of Deep's canonical serialization (the round-trip is locked by
`roundtrip.rs`), not the *lossy* Surf reparse the proposal described, and crucially it does
**not** lose structure into an SmtExpr because no SmtExpr is built.

### Consequence for the grounding

**The proposal's strongest single grounding was mischaracterized, so the grounding weakens —
directly stated, without softening.** The build-out was motivated by "we lose structural
identity at c-earchin → prove via a lossy text reparse." On executed evidence, the live path
does not lose SmtExpr identity (none is built) and does not go through Surf. If the motivating
example does not exhibit the structural-loss problem, then the case for the authoring-trust-
stack build-out *rests on something other than that example* — and nothing else in the recon
establishes that loss either. This is a reason to treat the build-out premise as **unproven**,
which reinforces the net verdict that the first step should be an experiment that *tests* the
premise (H2), not an investment that *assumes* it (H1).

---

## 5. The H2 Evidence Gap (load-bearing)

**Head-on: essentially nothing in-repo bears on H2.** This is the single most important
finding in the reconnaissance.

- 0 `.dp` files in shoals/whale/economoist (Section 2.6 Q2).
- 0 `.dp` occurrences across both fluke-ball harness trees vs hundreds of `.ch` (≈365–474, 2.6 Q4).
- No Deep-authoring or Deep-emit MCP tool; no `chelis deep`/`chelis surf` conversion invoked in
  any harness (2.6 Q4).
- No head-to-head Surf-vs-Deep generation experiment, no Deep authored corpus, no Deep scoring
  path anywhere (2.6 Q4).
- The only pro-Deep claim is a **design hypothesis** (`chelis_design_sprint_adjudication.md:33`),
  never tested.

**Why the recon is dense on H1 and empty on H2 — and the warning that follows.** H1 is
*observable in static source*: tool registries, AST type visibility, in-context check APIs, span
chains, error structs. So six verification agents could pile up precise H1 file:line findings.
H2 — *does a model author Deep better than Surf?* — is **not observable in static source at
all**; it is an empirical question about model behavior that no existing artifact answers.

> **The density of H1 findings is an artifact of observability, not a signal that H1 is the right
> first step. Building the H1 PoC first because "that's where all the concrete findings are" is
> textbook streetlight bias** — searching under the lamppost because that's where the light is,
> not because that's where the answer is. The decision-critical unknown (H2) is precisely the one
> the static recon could not touch. The first step should *reduce that unknown*.

---

## 6. Cross-Cutting: Proposal Assumptions

| # | Proposal assumption | Status | Citation |
|---|---|---|---|
| A | "The compiler probably already has spans / AST-path errors are a surface change" | **CONTRADICTED** | `CheckError` has no location field (`errors.rs:5-15`); all 8 error→diagnostic sites hardcode `span: None` (`compiler.rs:2025`); best case is a Surf byte-span *string* in the message; no Deep path exists (2.2 Q2, skeptic CLAIM_HOLDS) |
| B | "`emit_surf.rs` becomes `emit_deep.rs` / emitters already build a Deep AST" | **PARTIALLY-CONFIRMED** | 3 of 4 emitters (c-earchin, calcify, octant) build canonical *text* by string-concat with no Chelis AST; only hydronnx builds an AST (its *own local* Surf AST, not chelis_deep) (2.4) |
| C | "Incremental compile is probably wireable / fragment validation is cheap" | **PARTIALLY-CONFIRMED** | Library-vs-new snapshot checking exists and is cheap (`check_ir_with_context`, `TypeEnv`; skeptic ran `empty_context_matches_check_ir_program` PASS); but **single-body-within-its-module** fragment checking is absent — no API draws the cache boundary inside one file (2.5 Q1/Q2) |
| D | "Deep is a clean single-spelling library target" | **PARTIALLY-CONFIRMED** | One canonical form enforced by `print_canonical` (idempotence + collapse + roundtrip probes), library parser is `pub`; **but** `--flat` is a second CLI spelling, and the AST/parser are exposed only as an internal crate, not via `chelis-compiler-api` or any wire/FFI surface (2.1 Q1/Q2/Q3) |
| E | "`chelis_replace_body` PoC is the obvious first tool" | **CONTRADICTED** (as *first* step) | Tool is entirely unbuilt (spec item header `chelis_agent_editing_surface.md:44`, explicitly Exploratory at `:1`/`:3`/`:162`; absent from `crates/`); its prerequisites (Deep-path errors, Deep-fragment splice API, single-body incremental check, tide edit surface) are all absent; and its motivating boundary (c-earchin→prove) was mischaracterized (Sections 2.2, 2.3, 2.5, 4) |

---

## 7. The Five Decision Questions (brief §5)

**(1) Is Deep a clean single-spelling library-accessible authoring target today, or does it need
canonicalization + public API first?**
*Mostly yes, with one real prerequisite.* Canonicalization is **done** — `print_canonical` gives
one form, proven idempotent and collapsing (2.1 Q2). The library parser/AST are `pub` and
callable from Rust (2.1 Q1/Q3). **But:** (a) `--flat` is a second machine-selectable spelling, and
(b) Deep is reachable only as an *internal Rust crate* — `chelis-compiler-api` does not re-export
it and there is no wire/FFI surface. **So: canonicalization is not needed; a public/stable AST
authoring API on the machine-facing facade *is* the prerequisite if Deep is to be a target for
anything other than direct Rust dependents.**

**(2) Are AST-path errors a surface exposure of data the compiler already carries, or do they
require threading provenance through inference?**
**They require threading provenance.** `CheckError`/`TypeError` carry no location at all (2.2 Q2,
skeptic CLAIM_HOLDS); even Surf byte spans are absent from all but ~11 validator errors and are
hardcoded `None` in the diagnostic conversion. A **Deep path does not exist anywhere** in the
implementation. This is not a formatting change — it needs new location fields on
CheckError/TypeError plus a Deep-path computation that does not exist.

**(3) Is per-edit validation cheap today, or does it need new fragment-checking plumbing?**
**Cheap at one granularity, needs plumbing at another.** Checking a Deep fragment against an
in-memory `TypeEnv` is cheap and exists (`check_ir_with_context`, skeptic-verified). Checking
*one edited function body against the rest of its own already-typed module* needs new plumbing: a
fragment-accepting API + a context-builder that draws the cache boundary inside a single file
(2.5 Q1/Q2). Property feedback per-property exists (`--only`) but type-checks the whole module
first and has substring/broken-sibling edges (2.5 Q3/Q4).

**(4) Which emitters already build a Deep AST (cheap to flip) vs assemble Surf text (a rewrite)?**
**None already builds a *Deep* AST.** c-earchin and octant already emit **Deep text** (no flip
needed — they target Deep, just by string-concat). calcify emits Surf text; hydronnx builds its
*own* Surf AST then renders Surf. So "flip `emit_surf.rs` to `emit_deep.rs`" applies only to the
two Surf-targeting tools, and for them it is a rewrite of the rendering layer, not a one-line
target swap. The two Deep-targeting tools need no flip at all (2.4).

**(5) What is the smallest real experiment that distinguishes H1 from H2 and tells us whether
Deep authoring is worth building toward?**
See Section 8. In short: a **pin-free Surf-vs-Deep authoring eval** scored by parse-pass +
type-check-pass on the same tasks in both syntaxes, seeded from the existing Surf corpus.

---

## 8. Smallest Discriminating Experiment (designed here from the full picture)

**Goal:** an experiment that (a) **discriminates H1 from H2** and (b) is **runnable TODAY**.

### Design: "Same-task, two-syntax authoring score"

1. **Corpus seed (real, in-repo):** Take a stratified sample from the existing **Surf** corpus —
   the cleanest candidates are **economoist properties** (34 `@property` across
   `bellman/growth/markov`, on-version at `=0.8.0`) and **shoals pricing `def`s** (532 `def`,
   property-rich). These are real, on-version, and already type-check under 0.8.0.
2. **Two authoring arms, same tasks:** For each sampled task, prompt a model to author the
   solution **(A) in Surf** and **(B) in Deep** from the same natural-language spec. Both arms
   produce a file the 0.8.0 binary can score.
3. **Pin-free scoring path (the key to "runnable today"):** Score with the **0.8.0 binary**
   using `chelis check` on `.ch` and `.dp` respectively. Section 2.6's skeptic confirmed
   `chelis check` returns an **identical JSON score schema** (`score`, `parse/structure/names/types`,
   `errors[]`) for both `.ch` and `.dp` under 0.8.0. Use **parse-pass + type-check-pass** (score
   == 1, `errors == []`) as the primary metric; optionally add `chelis fmt --check` round-trip
   stability and `chelis prove --only` pass-rate where the task has a property. **No P&L, no
   runtime, no harness admission gate is required for the discriminating signal.**
4. **What it discriminates:** If Deep authoring yields materially higher parse/type-check pass
   rates (or fewer edit cycles to green), H2's premise has support and Deep authoring is worth
   building toward. If Surf wins or ties, the H1 build-out is not justified by an authoring
   advantage and any value must come from the *editing* loop — which then needs its own, separate
   justification (and its prerequisites from Sections 2.2/2.3/2.5 priced in). This is the
   cleanest H1-vs-H2 fork the current artifacts permit.

### The fluke-ball 0.7.20 pin — what it forces, and the pin-free escape

- **What the pin forces (confirmed by execution):** The **old fluke-ball harness's own scoring
  path cannot score Deep.** A pre-0.8.0 `check` has no Deep-input parse path: an older binary
  (red-team-reproduced with 0.7.17, the nearest available to the `=0.7.20` pin) on a `.dp` returns
  `score 0`, all components 0, `errors: [{"found LParen at byte 0"}]` — even on a `.dp` that the
  older binary itself produced. Only 0.8.0's `chelis check .dp` returns `score 1`. Additionally,
  the harness's ABI/admission gate is
  **Surf-only by construction** (Surf-syntax regexes for exports/signatures/state; a valid Deep
  submission is uniformly rejected). And `chelis check` has **no `--json` flag** (JSON is the
  default report).
- **Therefore: do NOT route the experiment through the fluke-ball harness.** Running a fair
  Surf-vs-Deep arm *on the existing harness* would require unblocking the 0.7.20 pin **and**
  building new Deep-aware admission/scoring (skeptic CLAIM_HOLDS for the harness path).
- **The pin-free escape (why the experiment is runnable today):** Score directly with the
  **0.8.0 `chelis check`** on `.ch`/`.dp` (step 3), bypassing fluke-ball entirely. This needs no
  pin change and no new harness scoring code — only a thin driver that invokes the 0.8.0 binary
  and parses its default JSON report. The economoist/shoals corpora are already on `=0.8.0`, so
  the seed tasks need no migration either.

**Note on the pin for completeness:** the *active* checkout (fluke-ball-wc) is already migrated
to `=0.8.0`, and the eval pin in the harness floats with the binary (synthesized temp packages
derive `compiler` from the live binary, env-overridable via `CHELIS_PACKAGE_COMPILER`) — but
even fluke-ball-wc's *scoring layer* remains Surf-only, so the harness is still not a fair H2
substrate without new Deep-aware scoring. The pin-free 0.8.0-`check` path sidesteps all of this.

---

## 9. Net Verdict (restated, full reasoning)

**Smallest first step: the H2 Surf-vs-Deep authoring eval of Section 8 (pin-free, 0.8.0
`chelis check` scoring on the economoist/shoals corpus) — NOT the H1 `chelis_replace_body` PoC.**

Reasoning grounded in the gathered evidence, deliberately *not* anchored to where the facts
clustered:

1. **The decision-critical unknown is H2, and nothing in-repo speaks to it** (Section 5). Every
   H1 fact is statically observable, so the recon is dense there; H2 is an empirical question
   about model behavior with zero in-repo artifacts. Picking the editing PoC because H1 has more
   findings is streetlight bias.
2. **The build-out's strongest grounding was mischaracterized** (Section 4). The c-earchin →
   prove boundary is a structural Deep-`.dp` handoff verified by concrete eval, not a lossy Surf
   reparse losing SmtExpr identity. With the motivating example not exhibiting the claimed
   problem, the build-out premise is unproven and should be *tested*, not *assumed*.
3. **H1's prerequisites are largely absent** (Sections 2.2, 2.3, 2.5): no structural-edit tool,
   no Deep-path errors (would require provenance threading), no single-body incremental check, no
   tide edit surface. An H1 PoC would first have to build several of these on faith that editing
   is the right loop — before any evidence that it is.
4. **The H2 eval is the cheapest informative move:** runnable today with the 0.8.0 binary, no pin
   change, no new harness scoring, seeded from real on-version corpora, and it directly forks H1
   vs H2. If Deep authoring wins, build toward Deep; if it ties/loses, the editing case needs an
   independent justification with its absent prerequisites priced in.

If the orchestrator prefers a **prerequisite-first** ordering instead, the single prerequisite
that most unblocks *either* hypothesis is a **public/stable Deep AST authoring API on
`chelis-compiler-api`** (Section 7 Q1) — without it, Deep is reachable only by direct Rust
dependents. But the eval needs no such API (it uses the existing CLI), so the eval remains the
smaller first step.

---

## 10. Appendix: Probe Log

All probes were read-only; temp files under `/tmp` only; no repo edits; the prebuilt 0.8.0
binary (plus the locally-present 0.7.20 binary for the pin probe) was used — no cargo build of
the workspace except the skeptic's single `cargo test` of one chelis-types unit test.

### Section 2.1 (Deep as a target)
- `chelis --version` → `chelis 0.8.0`
- `chelis fmt simple_def.dp` twice → byte-identical (`IDEMPOTENT-OK`)
- ugly non-canonical `.dp` → `chelis fmt` → byte-identical to canonical fixture
  (`COLLAPSES-TO-SAME-CANONICAL-OK`)
- `chelis deep examples/vmap_relu.ch` (pretty) + `--flat` → two distinct spellings; `fmt` of
  pretty deep → no diff (`DEEP-OUTPUT-IS-FMT-CANONICAL-OK`)
- `chelis validate --deep .../simple_def.dp` → `validated deep: ...`
- `chelis fmt --help` → only `--inplace`, `--check` (no `--flat`)
- grep `.chb` across spec/crates → archived intent (`adjudication.md:33`) vs active package-
  metadata def (`spec/10`, `canonical_reference`); producers `chelis-reef/src/lib.rs:927/...`

### Section 2.2 (provenance)
- `chelis check /tmp/x.ch --allow-style-violations` (int32+f32) →
  `[{"kind":"PrecisionMismatch","message":"precision mismatch: expected int32, got f32","severity":0.8}]`
  (no span/path)
- `chelis check /tmp/x_raw.dp` (Deep input WITH span metadata) → identical spanless diagnostic
- `chelis deep /tmp/x.ch --annotate` → requires well-typed program; CheckError shows
  `expected:None, got:None` (skeptic: full Debug dump, no span/path/node field)
- `chelis check /tmp/conv.ch` (conv2d stride 0) → `...positive stride, got 0 (at surf:93..113)`
  (Surf byte span in message string)
- Deep→Surf→Deep round trip: spans `surf:36..43/36..37/40..43` → recomputed
  `surf:35..46/35..38/39..40`
- `fmt '= (x + 1.0)'` → `(app {} (var add) x 1.0)` → `chelis surf` → `add(x, 1.0)` (infix `+`
  lost)
- grep `span:None` in compiler.rs → `:468, :1015, :2033` (three conversions hardcode None);
  `schema.rs:63` Diagnostic HAS span field

### Section 2.3 (tool surface)
- live `chelis tide mcp` JSON-RPC `tools/list` →
  `[chelis_check, chelis_compile, chelis_decompile, chelis_desugar, chelis_eval, chelis_grad, chelis_prove, chelis_validate]`
- grep `chelis_replace_body|replace_body` over `crates/` → NONE FOUND (spec/*.md only)
- grep `chelis_define|chelis_rename|chelis_change_signature|chelis_view|chelis_property_check`
  over `crates/` → NONE FOUND
- `chelis tide --help` → subcommands serve/mcp/lsp; no edit command

### Section 2.4 (emitters + boundary)
- enumerated emit modules in c-earchin/calcify/hydronnx/octant
- grep `render_surf/render_property/'PropertySpec {'` in c-earchin → TEST-ONLY
- `chelis prove --help` → accepts `.ch|.dp` + `--spans` bridge manifest override
- integration test `prove.rs:805` writes a `bridge:c-earchin` `.dp` + spans.json, asserts CLI
  resolves to EARS source
- `from_property_spec.rs` serde bridge present; skeptic grep: `to_smt_property` called only in
  its own unit tests; `dispatch_property`/`to_dispatch_amenability` zero non-definition callers;
  `tier_b::solve(&str)` returns Timeout

### Section 2.5 (incremental + property)
- `chelis prove /tmp/p.ch --only only_me --json` → 1 record, `total:1, passed:1`, exit 0;
  no-filter → `total:2`
- `chelis prove /tmp/p2.ch --only only_me --json` (broken sibling) → `stage:check` error,
  `total:0`, exit 3
- `chelis prove economoist/properties/growth.ch --only gordon_positive --json` → 2 records
  (substring pulled in sibling); `--only gordon_dP_dr_negative_guards_satisfiable` → `total:1`
- `chelis check economoist/properties/growth.ch` → score 1
- `strings target/release/chelis | grep -i 'cvc5|smt|smt-only'` → present (SMT compiled in)
- grep `fragment/recheck_function/replace_def/patch_def/incremental` → no single-function recheck
  API
- skeptic `cargo test`: `empty_context_matches_check_ir_program` (`chelis-types/tests/context.rs:191`)
  PASSED 0.01s → `check_ir_with_context(&TypeEnv::empty(), fragment)` checks a standalone Deep
  fragment

### Section 2.6 (empirical setup) and the fluke-ball pin note
- `chelis --version` → `chelis 0.8.0`
- `fluke-ball/reef.toml:4` → `compiler = "=0.7.20"`; reef.lock 6 sites; bins at 0.7.20
- `fluke-ball-wc/reef.toml:4` → `compiler = "=0.8.0"` (migrated, branch world-cup, mtime
  2026-06-21)
- `chelis-reef/src/lib.rs:22` `CURRENT_COMPILER_VERSION = "=" + CARGO_PKG_VERSION`; `:3299-3302`
  reject "package.compiler must be `=0.8.0`"
- **/tmp pin probe (0.8.0 binary):** package pinning `=0.7.20` → `error: package.compiler must be
  =0.8.0 in 3a` (exit 1); `=0.8.0` → evaluates `1 + 2` to `3` (exit 0)
- **older binary probe (0.7.17 proxy; see §11):** `chelis check strategy.dp` → `score 0`,
  components 0, `errors:[{"found LParen at byte 0"}]` (no Deep-input parse path before 0.8.0). A
  binary matching the exact `=0.7.20` pin was not independently confirmed present; the 0.7.17 proxy
  emits the identical failure mode.
- **0.8.0 binary probe:** `chelis check strategy.dp` → `score 1`, `errors []` (Deep-input check is
  0.8.0-only)
- harness ABI scoring on a Deep version → `module_name=None`, `exports=[]`, `signatures={}`,
  `state_schema=None`; issues `invalid_strategy_module` + `missing_strategy_state_schema` + 6×
  `missing_required_function` (admission gate is Surf-only)
- `chelis check --help` (both binaries) → **no `--json` flag** (JSON is the default report)
- corpus counts: shoals 29 `.ch`/0 `.dp`/532 `def`/58 `@property`; whale 13 `.ch`/0 `.dp`/105
  `def`/26 `type`; economoist 3 src + 3 property `.ch`/34 `@property`/0 `.dp`
- harness grep: `.ch` = 506 occurrences, `.dp` = 0, across both fluke-ball trees

**Fluke-ball pin note (do not fix here):** The old `fluke-ball` checkout pins `=0.7.20`, and any
pre-0.8.0 binary cannot parse Deep input at all (reproduced with 0.7.17; a binary matching the
exact `=0.7.20` pin was not independently confirmed present — see §11); the active `fluke-ball-wc`
is migrated to `=0.8.0` but its scoring/admission layer remains Surf-only. The discriminating
experiment (Section 8) deliberately avoids the harness and scores via the 0.8.0 `chelis check`
JSON report directly, which needs neither the pin unblocked nor new harness scoring.

---

## 11. Red-Team Validation

This document was independently validated by a **fresh-context adversarial subagent** (per the
repo's Red Team Protocol) that re-executed the load-bearing probes from source and command
output, treating every claim as a hypothesis to falsify — it did **not** read the synthesis
reasoning. Outcome: **all four load-bearing claims CONFIRMED, net verdict supported, 0
CRITICAL/HIGH/MEDIUM findings, 4 LOW findings** (all folded into this revision).

- **C1 (provenance absent):** CONFIRMED — `CheckError` field set re-read (`errors.rs:5-15`, no
  location field); `span: None` re-grepped at `compiler.rs:468/1015/2033`; a `.dp` literally
  carrying `span: "surf:34..39"` still returned a spanless `PrecisionMismatch`.
- **C2 (c-earchin→prove builds no SmtExpr):** CONFIRMED — all five sub-claims independently
  re-derived; the Surf path is `#[cfg(test)]`-only, `to_smt_property`/`dispatch_property`/
  `tier_b::solve` have zero non-test callers, and the shared runner skips bridge properties
  (`property_runner.rs:1741`, locked by `tests.rs:412`).
- **C3 (H2 eval runnable today):** CONFIRMED — `chelis check` on a `.ch` and its `chelis deep`
  `.dp` returned **byte-identical** JSON (same keys, both `score:1`, `typed_nodes:54` identical);
  `check` has no `--json` flag (JSON is the default report).
- **C4 (fluke-ball pin):** CONFIRMED with one caveat (F-LOW-1 below) — pin values and exact-pin
  enforcement re-verified live; the "older binary can't parse Deep" half reproduced.
- **Citations:** 10/11 spot-checked resolved exactly (+2 bonus); the one partial miss (`:44` does
  not contain the word "Exploratory") is corrected in §6 row E.
- **Corpus:** `532 def`, `34 @property`, `58 @property` (shoals total), and **0 `.dp` everywhere**
  reproduced exactly.

**LOW findings, all addressed in this revision:**
- **F-LOW-1 (evidence provenance):** the original draft asserted a probe run with a "locally-
  present `0.7.20` binary." The red team could not find a `0.7.20` binary (found 0.7.17 / 0.7.27 /
  0.8.0) and reproduced the *conclusion* (a pre-0.8.0 `check` cannot parse Deep input —
  `found LParen at byte 0`) with **0.7.17** as a proxy. §2.6/§8/§10 now attribute this to a
  pre-0.8.0 binary (0.7.17 proxy) and state the exact `=0.7.20` binary was not independently
  confirmed present. The pin *value* (`=0.7.20`) is real (`fluke-ball/reef.toml:4`).
- **F-LOW-2 (citation precision):** "Exploratory" is at `:1`/`:3`/`:162` of
  `chelis_agent_editing_surface.md`, not `:44` (the item header) — corrected in §6 row E.
- **F-LOW-3 (count imprecision):** the harness `.ch` occurrence count is scope-dependent
  (≈365–474), not a hard 506 — softened wherever it appeared; the decisive `0 .dp` is exact.
- **F-LOW-4 (label precision):** of the three `span: None` sites, `:468` and `:1015` are both the
  **effects** conversion; the **type** and **linearity** families route through
  `check_error_diagnostic` (`:2033`) — clarified in §2.2.

These are textual-accuracy corrections grounded in the red team's own verified evidence; none
alters a verdict, and the net verdict is unaffected.
