# Chelis Ecosystem Naming-Convention Snapshot

A descriptive study of naming conventions currently in use across the Chelis
ecosystem, produced as input to a future remediation pass. This file describes
what exists; it does not recommend changes.

---

## 1. Scope of the survey

### Real ecosystem repos walked

Real independent peer repos under `github.com/Chelis-Lang/`:

- `chelis/` — primary monorepo (compiler, runtime, stdlib reef package, bindings).
- `nautilus/` — numerical / stats / optimization library.
- `coral/` — typed-dataframes library.
- `octant/` — LaTeX-to-Chelis bridge (Rust binary peer).
- `shoals/` — quantitative-finance library.

### Excluded — git worktrees of `chelis`

`git worktree list` confirms these are working-tree views of `chelis/.git`,
not independent repos. Do not double-count:
`chelis-proof`, `chelis-tooling`, `chelis-research`, `chelis-phase-j-baseline`,
`chelis-coral`, plus the dozens of agent worktrees under `chelis/.claude/worktrees/`.

### Excluded — out of scope

`polaris/` is an unrelated Elixir/Optax library (confirmed by `mix.exs`), not part
of the Chelis ecosystem.

### Sample size

5,290 `.ch` files (excluding `examples/illustrative/`); 30,109 top-level `def`
occurrences; 144 distinct module declarations; 1,396 `type` declarations;
10,818 ADT constructors; 21 Rust crates (20 in chelis + 1 in octant); 5
`reef.toml` packages.

---

## 2. Constraints the language imposes (HARD)

These are enforced by the parser, resolver, or backend. Any future remediation
must live inside them.

| Constraint | Source |
|---|---|
| Surf identifiers split on first-character case at the lexer: lowercase-leading → `Ident` (values, functions, params, dims); uppercase-leading → `TypeIdent` (types, ADT constructors, module-ladder components). The parser uses this to disambiguate. | `crates/chelis-surf/src/lexer.rs:398-403` |
| Surf identifier charset: `[A-Za-z_][A-Za-z0-9_]*` — ASCII alphanumeric + underscore only. **No hyphens.** | `crates/chelis-surf/src/lexer.rs:325-368` |
| Module-path lookup: `module Foo.Bar` → `foo/bar.ch`. The desugarer lowercases each PascalCase component and joins with `/`. So `.ch` filenames must be all-lowercase / snake_case to be importable. | `spec/02-surf-syntax.md:79-91`, line 79: *"Module names are PascalCase. No nesting within a file."* |
| 23 reserved Surf keywords, all lowercase: `def sig type dim macro match with fn module import if then else grad vmap jit realize copy tensor cast export par true false`. Plus 7 reserved-for-Phase-2: `effect handler perform resume borrow where do`. | `crates/chelis-surf/src/lexer.rs:372-395` |
| Deep tag vocabulary: closed at ~60 tags. Single-token tags are bare lowercase (`app`, `var`, `lit`, `def`, `sig`, `module`, `import`); compound tags use **hyphenated** form (`t-fn`, `t-prim`, `t-tensor`, `t-var`, `pat-var`, `pat-lit`, `pat-ctor`, `d-name`, `d-var`, `d-lit`). Hardcoded in `crates/chelis-deep/src/validate.rs` and emitted by `crates/chelis-deep/src/printer.rs`. | `spec/03-deep-syntax.md:262+` |
| Deep `Symbol` lexer accepts hyphens: `[A-Za-z_][A-Za-z0-9_-]*`. This is intentional structure — the wider grammar exists so the closed compound-tag vocabulary can use hyphens as the compound separator. User-defined Deep symbols originate from Surf desugaring and inherit Surf's no-hyphen rule by construction; the asymmetry between Surf (no hyphens) and Deep (hyphens allowed) is a design choice, not a defect. | `crates/chelis-deep/src/lexer.rs:179-181` |
| `chelis fmt` does not rewrite identifier case. Mixed styles survive round-trips. | `crates/chelis-surf/src/format.rs` |
| C / HIP backends emit user names as-is; no symbol mangling. | `crates/chelis-backend-c/src/emit.rs:29-31` |

---

## 3. Snapshot — filesystem and manifest layer

### Top-level dirs (the checkout itself)

Uniform kebab-case across all five real ecosystem repos: `chelis`, `nautilus`,
`coral`, `octant`, `shoals`. Single-word names stay single-word; multi-word
names use hyphens. No exceptions.

### Rust crate dirs and `Cargo.toml` `package.name`

20 crates inside `chelis/crates/`, all kebab-case:

```
chelis-backend-c, chelis-backend-hip, chelis-backend-metal, chelis-cli,
chelis-compiler-api, chelis-cove, chelis-deep, chelis-e2e, chelis-effects,
chelis-ir, chelis-lsp, chelis-macros, chelis-python, chelis-reef,
chelis-runtime, chelis-shell, chelis-surf, chelis-tide, chelis-types,
chelis-validate
```

Plus `tree-sitter-chelis` at the workspace root (kebab). Plus the standalone
peer crate `octant` (kebab, single-word, no `chelis-` prefix — but it lives
in its own repo, so the prefix question only applies inside the monorepo).

`crates/chelis-python/Cargo.toml` has `package.name = "chelis-python"` and a
lib name of `chelis_python` (Rust standard hyphen→underscore coercion).

### `.rs` module files

Snake_case, uniform Rust standard: `parser.rs`, `lexer.rs`, `span_merge.rs`,
`optimize.rs`, `pipeline.rs`, `dag.rs`, `emit.rs`.

**Outlier — chelis core integration tests:** four files in
`crates/chelis-reef/tests/` carry the `phaseA` chunk inside an otherwise
snake_case filename:
- `phaseA_item6_from_github.rs`
- `phaseA_item7_bootstrap.rs`
- `phaseA_item8_autofetch_build.rs`
- `phaseA_item9_lockfile_origin.rs`

Every other `.rs` file in every walked Rust tree is pure snake. The `phaseA`
embedding here matches the project-wide phase-identifier convention (see §6
below) but produces mixed-case Rust filenames.

### `.ch` (Surf) source files

All-lowercase, snake_case where the basename has multiple words. Example
distribution: `linalg.ch`, `pricing.ch`, `pattern_matching.ch`,
`mlp.ch`, `cubic_hermite.ch`. No PascalCase, kebab, or mixed-case `.ch`
filenames found across any walked checkout — consistent with the language's
hard constraint that `module Foo.Bar` resolves to `foo/bar.ch`.

### `.dp` (Deep) source files

Confirmed extension: `.dp`, not `.deep`. Found primarily as test fixtures:
`crates/chelis-deep/tests/fixtures/` (`simple_def.dp`, `hello_tensor.dp`,
`metadata.dp`, `pattern_match.dp`, `pipeline.dp`) and `octant/references/`
(`call_price.dp`, `call_price_wrapped.dp`). All snake_case. No standalone
authored `.dp` files outside fixtures — Deep is generated, not authored.

### Reef package source dirs

The chelis-std reef package lives at `chelis/packages/chelis-std/` (not under
`crates/`). The monorepo therefore has two parallel package surfaces:
`crates/` (Rust workspace members) and `packages/` (reef packages — currently
just `chelis-std`).

### Python files

Snake_case where Chelis-authored. Walked locations:

- `chelis/scripts/`: `bench_phase_j.py`, `build_wrapped_spans_sidecar.py`, `bump_compiler_pins.py`, `nautilus_local_gate.py`, `test_bump_compiler_pins.py`.
- `chelis/bindings/python/`: `chelis/__init__.py`, `tests/manual_phase3b.py`, `tests/manual_phase3bii.py`.
- `chelis/benchmarks/pytorch/`: `linreg.py`, `mnist.py`, `transformer_block.py`.
- `chelis/assets/mascot/`: `generate_chev.py`.
- `nautilus/scripts/`: `bench_eval_startup.py`, `bench_vs_scipy.py`, `chelis_toolchain.py`, `extract_stability.py`, `gen_goldens.py`, `validate_book_examples.py`.
- `coral/scripts/`: `chelis_toolchain.py`, `repro_multimodule_bare_build.py`, `run_skill_checks.py`, `run_static_checks.py`, `validate_book_examples.py`.
- `octant/scripts/`: `corpus_smoke.py`.
- `shoals/scripts/`: `run_local_gate.py`.

All snake_case, no exceptions found across walked Chelis-authored Python.

### Shell scripts

Project policy is "never shell" (`CLAUDE.md` Scripting Language Policy).
**One outlier:** `nautilus/scripts/install_chelis_std.sh`. This violates the
documented policy.

### Spec & doc filenames — three coexisting conventions

`chelis/spec/` top-level (13 files) — **numeric-prefix + kebab**:
`00-context.md`, `01-nomenclature.md`, `02-surf-syntax.md`, `03-deep-syntax.md`,
`04-type-system.md`, `05-risc-primitives.md`, `06-transformations.md`,
`07-concurrency.md`, `08-backends.md`, `09-tide.md`, `10-serialization.md`,
`11-ffi.md`, `12-roadmap.md`.

`chelis/spec/design/` (~45 files) — mixed **snake_case** with **kebab-case**
within the same directory:
- snake_case (the majority): `phase1a_kernel_codegen.md`, `phase1b_fusion.md`,
  `phase1c_memory_planning.md`, `phase1d_flattening.md`, `phase1e_benchmarks.md`,
  `phase1f_executable_grammar.md`, `phase1_symbolic_dims.md`,
  `phase3j_pre_release.md`, `phase3m_rust_runtime_rewrite.md`,
  `phase3n_octant.md`, `phase5_host_scalar_ad.md`, `chelis_*` (~20 files)
- kebab-case (concentrated in the upstream-bugs subset):
  `grad-eval-host-runtime.md`, `host-emit-hashmap-iteration-nondeterminism.md`,
  `host-eval-perf-mc-rigor.md`, `producer-string-sanitization.md`,
  `reef-multi-source-roots.md`.

`chelis/docs/` — **snake_case**: `lin_rca_report.md`, `perf_baseline.md`,
`perf_baseline_investigation.md`, `manual_gates.md`, `phase_oracles.md`,
`loc_report.md`. Plus `chelis/docs/book/src/` chapters which are **kebab-case**
(`first-program.md`, `cli.md`, `install.md`, `reef.md`, `effects.md`,
`reference.md`).

`octant/docs/` — **mixed snake_case + kebab-case-with-versions**:
- snake: `architecture.md`, `chelis_octant_spec.md`, `error_messages.md`, `extending.md`, `supported_subset.md`
- kebab + version markers: `red-team-O3.md`, `red-team-O4.md`, `red-team-pre-v0.1.0.md`, `red-team-v0.2.0-final.md`, `red-team-v0.2.0-mid.md`, `v032-nautilus-investigation.md`

`nautilus/docs/`, `coral/docs/` — **SCREAMING_SNAKE_CASE for status reports**:
`BENCHMARK_FINDINGS.md`, `EVAL_STARTUP_FINDINGS.md`, `NAUTILUS_STATUS.md`,
`RELEASES.md`, `STATUS.md`, `UPSTREAM_BUGS.md`.

`shoals/spec/` — single file, simple lowercase: `phase3l.md`. No `docs/`
directory in shoals.

### `reef.toml` package manifests

All five reef packages follow the same pattern: `name = <kebab>` +
`module_prefix = <Pascal>`:

```
chelis-std    | name = "chelis-std"  | module_prefix = "Std"
nautilus      | name = "nautilus"    | module_prefix = "Nautilus"
coral         | name = "coral"       | module_prefix = "Coral"
shoals        | name = "shoals"      | module_prefix = "Shoals"
octant        | name = "octant"      | module_prefix = "Octant"
```

The chelis-std reef package uses the short prefix `Std` (not `ChelisStd`).
The package name is kebab; the prefix is PascalCase, forced by Surf's
case-split rule.

Cross-shell dependency convention: kebab-cased keys, semantic-version-string
values, no git-revision references in any active shell.

```toml
# Shoals (highest dep fan-in)
chelis-std = { version = "0.1.0" }
nautilus   = { version = "0.5.0" }
coral      = { version = "0.5.0" }
```

`shoals/reef.toml` uniquely sets `additional_sources = ["properties", "references"]`.

### Hidden config dirs

- `.github/workflows/`: `ci.yml`, `release.yml`, `nightly.yml` — all lowercase, no hyphens. Uniform across all five repos.
- `.claude/skills` symlinks to `agent-skills/` (per `CLAUDE.md` shared-skills policy). Skill dirs are kebab: `backend-numerics`, `cli-surface`, `example-corpus`, `phase-gate`, `redteam-exec`, `spec-sync`. Skill manifest inside is `SKILL.md` (PascalCase).
- `.claude/commands/`: `red-team.md` (kebab).
- VSCode extension at `chelis/editors/vscode/`: `package.json` `name = "chelis-vscode"`, syntax files `chelis.tmLanguage.json`, `chelis-deep.tmLanguage.json` (kebab).
- Tree-sitter grammars at `chelis/grammars/`: two parallel grammars `tree-sitter-chelis-surf` and `tree-sitter-chelis-deep` (kebab). Grammar entry `grammar.js`, queries at `queries/highlights.scm`, generated parser under `src/parser.c`.

---

## 4. Snapshot — in-source identifier layer

### Surf module ladder — 144 distinct modules

Composition: 44 two-component (`Org.Category`), 69 three-component
(`Org.Category.Subcategory`), 30 four-or-more component, 1 single-word.

Compound style breakdown for non-single-word components:

- **Title-case compound** (71 modules — the dominant style): `Nautilus.LinAlg`,
  `Nautilus.CurveFit`, `Coral.GroupBy`, `Coral.Frame`, `Shoals.Orderbook`,
  `Std.Loss.CrossEntropy`, `Std.Nn.RmsNorm`.
- **ALL-CAPS abbreviation component** (3 modules only): `Coral.Internal.HAMT`,
  `Std.Loss.KlDiv`, `Std.Tests.IO`. Rare but present.
- **Single-word** (everything else): `Coral.Frame`, `Shoals.Risk`,
  `Nautilus.Distance`, `Std.Test`.

Per-shell prefixes: `Nautilus.*`, `Coral.*`, `Shoals.*`, `Octant.*`, `Std.*`.

### Surf types and ADT constructors

Type declarations: 1,396, all PascalCase. ADT constructors: 10,818, all
PascalCase. Examples observed: `Frame`, `Column`, `GroupedFrame`, `Hamt`,
`ColumnType`, `KeyValue`, `YieldCurve`, `OrderBook`, `Order`, `Decimal`,
`Date`, `Duration`, `Tokenizer`, `Json`, `JsonInt`, `JsonBool`, `JsonArray`,
`JsonObject`, `JsonNull`, `AggSum`, `AggMean`, `AggMax`, `AggMin`, `AggCount`,
`RoundUp`, `RoundDown`, `RoundHalfEven`, `RoundHalfUp`, `Monday`, `Friday`,
`Saturday`, `BidsNode`, `Empty`, `Leaf`, `Collision`, `Activation`, `Relu`,
`Sigmoid`, `Option`. All use Title-case compounds where applicable; no
ALL-CAPS abbreviation styling found in type names (`Hamt` is the type
constructor; `HAMT` is the module that exports it — a within-pair mismatch).

### Surf functions and values — 30,109 occurrences, ~1,849 unique

About 93% pure snake_case (`predict`, `loss`, `inner_product`, `matvec`,
`from_pairs`, `with_column`, `rolling_mean`, `golden_section_search`,
`brent_minimize`, `gauss_legendre_5`, `cubic_hermite`, `simpsons`).

Frequent type-suffix patterns (numbers are total occurrences across the
ecosystem):

| Suffix | Count | Sample callers |
|---|---|---|
| `*_string` | 496 | parsers, formatters |
| `*_int` | 463 | tensor ops, Frame variants |
| `*_scalar` | 362 | finance / math |
| `*_bool` | 138 | predicates |
| `*_f32` | 118 | tensor monomorphic variants |
| `*_vec` | 16 | linear algebra |

Frequent prefix-namespacing patterns:

| Prefix | Count | Module |
|---|---|---|
| `test_*` | 9,653 | every test module across the ecosystem |
| `parse_*` | 1,064 | parsers |
| `decimal_*` | 988 | `Std.Decimal` |
| `tensor_*` | 927 | `Std.Tensor`-style modules |
| `la_*` | 51 | `Nautilus.LinAlg` only |
| `bs_*` | 2 | `Shoals.Pricing` only |
| `mc_*` | (some) | `Shoals.Stochastic` |
| `gbm_*` | (some) | `Shoals.Stochastic` |

### Surf parameters — bimodal

- **Math-heavy code (Shoals, Nautilus)**: single-letter parameters dominate.
  Shoals.Pricing parameters: `s` (spot), `k` (strike), `r` (rate), `sigma`,
  `t` (time), `s0`, `mu`. Nautilus.LinAlg parameters: `a`, `b`, `m`, `n`,
  `i`, `j`, `k`, `alpha`, `acc`.
- **Data-processing code (Coral)**: descriptive snake_case parameters.
  `col`, `col_name`, `col_keys`, `df`, `from_pairs`, `values`, `entries`,
  `idx`, `hash`.

No camelCase parameter names anywhere.

### Surf dimensions — top names by frequency

dtype tokens (not strictly dimensions, but appear in tensor type slots):
`f32` 20,497, `int64` 3,284, `int32` 76, `bool` 122.

Actual semantic dimensions: `seq` 2,170, `vocab` 1,520, `classes` 708,
`b` 456, `unit` 228, `out_dim` 152, `hidden` 152, `features` 152, `n` 105,
`in_dim` 76, `cols` 76, `p` 7, `m` 2, `k` 2.

All snake_case or single letter. No PascalCase or camelCase dimension names.

### "camelCase" hits in `.ch` sources

Targeted grep across all reef checkouts surfaced these and only these tokens
matching `[a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*`: `nAlice`, `nBob` (chelis examples
— `n` + capitalized proper noun, lexes as a single `Ident`); `c1I`, `dW`,
`tA`, `tI` (nautilus, math notation: index + uppercase math-symbol);
`rT` (shoals, same pattern). No general camelCase identifier convention
exists in `.ch` source.

### Rust identifiers

Rust standard throughout: PascalCase types/traits, snake_case functions/modules,
SCREAMING_SNAKE_CASE consts. Examples: `OctantError`, `SourceId`, `Span`,
`FunctionRegistry`, `MetaMap`, `MetaExpr`, `Expr`, `Atom`, `List`,
`DEFAULT_MODEL`, `API_URL`. No deviations.

### Python identifiers

PEP 8 throughout: `snake_case` functions, `PascalCase` classes, SCREAMING
constants. Examples: `load_api_key`, `chelis_version`, `build_and_compile_canary`,
`first_nonempty`, `extract_text`, `scrape_pdf_links`, `PDFLinkExtractor`,
`DEFAULT_MODEL`, `API_URL`, `RETRYABLE_CODES`, `FALLBACK_CHAIN`. No deviations
in Chelis-authored Python.

### Rust test function names

Snake_case throughout. Sample (~100 tests): `roundtrip_simple_def`,
`roundtrip_adt`, `roundtrip_example_mnist`, `span_id_returns_none_for_atom`,
`span_id_unicode_preserved_bit_for_bit`, `dim_params_produce_d_var`,
`effect_annotations_desugar_into_t_fn_metadata`, `abs_emits_fabsf`,
`add_rejects_mixed_precision`,
`ad_backward_nodes_carry_grad_marker_and_forward_span`,
`adv1_long_chain_5_ops`, `agreement_add`, `assert_check_clean`.

### Surf test function names

Pattern across all shells: `def test_*` for unit tests, `def example_*` for
illustrative example functions in the same files (Shoals).

---

## 5. Snapshot — Deep layer

- **Tag names**: closed ~60-tag vocabulary. Single-token tags are bare lowercase: `app`, `var`, `lit`, `def`, `sig`, `module`, `import`, `dim`, `deftype`, `variant`, `field`, `arm`, `meta`. Compound tags use **hyphens** as the compound separator: `t-fn`, `t-prim`, `t-tensor`, `t-var`, `pat-var`, `pat-lit`, `pat-ctor`, `d-name`, `d-var`, `d-lit`. Authoritative list lives in `spec/03-deep-syntax.md:262+`; hardcoded in `crates/chelis-deep/src/validate.rs` and emitted by `crates/chelis-deep/src/printer.rs`. Not user-controlled.
- **Symbol grammar**: `[A-Za-z_][A-Za-z0-9_-]*` — accepts hyphens. This is intentional, not a latent defect. The wider grammar exists so the closed compound-tag vocabulary can use hyphens (`t-fn`, `pat-ctor`, `d-name`) as the compound separator. User-defined Deep symbols (variable names, function names) originate from Surf desugaring and inherit Surf's no-hyphen rule by construction. The Surf-vs-Deep hyphen asymmetry is a working separation of concerns: Surf is the human authoring surface where operator ambiguity rules out hyphens; Deep is the compiler IR where compound tag names follow a deliberate naming convention.
- **Module paths in Deep**: appear as `(module {} foo.bar …)` with dot-separated lowercase components — the result of Surf's PascalCase → lowercase desugaring.
- **Snapshot test files (Octant)**: `cli__error_format__parse_error_format.snap`, `cli__error_format__type_error_format.snap`, `cli__error_format__unsupported_error_format.snap`. Insta-style double-underscore separator: `{context}__{section}__{test_name}.snap`. Octant is the only shell using this convention.

---

## 6. Project-level identifier conventions (cross-cutting)

### Phase identifiers

Current phase: `phaseA` — lowercase `phase` + uppercase letter. No hyphen,
no underscore between `phase` and the letter.

Historical phases (in `spec/design/` filenames and old commits): `phase1a`,
`phase1b`, `phase1c`, `phase1d`, `phase1e`, `phase1f`, `phase3a`, `phase3b`,
`phase3g`, `phase3j`, `phase3k`, `phase3m`, `phase3n`, `phase5`. Pattern:
lowercase `phase` + digit + lowercase letter.

Current Phase A is the first to use letter-only (no digit) phase id. The
case shift from `phase3j` (lowercase `j`) to `phaseA` (uppercase `A`) is the
new convention.

### Branch naming

`feat/phaseA-item9-lockfile-origin`, `feat/phaseA-item8-autofetch`,
`feat/phaseA-item7-bootstrap`, `fix/phaseA-chelis-std-runtime`. Pattern:
`{type}/{phase-identifier}-{item-slug-in-kebab}`. Type prefixes follow
conventional-commits style.

### Commit-message conventions

Conventional commits: `feat(phaseA-item9): …`, `test(phaseA-item8): …`,
`style(reef): …`, `docs(spec): …`. Scope strings use the same `phaseA-itemN`
pattern.

### CI workflow filenames

`ci.yml`, `release.yml`, `nightly.yml`. All lowercase, no separator.
Identical across all five ecosystem repos.

---

## 7. The chelis-std / chelis-runtime distinction (not a shoreline mismatch)

These are two different artifacts with overlapping naming. Documented and
reconciled in commit `bcda878 docs(spec): mark chelis-std as runtime, not
shell, in reef + canonical refs`.

- **`chelis-std`** is the language **runtime** reef package. It lives at
  `chelis/packages/chelis-std/`, has `name = "chelis-std"` and
  `module_prefix = "Std"` in its `reef.toml`, version-marches with the
  compiler, ships bundled with the toolchain, and is referenced from every
  reef shell as `chelis-std = { version = "0.1.0" }`. The
  `LockSource::Bundled` variant (commit `32b8b99`) tracks this as a distinct
  source kind (not network).

- **`chelis-runtime`** is a **Rust crate** at `chelis/crates/chelis-runtime/`.
  It supports the C backend at runtime — different artifact, different role.

The canonical reference (`spec/design/chelis_canonical_reference.md` lines
169-175) explicitly distinguishes runtime from shell:
> "Runtime vs shells. `chelis-std` is the language runtime, not a shell. It
> version-marches with the compiler, ships bundled with the toolchain, and
> cannot be substituted independently."

The same table marks `chelis-std` as `Runtime (compiler-bundled)` while
`nautilus`, `coral`, `shoals`, `octant`, `school`, `darwin`, `hull`, `beacon`
are marked as `Shell`. There is no remaining naming reconciliation work
here; the two names refer to two different things by design.

---

## 8. Inventory of observed inconsistencies

Concrete items the survey found that are not consistent with their
surrounding convention. Provided as input to a future remediation pass.

1. **`phaseA_item{6,7,8,9}_*.rs` integration tests** in
   `chelis/crates/chelis-reef/tests/` — camelCase `phaseA` chunk inside
   otherwise snake_case Rust filenames. Every other `.rs` file in the
   ecosystem is pure snake.

2. **Six Nautilus example modules with lowercase compound names** (no
   internal capital):
   - `Nautilus.Examplerootfind` (`nautilus/src/examplerootfind.ch`)
   - `Nautilus.Exampleodedemo`
   - `Nautilus.Exampledistributions`
   - `Nautilus.Exampleintegration`
   - `Nautilus.Exampleoptim`
   - `Nautilus.Apismoke`

   Spec rule (`spec/02-surf-syntax.md:79`) is "Module names are PascalCase."
   The body component should be `ExampleRootFind`, etc. Internal/test
   modules; no public API impact, but they violate the documented rule.

3. **`Coral.ApiSmoke` vs `Nautilus.Apismoke`** — same conceptual file
   (smoke-test module), two casings. `coral/src/apismoke.ch` declares the
   Title-case form; `nautilus/src/apismoke.ch` declares the lowercase form.

4. **`Hellotensor` module** — agent surveyed 144 module declarations and
   found one single-word lowercase-after-first form, `Hellotensor`. Likely
   the chelis "hello tensor" example; should be `HelloTensor` if read as
   "Hello Tensor".

5. **Module-compound styling: ALL-CAPS abbreviations vs Title-case
   compounds**, both in use across the ecosystem:
   - ALL-CAPS abbreviation: `Coral.Internal.HAMT`, `Std.Loss.KlDiv`,
     `Std.Tests.IO` (3 modules total).
   - Title-case compound: `Nautilus.LinAlg`, `Nautilus.CurveFit`,
     `Coral.GroupBy` (71 modules — the dominant form).
   No documented rule selects between them. Note also the within-pair
   mismatch: `Coral.Internal.HAMT` (the module) exports `Hamt` (the
   Title-case type constructor).

6. **`la_*` prefix policy in `Nautilus.LinAlg`** — applied to 51
   low-level helper functions (`la_vec_sub`, `la_vec_add`, `la_basis_n_f32`,
   `la_zeros_mat_like`, `la_identity_n`, `la_lu_fwd_step`, `la_vec_saxpy`,
   `la_qr_build_q`, etc.); absent from public/high-level wrappers
   (`transpose`, `matmul_wrap`, `gram`, `aat`, `diag`, `trace_mat`,
   `inner_product`, `frobenius_norm`, `scale_vec`, `matvec`,
   `qr_decompose`, `lu_solve`, `svd_n`, `eig_n`, `cholesky_n`). No
   documented rule for when to apply the prefix.

7. **`bs_*` prefix isolation in `Shoals.Pricing`** — only two functions
   carry it (`bs_call_scalar`, `bs_put_scalar`). Other pricing functions
   in the same module use bare names (`call_prices`, `put_prices`,
   `call_total`, `put_total`, `deltas_call`).

8. **`*_int` suffix in `Coral.Frame` is semantically misleading.** The
   suffix does not mean "returns int". It means "Frame-variant accepting
   an int column name" rather than a raw-tensor variant.
   - `is_nan` (tensor[n,f32]→tensor[n,bool]) vs `is_nan_int` (Frame[n], string→tensor[n,bool])
   - `any_nan` vs `any_nan_int`, `count_nan` vs `count_nan_int`,
     `fill_nan` vs `fill_nan_int`, `drop_nan` vs `drop_nan_int`.
   The bare form is the polymorphic / tensor version; the `_int`-suffixed
   form takes a Frame column name. The naming reads as type-suffix but the
   semantics is dispatch-form-suffix. This is not just inconsistent; it is
   misleading.

9. **`install_chelis_std.sh`** in `nautilus/scripts/` — single shell script
   in violation of the project-wide "never shell" policy
   (`CLAUDE.md` Scripting Language Policy).

10. **Doc-directory convention split, three coexisting styles in
    `chelis/spec/design/`:**
    - snake_case (~25 files): `phase1a_kernel_codegen.md`,
      `chelis_canonical_reference.md`, `chelis_project_plan.md`, ….
    - kebab-case (5 files, concentrated in upstream-bugs subset):
      `grad-eval-host-runtime.md`,
      `host-emit-hashmap-iteration-nondeterminism.md`,
      `host-eval-perf-mc-rigor.md`, `producer-string-sanitization.md`,
      `reef-multi-source-roots.md`.
    Directory does not have a documented convention.

11. **Per-shell `docs/` filename conventions diverge across repos.**
    - `chelis/docs/`: snake_case (`lin_rca_report.md`, `perf_baseline.md`).
    - `chelis/docs/book/src/`: kebab-case for book chapters
      (`first-program.md`, `cli.md`).
    - `nautilus/docs/`, `coral/docs/`: SCREAMING_SNAKE for status reports
      (`RELEASES.md`, `STATUS.md`, `UPSTREAM_BUGS.md`,
      `BENCHMARK_FINDINGS.md`).
    - `octant/docs/`: snake for architecture, kebab for versioned red-team
      reports (`red-team-O3.md`, `red-team-v0.2.0-final.md`).

12. **Octant's BUILTINS table emits a stale Nautilus path.**
    `octant/reef.toml` lines 9-16 contain a comment acknowledging that
    Octant still emits `Nautilus.Special.normal_cdf` while nautilus
    v0.5.0 moved `normal_cdf` to `Nautilus.Distributions`. Test fixtures
    are correct; the hardcoded BUILTINS map is not. Known, deferred — not
    a naming-convention issue per se, but a naming-data drift between
    shells.

13. **Deep `Symbol` grammar accepts hyphens; Surf does not — intentional
    structure, not a defect.** Earlier drafts of this snapshot framed the
    asymmetry as a latent round-trip risk. That framing was incorrect.
    Hyphens in Deep are the compound-tag separator: the closed vocabulary
    uses `t-fn`, `t-prim`, `t-tensor`, `t-var`, `pat-var`, `pat-lit`,
    `pat-ctor`, `d-name`, `d-var`, `d-lit`. User-defined Deep symbols
    originate from Surf desugaring and inherit Surf's no-hyphen rule by
    construction — there's no path by which a hyphenated user symbol can
    enter Deep today. The actual rule that matters is narrower: a
    user-defined Deep symbol (anything not in the closed tag vocabulary)
    must satisfy the Surf identifier charset `[A-Za-z_][A-Za-z0-9_]*`.
    The closed tag vocabulary is the allowlist; everything else passes
    through the same constraint Surf imposes. This rule is enforceable by
    the lint without changing the Deep lexer.

14. **Phase-identifier case shift from old to current.** Historical
    phase docs use lowercase letter (`phase3j`, `phase1a`); current
    Phase A uses uppercase letter (`phaseA`). Whether this is a
    deliberate convention change or an inconsistency is undocumented.

15. **`chelis_python` lib name** in `Cargo.toml`: package name
    `chelis-python`, lib name `chelis_python`. This is the standard Rust
    hyphen→underscore coercion for symbol names. Not strictly an
    inconsistency — it is the language norm — but worth noting as a
    shoreline crossing for readers tracking the kebab→snake boundary.

A lint pass over the actual file trees would likely surface a few more in
the same buckets. The 15 items above are what showed up across three
deeper Explore-agent passes.

---

## 9. Documented naming guidelines (current state of the spec)

Searched all five repos for naming guidance. Findings:

- **`spec/01-nomenclature.md`** defines layer terms (Surf, Deep,
  CLI commands, file extensions) but does not specify identifier-style
  rules.
- **`spec/02-surf-syntax.md:79-91`** documents *one* style rule:
  "Module names are PascalCase. No nesting within a file." Plus the
  on-disk mapping rule.
- **`spec/02-surf-syntax.md:115, 133`** document dimension and
  parameter placement, but not casing.
- **`spec/03-deep-syntax.md`** documents the closed tag
  vocabulary.
- **`spec/design/chelis_canonical_reference.md` §4** documents
  the dual-syntax architecture and the runtime-vs-shell distinction
  (lines 130-147 cover the ecosystem-naming table).
- **No `STYLE.md`, `CONVENTIONS.md`, or other style-guide-level
  document** exists in any repo.
- **`CLAUDE.md` Surf Style Guide** (project root, agent contract) gives
  guidance on *Surf code style* (when to use `def : T` vs
  `def -> T = …`, when to annotate types, naming for `h1` / `logits`
  / `loss` / `probs` intermediates) but does not cover the
  ecosystem-level naming-convention questions in §8 above.
- **Per-shell agent-skills**: 6 shared skills exist (`backend-numerics`,
  `cli-surface`, `example-corpus`, `phase-gate`, `redteam-exec`,
  `spec-sync`). None of them is a naming-style skill.

Summary: identifier-level naming rules (snake_case for values, PascalCase
for types, ABBREV vs Abbrev, prefix-policy, type-suffix-policy) are not
formally specified anywhere in the ecosystem.

---

## 10. References

- `spec/01-nomenclature.md` (existing — natural documentary home).
- `spec/02-surf-syntax.md` (Surf grammar, module-path mapping, reserved words).
- `spec/03-deep-syntax.md` (Deep tag vocabulary, symbol grammar).
- `spec/design/chelis_canonical_reference.md` (top-level reference; runtime-vs-shell distinction at lines 169-175; ecosystem-naming table at lines 130-147).
- `crates/chelis-surf/src/lexer.rs` (Surf identifier grammar — case split, charset, keywords).
- `crates/chelis-deep/src/lexer.rs` (Deep symbol grammar — note the hyphen asymmetry).
- `crates/chelis-surf/src/format.rs` (formatter behavior re: identifier case).
- `crates/chelis-backend-c/src/emit.rs` (backend symbol-emission behavior).
- `packages/chelis-std/reef.toml` (`name = "chelis-std"`, `module_prefix = "Std"`).
- `nautilus/reef.toml`, `coral/reef.toml`, `shoals/reef.toml`, `octant/reef.toml` — cross-shell convention witnesses (in their respective peer repos).
- Commit `bcda878 docs(spec): mark chelis-std as runtime, not shell, in reef + canonical refs` (resolved std/runtime documentation).
- Commit `32b8b99 feat(reef): add LockSource::Bundled variant for the chelis-std runtime` (lockfile-side support).
