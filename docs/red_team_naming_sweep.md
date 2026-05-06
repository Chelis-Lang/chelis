# Red Team — Naming Sweep Phase 4

## Header

- **Date:** 2026-05-06
- **Agent context:** fresh local subagent, no prior framing, working from
  `chelis/spec/01-nomenclature.md` and the post-sweep source state.
- **Build SHA used:** `592a9735672758b2849c8d2a8d21cf9e412e75af` (chelis main).
  Working tree shows `M README.md` only; lint binary built clean.
- **Lint binary:** `/home/jeff/Documents/scratch/chelis/target/debug/chelis`,
  command `chelis lint [--check] [--rule <id>] <path>`.
- **Repos verified:** chelis monorepo (incl. chelis-std runtime, chelis-lint
  crate), nautilus, coral, shoals, octant.
- **Lint test suite:** `cargo test -p chelis-lint` — 125/125 pass.

---

## Baseline state (informational, before any probe mutations)

`chelis lint --check <repo>` exit codes against post-sweep working trees:

| Repo     | exit | violations |
|----------|------|------------|
| chelis   | 1    | 26         |
| nautilus | 1    | 6          |
| coral    | 1    | 48         |
| shoals   | 0    | 0          |
| octant   | 0    | 0          |

**Every violation in the baseline is `prefix-namespace (§7.1)`** — closer-read
candidates the rule flags as "either drop or escalate to §7.1.1." Examples:

- chelis: `Std.Time` family (`date_*`, `days_*`, `day_*`), `Std.Optim`
  (`lamb_*`), `Std.Tokenizer` (`seed_*`).
- nautilus: `Nautilus.ExampleOdeDemo` (`eo_*`), `Nautilus.ExampleOptim`
  (`eop_*`).
- coral: `Coral.Frame` (`enum_*`, `key_*`, `list_*`, `ints_*`),
  `Coral.GroupBy` (`agg_*`, `sum_*`, `mean_*`, `min_*`, `max_*`),
  `Coral.Internal.Hamt` (`mask_*`, `hash_*`, `char_*`, `list_*`),
  `Coral.Io` (`csv_*`, `json_*`, `join_*`), `Coral.Join` (`left_*`),
  `Coral.Reshape` (`melt_*`).

CI status (`chelis/.github/workflows/ci.yml:42-49`): `chelis lint` is invoked
**without `--check`**, so it is informational; the comment promises a switch to
`--check` "once Wave A lands (the post-sweep state should report zero
violations)." The post-sweep state does not report zero violations. None of
nautilus/coral/shoals/octant CI invokes `chelis lint` at all.

---

## Probe 1 — Lint completeness per documented rule

| § | Rule id | Mutation | Result | Evidence |
|---|---------|----------|--------|----------|
| 2.9 | no-shell-scripts | `install.sh` placed in scratch dir | **PASS** | fired with `§2.9: shell script disallowed by project policy; port to Python or remove` |
| 6.2 | module-compound-titlecase | `module Coral.Internal.HAMT` | **PASS** | `§6.2: module component HAMT uses ALL-CAPS abbreviation; rewrite as Title-case (e.g., Hamt)` |
| 6.3 | module-pascal-components | `module Nautilus.Examplerootfind` | **PASS** | `§6.3: module component Examplerootfind looks like a multi-word compound but has no internal capital` |
| 7.1 | prefix-namespace | unrecognized `xyz_*` prefix on 2 defs | **PASS** | fired twice with shorthand-mismatch message |
| 7.1.1 | prefix-namespace allowlist | `bs_call_scalar`/`bs_put_scalar` in `Shoals.Pricing` accepted; `xyz_call_scalar`/`xyz_put_scalar` flagged | **PASS** | bs_ rc=0, xyz_ rc=1 |
| 7.2 | type-suffix-policy | `is_nan_int(f: Frame, name: string) -> tensor[n, bool]` | **PASS** | `§7.2: function is_nan_int has element-type suffix _int but first argument type Frame is a container — dispatch form is prohibited` |
| 8.1 | doc-filename-convention | `chelis/spec/foo.md` (no numeric prefix) | **CONDITIONAL/FAIL** | rc=0, no violation. The rule dispatches non-numeric `spec/` files to §8.2 by shape. A non-numbered `chelis/spec/foo.md` slips through silently. The rule still catches `99-bad_format.md` (numeric+underscore) as §8.1, and `02-foo-bar.md` as valid. The brief's literal expectation (§8.1 must fire on a non-numbered file in `chelis/spec/`) is not met. |
| 8.2 | doc-filename-convention | `spec/design/foo-bar.md` | **PASS** | `§8.2: filename foo-bar.md does not match … snake_case` |
| 8.3 | doc-filename-convention | `docs/foo-bar.md` | **PASS** | `§8.3: snake_case for narrative docs; SCREAMING_SNAKE_CASE for status reports` |
| 8.5 | doc-filename-convention | `docs/src/foo_bar.md` (mdBook tree) | **PASS** | `§8.5: kebab-case (mdBook URL stability)`. `SUMMARY.md` correctly exempt. |
| 9.1 | phase-identifier-case | `phaseA_test.rs` | **PASS** | `§9.1: filename phaseA_test.rs uses phaseA form; rename to phase_a` |
| 10.3 | snapshot-filename-pattern | `cli__error_format__bad-name.snap` (kebab in test_name) | **PASS** | `§10.3: snapshot filename does not match {context}__{section}__{test_name}.snap` |
| 11.1 / 1.5 | deep-user-symbol-charset | `(def {} my-func (params {} x) ...)` and a separate file with compound tags | **PASS** | hyphenated user symbols `my-func`, `ok-name` flagged; closed-vocabulary `t-fn`, `t-prim`, `var`, `app`, `def`, `params`, `lit` not flagged |

Probe 1 verdict: 12/13 PASS, 1 conditional fail on §8.1 (literal vs dispatch
intent). See classification below.

---

## Probe 2 — Sweep-completeness (revert each cleanup item)

Each cleanup item was reverted on a scratch tempdir copy and the lint
re-run.

| Cleanup | Revert | Lint output |
|---------|--------|-------------|
| Coral `is_nan_col → is_nan_int` (commit `5d9d726` family) | `def is_nan_int[n](df: Frame[n], col_name: string) -> tensor[n, bool]` | **PASS** §7.2 dispatch-form |
| Nautilus `ExampleRootFind → Examplerootfind` | `module Nautilus.Examplerootfind` | **PASS** §6.3 |
| chelis `phase_a_*.rs → phaseA_*.rs` | `phaseA_item9.rs` | **PASS** §9.1 |
| octant `red_team_o3.md → red-team-O3.md` | `docs/red-team-O3.md` | **PASS** §8.3 |
| chelis `Hellotensor → HelloTensor` revert | `module Demo.Hellotensor` | **PASS** §6.3 |
| coral `HAMT → Hamt` revert | `module Coral.Internal.HAMT` | **PASS** §6.2 |
| chelis-std `Std.IO → Std.Io` revert | `module Std.IO` | **PASS** §6.2 |

Probe 2 verdict: 7/7 PASS. Every cleanup is locked by the lint.

---

## Probe 3 — Exception scoping

| Probe | Result | Evidence |
|-------|--------|----------|
| `cli_doc.md` placed in mdBook tree (`docs/book/src/`) | **PASS** | fires §8.5 (kebab required) |
| Real `chelis/docs/book/src/SUMMARY.md` | **PASS** | rc=0, no violations (tool-required exception applied) |
| `lin-rca-report.md` placed in non-mdBook docs/ | **PASS** | fires §8.3 (kebab not allowed for narrative docs) |

Probe 3 verdict: 3/3 PASS — the mdBook exception is correctly path-scoped via
`book.toml` ancestor detection plus literal-filename exemption for SUMMARY.md
and README.md.

---

## Probe 4 — Sibling-sweep heuristic

| Probe | Result | Evidence |
|-------|--------|----------|
| `Nautilus.LinAlg → Nautilus.Linalg` (lowercase 2nd word) | **FAIL** | rc=0; not flagged. The §6.3 heuristic requires a `>= 7`-char run of lowercase after the leading capital, and "inalg" is 5 chars. |
| `inner_join → inner_join_int` on a `Frame[n]` first arg | **PASS** | fires §7.2 dispatch-form |
| `Std.Nn.LSTM` (ALL-CAPS abbreviation in a previously-clean ladder) | **PASS** | fires §6.2 |

Sibling-sweep extension test (additional sibling cases):

| Module | Result | Notes |
|--------|--------|-------|
| `Nautilus.Linalg` | not flagged | 5-char run, below threshold |
| `Nautilus.Curvefit` | flagged §6.3 | 7-char run |
| `Coral.Groupby` | not flagged | 6-char run |
| `Std.Nn.Rmsnorm` | not flagged | 6-char run |
| `Std.Loss.Crossentropy` | flagged §6.3 | 10-char run |

Probe 4 verdict: **partial FAIL on §6.3 sibling sweep**. The 7-character
lowercase-run threshold misses real compound violations of length 5-6
(`Linalg`, `Groupby`, `Rmsnorm`). The spec calls these out in §6.1 and §6.3
explicitly (`Nautilus.LinAlg`, `Coral.GroupBy`, `Std.Nn.RmsNorm`) yet the
inverse forms slip the lint. See severity classification.

---

## Probe 5 — Cross-shell consistency

Greps against the four shell repos plus chelis (excluding `target/`,
`.git/`, `.claude/worktrees/`, vendored `.venv*`):

- `Coral.Internal.HAMT` in source: **0 hits**. Matches in `chelis-lint` rule
  test fixtures (legitimate test of the rule) and in
  `docs/ecosystem_naming_snapshot.md` (narrative archaeology) only.
- Old Nautilus example modules (`ExampleRootFind` etc — checked for
  the `*lower`-tail forms): **0 hits in source `module` declarations**. All six
  modules are now PascalCase. The literal strings `Nautilus.Apismoke`,
  `Nautilus.Examplerootfind` etc. survive only in the pre-sweep snapshot doc
  and in the lint's own regression-test fixtures.
  - Note: `Nautilus.ODE` / `Nautilus.SDE` survive in
    `nautilus/docs/book/*.html` (rendered mdBook output) and in
    `chelis/spec/design/chelis_phase3_plan.md` (archaeology). Live source uses
    `Nautilus.Ode` / `Nautilus.Sde`. Rendered HTML drift is build-output
    staleness, not a source-level regression.
- `is_nan_int|any_nan_int|count_nan_int|fill_nan_int|drop_nan_int` outside
  coral: **0 source hits**. All matches inside `chelis-lint`'s
  type_suffix_policy.rs are deliberate test fixtures of the §7.2 rule.
- `Std\.IO[^a-zA-Z]` in source/active docs: **2 narrative hits** —
  `nautilus/spec/phase3j.md:35` (referring to a phase 3a stub),
  `chelis/.github/workflows/ci.yml:46` (comment about Std.IO sibling-sweep).
  Both are commentary, not live module references; no source `module Std.IO`
  declarations remain.

Probe 5 verdict: **PASS for source code, advisory drift in narrative docs**
(rendered mdBook HTML and design-archive markdown still mention old names).

---

## Probe 6 — Style/lint agreement

### Rules registered (`registry::all_rules`)

| Rule id | spec_ref | Spec section exists |
|---------|----------|---------------------|
| no-shell-scripts | §2.9 | yes |
| phase-identifier-case | §9.1 | yes |
| module-compound-titlecase | §6.2 | yes |
| module-pascal-components | §6.3 | yes |
| doc-filename-convention | §8 / 8.1 / 8.2 / 8.3 / 8.5 (dispatched per slot) | yes (no §8.4 dispatch — see below) |
| snapshot-filename-pattern | §10.3 | yes |
| type-suffix-policy | §7.2 | yes |
| prefix-namespace | §7.1 (allowlist for §7.1.1) | yes |
| deep-user-symbol-charset | §11.1 (refs §1.5) | yes |

Every cited section resolves to a real heading in
`chelis/spec/01-nomenclature.md`.

### Spec sections without an enforcing rule

- §1.1–§1.7 — hard language constraints, parser-enforced; lint not needed.
- §2.1, §2.2, §2.3, §2.4, §2.5, §2.6, §2.7, §2.8, §2.10, §2.11, §2.12 — file
  path/manifest conventions; no lint enforcement (only §2.9 has a rule).
- §3.1, §3.2, §3.3, §3.4 — Surf identifier conventions; no lint.
- §4 — Rust identifier rules; no lint.
- §5 — Python identifier rules; no lint.
- §6.1, §6.4 — module ladder structure & runtime/shell distinction; documentation only as called for in the brief.
- §7.1.1 — covered as exemption inside §7.1's `MODEL_NAMESPACE_PREFIXES`.
- §8.4 (versioned reports) — **transitively covered** by §8.3's snake check
  (kebab fails). The lint cites §8.3 as the spec_ref, not §8.4. Acceptable.
- §9.2, §9.3, §9.4 — branch/commit/CI workflow conventions; lint walks the
  filesystem, never git history. The CI workflow filename rule is also not
  enforced (no rule fires on `.github/workflows/Foo.yml`); see severity.
- §10.1, §10.2 — Surf and Rust test-function naming; no lint.
- §11.2 — documentation-only.

### §12 enforcement infrastructure

`crates/chelis-lint/src/exceptions.rs::verify_cross_refs` exists and has
unit-test coverage (10 tests: cross_ref resolution, glob matching, exception
application). However, `crates/chelis-cli/src/main.rs:4025` builds the runtime
exception list as `Vec::new()` and **never calls `verify_cross_refs`**. The
documented invariant ("entries with unresolvable cross_refs are build-time
errors") is enforced only in the lint crate's unit tests, not at the CLI's
startup. Any production exception added to that vector with a bogus cross_ref
would be silently applied. See severity.

Probe 6 verdict: rule/spec mapping is consistent; one MEDIUM gap in §12
runtime enforcement.

---

## Probe 7 — Documentation honesty

Snapshot §8 cleanup items, post-sweep:

| # | Claim | Verified |
|---|-------|----------|
| 1 | phaseA_*.rs → phase_a_*.rs | **PASS** — 5 phase_a_*.rs in `chelis/crates/chelis-cli/tests/`, 0 phaseA_*.rs |
| 2 | six Nautilus example modules | **PASS** — `module Nautilus.ExampleRootFind/ExampleIntegration/ExampleOptim/ExampleDistributions/ExampleOdeDemo/ApiSmoke` all PascalCase |
| 3 | Coral.ApiSmoke vs Nautilus.Apismoke | **PASS** — both `ApiSmoke` |
| 4 | Hellotensor → HelloTensor | **PASS** — `examples/hello_tensor.ch` declares `module HelloTensor`; lint test fixture confirms accept |
| 5 | HAMT, IO, KLDiv | **PASS** — zero `^module .*\.HAMT\|IO\|KLDiv` in any repo |
| 6 | la_ in LinAlg accepted | **PASS** — `chelis lint --rule prefix-namespace nautilus/src/linalg.ch` rc=0 |
| 7 | bs_ in Pricing accepted per §7.1.1 | **PASS** — `chelis lint --rule prefix-namespace shoals/src/pricing.ch` rc=0 |
| 8 | Coral _int → _col | **PASS** — `is_nan_col`, `any_nan_col`, `count_nan_col`, `fill_nan_col`, `drop_nan_col` all defined in `coral/src/frame.ch`; zero `_int` siblings |
| 9 | install_chelis_std.sh, smoke_macos_accelerate.sh ported | **PASS** — `nautilus/scripts/install_chelis_std.py` and `chelis/.github/scripts/smoke_macos_accelerate.py` exist; original `.sh` files removed (only ROCm wheel `.venv` artifacts remain, correctly skipped by walker) |
| 10 | kebab spec/design | **PASS** — zero kebab `.md` in `chelis/spec/design/` |
| 11 | octant red-team kebab | **PASS** — six `red_team_*.md` files in `octant/docs/`; zero `red-team-*.md` |
| 14 | phase identifier case | **PASS** in filenames; **branch refs in `.git/refs/heads/feat/phaseA-*` survive** but `.git/` is correctly skipped by the walker (and §9.2 says branches use kebab, not snake — so `phaseA` in branch names was the violation; the branches still exist as historical refs but aren't lint-checked) |

Snapshot §8 #12, #13, #15 declared out-of-scope per the brief; not verified.

Probe 7 verdict: every in-scope item resolves matches its claim. PASS.

---

## Severity classification of FAIL/CONDITIONAL findings

### CRITICAL — none

No rule documented in §§2-10 is completely unenforced where the spec promises
enforcement, and no cross-shell escape breaks builds.

### HIGH

**H1. §6.3 sibling-sweep coverage gap.**
The `module-pascal-components` rule fires only when a component has a
≥7-character lowercase run after the leading capital. Compound names with a
shorter second component slip past:

- `Nautilus.Linalg` (5-char run "inalg") — not flagged
- `Coral.Groupby` (6-char run "roupby") — not flagged
- `Std.Nn.Rmsnorm` (6-char run "msnorm") — not flagged

These are documented compounds in the spec (`Nautilus.LinAlg`, `Coral.GroupBy`,
`Std.Nn.RmsNorm` appear as positive examples in §6.1, §6.3, §7.1). The inverse
(lowercase second word) is a real §6.3 violation that the lint will not
catch. The rule should either (a) lower the threshold to e.g. 4 lowercase
chars after the cap, with the `KNOWN_SINGLE_WORDS` allowlist absorbing the
real single-word components, or (b) flip to a "compound-detection-by-known-
parts" approach.

**H2. CI gate not switched to `--check`; post-sweep lint not green.**
`chelis/.github/workflows/ci.yml:42-49` runs `chelis lint .` informationally
with a comment that promises the switch to `--check` "once Wave A lands."
Wave A is documented as landed, but `chelis lint --check` against the post-
sweep tree returns exit 1 with 26 violations on chelis, 6 on nautilus, 48 on
coral. Until those 80 prefix-namespace findings are either (i) reduced to
zero by §7.1 renames, (ii) absorbed into §7.1.1, or (iii) waived through
the exception list with cross-refs, the gate cannot be switched. The
documented "the lint is the persistent artifact: it prevents drift after a
cleanup pass" property in §12 is unrealized.

**H3. Nautilus/Coral/Shoals/Octant CI does not invoke `chelis lint`.**
Only chelis runs the lint at all. The settled-conventions are advertised
as "ecosystem-wide" (§12: "CI runs it as a gate on every PR") but the four
shell repos have no `chelis lint` step in their workflows. Drift in any
shell repo's `.ch` modules, helper-prefix patterns, or doc filenames is
not caught at PR time.

### MEDIUM

**M1. Exception list cross-ref validation not invoked at runtime.**
`exceptions::verify_cross_refs` is well-tested in unit tests but is never
called from `cmd_lint` in `chelis-cli/src/main.rs`. Today the runtime
exception list is `Vec::new()` so no harm; the moment anyone adds an
exception with a typo (`§8.6` instead of `§8.5`) or fabricated section
(`§99`), the lint will silently honor it. The §12 promise ("entries with
unresolvable cross_refs are build-time errors") would need
`verify_cross_refs(&exceptions, include_str!("…01-nomenclature.md"))` at
startup, returning a hard error if not Ok. Today this protection lives only
in tests of the validator function itself.

**M2. Probe 1 §8.1 conditional dispatch.**
A non-numbered file in `chelis/spec/` (e.g. `chelis/spec/foo.md`) silently
classifies as `Slot::SpecDesign` and passes the snake_case check. The
brief's literal expectation was that §8.1 fires; the rule's actual behavior
is shape-based dispatch. This is internally consistent but a spec reader
might assume `chelis/spec/foo.md` is rejected because §8.1 says "numeric
prefix + kebab-case for the top-level numbered language specs." If the
intent is that ALL files in `chelis/spec/` (top level only) must be
numbered+kebab, the rule needs a numeric-prefix-required gate for files
directly in `chelis/spec/`. Today, the rule allows snake_case as an
alternative form, which the spec does not authorize.

### LOW

**L1. CI workflow filename rule absent.**
§2.10 says CI workflow filenames are "lowercase, no separator." There is no
rule for `.github/workflows/*.yml` filename shape. Today this isn't broken
(the existing files are `ci.yml`, `release.yml`), but adding `Foo_Bar.yml`
to a workflows directory would not be caught. Not critical because no
shell repo currently has a deviant workflow filename.

**L2. Hidden-config naming (§2.11) not enforced.**
SCREAMING_SNAKE for `.claude/skills/*/SKILL.md`, kebab for
`.claude/commands/*.md`, kebab for skill directories — no rule fires on
violations. `.claude/skills/Some_Skill/` would not be flagged.

**L3. Rust/Python identifier rules (§4, §5) and Surf identifier rules
(§3.1-3.4) have no lint enforcement.** Mostly clippy/rustfmt/PEP8 covered
externally, but the spec does not say "external tools enforce this," it says
the lint is the persistent artifact. A docstring noting the external
toolchain coverage would close the gap.

**L4. Rendered mdBook HTML drift.**
`nautilus/docs/book/solvers/ode.html` and friends still contain `Nautilus.ODE`
literal strings even though the source `.md` and source `.ch` use
`Nautilus.Ode`. This is rendered build output, not a source regression.
Either the `book/` directory should be `.gitignore`d or it should be
regenerated. Not a lint concern, advisory only.

---

## Verdict

**Cleared for merge with HIGH-severity caveats; blocked on 3 issues.**

The lint mechanics are sound: every cleanup commit is locked by an
appropriate rule (Probe 2 7/7 PASS), the closed-vocabulary §11.1 guard works,
the §7.1.1 model-namespace allowlist works for both accept (bs_, mc_, gbm_,
fd_, lm_, cg_, airy_, beta_, chi_, det_, eig_, inv_, erf_) and reject (xyz_)
cases, and the mdBook §8.5 exception is correctly path-scoped.

The cleanup work landed and is locked against direct regression — Probes 1,
2, 3, and 7 substantially PASS. But three HIGH issues block "lint as
persistent post-sweep artifact":

1. **H1**: §6.3 sibling-sweep misses 5-6 char lowercase runs (`Linalg`,
   `Groupby`, `Rmsnorm`).
2. **H2**: post-sweep tree has 80 unresolved `prefix-namespace` findings; CI
   cannot switch to `--check`; the gate the §12 enforcement model relies on
   is still informational.
3. **H3**: only chelis CI runs the lint; the four shell repos have no lint
   gate in their workflows, undermining the "ecosystem-wide" §12 claim.

Plus two MEDIUM issues for the orchestrator's queue: M1 (runtime
verify_cross_refs not invoked) and M2 (§8.1 lets snake_case through silently
in `chelis/spec/`).

Recommend the orchestrator route H1, H2, H3 back to the appropriate Wave's
agents before declaring Phase 4 cleared. Probe 2's 7/7 lock is real: the
sweep itself is durable. The work that remains is the lint's coverage and
the gate's CI integration, both of which are spec invariants the current
state does not satisfy.
