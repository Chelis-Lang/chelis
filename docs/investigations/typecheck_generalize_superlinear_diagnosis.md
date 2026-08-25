# Type checking is superlinear in binding count — investigation

**Issue:** [chelis#1207](https://github.com/Chelis-Lang/chelis/issues/1207)

**Implementation plan:**
[`spec/design/typecheck_levels_generalization_plan.md`](../../spec/design/typecheck_levels_generalization_plan.md)

**Measured:** 2026-08-06 through 2026-08-10; implementation validated 2026-08-25

## Result

The pre-remediation `Env::generalize` applied the accumulated substitution to every
binding in scope three times for every generalization. The affected axis was
**binding count**, not definition count: local block bindings inside one definition
reproduced the same growth as top-level definitions.

All three user-facing lanes paid the cost because all three type-check before their
lane-specific work. At 200 generated bindings, `chelis check` was about 60 times
slower than `build` or `eval` and grew more steeply. `build` and `eval` nevertheless
showed approximately quadratic growth. The different `check` constant and exponent
remain unexplained and may conceal a second independent problem.

The original investigation went through two incorrect interpretations that this
record deliberately does not preserve as findings:

- A top-level-only corpus made the problem look like a definition-count problem.
  Local `lets` and `letsind` fixtures disproved that interpretation.
- A single-N lane comparison made `build` and `eval` look exempt. Per-lane scaling
  disproved that interpretation.

## Source mechanism

The pre-remediation `crates/chelis-types/src/env.rs` contained three separate
full-environment walks:

```rust
pub fn free_tvars(&self, subst: &Subst) -> HashSet<TypeVar> {
    let mut result = HashSet::new();
    for scheme in self.bindings.values() {
        let ty = subst.apply_scheme(scheme);
        // collect type variables
    }
    result
}

pub fn generalize(&self, ty: &Type, subst: &Subst) -> Scheme {
    let ty = subst.apply(ty);
    let env_tvars = self.free_tvars(subst);
    let env_dvars = self.free_dvars(subst);
    let env_rvars = self.free_rvars(subst);
    // quantify variables free in ty but not free in env
}
```

In the investigated source, `generalize` called all three collectors and each
collector applied the full substitution to every scheme in the environment. Seven
generalization sites covered sequential block bindings, local transformed
definitions, ordinary top-level definitions, implicit-generic signature/metadata
resolution, and recursive-group completion.

The resulting work has three measured factors:

1. the number of generalizations;
2. the number of environment bindings visited by each generalization; and
3. the cost of `apply_scheme` under the accumulated substitution.

That is why the affected fixtures grew faster than a simple quadratic while the
`flat` control stayed near linear.

## Binding-count corpus

N is the number of generated target bindings. Fixed module scaffolding, helper
definitions, entry definitions, and `out` bindings are excluded from N.

- `flat`: N top-level definitions with no inter-definition calls.
- `shallow`: N top-level definitions that each call one fixed helper.
- `lets`: N chained local bindings in one definition.
- `letsind`: N independent local bindings in one definition.

`chelis check`, min-of-three successful runs:

| shape | N=50 | N=100 | N=200 | per doubling |
|---|---:|---:|---:|---:|
| `flat` | 0.14 s | 0.18 s | 0.34 s | ×1.4, ×1.8 |
| `shallow` | 1.07 s | 4.71 s | 25.97 s | ×4.4, ×5.5 |
| `lets` | 0.82 s | 4.11 s | 25.72 s | ×5.0, ×6.3 |
| `letsind` | 0.84 s | 4.11 s | 25.84 s | ×4.9, ×6.3 |

The controls separate the claims:

- `lets ≈ shallow`: local and top-level bindings both paid.
- `letsind ≈ lets`: substitution chaining between the generated bindings is not
  required.
- `flat` stayed near linear despite the same environment growth because it created
  no inter-definition substitution workload.

The affected rows grow by about O(N^2.3) over this range. That fit describes the
observed corpus; it is not a language-level complexity guarantee.

## Split experiment

The load-bearing mechanism evidence holds total generated bindings at 200 and
changes only their layout: k definitions of m bindings, with `k × m = 200`.

| layout | 1×200 | 2×100 | 4×50 | 8×25 |
|---|---:|---:|---:|---:|
| measured | 25.72 s | 15.72 s | 11.00 s | 8.98 s |
| model | 25.72 | 15.6 | 10.8 | 9.1 |

A per-binding full-environment sweep predicts cost proportional to
`total × (G + k + m/2)`. With one fitted baseline term, `G ≈ 23`, the model is
within two percent for all four layouts. Splitting the body therefore reduces but
does not remove the cost: the baseline environment is still swept for every local
binding.

## Per-lane scaling

The same `lets` fixtures were run through every lane. Each timing was accepted only
after the command's success and output were checked; a fast error or stack overflow
was not recorded as a timing.

| N | `check` | `build` | `eval` |
|---:|---:|---:|---:|
| 200 | 26.50 s | 0.45 s | 0.38 s |
| 400 | 181.34 s (×6.8) | 1.23 s (×2.7) | 1.17 s (×3.1) |
| 800 | — | 5.22 s (×4.3) | 4.60 s (×3.9) |
| 1600 | — | stack overflow | — |

`build` and `eval` are approximately quadratic over the measured range. `check`
is closer to N^2.8 and has a much larger constant. Both routes eventually call
`analyze_prepared_with_library` and `complete_context_checks`. The gap is not
explained by monolithic versus layered checking: at N=200, measured `check` times
were 35.5 s inside a reef package and 26.5 s outside one.

## Real programs and shell evidence

`shoals/src/modelfit.ch`, a real module with 41 definitions and 433 lines, took
67.6 seconds to `chelis check` with the Shoals-pinned toolchain.

A separate 2026-08-10 report measured stock Chelis 0.18.4 in the same
linux/arm64 shell (2 CPU, 4 GB), using the same module, call shape, concrete 3×3
`f32` input, and roughly eight-line entry body:

| `Nautilus.LinAlg` entry | `chelis build` wall time |
|---|---:|
| `cholesky_n` | 7.1 s |
| `eig_n` | more than 600 s; killed at the timeout |

The 85-fold difference is consistent with importing a much larger library body and
therefore a larger binding population. It is evidence of the practical symptom, not
an isolated proof of this mechanism. Chelis#828 separately tracks `eig_n`'s runtime
cost; this report concerns the compile-time half.

## Relationship to chelis#1204 and chelis#1205

Chelis#1204 exposed the stall during its validation but did not create the three
environment sweeps. A 200-definition bare-reference fixture measured 0.72 s before
and 0.76 s after that change. Its call-graph walk can add separate overhead, but it
is a trigger and adjacent cost, not the cause recorded here.

Chelis#1205 contains two regimes:

- Its flat, named-binding regime belongs to chelis#1207.
- Its deeply nested expression regime does not. At 160 operations that fixture
  measured `check = 0.36 s` and `build = 6.52 s`; its build profile was in
  `chelis_ir::host::lower_*`.

Level-based generalization must not be claimed as a fix for the nested lowering
regime. The SCC/reachability pass (`function_inference_sccs`, `reaches_name`, and
`collect_top_level_calls`) is another separate optimization surface.

## Measurement limits

The measurements were taken on macOS arm64 except for the explicitly identified
shell result. Chelis#356 can wedge first execution and inflate absolute timings, so
ratios and shapes are more reliable than the constants.

Every accepted timing used an explicit success assertion. Two early apparent fast
runs were actually failures: a toolchain-pin rejection reported `total_nodes: 0`,
and a module-name mismatch exited before doing the intended work.

The original `sample` hot-leaf interpretation was retracted after its call-graph
self-time attribution could not be reproduced reliably. It is not used here. The
source walk, local/top-level controls, independent-binding control, split experiment,
and per-lane scaling carry the diagnosis.

## Implementation validation

The implementation replaces the three production sweeps with transactional solver
levels for type, dimension, rank, and precision-slot variables. Its retained sweep
implementation is compiled only by the `generalize-sweep-oracle` test feature. The
authoritative command on the repaired implementation tree was:

```sh
cargo nextest run --workspace \
  --ignore-default-filter \
  --features chelis-types/generalize-sweep-oracle \
  --no-fail-fast
```

It passed all 8,410 discovered tests with 222 skipped. Every observed production
generalization matched the retained sweep result exactly, and the independent-binding
structural test observed zero production environment-binding visits.

A fresh exact-head adversarial review found two cleanup gaps after the initial
implementation. First, resolving several deferred expand constraints could lower a
younger dimension variable before a later constraint rejected, leaving that level
change behind. Deferred resolution now runs as one cloned-solver transaction and
commits only when every constraint accepts; the negative regression also proves that
the rejected relation remains deferred. Second, the primary inference driver did not
poll cancellation between members of a recursive SCC. It now aborts the structured
recursive scope before returning a hard cancellation error, with an executable test
covering restoration of the parent level, recursion pins, authored prior bindings, and
temporary bindings. The 8,410-test result above includes both regressions.

The optimized exact-head checker was then run on generated, formatted fixtures. Each
timing was accepted only after asserting `score == 1` and an empty error list. The
table reports the minimum of three successful runs, matching the original corpus
method:

| shape | N=200 | N=400 | N=800 | per doubling |
|---|---:|---:|---:|---:|
| `flat` | 1.28 s | 1.37 s | 1.99 s | x1.1, x1.5 |
| `lets` | 1.26 s | 1.61 s | 3.33 s | x1.3, x2.1 |
| `letsind` | 1.74 s | 2.47 s | 5.22 s | x1.4, x2.1 |

The former x4.4 through x6.3 per-doubling binding-count curve is absent, and the
local-binding shapes now follow the same broad scaling regime as `flat`. A separate
successful N=800 lane comparison measured `check = 4.25 s`, `build = 7.99 s`, and
`eval = 7.34 s`; the former roughly 60-fold checker-lane gap did not survive this
change.

The maintained Shoals 0.18.5 bump supplied the current-language real-program corpus;
the repository's 0.14-pinned main checkout rejects legacy syntax before type
checking. The fastest of two completed successful optimized runs of
`src/modelfit.ch` was 50.86 seconds. The maintained bump also changes grammar and
dependency revisions, so this is not a controlled one-for-one comparison with the
historical 67.6-second result and does not support a percentage-improvement claim. It
does establish substantial residual latency, which is not attributed to levels.
Chelis#1316 tracks the independent SCC/reachability investigation, while chelis#1205
continues to own deeply nested lowering. No result here claims either surface was
fixed.

## Implemented remediation

The implementation uses eager, solver-local level-based generalization: it mints
inference variables at the current binding level, lowers a younger variable's level
whenever unification makes it reachable from an older one, and generalizes only
variables above the enclosing boundary. This replaces all three environment sweeps
together; a preliminary "collapse three sweeps into one" optimization is subsumed
rather than composed.

The algorithm follows Rémy's level formulation. Kiselyov's
[*Efficient and Insightful Generalization*](https://okmij.org/ftp/ML/generalization.html)
provides an executable exposition. Chelis must apply the rule to type, dimension,
rank, and precision-slot variables and preserve its existing deferred-shape and
recursive-instantiation exclusions. The detailed repository-specific contract is in
the linked implementation plan.

## Reproducing the generated corpus

The repository generator creates all four scaling shapes and the split experiment:

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/typecheck_generalization_fixtures.py \
  --output-dir /path/to/new-fixture-directory \
  --bindings 50 100 200 \
  --split-total 200 \
  --split-counts 1 2 4 8
```

It refuses nonpositive or duplicate counts, indivisible splits, and any existing
target file. Filenames and source order are deterministic. Its unit tests lock exact
small snapshots for every shape, target binding counts, split allocation, validation,
ordering, and overwrite refusal:

```sh
uv run --managed-python --python 3.11 --no-project python -m unittest discover \
  -s scripts -p 'test_typecheck_generalization_fixtures.py'
```

Format and check every generated file before timing it. A measurement runner must
also reject nonzero exits, reported diagnostics, `total_nodes: 0`, and stack
overflow rather than recording them as fast successes.
