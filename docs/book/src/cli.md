# CLI Workflow

Chelis ships one CLI with machine-facing and human-facing subcommands.

## Core Commands

- `chelis fmt` canonicalizes Surf or Deep source.
- `chelis lint` enforces naming and style conventions from `spec/01-nomenclature.md`.
- `chelis check` parses, desugars, type-checks, and reports fitness/errors.
  Accepts both Surf (`.ch`) and already-lowered Deep (`.dp`) inputs; a
  `.dp` skips desugaring and is type/effect/linearity-checked directly,
  emitting the same JSON report shape as the `.ch` path.
- `chelis deep` prints canonical Deep for a Surf program.
- `chelis surf` decompiles Deep back to Surf.
- `chelis eval` runs the host/runtime evaluator. `chelis eval --file`
  accepts both `.ch` and `.dp` inputs.
- `chelis test` discovers and runs Chelis-native Reef package tests.
- `chelis prove` discovers and runs executable properties.
- `chelis validate` runs the executable-grammar validator on the input.
- `chelis build` emits C or HIP source plus runtime artifacts and compile flags.
- `chelis tide` exposes the HTTP/MCP tooling surface.

## Lint Traversal Policy

Directory linting composes Chelis's shipped baseline exclusions with the nearest
ancestor `chelis-lint.toml`. Discovery resolves relative targets against the
invocation working directory first, so `chelis lint --check .` from a
subdirectory applies the same repository policy as the absolute spelling. Repository entries are strict, versioned TOML with
a gitignore-style pattern, a closed class, and a cross-reference resolving in
the policy's declared spec:

```toml
version = 1
spec = "spec/01-nomenclature.md"

[[exclude]]
pattern = "path/to/generated/"
class = "generated"
cross_ref = "§12.2"
```

The allowed classes are `infrastructure`, `build`, `dependency`, `generated`,
and `immutable`. Invalid policy fails lint before the walk. Non-file or broken
policy paths and policy or spec links resolving outside the policy root also
fail; internal links remain valid. `.gitignore`, `.ignore`, parent and global Git
ignores, `.git/info/exclude`, and hidden-file defaults do not affect lint
scope. A directly named file or directory remains lintable when only its
exclusion pattern would reject it; it must still be a regular file or directory
(or a link resolving to one) inside the policy root. A directly named root
that exists but fails that admission — a socket or FIFO, a link resolving to
a different entry kind, an escaping link, or a broken link — fails
`chelis lint` loudly with the root path and rejection reason, exactly like a
nonexistent root; it never exits 0 as an empty lint. A link-final target is
not resolved before the walk, so its identity reaches that boundary check.
Matching nested
descendants are pruned. Non-explicit discovered entries must be directories,
regular files, or symlinks resolving to the same entry kind. Special entries
such as sockets and FIFOs, aliases into an excluded tree, broken links, and
targets outside the policy root are omitted, while internal links to admitted
regular files retain their link-path surface. An explicitly named excluded directory
keeps its depth-zero override for internal link targets, but separately
excluded descendants still apply. Rules also apply this policy to ancillary
metadata: an excluded Cargo manifest or machine-local manifest above the
policy root cannot grant the §8.3 package-name exception to an admitted
documentation filename. A link path above the policy root remains machine-local
even when its target resolves to an admitted internal manifest. Admitted sibling
workspace manifests remain visible when lint targets a documentation
subdirectory or explicit file, including crates exposed through an internal
directory symlink. A crate directory link resolving outside the policy root is
rejected. Use rule-specific exceptions or inline
`allow`/`keep` when a path must still contribute to other lint rules.

## Style Gate (Built-In on Every Build)

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` enforce a **style gate** on the input file before
the front-end pipeline runs. The gate is two checks in one:

1. **Formatter check**: the file must be byte-identical to its
   canonical re-print (the same comparison `chelis fmt --check` does).
2. **Lint check**: the file must pass every `chelis lint` rule that
   applies to its surface in the blocking rule registry.

A failure prints a one-line-per-issue diagnostic to stderr and exits
non-zero. The error message tells you exactly what to run:

```text
error: app.ch: not canonically formatted; run `chelis fmt --inplace app.ch` to fix
       app.ch:7: surf-value-snake-case (§3.2): function `BadName` is not snake_case
2 issue(s) blocked the build; pass `--allow-style-violations` to bypass (CI must not).
```

The gate runs only on the user-supplied source. Reef-imported library
decls are not re-checked here. They were already gated when the
package was published.

Advisory lint rules are not part of the style gate. They may print as
warnings on user-facing commands, but they do not block `build`,
`check`, `validate`, `eval --file`, or `lint --check`.

### Bypass flags

| Surface | Bypass | When to use |
|---|---|---|
| `--allow-style-violations` (CLI flag) | Per-command opt-out; logs a stderr warning. | Emergency local builds and one-off migrations. **CI must not pass this flag.** |
| `CHELIS_STYLE_GATE_DISABLE=1` (env var) | Process-wide opt-out. | The integration-test corpus only, used by tests that synthesize ad-hoc Surf to exercise pipeline behavior independently of style. **Not for production builds or end-user scripts.** |

`chelis eval EXPR` (the inline-expression form) is unaffected. There
is no on-disk source to canonicalize, so the gate does not apply.

`--allow-style-violations` bypasses only the style gate. It does not
turn parse, type, effect, validation, evaluation, or backend errors into
warnings.

### Severity behavior

| Severity | Example | Output | Exit behavior |
|---|---|---|---|
| Blocking violation | non-canonical formatting, `surf-value-snake-case`, `no-em-dash-in-public-strings` | one issue per line | `lint --check` exits non-zero; built-in style gate blocks unless bypassed |
| Warning/advisory | `redundant-linearity-call`, `prefer-pipe-operator` | prefixed with `warning:` or `advisory:` | never makes `lint --check` fail and is excluded from the built-in style gate |

`redundant-linearity-call` warns on explicit `copy()` and `drop()`
calls. Existing fixtures and migration baselines may keep those calls
when they prove compatibility or preserve before/after evidence. New
human-facing examples should use implicit linearity unless the explicit
form is the subject of the example.

`chelis lint --fix <path>` applies available source rewrites in-place.
The warning-only `redundant-linearity-call` and `prefer-pipe-operator`
rules are diagnostic-only until their fixers have semantic proof that a
rewrite preserves ownership and call argument behavior.
`chelis lint --rules a,b <path>` runs a comma-separated subset, and
`chelis lint --list` prints the registered rules, severities, spec
references, and summaries.

## Typical Loop

```sh
chelis fmt --inplace app.ch
chelis lint --check app.ch

chelis check app.ch
chelis eval --file app.ch
chelis build app.ch --target c --output out/
```

Use this loop for project files and generated shell output. `fmt` makes the source
canonical, `lint --check` catches naming/style drift, and the later commands re-run the
same gate before doing semantic work.

When `chelis eval --file` runs from inside a Reef package root, ad hoc
snippet files can import package modules even if the snippet file
itself lives outside `src/` and does not declare a top-level `module`.

### Targeted evaluation and root manifests

`chelis eval --target eval|c|hip|metal` computes the root manifest against the
selected backend's capabilities. The default is `eval`. This is useful when a
program must be compared with a generated artifact: for example, an f64 root
can use the Tensor lane under `eval` while the same root routes through the
Host lane under `c`. An unknown target is an error; it never falls back to a
different capability set.

Every successful `chelis eval --json` response includes a `manifest` object:
the selected `target`, ordered `entries` (`name`, `lane`, and
`required_inputs`), and `requires_main`. The `roots` array has exactly the
selected manifest names in the same order. Tuple roots and statically fixed
ADT roots use dotted component names. If a lane cannot produce an owed root,
evaluation exits nonzero instead of returning a partial JSON document.

### Bounding a slow evaluation

Interactively, Ctrl-C stops a running `chelis eval` immediately. For
unattended and scripted runs — CI, agent harnesses, batch jobs — use
`--timeout`:

```sh
chelis eval --timeout 30 --file slow.ch
```

On trip the command prints

```text
error: evaluation timed out after 30s (--timeout)
```

to stderr and exits non-zero, so a mis-sized or accidentally quadratic
program fails loudly instead of being indistinguishable from one that is
still making progress. Without the flag there is no timeout; evaluation
runs to completion.

The timeout is cooperative: the evaluator checks for it at every node
visit, and the front end checks it at every phase boundary and at every
top-level declaration inside the passes that dominate a large compile
(chelis#930), so both a compile-bound and an evaluation-bound program
unwind cleanly rather than being killed mid-write.

If cooperative unwinding does complete, that is the whole message. If it
does not, a backstop terminates the process and says so:

```text
error: evaluation timed out after 30s (--timeout); cancellation did not complete within 5s, forced exit
```

The suffix is worth reading. It means the process was killed rather than
unwound, so destructors did not run and buffered output was not flushed.
The usual cause is a machine under heavy load, where the cooperative
unwind competes for CPU against a fixed wall-clock grace period.

That backstop is defence in depth, not the mechanism: the polling above is
not exhaustive — the style gate, Reef graph resolution, and lowering's
whole-program walk do not poll, and a compiler pass that genuinely wedges
would never reach a check point. `--timeout` promises an
unconditional loud failure, so the process-level stop remains.

The same cancellation mechanism is what makes `KeyboardInterrupt` work
promptly in the Python bindings (chelis#914, chelis#930); see
`bindings/python/README.md`.

## Native Test Loop

`chelis test` runs `def test_*()` functions in `tests/**/*.ch` files:

```sh
chelis test
chelis test tests/
chelis test tests/core.ch
chelis test tests/ --filter pricing --timeout 10 --suite-timeout 120 --batch-mode auto
chelis test tests/ --json --batch-mode file --jobs 1
chelis test tests_neg/ --expect neg --json
chelis test tests_blocked/ --expect blocked
```

Directory runs use `--batch-mode auto` by default: eligible files are compiled
as one suite batch so the fixed Reef context and test-source compile costs are
paid once. Files with top-level module-init bindings or top-level name
collisions use the per-file worker path.

The batch is one compilation unit with one top-level scope, so a name
collision is any way two batched files can disagree about what a name means:
both declaring it, one declaring what another explicitly imported (in either
order), both importing it from different modules, or a wildcard import whose
names cannot be enumerated without resolving the package graph. Importing the
same name from the same module is not a collision. Demotion is not reported by
default, because the per-file path gives the same rows and the same exit code.
Set `CHELIS_TEST_EXPLAIN_BATCHING=1` to print one stderr line per demoted file
naming its reason, which is what to reach for when a suite has quietly lost the
batch path and you need to know which names to rename.

If a batch worker cannot be started, crashes, times out, or exits without
usable rows, the parent falls back to per-file workers and says so on every
channel that a reader might be capturing: an attributed stderr note naming the
reason and every file the batch had claimed, a ` (batch abandoned: ran
per-file)` marker on the plain summary line, and under `--json` a
`batch_fallback` record before the rows plus a `batch_fallback` flag on the
summary. All three are absent when the batch completes, and a clean run's
summary line and summary record are unchanged. The exit code still tracks test
outcomes only, because every selected test still ran.

Use `--batch-mode file` to force per-file subprocess isolation while debugging.
`--jobs auto` caps worker concurrency on file-worker paths; pass `--jobs 1` for
serial file execution. Output remains stable in discovery order for both plain
text and NDJSON.

`--timeout` is a per-test budget (30 seconds by default).
`--suite-timeout` is an independent bound around the complete command,
including Reef/context preparation, workers, output collection, and
finalization (600 seconds by default). On suite expiry, Chelis terminates the
suite process group and exits `1`; it does not fall back to a second execution
mode. The value must be at least one second, and the termination grace is
inside the stated deadline. Plain output marks the suite incomplete. JSON output remains NDJSON,
retains completed test rows and completed `--expect` verdicts, and ends with a
`suite.status:"timeout"` record plus a mode-correct summary carrying
`incomplete:true`. Whole-suite supervision currently requires Unix process
groups (the supported Linux and macOS release targets); other targets fail
closed before starting `chelis test`.

The public `test` route cannot be converted into an unsupervised worker by an
environment variable. On Unix the suite is forked directly, with no hidden
suite-worker CLI route. A private lifecycle pipe removes progress state and
kills the suite process group if the supervisor disappears. If the suite
leader is killed, the public supervisor reaps its remaining process group
before collecting output and reports the suite as incomplete. Internally,
batch progress uses a supervisor-owned
temporary record file; stderr is forwarded byte-for-byte and is never used as
the progress protocol. Captured output is also forwarded within the
whole-command deadline: a consumer that stops reading causes exit `1` instead
of an unbounded write. The incomplete diagnostic uses a bounded one-second
best-effort reporting grace. Stderr is delivered before a normal stdout
summary, preventing a blocked diagnostic stream from leaving machine output
that appears perfectly successful.

`--expect neg|blocked` treats each discovered `.ch` file as one
expected-failure probe paired with a same-stem `.expect` sidecar. Sidecar line
1 is the required diagnostic substring; `blocked` sidecars must also contain
an auditable blocker citation such as `chelis#NNN`. A probe may express its
failure as either a failing `test_*` row or a bare file-level compile/check
diagnostic: the latter is preserved as the adapter input even when the file
declares no `test_*` function (chelis#967). A clean file with no `test_*` and
no compile/check failure is still a `config-error`, rather than an
expected-failure success. Plain and NDJSON modes emit one verdict per file
plus a mode-specific summary. A `wrong-diagnostic` or `drifted` NDJSON record
also carries the compiler messages in its `"got"` string array, so machine
consumers do not need a second plain-text run to diagnose the mismatch.

## Property Proof Loop

`chelis prove` runs first-class Surf properties and bridge-emitted Deep property
witnesses:

```sh
chelis prove
chelis prove properties/
chelis prove src/model.ch --only invariant_* --samples 1000 --seed 42
chelis prove references/vocabulary_miss.dp --spans references/vocabulary_miss.spans.json --json
```

With no path, discovery scans the current package's `properties/**/*.ch` and
`src/**/*.ch`, skipping `tests/`, hidden directories, `target/`, `dist/`, and
dependency trees. Explicit `.ch` inputs discover properties only in that file;
imports are for name resolution, not discovery. Explicit `.dp` inputs are
validated first, then scanned for canonical `chelis_role: "property"`
metadata. SMT-amenable scalar Deep properties lower directly to Tier B;
unsupported Deep property shapes follow the normal requested-tier policy.
Surf properties may also prove an applied single-target scalar gradient such
as `grad(price, wrt=rate)(args...)` when the differentiated lambda or pure
top-level function has `f32`/`f64` parameters, produces an `f32`/`f64` result,
and uses scalar arithmetic, blocks, and inlineable scalar helpers. These proofs
are still over the reals and carry the `real_arithmetic` qualifier. Tensor or
multi-target gradients, non-floating results, conditionals, casts in
differentiated bodies, effects, recursion, nested transforms, helper-depth
overflow, and unsupported differentiated operations return a specific
unsupported reason under `--tier smt-only`; `auto` may fuzz-validate them
instead. Conditional scalar gradients and differentiated casts remain outside
the prover subset until the compiler can build them.

Contract-backed Reef proofs resolve their implementation through the linker.
For example, a property importing `Nautilus.Stats.quantile_vec` can request
`std.quantile.monotonicity`; Tier B couples two calls only when their tensor
argument is the same compiler-bound dataset and reports
`Nautilus.Stats.quantile_vec` in the consumed assumption record. A local
function, a root declaration that imitates a linker name, two different
datasets, or a contract request with no trusted call is unsupported rather
than assumed. The quantile range and boundary contracts do not yet have this
source bridge.

Under `--tier fuzz-only` (and an `auto` fallback), guarded `f32`/`f64`
properties use deterministic constraint-directed generation for conjunctions
of scalar interval and binder-order comparisons. This keeps narrow finance
domains such as ordered VaR confidence levels non-vacuous. Explicit and
negative bounds are not clipped to the uniform generator's `[-10, 10]` range;
strict order spacing follows the binders' actual IEEE representation, and
non-strict orders may generate equal values. Unsupported, inconsistent, or
machine-unrepresentable scalar guard shapes exit `2` instead of producing a
green result. One-sided finite intervals synthesize their missing endpoint
inside the binder dtype's finite range, including near the `f32`/`f64`
extrema. This Tier-C behavior is identical in normal SMT and non-SMT CLI builds
and through Tide. Shared-runner user-property records on the CLI and Tide
always disclose `proof_tier` (`none` for a terminal outcome), and sampling
records additionally disclose `sampling_method` plus accepted, attempted, and
rejected sample counts; guarded records repeat this evidence in the
precondition non-vacuity record.

`--tier induction-only` proves a conservative class of general-size Surf
properties by separately dispatching a concrete base and symbolic step to SMT.
The compiler AST must show one `int*` binder with `n >= 0`, one scalar model
call, and an exact `if n <= 0 ... else ... f(n - 1, unchanged_args...)`
recurrence. Both cases and their non-vacuity checks must pass; the JSON record
uses `proof_tier:"induction"` and discloses both statuses under `induction`.
Unsupported recursion is terminal and never falls back to fuzz. Proofs retain
the `real_arithmetic` qualifier. The old caller-classified Tier-D scaffold is
still disconnected and fail-closed, and `ASSUMED` is never proof evidence.
Compiler-inlined transparent aliases are accepted when they expose that exact
recurrence. Literal-dead branches are retained in the solver goal rather than
discarded by the classifier. Deep properties have no structural-recursion
ownership record yet, so `induction-only` on Deep is terminal `unsupported`
with zero samples on both CLI and Tide.

Exit codes are stable for CI: `0` pass, `1` counterexample, `2` selected
property unsupported by the v1 generator, and `3` setup/input/config error.
`--json` emits NDJSON property records followed by one summary record.
For Reef-package Surf modules, that summary includes
`dependency_graph:{status:"complete",declarations:[...],edges:[...]}`. Nodes
carry stable compiler declaration IDs plus package/module and package-relative
source-span ownership; edges use those IDs and come from linker-resolved AST,
not downstream source parsing. A complete graph may be explicitly empty.
Bare Surf, Deep, a failed attribution pass, or any incomplete member of a
multi-input selection instead emits
`dependency_graph:{status:"unavailable",reason:"..."}`. The deprecated,
name-only `dependency_edges` field remains for one compatibility release.

## Reef Artifact Verification

Release workflows can validate a prebuilt Reef shell and its source archive
without installing either file:

```sh
chelis reef verify-artifact \
  --shell dist/example-1.2.3.chb \
  --archive dist/example-1.2.3.tar.zst
```

Verification strictly consumes the complete CHB, requires its bytes and
metadata ordering to be canonical, validates structural invariants across the
versioned `CHELCHB` envelope, including canonical quantified type-variable
restriction metadata, and checks the archive bytes against the CHB's embedded SHA-256.
Appended bytes, truncation, malformed metadata, and a mismatched archive fail
before any registry state is written. `reef install` uses this same verifier.

Pass `--json` for release automation. Stdout is JSON only, stderr is empty,
and exit status is zero exactly when `valid` is `true` and `errors` is empty.
The success report includes the package identity, compiler pin, and both
computed SHA-256 digests. This is content and structural validation, not
publisher authentication; release transport should separately pin or sign the
CHB digest.

## Output Contract

- `check` is machine-facing: perfect score implies an empty error list.
- `deep` defaults to canonical pretty output.
- `build` emits source and runtime artifacts; it does not invoke
  `gcc` or `hipcc` for you.
- `lint` prints `path:line:col: rule_id (§spec_ref): message`; with
  `--check` it exits non-zero on any blocking violation. Advisory
  warnings are prefixed with `warning:` and do not affect the exit code.
- `reef verify-artifact --json` emits one JSON document where
  `valid == errors.is_empty()` and exits non-zero for an invalid pair.

## Shell Author Checklist

- Generate Surf when humans will edit the result; generate Deep when a tool needs the
  canonical AST shape.
- Run `chelis fmt --inplace` before persisting generated `.ch` or `.dp` files.
- Run `chelis check` on every generated entry point before publishing a shell artifact.
- Treat `--allow-style-violations` as a local escape hatch, not part of a package build.
- In pipe-stage Surf, `x |> f(y)` means `f(x, y)`. Use
  `x |> fn (v) -> f(y, v)` when the piped value belongs later.

For exact CLI semantics, use the numbered specs plus the CLI
integration tests in the repo (notably
`crates/chelis-cli/tests/style_gate.rs` for the gate itself and
`crates/chelis-cli/tests/cli.rs` for end-to-end pipeline behavior).
