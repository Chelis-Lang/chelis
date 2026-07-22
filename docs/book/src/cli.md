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
ancestor `chelis-lint.toml`. Repository entries are strict, versioned TOML with
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
and `immutable`. Invalid policy fails lint before the walk. `.gitignore`,
`.ignore`, parent and global Git ignores, `.git/info/exclude`, and hidden-file
defaults do not affect lint scope. A directly named file or directory remains
lintable even when its path matches an exclusion; matching nested descendants
are pruned. Use rule-specific exceptions or inline `allow`/`keep` when a path
must still contribute to other lint rules.

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

## Native Test Loop

`chelis test` runs `def test_*()` functions in `tests/**/*.ch` files:

```sh
chelis test
chelis test tests/
chelis test tests/core.ch
chelis test tests/ --filter pricing --timeout 10 --batch-mode auto
chelis test tests/ --json --batch-mode file --jobs 1
```

Directory runs use `--batch-mode auto` by default: eligible files are compiled
as one suite batch so the fixed Reef context and test-source compile costs are
paid once. Files with top-level module-init bindings or top-level name
collisions use the per-file worker path. If a batch worker crashes, times out,
or emits incomplete rows, the parent falls back to per-file workers.

Use `--batch-mode file` to force per-file subprocess isolation while debugging.
`--jobs auto` caps worker concurrency on file-worker paths; pass `--jobs 1` for
serial file execution. Output remains stable in discovery order for both plain
text and NDJSON.

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

Exit codes are stable for CI: `0` pass, `1` counterexample, `2` selected
property unsupported by the v1 generator, and `3` setup/input/config error.
`--json` emits NDJSON property records followed by one summary record.

## Output Contract

- `check` is machine-facing: perfect score implies an empty error list.
- `deep` defaults to canonical pretty output.
- `build` emits source and runtime artifacts; it does not invoke
  `gcc` or `hipcc` for you.
- `lint` prints `path:line:col: rule_id (§spec_ref): message`; with
  `--check` it exits non-zero on any blocking violation. Advisory
  warnings are prefixed with `warning:` and do not affect the exit code.

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
