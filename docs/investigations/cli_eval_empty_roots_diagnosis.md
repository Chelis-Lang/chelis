# `chelis eval --file` silent-no-output diagnosis (G7 CLI)

Diagnosis pass for the CLI-side half of the G7 sub-bug surfaced by the
Item 2 sibling-sweep. The IR-lowering half (`match` lowering) is
reserved for a dedicated phase. This note covers the CLI half only.

Cross-references:

- Sweep findings: `docs/investigations/item2_sibling_sweep_findings.md`
  section **G7**, sub-bug 2 (eval-side silent no-output).
- Orchestrator plan: `.claude/plans/build-up-a-plan-mossy-meteor.md`
  (workstream context for the G7 CLI dispatch).
- Pinning test: `crates/chelis-cli/tests/cli.rs`
  `eval_def_only_emits_warning_on_stderr` (commit `5128031`).

No code changes in this commit.

> **Status update (chelis#639, 2026-07-21):** The text-mode warning described
> below has shipped. JSON mode now emits the same warning on stderr when both
> `roots` and `transcript` are empty, while preserving `{"roots":[]}\n` on
> stdout and exit `0`. Direct-file and Reef-context behavior is pinned in
> `crates/chelis-cli/tests/cli.rs` and
> `crates/chelis-cli/tests/eval_in_reef_context.rs`.

## Bug surface

A Surf input that contains only `def` declarations and no top-level
evaluable expression produces an `EvalResult` whose `roots` vector and
`transcript` vector are both empty. `format_eval_result`
(`crates/chelis-cli/src/main.rs` near L4457) returns the empty string
for that shape. Both `run_eval_emit` (L890) and `run_eval_in_context`
(L837) short-circuit on the empty formatted string and return `Ok(())`
with no print and no stderr breadcrumb.

End-to-end behavior on `main` today:

```
$ cat def_only.ch
def double(x: &tensor[3, f32]) -> tensor[3, f32] = x + x
def triple(x: &tensor[3, f32]) -> tensor[3, f32] = x + x + x

$ CHELIS_STYLE_GATE_DISABLE=1 chelis eval --file def_only.ch
$ echo $?
0
```

Both stdout and stderr are empty. A human interactively typing this
gets no signal that they wrote an unevaluable program; a CI job
piping stdout into a downstream consumer gets an empty stream that
looks like an evaluation produced no rows.

## Bug sites (both share the same shape)

1. `crates/chelis-cli/src/main.rs` `run_eval_emit` (L890): legacy
   eval path. Reached when `chelis eval --file <foo.ch>` runs outside
   a reef package, or inside a reef package that the Phase H context
   builder can't yet hash (`HashUnsupported` fallback).
2. `crates/chelis-cli/src/main.rs` `run_eval_in_context` (L837):
   Phase H eval-in-context path. Reached when `chelis eval --file
   <foo.ch>` runs inside a reef package whose graph the context
   builder can hash.

Both call `format_eval_result` and both short-circuit on an empty
formatted string. The fix needs to land on both arms or the bug
silently moves between them depending on package layout.

## Option survey

### (a) stderr warning, exit 0 — chosen

- Emit `warning: input contains only def declarations; nothing to
  evaluate` on stderr at each of the two empty-formatted-string
  short-circuits.
- Return `Ok(())` so scripted consumers that previously saw exit 0 +
  empty stdout keep seeing exit 0 + empty stdout, plus a stderr
  breadcrumb they can ignore.
- Style matches the two existing CLI warning sites
  (`eprintln!("warning: {...}")` near L1316 and L4574).

Why this wins:

- Humans get a visible signal: the silent footgun is fixed.
- Scripts that already pipe stdout downstream and ignore stderr
  keep working byte-for-byte on stdout.
- Cost is two `eprintln!` lines plus one test flip.

### (b) non-zero exit — rejected

- Would surface the issue louder but retroactively breaks any
  scripted consumer that today shells out `chelis eval --file
  <generated.ch>` and treats exit 0 as "no errors, proceed".
- The Surf input is structurally valid; it just doesn't produce a
  value. That's a UX gap, not a compile error. Conflating it with a
  hard failure penalizes the wrong consumers.

### (c) no change — rejected

- The silent-success-is-defensible reading ("the user wrote a library
  module, of course there are no roots") is operationally bad. The
  CLI subcommand is `eval`, not `check`; the user who typed `eval`
  asked for an evaluation, and getting nothing back without
  explanation is a footgun. The corpus of investigations already
  contains the sweep entry that calls this out as a bug.

## Acceptance

- Empty roots + empty transcript produces a stderr warning on both
  the legacy and Phase H paths.
- Exit code stays 0.
- Stdout stays empty on the def-only input (no spurious blank line,
  no transcript artifact).
- The pinning test
  `crates/chelis-cli/tests/cli.rs::eval_def_only_emits_warning_on_stderr`
  flips from `#[ignore]` to running and passes.
- The full workspace gate (`cargo test --workspace`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo fmt --all --
  --check`) stays green, and `chelis lint --check .` stays exit 0.
