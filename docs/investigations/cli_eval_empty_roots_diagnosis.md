# `chelis eval` empty-result diagnosis (G7 CLI, chelis#639)

This note covers the CLI-side half of the G7 silent-no-output bug. The
IR-lowering half (`match` lowering) remains a separate workstream. The
original fix added a text-mode warning; chelis#639 closes the later JSON
asymmetry without changing evaluation semantics or the JSON wire document.

Cross-references:

- Sweep findings: `docs/investigations/item2_sibling_sweep_findings.md`
  section **G7**, sub-bug 2 (eval-side silent no-output).
- Direct-file pins: `crates/chelis-cli/tests/cli.rs`
  `eval_def_only_emits_warning_on_stderr` and
  `eval_json_def_only_emits_empty_roots_json`.
- Reef-context pin: `crates/chelis-cli/tests/eval_in_reef_context.rs`
  `cmd_eval_json_reef_package_def_only_warns_with_wire_compatible_stdout`.

## Bug surface

A Surf input containing only `def` declarations and no top-level evaluable
expression produces an `EvalResult` whose `roots` and `transcript` vectors
are both empty. Human formatting produces an empty string. JSON serialization
produces the valid document `{"roots":[]}` because an empty transcript is
omitted.

The original G7 fix made human mode emit this stderr breadcrumb while keeping
empty stdout and exit `0`:

```text
warning: input contains only def declarations; nothing to evaluate
```

JSON mode initially skipped that warning. A direct file or a file routed
through Reef-context evaluation therefore exited `0`, wrote
`{"roots":[]}\n` to stdout, and left stderr empty. That asymmetry hid the
same authoring mistake from notebook and scripted operators and contributed
to the package-evaluation misdiagnosis in chelis#636.

Nullary function declarations remaining non-evaluable is intentional. The
bug is the missing operator diagnostic, not root selection.

## Emission sites

The CLI has two successful JSON-result emitters:

1. `run_eval_json_emit` serves direct files, Deep files, fallback evaluation,
   and inline expressions.
2. The JSON arm of `run_eval_in_context` serves files routed through the
   Reef-context fast path.

Both serialize and print the original `EvalResult`. After successful
serialization, both apply one shared structured predicate:

```text
result.roots.is_empty() && result.transcript.is_empty()
```

Only that shape calls the existing `warn_eval_no_roots` stderr helper. This
keeps a transcript-only result observable and warning-free, and it prevents
evaluation or serialization failures from being replaced by the warning.

## Chosen behavior: stderr warning, exit 0

For an empty successful result:

- Human mode keeps empty stdout, emits the warning once on stderr, and exits
  `0`.
- JSON mode writes exactly `{"roots":[]}\n` to stdout, emits the same warning
  once on stderr, and exits `0`.

Why this wins:

- Humans and operators get a visible explanation instead of silent success.
- JSON consumers still receive one byte-stable, parseable stdout document.
- The structurally valid input remains a successful evaluation rather than a
  compile error.
- Existing root selection, including nullary-function behavior, is unchanged.

A nonzero exit was rejected because the input is valid and existing consumers
rely on exit `0`. Adding a warning field to JSON was rejected because it would
change the machine-facing schema.

## Acceptance

- Empty roots plus an empty transcript emits the warning exactly once in both
  direct-file and Reef-context JSON dispatch.
- JSON stdout remains byte-exact `{"roots":[]}\n`; text stdout remains empty.
- Exit status remains `0`.
- Results with roots or transcript entries do not emit the warning.
- Evaluation failures remain nonzero with empty stdout and their original
  stderr diagnostic.
- Focused CLI tests, the repository local gate, and strict OpenSpec validation
  are the executable completion evidence.
