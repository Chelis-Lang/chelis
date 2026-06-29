# Tide: Interactive And Agent Mode

**Status:** Phase 0i REPL shipped; Phase 2e agent API + MCP shipped; Phase 2f LSP
server + VS Code-compatible extension scaffold shipped; Phase 2g `chelis cove`
single-file TUI shipped.
This document records the intended architecture so the rest of the docs stay aligned
while implementation catches up.

## 1. Scope

Phase 0i covers the first interactive developer workflow:

- `chelis tide` for a REPL
- `chelis deep` for Surf to Deep inspection
- `chelis surf` for canonical Deep to Surf decompilation on the default path
- `chelis fmt` for Surf formatting
- `chelis eval expr` for one-shot evaluation

Later phases extend Tide into language-server support and the Cove TUI.

## 2. Interactive Execution Strategy

The REPL and `chelis eval` use the IR evaluator in `chelis-ir/src/eval.rs`.
That means:

- parse input
- type-check it
- lower it to the RISC DAG
- evaluate the DAG directly

This avoids spawning an external C compiler for small interactive workloads.

For larger workloads, `chelis build` continues to use the production C backend.

## 3. Latency Policy

If interactive latency later becomes a problem, the escalation order is fixed:

1. IR evaluator
2. cached C artifacts
3. persistent compiler helper process
4. JIT only if the first three fail on measured workloads

This is a policy decision, not an open exploration track.

## 4. Command Semantics

### `chelis tide`

Launch an interactive REPL.
It should accept Deep expressions directly and may accept Surf input when the front-end
path is available.

### `chelis tide serve`

Launch the HTTP/JSON compiler service.
The shipped surface includes `/parse`, `/desugar`, `/check`, `/lower`, `/compile`,
`/eval`, `/grad`, `/validate`, `/decompile`, `/replace_function_body`, and
`/batch`.
The public contract uses explicit wire-model types rather than serialized compiler
internals.

### `chelis tide mcp`

Launch the MCP server on stdio.
The shipped MCP tool surface is:

- `chelis_check`
- `chelis_compile`
- `chelis_desugar`
- `chelis_decompile`
- `chelis_eval`
- `chelis_grad`
- `chelis_replace_function_body`
- `chelis_prove`
- `chelis_validate`

`chelis_replace_function_body` accepts Deep strings for
`{module, function_name, new_body}` and returns canonical Deep
`{changed_def_deep, module_deep}` only after the post-splice whole-module
check is clean. Parse, splice, type, effect, and linearity failures return the
normal structured failure envelope with `ok:false`, `stage`, and typed
diagnostics; failed replacements do not carry a replacement result.

`chelis_prove` runs the same property + derived producer-obligation
verification as the CLI `chelis prove` on the same module source — the
obligation collection, synthesis, assumption injection, and tiered
dispatch live in `chelis-prove` and are shared across both surfaces
(`opaque_invariants_rfc.md` D-PARITY). The MCP response carries the
derived `obligation` records and a summary `obligations` count alongside
the property result; a prove through tide is identical to the CLI on the
same module (locked by a cross-surface parity test).

**SMT proofs are over the reals.** A `proof_tier:"smt"` obligation or
property is discharged by the solver over the reals while runtime
arithmetic is IEEE floating-point; such artifacts carry
`arith_model:"real"`. No float-level soundness is claimed from a Tier B
proof. Admission policies that quote the composed opaque-invariant
guarantee quote this gap. See `spec/design/chelis_property_spec.md`
(Derived obligation records / Tier B SMT note) for the record schema.

### `chelis tide lsp`

Launch the Tide LSP server on stdio.
The shipped v1 surface is full-file recomputation on each open/change and provides:

- diagnostics
- completion
- hover
- go-to-definition
- `workspace/executeCommand` commands for Deep view and fitness status

The server only promises information the compiler can actually produce today.
For `.dp` files the shipped surface is diagnostics-first; the richer editor features are
centered on Surf.

### `chelis eval expr`

Evaluate a single expression through the evaluator fast path.
No C emission or external compiler process is involved.

### `chelis deep file.ch`

Show the canonical Deep form of Surf input.
Default output is width-aware pretty Deep; `--flat` preserves flat per-form output while
keeping top-level forms separated.

### `chelis surf file.dp`

Decompile Deep into formatter-canonical Surf on the default path.
`--verbose` is the explicit best-effort debug form.

### `chelis fmt file.ch`

Format Surf or Deep using the compiler-owned canonical style.
`--check` validates canonical formatting without rewriting the file.

## 5. Output Expectations

Interactive tooling should surface:

- values from evaluation
- inferred types where useful
- fitness-oriented diagnostics when programs fail to type-check

`chelis check` remains the machine-friendly command for explicit structured fitness
output.

## 6. Agreement Testing

Tide-related evaluation must agree numerically with the production backend.
The Phase 0i test strategy includes evaluator-agreement tests:

- evaluate via the IR evaluator
- evaluate via the C backend
- compare results across the shared spec test corpus

## 7. Later Tide Work

Phase 2 extends Tide beyond the REPL:

- 2e: HTTP / JSON compiler API plus MCP server for coding agents
- 2f: LSP support with diagnostics, hover, completion, go-to-definition, Surf/Deep
  visibility, and TextMate grammar for instant highlighting
- 2g: `chelis cove` terminal UI with live checking, Surf/Deep toggling, and tree-sitter
  grammar for incremental terminal highlighting
- batch interfaces for agent loops and corpus collection

## 8. Phase 2e Contract Notes

- `/eval` takes named input bindings and resolves them by `Load.name`; the evaluator
  itself remains unchanged.
- `/grad` is DAG-level in the shipped Phase 2e surface and returns differentiated DAG
  JSON plus node mappings, not a source-level differentiated Surf or Deep program.
- The 2e automated acceptance oracle is `cargo test -p chelis-tide --test api`.
- MCP-agent end-to-end validation remains a documented manual gate rather than part of
  the default workspace run.

## 9. Phase 2f Contract Notes

- `chelis tide lsp` is the shipped stdio entrypoint; there is no separate `chelis-lsp`
  binary on the user-facing CLI surface.
- The v1 LSP does not depend on `salsa`; it recomputes from the full current document.
- The bundled VS Code-compatible extension lives in `editors/vscode/` and includes
  TextMate grammars for Surf and Deep so syntax highlighting works before the LSP is
  ready.
- The Deep toggle is a read-only command that shows canonical Deep; it does not attempt
  bidirectional Surf/Deep editing.
- Automated coverage for 2f is library-level: `cargo test -p chelis-lsp` exercises the
  analysis engine and command preparation. Full editor-host protocol behavior remains a
  manual gate rather than a claimed automated proof.
- The 2f manual acceptance oracle is opening a `.ch` file through the extension and
  verifying immediate syntax highlighting, diagnostics, hover, completion, definition
  lookup, Deep view, and fitness status in one session.

## 10. Phase 2g Contract Notes

- `chelis cove` is the shipped TUI entrypoint; `chelis cove --file examples/mnist.ch`
  opens a specific file.
- The shipped v1 is single-file and direct-library: it calls `chelis-tide::compiler`
  helpers in-process rather than talking to a background daemon.
- The shipped pane layout is Surf editor, read-only Deep view, diagnostics/fitness, and
  output.
- The output pane supports compile-preview and evaluator execution with auto-generated
  zero-filled named bindings for `Load` nodes whose shapes are known.
- The bundled tree-sitter grammars live in `grammars/tree-sitter-chelis-surf/` and
  `grammars/tree-sitter-chelis-deep/`; Cove uses them for Surf and Deep highlighting.
- Agent-mode hosting inside Cove is deferred; the shipped 2g surface does not embed an
  MCP-driven assistant session.
- Automated coverage for 2g is non-UI only: the live pipeline helpers, zero-binding
  eval, file loading, and CLI surface are tested, but the terminal event loop and panel
  behavior remain manual-gate territory.
- The 2g manual acceptance oracle is:
  `cargo run -p chelis-cli -- cove --file examples/mnist.ch`
  and confirming that editing updates Deep/diagnostics live, `Ctrl-S` saves, and
  compile/eval actions populate the output pane.
