# tide

## Purpose

Define Tide, the interactive and agent-mode surface: the evaluator-first interactive-execution
strategy and the fixed latency-escalation policy, the interactive/decompile/format command
semantics, the HTTP/JSON and MCP agent services with their structured wire and edit contracts
(whole-module-check gate, preimage guard, cascade completeness), prove parity and the SMT
real-arithmetic disclosure, and evaluator-agreement testing. This is the current truth of
Chelis's interactive and agent tooling.

## Requirements

### Requirement: Interactive execution strategy

The REPL and `chelis eval` SHALL use the IR evaluator: parse, type-check, lower to the RISC DAG,
and evaluate the DAG directly, avoiding an external C compiler for small interactive workloads.
Larger workloads SHALL continue to use the production C backend via `chelis build`.

#### Scenario: eval uses the evaluator fast path

- **WHEN** `chelis eval expr` is run
- **THEN** it evaluates through the IR evaluator with no C emission or external compiler process

#### Scenario: Build path is separate from interactive eval

- **WHEN** a large workload is compiled
- **THEN** it uses `chelis build`'s production C backend rather than the interactive evaluator

### Requirement: Fixed latency-escalation policy

If interactive latency becomes a problem, the escalation order SHALL be fixed: IR evaluator,
then cached C artifacts, then a persistent compiler helper process, then JIT only if the first
three fail on measured workloads. This SHALL be a policy decision, not an open exploration track.

#### Scenario: Escalation follows the fixed order

- **WHEN** interactive latency needs improvement
- **THEN** the response escalates in the fixed order (evaluator → cached C → helper process → JIT)

#### Scenario: JIT is the last resort

- **WHEN** JIT is considered
- **THEN** it is adopted only after the first three tiers fail on measured workloads, not speculatively

### Requirement: Interactive and inspection command semantics

`chelis tide` SHALL launch a REPL accepting Deep (and Surf where available); `chelis deep
file.ch` SHALL show canonical Deep (width-aware by default, `--flat` for per-form output);
`chelis surf file.dp` SHALL decompile to formatter-canonical Surf (`--verbose` for best-effort
debug); and `chelis fmt` SHALL apply the canonical style, with `--check` validating without
rewriting.

#### Scenario: deep shows canonical Deep

- **WHEN** `chelis deep file.ch` runs
- **THEN** it prints width-aware canonical Deep, or flat per-form output under `--flat`

#### Scenario: fmt --check does not rewrite

- **WHEN** `chelis fmt --check file.ch` runs on non-canonical source
- **THEN** it reports the formatting failure without modifying the file

### Requirement: Agent service structured edit contract

`chelis tide serve` and `chelis tide mcp` SHALL expose the compiler over HTTP/JSON and MCP using
explicit wire-model types rather than serialized compiler internals. Structural edit tools
(`replace_function_body`, `add_function`) SHALL return a canonical-Deep result only after the
post-edit whole-module check is clean; parse/splice/type/effect/linearity failures SHALL return
a structured `ok:false` failure envelope with no result payload.

#### Scenario: Clean edit returns canonical Deep

- **WHEN** `chelis_replace_function_body` splices a valid body and the whole-module check passes
- **THEN** it returns canonical `{changed_def_deep, module_deep}`

#### Scenario: Failed edit returns a structured envelope, no result

- **WHEN** a spliced body fails the type/effect/linearity check
- **THEN** the response is `ok:false` with a stage and typed diagnostics and no replacement result

### Requirement: Preimage-guarded and cascade-checked edits

Deep authoring edits that include a stale preimage SHALL fail at `stage:"preimage"` with no
result payload, using `preimage_sha256` computed over canonical function `(def ...)` nodes.
`rename` and `change_signature` SHALL perform structural cascade-completeness checks over the
query substrate before reporting success, with whole-module validation as the final gate.

#### Scenario: Stale preimage rejects the edit

- **WHEN** an edit request carries a `preimage_sha256` that no longer matches the canonical def
- **THEN** it fails at `stage:"preimage"` with no result

#### Scenario: Rename cascade is checked before success

- **WHEN** `chelis_rename` runs
- **THEN** it verifies cascade completeness over the query substrate and gates on whole-module validation before reporting success

### Requirement: Prove parity and real-arithmetic disclosure

`chelis_prove` over MCP SHALL run the same property and derived producer-obligation verification
as the CLI `chelis prove` on the same module, returning the derived obligation records and a
summary count. An SMT-tier proof SHALL be discharged over the reals while runtime arithmetic is
IEEE floating-point, and such artifacts SHALL carry `arith_model:"real"` with no float-level
soundness claimed.

#### Scenario: MCP prove matches the CLI

- **WHEN** `chelis_prove` runs on a module through MCP
- **THEN** it returns the same obligations and result as the CLI `chelis prove` on that module

#### Scenario: SMT proof discloses the real-arithmetic gap

- **WHEN** an obligation is discharged at `proof_tier:"smt"`
- **THEN** it carries `arith_model:"real"` and claims no float-level soundness

### Requirement: Evaluator-agreement testing

Tide-related evaluation SHALL agree numerically with the production backend: the test strategy
SHALL evaluate via the IR evaluator, evaluate via the C backend, and compare results across the
shared spec test corpus.

#### Scenario: Evaluator matches the C backend on the corpus

- **WHEN** a corpus program is evaluated both via the IR evaluator and the C backend
- **THEN** the results agree numerically

#### Scenario: A divergence is a defect

- **WHEN** the IR evaluator and C backend disagree on a corpus program
- **THEN** it is a defect, because Tide evaluation must agree with the production backend
