## Phase 1f: Executable Grammar (`chelis validate`)

**Goal:** Extract PEG grammars from spec prose into a standalone validation tool.

**Authoritative oracle:**

```sh
cargo test -p chelis-e2e --test phase1f_validate
```

### What the Agent Builds

A new CLI subcommand using a PEG parser generator (e.g., `pest` crate) that validates Chelis syntax independently from the full compiler.

```
chelis validate --surf file.ch      # syntax-only validation against Surf PEG
chelis validate --deep file.dp      # syntax + tag vocabulary validation against Deep PEG
chelis validate --desugar file.ch   # parse Surf → desugar → validate Deep output
```

**Implementation approach:**

1. Extract the PEG grammars from `spec/02-surf-syntax.md` §4 and `spec/03-deep-syntax.md` into `pest` grammar files (`.pest`)
2. Build a `chelis-validate` crate that uses `pest` to parse against these grammars
3. The Deep validator additionally checks: tag vocabulary (only 61 valid tags), 3-tuple structure (every node has `{}`), arity rules (correct number of children per tag)
4. Wire into the CLI as the `validate` subcommand

**This is a conformance tool, not a replacement for the parser.** The full compiler's parser is hand-written (Pratt + recursive descent). The `pest`-based validator is an independent second implementation derived from the spec grammar. If they disagree about whether a program is valid, that's a bug in one of them — and the disagreement is the valuable finding.

### Current Implementation

- `crates/chelis-validate` owns the standalone `pest` grammars and validation entrypoints
- `chelis validate --surf file.ch` validates the shipped Surf surface, including script-style
  top-level bindings used by the executable examples, semicolon-separated block/par
  forms, and ordinary identifiers such as `axis` outside `vmap(..., axis=...)`
- `chelis validate --deep file.dp` validates Deep PEG structure plus the closed tag
  vocabulary, metadata-map requirement, arity/helper-form invariants, and dotted module/import
  path names emitted by canonical Deep
- `chelis validate --desugar file.ch` reuses the compiler Surf parser/desugarer and then
  validates the emitted canonical Deep
- the oracle suite checks validator/compiler agreement across:
  - executable examples in `examples/`
  - illustrative syntax examples in `examples/illustrative/`
  - `SKILL.md` Surf and Deep teaching blocks
  - curated positive spec fixtures
  - curated negative Surf and Deep fixtures

### Acceptance Notes

- This remains a conformance tool, not a parser replacement.
- The validator is intentionally strict about canonical Deep structure.
- Phase 1f shipping does not, by itself, declare all of Phase 1 complete; the broader
  phase gate remains the red-team checkpoint in the Phase 1 plan.

### Execution Strategy

```
Commit 1: Extract PEG grammars into .pest files
Commit 2: Deep validator (parse + tag vocabulary + arity)
Commit 3: Surf validator
Commit 4: Desugar mode
Commit 5: Conformance tests — validator vs compiler on full spec test suite
```
