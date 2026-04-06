## Phase 1f: Executable Grammar (`chelis validate`)

**Goal:** Extract PEG grammars from spec prose into a standalone validation tool.

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
3. The Deep validator additionally checks: tag vocabulary (only 56 valid tags), 3-tuple structure (every node has `{}`), arity rules (correct number of children per tag)
4. Wire into the CLI as the `validate` subcommand

**This is a conformance tool, not a replacement for the parser.** The full compiler's parser is hand-written (Pratt + recursive descent). The `pest`-based validator is an independent second implementation derived from the spec grammar. If they disagree about whether a program is valid, that's a bug in one of them — and the disagreement is the valuable finding.

### Test Strategy (~8 tests)

- [ ] Every example in SKILL.md validates (both Surf and Deep)
- [ ] Every example in spec docs validates
- [ ] Known invalid programs are rejected with clear messages
- [ ] Validator and compiler parser agree on 100% of the spec test suite programs
- [ ] Deep validator rejects: unknown tags, missing `{}` metadata, wrong arity
- [ ] Surf validator rejects: unknown keywords, malformed operators, missing delimiters
- [ ] `--desugar` mode: desugared output passes Deep validation

### Execution Strategy

```
Commit 1: Extract PEG grammars into .pest files
Commit 2: Deep validator (parse + tag vocabulary + arity)
Commit 3: Surf validator
Commit 4: Desugar mode
Commit 5: Conformance tests — validator vs compiler on full spec test suite
```
