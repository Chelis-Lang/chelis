# Chelis Project Rules

## Quality Standards

### Spec-First Development
- Before writing implementation, write test stubs derived from the spec
- Every spec requirement must have a corresponding test BEFORE the code exists
- If the spec says "X is a type error," write `check_err(X)` before implementing the checker
- Phase is not done until every spec requirement has both a positive and negative test

### Negative Test Parity
- For every test that checks something WORKS, write a test that checks the corresponding FAILURE
- "add(tensor, tensor) succeeds" must be paired with "add(tensor, int) fails," "add(tensor[f32], tensor[bf16]) fails," "add with wrong arity fails"
- If you can't think of a negative test, you don't understand the spec well enough

### Don't Trust Green
- Passing tests prove the code matches the tests, NOT the spec
- After green CI, ask: "what spec requirements have no test?"
- Silent fallbacks (default values, empty error vecs, unwrap_or) are bug factories — audit them

### Phase Completion Criteria
- Do NOT claim a phase is done based on "N tests passing, clippy clean"
- A phase is done when: every spec requirement is tested, adversarial inputs are tried, and the red team finds fewer than 2 issues
- Budget for at least one real adversarial review cycle that finds issues

## Red Team Protocol

### Who Red Teams
- The red team agent must be FRESH — no shared context with the implementation agent
- It must read ONLY the spec and the code, NOT the plan or implementation notes
- It must EXECUTE tests, not just read source code

### What Red Teams Do
- Write and RUN adversarial test cases against the implementation
- Try inputs the spec says should fail — verify they actually fail
- Try inputs the spec says should pass — verify they actually pass  
- Check exact output values, not just "no errors"
- Construct the worst possible inputs: edge cases, type mismatches, empty inputs, huge inputs, wrong arities
- A report with zero findings is suspicious — send it back

### Red Team Prompt Template
```
You are an ADVERSARIAL TESTER. Your job is to FIND PROBLEMS.
DO NOT PLAN. Execute. Run tests. Report findings.
A report with zero findings means you didn't look hard enough.

Read the spec at [path]. Read the code at [path].
Write and RUN test cases that should fail per spec — verify they DO fail.
Write and RUN test cases that should pass per spec — verify they DO pass.
Use non-trivial input values. Check exact outputs.
```

## Commit Hygiene

### No AI Attribution
- commit-msg hook enforces: no Co-Authored-By, Authored-By, or Generated-with referencing any AI tool
- CI job `no-ai-authorship` checks all commits
- This covers Claude, Anthropic, OpenAI, Codex, ChatGPT, Copilot, and generic AI/LLM

### Commit Messages
- Describe WHAT changed and WHY, not HOW
- No "comprehensive," "robust," or other filler adjectives
- Include test counts only when they represent real coverage changes

## Agent Team Patterns

### Agents Enter Plan Mode
- Agents frequently enter plan mode instead of executing
- Always launch implementation agents with `mode: bypassPermissions`
- Include "DO NOT PLAN. Write code. Run tests." in agent prompts
- If an agent returns a plan instead of code, note it failed and do the work directly

### Agent Supervision
- Don't let agents run unsupervised on large tasks — they stall
- Decompose into focused pieces (one file, one feature)
- Monitor progress — if output file size stops growing, intervene
- Prefer doing focused work directly over launching agents for small fixes

## Chelis-Specific

### Deep AST Format
- Every node is a 3-tuple: `(tag {} children...)`
- `{}` is metadata map, always present at element[1]
- 56-tag closed vocabulary (see spec/03-deep-syntax.md §2)
- Function application: `(app {} (var {} f) (var {} x))`
- RISC primitives are built-in functions via `(var {} name)`, NOT tags

### Type System
- No implicit precision promotion — mixed precision is always a type error
- Named tensor dimensions match by name, not position
- No broadcasting — explicit expand required
- Integer literals default to int32, float literals to f32

### Build
```
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

### C Backend Prerequisites
- gcc, openblas-devel (libopenblas-dev on Ubuntu), valgrind (optional)
