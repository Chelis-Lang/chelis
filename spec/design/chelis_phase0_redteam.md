# Phase 0 Completion: Red Team Validation

## Your Task

Verify that Phase 0 of the Chelis compiler is complete, correct, and ready for Phase 1 (GPU backend). Phase 0 spans nine sub-phases (0a through 0i). Each produced specific deliverables. Your job is to test every claim, find what's actually broken, and produce a prioritized list of must-fix items before the project moves on.

This is adversarial. Assume things are broken until proven otherwise. Run the code. Read the output. Don't trust descriptions.

---

## Ground Rules

1. **Run everything.** Don't review code by reading it. Compile it. Execute the tests. Run the CLI commands. Feed bad input and see what happens.
2. **Check the claims against reality.** The project plan makes specific claims about what works. Verify each one.
3. **Report what you find, not what you expect.** If something passes that you expected to fail, say so. If something fails that should pass, say so.
4. **Classify every finding.** CRITICAL = blocks Phase 1. HIGH = should fix before Phase 1 but not blocking. MEDIUM = tech debt. LOW = cosmetic.

---

## Check 1: Does It Build?

```bash
cargo build --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

- [ ] `cargo build` succeeds with zero errors
- [ ] `cargo clippy` is clean (no warnings with `-D warnings`)
- [ ] `cargo fmt` reports no formatting diffs
- [ ] No unused dependencies in any Cargo.toml (`cargo machete` if available)

Report: compiler version, any warnings suppressed by `#[allow(...)]`, total build time.

## Check 2: Do All Tests Pass?

```bash
cargo test --workspace
cargo test --workspace -- --ignored  # if any ignored tests exist
```

- [ ] All tests pass (report total count)
- [ ] No `#[ignore]`d tests that should be running (check each — is the ignore justified?)
- [ ] No tests that pass by coincidence (e.g., assert against a hardcoded value that happens to match)
- [ ] Test count per crate:

| Crate | Expected minimum | Actual |
|---|---|---|
| chelis-deep | 15+ | |
| chelis-surf | 30+ | |
| chelis-types | 40+ | |
| chelis-ir | 30+ | |
| chelis-backend-c | 30+ | |
| chelis-e2e | 20+ | |
| chelis-cli | 5+ | |

If any crate is significantly below the expected minimum, flag it.

## Check 3: Phase 0a — Scaffold

- [ ] README.md exists and has: project description, build instructions, links to spec docs
- [ ] `docs/book/` exists and covers developer-facing repo usage
- [ ] `spec/` directory contains: 02-surf-syntax.md, 03-deep-syntax.md, 04-type-system.md, 05-risc-primitives.md
- [ ] CI exists (GitHub Actions or equivalent) and is green
- [ ] `packages/chelis-std/SKILL.md` exists and `skill_suite.rs` validates its examples

## Check 4: Phase 0b — Deep Parser

```bash
cargo test -p chelis-deep
```

- [ ] Can parse all 56 tags in the vocabulary
- [ ] Round-trip: parse → print → parse produces identical AST
- [ ] Rejects malformed s-expressions with clear errors
- [ ] Handles nested block comments `{- {- ... -} -}`
- [ ] The canonical printer produces deterministic output (same input → same output, always)

Test: take every Deep code block from SKILL.md, parse each one, verify it round-trips.

## Check 5: Phase 0c — Surf Parser + Desugarer

```bash
cargo test -p chelis-surf
```

- [ ] Parses all example programs from `spec/02-surf-syntax.md` §8
- [ ] Parses `examples/mnist.ch` without errors
- [ ] Desugaring produces valid Deep (parse the Deep output, verify it validates)
- [ ] Operator precedence: `a + b * c` parses as `a + (b * c)`, not `(a + b) * c`
- [ ] Pipe precedence: `x |> f |> g` parses as `(pipe x f g)`
- [ ] Match arms use `=>` (not `->`)
- [ ] Short block bindings work: `{ x = 1; x + 1 }`
- [ ] `let` and `in` are valid identifiers after keyword removal: `{ let = 1; in = 2; add(let, in) }`
- [ ] Reserved keywords exclude `let` and `in`; using `def` as a variable name still errors
- [ ] Trailing commas accepted in parameter lists, arguments, record fields, imports

Test: write 5 intentionally malformed Surf programs and verify each produces a parse error (not a panic or silent success).

## Check 6: Phase 0d — Type Checker

```bash
cargo test -p chelis-types
```

- [ ] HM inference works: `def f(x, y) = add(x, y)` infers tensor types
- [ ] Named dimensions are nominal: `tensor[batch, f32]` and `tensor[seq, f32]` do NOT unify
- [ ] Precision mismatch is an error: `add(tensor[n, f32], tensor[n, bf16])` → type error
- [ ] No implicit broadcasting: tensors with different dimension counts → error
- [ ] ADTs with exhaustive matching: missing a variant in `match` → error
- [ ] Fitness scoring: a correct program → fitness 1.0, a partially broken program → 0 < fitness < 1
- [ ] All Tier 1 built-ins (RISC primitives) have correct type signatures in the environment
- [ ] All Tier 2 built-ins (relu, sigmoid, softmax, matmul, etc.) are recognized
- [ ] `cmplt` and comparison ops return bool tensors, not numeric tensors
- [ ] Reduction ops (sum, mean, max_reduce, softmax) require an explicit axis argument

Test: feed 5 programs with known type errors and verify each produces the correct error type (DimensionMismatch, PrecisionMismatch, UnboundVariable, ArityMismatch, NonExhaustiveMatch).

## Check 7: Phase 0e — IR Lowering

```bash
cargo test -p chelis-ir
```

- [ ] All Tier 1 ops produce correct DAG nodes
- [ ] Tier 2 ops decompose to RISC primitives (relu → MaxElem + Const(0), etc.)
- [ ] `matmul(A, B)` decomposes to expand + mul + sum subgraph
- [ ] `softmax(x, axis)` decomposes correctly (exp, sum, div)
- [ ] DAG verifier (`verify.rs`) catches: cycles, type mismatches, duplicate Load names with different types
- [ ] DAG is topologically ordered by construction
- [ ] DCE removes dead nodes
- [ ] CSE deduplicates identical subgraphs
- [ ] Constant folding: `add(const(1), const(2))` → `const(3)`

Test: build a DAG with a known dead branch, verify DCE removes it. Build a DAG with duplicate subexpressions, verify CSE merges them.

## Check 8: Phase 0f — C Backend

```bash
cargo test -p chelis-backend-c
```

- [ ] Generated C compiles with `gcc -O2 -fopenmp -lopenblas -lm`
- [ ] Stride-aware indexing on all elementwise ops (not flat `data[i]`)
- [ ] `#pragma omp parallel for` appears on elementwise loops
- [ ] `cmplt` emits `1.0f / 0.0f`, not C bool
- [ ] Reduction ops use `#pragma omp parallel for` on outer loop
- [ ] `reshape` after `permute` calls contiguity check
- [ ] `expand` sets stride to 0
- [ ] BLAS pattern match: `matmul` DAG → `cblas_sgemm` in generated C
- [ ] BLAS is opt-in (not unconditionally in link flags)
- [ ] `CHELIS_F32` defines, not magic numbers
- [ ] Load nodes use name-based mapping (not positional)
- [ ] Evaluator and C backend produce identical numerical results (within 1e-6) on at least 5 test DAGs

Test: generate C for `add(const(1), const(2))`, compile, run, verify output is 3.0. Repeat for mul, neg, relu, exp, sum, matmul.

## Check 9: Phase 0g — Automatic Differentiation

```bash
cargo test -p chelis-ir --test '*grad*'  # or however grad tests are organized
```

- [ ] Every differentiable Tier 1 op has a finite-difference verified adjoint
- [ ] `Log` adjoint is `Div(g, x)` (not the roundabout `Exp(Neg(Log(...)))` form)
- [ ] `Sqrt` adjoint is `Div(g, Mul(Const(2), Sqrt(x)))` (reuses forward node)
- [ ] `MaxReduce` has a real subgradient (not zero gradient — needed for softmax backward)
- [ ] Gradient accumulation: `mul(x, x)` → gradient is `2x` (both uses of `x` contribute)
- [ ] Second-order: `grad(grad(x^2))` → 2 (constant)
- [ ] Tensor-shape adjoints: `sum(x, axis=0)` on 2×3 tensor → gradient is all-ones 2×3
- [ ] `expand` adjoint is `sum` over the expanded axis
- [ ] `reshape` adjoint reshapes back to original shape
- [ ] `permute` adjoint uses inverse permutation
- [ ] Non-differentiable ops (`cmplt`, `const`) produce zero gradient without error

Test: build the softmax → log → mul → neg → sum chain (cross-entropy loss), differentiate, verify gradient matches `softmax(logits) - labels` numerically.

## Check 10: Phase 0h — MNIST End-to-End

- [ ] `examples/mnist.ch` exists and contains the MLP model in Surf
- [ ] `examples/mnist.ch` parses through the full pipeline: Surf → Deep → type check → lower → DAG
- [ ] Tier 2 built-in recognition smoke test: `relu`, `softmax`, `matmul` in mnist.ch are recognized as built-ins during lowering (not treated as unknown functions)
- [ ] IDX data loader reads real MNIST files correctly (60k train, 10k test)
- [ ] Training runs: loss decreases over epochs
- [ ] **Test accuracy >90% on real MNIST after 5+ epochs** (the actual milestone)
- [ ] Gradients match finite-difference spot check on the MNIST model
- [ ] Evaluator and C backend agree on one batch of MNIST forward pass

If MNIST data isn't present, the test should be `#[ignore]` with a clear message about where to get it — but you should download it and run the full test.

## Check 11: Phase 0i — CLI + Tide

Test every CLI subcommand:

```bash
chelis deep examples/mnist.ch          # should print canonical Deep
chelis check examples/mnist.ch         # should print fitness report JSON
chelis eval "mul(x, x)"               # or whatever simple expression works
chelis build examples/mnist.ch         # should produce .c + .h files
chelis fmt examples/mnist.ch           # should be idempotent
chelis surf <(chelis deep examples/mnist.ch)  # Deep → Surf decompilation
chelis tide                            # REPL — type an expression, get a result
```

- [ ] `chelis deep` produces valid canonical Deep (re-parse the output)
- [ ] `chelis check` produces JSON with fitness score and errors array
- [ ] `chelis eval` evaluates and prints a result
- [ ] `chelis build` produces compilable C
- [ ] `chelis fmt` is idempotent: `chelis fmt file.ch && chelis fmt file.ch` produces no diff
- [ ] `chelis surf` produces valid Surf (re-parse the output)
- [ ] `chelis tide` accepts input and produces output (basic smoke test)

Negative tests:
- [ ] `chelis deep nonexistent.ch` → clear error message, non-zero exit code
- [ ] `chelis check` on a program with type errors → fitness < 1.0, structured errors in JSON
- [ ] `chelis eval` on a type-incorrect expression → error message, not panic

## Check 12: SKILL.md Validation

```bash
cargo test -p chelis-e2e --test skill_suite
```

- [ ] All Surf examples in SKILL.md parse and type-check with fitness ≥ 0.9
- [ ] All Deep examples in SKILL.md parse and validate
- [ ] The eval harness (`scripts/skill_eval.py`) exists and is documented
- [ ] The 9/10 result is reproducible (or at least the automated `skill_suite` passes)

## Check 13: Spec Consistency

Cross-reference the implementation against the spec documents:

- [ ] The Deep parser accepts all 56 tags listed in `spec/03-deep-syntax.md`
- [ ] The Deep parser rejects tags NOT in the 56-tag vocabulary
- [ ] The type checker's built-in scope matches `spec/05-risc-primitives.md` §built-in-scope
- [ ] The desugaring table in `spec/02-surf-syntax.md` §5 matches the actual desugarer output for each construct
- [ ] The fitness scoring formula in `spec/04-type-system.md` matches the implementation
- [ ] `normalize` is NOT in the stable built-in scope (moved to unstable per SKILL.md revision)

Test: pick 5 entries from the desugaring table, write the Surf form, desugar, compare the Deep output against the spec. Report any discrepancies.

## Check 14: Architectural Discipline (Salsa Readiness)

The project plan requires pure-function crate boundaries for future salsa migration:

- [ ] `chelis_surf::parse` takes source string, returns AST. No mutable state.
- [ ] `chelis_surf::desugar` takes Surf AST, returns Deep AST. No mutable state.
- [ ] `chelis_types::infer` takes Deep AST, returns typed result. No mutable state persisting across calls.
- [ ] `chelis_ir::lower` takes typed AST, returns DAG. No mutable state.
- [ ] `chelis_backend_c::codegen` takes DAG, returns C source string. No mutable state.
- [ ] No global mutable state (`static mut`, `lazy_static` with mutation, `thread_local` with mutation) in any crate.
- [ ] Two calls to the same function with the same input produce the same output.

Test: call `codegen` twice with the same DAG, verify identical output. Call `infer` twice with the same AST, verify identical result.

## Check 15: Known Landmines

These are specific issues flagged during development. Verify each is resolved:

- [ ] The positional-load ABI bug in `emit.rs` was fixed — Load nodes use name-based mapping
- [ ] BLAS is opt-in, not unconditionally surfaced in link flags
- [ ] Duplicate `Load("x")` nodes with different types are caught by the verifier (not just by codegen)
- [ ] `grad_dag` returns `HashMap<NodeId, NodeId>` and the training loop correctly maps Load names → NodeIds → gradient NodeIds
- [ ] Dimensions are concrete at IR lowering time (symbolic dimensions are a Phase 1 concern, not a Phase 0 bug)
- [ ] The `chelis eval` command handles the scalar tensor issue (rank-0 `tensor[f32]`, not bare `f32`)

---

## Output Format

For each check item:
```
CHECK N.M: [PASS / FAIL / PARTIAL / SKIP]
Evidence: [command run, output observed, file/line referenced]
Issue: [if not PASS — what's wrong]
Severity: [CRITICAL / HIGH / MEDIUM / LOW]
```

At the end, produce:

1. **Summary table:** total checks, passes, fails, by severity
2. **Must-fix list:** all CRITICAL and HIGH items, prioritized
3. **Tech debt list:** all MEDIUM items
4. **Overall verdict:** "Phase 0 complete — ready for Phase 1" or "Phase 0 incomplete — N items must be resolved"

**The bar for "ready for Phase 1":** Zero CRITICAL items. Fewer than 3 HIGH items. MNIST trains to >90% accuracy. All CLI commands work. Tests pass. Specs match implementation.
