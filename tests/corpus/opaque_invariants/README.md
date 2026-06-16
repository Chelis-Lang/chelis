# Opaque-Invariants Differential Corpus (RFC D-CORPUS)

In-tree, mechanically generated corpus that exercises the opaque-types
feature's shipped rejection, well-formedness, obligation, injection, and
generator-starvation paths, with **measured** diagnostic coverage and an
executable **solver-free** regression gate for `chelis check`.

Owning decision: `spec/design/opaque_invariants_rfc.md` §15 (D-CORPUS).

## What is here

| File | Role |
| --- | --- |
| `generate_corpus.py` | THE PRODUCER. Emits `programs/` + `manifest.json`. The only writer; the corpus is regenerated, never hand-patched. |
| `programs/*.dp`, `programs/*.ch` | The generated Chelis programs. |
| `manifest.json` | Per-program: lane, filename, pinned exit code, `prove_args`, and the `targets` (the diagnostic / status tokens the program is designed to hit). |
| `coverage_runner.py` | Unit 2. Runs every program against the live binary and MEASURES that every target token is hit by ≥ 1 program. Fails the gate on any uncovered token. |
| `solver_free.py` | Unit 3. The load-bearing deliverable: asserts `chelis check` stays solver-free on the corpus. |
| `test_generate_corpus.py`, `test_coverage_runner.py`, `test_solver_free.py` | `unittest` unit tests for the three scripts (no binary needed). |

The Rust drivers that make this an oracle in the cargo gate:

- `crates/chelis-cli/tests/opaque_corpus_gate.rs` — DEFAULT gate (non-smt):
  the solver-free assertion + check-lane coverage + corpus-in-sync.
- `crates/chelis-cli/tests/opaque_corpus_smt.rs` — `--features smt`: full
  (both-lane) coverage + prove performance sanity.

## Two lanes

- **`check`** — run with the DEFAULT (non-smt) `chelis` binary via
  `chelis check`. Covers the six opacity rejections (broadened), the sixth
  (unexported-producer) rejection, `@opaque`-outside-a-named-module,
  `DuplicateModule`, `ReservedLinkerName`, `DuplicateDefinition`, and the
  eight invariant well-formedness error classes. Rejection / WF programs are
  `.dp` (the lexical multi-module / hand-encoded form) so the checker-pass
  `OpaqueTypeViolation` fires deterministically instead of a parser-level Surf
  twin; a `.ch` covers the Surf-surface `@opaque`-outside-module declaration
  error.
- **`prove`** — run with the `--features smt` binary via `chelis prove
  --json`. Covers obligation statuses (`passed` / `failed` / `unsupported` /
  `error`), producer-set decomposition (Direct / Option / tuple / constant) +
  covered-or-rejected containers (List, record wrapper, the RT-3 type-alias
  record field, caller-receives signature rejection), assumption injection
  over an opaque binder, the exact-`==` generator-starvation diagnostic and
  its `--invariant-min-rate 0.0` legacy-error twin, the type-check-failure
  prove Error, and the many-producer perf-sanity module.

The obligation surface only compiles under the `smt` feature (mirroring
`prove_invariant_obligations.rs`), so prove-lane coverage is the `smt`
companion; the default gate covers the check lane only and reports the prove
lane as SKIPPED (never silently dropped).

## The coverage contract (MEASURED, not assumed)

`coverage_runner.py` runs every program against the live binary and records,
per target token, the set of program ids that hit it. A token with **zero**
hits fails the gate. Coverage is measured against shipped behavior; if a
fixture stops triggering its diagnostic (a regression, or a message change),
the gate goes red. The `manifest.json` `targets` list is the full set the
runner must show as covered. No target is silently omitted: every targeted
path appears in the manifest and must be hit.

The runner also pins each program's exit code (`expect_exit`); a drift from
the pinned exit is a corpus-drift finding that also fails the gate.

## The solver-free gate (the load-bearing W7 deliverable)

`solver_free.py` makes the RFC's "no solver in the check loop" guarantee
executable in three independent assertions:

- **A. NO-LINK.** The default (non-smt) `chelis` binary links **zero** cvc5
  symbols (`nm -C` is empty). The `--features smt` binary DOES link cvc5
  (≈ 33k symbols) — a control proving the symbol probe discriminates, so the
  absence in the default build is meaningful. This is structural: the crate
  that ships `chelis check` (`chelis-types`) has no dependency on
  `chelis-prove`/cvc5, so the solver is not in `chelis check`'s dependency
  graph at all.
- **B. IDENTICAL CHECK.** The non-smt and smt builds produce byte-identical
  `chelis check` verdicts (exit code + sorted `(kind, message)` errors) on
  every check-lane program. If check consulted the solver, the cvc5-linked
  build could diverge; identity proves the check verdict is
  solver-independent even where the solver is present. (Cross-build → manual
  gate, below.)
- **C. CHECK-COVERS.** The non-smt binary `chelis check`s every check-lane
  program to its pinned exit code — check needs no solver to run the corpus.

The default-gate Rust test runs A + C against the freshly built non-smt
binary. B is the cross-build identity arm and is a documented manual gate.

## Running

Regenerate the corpus (after editing `generate_corpus.py`):

```sh
.venv/bin/python tests/corpus/opaque_invariants/generate_corpus.py
```

Script unit tests (no binary needed):

```sh
.venv/bin/python -m unittest \
  tests.corpus.opaque_invariants.test_generate_corpus \
  tests.corpus.opaque_invariants.test_coverage_runner \
  tests.corpus.opaque_invariants.test_solver_free
```

The cargo oracle (the W7 acceptance surface):

```sh
# default gate (non-smt): solver-free + check-lane coverage + in-sync
cargo nextest run -p chelis-cli --test opaque_corpus_gate

# smt companion: full coverage (both lanes) + prove perf-sanity
LD_LIBRARY_PATH=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib \
  cargo nextest run -p chelis-cli --features smt --test opaque_corpus_smt
```

## Manual gate: cross-build check-identity (assertion B)

Assertion B compares two builds, so it cannot run inside one cargo
invocation. Build both binaries to distinct target dirs, then run
`solver_free.py` with both:

```sh
CARGO_TARGET_DIR=target-nonsmt cargo build -p chelis-cli
LD_LIBRARY_PATH=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib \
  cargo build -p chelis-cli --features smt   # target/debug is the smt build

.venv/bin/python tests/corpus/opaque_invariants/solver_free.py \
  --nonsmt-bin "$PWD/target-nonsmt/debug/chelis" \
  --smt-bin    "$PWD/target/debug/chelis" \
  --require-control
```

Expected success condition: exit 0, with
`cvc5 symbols = 0 (PASS)`, a discriminating control
(`smt binary cvc5 symbols = <large>`), `0 mismatch(es)`, `0 exit-code
drift(s)`, and `SOLVER-FREE GATE: PASS`. Owner: W7 (opaque types). Not part of
the default `cargo nextest` gate (it requires two builds).

## Hull-side reference support: OUT-OF-REPO FOLLOW-UP

A Hull-side reference implementation of the opaque-types feature (so the
opaque/invariant paths could be cross-checked against Hull's differential
verdicts the way `tests/conformance/hull/` cross-checks the core language) is
an **out-of-repo follow-up**, tracked against the Hull repo. It is explicitly
NOT claimed by this corpus: this corpus is a chelis-internal
diagnostic-coverage + solver-free regression gate, not a Hull differential
agreement corpus. The Hull conformance machinery
(`tests/conformance/hull/`) is the shape reference, not a dependency.
