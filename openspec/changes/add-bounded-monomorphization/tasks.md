# Tasks: add-bounded-monomorphization

Authoritative acceptance oracle (D7): `cargo nextest run -p chelis-cli --test
recursive_generic_monomorphization --no-fail-fast` — green means every positive
scenario compiles, links, and runs with eval-lane-exact output and every negative
scenario rejects at check time with the atom-citing diagnostic. Nothing in this
change is done while that oracle is red.

## 1. Citation hygiene (independent, lands first)

- [x] 1.1 Flip the branded rejection citation in `crates/chelis-ir/src/host.rs`
      from `chelis#941` to `chelis#1158`; update the two message assertions in
      `crates/chelis-cli/tests/issue_935_nullary_generic_adt.rs`
- [x] 1.2 Update the chelis#941 boundary entry in
      `spec/design/loud_unsupported.md` to name chelis#1158 as the live tracker
- [ ] 1.3 Run `python3 scripts/gate.py --local`; land this group as its own PR
      (gate green 2026-08-06 — see 6.1. The separate-PR option was overtaken:
      the full implementation completed in the same change set, so the
      citation flip ships with it; splitting now would be churn. PR landing
      awaits maintainer go-ahead.)

## 2. Spec authoring and failing tests (spec-first)

- [x] 2.1 Probe current checker behavior (D5): author a polymorphic-recursion
      program (`f` over `a` recursively calling `f` at `Box[a]`) and record whether
      `chelis check` / `chelis eval` accept it today; decide the BREAKING bullet's
      final wording from the result
      **Result (2026-08-05): `chelis check` ACCEPTS (exit 0, zero errors) and
      `chelis eval` executes it (`main = 3`); `chelis build` rejects at lowering
      with the branded chelis#1158 diagnostic. The change IS BREAKING for the
      eval lane: the CHANGELOG carries a BREAKING note with the reproducer.**
- [x] 2.2 Author the uniform-recursive-instantiation atoms in
      `spec/04-type-system.md` §3.1, allocating `[04-INF-N]` numbers against
      current `main`; state the rule (in-group recursive calls typed at the
      caller's own instantiation; violation is a check-time type error carrying
      the atom per [05-UNS-5]) without implementation status
- [x] 2.3 Create `crates/chelis-cli/tests/recursive_generic_monomorphization.rs`
      with test stubs mapped one-to-one to every scenario in both delta specs:
      direct recursion (one and two instantiations), mutual recursion, memoized
      termination/symbol-count, link-clean build + run vs eval output, check-time
      polymorphic-recursion rejection in `eval` and `build`, lane-uniform
      diagnostic equality, and no-closed-issue-citation; all red except the
      currently-rejecting negatives
- [x] 2.4 Add the chelis#941 minimized `Box[a]`/`loop` reproducer to the suite as
      the named regression case (suite is 4 green / 9 red: the four green are
      the currently-rejecting negatives and currently-accepting checks, per
      the spec-first expectation)

## 3. Checker: uniform recursive instantiation (D5)

- [x] 3.1 In `chelis-types::infer`, compute recursive binding groups (SCCs over
      the top-level def call graph) during checking (reused the existing
      `function_inference_sccs` planner; extended its call-graph edges to
      include bare `(var f)` references so aliased recursion forms a group)
- [x] 3.2 Enforce in-group calls typed at the caller's own instantiation; emit an
      `ErrorWitness`-conformant diagnostic naming the function, both
      instantiations, and the deciding §3.1 atom (`infer/recursion.rs`:
      in-group instantiations recorded during group inference, pinned against
      let-generalization, validated post-group; fully concrete arguments are
      admitted for inference-introduced signature variables — [04-INF-2] was
      amended to state that admission rule — while authored binders stay
      strict)
- [x] 3.3 Negative-parity coverage in `chelis-types` unit tests: direct
      polymorphic recursion, mutual polymorphic recursion (uniform within one
      edge, growing across the cycle), and the accepted uniform twins of each
      (`infer/tests/recursion_uniformity.rs`, plus let-alias positive/negative
      pair and out-of-group freedom)
- [x] 3.4 Verify lane uniformity: `eval --file` and `build` report the identical
      check-time error (oracle's negative half green; both lanes report the
      identical CheckError, wrapped in per-lane invocation framing)

## 4. Lowering: bounded memoized specialization (D4)

- [x] 4.1 Add the specialization memo keyed by `(def name, canonical checked type
      application)` with per-`lower_host_program`-invocation lifetime, and the
      worklist that emits one specialized definition per key; insert the memo
      entry before lowering the body so in-progress symbols resolve
      (`MONO_SPECIALIZATIONS` thread-local in `host.rs`: memo + in-progress
      stack registered before body lowering; recursion drives the worklist)
- [x] 4.2 Implement deterministic symbol mangling from the canonical type
      identity; golden test asserts the exact emitted symbol set for a
      two-instantiation program across two consecutive builds
      (`<def>__mono_<fnv1a-16hex>` over the canonical `HostTypeTerm`
      signature; `specialized_symbol_set_is_deterministic_across_builds`
      asserts set equality across two builds plus exact cardinality)
- [x] 4.3 Rewire the recursion-detected fallback in `host.rs` from the branded
      rejection to a specialization request; leave the non-recursive inlining
      path untouched (`lower_recursive_generic_call` replaces the rejection;
      the `callee_is_nonrecursive_type_polymorphic` inline branch is
      unchanged; the two issue_935 branded-failure tests flipped to
      compile-success assertions)
- [x] 4.4 Emit specialized definitions in `lower_host_program` alongside the
      existing generic-definition skip; confirm emitted C contains no reference
      to an omitted generic definition (link-clean assertion via the suite's
      compile step; specializations drain into `host.functions` before the
      refinement fixpoint so prototypes/conformance treat them uniformly)
- [x] 4.5 Keep the fail-closed residue: any surviving unspecializable generic
      call still rejects branded, now citing chelis#1158; negative fixture
      asserts no C artifact is written on rejection (residue = a call whose
      checked type application never resolves concrete, e.g. `loop(Empty)`
      with unconstrained `a` at top level;
      `surviving_unsupported_call_fails_closed_without_artifacts` green)
- [x] 4.6 Verify HIP and Metal host lanes consume the shared specialization path,
      or record the exact gap as [05-UNS] residue citing chelis#1158 in the
      owning docs
      **Verified 2026-08-05: `--target hip` and `--target metal` both route
      host code through the shared `chelis-ir` lowering + shared host
      emitter — the emitted `.cpp` for the recursive-generic probe contains
      the specialized symbol's prototype, definition, and recursive call on
      both targets, and both builds exit 0. No divergent host lane exists;
      no residue to record.**
- [x] 4.7 Confirm Tier-2 precision and Tier-3 rank specialization suites are
      green unchanged (regression oracle for the untouched paths)
      (full `-p chelis-ir -p chelis-backend-c -p chelis-cli -p chelis-types`
      nextest run: 4711 tests, green after wiring the new example into the
      parity corpus; the only other failures were two `issue_914` timeout
      tests that pass on a quiet machine — the documented CPU-contention
      signature, re-run green)

## 5. Docs, examples, and spec sync

- [x] 5.1 Add an executable recursive-generic example to `examples/` that
      survives `chelis fmt --check` and `chelis check` (Phase 0 path)
      (`examples/recursive_generic.ch`; fmt --check clean, check clean, eval
      prints `5`)
- [x] 5.2 Retire the boundary entry in `spec/design/loud_unsupported.md` to a
      delivered-behavior record pointing at the oracle suite
- [x] 5.3 Update `docs/book` build/eval pages where they describe the generic
      lowering boundary; add `CHANGELOG.md` entries (feature; BREAKING note if
      task 2.1 found acceptance)
      (no `docs/book` page describes the recursive-generic lowering boundary
      — verified by search — so the book half is a no-op; CHANGELOG carries
      the Added entry and the BREAKING checker note with the task-2.1
      reproducer)
- [x] 5.4 Verify every scenario in both delta specs maps to at least one test in
      the oracle suite; note the mapping in the suite's module docs (module
      doc of `recursive_generic_monomorphization.rs` carries the full
      11-scenario map: 7 generic-monomorphization + 4 type-system)

## 6. Validation and acceptance

- [x] 6.1 `python3 scripts/gate.py --local` green (clippy `-D warnings`, fmt,
      `chelis lint --check .`, rustdoc trio, checkpoint fixture, changed-crate
      nextest) in an isolated `CARGO_TARGET_DIR`
      (green 2026-08-06; the one post-gate edit — a one-line `current_dir`
      hygiene fix in `style_gate.rs` stopping `build --allow-style-violations`
      from writing artifacts into `crates/chelis-cli/` — was re-verified with
      `cargo fmt --check`, `cargo clippy -p chelis-cli --tests -D warnings`,
      and the 18/18 style_gate suite)
- [x] 6.2 Acceptance oracle green: `cargo nextest run -p chelis-cli --test
      recursive_generic_monomorphization --no-fail-fast` (13/13; plus the
      issue_935 suite 10/10 and parity 21/21 including
      `parity_recursive_generic`'s byte-exact eval/C comparison)
- [ ] 6.3 Red team per protocol: fresh local subagent executes the oracle suite,
      probes adversarial variants (mutual recursion crossing three defs,
      instantiation reuse across separate call sites, polymorphic recursion
      hidden behind a local alias), and checks docs claims against shipped
      behavior; record findings here
      **Round 1 (2026-08-06, fresh local subagent in herdr pane w36:pD):**
      confirmed one MAJOR finding before stalling mid-run (agent hung after
      ~30 min of work; handle closed per protocol):
      - **Permuted in-group instantiation miscompiled.** `swap[a, b]`
        recursing at `(b, a)` — admitted by [04-INF-2] as a renaming of the
        caller's own parameters — was wired by the lowering's in-progress
        reuse branch back to the caller's own symbol, emitting C that passes
        `bool` where `chelis_adt*` is expected (clang: incompatible integer
        to pointer conversion) while `chelis build` reported success.
        **Fixed** in the same session: `lower_recursive_generic_call` now
        derives the edge's own checked type application first (each orbit
        member gets its own memoized specialization; renaming orbits are
        finite) and reuses the innermost in-progress symbol only for
        underived (unconstrained) edges such as `loop(Empty)`. Locked by
        `permuted_recursive_instantiation_specializes_per_orbit_member`
        (two orbit symbols, native compile, run output matches eval).
      **Round 2 (2026-08-06, fresh local subagent in herdr pane w36:pE,
      completed with report at `target/redteam/report.md`):** ran the
      oracle (14/14 at the time), issue_935 + parity, and 18 adversarial
      probes (3-def SCC, 3-param rotation, permutation+narrowing,
      alias/argument-passed/sig-authored polymorphic recursion, determinism
      byte-comparison, docs-claims audit). Verdict REDTEAM-FINDINGS:
      - **MAJOR (fixed):** permuted edge carrying an unconstrained argument
        (`tri[a, b, c]` recursing at `(b, a, Empty)`) — the round-1 fix's
        all-or-nothing derivation dropped the derived permuted slots and
        blind-reused the caller's own symbol; silent build success,
        uncompilable C. Fixed by per-slot completion: unresolved slots fill
        from the innermost in-progress same-def specialization's
        corresponding parameter, derived slots are kept, and the merged
        application goes through the ordinary memo. Locked by
        `permuted_edge_with_unconstrained_argument_specializes_correctly`.
      - **MINOR (documented residue):** a mutually recursive cross-member
        edge whose argument leaves the callee parameter unconstrained
        (`even2[a]` → `odd2(Empty, ...)`) is checker-admitted and
        eval-executable but fails closed at build (branded, chelis#1158, no
        artifact) — host lowering has no positional correspondence across
        different defs' parameters. Recorded in
        `spec/design/loud_unsupported.md` and locked by
        `mutual_unconstrained_cross_edge_fails_closed`.
      - **NIT (fixed):** the monomorphic-caller rejection named an empty
        instantiation `[]` and suggested reusing nonexistent type
        parameters; the diagnostic now names the actual constraint. Locked
        by `monomorphic_caller_in_group_names_the_missing_type_parameters`.
      Beyond the probe findings, round 2 also audited this implementation
      against the adversarial review of PR #1202 (a parallel, independently
      built implementation of chelis#1158 that surfaced the same day) and
      confirmed four shared defect classes in our architecture: probe-driven
      specialization-state pollution/nondeterminism, specialized symbols
      leaking into the published header and tensor-entry selection, the
      symbolic-dimension blind spot in `is_unresolved` concreteness gating,
      and the `Debug`-formatted interning key with undetected hash
      collisions. Those findings are recorded as their own follow-up
      OpenSpec change, `openspec/changes/harden-bounded-monomorphization`
      (proposal + design + delta on `generic-monomorphization` + spec-first
      tasks), which depends on this change and archives after it; its task
      6.6 carries the residual adversarial validation for that surface. The
      disposition of PR #1202 itself (merge, close in favor, or split) is a
      maintainer decision recorded on chelis#1158.
- [ ] 6.4 Full CI green including macOS Smoke (authoritative workspace oracle);
      chelis#1158 closes on the merged green oracle with a comment naming the
      suite; notify coral to re-cite/retire its 6 citation sites on the next
      conform bump
