# Tasks: add-bounded-monomorphization

Authoritative acceptance oracle (D7): `cargo nextest run -p chelis-cli --test
recursive_generic_monomorphization --no-fail-fast` — green means every positive
scenario compiles, links, and runs with eval-lane-exact output and every negative
scenario rejects at check time with the atom-citing diagnostic. Nothing in this
change is done while that oracle is red.

## 1. Citation hygiene (independent, lands first)

- [ ] 1.1 Flip the branded rejection citation in `crates/chelis-ir/src/host.rs`
      from `chelis#941` to `chelis#1158`; update the two message assertions in
      `crates/chelis-cli/tests/issue_935_nullary_generic_adt.rs`
- [ ] 1.2 Update the chelis#941 boundary entry in
      `spec/design/loud_unsupported.md` to name chelis#1158 as the live tracker
- [ ] 1.3 Run `python3 scripts/gate.py --local`; land this group as its own PR

## 2. Spec authoring and failing tests (spec-first)

- [ ] 2.1 Probe current checker behavior (D5): author a polymorphic-recursion
      program (`f` over `a` recursively calling `f` at `Box[a]`) and record whether
      `chelis check` / `chelis eval` accept it today; decide the BREAKING bullet's
      final wording from the result
- [ ] 2.2 Author the uniform-recursive-instantiation atoms in
      `spec/04-type-system.md` §3.1, allocating `[04-INF-N]` numbers against
      current `main`; state the rule (in-group recursive calls typed at the
      caller's own instantiation; violation is a check-time type error carrying
      the atom per [05-UNS-5]) without implementation status
- [ ] 2.3 Create `crates/chelis-cli/tests/recursive_generic_monomorphization.rs`
      with test stubs mapped one-to-one to every scenario in both delta specs:
      direct recursion (one and two instantiations), mutual recursion, memoized
      termination/symbol-count, link-clean build + run vs eval output, check-time
      polymorphic-recursion rejection in `eval` and `build`, lane-uniform
      diagnostic equality, and no-closed-issue-citation; all red except the
      currently-rejecting negatives
- [ ] 2.4 Add the chelis#941 minimized `Box[a]`/`loop` reproducer to the suite as
      the named regression case

## 3. Checker: uniform recursive instantiation (D5)

- [ ] 3.1 In `chelis-types::infer`, compute recursive binding groups (SCCs over
      the top-level def call graph) during checking
- [ ] 3.2 Enforce in-group calls typed at the caller's own instantiation; emit an
      `ErrorWitness`-conformant diagnostic naming the function, both
      instantiations, and the deciding §3.1 atom
- [ ] 3.3 Negative-parity coverage in `chelis-types` unit tests: direct
      polymorphic recursion, mutual polymorphic recursion (uniform within one
      edge, growing across the cycle), and the accepted uniform twins of each
- [ ] 3.4 Verify lane uniformity: `eval --file` and `build` report the identical
      check-time error (oracle's negative half green)

## 4. Lowering: bounded memoized specialization (D4)

- [ ] 4.1 Add the specialization memo keyed by `(def name, canonical checked type
      application)` with per-`lower_host_program`-invocation lifetime, and the
      worklist that emits one specialized definition per key; insert the memo
      entry before lowering the body so in-progress symbols resolve
- [ ] 4.2 Implement deterministic symbol mangling from the canonical type
      identity; golden test asserts the exact emitted symbol set for a
      two-instantiation program across two consecutive builds
- [ ] 4.3 Rewire the recursion-detected fallback in `host.rs` from the branded
      rejection to a specialization request; leave the non-recursive inlining
      path untouched
- [ ] 4.4 Emit specialized definitions in `lower_host_program` alongside the
      existing generic-definition skip; confirm emitted C contains no reference
      to an omitted generic definition (link-clean assertion via the suite's
      compile step)
- [ ] 4.5 Keep the fail-closed residue: any surviving unspecializable generic
      call still rejects branded, now citing chelis#1158; negative fixture
      asserts no C artifact is written on rejection
- [ ] 4.6 Verify HIP and Metal host lanes consume the shared specialization path,
      or record the exact gap as [05-UNS] residue citing chelis#1158 in the
      owning docs
- [ ] 4.7 Confirm Tier-2 precision and Tier-3 rank specialization suites are
      green unchanged (regression oracle for the untouched paths)

## 5. Docs, examples, and spec sync

- [ ] 5.1 Add an executable recursive-generic example to `examples/` that
      survives `chelis fmt --check` and `chelis check` (Phase 0 path)
- [ ] 5.2 Retire the boundary entry in `spec/design/loud_unsupported.md` to a
      delivered-behavior record pointing at the oracle suite
- [ ] 5.3 Update `docs/book` build/eval pages where they describe the generic
      lowering boundary; add `CHANGELOG.md` entries (feature; BREAKING note if
      task 2.1 found acceptance)
- [ ] 5.4 Verify every scenario in both delta specs maps to at least one test in
      the oracle suite; note the mapping in the suite's module docs

## 6. Validation and acceptance

- [ ] 6.1 `python3 scripts/gate.py --local` green (clippy `-D warnings`, fmt,
      `chelis lint --check .`, rustdoc trio, checkpoint fixture, changed-crate
      nextest) in an isolated `CARGO_TARGET_DIR`
- [ ] 6.2 Acceptance oracle green: `cargo nextest run -p chelis-cli --test
      recursive_generic_monomorphization --no-fail-fast`
- [ ] 6.3 Red team per protocol: fresh local subagent executes the oracle suite,
      probes adversarial variants (mutual recursion crossing three defs,
      instantiation reuse across separate call sites, polymorphic recursion
      hidden behind a local alias), and checks docs claims against shipped
      behavior; record findings here
- [ ] 6.4 Full CI green including macOS Smoke (authoritative workspace oracle);
      chelis#1158 closes on the merged green oracle with a comment naming the
      suite; notify coral to re-cite/retire its 6 citation sites on the next
      conform bump
