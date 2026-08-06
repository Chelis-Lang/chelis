# Tasks: harden-bounded-monomorphization

Authoritative acceptance oracle (D5): `cargo nextest run -p chelis-cli --test
recursive_generic_monomorphization --no-fail-fast` — the owning suite from
`add-bounded-monomorphization`, extended with this change's scenarios. Green
means every determinism, surface-hygiene, symbolic-dim, and identity scenario
passes and every pre-existing scenario stays green. Nothing in this change is
done while that oracle is red.

Prerequisite: `add-bounded-monomorphization` is implemented on this branch; this
change layers on its lowering machinery and archives after it.

## 1. Failing tests first (spec-first)

- [x] 1.1 Byte-determinism test: construct the probe-trigger fixture (a probed
      caller preceding the defs it wraps in source order, containing recursive
      generic calls), build ≥10 times, assert whole-`.c` byte identity. Expected
      red or flaky-red pre-fix; if it cannot be made to fail pre-fix on this
      machine, record that in the test's doc comment and rely on task 2.2's
      mutation check
      (`emitted_c_is_byte_identical_across_repeated_builds`: 10 builds of the
      `caller` → `wrap_int`/`wrap_bool` → `depth` fixture; mutation-verified
      red per task 2.2)
- [x] 1.2 Probe-pollution test: a probe-only instantiation (reachable only
      through a body the real lowering never takes through the mono path) must
      not appear in the emitted C; a probe-only specialization that cannot
      lower must not fail the build
      (folded into 1.1's exactly-two-specializations assertion — a genuinely
      probe-only instantiation was not constructible from source because the
      real pass lowers every probed def with the same memo; the guard makes
      probe-only emission structurally impossible regardless. Probe failures
      no longer fail the build: `top_level_fn_helper_summary_rejects` treats
      a failed speculative lowering as "no rejection", and a genuine callee
      defect resurfaces when the callee lowers for real)
- [x] 1.3 Entry-selection minimal pair: a program whose authored tensor entry
      coexists with a tensor-shaped specialization; assert the compiled entry
      ABI equals the generic-free twin's. Expected red (ABI flip) pre-fix
      (locked at the chokepoint with a synthetic `ConcreteHostProgram`:
      `preferred_tensor_entry_ignores_specializations` and
      `tensor_signature_classification_ignores_specializations` in
      `chelis-ir::host::tests` — red pre-fix, since `.rev().find` previously
      selected the appended specialization)
- [x] 1.4 Header-exclusion test: build a recursive-generic program, assert the
      emitted `.h` contains no `__mono_` identifier and the `.c` still compiles
      and links with the reported flags. Expected red pre-fix
      (`published_header_omits_specialized_symbols`; red pre-fix)
- [x] 1.5 Symbolic-dim pair: recorded current behavior by experiment
      (2026-08-06, three probes): (a) concrete tensor payload — builds, runs,
      output matches eval; (b) `tensor[n, f32]` payload through a dim-generic
      caller (the coral `Hamt` shape) — builds, runs, `2 == 2` eval parity,
      one specialization; (c) attempted dim disagreement through unsigned
      recursion — the checker's dim join types it, and it lowers and runs
      correctly (dims are erased at the `chelis_tensor*` ABI). **The PR #1202
      ICE premise does not map onto this architecture**: a specialization's
      parameter types are the call site's own checked types, so no separately
      instantiated signature exists to disagree with. The delta requirement
      was amended accordingly (positive scenario locked by
      `symbolic_dim_payload_specializes_with_eval_parity`; the negative
      recast as the unrepresentability contract); D3's comparison guard is
      NOT implemented — it would be a vacuous check of a by-construction
      equality
- [x] 1.6 Reef test: package def called via qualified and short spellings at one
      instantiation emits exactly one specialized definition, named from the
      package-internal identity. Expected red if spellings mint duplicates
      (locked at the resolution seam:
      `mono_interning_identity_is_the_definitions_own_name` proves both
      spellings resolve to the definition's own name and produce one
      canonical key and one symbol; reef call sites are already rewritten to
      the package-internal spelling before the checker runs, so the
      spelling-divergence risk lives in module-qualified programs, which the
      test covers directly)
- [x] 1.7 Unit tests in `chelis-ir`: canonical-key stability (golden rendering,
      not `{:?}`), `is_monomorphized_specialization` positive/negative parity
      (16-hex suffix matches; user-plausible names do not), collision map
      (two keys forced to one symbol ⇒ loud error)
      (`mono_specialization_key_is_stable`,
      `mono_specialization_predicate_matches_only_minted_symbols`,
      `mono_symbol_collision_is_loud`)

## 2. Probe isolation (D1)

- [x] 2.1 Add the snapshot/restore guard over `MONO_SPECIALIZATIONS` and install
      it in `top_level_fn_helper_summary_rejects` (and any other speculative
      lowering entry a `grep` for discarded `lower_host_function` results
      finds); iterate probe name-sets in sorted order in
      `expr_calls_summary_rejecting_top_level_fn`
      (`MonoProbeGuard` snapshot/restore; the grep found exactly one
      speculative entry — `top_level_fn_helper_summary_rejects`; the other
      `lower_host_function` caller is the real pass in `lower_host_program`)
- [x] 2.2 Mutation-verify determinism: with the guard disabled, tasks 1.1/1.2 go
      red on the trigger shape; record the observed divergence in this file,
      then re-enable
      **Mutation record (2026-08-06): guard disabled alone stayed green — the
      sorted probe iteration already pins emission order on this shape. With
      BOTH the guard and the sort disabled, the byte-determinism test failed
      immediately ("diverged on rebuild 1"): probe HashSet order leaked into
      specialization emission order across processes. Both defenses
      restored; test green.**
- [x] 2.3 Confirm the memo-without-definition residue is closed by restore
      semantics (a failed probe leaves no memo entry); add the regression
      assertion to the 1.2 test
      (closed structurally: the guard restores the WHOLE state — memo,
      symbol map, functions, in-progress stack — so a failed probe leaves
      neither a memo entry nor a definition; and a probe failure no longer
      propagates as a build failure)

## 3. Surface hygiene (D2)

- [x] 3.1 Add `is_monomorphized_specialization` beside the mangler in `host.rs`;
      lock predicate and mangling to each other with the 1.7 unit test
- [x] 3.2 Exclude specializations in `preferred_tensor_entry_name` and
      `function_has_tensor_signature`; verify every entry-selection consumer
      (CLI, HIP, Metal, compiler-api) routes through those two chokepoints, and
      record the verification here
      (verified by grep: `preferred_tensor_entry_name` and
      `function_has_tensor_signature` are the only tensor-entry selection
      exports, and every consumer — chelis-cli's build/eval entry paths, the
      HIP and Metal targets' entry resolution, and chelis-compiler-api —
      calls one of the two; no consumer re-implements the signature scan)
- [x] 3.3 Filter the published header in `host_emit.rs` (header legs only; the
      `.c`-internal prototype pass stays unfiltered); tasks 1.3/1.4 green
      (`emit_host_declarations` gains `include_specializations`; the
      `emit_host_header`/`emit_host_abi_header` legs filter, the in-`.c`
      prototype leg does not)

## 4. Symbolic-dimension guard (D3)

- [x] 4.1 Implement the instantiated-vs-supplied comparison (tensor shapes
      included) at the interning call site in `lower_recursive_generic_call`;
      route the first disagreement to the branded `[05-UNS-1]` rejection naming
      both types
      **Not implemented, deliberately (see 1.5's experiment record): the
      comparison would check a by-construction equality — specialization
      parameter types are derived from the call site's own checked types, so
      the two sides are the same values. The delta requirement was amended
      to state the unrepresentability contract instead.**
- [x] 4.2 Task 1.5 green on both halves; confirm the eval lane still executes
      the dim-disagreeing program unchanged (lowering-stage rejection only)
      (positive half green via
      `symbolic_dim_payload_specializes_with_eval_parity`; the "disagreeing"
      probe programs execute identically in eval and native — no rejection
      exists to compare, per the amended requirement)

## 5. Canonical identity, key, and collisions (D4)

- [x] 5.1 Replace the `Debug`-formatted interning key with the canonical
      signature rendering; golden stability test from 1.7
      (`write_canonical_host_type_key` + `mono_specialization_key`; every
      `HostTypeTerm` variant has an explicit spelling, unresolved variants
      included for totality though the concreteness gate excludes them)
- [x] 5.2 Normalize the interned callee identity to the package-internal name;
      minted symbols become `<package-internal-def>__mono_<hash>`; task 1.6
      green
      (`find_top_level_def_named` resolves any accepted spelling to the
      definition's own name; `lower_recursive_generic_call` interns and
      mints from that canonical identity, and the in-progress stack matches
      on it)
- [x] 5.3 Add the symbol→key reverse map and the loud collision failure; 1.7
      collision test green
      (`symbol_keys` map + `register_mono_symbol_key`; a collision rejects
      with an internal diagnostic naming both canonical signatures)

## 6. Docs, validation, and acceptance

- [x] 6.1 Update `spec/design/loud_unsupported.md`: record the verified
      symbolic-dimension invariant and the delivered probe, surface, identity,
      and collision behavior
- [x] 6.2 `CHANGELOG.md`: header no longer declares specialized symbols;
      tensor-entry selection excludes them; probe state is isolated; reef'd
      specialization symbol names are normalized; credit the PR #1202
      adversarial review as the source of the findings
- [x] 6.3 `openspec validate --all --strict --no-interactive` green with both
      this change and `add-bounded-monomorphization` active
      (34 passed, 0 failed)
- [x] 6.4 `python3 scripts/gate.py --local` green in an isolated
      `CARGO_TARGET_DIR`
      (green on 2026-08-06; the gate assigned a target directory inside this
      worktree; final `chelis-types` stage: 1,289/1,289)
- [x] 6.5 Acceptance oracle green: `cargo nextest run -p chelis-cli --test
      recursive_generic_monomorphization --no-fail-fast`
      (19/19 after the symbolic-dimension scenario joined the existing 18)
- [ ] 6.6 Red team per protocol: fresh local subagent runs the oracle, then
      probes adversarially — probe-trigger permutations (wrapper order, nested
      probes during specialization lowering), a specialization named like a
      user def, symbolic-dim disagreement through mutual recursion, reef
      spelling mixes across modules — and checks docs claims against shipped
      behavior; record findings here
