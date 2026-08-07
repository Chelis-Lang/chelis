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
- [x] 1.2 Probe-pollution test: a probe-only instantiation must not appear in
      the emitted C. A probe-only specialization that cannot lower must not
      fail the build
      (`emitted_c_is_byte_identical_across_repeated_builds` locks the emitted
      specialization set. The `chelis-ir` seam test
      `failed_mono_probe_restores_state_and_defers_the_real_error` forces
      `lower_host_function` to fail inside `MonoProbeGuard`. It verifies
      `Ok(false)`, exact restoration of all four state fields, and the exact
      genuine error from later real lowering)
- [x] 1.3 Entry-selection minimal pair: a program whose authored tensor entry
      coexists with a tensor-shaped specialization; assert the compiled entry
      ABI equals the generic-free twin's. Expected red (ABI flip) pre-fix
      (locked at the chokepoint with a synthetic `ConcreteHostProgram`:
      `preferred_tensor_entry_ignores_specializations` and
      `tensor_signature_classification_ignores_specializations` in
      `chelis-ir::host::tests` — red pre-fix, since `.rev().find` previously
      selected the appended specialization)
- [x] 1.4 Header and linkage tests: assert that a real specialization stays out
      of the published header, an authored mangling-shaped name stays in it,
      generated C links and runs, and two object-mode outputs with one
      specialization symbol link into one relocatable object
      (`published_header_omits_specialized_symbols`,
      `authored_mono_shaped_name_stays_in_the_published_surface`, and
      `specializations_link_cleanly_across_generated_objects`; all three
      defects reproduced before their fixes)
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
      was amended accordingly. The positive scenario is locked by
      `symbolic_dim_payload_specializes_with_eval_parity`. The negative case
      is the unrepresentability contract. Host lowering contains no redundant
      comparison of a checked type with itself
- [x] 1.6 Reef tests: an exact and a short reference to one definition share
      one canonical key in the unit seam. An executable Reef package also uses
      two recursive generic definitions with one terminal name. Native output
      must equal eval and C must contain both specialization identities
      (`mono_interning_identity_is_the_definitions_own_name` and
      `package_defs_with_one_terminal_name_keep_distinct_specializations`;
      the package test reproduced a silent `32` versus `5` miscompile before
      exact identity lookup won over terminal fallback)
- [x] 1.7 Unit tests in `chelis-ir`: failed-probe restoration, canonical-key
      stability, explicit function provenance, and collision-map rejection
      (`failed_mono_probe_restores_state_and_defers_the_real_error`,
      `mono_specialization_key_is_stable`,
      `mono_specialization_provenance_is_explicit`,
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
      semantics. Add the regression assertion to the 1.2 test
      (`failed_mono_probe_restores_state_and_defers_the_real_error` seeds the
      memo, symbol map, function list, and in-progress stack. It forces the
      probe error, then compares every field with the pre-probe snapshot)

## 3. Surface hygiene (D2)

- [x] 3.1 Add `HostFunctionOrigin::{Authored, Monomorphized}` and preserve it
      through concrete-host and C-ABI projection. Lock explicit provenance
      with the 1.7 unit test and the authored mangling-shaped CLI test
- [x] 3.2 Exclude specializations in `preferred_tensor_entry_name` and
      `function_has_tensor_signature` by provenance. Verify every
      entry-selection consumer (CLI, HIP, Metal, compiler-api) routes through
      those two chokepoints
      (verified by grep: both exports are the only tensor-entry selectors, and
      every consumer calls one. No consumer repeats the signature scan)
- [x] 3.3 Filter published headers by provenance. Keep the internal prototype
      pass complete. Give specialization declarations and definitions
      translation-unit-local linkage in binary mode and object mode. Keep
      authored object exports external
      (`published_header_omits_specialized_symbols`,
      `authored_mono_shaped_name_stays_in_the_published_surface`, and
      `specializations_link_cleanly_across_generated_objects` are green)

## 4. Symbolic-dimension contract (D3)

- [x] 4.1 Record and lock the delivered dimension invariant: specialization
      parameter types come from checked argument expressions, and the result
      type comes from the checked application. A second instantiated dimension
      signature does not exist at this boundary
- [x] 4.2 Run positive and negative dimension probes. The symbolic payload
      passes with native and eval parity. A concrete dimension mismatch rejects
      in the checker with `DimensionMismatch` before host lowering

## 5. Canonical identity, key, and collisions (D4)

- [x] 5.1 Replace the `Debug`-formatted interning key with the canonical
      signature rendering; golden stability test from 1.7
      (`write_canonical_host_type_key` + `mono_specialization_key`; every
      `HostTypeTerm` variant has an explicit spelling, unresolved variants
      included for totality though the concreteness gate excludes them)
- [x] 5.2 Normalize the interned callee identity to the package-internal name.
      Resolve an exact canonical name before terminal fallback. Accept a
      terminal fallback only when it is unique
      (`find_top_level_def_named` now preserves two package definitions named
      `depth`; the executable Reef test and native/eval parity are green)
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
      (final run on 2026-08-06 used `NEXTEST_TEST_THREADS=4` after two
      concurrent-gate runs missed only the two watchdog timing tests. The
      focused timeout suite passed 6/6. Final gate stages passed 415/415,
      2,083/2,083, 942/942, and 1,289/1,289)
- [x] 6.5 Acceptance oracle green: `cargo nextest run -p chelis-cli --test
      recursive_generic_monomorphization --no-fail-fast`
      (22/22 after symbolic dimensions, authored-name provenance, object
      linkage, and package identity joined the suite)
- [x] 6.6 Red team per protocol: fresh local subagent runs the oracle, then
      probes adversarially — probe-trigger permutations (wrapper order, nested
      probes during specialization lowering), a specialization named like a
      user def, symbolic-dim disagreement through mutual recursion, reef
      spelling mixes across modules — and checks docs claims against shipped
      behavior; record findings here
      (round 1, `target/redteam/hardening/report.md`: REJECT. BLOCKER: first
      terminal-name match caused a `32` eval result to compile as `5` across two
      package definitions. MAJOR: an authored mangling-shaped name vanished
      from its header. MAJOR: object-mode specializations had external linkage
      and collided across translation units. MAJOR: proposal, design, tasks,
      and oracle coverage disagreed. Each code defect now has a red-before-fix
      test in the owning oracle. The planning artifacts now describe the
      delivered architecture. Round 2, `target/redteam/hardening-round2/report.md`:
      REJECT. All exercised behavior passed, but direct executable evidence for
      failed-probe restoration was absent. The new `chelis-ir` seam test forces
      that error and locks `Ok(false)`, complete state restoration, and later
      real-error attribution. Round 3,
      `target/redteam/hardening-round3/report.md`: ACCEPT. The oracle passed
      22/22, the failed-probe seam passed, six hardening locks passed, and an
      independent symbolic-dimension probe had native/eval parity. The one
      minor mutation-record inconsistency in the proposal and design is fixed.)
