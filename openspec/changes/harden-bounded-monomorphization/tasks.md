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

- [ ] 1.1 Byte-determinism test: construct the probe-trigger fixture (a probed
      caller preceding the defs it wraps in source order, containing recursive
      generic calls), build ≥10 times, assert whole-`.c` byte identity. Expected
      red or flaky-red pre-fix; if it cannot be made to fail pre-fix on this
      machine, record that in the test's doc comment and rely on task 2.2's
      mutation check
- [ ] 1.2 Probe-pollution test: a probe-only instantiation (reachable only
      through a body the real lowering never takes through the mono path) must
      not appear in the emitted C; a probe-only specialization that cannot
      lower must not fail the build
- [ ] 1.3 Entry-selection minimal pair: a program whose authored tensor entry
      coexists with a tensor-shaped specialization; assert the compiled entry
      ABI equals the generic-free twin's. Expected red (ABI flip) pre-fix
- [ ] 1.4 Header-exclusion test: build a recursive-generic program, assert the
      emitted `.h` contains no `__mono_` identifier and the `.c` still compiles
      and links with the reported flags. Expected red pre-fix
- [ ] 1.5 Symbolic-dim pair: (positive) recursive trie generic at a
      `tensor[n, f32]` payload builds, links, runs, and matches eval byte-exact;
      (negative) a dim-disagreeing recursive call rejects with the branded
      diagnostic naming both types, no internal-error text, no artifact.
      Record current behavior of each (expected: positive untested/unknown,
      negative ICEs at C ABI projection)
- [ ] 1.6 Reef test: package def called via qualified and short spellings at one
      instantiation emits exactly one specialized definition, named from the
      package-internal identity. Expected red if spellings mint duplicates
- [ ] 1.7 Unit tests in `chelis-ir`: canonical-key stability (golden rendering,
      not `{:?}`), `is_monomorphized_specialization` positive/negative parity
      (16-hex suffix matches; user-plausible names do not), collision map
      (two keys forced to one symbol ⇒ loud error)

## 2. Probe isolation (D1)

- [ ] 2.1 Add the snapshot/restore guard over `MONO_SPECIALIZATIONS` and install
      it in `top_level_fn_helper_summary_rejects` (and any other speculative
      lowering entry a `grep` for discarded `lower_host_function` results
      finds); iterate probe name-sets in sorted order in
      `expr_calls_summary_rejecting_top_level_fn`
- [ ] 2.2 Mutation-verify determinism: with the guard disabled, tasks 1.1/1.2 go
      red on the trigger shape; record the observed divergence in this file,
      then re-enable
- [ ] 2.3 Confirm the memo-without-definition residue is closed by restore
      semantics (a failed probe leaves no memo entry); add the regression
      assertion to the 1.2 test

## 3. Surface hygiene (D2)

- [ ] 3.1 Add `is_monomorphized_specialization` beside the mangler in `host.rs`;
      lock predicate and mangling to each other with the 1.7 unit test
- [ ] 3.2 Exclude specializations in `preferred_tensor_entry_name` and
      `function_has_tensor_signature`; verify every entry-selection consumer
      (CLI, HIP, Metal, compiler-api) routes through those two chokepoints, and
      record the verification here
- [ ] 3.3 Filter the published header in `host_emit.rs` (header legs only; the
      `.c`-internal prototype pass stays unfiltered); tasks 1.3/1.4 green

## 4. Symbolic-dimension guard (D3)

- [ ] 4.1 Implement the instantiated-vs-supplied comparison (tensor shapes
      included) at the interning call site in `lower_recursive_generic_call`;
      route the first disagreement to the branded `[05-UNS-1]` rejection naming
      both types
- [ ] 4.2 Task 1.5 green on both halves; confirm the eval lane still executes
      the dim-disagreeing program unchanged (lowering-stage rejection only)

## 5. Canonical identity, key, and collisions (D4)

- [ ] 5.1 Replace the `Debug`-formatted interning key with the canonical
      signature rendering; golden stability test from 1.7
- [ ] 5.2 Normalize the interned callee identity to the package-internal name;
      minted symbols become `<package-internal-def>__mono_<hash>`; task 1.6
      green
- [ ] 5.3 Add the symbol→key reverse map and the loud collision failure; 1.7
      collision test green

## 6. Docs, validation, and acceptance

- [ ] 6.1 Update `spec/design/loud_unsupported.md`: add the symbolic-dim
      residue (terms resolved, dims disagreed — diagnostic names two types) to
      the delivered-behavior record
- [ ] 6.2 `CHANGELOG.md`: header no longer declares specialized symbols;
      symbolic-dim disagreement rejects cleanly instead of ICEing; reef'd
      specialization symbol names normalized; credit the PR #1202 adversarial
      review as the source of the findings
- [ ] 6.3 `openspec validate --all --strict --no-interactive` green with both
      this change and `add-bounded-monomorphization` active
- [ ] 6.4 `python3 scripts/gate.py --local` green in an isolated
      `CARGO_TARGET_DIR`
- [ ] 6.5 Acceptance oracle green: `cargo nextest run -p chelis-cli --test
      recursive_generic_monomorphization --no-fail-fast`
- [ ] 6.6 Red team per protocol: fresh local subagent runs the oracle, then
      probes adversarially — probe-trigger permutations (wrapper order, nested
      probes during specialization lowering), a specialization named like a
      user def, symbolic-dim disagreement through mutual recursion, reef
      spelling mixes across modules — and checks docs claims against shipped
      behavior; record findings here
