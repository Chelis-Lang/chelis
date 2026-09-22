## Delivery scopes and dependencies

[#1354](https://github.com/Chelis-Lang/chelis/issues/1354) remains the umbrella and
final acceptance tracker. The sections below group obligations, not one mandatory
all-at-once implementation phase. Deliver the following independently reviewable
scopes; their native GitHub blocked-by relationships mirror this table.

| Scope | Issue | Prerequisites | Owned boundary |
|---|---|---|---|
| Identity production | [#2394](https://github.com/Chelis-Lang/chelis/issues/2394) | None | Public contract, D1/D2 producer/decoder, independent CLI/Python expectations, Cargo/Nix build inputs and explicit provenance |
| Shared verification | [#2395](https://github.com/Chelis-Lang/chelis/issues/2395) | #2394 | D3/D4/D5 resolver, live freshness, verified staging/receipts, shared adapter and representation-runner pins, core oracle machinery |
| Native distributions | [#2396](https://github.com/Chelis-Lang/chelis/issues/2396) | #2394 | Exact Nix/release/container outputs, sealed bundles, installer validation and installed canary |
| CLI activation | [#2397](https://github.com/Chelis-Lang/chelis/issues/2397) | #2395, #2396 | All production C/HIP/Metal staging callers, CLI-owned selectors/native drivers and CLI mutation witnesses |
| Python activation and wheels | [#2398](https://github.com/Chelis-Lang/chelis/issues/2398) | #2395 | Extension-owned bundle, compile/stage/link/call, persisted load admission and Python mutation/wheel witnesses |
| CPU harnesses | [#2399](https://github.com/Chelis-Lang/chelis/issues/2399) | #2395 | C backend unit/integration, E2E/spec runners and benchmarks; CLI-owned drivers stay in #2397 |
| HIP harnesses | [#2400](https://github.com/Chelis-Lang/chelis/issues/2400) | #2395 | Conventional and Phase-2 exact-pin consumers, intended feature recipes and bounded HIP evidence |
| Metal harnesses | [#2401](https://github.com/Chelis-Lang/chelis/issues/2401) | #2395 | Metal selection/staging/linking and separately reported macOS/Metal evidence |

Each scope takes its applicable fixtures, positive/negative execution, documentation,
changelog, selector dispositions and adversarial/hosted evidence from the obligations
below. CPU families may use separate PRs under their issue; HIP and Metal do not wait
for each other or Python. Native package validation uses the producer's common
decoder/comparison contract, not a second compatibility policy.

Producer/package support may land before enforcement with the parent bug still open.
Every activated consumer must be complete through its own publication/load boundary
and ship with matching producer outputs: no legacy fallback or identity bypass.
Keep Python compilation and persisted `load` admission together; do not create a new
native cache subsystem. A partial scoped run identifies its coverage, never a passing
parent aggregate. #2395 owns shared oracle machinery; each later scope supplies its
rows. Final full-oracle execution, evidence reconciliation and closure remain #1354.

## 1. Contract and discriminating fixtures

These are implementation tasks, not completed by authoring or validating this plan.
The authoritative completion oracle is the planned command
`.venv/bin/python scripts/runtime_artifact_identity_oracle.py` described in `design.md`.
Every execution claim remains bounded to its recorded consumers/configurations.

- [ ] 1.1 Reconcile the superseded draft PR #2036 and local issue-1354 sketches with this plan; preserve their work and do not mark unexecuted tests as evidence. Enumerate all runtime selectors/producers from D6 and disposition every occurrence through its owning scope.
- [ ] 1.2 Author the public identity, override, staging/failure, source-freshness, and receipt rules in `spec/08-backends.md` §2 and `spec/11-ffi.md` §§1.4/2; cross-reference them from the captured backend/FFI and Nix-package capabilities without transferring authority or adding a MODIFIED delta to this change.
- [ ] 1.3 Build real matching/incompatible fixture archives using exact Cargo `compiler-artifact` outputs. Add red-first cases for all identity dimensions, missing/malformed/duplicate/unsupported records, identical copies, distinct-byte ambiguity, equal/reversed mtimes, and reversed enumeration. A valid unique match must succeed.
- [ ] 1.4 Add override and staging cases: valid directory/exact-file pins, conflicting or invalid overrides with a valid default nearby, header mismatch, byte replacement during staging, and absent/stale cached-artifact receipts. Check failure before linking/usable publication, not arbitrary eventual errors.
- [ ] 1.5 Establish real CLI build/stage/link/run and Python compile/load/call positive witnesses with exact numerical results and byte-bound receipts. Reuse existing callable and Phase 3m fixtures where suitable; do not select the fixtures by first/newest archive.
- [ ] 1.6 Add runtime-source and transitive-dependency mutation controls in an isolated checkout: retain good A, mutate to B, show no-rebuild freshness failure, consistently rebuild and execute B so wrong behavior kills each witness, then restore/rebuild and recover. Do not use Git's shared stash or mutate another worktree.

## 2. Identity production and shared selection

- [ ] 2.1 Implement the single runtime-input descriptor producer inside the producing crates' build steps, resolving the closure from the lockfile, manifests, and workspace sources without invoking Cargo and recording only directly observed compile configuration, never values inferred from Cargo config files or profile names (D1). Prove source additions/deletions and dependency mutations invalidate it, plain `cargo build -p chelis-runtime` still produces it, and a missing or unrecordable input fails closed.
- [ ] 2.2 Embed expected descriptors in CLI/Python builds and the versioned private identity record in static runtime archives. Implement supported archive/object decoding and strict schema admission without executing candidates or adding a public numeric ABI.
- [ ] 2.3 Implement the shared resolver's unique-content selection and structured errors; verify matching archives remain usable alongside rejected candidates and all ordering/timestamp permutations preserve the outcome.
- [ ] 2.4 Implement explicit source-worktree versus sealed-distribution provenance and live-source validation. Missing development inputs cannot select distribution mode, and Python expectation cannot come from an adjacent CLI.
- [ ] 2.5 Implement verified copy/hash/header checks, atomic successful publication, the staged-archive line in the CLI's stdout artifact report (stderr stays empty on success), and staged/link receipts. Preserve the distinction between compiler image, runtime build identity, archive digest, and native artifact digest.

## 3. Producer-first and per-consumer cutover

- [ ] 3.1 Activate the shared owner for every CLI C/HIP/Metal staging caller in #2397, after shared verification and native distributions are ready. Migrate CLI-owned selectors/native drivers and execute the complete CLI staging/link/run and mutation witnesses with that slice.
- [ ] 3.2 Migrate the shared harness adapter and representation-runner pin contract in #2395, CLI-owned harnesses in #2397, CPU C/E2E/benchmark families in #2399, HIP in #2400, and Metal in #2401. Validate `CHELIS_RUNTIME_LIB` pins, including the consistent directory-plus-pin pair `scripts/runtime_representation_phase1.py::runtime_pin` sets; delete each claimed path's first/newest/canonical guessing and unchecked reuse, including no-override fallbacks. Keep evidence-backed dispositions for nonconsumers.
- [ ] 3.3 Establish common Cargo/Nix producer inputs in #2394; complete exact native package assembly in #2396. Supply the lockfile, manifests, and local sources each per-crate derivation needs, remove Nix first-match packaging, and validate archive/header identity in combined and standalone outputs and existing native package checks.
- [ ] 3.4 Update release/container assembly, installer validation and the installed-artifact canary in #2396 to preserve/check embedded identity. Exercise relocation and clean source-free native installs. Python wheel assembly and its source-free witness belong to #2398 with Python activation.
- [ ] 3.5 Rebuild/reinstall matching consumer/runtime/header distributions before activating each consumer, and invalidate incompatible compiled artifacts on its claimed path. Other consumers may remain unmigrated with #1354 open; an activated consumer never accepts an unstamped legacy fallback, and failed deployments remain loud until its matching bundle is installed.
- [ ] 3.6 Activate Python selection, verified explicit native linking, persisted receipt/native-byte admission and subsequent `load` together in #2398, with existing `ChelisError` categories. Produce the extension's own matching wheel runtime/header bundle and execute compile/load/call, persisted reload, source/dependency mutation and relocated source-free wheel witnesses. Interpreter/wheel evidence remains local/manual under the existing lane boundary.

## 4. Documentation and strict validation

- [ ] 4.1 Reconcile `spec/design/phase3m_rust_runtime_rewrite.md` and packaging/current-state docs with the decided contract; document rebuild/remedy and rollback behavior, and add the implementation changelog fragment.
- [ ] 4.2 Implement the named Python oracle's shared machinery and core rows in #2395 using existing exact-artifact/digest helpers; each consumer/package/harness scope adds its executed rows. Register the script/tests in `.config/ci-test-targets.toml`, document its authority in `docs/phase_oracles.md`, and document accelerator commands/limits in `docs/manual_gates.md`. Scoped receipts cannot pass the full aggregate while mandatory rows or prerequisites are missing.
- [ ] 4.3 Run `openspec validate bind-runtime-artifact-identity --strict --no-interactive` and `openspec validate --all --strict --no-interactive`; verify every requirement has positive and negative execution coverage or an explicitly bounded hardware row.
- [ ] 4.4 Run the authoritative runtime identity oracle in an isolated target/environment; inspect actual CLI, Python, source-mutation, producer/package, and receipt results rather than only the aggregate exit status. Keep the acceptance claim inactive until its mandatory rows pass.

## 5. Adversarial and hosted acceptance

- [ ] 5.1 Conduct a fresh local red-team round against the implementation and real witnesses: reintroduce mtime/first-match selection, bypass override validation, swap copied/cached bytes, omit a dependency from identity, and leave a stale correct runtime beside a rebuilt mutant. Repair findings and have the standing reviewer verify them.
- [ ] 5.2 Run the CLI witness, the `chelis-python` compile/load/call job, and the Nix package checks on the hosted Linux/macOS lanes that already build them, at the exact implementation head; record the wheel and interpreter-level Python witnesses as local evidence. Required prerequisites cannot silently skip; retain source/configuration identities, archive/native digests, commands, and run links.
- [ ] 5.3 Exercise C/HIP/Metal artifact staging without claiming GPU execution; run applicable existing hardware/manual lanes and record executed versus unrun configurations separately. Resolve all remaining selector dispositions before any class-closure claim.
- [ ] 5.4 Inspect exact-head hosted evidence and the mutation failures plus restored positive controls; update review/oracle records and close #1354 only after the implementation is confirmed on main. Merging this planning change or draft #2036 is not that oracle.
