## 1. Contract and discriminating fixtures

These are implementation tasks, not completed by authoring or validating this plan.
The authoritative completion oracle is the planned command
`.venv/bin/python scripts/runtime_artifact_identity_oracle.py` described in `design.md`.
Every execution claim remains bounded to its recorded consumers/configurations.

- [ ] 1.1 Reconcile draft PR #2036 and the local issue-1354 sketches with this plan; preserve their work and do not mark unexecuted tests as evidence. Enumerate all runtime selectors/producers from D6 and disposition every occurrence.
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

## 3. Coordinated consumer and producer cutover

- [ ] 3.1 Replace CLI C/HIP/Metal and Python selectors with the shared owner; link verified staged paths explicitly in Python. Validate persisted receipts/native bytes for compiled-artifact reuse and subsequent load, with existing `ChelisError` failure categories.
- [ ] 3.2 Migrate the shared harness adapter and every inventoried C/CLI/E2E/HIP/Metal test and benchmark selector. Validate `CHELIS_RUNTIME_LIB` pins, including the consistent directory-plus-pin pair `scripts/runtime_representation_phase1.py::runtime_pin` sets for the registered runtime-representation oracle, delete obsolete first/newest/canonical-file guessing, and keep an evidence-backed disposition for nonconsumers.
- [ ] 3.3 Make Cargo and separate Nix compiler/runtime derivations consume the same runtime recipe and exact producer output; supply the lockfile, manifests, and local sources each per-crate derivation needs to produce the descriptor, and remove Nix first-match archive packaging. Validate archive/header identity in combined and standalone outputs and existing native package checks.
- [ ] 3.4 Update release/container assembly, installer validation, and installed-artifact canary to preserve/check embedded identity. Supply a matching runtime/header bundle in Python wheels; exercise relocation and clean source-free installs with no neighboring CLI as local and manual-gate evidence, since no hosted lane builds a wheel.
- [ ] 3.5 Rebuild paired consumer/runtime distributions and invalidate incompatible compiled artifacts. Do not activate a partial cutover through permissive legacy fallback; failed deployments remain loud until the matching bundle is installed.

## 4. Documentation and strict validation

- [ ] 4.1 Reconcile `spec/design/phase3m_rust_runtime_rewrite.md` and packaging/current-state docs with the decided contract; document rebuild/remedy and rollback behavior, and add the implementation changelog fragment.
- [ ] 4.2 Implement the named Python oracle using existing exact-artifact/digest helpers. Register its script/tests in `.config/ci-test-targets.toml`, document its authority in `docs/phase_oracles.md`, and document accelerator commands/limits in `docs/manual_gates.md`.
- [ ] 4.3 Run `openspec validate bind-runtime-artifact-identity --strict --no-interactive` and `openspec validate --all --strict --no-interactive`; verify every requirement has positive and negative execution coverage or an explicitly bounded hardware row.
- [ ] 4.4 Run the authoritative runtime identity oracle in an isolated target/environment; inspect actual CLI, Python, source-mutation, producer/package, and receipt results rather than only the aggregate exit status. Keep the acceptance claim inactive until its mandatory rows pass.

## 5. Adversarial and hosted acceptance

- [ ] 5.1 Conduct a fresh local red-team round against the implementation and real witnesses: reintroduce mtime/first-match selection, bypass override validation, swap copied/cached bytes, omit a dependency from identity, and leave a stale correct runtime beside a rebuilt mutant. Repair findings and have the standing reviewer verify them.
- [ ] 5.2 Run the CLI witness, the `chelis-python` compile/load/call job, and the Nix package checks on the hosted Linux/macOS lanes that already build them, at the exact implementation head; record the wheel and interpreter-level Python witnesses as local evidence. Required prerequisites cannot silently skip; retain source/configuration identities, archive/native digests, commands, and run links.
- [ ] 5.3 Exercise C/HIP/Metal artifact staging without claiming GPU execution; run applicable existing hardware/manual lanes and record executed versus unrun configurations separately. Resolve all remaining selector dispositions before any class-closure claim.
- [ ] 5.4 Inspect exact-head hosted evidence and the mutation failures plus restored positive controls; update review/oracle records and close #1354 only after the implementation is confirmed on main. Merging this planning change or draft #2036 is not that oracle.
