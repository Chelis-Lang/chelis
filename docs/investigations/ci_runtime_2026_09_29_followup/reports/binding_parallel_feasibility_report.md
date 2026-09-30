# Binding census parallel feasibility — 465a3623be19bd68ada6c39aa7552b63e0ccf8c3

**Verdict: feasible as a small control-flow PR candidate; speedup unmeasured.** The binding test's `bindings-discovery` process can run binding-specific work alongside live wire verification and join both current-source witnesses. No builds or tests run.

## Earliest split and required join

`capacity_census_typed.py:340-397` runs compiler-JSON, native, then discovery. `capacity_census_compiler_json.py:345-390` already executes supervised Python controls, the exact `compiler_json_payloads` libtest and compiled registration before wire. Its later binding driver/MIR collection, 22 construction controls and five MIR controls use their own compiled evidence. `verify_native_bindings` (`capacity_census_native_authority.py:175-221`) independently obtains current registration, chelis-python/chelis-abi rustdoc, MIR ownership and boundary execution.

The first *semantic* wire dependency is `capacity_census_compiler_json.py:369-372`: `SchemaWireGraph` needs the actual wire witness's documents. Compiler-JSON exposure also needs `wire.schema.classifications` (`:317-330`); `discover_bindings` requires validated compiler-JSON and native witnesses (`capacity_census_bindings.py:274-292`). Split after `bindings-discovery` checks registration provenance. Run compiler-JSON's independent phase then native verification on one worker while `verify_wire_census` runs on the calling thread. Only after both succeed, create the schema graph and final compiler-JSON witness, discover rows and write the report. The worker issues no provisional numeric authority.

## Actual artifact and lock intersections

Both modes currently use `target/agents/729-capacity-rustdoc` (`capacity_census_typed.py:37,312-321`), an intentional serial reuse enforced by `test_capacity_census_typed.py:75-130`. Parallel builds there are unsafe. Wire starts with `cargo clean -p chelis-types -p chelis-vocab -p chelis-compiler-api --target-dir <shared>` (`capacity_census_wire_adapters.py:1060-1105`). Wire and binding both produce `doc/chelis_python.json`, `debug/examples/*` and `debug/deps/*` (`capacity_census_wire_schema.py:1734-1857`; `capacity_census_typed.py:250-283`; `capacity_census_native_registration.py:116-151`). A common Cargo target lock serializes builds, while overlapping `_target_lease(<shared>)` rejects (`capacity_census_wire_adapters.py:1011-1030`). Wire cache publication seals exact `debug/deps` artifacts, which later binding builds could invalidate (`capacity_census_cache_publication.py:299-484`).

Compiled call scopes use distinct `<shared>/wire-invocations/cargo` and `<shared>/binding-invocations/cargo` paths (`capacity_census_wire_calls.py:781-850`); other Cargo/rustdoc work still intersects. Sequence locks differ; scratch names are unique.

Keep wire on the existing target for its standalone test. Select a sibling internally for **all** binding-specific work, e.g. `target/agents/729-capacity-binding-proof`. Cargo, rustdoc, binaries, native captures and receipts then have disjoint paths. Keep compiler-JSON before native inside that binding target because those scopes intentionally share its invocation Cargo directory and driver. No caller-supplied target, document or receipt grants authority.

## Verdicts and negative controls

Always await the worker, including on wire failure; report both failures and issue no final witness/report unless both pass. Require `wire.source_sha256 == binding_start == binding_end == native.source_sha256 == source_identity(root)` at join. Revalidate wire, native and final `VerifiedCompilerJsonBindings` before discovery. Preserve its private constructor and wire field, plus the Rust report-shape and baseline/bijection checks (`tests/support/capacity_census_compiler_json.rs:51-145`; `crates/chelis-python/tests/capacity_census_bindings.rs:375-585`).

Join wire schema/codec, acceptance, cache and publication receipts (`capacity_census_wire_verifier.py:116-154`); compiler-JSON Python/libtest selections, registration processes/binaries, five conversion owners, 22 construction and five MIR outcomes (`capacity_census_compiler_json.py:28-53,259-343`); and native registration, two rustdoc documents, four MIR-owned exposures, 37 selected tests and 50 captures (`capacity_census_native_authority.py:175-221`). Witnesses recheck source, artifact/provenance hashes and selection equality. Reject missing/skipped/failed cases, helper/wrong-root codecs, invalid construction/native ownership, stale source, changed artifacts/logs and missing wire classifications. Saved JSON and baselines remain comparison data only.

## Small candidate, tests and cost

Files: `scripts/capacity_census_typed.py` (sibling target, concurrent join); `scripts/capacity_census_compiler_json.py` (collection/finalization split); `scripts/test_capacity_census_typed.py` (path isolation and overlap); `scripts/test_capacity_census_compiler_json.py` (join/failure controls). The other agent's publication-probe files need no edit.

Positive: event-controlled unit test proves binding work starts before wire completes and no report precedes both; live `chelis-python::capacity_census_bindings` retains 17 rows, eight final numeric authorities and baseline equality; run standalone wire census. Negative: fail each branch separately and prove the other is awaited with no report; change source before join; corrupt an artifact/selection packet; contend for each target lease; reject a saved wire receipt. Keep existing mutation suites and Rust report-shape controls.

Cost risk: separate targets recompile shared dependencies (`capacity_census_typed.py:18-35`). Cargo cache locks, CPU contention and the wire-probe change could erase savings. Compare cold/warm wall time and peak CI CPU before claiming a cut. A serial mega-test only moves work between jobs; cross-job saved JSON cannot construct `VerifiedWireCensus`. This proposal joins concurrent proof in the source-bound binding test.
