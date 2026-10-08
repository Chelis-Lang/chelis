# Beacon subprocess shim (chelis#439)

Status: design APPROVED. This is the chelis-SIDE contract for the in-tree
transport-only shim that routes a `GoalShape::BoxRange` goal to an out-of-tree
`chelis-beacon` verifier binary. Beacon's verifier logic stays out of tree; this
document pins only the transport, the integrity mapping, and the byte flow.

Companion: `docs/design/phase2_seam_contract.md` (the five frozen surfaces this
shim consumes WITHOUT changing). This shim changes none of them.

## 1. Shape

A thin in-tree `DischargeEngine` (`BeaconShim`, in
`crates/chelis-prove/src/beacon_shim.rs`):

- `fitness(goal)` is true iff `goal.shape` is `GoalShape::BoxRange`.
- `discharge(goal, timeout_ms)` builds a request JSON, spawns the pinned
  `chelis-beacon` binary through the shared timeout supervisor, parses the
  returned `CheckReport` JSON, and maps it to a `Discharge` per §5 — all built
  through `Discharge::new` (the integrity gate).

The shim is TRANSPORT ONLY: no verifier logic, no solver linkage. It adds no
cvc5-named symbol, so the default (non-smt) build stays solver-free (the
`check_is_solver_free_on_the_corpus` gate). It is available in the DEFAULT build
(`std::process` + `serde_json` + `base64` + `sha2`).

## 2. The byte flow (content-addressed store, NO frozen-surface change)

The frozen `DischargeEngine::discharge(&self, goal: &Goal, timeout_ms)` gives the
shim only the `Goal`, whose `IrHandle` carries `dag_hash` + `root_index` and NO
bytes (frozen by the seam contract §2). The serialized exact-version `WireDag` v6 bytes the
shim must transport to Beacon live in `graph_extract::ExtractedGoal.wire_dag_bytes`,
which never enters the `Goal`.

A content-addressed `WireDagByteStore` bridges the two WITHOUT touching the
frozen seam:

- `WireDagByteStore` maps `dag_hash` (lowercase-hex sha256, the same key the WI-3
  producer computes) to the exact `wire_dag_bytes`.
- The DISPATCH SITE (the caller that runs the WI-3 producer and registers the
  shim) owns the store and populates it from `ExtractedGoal { wire_dag_bytes,
  dag_hash, .. }` at extraction time. The store is passed into
  `BeaconShim::new(binary, store)`; the shim closes over its own clone (the store
  is internally `Arc`-shared and cheap to clone).
- In `discharge`, the shim reads `goal.ir.dag_hash()` and looks the bytes up in
  the store.

This needs ZERO change to `IrHandle`, `Goal`, the `DischargeEngine` trait, or the
registry. The store is a NEW type the shim and the dispatch site share, not a
field on any frozen surface. The registry is untouched except the caller's one
`register(Box::new(BeaconShim::new(..)))` call.

Rejected alternative: a global/static store (hidden coupling, test-isolation
hazard). Rejected alternative: path-only transport (a path can drift from the
hash; the contract requires inline bytes whose decoded sha256 equals the handle's
`dag_hash`).

## 3. Binary discovery (Q2)

The `chelis-beacon` binary is located by, in order:

1. an EXPLICIT path supplied at shim construction (`BeaconShim::new`), or
2. the `CHELIS_BEACON_BIN` environment variable as a FALLBACK.

There is NO walk-up filesystem detection. If NEITHER is set, the shim is NOT
registered / does not fit: a `BoxRange` goal then takes the existing no-fit path
(`Soundness::Untrusted` + empty `QualifierSet` + `TierBResult::Error` →
`CompositeVerdict::Unsupported`), never a crash. Construction returns `None` (or
an unregistered shim) when no binary is configured; it never panics.

## 4. Request JSON (Q5/Q6 — inline base64 of the EXACT bytes)

The request handed to `chelis-beacon dispatch --request -` (stdin) or a temp file
is:

```json
{
  "schema_version": 2,
  "wire_dag_v6_base64": "<base64 of the EXACT WireDag v6 bytes from the store>",
  "expected_dag_sha256": "<goal.ir.dag_hash()>",
  "root_index": <goal.ir.root_index()>,
  "inputs": [{"name": "...", "lo": <f64>, "hi": <f64>}, ...],
  "output": {"output": "...", "lo": <f64>, "hi": <f64>},
  "oracle": <string|null>,
  "split_max_depth": null,
  "split_max_boxes": null
}
```

INVARIANTS:

- `wire_dag_v6_base64` is base64 of the EXACT bytes the store holds — NO parse,
  reformat, or re-serialize between the WI-3 bytes and the base64. The bytes the
  producer hashed are the bytes that travel.
- `expected_dag_sha256` is `goal.ir.dag_hash()`. Before trusting any report, the
  shim recomputes sha256 over the DECODED buffer and asserts it equals
  `expected_dag_sha256`. If they differ → fail-closed (§5, hash-mismatch row).
  This round-trips by construction (the store key IS the producer's hash); the
  recompute is defense-in-depth against store corruption / a wrong-keyed insert.
- `inputs` / `output` are the goal's `IntervalBox` / `OutputRange` bounds.
- `oracle` defaults to `null`, preserving the original #439 request contract.
  The verified zonotope dispatch mode is opt-in and emits exactly
  `"zonotope_verified"`; selecting it does not change the fail-closed
  CheckReport mapping in §5.

Byte-store-over-path is the Phase 2E choice; a byte-STORE on disk (rather than
inline base64) is the documented future contract IF a request-size ceiling is
ever hit. Path-only is rejected.

## 5. CheckReport → Discharge mapping (the soundness guard)

Every row is built through `Discharge::new`. The integrity principle: ONLY a
Beacon report that is itself oracle-verified earns a proof/refutation soundness;
anything unverified, errored, timed-out, or hash-mismatched is `Untrusted` and
carries NO proof qualifier. Never launder an unverified/error report into a
green.

| Beacon `CheckReport`                              | `TierBResult`              | `Soundness`        | `QualifierSet`              | verdict projection             |
|---------------------------------------------------|----------------------------|--------------------|-----------------------------|--------------------------------|
| `proved` (oracle-verified, or flag absent)        | `Proved`                   | `SoundApproximate` | `{SoundOverApproximation}`  | `sound_approximate` (NOT proven) |
| `proved` + `oracle_verified: false`               | `Error(reason)`            | `Untrusted`        | `{}` empty                  | `Unsupported` (self-contradictory, fail-closed) |
| `proved_oracle_unverified`                        | `Error(reason)`            | `Untrusted`        | `{}` empty                  | `Unsupported`                  |
| `refuted` + oracle-verified flag TRUE             | `Disproved(counterexample)`| `SoundApproximate` | `{SoundOverApproximation}`  | `disproved`/`failed`           |
| `refuted` + flag FALSE or ABSENT                  | `Disproved(counterexample)`| `Untrusted`        | `{}` empty                  | `Unsupported` (conservative)   |
| nonzero exit / spawn failure / unparseable report | `Error(stderr in evidence)`| `Untrusted`        | `{}` empty                  | `Unsupported` (fail-closed)    |
| timeout (hard kill at `timeout_ms`)               | `Error("beacon timeout")`  | `Untrusted`        | `{}` empty                  | `Unsupported` (never silent pass) |
| binary not configured (Q2)                        | — shim not registered → existing no-fit | `Untrusted` | `{}` empty           | `Unsupported`                  |
| `dag_hash` mismatch (decoded sha256 ≠ expected)   | `Error("wire_dag hash mismatch")` | `Untrusted` | `{}` empty                  | `Unsupported` (fail-closed)    |

`proved` carries `SoundOverApproximation` (Beacon's interval pass is a sound
over-approximation, never an exact proof — it projects to `sound_approximate`,
never `proven`). `proved_oracle_unverified` is the integrity guard: a proof whose
soundness oracle did not verify is `Untrusted` with NO proof qualifier, so it
cannot read as proven-modulo-anything.

The proof-side discriminator is the `proved_oracle_unverified` TOKEN, not the
`oracle_verified` flag, so a `proved` verdict with an ABSENT flag is the normal
verified case and stays `SoundApproximate`. But a `proved` verdict carrying an
explicit `oracle_verified: false` is SELF-CONTRADICTORY; rather than let it read
as `SoundApproximate` (an asymmetry a regressed Beacon could exploit), the shim
fails it closed to `Untrusted`, symmetric with the refuted arm (defense in
depth).

REFUTED soundness mirrors the proved guard on the SAME oracle-verified flag that
splits `proved` vs `proved_oracle_unverified`: an oracle-verified refutation is a
sound refutation (`SoundApproximate` + `SoundOverApproximation`, symmetric to the
proved row); an UNVERIFIED refutation, OR a refutation where Beacon's report
exposes NO verification flag, defaults to `Untrusted` (conservative — same
principle as `proved_oracle_unverified`). The result is `Disproved` either way
(the goal did not hold); only the soundness stamp differs. Fail-safe regardless
of Beacon's eventual answer.

### Beacon-contract follow-up (tracked)

Beacon should pin whether a refutation report carries an oracle-verified flag
SYMMETRIC to the proof side's `proved` vs `proved_oracle_unverified` split. Until
it does, this shim DEFAULTS an un-flagged refutation to `Untrusted`. This is a
Beacon-side contract item, carried to the Beacon agent; the shim is fail-safe
under either answer.

## 6. Failure posture (Q3/Q4 — fail-closed, never green)

- Spawn failure, nonzero exit, unparseable report → `Untrusted` +
  `TierBResult::Error`, with captured stderr in the `Discharge` evidence.
- Timeout: the shared Beacon supervisor kills and reaps the child at
  `timeout_ms`, returning `Untrusted` + `Error("beacon timeout")`. A denied
  wait or teardown operation returns a branded unsupported error.
- Input deadlock safety: the supervisor writes input to a temporary file before
  spawning. The `--request -` protocol reads that file on stdin, so a child
  that never reads cannot block the parent before its timeout starts. Large
  requests use the `--request <path>` protocol. The transport choice never
  affects the soundness mapping.
- Store miss (hash populated but bytes absent) or unpopulated `goal.ir` →
  `Untrusted` + `TierBResult::Error` (distinct from binary-not-configured, which
  is no-fit). Never a crash.
- `dag_hash` mismatch on the decoded buffer → fail-closed `Untrusted` + `Error`.

A `BeaconShim` discharge NEVER panics on a malformed report, a missing binary, a
slow binary, or a corrupt store. Every path resolves to a `Discharge` built
through `Discharge::new`.

## 7. Gating

Transport-only: the shim ships in the DEFAULT build and adds no solver-named
symbol. The `base64` crate dependency is transport
utilities and introduce no cvc5 symbol, so `check_is_solver_free_on_the_corpus`
(which greps `nm -C | grep -i cvc5`) stays green in the default build. `sha2` and
`serde_json` are already chelis-prove deps; `tempfile` is already in-workspace.

## 8. Acceptance oracle

The owning oracle for this work is the spec-first integration test suite
`crates/chelis-prove/tests/beacon_shim_mock.rs` (it is an INTEGRATION test, not
a `src/` unit module, because it needs `CARGO_BIN_EXE_mock-chelis-beacon`, which
cargo defines only for integration tests). It drives the compiled
`mock-chelis-beacon` test helper (`crates/chelis-prove/src/bin/mock_chelis_beacon.rs`)
emitting canned `CheckReport` JSON and asserts EACH §5 mapping row plus its
negative twin, including:

- `proved` → `SoundApproximate` + `SoundOverApproximation`;
- `proved` with `oracle_verified: false` → fail-closed `Untrusted` (the
  self-contradictory case, symmetric with the refuted arm);
- `proved_oracle_unverified` → `Untrusted` + empty;
- oracle-verified `refuted` → `Disproved` + `SoundApproximate`;
- unverified / no-flag `refuted` → `Disproved` + `Untrusted`;
- nonzero exit → `Untrusted` + stderr-in-evidence;
- timeout → `Untrusted` + `Error`;
- binary-not-configured → no-fit / `Unsupported` (no crash);
- `dag_hash` mismatch → fail-closed `Error`;
- a large (256 KiB+) request against a non-draining child HARD-KILLS at the
  timeout (the stdin auto-fallback to the temp-file transport; §6) and a large
  `proved` verdict is still delivered;
- request JSON carries the EXACT base64 bytes and `expected_dag_sha256 ==
  dag_hash` (no parse/reformat between the WI-3 bytes and base64).

No real `chelis-beacon` binary is required in default CI; the mock harness is
the transport/mapping oracle. The live cross-repo gate is
`crates/chelis-prove/tests/beacon_e2e.rs`, run ignored with
`CHELIS_BEACON_BIN=/path/to/chelis-beacon`. It exercises the real shim against a
real Arb-enabled Beacon binary, uses Chelis-produced WI-3 bytes, asserts
Beacon-side exact-byte evidence, and keeps corrupted-byte / wrong-range
negatives non-proof.
