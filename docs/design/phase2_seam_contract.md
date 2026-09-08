# Phase 2 seam contract: the five surfaces Beacon pins against

Status: SIGNED OFF by the Beacon shell agent (`Chelis-Lang/beacon`,
`~/Documents/scratch/beacon-bakeoff`) with one correction to ① (recorded
below). Surfaces 1, 3, 4 confirmed usable as shipped; surface 2 (`IrHandle`)
is now frozen at the hash-addressed exact-version `WireDag` v6 shape the correction
specified, and WI-3 populates against it.

The verification-stack dispatch layer (WI-3 graph-extraction producer, WI-9
`DischargeEngine` registry) exposes a small frozen API that an out-of-tree shell
engine (Beacon) consumes. Beacon pins against these surfaces; once published they
must not churn, because a shape change after Beacon pins is the expensive failure.
**Freeze first, populate second:** WI-3 populates `Goal.ir` and produces
`GoalShape::BoxRange` goals only after this contract is signed off.

All five live in `crates/chelis-prove/src/discharge.rs` on `main`/`phase2-wave1`
unless noted. Signatures below are transcribed verbatim from the shipped code.

## 1. `Goal`

```rust
pub struct Goal {
    pub shape: GoalShape,
    pub ir: IrHandle,
}

impl Goal {
    pub fn smt(property: SmtProperty) -> Self;                      // unpopulated ir
    pub fn box_range(inputs: IntervalBox, output: OutputRange)      // validates lo <= hi
        -> Result<Self, GoalError>;
    pub fn with_ir(mut self, ir: IrHandle) -> Self;                 // WI-3 producer surface
    pub fn as_smt(&self) -> Option<&SmtProperty>;
}
```

`box_range` rejects an inverted/NaN interval as `GoalError::IllFormed`. Stable.

## 2. `IrHandle`  — CONFIRMED ① (Beacon correction applied)

```rust
pub struct IrHandle {                  // private fields
    dag_hash: Option<String>,          // lowercase-hex sha256 of the serialized WireDag v6 bytes
    root_index: Option<u64>,           // which WireDag.roots entry the goal's output selects
}

impl IrHandle {
    pub fn unpopulated() -> Self;
    pub fn from_wire_dag(dag_hash: String, root_index: u64) -> Self;  // WI-3 producer surface
    pub const fn is_populated(&self) -> bool;
    pub fn dag_hash(&self) -> Option<&str>;
    pub const fn root_index(&self) -> Option<u64>;
}
```

**① RESOLVED.** Beacon's sign-off corrected the default assumption: Beacon does
not consume a `node: u64` index, nor a borrow of an in-memory `chelis_ir::Dag`.
It consumes the SERIALIZED exact-version `WireDag` JSON bytes out of process:
it parses the slice, requires the current `WIRE_DAG_SCHEMA_VERSION` (version 7
in the chelis#1277 Slice A change), validates the complete
cross-node wire contract, computes a sha256 over those bytes, and
selects the output by `root_index`. So `IrHandle` addresses that artifact by its
content hash (lowercase hex) plus a root index, NOT by a node id.

The version requirement is exact. Missing, older, and future schema versions
are rejected before any `WireRiscOp` is decoded; there is no compatibility
reader. `IrHandle`'s hash-plus-index shape remains independent of the schema
payload it addresses.

`IrHandle` still holds only a hash + index: it does NOT carry a `chelis_ir::Dag`
or a `WireDag` value, so populating it pulls no live IR dependency into
`chelis-prove`'s public API. The WI-3 producer (which DOES have the IR in scope)
serializes the exact-version `WireDag` v6, validates its version and complete
cross-node contract at the producer
boundary, hashes the bytes, and hands the digest + root index here via
`from_wire_dag`. A bare `from_node(u64)` is removed: it addressed nothing a
cross-process consumer could resolve.

**Finite-float precondition on the content-address path.** The hash is taken
over the JSON serialization of the `WireDag`, and `serde_json` serializes a
non-finite f64 (`NaN` / `+inf` / `-inf`) as the JSON token `null`. That breaks
content addressing two ways: the bytes no longer parse back as a `WireDag` (a
consumer's `from_validated_json` fails on `null`-where-`f64`-expected, *after*
the self-consistent hash already matched, so it is silent at the producer), and
`+inf` / `-inf` / `NaN` all collapse to the same `null`, so three distinct DAGs
would share one hash. **The content-address path therefore requires finite
floats: the WI-3 producer rejects any `WireDag` carrying a non-finite node-op
float at its boundary** (before serialize + hash), the same fail-closed posture
as the `schema_version` check. A canonical non-finite representation — to support
content-addressing DAGs that legitimately contain `inf`/`NaN` — is a tracked
follow-up requiring a coordinated `WireDag`-JSON-format change with Beacon's
consume path; it is out of scope for WI-3.

## 3. `GoalShape::BoxRange`  — OPEN QUESTION ② for Beacon

```rust
pub enum GoalShape {
    Smt(SmtProperty),
    BoxRange { inputs: IntervalBox, output: OutputRange },
}

pub struct IntervalBox { pub dims: Vec<(String, f64, f64)> }   // (name, lo, hi) per input
pub struct OutputRange { pub output: String, pub lo: f64, pub hi: f64 }
```

Per-named-dim `[lo, hi]` input box; a single named output asserted within
`[lo, hi]`. `f64` bounds.

**② Is this shape what Beacon's interval evaluator actually consumes?** Specifically:
- Per-named-scalar-dim boxes sufficient, or do you need affine / zonotope input
  forms, symbolic dims, or per-tensor-element boxes?
- A single named output enough, or multiple outputs / a vector output range?
- `f64` bounds acceptable (vs rational / arbitrary-precision)?

The WS-7 gradient goal (WI-10, Wave 2) reuses this same `BoxRange` shape for a
verified-Greeks-over-a-region goal — so confirming ② also fixes the gradient-goal
shape. Flag if the gradient case needs anything the forward case does not.

## 4. `DischargeEngine` trait

```rust
pub trait DischargeEngine {
    fn name(&self) -> &'static str;
    fn fitness(&self, goal: &Goal) -> bool;                          // WI-9 routing selector
    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge;
}
```

Beacon implements this for its interval engine: `fitness` returns true for
`GoalShape::BoxRange`, `discharge` runs the forward interval pass and returns a
`Discharge` (built only via `Discharge::new`, which enforces the per-qualifier
minimum-soundness map — a sound-over-approximation result carries
`Qualifier::SoundOverApproximation` at `Soundness::SoundApproximate`). Stable.

## 5. WI-9 registry external-registration entry point  — LANDED IN WAVE 2

The registry that selects an engine by `fitness()` is WI-9 (Wave 2), now landed
as `chelis_prove::DischargeRegistry`. The external-registration entry point — how
Beacon's out-of-tree engine registers so a `BoxRange` goal routes to it — is:

```rust
pub struct DischargeRegistry { /* private */ }

impl DischargeRegistry {
    pub fn new() -> Self;                                     // empty registry
    pub fn with_builtin_engines() -> Self;                   // in-tree engines for the active feature lane
    pub fn register(&mut self, engine: Box<dyn DischargeEngine>); // the Beacon rail
    pub fn dispatch(&self, goal: &Goal, timeout_ms: u64) -> Discharge;
    pub fn selected_engine_name(&self, goal: &Goal) -> Option<&'static str>;
}
```

Beacon constructs `DischargeRegistry::with_builtin_engines()` and `register`s its
boxed interval engine on top, then `dispatch`es. The registry stores boxed trait
objects (not a closed enum) precisely so an out-of-tree engine can register
without touching this crate.

**Deterministic selection / tie-break.** `dispatch` selects the FIRST registered
engine whose `fitness(goal)` is true, in registration order. Registration order
IS the priority: an engine registered earlier is selected ahead of a later one
when both fit a goal. So Beacon's interval / Arb sound lane registers ahead of any
weaker fallback to own the `BoxRange` lane. The tie-break is explicit and
registerable, not accidental.

**No-fit path (unchanged guarantee).** A goal that no registered engine fits
yields a `Discharge` at `Soundness::Untrusted` with an empty `QualifierSet` and a
`TierBResult::Error` result (built through `Discharge::new`, so the integrity
invariant holds). It projects to `CompositeVerdict::Unsupported` — never a silent
pass, never a green, never `Proven`. No new soundness or verdict variant is
introduced.

**Dual-lane preservation.** `with_builtin_engines()` registers the in-tree engine
per feature so neither feature lane changes behavior: under `--features smt` it
registers `Cvc5Engine`; in the default (solver-free) build it registers the
solver-free `SolvePropertyEngine`, which wraps the same unconditional
`solve_property` call the non-smt path used directly before WI-9. The default
binary still links zero cvc5-named symbols (the `check_is_solver_free_on_the_corpus`
gate).

## 6. WI-10 AD-as-verification-target rail  — LANDED IN WAVE 2

The AD rail (`chelis_prove::ad_rail`) makes the gradient (adjoint) graph
dispatchable like any other property. It is the gradient analogue of the WI-3
forward producer: where WI-3 turns a forward program's output into a box/range
goal, the AD rail turns the GRADIENT of a program into a fan-out of box/range
goals — one per gradient target — for a "verified-bounded-sensitivities" goal
(Greeks over an input region).

```rust
pub struct GradTargetRange { pub target: String, pub lo: f64, pub hi: f64 }
pub struct AdRailRequest {
    pub source: String,
    pub source_kind: SourceKind,
    pub output_name: String,           // the scalar output to differentiate
    pub wrt_names: Vec<String>,        // the inputs differentiated w.r.t. (each = one Greek)
    pub input_box: IntervalBox,        // shared region; emitted name-sorted (WI-3 shape)
    pub target_ranges: Vec<GradTargetRange>,
}
pub struct GradGoal { pub target: String, pub extracted: ExtractedGoal }

pub fn grad_goals_from_request(req: &AdRailRequest) -> Result<Vec<GradGoal>, AdRailError>;
pub fn dispatch_grad_goals(
    registry: &DischargeRegistry, goals: &[GradGoal], timeout_ms: u64,
) -> Vec<(String, Discharge)>;
```

**Fan-out, not packing.** `chelis_compiler_api::compiler::grad` lowers the
combined forward+backward DAG ONCE and returns `grad_nodes_by_name`: a
`BTreeMap<String, usize>` from each `wrt` input NAME to the ROOT INDEX of that
input's gradient. That map is exactly the named-roots shape
`box_range_goal_from_wire_dag` resolves against, so the rail reuses the WI-3
core verbatim. A gradient over N targets becomes N SEPARATE box/range goals
(one scalar output — one Greek — each), NOT one goal packing a vector of
sensitivities (the Beacon agent's hard requirement: an interval engine bounds
one scalar output per goal). All N goals share the ONE gradient-DAG content
hash and the same name-keyed input box, differing only in root index and output
range.

**No in-tree fit (what this wave lands).** There is no `BoxRange` engine
in-tree (cvc5 fits only `Smt`; Beacon is out-of-tree), so each gradient goal
dispatched through `with_builtin_engines()` is NO-FIT — the canonical
`Soundness::Untrusted` + empty `QualifierSet` → `CompositeVerdict::Unsupported`,
never green. This wave lands the plumbing (grad → per-target goal → dispatch)
plus the honest no-fit outcome; the real verified-Greeks discharge is Beacon's,
out of tree. An out-of-tree interval engine registered on top of
`with_builtin_engines()` routes each gradient goal to it instead, exactly like
any other `BoxRange` goal.

**Boundary checks.** The gradient `WireDag` flows through the same WI-3
fail-closed boundary as a forward DAG: a schema above `WIRE_DAG_SCHEMA_VERSION`,
a non-finite node-op float, or an inverted/NaN output range is rejected before
hashing. An unknown gradient target (a `target` not among `wrt_names`) fails the
WHOLE fan-out with `AdRailError::UnknownGradTarget` — no partial goal set, no
wrong-root goal.

## Sign-off

Beacon agent: confirm ① and ② (and that 1/4 are usable as shipped). Record the
answers here; WI-3 then populates against the frozen shapes.
