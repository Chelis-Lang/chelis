# Phase 2 seam contract: the five surfaces Beacon pins against

Status: PROPOSED, pending sign-off from the Beacon shell agent
(`Chelis-Lang/beacon`, `~/Documents/scratch/beacon-bakeoff`).

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

## 2. `IrHandle`  — OPEN QUESTION ① for Beacon

```rust
pub struct IrHandle { node: Option<u64> }   // private field

impl IrHandle {
    pub const fn unpopulated() -> Self;
    pub const fn from_node(node: u64) -> Self;   // WI-3 producer surface
    pub const fn is_populated(&self) -> bool;
    pub const fn node(&self) -> Option<u64>;
}
```

Today `IrHandle` is a bare node index, deliberately opaque so adding a payload
later does not change the `Goal` shape OR pull a `chelis-ir` dependency into
`chelis-prove` before a consumer needs it. It is NOT the serialized `WireDag`.

**① How does Beacon's interval evaluator resolve a populated `IrHandle` to the
actual `chelis_ir::Dag`?** Two options, pick the one your evaluator needs:
- (a) `IrHandle` stays a `node: u64` index, and the `Dag` is handed to Beacon
  separately (e.g. as a second argument or via the registry), so `chelis-prove`
  keeps zero `chelis-ir` dependency. WI-3 populates `from_node(id)`.
- (b) `IrHandle` carries a richer payload (a handle into / borrow of the
  `chelis_ir::Dag`), which pulls `chelis-ir` into `chelis-prove`'s public API.

Default assumption if unspecified: (a). Confirm or correct.

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

## 5. WI-9 registry external-registration entry point  — DESIGNED IN WAVE 2

The registry that selects an engine by `fitness()` is WI-9 (Wave 2). Its
external-registration entry point — how Beacon's out-of-tree engine registers so a
`BoxRange` goal routes to it — will be added to this contract and signed off before
WS-7's gradient-goal shape lands. Until WI-9 exists, a `BoxRange` goal with no
registered fitting engine yields a `Discharge` at `Soundness::Untrusted` with an
empty `QualifierSet`, projecting to `CompositeVerdict::Unsupported` (never a silent
pass).

## Sign-off

Beacon agent: confirm ① and ② (and that 1/4 are usable as shipped). Record the
answers here; WI-3 then populates against the frozen shapes.
