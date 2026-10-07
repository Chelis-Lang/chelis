import Aeneas

/-!
# Stack budget: TLS guards versus the explicit bail record

Model proofs relating the old TLS stack-exhaustion machinery of the checker to the explicit
state of the FCIS refactor, for every native stack behaviour.

Old (`crates/chelis-types/src/infer/stack.rs`, TLS):

```rust
static RECURSION_DEPTH: Cell<usize>;                                  // RecursionDepthGuard
static STACK_EXHAUSTED: RefCell<Option<(String, Option<String>)>>;    // first bail
static STACK_SCOPE_DEPTH: Cell<usize>;                                // StackExhaustionScope
let exceeded = match stacker::remaining_stack() {
    Some(remaining) => remaining < STACK_RED_ZONE_BYTES,
    None => RecursionDepthGuard::depth() > FALLBACK_MAX_DEPTH,
};
if exceeded { … if slot.is_none() { *slot = Some((site, span_id)); } }
// StackExhaustionScope::enter: if depth == 0 { STACK_EXHAUSTED = None }; depth + 1
// drain_into: STACK_EXHAUSTED.take() -> push stack_depth_error
// Drop: RECURSION_DEPTH / STACK_SCOPE_DEPTH saturating_sub(1)   (return AND unwind)
```

New (`native.rs`, `infer/stack.rs`, `session.rs`): the same fields owned by
`NativeServices { depth }` and `DiagnosticSink { stack_bails: StackBailRecord { first,
scope_depth } }`; `frame_exhausted` = `enter_frame(); if stack_exhausted() { record(..);
exit_frame(); return true }`; the walker calls `exit_frame()` on every return path; check
entries call `enter_stack_scope` / `drain_stack_bail` / `exit_stack_scope` explicitly.
`StackBailRecord::enter_scope` (`infer/stack.rs:118-125`) resets the bail AND the services'
guarded-frame depth (`NativeServices::reset_frames`) when it enters the OUTERMOST scope, and
`Explicit.enterScope` models both resets. The services last across checks. The bail record
(scope depth and first bail) is new for each check, because each check owns a new sink.

Trusted-base assumptions (`verification/trusted_base.json`, section `runtime_assumptions`):

* RA-g: `stacker::remaining_stack()` is an arbitrary oracle (`Oracle`), one answer per guard
  check; nothing about native stack consumption is assumed.
* RA-h: the model counts depth in `Nat`. Release builds wrap `depth += 1` (`native.rs:110`),
  while the Lean model of `enter_frame` fails at `usize::MAX` (`cn_enter_frame_overflow`).
* RA-d: `stacker::grow` runs its closure on the same thread and returns its value, so
  `with_grown_stack` is plain application.

Section C maps every run over an `Oracle` to a run over a schedule of the Lean model
`chelis_types.native.NativeServices` (field `stack`, clock `stackTaken`), and relates each step
to the Lean models of `enter_frame`, `stack_exhausted`, `exit_frame` and `reset_frames`.
-/

namespace ChelisBridge.Runtime.Stack

/-- Walker site label (`&str` in Rust). -/
abbrev Site := Nat
/-- Optional external-source span id. -/
abbrev SpanId := Option Nat
/-- A recorded bail `(site, span_id)`. -/
abbrev Bail := Site × SpanId
/-- Native thread identity. -/
abbrev ThreadId := Nat

/-- `STACK_RED_ZONE_BYTES`. -/
def STACK_RED_ZONE_BYTES : Nat := 128 * 1024
/-- `FALLBACK_MAX_DEPTH`. -/
def FALLBACK_MAX_DEPTH : Nat := 20

/-- One answer of `stacker::remaining_stack()`. -/
abbrev Probe := Option Nat
/-- Any native stack behaviour: the `n`-th guard check on the thread sees `oracle n`. -/
abbrev Oracle := Nat → Probe

/-- The guard predicate shared by `stack_guard_tripped` (old) and
`NativeServices::stack_exhausted` (new). -/
def exceeded (probe : Probe) (depth : Nat) : Bool :=
  match probe with
  | some remaining => decide (remaining < STACK_RED_ZONE_BYTES)
  | none => decide (depth > FALLBACK_MAX_DEPTH)

/-- First-write-wins record update. -/
def recordBail (b : Bail) : Option Bail → Option Bail
  | none => some b
  | some x => some x

/-- Checker control relevant to the stack machinery. -/
inductive Walk where
  /-- Observable non-recursive work. -/
  | work (e : Nat)
  | seq (a b : Walk)
  /-- A guarded recursive walker: `stack_guard!(site, expr, bail)` then `body`. -/
  | guarded (site : Site) (span : SpanId) (body : Walk)
  /-- An early `return` / `?` out of the innermost guarded walker or check unit. -/
  | early
  /-- A public check entry: scope enter, pipeline `body`, drain, scope exit. -/
  | unit (body : Walk)
  /-- An unexpected panic. -/
  | panic
  deriving Repr

/-- How a walk exits. -/
inductive Exit where
  | normal
  | early
  | unwind
  deriving DecidableEq, Repr

/-- Observable events: work and the bail each drain reports (`None`: nothing pushed). -/
inductive Ev where
  | work (e : Nat)
  | drained (b : Option Bail)
  deriving DecidableEq, Repr

/-- A function (walker or check entry) catches `early`: it returns normally. -/
def Exit.catchEarly : Exit → Exit
  | .early => .normal
  | e => e

/-! ## Old semantics: per-thread TLS and RAII guards -/

/-- The old TLS of one thread, plus the ghost oracle clock `tick`. -/
structure Tls where
  /-- `RECURSION_DEPTH`. -/
  recDepth : Nat
  /-- `STACK_SCOPE_DEPTH`. -/
  scope : Nat
  /-- `STACK_EXHAUSTED`. -/
  exhausted : Option Bail
  /-- Number of guard checks this thread has made (index into its oracle). -/
  tick : Nat
  deriving DecidableEq, Repr

/-- `RecursionDepthGuard::enter()` followed by the probe (`tick` advances). -/
def Tls.enterFrame (x : Tls) : Tls := { x with recDepth := x.recDepth + 1, tick := x.tick + 1 }
/-- Tripped guard: first-write-wins record, then the guard's drop on `return $bail`. -/
def Tls.bail (b : Bail) (x : Tls) : Tls :=
  { x with exhausted := recordBail b x.exhausted, recDepth := x.recDepth - 1 }
/-- `RecursionDepthGuard::drop`: `saturating_sub(1)`. -/
def Tls.exitFrame (x : Tls) : Tls := { x with recDepth := x.recDepth - 1 }
/-- `StackExhaustionScope::enter()`: reset only when entering the outermost scope. -/
def Tls.enterScope (x : Tls) : Tls :=
  { x with exhausted := if x.scope = 0 then none else x.exhausted, scope := x.scope + 1 }
/-- `drain_into` (`take`) followed by the scope's drop. -/
def Tls.drainExit (x : Tls) : Tls := { x with exhausted := none, scope := x.scope - 1 }
/-- `StackExhaustionScope::drop` alone (unwinding: no drain). -/
def Tls.exitScope (x : Tls) : Tls := { x with scope := x.scope - 1 }

/-- All threads' TLS. -/
abbrev Threads := ThreadId → Tls

/-- Update one thread's TLS. -/
def modifyAt (T : Threads) (t : ThreadId) (f : Tls → Tls) : Threads :=
  fun u => if u = t then f (T t) else T u

@[simp] theorem modifyAt_same (T : Threads) (t : ThreadId) (f : Tls → Tls) :
    modifyAt T t f t = f (T t) := by simp [modifyAt]

@[simp] theorem modifyAt_other (T : Threads) (t u : ThreadId) (f : Tls → Tls) (h : u ≠ t) :
    modifyAt T t f u = T u := by simp [modifyAt, h]

/-- Old semantics on thread `t`. RAII drops (`RecursionDepthGuard`, `StackExhaustionScope`) run on
every exit including unwinding; `drain_into` runs only on returning paths. The guard reads the
probe and the depth after `RecursionDepthGuard::enter()` (depth + 1). -/
def oldExec (o : Oracle) (t : ThreadId) : Walk → Threads → Threads × List Ev × Exit
  | .work e, T => (T, [.work e], .normal)
  | .seq a b, T =>
      match oldExec o t a T with
      | (T1, tr1, .normal) =>
          match oldExec o t b T1 with
          | (T2, tr2, e) => (T2, tr1 ++ tr2, e)
      | (T1, tr1, e) => (T1, tr1, e)
  | .guarded site span body, T =>
      if exceeded (o (T t).tick) ((T t).recDepth + 1) then
        (modifyAt (modifyAt T t Tls.enterFrame) t (Tls.bail (site, span)), [], .normal)
      else
        match oldExec o t body (modifyAt T t Tls.enterFrame) with
        | (T3, tr, e) => (modifyAt T3 t Tls.exitFrame, tr, e.catchEarly)
  | .early, T => (T, [], .early)
  | .unit body, T =>
      match oldExec o t body (modifyAt T t Tls.enterScope) with
      | (T2, tr, .unwind) => (modifyAt T2 t Tls.exitScope, tr, .unwind)
      | (T2, tr, _) => (modifyAt T2 t Tls.drainExit, tr ++ [.drained (T2 t).exhausted], .normal)
  | .panic, T => (T, [], .unwind)

/-! ## New semantics: explicit state threaded by `&mut` -/

/-- `NativeServices.depth`, `StackBailRecord { first, scope_depth }`, and the oracle clock. -/
structure Explicit where
  depth : Nat
  scopeDepth : Nat
  first : Option Bail
  tick : Nat
  deriving DecidableEq, Repr

/-- `enter_frame()` followed by `stack_exhausted()` (`tick` advances). -/
def Explicit.enterFrame (N : Explicit) : Explicit := { N with depth := N.depth + 1, tick := N.tick + 1 }
/-- `record(site, span)` then `exit_frame()` inside `frame_exhausted`. -/
def Explicit.bail (b : Bail) (N : Explicit) : Explicit :=
  { N with first := recordBail b N.first, depth := N.depth - 1 }
/-- `exit_frame()`: `saturating_sub(1)`. -/
def Explicit.exitFrame (N : Explicit) : Explicit := { N with depth := N.depth - 1 }
/-- `enter_scope()`: entering the OUTERMOST scope resets the bail record and the services'
guarded-frame depth (`reset_frames`); a nested entry only counts the scope. -/
def Explicit.enterScope (N : Explicit) : Explicit :=
  { N with depth := if N.scopeDepth = 0 then 0 else N.depth,
           first := if N.scopeDepth = 0 then none else N.first, scopeDepth := N.scopeDepth + 1 }
/-- `drain_stack_bail()` (`take_error`) then `exit_stack_scope()`. -/
def Explicit.drainExit (N : Explicit) : Explicit :=
  { N with first := none, scopeDepth := N.scopeDepth - 1 }

/-- New semantics: every depth/scope exit is an explicit call on a returning path; nothing runs
during unwinding (the invocation's explicit state is abandoned with it). -/
def newExec (o : Oracle) : Walk → Explicit → Explicit × List Ev × Exit
  | .work e, N => (N, [.work e], .normal)
  | .seq a b, N =>
      match newExec o a N with
      | (N1, tr1, .normal) =>
          match newExec o b N1 with
          | (N2, tr2, e) => (N2, tr1 ++ tr2, e)
      | (N1, tr1, e) => (N1, tr1, e)
  | .guarded site span body, N =>
      if exceeded (o N.tick) (N.depth + 1) then
        (N.enterFrame.bail (site, span), [], .normal)
      else
        match newExec o body N.enterFrame with
        | (N3, tr, .unwind) => (N3, tr, .unwind)
        | (N3, tr, e) => (N3.exitFrame, tr, e.catchEarly)
  | .early, N => (N, [], .early)
  | .unit body, N =>
      match newExec o body N.enterScope with
      | (N2, tr, .unwind) => (N2, tr, .unwind)
      | (N2, tr, _) => (N2.drainExit, tr ++ [.drained N2.first], .normal)
  | .panic, N => (N, [], .unwind)

/-- The old TLS slot of a thread as the new explicit state. -/
def proj (x : Tls) : Explicit :=
  { depth := x.recDepth, scopeDepth := x.scope, first := x.exhausted, tick := x.tick }

@[simp] theorem proj_enterFrame (x : Tls) : proj x.enterFrame = (proj x).enterFrame := rfl
@[simp] theorem proj_bail (b : Bail) (x : Tls) : proj (x.bail b) = (proj x).bail b := rfl
@[simp] theorem proj_exitFrame (x : Tls) : proj x.exitFrame = (proj x).exitFrame := rfl
/-- The old outermost scope entry keeps `RECURSION_DEPTH`, the new one resets the depth: they
agree when the outermost entry happens at depth 0. -/
theorem proj_enterScope (x : Tls) (h : x.scope = 0 → x.recDepth = 0) :
    proj x.enterScope = (proj x).enterScope := by
  by_cases hs : x.scope = 0
  · simp [proj, Tls.enterScope, Explicit.enterScope, hs, h hs]
  · simp [proj, Tls.enterScope, Explicit.enterScope, hs]
@[simp] theorem proj_drainExit (x : Tls) : proj x.drainExit = (proj x).drainExit := rfl

/-! ## B. Equivalence for every oracle -/

/-- Old RAII bookkeeping is balanced on EVERY exit path, including unwinding. -/
theorem old_balanced (o : Oracle) (t : ThreadId) (w : Walk) (T : Threads) :
    ((oldExec o t w T).1 t).recDepth = (T t).recDepth ∧
    ((oldExec o t w T).1 t).scope = (T t).scope := by
  induction w generalizing T with
  | work e => simp [oldExec]
  | seq a b iha ihb =>
    obtain ⟨a1, a2⟩ := iha T
    rcases hA : oldExec o t a T with ⟨T1, tr1, e1⟩
    rw [hA] at a1 a2
    cases e1 with
    | normal =>
      obtain ⟨b1, b2⟩ := ihb T1
      rcases hB : oldExec o t b T1 with ⟨T2, tr2, e2⟩
      rw [hB] at b1 b2
      simp only [oldExec, hA, hB]
      exact ⟨b1.trans a1, b2.trans a2⟩
    | early => simp only [oldExec, hA]; exact ⟨a1, a2⟩
    | unwind => simp only [oldExec, hA]; exact ⟨a1, a2⟩
  | guarded site span body ih =>
    by_cases hex : exceeded (o (T t).tick) ((T t).recDepth + 1) = true
    · simp [oldExec, hex, Tls.enterFrame, Tls.bail]
    · obtain ⟨b1, b2⟩ := ih (modifyAt T t Tls.enterFrame)
      rcases hB : oldExec o t body (modifyAt T t Tls.enterFrame) with ⟨T3, tr, e⟩
      rw [hB] at b1 b2
      simp only [oldExec, hex, if_false, Bool.false_eq_true, hB, modifyAt_same] at b1 b2 ⊢
      simp only [Tls.exitFrame, Tls.enterFrame] at b1 b2 ⊢
      exact ⟨by omega, b2⟩
  | early => simp [oldExec]
  | unit body ih =>
    obtain ⟨b1, b2⟩ := ih (modifyAt T t Tls.enterScope)
    rcases hB : oldExec o t body (modifyAt T t Tls.enterScope) with ⟨T2, tr, e⟩
    rw [hB] at b1 b2
    simp only [modifyAt_same, Tls.enterScope] at b1 b2
    cases e <;> simp [oldExec, hB, Tls.exitScope, Tls.drainExit, b1, b2]
  | panic => simp [oldExec]

/-- Every OUTERMOST check unit of the walk starts at recursion depth 0, for a walk run from TLS
scope depth `s` and recursion depth `d`. Old RAII bookkeeping is balanced (`old_balanced`), so
these depths are static: each guarded walker adds one frame, each check unit one scope. Public
check entries satisfy it: they open the outermost unit outside every guarded walker. -/
def BaseUnits : Nat → Nat → Walk → Prop
  | _, _, .work _ => True
  | s, d, .seq a b => BaseUnits s d a ∧ BaseUnits s d b
  | s, d, .guarded _ _ body => BaseUnits s (d + 1) body
  | _, _, .early => True
  | s, d, .unit body => (s = 0 → d = 0) ∧ BaseUnits (s + 1) d body
  | _, _, .panic => True

/-- Inside an open scope no unit is outermost, so the condition holds for every walk. -/
theorem baseUnits_of_scope_ne_zero (s d : Nat) (w : Walk) (hs : s ≠ 0) : BaseUnits s d w := by
  induction w generalizing s d with
  | work _ => simp only [BaseUnits]
  | seq a b iha ihb => simp only [BaseUnits]; exact ⟨iha s d hs, ihb s d hs⟩
  | guarded _ _ body ih => simp only [BaseUnits]; exact ih s (d + 1) hs
  | early => simp only [BaseUnits]
  | unit body ih => simp only [BaseUnits]; exact ⟨fun h => absurd h hs, ih (s + 1) d (by omega)⟩
  | panic => simp only [BaseUnits]

/-- **B.** For every oracle (native stack behaviour), every walk whose outermost check units
start at recursion depth 0 (`BaseUnits`), and every thread state: the old TLS machinery and the
explicit state produce the same observable events — in particular the same reported FIRST bail
`(site, span)` at every drain — and the same exit; on every returning exit the TLS of the thread
equals the explicit state (record, reset/drain, depth and scope bookkeeping); and other threads'
TLS are untouched. -/
theorem first_bail_equivalence (o : Oracle) (t : ThreadId) (w : Walk) (T : Threads)
    (hw : BaseUnits (T t).scope (T t).recDepth w) :
    (oldExec o t w T).2 = (newExec o w (proj (T t))).2 ∧
    ((oldExec o t w T).2.2 ≠ .unwind →
      proj ((oldExec o t w T).1 t) = (newExec o w (proj (T t))).1) ∧
    (∀ u, u ≠ t → (oldExec o t w T).1 u = T u) := by
  induction w generalizing T with
  | work e => simp [oldExec, newExec]
  | seq a b iha ihb =>
    simp only [BaseUnits] at hw
    obtain ⟨a1, a2, a3⟩ := iha T hw.1
    obtain ⟨hd, hs⟩ := old_balanced o t a T
    rcases hA : oldExec o t a T with ⟨T1, tr1, e1⟩
    rcases hN : newExec o a (proj (T t)) with ⟨N1, ntr1, ne1⟩
    rw [hA, hN] at a1 a2
    rw [hA] at a3 hd hs
    simp only [Prod.mk.injEq] at a1 a2 a3 hd hs
    obtain ⟨rfl, rfl⟩ := a1
    cases e1 with
    | normal =>
      have hrel : proj (T1 t) = N1 := a2 (by simp)
      obtain ⟨b1, b2, b3⟩ := ihb T1 (by rw [hd, hs]; exact hw.2)
      rw [hrel] at b1 b2
      rcases hB : oldExec o t b T1 with ⟨T2, tr2, e2⟩
      rcases hNB : newExec o b N1 with ⟨N2, ntr2, ne2⟩
      rw [hB, hNB] at b1 b2
      rw [hB] at b3
      simp only [Prod.mk.injEq] at b1 b2 b3
      obtain ⟨rfl, rfl⟩ := b1
      simp only [oldExec, newExec, hA, hN, hB, hNB]
      exact ⟨by first | trivial | rfl, b2, fun u hu => (b3 u hu).trans (a3 u hu)⟩
    | early =>
      simp only [oldExec, newExec, hA, hN]
      exact ⟨by first | trivial | rfl, fun _ => a2 (by simp), a3⟩
    | unwind =>
      simp only [oldExec, newExec, hA, hN]
      exact ⟨by first | trivial | rfl, fun h => absurd rfl h, a3⟩
  | guarded site span body ih =>
    simp only [BaseUnits] at hw
    by_cases hex : exceeded (o (T t).tick) ((T t).recDepth + 1) = true
    · have hex' : exceeded (o (proj (T t)).tick) ((proj (T t)).depth + 1) = true := hex
      simp only [oldExec, newExec, hex, hex', if_true]
      refine ⟨by first | trivial | rfl, fun _ => by simp, fun u hu => by simp [hu]⟩
    · have hex' : ¬ exceeded (o (proj (T t)).tick) ((proj (T t)).depth + 1) = true := hex
      obtain ⟨b1, b2, b3⟩ := ih (modifyAt T t Tls.enterFrame) (by simpa [Tls.enterFrame] using hw)
      simp only [modifyAt_same, proj_enterFrame] at b1 b2
      rcases hB : oldExec o t body (modifyAt T t Tls.enterFrame) with ⟨T3, tr, e⟩
      rcases hNB : newExec o body (proj (T t)).enterFrame with ⟨N3, ntr, ne⟩
      rw [hB, hNB] at b1 b2
      rw [hB] at b3
      simp only [Prod.mk.injEq] at b1 b2 b3
      obtain ⟨rfl, rfl⟩ := b1
      simp only [oldExec, newExec, hex, hex', if_false, Bool.false_eq_true, hB, hNB]
      cases e with
      | unwind =>
        exact ⟨by first | trivial | rfl, fun h => absurd rfl h, fun u hu => by simp [hu, b3 u hu]⟩
      | normal =>
        exact ⟨by first | trivial | rfl, fun _ => by simp [← b2 (by simp)], fun u hu => by simp [hu, b3 u hu]⟩
      | early =>
        exact ⟨by first | trivial | rfl, fun _ => by simp [← b2 (by simp)], fun u hu => by simp [hu, b3 u hu]⟩
  | early => simp [oldExec, newExec]
  | unit body ih =>
    simp only [BaseUnits] at hw
    obtain ⟨b1, b2, b3⟩ := ih (modifyAt T t Tls.enterScope) (by simpa [Tls.enterScope] using hw.2)
    simp only [modifyAt_same, proj_enterScope _ hw.1] at b1 b2
    rcases hB : oldExec o t body (modifyAt T t Tls.enterScope) with ⟨T2, tr, e⟩
    rcases hNB : newExec o body (proj (T t)).enterScope with ⟨N2, ntr, ne⟩
    rw [hB, hNB] at b1 b2
    rw [hB] at b3
    simp only [Prod.mk.injEq] at b1 b2 b3
    obtain ⟨rfl, rfl⟩ := b1
    simp only [oldExec, newExec, hB, hNB]
    cases e with
    | unwind =>
      exact ⟨by first | trivial | rfl, fun h => absurd rfl h, fun u hu => by simp [hu, b3 u hu]⟩
    | normal =>
      have hrel := b2 (by simp)
      refine ⟨?_, fun _ => ?_, fun u hu => by simp [hu, b3 u hu]⟩
      · rw [← hrel]; rfl
      · simp [← hrel]
    | early =>
      have hrel := b2 (by simp)
      refine ⟨?_, fun _ => ?_, fun u hu => by simp [hu, b3 u hu]⟩
      · rw [← hrel]; rfl
      · simp [← hrel]
  | panic => simp [oldExec, newExec]

/-- The explicit bookkeeping is balanced on every returning exit of a walk whose outermost
check units start at depth 0. (An outermost unit entered at a nonzero depth returns at depth 0:
`enter_scope` resets the depth.) -/
theorem new_balanced (o : Oracle) (w : Walk) (N : Explicit)
    (hw : BaseUnits N.scopeDepth N.depth w)
    (h : (newExec o w N).2.2 ≠ .unwind) :
    (newExec o w N).1.depth = N.depth ∧ (newExec o w N).1.scopeDepth = N.scopeDepth := by
  let x : Tls := { recDepth := N.depth, scope := N.scopeDepth, exhausted := N.first, tick := N.tick }
  let T : Threads := fun _ => x
  have hN : proj (T 0) = N := rfl
  obtain ⟨e1, e2, _⟩ := first_bail_equivalence o 0 w T hw
  obtain ⟨b1, b2⟩ := old_balanced o 0 w T
  rw [hN] at e1 e2
  have hexit : (oldExec o 0 w T).2.2 ≠ .unwind := by rw [e1]; exact h
  have := e2 hexit
  rw [← this]
  exact ⟨b1, b2⟩

/-! ## Invocation boundary: fresh explicit services versus reused thread TLS -/

/-- A fresh invocation's explicit state: `NativeServices::new(..)` (depth 0) and
`StackBailRecord::default()`. -/
def fresh (tick : Nat) : Explicit := { depth := 0, scopeDepth := 0, first := none, tick := tick }

/-- A top-level check entry on a thread whose TLS is at the base (depth 0, no open scope —
which `old_balanced` re-establishes after every invocation, including unwinding ones) behaves
exactly like the new entry on FRESH explicit services: a stale `STACK_EXHAUSTED` left by an
earlier unwound invocation is reset by the outermost scope. -/
theorem invocation_boundary (o : Oracle) (t : ThreadId) (body : Walk) (T : Threads)
    (hrec : (T t).recDepth = 0) (hscope : (T t).scope = 0) :
    (oldExec o t (.unit body) T).2 = (newExec o (.unit body) (fresh (T t).tick)).2 ∧
    ((oldExec o t (.unit body) T).1 t).recDepth = 0 ∧ ((oldExec o t (.unit body) T).1 t).scope = 0 := by
  have hw : BaseUnits (T t).scope (T t).recDepth (.unit body) := by
    simp only [BaseUnits]
    exact ⟨fun _ => hrec, baseUnits_of_scope_ne_zero _ _ _ (by omega)⟩
  obtain ⟨e1, _, _⟩ := first_bail_equivalence o t (.unit body) T hw
  obtain ⟨b1, b2⟩ := old_balanced o t (.unit body) T
  refine ⟨?_, b1.trans hrec, b2.trans hscope⟩
  rw [e1]
  have : proj (T t) = { fresh (T t).tick with first := (T t).exhausted } := by
    simp [proj, fresh, hrec, hscope]
  rw [this]
  simp [newExec, fresh, Explicit.enterScope]

/-- Counterexample for the base-state hypothesis of `invocation_boundary`: on a platform that
cannot report remaining stack, a thread entering an invocation at TLS depth 20 trips the first
guard, while fresh explicit services (depth 0) do not. -/
theorem nonzero_entry_depth_counterexample :
    let T : Threads := fun _ => { recDepth := 20, scope := 0, exhausted := none, tick := 0 }
    let w := Walk.unit (.guarded 7 none (.work 1))
    (oldExec (fun _ => none) 0 w T).2.1 = [.drained (some (7, none))] ∧
    (newExec (fun _ => none) w (fresh 0)).2.1 = [.work 1, .drained none] := by
  decide

/-- **O11.** At an outermost check unit (its bail record is new, so the scope depth is 0) any
explicit state behaves exactly like fresh services with the same oracle clock: events, exit and
final state agree, because `enter_scope` resets both the bail and the guarded-frame depth. -/
theorem outermost_unit_forgets_state (o : Oracle) (body : Walk) (N : Explicit)
    (h : N.scopeDepth = 0) :
    newExec o (.unit body) N = newExec o (.unit body) (fresh N.tick) := by
  have : N.enterScope = (fresh N.tick).enterScope := by
    simp [Explicit.enterScope, fresh, h]
  simp only [newExec, this]

/-- **O11.** Services reused after ANY earlier run, in particular after a check that unwound out
of guarded walkers and so never ran their frame exits, give the next check (with its new bail
record, `StackBailRecord::default()`) the same events, exit and final state as fresh services with
the same oracle clock. Native counterpart: the `native.rs` test
`check_after_a_caught_unwind_starts_like_fresh_services`. -/
theorem reuse_after_unwind (o : Oracle) (prev body : Walk) (N : Explicit) :
    newExec o (.unit body) { (newExec o prev N).1 with scopeDepth := 0, first := none } =
      newExec o (.unit body) (fresh (newExec o prev N).1.tick) :=
  outermost_unit_forgets_state o body _ rfl

/-- `n` nested guarded walkers around `w`. -/
def descend : Nat → Walk → Walk
  | 0, w => w
  | n + 1, w => .guarded 0 none (descend n w)

/-- **O11 witness** (the scenario of the native test). On a platform that cannot report remaining
stack, a check that unwinds out of `FALLBACK_MAX_DEPTH` guarded walkers leaves the services at
depth 20, where the next guard check would trip the fallback cap. The outermost scope of the
next check resets the depth, so its guarded walker recurses and its drain reports no bail. -/
theorem reuse_after_unwind_witness :
    let o : Oracle := fun _ => none
    let prev := Walk.unit (descend FALLBACK_MAX_DEPTH .panic)
    let N1 := (newExec o prev (fresh 0)).1
    (newExec o prev (fresh 0)).2.2 = .unwind ∧ N1.depth = FALLBACK_MAX_DEPTH ∧
      exceeded none (N1.depth + 1) = true ∧
      (newExec o (.unit (.guarded 7 none (.work 1))) { N1 with scopeDepth := 0, first := none }).2 =
        ([.work 1, .drained none], .normal) := by
  decide

/-! ## Reset/drain for nested scopes -/

/-- A nested check unit (scope depth > 0) does not reset the outer bail, and the drain reports
the recorded first bail. -/
theorem nested_unit_keeps_bail (o : Oracle) (N : Explicit) (b : Bail)
    (hscope : N.scopeDepth ≠ 0) (hfirst : N.first = some b) :
    (newExec o (.unit (.work 0)) N).2.1 = [.work 0, .drained (some b)] := by
  simp [newExec, Explicit.enterScope, hscope, hfirst]

/-- Syntactic predicate: no check unit inside. -/
def NoUnit : Walk → Prop
  | .work _ => True
  | .seq a b => NoUnit a ∧ NoUnit b
  | .guarded _ _ body => NoUnit body
  | .early => True
  | .unit _ => False
  | .panic => True

/-- First write wins: within one check unit, once a bail is recorded it is never replaced. -/
theorem first_write_wins (o : Oracle) (w : Walk) (hw : NoUnit w) (N : Explicit) (b : Bail)
    (h : N.first = some b) : (newExec o w N).1.first = some b := by
  induction w generalizing N with
  | work e => simpa [newExec] using h
  | seq a c iha ihc =>
    simp only [NoUnit] at hw
    have ha := iha hw.1 N h
    rcases hA : newExec o a N with ⟨N1, tr1, e1⟩
    rw [hA] at ha
    cases e1 with
    | normal =>
      have hc := ihc hw.2 N1 ha
      rcases hC : newExec o c N1 with ⟨N2, tr2, e2⟩
      rw [hC] at hc
      simp only [newExec, hA, hC]
      exact hc
    | early => simp only [newExec, hA]; exact ha
    | unwind => simp only [newExec, hA]; exact ha
  | guarded site span body ih =>
    simp only [NoUnit] at hw
    by_cases hex : exceeded (o N.tick) (N.depth + 1) = true
    · simp [newExec, hex, h, recordBail, Explicit.bail, Explicit.enterFrame]
    · have hb := ih hw N.enterFrame h
      rcases hB : newExec o body N.enterFrame with ⟨N3, tr, e⟩
      rw [hB] at hb
      simp only [newExec, hex, if_false, Bool.false_eq_true, hB]
      cases e <;> simpa [Explicit.exitFrame] using hb
  | early => simpa [newExec] using h
  | unit body _ => exact absurd hw id
  | panic => simpa [newExec] using h

/-! ## C. Lean-model stack schedule (`chelis_types.native.NativeServices`)

The Lean model of `NativeServices::stack_exhausted` returns the next element of the oracle
schedule `stack` and advances its clock `stackTaken`. The `Explicit.tick` of a run is that
clock. `probes` lists the native guard answers of a run (`exceeded(remaining_stack(), depth)`,
`native.rs:99-104`), and `cnExec` runs the same walk with the answers read from a schedule.
Every run over an `Oracle` is a run over the schedule of its own probes
(`cnExec_of_probes`, `oracle_run_schedule`). The `cn_*` lemmas relate each step of the
explicit state to the Lean models of the primitives. -/

/-- `newExec` with the guard answer read from the schedule `s` at the oracle clock (the Lean
model of `stack_exhausted`) instead of `exceeded (o tick) (depth + 1)`. -/
def cnExec (s : Nat → Bool) : Walk → Explicit → Explicit × List Ev × Exit
  | .work e, N => (N, [.work e], .normal)
  | .seq a b, N =>
      match cnExec s a N with
      | (N1, tr1, .normal) =>
          match cnExec s b N1 with
          | (N2, tr2, e) => (N2, tr1 ++ tr2, e)
      | (N1, tr1, e) => (N1, tr1, e)
  | .guarded site span body, N =>
      if s N.tick then
        (N.enterFrame.bail (site, span), [], .normal)
      else
        match cnExec s body N.enterFrame with
        | (N3, tr, .unwind) => (N3, tr, .unwind)
        | (N3, tr, e) => (N3.exitFrame, tr, e.catchEarly)
  | .early, N => (N, [], .early)
  | .unit body, N =>
      match cnExec s body N.enterScope with
      | (N2, tr, .unwind) => (N2, tr, .unwind)
      | (N2, tr, _) => (N2.drainExit, tr ++ [.drained N2.first], .normal)
  | .panic, N => (N, [], .unwind)

/-- The guard answers of a `newExec` run, in clock order. -/
def probes (o : Oracle) : Walk → Explicit → List Bool
  | .work _, _ => []
  | .seq a b, N =>
      match newExec o a N with
      | (N1, _, .normal) => probes o a N ++ probes o b N1
      | _ => probes o a N
  | .guarded _ _ body, N =>
      if exceeded (o N.tick) (N.depth + 1) then [true] else false :: probes o body N.enterFrame
  | .early, _ => []
  | .unit body, N => probes o body N.enterScope
  | .panic, _ => []

/-- Each guard check advances the oracle clock by one, and nothing else does. -/
theorem newExec_tick (o : Oracle) (w : Walk) (N : Explicit) :
    (newExec o w N).1.tick = N.tick + (probes o w N).length := by
  induction w generalizing N with
  | work e => simp [newExec, probes]
  | seq a b iha ihb =>
    have ha := iha N
    rcases hA : newExec o a N with ⟨N1, tr1, e1⟩
    rw [hA] at ha
    simp only at ha
    cases e1 with
    | normal =>
      have hb := ihb N1
      rcases hB : newExec o b N1 with ⟨N2, tr2, e2⟩
      rw [hB] at hb
      simp only at hb
      simp only [newExec, probes, hA, hB, List.length_append]
      omega
    | early => simp only [newExec, probes, hA]; exact ha
    | unwind => simp only [newExec, probes, hA]; exact ha
  | guarded site span body ih =>
    by_cases hex : exceeded (o N.tick) (N.depth + 1) = true
    · simp [newExec, probes, hex, Explicit.enterFrame, Explicit.bail]
    · have hb := ih N.enterFrame
      rcases hB : newExec o body N.enterFrame with ⟨N3, tr, e⟩
      rw [hB] at hb
      have ht : N.enterFrame.tick = N.tick + 1 := rfl
      simp only [ht] at hb
      cases e <;>
        simp only [newExec, probes, hex, if_false, Bool.false_eq_true, hB, List.length_cons,
          Explicit.exitFrame] <;> omega
  | early => simp [newExec, probes]
  | unit body ih =>
    have hb := ih N.enterScope
    rcases hB : newExec o body N.enterScope with ⟨N2, tr, e⟩
    rw [hB] at hb
    have ht : N.enterScope.tick = N.tick := rfl
    simp only [ht] at hb
    cases e <;> simp only [newExec, probes, hB, Explicit.drainExit] <;> omega
  | panic => simp [newExec, probes]

/-- **O12(a).** A schedule that agrees with the native guard answers of a run (each answer at
its clock position) gives the same run: events, exit and final explicit state. -/
theorem cnExec_of_probes (o : Oracle) (s : Nat → Bool) (w : Walk) (N : Explicit)
    (hs : ∀ i b, (probes o w N)[i]? = some b → s (N.tick + i) = b) :
    cnExec s w N = newExec o w N := by
  induction w generalizing N with
  | work e => rfl
  | seq a b iha ihb =>
    have tickA := newExec_tick o a N
    rcases hA : newExec o a N with ⟨N1, tr1, e1⟩
    rw [hA] at tickA
    simp only at tickA
    cases e1 with
    | normal =>
      have hp : probes o (.seq a b) N = probes o a N ++ probes o b N1 := by
        simp only [probes, hA]
      rw [hp] at hs
      have ha : cnExec s a N = newExec o a N := iha N fun i x h => by
        have hi := (List.getElem?_eq_some_iff.mp h).1
        exact hs i x (by rw [List.getElem?_append_left hi]; exact h)
      have hb : cnExec s b N1 = newExec o b N1 := ihb N1 fun i x h => by
        have := hs ((probes o a N).length + i) x (by
          rw [List.getElem?_append_right (by omega), Nat.add_sub_cancel_left]; exact h)
        rw [tickA, Nat.add_assoc]; exact this
      rw [hA] at ha
      simp only [cnExec, newExec, ha, hA, hb]
    | early =>
      have hp : probes o (.seq a b) N = probes o a N := by simp only [probes, hA]
      rw [hp] at hs
      have ha := iha N hs
      rw [hA] at ha
      simp only [cnExec, newExec, ha, hA]
    | unwind =>
      have hp : probes o (.seq a b) N = probes o a N := by simp only [probes, hA]
      rw [hp] at hs
      have ha := iha N hs
      rw [hA] at ha
      simp only [cnExec, newExec, ha, hA]
  | guarded site span body ih =>
    by_cases hex : exceeded (o N.tick) (N.depth + 1) = true
    · have h0 : s N.tick = true := by
        have := hs 0 true (by simp [probes, hex])
        simpa using this
      simp [cnExec, newExec, hex, h0]
    · have h0 : s N.tick = false := by
        have := hs 0 false (by simp [probes, hex])
        simpa using this
      have hb : cnExec s body N.enterFrame = newExec o body N.enterFrame := ih N.enterFrame
        fun i x h => by
          have := hs (i + 1) x (by
            simp only [probes, hex, if_false, Bool.false_eq_true, List.getElem?_cons_succ]
            exact h)
          simpa [Explicit.enterFrame, Nat.add_assoc, Nat.add_comm 1 i] using this
      simp only [cnExec, newExec, h0, hex, if_false, Bool.false_eq_true, hb]
  | early => rfl
  | unit body ih =>
    have hb : cnExec s body N.enterScope = newExec o body N.enterScope := ih N.enterScope
      fun i x h => by
        have := hs i x (by simpa [probes] using h)
        simpa [Explicit.enterScope] using this
    simp only [cnExec, newExec, hb]
  | panic => rfl

/-- **O12(a).** Every run over a native stack oracle is a run over a Lean-model stack schedule:
the schedule that lists the run's native guard answers at their clock positions. -/
theorem oracle_run_schedule (o : Oracle) (w : Walk) (N : Explicit) :
    ∃ s : Nat → Bool, cnExec s w N = newExec o w N :=
  ⟨fun n => ((probes o w N)[n - N.tick]?).getD false,
    cnExec_of_probes o _ w N fun i x h => by simp [h]⟩

open Aeneas.Std in
/-- The explicit state that a Lean-model `NativeServices` value carries together with the
sink's bail record: the guarded-frame depth and the stack-oracle clock. -/
def ofServices (c : chelis_types.native.NativeServices) (scopeDepth : Nat)
    (first : Option Bail) : Explicit :=
  { depth := c.depth.val, scopeDepth := scopeDepth, first := first, tick := c.stackTaken }

open Aeneas.Std in
private theorem usize_max_eq : Usize.max = 2 ^ System.Platform.numBits - 1 := by
  simp [Usize.max_def, Usize.numBits_def]

open Aeneas.Std in
/-- **O12(b).** Below `usize::MAX` the Lean model of `enter_frame` adds one to the depth. -/
theorem cn_enter_frame (c : chelis_types.native.NativeServices) (h : c.depth.val < Usize.max) :
    ∃ d : Usize, d.val = c.depth.val + 1 ∧
      chelis_types.native.NativeServices.enter_frame c = .ok { c with depth := d } := by
  have hadd := UScalar.add_equiv c.depth 1#usize
  have hm := usize_max_eq
  unfold chelis_types.native.NativeServices.enter_frame
  rcases hr : c.depth + 1#usize with d | e | _
  · rw [hr] at hadd
    exact ⟨d, by simpa using hadd.2.1, by simp⟩
  · rw [hr] at hadd
    exfalso
    apply hadd
    simp only [UScalar.inBounds, UScalarTy.Usize_numBits_eq]
    simp
    omega
  · rw [hr] at hadd
    exact hadd.elim

open Aeneas.Std in
/-- **O12(b), RA-h.** At `usize::MAX` the Lean model of `enter_frame` fails, as Rust's checked
`+=` does in debug builds. Release builds wrap instead (trust entry RA-h). -/
theorem cn_enter_frame_overflow (c : chelis_types.native.NativeServices)
    (h : c.depth.val = Usize.max) :
    chelis_types.native.NativeServices.enter_frame c = .fail .integerOverflow := by
  unfold chelis_types.native.NativeServices.enter_frame
  have hm := usize_max_eq
  have hpos : 0 < 2 ^ System.Platform.numBits := Nat.two_pow_pos _
  have heq : c.depth + 1#usize = UScalar.tryMk .Usize (c.depth.val + 1) := rfl
  have ho := UScalar.tryMkOpt_eq .Usize (c.depth.val + 1)
  rcases hopt : UScalar.tryMkOpt .Usize (c.depth.val + 1) with _ | d
  · rw [heq]
    simp only [UScalar.tryMk, Result.ofOption, hopt]
    rfl
  · rw [hopt] at ho
    exfalso
    have := ho.2
    simp [UScalar.inBounds] at this
    omega

open Aeneas.Std in
/-- **O12(b).** `enter_frame` then `stack_exhausted` (the start of `frame_exhausted`,
`infer/stack.rs:185-197`) below `usize::MAX`: the answer is the schedule element at the clock,
the services move as `Explicit.enterFrame`, and the oracles are unchanged. -/
theorem cn_enter_probe (c : chelis_types.native.NativeServices) (scopeDepth : Nat)
    (first : Option Bail) (h : c.depth.val < Usize.max) :
    ∃ c', (do
        let c ← chelis_types.native.NativeServices.enter_frame c
        chelis_types.native.NativeServices.stack_exhausted c) = .ok (c.stack c.stackTaken, c') ∧
      ofServices c' scopeDepth first = (ofServices c scopeDepth first).enterFrame ∧
      c'.stack = c.stack ∧ c'.cancel = c.cancel ∧ c'.cancelTaken = c.cancelTaken := by
  obtain ⟨d, hd, he⟩ := cn_enter_frame c h
  refine ⟨{ c with depth := d, stackTaken := c.stackTaken + 1 }, ?_, ?_, rfl, rfl, rfl⟩
  · rw [he]; rfl
  · simp [ofServices, Explicit.enterFrame, hd]

open Aeneas.Std in
/-- **O12(b).** `exit_frame` moves the services as `Explicit.exitFrame` (saturating). -/
theorem cn_exit_frame (c : chelis_types.native.NativeServices) (scopeDepth : Nat)
    (first : Option Bail) :
    ∃ c', chelis_types.native.NativeServices.exit_frame c = .ok c' ∧
      ofServices c' scopeDepth first = (ofServices c scopeDepth first).exitFrame ∧
      c'.stack = c.stack ∧ c'.stackTaken = c.stackTaken := by
  refine ⟨_, rfl, ?_, rfl, rfl⟩
  have hlt := c.depth.hBounds
  have h1 : (1#usize : Usize).val = 1 := rfl
  have hval : (UScalar.saturating_sub c.depth 1#usize).val = c.depth.val - 1 := by
    simp only [UScalar.saturating_sub, Nat.zero_max, h1]
    show (BitVec.ofNat _ _).toNat = _
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt]
    omega
  simp [ofServices, Explicit.exitFrame, hval]

open Aeneas.Std in
/-- **O12(b).** `reset_frames` sets the depth to 0, as the outermost `Explicit.enterScope` does. -/
theorem cn_reset_frames (c : chelis_types.native.NativeServices) (scopeDepth : Nat)
    (first : Option Bail) :
    ∃ c', chelis_types.native.NativeServices.reset_frames c = .ok c' ∧
      ofServices c' scopeDepth first = { ofServices c scopeDepth first with depth := 0 } ∧
      c'.stack = c.stack ∧ c'.stackTaken = c.stackTaken :=
  ⟨_, rfl, by simp [ofServices], rfl, rfl⟩

end ChelisBridge.Runtime.Stack
