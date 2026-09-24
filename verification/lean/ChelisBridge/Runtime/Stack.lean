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

New (`wip/fcis-types`: `native.rs`, `infer/stack.rs`, `session.rs`): the same fields owned by
`NativeServices { depth }` and `DiagnosticSink { stack_bails: StackBailRecord { first,
scope_depth } }`; `frame_exhausted` = `enter_frame(); if stack_exhausted() { record(..);
exit_frame(); return true }`; the walker calls `exit_frame()` on every return path; check
entries call `enter_stack_scope` / `drain_stack_bail` / `exit_stack_scope` explicitly.

Trusted-base assumption: `stacker::remaining_stack()` is an arbitrary oracle (`Oracle`), one
answer per guard check; nothing about native stack consumption is assumed.
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
/-- `enter_scope()`. -/
def Explicit.enterScope (N : Explicit) : Explicit :=
  { N with first := if N.scopeDepth = 0 then none else N.first, scopeDepth := N.scopeDepth + 1 }
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
@[simp] theorem proj_enterScope (x : Tls) : proj x.enterScope = (proj x).enterScope := rfl
@[simp] theorem proj_drainExit (x : Tls) : proj x.drainExit = (proj x).drainExit := rfl

/-! ## B. Equivalence for every oracle -/

/-- **B.** For every oracle (native stack behaviour), every walk and every thread state: the old
TLS machinery and the explicit state produce the same observable events — in particular the
same reported FIRST bail `(site, span)` at every drain — and the same exit; on every returning
exit the TLS of the thread equals the explicit state (record, reset/drain, depth and scope
bookkeeping); and other threads' TLS are untouched. -/
theorem first_bail_equivalence (o : Oracle) (t : ThreadId) (w : Walk) (T : Threads) :
    (oldExec o t w T).2 = (newExec o w (proj (T t))).2 ∧
    ((oldExec o t w T).2.2 ≠ .unwind →
      proj ((oldExec o t w T).1 t) = (newExec o w (proj (T t))).1) ∧
    (∀ u, u ≠ t → (oldExec o t w T).1 u = T u) := by
  induction w generalizing T with
  | work e => simp [oldExec, newExec]
  | seq a b iha ihb =>
    obtain ⟨a1, a2, a3⟩ := iha T
    rcases hA : oldExec o t a T with ⟨T1, tr1, e1⟩
    rcases hN : newExec o a (proj (T t)) with ⟨N1, ntr1, ne1⟩
    rw [hA, hN] at a1 a2
    rw [hA] at a3
    simp only [Prod.mk.injEq] at a1 a2 a3
    obtain ⟨rfl, rfl⟩ := a1
    cases e1 with
    | normal =>
      have hrel : proj (T1 t) = N1 := a2 (by simp)
      obtain ⟨b1, b2, b3⟩ := ihb T1
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
    by_cases hex : exceeded (o (T t).tick) ((T t).recDepth + 1) = true
    · have hex' : exceeded (o (proj (T t)).tick) ((proj (T t)).depth + 1) = true := hex
      simp only [oldExec, newExec, hex, hex', if_true]
      refine ⟨by first | trivial | rfl, fun _ => by simp, fun u hu => by simp [hu]⟩
    · have hex' : ¬ exceeded (o (proj (T t)).tick) ((proj (T t)).depth + 1) = true := hex
      obtain ⟨b1, b2, b3⟩ := ih (modifyAt T t Tls.enterFrame)
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
    obtain ⟨b1, b2, b3⟩ := ih (modifyAt T t Tls.enterScope)
    simp only [modifyAt_same, proj_enterScope] at b1 b2
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

/-- The explicit bookkeeping is balanced on every returning exit. -/
theorem new_balanced (o : Oracle) (w : Walk) (N : Explicit)
    (h : (newExec o w N).2.2 ≠ .unwind) :
    (newExec o w N).1.depth = N.depth ∧ (newExec o w N).1.scopeDepth = N.scopeDepth := by
  let x : Tls := { recDepth := N.depth, scope := N.scopeDepth, exhausted := N.first, tick := N.tick }
  let T : Threads := fun _ => x
  have hN : proj (T 0) = N := rfl
  obtain ⟨e1, e2, _⟩ := first_bail_equivalence o 0 w T
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
  obtain ⟨e1, _, _⟩ := first_bail_equivalence o t (.unit body) T
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

/-- Counterexample for "the explicit state is abandoned after an unwind": the explicit depth
is not decremented by unwinding (no `Drop`), so reusing the state would diverge from the TLS
(whose guard drop runs during unwinding). -/
theorem reuse_after_unwind_counterexample :
    let T : Threads := fun _ => { recDepth := 0, scope := 0, exhausted := none, tick := 0 }
    let w := Walk.guarded 7 none .panic
    ((oldExec (fun _ => none) 0 w T).1 0).recDepth = 0 ∧
    (newExec (fun _ => none) w (fresh 0)).1.depth = 1 := by
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

end ChelisBridge.Runtime.Stack
