/-!
# Unwind, panic hook and finalizers: native unwinding versus the explicit `Result` channel

The central relation of the lowering runtime shell. A small language models lowering control
(scopes with finalizers, typed raises, speculative suppression, fatal re-raise, nested
`catch_lowering`, unexpected panics with arbitrary payloads, public panicking entries, native
aborts) over per-thread suppression flags read by the process-global panic hook.

* OLD semantics (`crates/chelis-ir/src/lowering_runtime.rs` on the base branch): a raise is
  `std::panic::panic_any(payload)`; the hook runs before any finalizer, unwinding runs the
  finalizers innermost-first, and `catch_lowering` converts the caught payload.
* NEW semantics (`wip/fcis-ir-lowering`): a raise is `observe_raise` (panic + immediate catch:
  the hook runs with the same thread, flags and payload before any finalizer) and returns
  `Err(LoweringFailure)`, which propagates through `?`; each scope's exit runs on the `Err` path in
  the same innermost-first order; `catch_lowering` maps `Err` with `into_diagnostic`; unexpected
  panics still unwind natively; public panicking entries continue with
  `LoweringFailure::resume_at_boundary` (`resume_unwind`: no second hook).

Trusted-base assumptions (native semantics taken as a model, not proved):

* Rust calls the panic hook exactly once per `panic_any`, before any destructor of the panicking
  frame, on the panicking thread; `resume_unwind` does not call the hook.
* Unwinding runs the destructors of live locals innermost-first; a destructor that panics while
  unwinding is still reported to the hook and then aborts the process ("panic in a destructor
  during cleanup"); a destructor that panics on a normal return starts an ordinary unwind.
* `catch_unwind` stops the unwind and returns the payload; `stacker::grow` is transparent.
* The hook installed by `install_chelis_panic_hook` forwards to the previous hook iff neither of
  the current thread's flags is set (`!(suppress_unrepresentable || suppress_output)`); it is
  installed once, at the first `catch_lowering`.
* Destructors of the typed payloads (`UnrepresentableDag`, `LowerDiagnostic`) and of
  `LoweringFailure` have no observable effect (plain data); only arbitrary payloads may have one.
* A native abort (e.g. stack overflow) runs no destructor.
-/

namespace ChelisBridge.Runtime.Unwind

/-- Native thread identity. -/
abbrev ThreadId := Nat

/-- `LowerDiagnostic { message, span, span_id, fatal }` (span and span id merged into `span`). -/
structure Diag where
  msg : String
  span : Option Nat
  fatal : Bool
  deriving DecidableEq, Repr

/-- `LowerDiagnostic::fatal(self)`: `self.fatal = true`. -/
def Diag.makeFatal (d : Diag) : Diag := { d with fatal := true }

/-- The fixed diagnostic of an unrepresentable DAG. -/
def unrepDiag : Diag :=
  { msg := "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead",
    span := none, fatal := false }

/-- Panic payloads (`Box<dyn Any + Send>`) distinguished by the conversion. -/
inductive Payload where
  /-- `LowerDiagnostic`. -/
  | diag (d : Diag)
  /-- `UnrepresentableDag`. -/
  | unrep
  /-- `String` or `&str`. -/
  | str (s : String)
  /-- Any other payload; `observableDrop` marks a destructor with an observable effect. -/
  | other (id : Nat) (observableDrop : Bool)
  deriving DecidableEq, Repr

/-- `panic_payload_to_lower_diagnostic`:
`UnrepresentableDag → fixed message; LowerDiagnostic → clone; String/&str → message;
other → "internal lowering error"`. -/
def convert : Payload → Diag
  | .unrep => unrepDiag
  | .diag d => d
  | .str s => { msg := s, span := none, fatal := false }
  | .other _ _ => { msg := "internal lowering error", span := none, fatal := false }

/-- `LoweringFailure::{Unrepresentable, Diagnostic}` (new explicit channel). -/
inductive Failure where
  | unrep
  | diag (d : Diag)
  deriving DecidableEq, Repr

/-- The payload a failure unwinds with (`unwind_with` / `resume_at_boundary`). -/
def Failure.payload : Failure → Payload
  | .unrep => .unrep
  | .diag d => .diag d

/-- `LoweringFailure::into_diagnostic`. -/
def Failure.intoDiag : Failure → Diag
  | .unrep => unrepDiag
  | .diag d => d

/-- One thread's `LoweringRuntime` flags. -/
structure Flags where
  /-- `suppress_unrepresentable`. -/
  unrep : Bool
  /-- `suppress_output`. -/
  out : Bool
  deriving DecidableEq, Repr

/-- Global state: every thread's flags (`NATIVE_LOWERING_RUNTIME` TLS), whether the chelis hook
is installed (process-global), and the session-owned invocation state as cells. -/
structure State where
  flags : ThreadId → Flags
  installed : Bool
  cells : Nat → Int

/-- Update the flags of thread `t`. -/
def State.modFlags (s : State) (t : ThreadId) (f : Flags → Flags) : State :=
  { s with flags := fun u => if u = t then f (s.flags t) else s.flags u }

/-- `suppress_output.set(v)` on thread `t`. -/
def State.setOut (s : State) (t : ThreadId) (v : Bool) : State :=
  s.modFlags t (fun fl => { fl with out := v })

/-- `suppress_unrepresentable.set(v)` on thread `t`. -/
def State.setUnrep (s : State) (t : ThreadId) (v : Bool) : State :=
  s.modFlags t (fun fl => { fl with unrep := v })

/-- Write one session cell. -/
def State.setCell (s : State) (k : Nat) (v : Int) : State :=
  { s with cells := fun j => if j = k then v else s.cells j }

/-- Scope guards (finalizer effects). -/
inductive Guard where
  /-- A local with an observable destructor only (e.g. the test's `RecordDrop`). -/
  | noop
  /-- `with_suppress_unrepresentable_panic`: enter sets the flag, `ClearOnDrop` clears it to
  `false` (NOT restore). -/
  | speculate
  /-- Restore-previous (`HostInvocationGuard`, `ActiveTypeSubstGuard`, inlining/preflight): enter
  installs `v` in cell `key`, exit writes back the previous value. -/
  | restore (key : Nat) (v : Int)
  /-- Rollback (`MonoProbeGuard`): snapshot all cells at entry, restore the snapshot at exit. -/
  | rollback
  deriving DecidableEq, Repr

/-- Scope entry effect. -/
def Guard.enter (t : ThreadId) : Guard → State → State
  | .noop, s => s
  | .speculate, s => s.setUnrep t true
  | .restore k v, s => s.setCell k v
  | .rollback, s => s

/-- Scope exit (finalizer) effect; `saved` is the cell snapshot taken at entry. -/
def Guard.exit (t : ThreadId) (saved : Nat → Int) : Guard → State → State
  | .noop, s => s
  | .speculate, s => s.setUnrep t false
  | .restore k _, s => s.setCell k (saved k)
  | .rollback, s => { s with cells := saved }

/-- Lowering-control programs. -/
inductive Prog where
  | pure (v : Nat)
  /-- Observable work. -/
  | work (e : Nat)
  /-- A body effect on session state. -/
  | setCell (k : Nat) (v : Int)
  /-- Sequencing with the value (`?` in the new code). -/
  | bind (p : Prog) (k : Nat → Prog)
  /-- A guard scope: `finPanic = some q` models a finalizer that panics with `q` (e.g. a
  `RefCell` borrow failure in `Drop`). -/
  | scope (g : Guard) (fid : Nat) (finPanic : Option Payload) (body : Prog)
  /-- `raise_lowering_diagnostic(d)`. -/
  | raise (d : Diag)
  /-- `raise_fatal_lowering_diagnostic(d)`. -/
  | raiseFatal (d : Diag)
  /-- A raise of the unrepresentable marker (`raise_unrepresentable`). -/
  | raiseUnrep
  /-- An unexpected `panic_any(p)` with an arbitrary payload. -/
  | panic (p : Payload)
  /-- A non-unwinding native abort. -/
  | abort
  /-- `catch_lowering(body)`, continuing with `onOk v` or `onErr diagnostic`. -/
  | catchL (body : Prog) (onOk : Nat → Prog) (onErr : Diag → Prog)
  /-- A public panicking entry point (its signature cannot return a lowering failure). -/
  | entry (body : Prog)

/-- Observable events. -/
inductive Ev where
  | work (t : ThreadId) (e : Nat)
  /-- A panic-hook invocation: thread, that thread's flags, hook installed?, payload. -/
  | hook (t : ThreadId) (fl : Flags) (installed : Bool) (p : Payload)
  /-- A scope finalizer ran. -/
  | fin (t : ThreadId) (fid : Nat)
  /-- An observable payload destructor ran, with the thread's flags at that time. -/
  | payloadDrop (t : ThreadId) (id : Nat) (fl : Flags)
  deriving DecidableEq, Repr

/-- The hook event of a panic with payload `p` on thread `t` in state `s`. -/
def hookEv (t : ThreadId) (s : State) (p : Payload) : Ev := .hook t (s.flags t) s.installed p

/-- Payload destruction events. -/
def dropEvs (t : ThreadId) (fl : Flags) : Payload → List Ev
  | .other id true => [.payloadDrop t id fl]
  | _ => []

/-- Old outcomes: value, native unwind with payload, abort. -/
inductive NRes where
  | ok (v : Nat)
  | unwind (p : Payload)
  | abort
  deriving DecidableEq, Repr

/-- New outcomes: additionally the explicit `Err(LoweringFailure)`. -/
inductive XRes where
  | ok (v : Nat)
  | err (f : Failure)
  | unwind (p : Payload)
  | abort
  deriving DecidableEq, Repr

/-- The raise payload: `if !diagnostic.fatal && self.unrepresentable_panic_suppressed()
{ panic_any(UnrepresentableDag) } panic_any(diagnostic)`. -/
def raisePayload (fl : Flags) (d : Diag) : Payload :=
  if !d.fatal && fl.unrep then .unrep else .diag d

/-- The new raise failure (same decision). -/
def raiseFailure (fl : Flags) (d : Diag) : Failure :=
  if !d.fatal && fl.unrep then .unrep else .diag d

/-- Old scope exit given the body outcome: finalizer effect and event on return and on unwind;
a panicking finalizer starts an ordinary unwind on return but aborts while unwinding. -/
def oldScopeExit (t : ThreadId) (g : Guard) (saved : Nat → Int) (fid : Nat)
    (fp : Option Payload) (s1 : State) (tr1 : List Ev) : NRes → State × List Ev × NRes
  | .abort => (s1, tr1, .abort)
  | .ok v =>
      match fp with
      | none => (g.exit t saved s1, tr1 ++ [.fin t fid], .ok v)
      | some q => (g.exit t saved s1,
          tr1 ++ [.fin t fid, hookEv t (g.exit t saved s1) q], .unwind q)
  | .unwind p =>
      match fp with
      | none => (g.exit t saved s1, tr1 ++ [.fin t fid], .unwind p)
      | some q => (g.exit t saved s1,
          tr1 ++ [.fin t fid, hookEv t (g.exit t saved s1) q], .abort)

/-- OLD semantics on thread `t` (native unwinding). -/
def old (t : ThreadId) : Prog → State → State × List Ev × NRes
  | .pure v, s => (s, [], .ok v)
  | .work e, s => (s, [.work t e], .ok 0)
  | .setCell k v, s => (s.setCell k v, [], .ok 0)
  | .bind p k, s =>
      match old t p s with
      | (s1, tr1, .ok v) =>
          match old t (k v) s1 with
          | (s2, tr2, r) => (s2, tr1 ++ tr2, r)
      | (s1, tr1, .unwind p) => (s1, tr1, .unwind p)
      | (s1, tr1, .abort) => (s1, tr1, .abort)
  | .scope g fid fp body, s =>
      match old t body (g.enter t s) with
      | (s1, tr1, r) => oldScopeExit t g s.cells fid fp s1 tr1 r
  | .raise d, s => (s, [hookEv t s (raisePayload (s.flags t) d)], .unwind (raisePayload (s.flags t) d))
  | .raiseFatal d, s =>
      -- `self.suppress_output.set(true)`, then raise the fatal diagnostic.
      let s' := s.setOut t true
      (s', [hookEv t s' (raisePayload (s'.flags t) d.makeFatal)],
        .unwind (raisePayload (s'.flags t) d.makeFatal))
  | .raiseUnrep, s => (s, [hookEv t s .unrep], .unwind .unrep)
  | .panic p, s => (s, [hookEv t s p], .unwind p)
  | .abort, s => (s, [], .abort)
  | .catchL body onOk onErr, s =>
      -- `install_chelis_panic_hook(); self.suppress_output.set(true); let _guard = ClearOnDrop(..);
      -- catch_unwind(..).map_err(|payload| panic_payload_to_lower_diagnostic(&*payload))`:
      -- conversion and payload drop happen while the guard is live, then the guard clears.
      match old t body ({ s with installed := true }.setOut t true) with
      | (s1, tr1, .ok v) =>
          match old t (onOk v) (s1.setOut t false) with
          | (s2, tr2, r) => (s2, tr1 ++ tr2, r)
      | (s1, tr1, .unwind p) =>
          match old t (onErr (convert p)) (s1.setOut t false) with
          | (s2, tr2, r) => (s2, tr1 ++ dropEvs t (s1.flags t) p ++ tr2, r)
      | (s1, tr1, .abort) => (s1, tr1, .abort)
  | .entry body, s => old t body s

/-- New scope exit: the explicit exit (or the guard's drop on a normal return) runs on `Ok` and
`Err`; only a genuine native unwind still runs it as a destructor during unwinding. -/
def newScopeExit (t : ThreadId) (g : Guard) (saved : Nat → Int) (fid : Nat)
    (fp : Option Payload) (s1 : State) (tr1 : List Ev) : XRes → State × List Ev × XRes
  | .abort => (s1, tr1, .abort)
  | .ok v =>
      match fp with
      | none => (g.exit t saved s1, tr1 ++ [.fin t fid], .ok v)
      | some q => (g.exit t saved s1,
          tr1 ++ [.fin t fid, hookEv t (g.exit t saved s1) q], .unwind q)
  | .err f =>
      match fp with
      | none => (g.exit t saved s1, tr1 ++ [.fin t fid], .err f)
      | some q => (g.exit t saved s1,
          tr1 ++ [.fin t fid, hookEv t (g.exit t saved s1) q], .unwind q)
  | .unwind p =>
      match fp with
      | none => (g.exit t saved s1, tr1 ++ [.fin t fid], .unwind p)
      | some q => (g.exit t saved s1,
          tr1 ++ [.fin t fid, hookEv t (g.exit t saved s1) q], .abort)

/-- `resume_at_boundary` at a public entry: `Err(f)` continues as a native unwind with the
failure's payload and no hook call. -/
def XRes.resume : XRes → XRes
  | .err f => .unwind f.payload
  | r => r

/-- NEW semantics on thread `t` (explicit `Result` channel). -/
def new (t : ThreadId) : Prog → State → State × List Ev × XRes
  | .pure v, s => (s, [], .ok v)
  | .work e, s => (s, [.work t e], .ok 0)
  | .setCell k v, s => (s.setCell k v, [], .ok 0)
  | .bind p k, s =>
      match new t p s with
      | (s1, tr1, .ok v) =>
          match new t (k v) s1 with
          | (s2, tr2, r) => (s2, tr1 ++ tr2, r)
      | (s1, tr1, .err f) => (s1, tr1, .err f)
      | (s1, tr1, .unwind p) => (s1, tr1, .unwind p)
      | (s1, tr1, .abort) => (s1, tr1, .abort)
  | .scope g fid fp body, s =>
      match new t body (g.enter t s) with
      | (s1, tr1, r) => newScopeExit t g s.cells fid fp s1 tr1 r
  | .raise d, s =>
      -- `observe_raise(failure)`: hook with the same flags and payload, then `Err`.
      (s, [hookEv t s (raiseFailure (s.flags t) d).payload], .err (raiseFailure (s.flags t) d))
  | .raiseFatal d, s =>
      let s' := s.setOut t true
      (s', [hookEv t s' (raiseFailure (s'.flags t) d.makeFatal).payload],
        .err (raiseFailure (s'.flags t) d.makeFatal))
  | .raiseUnrep, s => (s, [hookEv t s .unrep], .err .unrep)
  | .panic p, s => (s, [hookEv t s p], .unwind p)
  | .abort, s => (s, [], .abort)
  | .catchL body onOk onErr, s =>
      -- `Ok(result) => result.map_err(LoweringFailure::into_diagnostic)`,
      -- `Err(payload) => Err(panic_payload_to_lower_diagnostic(&*payload))`, guard live throughout.
      match new t body ({ s with installed := true }.setOut t true) with
      | (s1, tr1, .ok v) =>
          match new t (onOk v) (s1.setOut t false) with
          | (s2, tr2, r) => (s2, tr1 ++ tr2, r)
      | (s1, tr1, .err f) =>
          match new t (onErr f.intoDiag) (s1.setOut t false) with
          | (s2, tr2, r) => (s2, tr1 ++ tr2, r)
      | (s1, tr1, .unwind p) =>
          match new t (onErr (convert p)) (s1.setOut t false) with
          | (s2, tr2, r) => (s2, tr1 ++ dropEvs t (s1.flags t) p ++ tr2, r)
      | (s1, tr1, .abort) => (s1, tr1, .abort)
  | .entry body, s =>
      match new t body s with
      | (s1, tr, r) => (s1, tr, r.resume)

/-! ## The relation -/

/-- How an old observer sees a new outcome: an `Err` still in flight corresponds to the unwind
of its payload. -/
def XRes.embed : XRes → NRes
  | .ok v => .ok v
  | .err f => .unwind f.payload
  | .unwind p => .unwind p
  | .abort => .abort

/-- Embed a whole new observation. -/
def embedObs (x : State × List Ev × XRes) : State × List Ev × NRes := (x.1, x.2.1, x.2.2.embed)

/-- No finalizer panics (explicit hypothesis; true once guards no longer borrow `RefCell`s). -/
def FinPanicFree : Prog → Prop
  | .pure _ => True
  | .work _ => True
  | .setCell _ _ => True
  | .bind p k => FinPanicFree p ∧ ∀ v, FinPanicFree (k v)
  | .scope _ _ fp body => fp = none ∧ FinPanicFree body
  | .raise _ => True
  | .raiseFatal _ => True
  | .raiseUnrep => True
  | .panic _ => True
  | .abort => True
  | .catchL body onOk onErr => FinPanicFree body ∧ (∀ v, FinPanicFree (onOk v)) ∧
      (∀ d, FinPanicFree (onErr d))
  | .entry body => FinPanicFree body

@[simp] theorem convert_payload (f : Failure) : convert f.payload = f.intoDiag := by
  cases f <;> rfl

@[simp] theorem dropEvs_payload (t : ThreadId) (fl : Flags) (f : Failure) :
    dropEvs t fl f.payload = [] := by
  cases f <;> rfl

@[simp] theorem raiseFailure_payload (fl : Flags) (d : Diag) :
    (raiseFailure fl d).payload = raisePayload fl d := by
  unfold raiseFailure raisePayload
  split <;> rfl

theorem scopeExit_embed (t : ThreadId) (g : Guard) (saved : Nat → Int) (fid : Nat)
    (s1 : State) (tr1 : List Ev) (r : XRes) :
    oldScopeExit t g saved fid none s1 tr1 r.embed = embedObs (newScopeExit t g saved fid none s1 tr1 r) := by
  cases r <;> rfl

/-- **D. Unwind equivalence.** For every program whose finalizers do not panic, on every thread
and from every state, the OLD native-unwinding semantics and the NEW explicit-`Result` semantics
produce the same final state (all threads' flags, hook installation, session cells), the same
observable trace (hook events with thread, flags and payload; finalizer events; payload
destructor events, all in the same order) and the same outcome, reading an in-flight `Err` as
the unwind of its payload. -/
theorem unwind_equivalence (t : ThreadId) (p : Prog) (hp : FinPanicFree p) (s : State) :
    old t p s = embedObs (new t p s) := by
  induction p generalizing s with
  | pure v => rfl
  | work e => rfl
  | setCell k v => rfl
  | bind p k ihp ihk =>
    obtain ⟨hp1, hp2⟩ := hp
    simp only [old, new, ihp hp1]
    rcases h : new t p s with ⟨s1, tr1, r⟩
    cases r with
    | ok v =>
      simp only [embedObs, XRes.embed, ihk v (hp2 v)]
    | err f => rfl
    | unwind q => rfl
    | abort => rfl
  | scope g fid fp body ih =>
    obtain ⟨rfl, hb⟩ := hp
    simp only [old, new, ih hb]
    rcases new t body (g.enter t s) with ⟨s1, tr1, r⟩
    exact scopeExit_embed t g s.cells fid s1 tr1 r
  | raise d => simp [old, new, embedObs, XRes.embed]
  | raiseFatal d => simp [old, new, embedObs, XRes.embed]
  | raiseUnrep => rfl
  | panic q => rfl
  | abort => rfl
  | catchL body onOk onErr ih ihOk ihErr =>
    obtain ⟨hb, hok, herr⟩ := hp
    have hB := ih hb ({ s with installed := true }.setOut t true)
    rcases hN : new t body ({ s with installed := true }.setOut t true) with ⟨s1, tr1, r⟩
    rw [hN] at hB
    cases r with
    | ok v =>
      have hK := ihOk v (hok v) (s1.setOut t false)
      simp only [old, new, hB, hN, embedObs, XRes.embed, hK]
    | err f =>
      have hK := ihErr f.intoDiag (herr _) (s1.setOut t false)
      simp only [old, new, hB, hN, embedObs, XRes.embed, hK, convert_payload, dropEvs_payload,
        List.append_nil]
    | unwind q =>
      have hK := ihErr (convert q) (herr _) (s1.setOut t false)
      simp only [old, new, hB, hN, embedObs, XRes.embed, hK]
    | abort =>
      simp only [old, new, hB, hN, embedObs, XRes.embed]
  | entry body ih =>
    have hB := ih hp s
    rcases hN : new t body s with ⟨s1, tr, r⟩
    rw [hN] at hB
    cases r <;> simp only [old, new, hB, hN, embedObs, XRes.embed, XRes.resume]

/-- At a public entry boundary the two semantics coincide exactly (no `Err` escapes it). -/
theorem entry_equivalence (t : ThreadId) (p : Prog) (hp : FinPanicFree p) (s : State) :
    old t p s = embedObs (new t (.entry p) s) ∧ ∀ f, (new t (.entry p) s).2.2 ≠ .err f := by
  refine ⟨?_, ?_⟩
  · rw [unwind_equivalence t p hp s]
    simp only [new]
    rcases new t p s with ⟨s1, tr, r⟩
    cases r <;> rfl
  · intro f
    simp only [new]
    rcases new t p s with ⟨s1, tr, r⟩
    cases r <;> simp [XRes.resume]

/-! ## Outcome classification -/

/-- The five outcome classes observed by an embedder. -/
inductive Outcome where
  | success (v : Nat)
  /-- A semantic rejection (typed diagnostic, fatal or not). -/
  | rejection (d : Diag)
  /-- Unsupported: the unrepresentable marker. -/
  | unsupported
  /-- An unexpected unwind with a foreign payload. -/
  | unexpected (p : Payload)
  | abort
  deriving DecidableEq, Repr

/-- Classify an old (native) outcome. -/
def classify : NRes → Outcome
  | .ok v => .success v
  | .unwind (.diag d) => .rejection d
  | .unwind .unrep => .unsupported
  | .unwind p => .unexpected p
  | .abort => .abort

/-- Classify a new outcome (an `Err` is classified by the payload it would resume with). -/
def classifyNew (r : XRes) : Outcome := classify r.embed

/-- **D. Outcomes stay distinct.** The classification of native outcomes is injective; the new
semantics yields the same class as the old for every panic-free program; and in the new explicit
channel success, semantic rejection, unsupported, unexpected unwind and abort are pairwise
distinct, with distinct failures never merged. -/
theorem outcomes_distinct :
    (∀ r₁ r₂ : NRes, classify r₁ = classify r₂ → r₁ = r₂) ∧
    (∀ t p s, FinPanicFree p → classify (old t p s).2.2 = classifyNew (new t p s).2.2) ∧
    (∀ f g : Failure, classifyNew (.err f) = classifyNew (.err g) → f = g) ∧
    (∀ v d, classifyNew (.ok v) ≠ classifyNew (.err (.diag d))) ∧
    (∀ v, classifyNew (.ok v) ≠ classifyNew (.err .unrep)) ∧
    (∀ d, classifyNew (.err (.diag d)) ≠ classifyNew (.err .unrep)) ∧
    (∀ f, classifyNew (.err f) ≠ classifyNew .abort) ∧
    (∀ f s, classifyNew (.err f) ≠ classifyNew (.unwind (.str s))) ∧
    (∀ f i b, classifyNew (.err f) ≠ classifyNew (.unwind (.other i b))) ∧
    (∀ s, classifyNew .abort ≠ classifyNew (.unwind (.str s))) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro r₁ r₂ h
    cases r₁ with
    | ok v => cases r₂ with
      | ok w => simp_all [classify]
      | unwind q => cases q <;> simp_all [classify]
      | abort => simp_all [classify]
    | unwind p => cases r₂ with
      | ok w => cases p <;> simp_all [classify]
      | unwind q => cases p <;> cases q <;> simp_all [classify]
      | abort => cases p <;> simp_all [classify]
    | abort => cases r₂ with
      | ok w => simp_all [classify]
      | unwind q => cases q <;> simp_all [classify]
      | abort => rfl
  · intro t p s hp
    rw [unwind_equivalence t p hp s]
    rfl
  · intro f g h
    cases f <;> cases g <;> simp_all [classifyNew, classify, XRes.embed, Failure.payload]
  · intro v d; simp [classifyNew, classify, XRes.embed, Failure.payload]
  · intro v; simp [classifyNew, classify, XRes.embed, Failure.payload]
  · intro d; simp [classifyNew, classify, XRes.embed, Failure.payload]
  · intro f; cases f <;> simp [classifyNew, classify, XRes.embed, Failure.payload]
  · intro f s; cases f <;> simp [classifyNew, classify, XRes.embed, Failure.payload]
  · intro f i b; cases f <;> simp [classifyNew, classify, XRes.embed, Failure.payload]
  · intro s; simp [classifyNew, classify, XRes.embed]

/-! ## Why "finalizers do not panic" is necessary -/

/-- The all-clear initial state. -/
def s0 : State := { flags := fun _ => ⟨false, false⟩, installed := false, cells := fun _ => 0 }

/-- **Counterexample.** A raise inside a scope whose finalizer panics: natively the finalizer
runs while unwinding, so the process aborts ("panic in a destructor during cleanup"); in the new
semantics it runs on the `Err` return path and starts an ordinary unwind. -/
theorem panicking_finalizer_counterexample :
    let p := Prog.scope .noop 0 (some (.str "finalizer")) (.raise ⟨"rejected", none, false⟩)
    (old 0 p s0).2.2 = .abort ∧
    (new 0 (.entry p) s0).2.2 = .unwind (.str "finalizer") ∧
    (old 0 p s0).2.2 ≠ (embedObs (new 0 (.entry p) s0)).2.2 := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  cases h

/-! ## Reentrancy and isolation -/

/-- **D. Nested catch clears.** Whatever the body does and whatever the enclosing value of
`suppress_output` was (in particular `true` inside an outer catch), a returning
`catch_lowering` leaves the flag `false` — it clears, it does not restore — in both semantics. -/
theorem nested_catch_clears (t : ThreadId) (body : Prog) (s : State) :
    ((new t (.catchL body .pure (fun _ => .pure 0)) s).2.2 ≠ .abort →
      ((new t (.catchL body .pure (fun _ => .pure 0)) s).1.flags t).out = false) ∧
    ((old t (.catchL body .pure (fun _ => .pure 0)) s).2.2 ≠ .abort →
      ((old t (.catchL body .pure (fun _ => .pure 0)) s).1.flags t).out = false) := by
  refine ⟨?_, ?_⟩
  · rcases hN : new t body ({ s with installed := true }.setOut t true) with ⟨s1, tr1, r⟩
    cases r <;> simp only [new, hN] <;> simp [State.setOut, State.modFlags]
  · rcases hO : old t body ({ s with installed := true }.setOut t true) with ⟨s1, tr1, r⟩
    cases r <;> simp only [old, hO] <;> simp [State.setOut, State.modFlags]

/-- Restore-previous would be wrong: inside an outer catch (`out = true`), the Rust observation
"after-inner-return" requires the flag to be `false` after the inner catch, so the next panic is
visible to the previous hook. -/
theorem clear_not_restore :
    let outer : State := (s0.setOut 0 true)
    ((new 0 (.catchL (.pure 17) .pure (fun _ => .pure 0)) { outer with installed := true }).1.flags 0).out
      = false := rfl

/-- Fatal re-raise re-enables output suppression before its hook runs, in both semantics. -/
theorem fatal_reraise_reenables (t : ThreadId) (d : Diag) (s : State) :
    (new t (.raiseFatal d) s).2.1 =
      [.hook t { s.flags t with out := true } s.installed (.diag d.makeFatal)] ∧
    (old t (.raiseFatal d) s).2.1 =
      [.hook t { s.flags t with out := true } s.installed (.diag d.makeFatal)] ∧
    ((new t (.raiseFatal d) s).1.flags t).out = true := by
  simp [new, old, hookEv, State.setOut, State.modFlags, raiseFailure, raisePayload, Failure.payload,
    Diag.makeFatal]

private theorem modFlags_other (s : State) (t u : ThreadId) (f : Flags → Flags) (h : u ≠ t) :
    (s.modFlags t f).flags u = s.flags u := by
  simp [State.modFlags, h]

private theorem enter_other (t u : ThreadId) (g : Guard) (s : State) (h : u ≠ t) :
    (g.enter t s).flags u = s.flags u := by
  cases g <;> simp [Guard.enter, State.setUnrep, State.setCell, modFlags_other _ _ _ _ h]

private theorem exit_other (t u : ThreadId) (g : Guard) (saved : Nat → Int) (s : State)
    (h : u ≠ t) : (g.exit t saved s).flags u = s.flags u := by
  cases g <;> simp [Guard.exit, State.setUnrep, State.setCell, modFlags_other _ _ _ _ h]

/-- Thread isolation: running on thread `t` never changes another thread's flags (both
semantics, no hypotheses). -/
theorem other_threads_unchanged (t u : ThreadId) (h : u ≠ t) (p : Prog) :
    (∀ s, (new t p s).1.flags u = s.flags u) ∧ (∀ s, (old t p s).1.flags u = s.flags u) := by
  induction p with
  | pure v => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | work e => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | setCell k v => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | bind p k ihp ihk =>
    refine ⟨fun s => ?_, fun s => ?_⟩
    · have h1 := ihp.1 s
      rcases hN : new t p s with ⟨s1, tr1, r⟩
      rw [hN] at h1
      dsimp only at h1
      cases r with
      | ok v =>
        simp only [new, hN]
        exact ((ihk v).1 s1).trans h1
      | err f => simp only [new, hN]; exact h1
      | unwind q => simp only [new, hN]; exact h1
      | abort => simp only [new, hN]; exact h1
    · have h1 := ihp.2 s
      rcases hN : old t p s with ⟨s1, tr1, r⟩
      rw [hN] at h1
      dsimp only at h1
      cases r with
      | ok v =>
        simp only [old, hN]
        exact ((ihk v).2 s1).trans h1
      | unwind q => simp only [old, hN]; exact h1
      | abort => simp only [old, hN]; exact h1
  | scope g fid fp body ih =>
    refine ⟨fun s => ?_, fun s => ?_⟩
    · have h1 := ih.1 (g.enter t s)
      rw [enter_other t u g s h] at h1
      rcases hN : new t body (g.enter t s) with ⟨s1, tr1, r⟩
      rw [hN] at h1
      dsimp only at h1
      simp only [new, hN]
      cases r <;> cases fp <;> simp [newScopeExit, exit_other t u g _ _ h, h1]
    · have h1 := ih.2 (g.enter t s)
      rw [enter_other t u g s h] at h1
      rcases hN : old t body (g.enter t s) with ⟨s1, tr1, r⟩
      rw [hN] at h1
      dsimp only at h1
      simp only [old, hN]
      cases r <;> cases fp <;> simp [oldScopeExit, exit_other t u g _ _ h, h1]
  | raise d => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | raiseFatal d =>
    exact ⟨fun s => by simp only [new]; exact modFlags_other s t u _ h,
      fun s => by simp only [old]; exact modFlags_other s t u _ h⟩
  | raiseUnrep => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | panic q => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | abort => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | catchL body onOk onErr ih ihOk ihErr =>
    have hin : ∀ s : State, ({ s with installed := true }.setOut t true).flags u = s.flags u :=
      fun s => modFlags_other _ t u _ h
    have hout : ∀ s : State, (s.setOut t false).flags u = s.flags u :=
      fun s => modFlags_other _ t u _ h
    refine ⟨fun s => ?_, fun s => ?_⟩
    · have h1 := ih.1 ({ s with installed := true }.setOut t true)
      rw [hin] at h1
      rcases hN : new t body ({ s with installed := true }.setOut t true) with ⟨s1, tr1, r⟩
      rw [hN] at h1
      dsimp only at h1
      cases r with
      | ok v =>
        simp only [new, hN]
        exact (((ihOk v).1 (s1.setOut t false)).trans (hout s1)).trans h1
      | err f =>
        simp only [new, hN]
        exact (((ihErr f.intoDiag).1 (s1.setOut t false)).trans (hout s1)).trans h1
      | unwind q =>
        simp only [new, hN]
        exact (((ihErr (convert q)).1 (s1.setOut t false)).trans (hout s1)).trans h1
      | abort => simp only [new, hN]; exact h1
    · have h1 := ih.2 ({ s with installed := true }.setOut t true)
      rw [hin] at h1
      rcases hN : old t body ({ s with installed := true }.setOut t true) with ⟨s1, tr1, r⟩
      rw [hN] at h1
      dsimp only at h1
      cases r with
      | ok v =>
        simp only [old, hN]
        exact (((ihOk v).2 (s1.setOut t false)).trans (hout s1)).trans h1
      | unwind q =>
        simp only [old, hN]
        exact (((ihErr (convert q)).2 (s1.setOut t false)).trans (hout s1)).trans h1
      | abort => simp only [old, hN]; exact h1
  | entry body ih =>
    refine ⟨fun s => ?_, fun s => ih.2 s⟩
    have h1 := ih.1 s
    rcases hN : new t body s with ⟨s1, tr1, r⟩
    rw [hN] at h1
    dsimp only at h1
    simp only [new, hN]
    exact h1

/-- Two global states look the same from thread `t`: its flags, hook installation, cells. -/
def Agree (t : ThreadId) (a b : State) : Prop :=
  a.flags t = b.flags t ∧ a.installed = b.installed ∧ a.cells = b.cells

private theorem agree_modFlags {t : ThreadId} {a b : State} (h : Agree t a b) (f : Flags → Flags) :
    Agree t (a.modFlags t f) (b.modFlags t f) := by
  obtain ⟨h1, h2, h3⟩ := h
  exact ⟨by simp [State.modFlags, h1], h2, h3⟩

private theorem agree_setCell {t : ThreadId} {a b : State} (h : Agree t a b) (k : Nat) (v : Int) :
    Agree t (a.setCell k v) (b.setCell k v) := by
  obtain ⟨h1, h2, h3⟩ := h
  exact ⟨h1, h2, by simp [State.setCell, h3]⟩

private theorem agree_enter {t : ThreadId} {a b : State} (h : Agree t a b) (g : Guard) :
    Agree t (g.enter t a) (g.enter t b) := by
  cases g with
  | noop => exact h
  | speculate => simp only [Guard.enter, State.setUnrep]; exact agree_modFlags h _
  | restore k v => exact agree_setCell h k v
  | rollback => exact h

private theorem agree_exit {t : ThreadId} {a b : State} (h : Agree t a b) (g : Guard)
    (saved : Nat → Int) : Agree t (g.exit t saved a) (g.exit t saved b) := by
  cases g with
  | noop => exact h
  | speculate => simp only [Guard.exit, State.setUnrep]; exact agree_modFlags h _
  | restore k v => exact agree_setCell h k _
  | rollback => exact ⟨h.1, h.2.1, rfl⟩

private theorem agree_install {t : ThreadId} {a b : State} (h : Agree t a b) :
    Agree t { a with installed := true } { b with installed := true } :=
  ⟨h.1, rfl, h.2.2⟩

private theorem hookEv_agree {t : ThreadId} {a b : State} (h : Agree t a b) (p : Payload) :
    hookEv t a p = hookEv t b p := by
  simp [hookEv, h.1, h.2.1]

private theorem newScopeExit_agree {t : ThreadId} {a b : State} (h : Agree t a b) (g : Guard)
    (saved : Nat → Int) (fid : Nat) (fp : Option Payload) (tr : List Ev) (r : XRes) :
    (newScopeExit t g saved fid fp a tr r).2 = (newScopeExit t g saved fid fp b tr r).2 ∧
    Agree t (newScopeExit t g saved fid fp a tr r).1 (newScopeExit t g saved fid fp b tr r).1 := by
  have he := agree_exit h g saved
  cases r <;> cases fp <;> simp [newScopeExit, hookEv_agree he, he, h]

/-- The panic hook reads only the CURRENT thread's flags: what thread `t` observes (trace and
outcome) depends only on its own flags, the hook installation and the session cells, never on
another thread's suppression state. -/
theorem hook_reads_current_thread (t : ThreadId) (p : Prog) :
    ∀ a b, Agree t a b → (new t p a).2 = (new t p b).2 ∧ Agree t (new t p a).1 (new t p b).1 := by
  induction p with
  | pure v => intro a b h; exact ⟨by first | trivial | rfl, h⟩
  | work e => intro a b h; exact ⟨by first | trivial | rfl, h⟩
  | setCell k v => intro a b h; exact ⟨by first | trivial | rfl, agree_setCell h k v⟩
  | bind p k ihp ihk =>
    intro a b h
    obtain ⟨e1, g1⟩ := ihp a b h
    rcases hA : new t p a with ⟨a1, tra, ra⟩
    rcases hB : new t p b with ⟨b1, trb, rb⟩
    rw [hA, hB] at e1 g1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨rfl, rfl⟩ := e1
    cases ra with
    | ok v =>
      obtain ⟨e2, g2⟩ := ihk v a1 b1 g1
      simp only [new, hA, hB, e2]
      exact ⟨by first | trivial | rfl, g2⟩
    | err f => simp only [new, hA, hB]; exact ⟨by first | trivial | rfl, g1⟩
    | unwind q => simp only [new, hA, hB]; exact ⟨by first | trivial | rfl, g1⟩
    | abort => simp only [new, hA, hB]; exact ⟨by first | trivial | rfl, g1⟩
  | scope g fid fp body ih =>
    intro a b h
    obtain ⟨e1, g1⟩ := ih (g.enter t a) (g.enter t b) (agree_enter h g)
    rcases hA : new t body (g.enter t a) with ⟨a1, tra, ra⟩
    rcases hB : new t body (g.enter t b) with ⟨b1, trb, rb⟩
    rw [hA, hB] at e1 g1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨rfl, rfl⟩ := e1
    simp only [new, hA, hB, h.2.2]
    exact newScopeExit_agree g1 g b.cells fid fp tra ra
  | raise d =>
    intro a b h
    simp only [new, hookEv_agree h, h.1]
    exact ⟨by first | trivial | rfl, h⟩
  | raiseFatal d =>
    intro a b h
    have h' := agree_modFlags h (fun fl => { fl with out := true })
    simp only [new, State.setOut, hookEv_agree h', h'.1]
    exact ⟨by first | trivial | rfl, h'⟩
  | raiseUnrep => intro a b h; simp only [new, hookEv_agree h]; exact ⟨by first | trivial | rfl, h⟩
  | panic q => intro a b h; simp only [new, hookEv_agree h]; exact ⟨by first | trivial | rfl, h⟩
  | abort => intro a b h; exact ⟨by first | trivial | rfl, h⟩
  | catchL body onOk onErr ih ihOk ihErr =>
    intro a b h
    have h0 := agree_modFlags (agree_install h) (fun fl => { fl with out := true })
    obtain ⟨e1, g1⟩ := ih _ _ h0
    rcases hA : new t body ({ a with installed := true }.setOut t true) with ⟨a1, tra, ra⟩
    rcases hB : new t body ({ b with installed := true }.setOut t true) with ⟨b1, trb, rb⟩
    simp only [State.setOut] at hA hB
    rw [hA, hB] at e1 g1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨rfl, rfl⟩ := e1
    have hc := agree_modFlags g1 (fun fl => { fl with out := false })
    cases ra with
    | ok v =>
      obtain ⟨e2, g2⟩ := ihOk v _ _ hc
      simp only [new, State.setOut, hA, hB] at e2 g2 ⊢
      simp only [e2]
      exact ⟨by first | trivial | rfl, g2⟩
    | err f =>
      obtain ⟨e2, g2⟩ := ihErr f.intoDiag _ _ hc
      simp only [new, State.setOut, hA, hB] at e2 g2 ⊢
      simp only [e2]
      exact ⟨by first | trivial | rfl, g2⟩
    | unwind q =>
      obtain ⟨e2, g2⟩ := ihErr (convert q) _ _ hc
      simp only [new, State.setOut, hA, hB] at e2 g2 ⊢
      have hf : a1.flags t = b1.flags t := g1.1
      simp only [e2, hf]
      exact ⟨by first | trivial | rfl, g2⟩
    | abort => simp only [new, State.setOut, hA, hB]; exact ⟨by first | trivial | rfl, g1⟩
  | entry body ih =>
    intro a b h
    obtain ⟨e1, g1⟩ := ih a b h
    rcases hA : new t body a with ⟨a1, tra, ra⟩
    rcases hB : new t body b with ⟨b1, trb, rb⟩
    rw [hA, hB] at e1 g1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨rfl, rfl⟩ := e1
    simp only [new, hA, hB]
    exact ⟨by first | trivial | rfl, g1⟩

/-! ## Replay of the native observation test

`lowering_runtime::tests::native_hook_and_unwind_observations` records `Hook(label)` when the
previous hook is reached (chelis hook not installed, or both flags clear) and `Drop(name)` for
finalizers and payload destructors. -/

/-- What the test's recorder observes. -/
inductive Observed where
  | hook (p : Payload)
  | drop (fid : Nat)
  | payloadDrop (id : Nat)
  deriving DecidableEq, Repr

/-- Project a trace to the recorder's observations. -/
def observed : List Ev → List Observed
  | [] => []
  | .hook _ fl installed p :: tr =>
      if installed && (fl.unrep || fl.out) then observed tr else .hook p :: observed tr
  | .fin _ fid :: tr => .drop fid :: observed tr
  | .payloadDrop _ id _ :: tr => .payloadDrop id :: observed tr
  | .work _ _ :: tr => observed tr

/-- `catch_lowering(|| { let _drop = RecordDrop("typed"); raise(diagnostic) })`:
`Err(diagnostic)` and events `[Drop("typed")]` (hook suppressed). -/
theorem replay_typed :
    let d : Diag := ⟨"typed-rejection", some 7, false⟩
    let p := Prog.catchL (.scope .noop 1 none (.raise d)) (fun _ => .pure 0) (fun e => .pure e.msg.length)
    observed (new 0 p s0).2.1 = [.drop 1] ∧ observed (old 0 p s0).2.1 = [.drop 1] := by
  exact ⟨rfl, rfl⟩

/-- Outer catch around an inner returning catch, then `panic_any("after-inner-return")`:
events `[Hook("after-inner-return"), Drop("outer-return")]` — visible because the inner catch
CLEARED the outer flag. -/
theorem replay_after_inner_return :
    let p := Prog.catchL
      (.scope .noop 2 none
        (.bind (.catchL (.pure 17) .pure (fun _ => .pure 0)) (fun _ => .panic (.str "after-inner-return"))))
      .pure (fun _ => .pure 0)
    observed (new 0 p s0).2.1 = [.hook (.str "after-inner-return"), .drop 2] ∧
    observed (old 0 p s0).2.1 = [.hook (.str "after-inner-return"), .drop 2] := by
  exact ⟨rfl, rfl⟩

/-- Unwinding inner catch: `[Drop("inner-unwind"), Hook("after-inner-unwind"),
Drop("outer-unwind")]`. -/
theorem replay_after_inner_unwind :
    let p := Prog.catchL
      (.scope .noop 3 none
        (.bind (.catchL (.scope .noop 4 none (.panic (.str "quiet-inner"))) .pure (fun _ => .pure 0))
          (fun _ => .panic (.str "after-inner-unwind"))))
      .pure (fun _ => .pure 0)
    observed (new 0 p s0).2.1 = [.drop 4, .hook (.str "after-inner-unwind"), .drop 3] ∧
    observed (old 0 p s0).2.1 = [.drop 4, .hook (.str "after-inner-unwind"), .drop 3] := by
  exact ⟨rfl, rfl⟩

/-- Fatal re-raise under speculation after an inner catch: `Err(fatal)` with no hook visible. -/
theorem replay_fatal_reraise :
    let fatal : Diag := ⟨"typed-rejection", some 7, true⟩
    let p := Prog.catchL
      (.scope .speculate 5 none
        (.catchL (.raise fatal) (fun _ => .pure 0) (fun caught => .raiseFatal caught)))
      .pure (fun _ => .pure 0)
    observed (new 0 p s0).2.1 = [.drop 5] ∧ observed (old 0 p s0).2.1 = [.drop 5] := by
  exact ⟨rfl, rfl⟩

/-- A payload with an observable destructor is dropped while output suppression is still live
(`[Drop("payload")]`, flags `out = true` at destruction). -/
theorem replay_payload_drop :
    let p := Prog.catchL (.panic (.other 9 true)) .pure (fun _ => .pure 0)
    (new 0 p s0).2.1 = [.hook 0 ⟨false, true⟩ true (.other 9 true), .payloadDrop 0 9 ⟨false, true⟩] ∧
    observed (old 0 p s0).2.1 = [.payloadDrop 9] := by
  exact ⟨rfl, rfl⟩

end ChelisBridge.Runtime.Unwind
