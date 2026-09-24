/-!
# Cancellation: relaxed-atomic token versus the explicit polling core

Model proofs relating the native cancellation primitive to the explicit-state core.

Native source (`crates/chelis-types/src/cancel.rs`):

```rust
pub struct CancelToken { flag: Arc<AtomicBool> }
pub fn cancel(&self) { self.flag.store(true, Ordering::Relaxed); }
pub fn is_cancelled(&self) -> bool { self.flag.load(Ordering::Relaxed) }
```

and, after the FCIS refactor (`crates/chelis-types/src/native.rs`):

```rust
/// Model: next element of a monotone boolean observation schedule
pub fn cancellation_observed(&mut self) -> bool {
    self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
}
```

Trusted-base assumptions made explicit here (see `modOrder`, `Permitted`):

* the only writer of the flag is `cancel()`, which stores `true`; the initial value is `false`
  (`AtomicBool::new(false)` via `Default`), so the modification order is `[false]` or
  `[false, true]` (repeated `true` stores are indistinguishable for loads);
* relaxed loads by one polling thread obey read-read coherence and read from the modification
  order, and nothing else is guaranteed (no bound on when `true` becomes visible).
-/

namespace ChelisBridge.Runtime.Cancellation

/-! ## A.1 Modification order and permitted observation sequences -/

/-- Modification order of the token's `AtomicBool`. `written` records whether any
`cancel()` store exists (`self.flag.store(true, Ordering::Relaxed)`). -/
def modOrder (written : Bool) : List Bool :=
  if written then [false, true] else [false]

/-- Monotone Boolean sequence: `false* true*` (once `true`, always `true`). -/
def Mono (obs : List Bool) : Prop :=
  obs.Pairwise (fun a b => a = true → b = true)

/-- Observation sequences of one polling thread permitted by the relaxed-atomics model:
each load reads some position of the modification order, and by read-read coherence the
positions read by successive loads of one thread are non-decreasing. No visibility bound. -/
def Permitted (written : Bool) (obs : List Bool) : Prop :=
  ∃ idx : List Nat, idx.Pairwise (· ≤ ·) ∧ (∀ i ∈ idx, i < (modOrder written).length) ∧
    obs = idx.map (fun i => (modOrder written).getD i false)

private theorem getD_written (i : Nat) (h : i < 2) :
    ([false, true] : List Bool).getD i false = decide (i = 1) := by
  match i, h with
  | 0, _ => rfl
  | 1, _ => rfl

private theorem permitted_true_iff (obs : List Bool) : Permitted true obs ↔ Mono obs := by
  constructor
  · rintro ⟨idx, hsorted, hbound, rfl⟩
    simp only [modOrder, if_true, List.length_cons, List.length_nil] at hbound
    unfold Mono
    rw [List.pairwise_map]
    refine hsorted.imp_of_mem ?_
    intro a b ha hb hab
    simp only [modOrder, if_true]
    rw [getD_written a (hbound a ha), getD_written b (hbound b hb)]
    have := hbound a ha
    have := hbound b hb
    simp only [decide_eq_true_eq]
    omega
  · intro hmono
    refine ⟨obs.map (fun b => if b then 1 else 0), ?_, ?_, ?_⟩
    · rw [List.pairwise_map]
      refine hmono.imp ?_
      intro a b hab
      cases a <;> cases b <;> simp_all
    · intro i hi
      simp only [List.mem_map] at hi
      obtain ⟨b, _, rfl⟩ := hi
      cases b <;> simp [modOrder]
    · rw [List.map_map]
      conv => lhs; rw [← List.map_id obs]
      apply List.map_congr_left
      intro b _
      cases b <;> rfl

private theorem permitted_false_iff (obs : List Bool) :
    Permitted false obs ↔ ∀ b ∈ obs, b = false := by
  constructor
  · rintro ⟨idx, _, hbound, rfl⟩ b hb
    simp only [List.mem_map] at hb
    obtain ⟨i, hi, rfl⟩ := hb
    have := hbound i hi
    simp [modOrder] at this
    have : i = 0 := by omega
    subst this
    rfl
  · intro hall
    refine ⟨obs.map (fun _ => 0), ?_, ?_, ?_⟩
    · rw [List.pairwise_map]
      exact List.pairwise_of_forall (fun _ _ => Nat.le_refl 0)
    · intro i hi
      simp only [List.mem_map] at hi
      obtain ⟨_, _, rfl⟩ := hi
      simp [modOrder]
    · rw [List.map_map]
      conv => lhs; rw [← List.map_id obs]
      apply List.map_congr_left
      intro b hb
      simp [hall b hb, modOrder]

/-- **A.1.** With a `cancel()` store, the permitted observation sequences are exactly the
monotone ones; without one, exactly the all-`false` ones. -/
theorem permitted_iff_monotone :
    (∀ obs, Permitted true obs ↔ Mono obs) ∧
    (∀ obs, Permitted false obs ↔ ∀ b ∈ obs, b = false) :=
  ⟨permitted_true_iff, permitted_false_iff⟩

/-- Visibility is not guaranteed within any bound: even after `cancel()`, every finite
number of polls may still observe `false`. -/
theorem no_visibility_bound (n : Nat) : Permitted true (List.replicate n false) := by
  rw [permitted_true_iff]
  unfold Mono
  simp [List.pairwise_replicate]

/-! ## A.2 The explicit polling core -/

/-- An abstract polling program: work steps with cancellation polls between them. -/
inductive Step where
  /-- An observable unit of work (a node visit, a declaration, a phase). -/
  | work (e : Nat)
  /-- A polling site: `NativeServices::cancellation_observed` (old: `is_cancelled` on the
  cached `Arc`, or `cancellation_requested()`); a `true` observation rejects. -/
  | poll
  deriving DecidableEq, Repr

/-- Final outcome: the uncancelled result, or the cancellation rejection
(`cancellation_check_error()` / `EVAL_CANCELLED_MSG`) at poll number `k` (0-based). -/
inductive Outcome where
  | completed
  | cancelled (k : Nat)
  deriving DecidableEq, Repr

/-- The observation stream of one polling thread: element `n` is what the `n`-th load returns. -/
abbrev Stream := Nat → Bool

/-- Drop the first observation. -/
def Stream.tail (s : Stream) : Stream := fun n => s (n + 1)

/-- Renumber a cancellation index after one earlier poll. -/
def Outcome.bump : Outcome → Outcome
  | .completed => .completed
  | .cancelled k => .cancelled (k + 1)

/-- The explicit core: each poll consumes the next stream element (the relaxed load), and a
`true` observation stops with the cancellation rejection. Returns the performed work. -/
def run : List Step → Stream → List Nat × Outcome
  | [], _ => ([], .completed)
  | .work e :: p, s => (e :: (run p s).1, (run p s).2)
  | .poll :: p, s =>
      if s 0 then ([], .cancelled 0) else ((run p s.tail).1, (run p s.tail).2.bump)

/-- Number of polling sites. -/
def pollCount : List Step → Nat
  | [] => 0
  | .work _ :: p => pollCount p
  | .poll :: p => pollCount p + 1

/-- All work of the program. -/
def works : List Step → List Nat
  | [] => []
  | .work e :: p => e :: works p
  | .poll :: p => works p

/-- The work performed before poll number `j`. -/
def worksBefore : List Step → Nat → List Nat
  | [], _ => []
  | .work e :: p, j => e :: worksBefore p j
  | .poll :: _, 0 => []
  | .poll :: p, j + 1 => worksBefore p j

/-- **A.2.** For every observation stream, the outcome is the uncancelled result (all work
performed) iff no poll observes `true`; otherwise it is the cancellation rejection at the
FIRST observing poll `j`, with exactly the work before that poll performed. No hypothesis on
the stream is needed, so in particular this holds for every `Permitted` stream. -/
theorem polling_outcome (p : List Step) (s : Stream) :
    ((∀ j < pollCount p, s j = false) → run p s = (works p, .completed)) ∧
    (∀ j, j < pollCount p → s j = true → (∀ i < j, s i = false) →
      run p s = (worksBefore p j, .cancelled j)) := by
  induction p generalizing s with
  | nil => exact ⟨fun _ => rfl, fun j hj => by simp [pollCount] at hj⟩
  | cons st p ih =>
    cases st with
    | work e =>
      obtain ⟨h1, h2⟩ := ih s
      refine ⟨fun hall => ?_, fun j hj hs hmin => ?_⟩
      · simp only [run, works, h1 hall]
      · simp only [run, worksBefore, h2 j hj hs hmin]
    | poll =>
      obtain ⟨h1, h2⟩ := ih s.tail
      refine ⟨fun hall => ?_, fun j hj hs hmin => ?_⟩
      · have h0 : s 0 = false := hall 0 (by simp [pollCount])
        have hall' : ∀ j < pollCount p, s.tail j = false :=
          fun j hj => hall (j + 1) (by simp [pollCount]; omega)
        simp only [run, h0, works, h1 hall', Outcome.bump]
        rfl
      · cases j with
        | zero => simp only [run, hs, worksBefore, if_true]
        | succ k =>
          have h0 : s 0 = false := hmin 0 (by omega)
          have hk : k < pollCount p := by simp [pollCount] at hj; omega
          simp only [run, h0, worksBefore,
            h2 k hk hs (fun i hi => hmin (i + 1) (by omega)), Outcome.bump]
          rfl

/-! ## A.3 Freezing the flag at entry is not equivalent -/

/-- Forbidden alternative: read the flag once at entry and poll the cached `Bool`
(fcis-design: "NEVER read the flag once at entry and freeze it"). -/
def runFrozen (p : List Step) (s : Stream) : List Nat × Outcome :=
  let b := s 0
  run p (fun _ => b)

/-- The stream `[false, true, true, …]`. -/
def lateCancel : Stream := fun n => decide (1 ≤ n)

/-- **A.3.** On the permitted observation sequence `[false, true]` the per-poll core rejects at
the second poll, while the frozen-at-entry variant completes: they are not equivalent. -/
theorem frozen_entry_not_equivalent :
    Permitted true [lateCancel 0, lateCancel 1] ∧
    run [.poll, .work 1, .poll] lateCancel = ([1], .cancelled 1) ∧
    runFrozen [.poll, .work 1, .poll] lateCancel = ([1], .completed) := by
  refine ⟨?_, rfl, rfl⟩
  rw [permitted_true_iff]
  unfold Mono lateCancel
  decide

/-! ## A.4 Scoped token installation versus the TLS `CancelTokenGuard` -/

/-- Native thread identity. -/
abbrev ThreadId := Nat

/-- Token identity (`Arc<AtomicBool>` pointer identity): all clones share it. -/
abbrev Token := Nat

/-- The old per-thread TLS slot
`static CANCEL_TOKEN: RefCell<Option<CancelToken>>` for every thread. -/
abbrev Slots := ThreadId → Option Token

/-- Point update of one thread's slot. -/
def Slots.update (s : Slots) (t : ThreadId) (v : Option Token) : Slots :=
  fun u => if u = t then v else s u

/-- Programs over token scopes. -/
inductive TProg where
  /-- A polling site; it observes the identity of the token it polls. -/
  | poll
  | seq (a b : TProg)
  /-- Old: `let _g = install_cancel_token(tok)`; new: the inner invocation's
  `NativeServices::new(Some(tok))` passed explicitly to `body`. -/
  | install (tok : Token) (body : TProg)
  /-- An unexpected panic (unwind). -/
  | panic
  /-- `std::panic::catch_unwind(body)`, result discarded. -/
  | catchUnwind (body : TProg)
  deriving Repr

/-- Exit mode of a program: normal return or unwinding. -/
inductive Exit where
  | normal
  | unwind
  deriving DecidableEq, Repr

/-- OLD semantics on thread `t`: TLS slots with the restore-on-drop guard.

* `install_cancel_token`: `let previous = CANCEL_TOKEN.with(|cell| cell.replace(Some(token)));`
* `CancelTokenGuard::drop` (normal return AND unwinding):
  `CANCEL_TOKEN.with(|cell| *cell.borrow_mut() = previous);`
* polls read `CANCEL_TOKEN.with(|cell| cell.borrow() …)` of the current thread. -/
def oldRun (t : ThreadId) : TProg → Slots → Slots × List (Option Token) × Exit
  | .poll, s => (s, [s t], .normal)
  | .seq a b, s =>
      match oldRun t a s with
      | (s1, tr1, .normal) =>
          match oldRun t b s1 with
          | (s2, tr2, e) => (s2, tr1 ++ tr2, e)
      | (s1, tr1, .unwind) => (s1, tr1, .unwind)
  | .install tok body, s =>
      match oldRun t body (s.update t (some tok)) with
      | (s1, tr, e) => (s1.update t (s t), tr, e)
  | .panic, s => (s, [], .unwind)
  | .catchUnwind body, s =>
      match oldRun t body s with
      | (s1, tr, _) => (s1, tr, .normal)

/-- NEW semantics: the token is an explicit argument (the invocation's `NativeServices`);
no ambient state exists, so nothing needs restoring. -/
def newRun (cur : Option Token) : TProg → List (Option Token) × Exit
  | .poll => ([cur], .normal)
  | .seq a b =>
      match newRun cur a with
      | (tr1, .normal) =>
          match newRun cur b with
          | (tr2, e) => (tr1 ++ tr2, e)
      | (tr1, .unwind) => (tr1, .unwind)
  | .install tok body => newRun (some tok) body
  | .panic => ([], .unwind)
  | .catchUnwind body => ((newRun cur body).1, .normal)

private theorem update_self (s : Slots) (t : ThreadId) : s.update t (s t) = s := by
  funext u
  simp only [Slots.update]
  split <;> simp_all

private theorem update_update (s : Slots) (t : ThreadId) (v w : Option Token) :
    (s.update t v).update t w = s.update t w := by
  funext u
  simp only [Slots.update]
  split <;> simp_all

private theorem update_at (s : Slots) (t : ThreadId) (v : Option Token) :
    s.update t v t = v := by
  simp [Slots.update]

/-- **A.4.** The explicit scope equals the TLS guard semantics, for nested scopes of any depth
and both exit modes: every poll sees the same token, and on normal return AND unwind the whole
slot map (in particular the installing thread's previous token) is restored. -/
theorem scoped_token_restores (t : ThreadId) (p : TProg) (s : Slots) :
    oldRun t p s = (s, newRun (s t) p) := by
  induction p generalizing s with
  | poll => rfl
  | seq a b iha ihb =>
    simp only [oldRun, newRun, iha]
    rcases h : newRun (s t) a with ⟨tr1, e1⟩
    cases e1 with
    | normal => simp only [ihb]
    | unwind => rfl
  | install tok body ih =>
    simp only [oldRun, newRun, ih, update_update, update_self, update_at]
  | panic => rfl
  | catchUnwind body ih =>
    simp only [oldRun, newRun, ih]

/-- Thread isolation, independent of restoration: operations on thread `t` never write the
slot of another thread, and what thread `t` observes depends only on its own slot. -/
theorem thread_isolation (t : ThreadId) (p : TProg) :
    (∀ s u, u ≠ t → (oldRun t p s).1 u = s u) ∧
    (∀ s₁ s₂, s₁ t = s₂ t → (oldRun t p s₁).2 = (oldRun t p s₂).2) := by
  refine ⟨?_, ?_⟩
  · intro s u _
    rw [scoped_token_restores]
  · intro s₁ s₂ h
    rw [scoped_token_restores, scoped_token_restores, h]

/-- Two threads' token scopes commute (sequential composition in either order yields the same
slots and the same per-thread observations). -/
theorem threads_commute (t u : ThreadId) (p q : TProg) (s : Slots) :
    (oldRun u q (oldRun t p s).1).1 = (oldRun t p (oldRun u q s).1).1 ∧
    (oldRun u q (oldRun t p s).1).2 = (oldRun u q s).2 ∧
    (oldRun t p (oldRun u q s).1).2 = (oldRun t p s).2 := by
  simp [scoped_token_restores]

end ChelisBridge.Runtime.Cancellation
