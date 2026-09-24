/-!
# Global identity supplies

Model proofs for the two process-global identity supplies of lowering.

Host inference variables (`crates/chelis-ir/src/host.rs`):

```rust
static NEXT_HOST_INFERENCE_VAR: AtomicU32 = AtomicU32::new(0);
NEXT_HOST_INFERENCE_VAR.fetch_add(1, Ordering::Relaxed)   // wrapping
```

Capacity scopes (`crates/chelis-ir/src/capacity_key.rs`):

```rust
static NEXT_CAPACITY_SCOPE: AtomicU64 = AtomicU64::new(1);
NEXT_CAPACITY_SCOPE
    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| current.checked_add(1))
    .expect("capacity-scope identity space exhausted");
```

Trusted-base assumption (`allocU32`, `allocU64`): every read-modify-write on one atomic
location reads the value written by the immediately preceding RMW in that location's
modification order, even with `Relaxed` (C++/Rust atomics: RMW atomicity). An arbitrary
interleaving of threads and invocations is therefore a schedule `List α` of RMW events in
modification order, each labelled by its issuer; a failed `fetch_update` (closure returned
`None`) performs no store.
-/

namespace ChelisBridge.Runtime.Ids

/-! ## C.1 Host inference variables: wrapping `AtomicU32::fetch_add` -/

/-- Run the RMWs of a schedule in modification order from counter `c`: each event receives the
value it read (`fetch_add` returns the previous value) and stores `c + 1` (wrapping). -/
def allocU32 {α : Type} : List α → UInt32 → List (α × UInt32) × UInt32
  | [], c => ([], c)
  | a :: sched, c =>
      match allocU32 sched (c + 1) with
      | (xs, c') => ((a, c) :: xs, c')

/-- Identities handed out by a schedule. -/
def idsU32 {α : Type} (sched : List α) (c : UInt32) : List UInt32 :=
  (allocU32 sched c).1.map (·.2)

private theorem ofNat_succ (n : Nat) :
    UInt32.ofNat (n + 1) = UInt32.ofNat n + 1 := by
  apply UInt32.toNat_inj.mp
  have h1 : (1 : UInt32).toNat = 1 := rfl
  simp only [UInt32.toNat_ofNat', UInt32.toNat_add, h1]
  omega

/-- Closed form: the `i`-th allocation of a schedule started at `c` is `c + i` (mod 2^32),
and the counter advances by the schedule length. -/
theorem allocU32_closed {α : Type} (sched : List α) (c : UInt32) :
    idsU32 sched c = (List.range sched.length).map (fun i => c + UInt32.ofNat i) ∧
    (allocU32 sched c).2 = c + UInt32.ofNat sched.length := by
  induction sched generalizing c with
  | nil => simp [idsU32, allocU32]
  | cons a sched ih =>
    obtain ⟨h1, h2⟩ := ih (c + 1)
    simp only [idsU32] at h1
    simp only [idsU32, allocU32, List.map_cons, h1, h2, List.length_cons, List.range_succ_eq_map,
      List.map_map]
    refine ⟨?_, ?_⟩
    · simp only [List.cons.injEq]
      refine ⟨by simp, ?_⟩
      apply List.map_congr_left
      intro i _
      simp only [Function.comp, ofNat_succ]
      rw [UInt32.add_assoc, UInt32.add_comm 1]
    · rw [ofNat_succ, UInt32.add_assoc, UInt32.add_comm 1]

private theorem ofNat_inj_lt (i j : Nat) (hi : i < 2 ^ 32) (hj : j < 2 ^ 32)
    (h : UInt32.ofNat i = UInt32.ofNat j) : i = j := by
  have := congrArg UInt32.toNat h
  simp only [UInt32.toNat_ofNat'] at this
  omega

/-- **C.1.** For any interleaving of threads and invocations (any schedule of RMW events) and
any starting counter, the allocated identities are pairwise distinct as long as the total
number of allocations is at most `2^32`. (At process start `c = 0`.) -/
theorem distinct_below_bound {α : Type} (sched : List α) (c : UInt32)
    (hbound : sched.length ≤ 2 ^ 32) : (idsU32 sched c).Nodup := by
  rw [(allocU32_closed sched c).1]
  unfold List.Nodup
  rw [List.pairwise_map]
  refine (List.pairwise_lt_range (n := sched.length)).imp_of_mem ?_
  intro i j hi hj hij heq
  simp only [List.mem_range] at hi hj
  have h := (UInt32.add_right_inj c).mp heq
  have := ofNat_inj_lt i j (by omega) (by omega) h
  omega

/-- **C.1 counterexample.** The no-wrap bound cannot be dropped: in any schedule of
`2^32 + 1` allocations the first and the last identity coincide, so equality of host
inference variables identifies distinct allocations. -/
theorem u32_wrap_counterexample {α : Type} (sched : List α) (c : UInt32)
    (hlen : sched.length = 2 ^ 32 + 1) :
    (idsU32 sched c)[0]? = some c ∧ (idsU32 sched c)[2 ^ 32]? = some c ∧
    ¬ (idsU32 sched c).Nodup := by
  have hclosed := (allocU32_closed sched c).1
  have h0 : (idsU32 sched c)[0]? = some c := by
    rw [hclosed, List.getElem?_map, List.getElem?_range (by omega)]
    simp
  have hN : (idsU32 sched c)[2 ^ 32]? = some c := by
    rw [hclosed, List.getElem?_map, List.getElem?_range (by omega)]
    simp only [Option.map_some]
    congr 1
    have : UInt32.ofNat (2 ^ 32) = 0 := by decide
    rw [this, UInt32.add_zero]
  refine ⟨h0, hN, ?_⟩
  intro hnd
  rw [hclosed, hlen, List.range_succ, List.map_append] at hnd
  unfold List.Nodup at hnd
  rw [List.pairwise_append] at hnd
  have h0mem : c + UInt32.ofNat 0 ∈ (List.range (2 ^ 32)).map (fun i => c + UInt32.ofNat i) :=
    List.mem_map.mpr ⟨0, List.mem_range.mpr (by decide), rfl⟩
  apply hnd.2.2 _ h0mem (c + UInt32.ofNat (2 ^ 32)) (by simp)
  have : UInt32.ofNat (2 ^ 32) = UInt32.ofNat 0 := by decide
  rw [this]

/-! ## C.2 Persistence: the supply is threaded, never reset -/

/-- Invocations run one after another (in modification order); the core threads the supply
state from each invocation to the next and never resets it. -/
def runInvocations {α : Type} : List (List α) → UInt32 → List (List UInt32) × UInt32
  | [], c => ([], c)
  | inv :: rest, c =>
      match runInvocations rest (allocU32 inv c).2 with
      | (ids, c') => (idsU32 inv c :: ids, c')

private theorem allocU32_append {α : Type} (a b : List α) (c : UInt32) :
    idsU32 (a ++ b) c = idsU32 a c ++ idsU32 b (allocU32 a c).2 ∧
    (allocU32 (a ++ b) c).2 = (allocU32 b (allocU32 a c).2).2 := by
  induction a generalizing c with
  | nil => simp [idsU32, allocU32]
  | cons x a ih =>
    obtain ⟨h1, h2⟩ := ih (c + 1)
    simp only [idsU32] at h1 ⊢
    simp [allocU32, h1, h2]

/-- Threading the supply through consecutive invocations is the same as allocating along the
concatenated schedule, so the no-wrap distinctness spans all invocations of the process. -/
theorem persistence {α : Type} (invs : List (List α)) (c : UInt32) :
    (runInvocations invs c).1.flatten = idsU32 invs.flatten c ∧
    (runInvocations invs c).2 = (allocU32 invs.flatten c).2 := by
  induction invs generalizing c with
  | nil => simp [runInvocations, idsU32, allocU32]
  | cons inv rest ih =>
    obtain ⟨h1, h2⟩ := ih (allocU32 inv c).2
    obtain ⟨a1, a2⟩ := allocU32_append inv rest.flatten c
    simp [runInvocations, h1, h2, a1, a2]

/-- Distinctness across invocations under the global bound. -/
theorem distinct_across_invocations {α : Type} (invs : List (List α)) (c : UInt32)
    (hbound : invs.flatten.length ≤ 2 ^ 32) : (runInvocations invs c).1.flatten.Nodup := by
  rw [(persistence invs c).1]
  exact distinct_below_bound _ c hbound

/-- Counterexample: a supply reset per invocation (not threaded) would hand the same identity
to two invocations. -/
theorem reset_counterexample :
    idsU32 [()] 0 = idsU32 [()] 0 ∧ ¬ (idsU32 [()] 0 ++ idsU32 [()] 0).Nodup := by
  refine ⟨rfl, ?_⟩
  decide

/-! ## C.3 Capacity scopes: checked `AtomicU64::fetch_update` -/

/-- Largest `u64`. -/
def u64Max : Nat := 2 ^ 64 - 1

/-- Run a schedule of `fresh()` calls: `current.checked_add(1)` succeeds (storing `current + 1`
and returning `current`) iff `current < u64::MAX`; otherwise `fetch_update` returns `Err`,
nothing is stored and `.expect(..)` panics (`none`). Counters are kept as `Nat` below `2^64`. -/
def allocU64 {α : Type} : List α → Nat → List (α × Option Nat) × Nat
  | [], c => ([], c)
  | a :: sched, c =>
      if c < u64Max then
        match allocU64 sched (c + 1) with
        | (xs, c') => ((a, some c) :: xs, c')
      else
        match allocU64 sched c with
        | (xs, c') => ((a, none) :: xs, c')

/-- Identities successfully handed out (panicking calls produce none). -/
def successesU64 {α : Type} (sched : List α) (c : Nat) : List Nat :=
  (allocU64 sched c).1.filterMap (·.2)

private theorem successes_ge {α : Type} (sched : List α) (c : Nat) :
    ∀ x ∈ successesU64 sched c, c ≤ x := by
  induction sched generalizing c with
  | nil => simp [successesU64, allocU64]
  | cons a sched ih =>
    intro x hx
    simp only [successesU64, allocU64] at hx
    split at hx
    · simp only [List.filterMap_cons, List.mem_cons] at hx
      rcases hx with rfl | hx
      · exact Nat.le_refl _
      · have := ih (c + 1) x hx
        omega
    · simp only [List.filterMap_cons] at hx
      exact ih c x hx

/-- **C.3.** Checked capacity scopes are pairwise distinct under every interleaving with no
bound hypothesis (they are strictly increasing); exhaustion panics instead of repeating. -/
theorem capacity_distinct {α : Type} (sched : List α) (c : Nat) :
    (successesU64 sched c).Pairwise (· < ·) := by
  induction sched generalizing c with
  | nil => simp [successesU64, allocU64]
  | cons a sched ih =>
    simp only [successesU64, allocU64]
    split
    · simp only [List.filterMap_cons, List.pairwise_cons]
      exact ⟨fun x hx => Nat.lt_of_lt_of_le (Nat.lt_succ_self c) (successes_ge sched (c + 1) x hx),
        ih (c + 1)⟩
    · simp only [List.filterMap_cons]
      exact ih c

/-- **C.3 bound.** From the initial value `1`, every allocation succeeds while the total count
is below `2^64 - 1` (i.e. at most `2^64 - 2` allocations). -/
theorem capacity_succeeds_below_bound {α : Type} (sched : List α) (c : Nat)
    (hbound : c + sched.length ≤ u64Max) :
    (allocU64 sched c).1.map (·.2) = (List.range sched.length).map (fun i => some (c + i)) := by
  induction sched generalizing c with
  | nil => simp [allocU64]
  | cons a sched ih =>
    simp only [List.length_cons] at hbound
    have hlt : c < u64Max := by omega
    simp only [allocU64, hlt, if_true, List.map_cons, List.length_cons, List.range_succ_eq_map,
      List.map_map, ih (c + 1) (by omega)]
    simp only [List.cons.injEq, Nat.add_zero, true_and]
    apply List.map_congr_left
    intro i _
    simp only [Function.comp]
    congr 1
    omega

/-- Exhaustion is reported, never wrapped: the `(2^64 - 1)`-th allocation from `1` panics. -/
theorem capacity_exhaustion_panics {α : Type} (a : α) :
    (allocU64 [a] u64Max).1 = [(a, none)] := by
  simp [allocU64]

end ChelisBridge.Runtime.Ids
