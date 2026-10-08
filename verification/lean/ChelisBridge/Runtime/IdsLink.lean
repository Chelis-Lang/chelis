import ChelisBridge.Runtime.Ids
import Aeneas

/-!
# Identity supplies: native schedules versus their Lean models (O15)

Links the schedule models of `Runtime/Ids.lean` to the Lean models of the primitives
(`Aeneas/Std/ChelisNative.lean`):

* `fresh_host_inference_var` (`chelis_ir.lowering_runtime.LoweringRuntime`): oracle `hostVar`,
  clock `hostVarsTaken`. Natively a relaxed `fetch_add` on the process-global
  `NEXT_HOST_INFERENCE_VAR: AtomicU32` that starts at 0.
* `fresh_id` (`chelis_ir.capacity_key.CapacityScopeSupply`): oracle `draw`, clock `taken`.
  Natively a checked relaxed `fetch_update` on the process-global `NEXT_CAPACITY_SCOPE: AtomicU64`
  that starts at 1 (`capacity_key.rs:254`); `none` is the exhaustion panic.
* `fresh_preflight_frame` (`LoweringRuntime`): the thread-local wrapping counter
  `nextPreflightFrame`.

Trusted-base assumption RA-g (`verification/trusted_base.json`, section `runtime_assumptions`):
every read-modify-write on one atomic location reads the value of the preceding one in its
modification order, so an interleaving is a global schedule of issuer labels (`Runtime/Ids.lean`).
A thread's Lean-model oracle is the projection of that schedule onto the thread.

The finite schedules describe only recorded native events. The capacity oracle returns `none`
outside its schedule, but that default is not evidence of native exhaustion.
-/

namespace ChelisBridge.Runtime.IdsLink

open ChelisBridge.Runtime.Ids
open Aeneas.Std

/-! ## Host inference variables -/

/-- The identities that thread `t` receives, in its own order, from a global schedule of
`fetch_add` events (each labelled by its issuing thread) started at the process value 0. -/
def hostVarsOf (t : Nat) (sched : List Nat) : List UInt32 :=
  ((allocU32 sched 0).1.filter (fun e => e.1 = t)).map (·.2)

/-- A native `u32` as the Lean-model scalar. -/
def toU32 (x : UInt32) : U32 := ⟨x.toBitVec⟩

theorem toU32_injective : Function.Injective toU32 := by
  intro x y h
  have : x.toBitVec = y.toBitVec := congrArg UScalar.bv h
  exact UInt32.toBitVec_inj.mp this

/-- `k` successive calls of the Lean model of `fresh_host_inference_var`. -/
def hostDraws : chelis_ir.lowering_runtime.LoweringRuntime → Nat →
    Result (List U32 × chelis_ir.lowering_runtime.LoweringRuntime)
  | s, 0 => .ok ([], s)
  | s, k + 1 => do
      let r ← chelis_ir.lowering_runtime.LoweringRuntime.fresh_host_inference_var s
      let rs ← hostDraws r.2 k
      .ok (r.1 :: rs.1, rs.2)

/-- `k` draws of the Lean model return the next `k` oracle elements and advance the clock. -/
theorem hostDraws_eq (s : chelis_ir.lowering_runtime.LoweringRuntime) (k : Nat) :
    hostDraws s k = .ok ((List.range k).map fun i => s.hostVar (s.hostVarsTaken + i),
      { s with hostVarsTaken := s.hostVarsTaken + k }) := by
  induction k generalizing s with
  | zero => simp [hostDraws]
  | succ k ih =>
    simp only [hostDraws, chelis_ir.lowering_runtime.LoweringRuntime.fresh_host_inference_var,
      bind_tc_ok, ih, List.range_succ_eq_map, List.map_cons, List.map_map, Nat.add_zero]
    simp only [Result.ok.injEq, Prod.mk.injEq, List.cons.injEq, true_and]
    refine ⟨List.map_congr_left fun i _ => ?_, ?_⟩
    · simp only [Function.comp, Nat.add_assoc, Nat.add_comm 1 i]
    · simp only [Nat.add_assoc, Nat.add_comm 1 k]

/-- The identities of one thread are a sublist of all identities of the schedule. -/
theorem hostVarsOf_sublist (t : Nat) (sched : List Nat) :
    (hostVarsOf t sched).Sublist (idsU32 sched 0) :=
  List.filter_sublist.map _

/-- **O15.** A Lean-model runtime whose `hostVar` oracle agrees with the projection of the
global schedule onto thread `t` (from the process start, clock 0) hands out, in its first `k`
draws, exactly the first `k` identities of that projection. -/
theorem host_var_draws (t : Nat) (sched : List Nat)
    (s : chelis_ir.lowering_runtime.LoweringRuntime)
    (hs : ∀ n, n < (hostVarsOf t sched).length →
      s.hostVar n = toU32 ((hostVarsOf t sched).getD n 0))
    (h0 : s.hostVarsTaken = 0) (k : Nat) (hk : k ≤ (hostVarsOf t sched).length) :
    hostDraws s k = .ok (((hostVarsOf t sched).take k).map toU32,
      { s with hostVarsTaken := k }) := by
  rw [hostDraws_eq, h0, Nat.zero_add]
  congr 2
  apply List.ext_getElem
  · simp; omega
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range, Nat.zero_add, List.getElem_take]
    simp only [List.length_map, List.length_range] at h1
    rw [hs i (by omega)]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega : i < (hostVarsOf t sched).length)]

/-- **O15.** Below `2^32` allocations in the whole process, the identities a thread draws
through the Lean model are pairwise distinct (`Ids.distinct_below_bound`). -/
theorem host_var_draws_distinct (t : Nat) (sched : List Nat) (hbound : sched.length ≤ 2 ^ 32)
    (k : Nat) : (((hostVarsOf t sched).take k).map toU32).Nodup := by
  have hall := distinct_below_bound sched 0 hbound
  have hsub := (List.take_sublist k _).trans (hostVarsOf_sublist t sched)
  exact (hall.sublist hsub).map toU32_injective

/-- Matched host draws remain distinct within the global allocation bound. -/
theorem matched_host_var_draws_distinct (t : Nat) (sched : List Nat)
    (s : chelis_ir.lowering_runtime.LoweringRuntime)

    (hs : ∀ n, n < (hostVarsOf t sched).length →
      s.hostVar n = toU32 ((hostVarsOf t sched).getD n 0))

    (h0 : s.hostVarsTaken = 0) (k : Nat) (hk : k ≤ (hostVarsOf t sched).length)
    (hbound : sched.length ≤ 2 ^ 32) :

    ∃ ids s', hostDraws s k = .ok (ids, s') ∧ ids.Nodup :=
  ⟨_, _, host_var_draws t sched s hs h0 k hk,
    host_var_draws_distinct t sched hbound k⟩

/-! ## Capacity scopes -/

/-- The `fresh_id` results of thread `t`, in its own order, in a global schedule of checked
`fetch_update` events started at the process value 1 (`none`: the exhaustion panic). -/
def capacityOf (t : Nat) (sched : List Nat) : List (Option Nat) :=
  ((allocU64 sched 1).1.filter (fun e => e.1 = t)).map (·.2)

/-- A native `u64` below `2^64` as the Lean-model scalar. -/
def toU64 (n : Nat) : U64 := ⟨BitVec.ofNat 64 n⟩

/-- The Lean-model `draw` oracle of thread `t`: the projection of the global schedule. -/
def capacityOracle (t : Nat) (sched : List Nat) : Nat → Option U64 :=
  fun n => ((capacityOf t sched).getD n none).map toU64

private theorem allocU64_from_max {α : Type} (sched : List α) :
    successesU64 sched u64Max = [] := by
  induction sched with
  | nil => simp [successesU64, allocU64]
  | cons a sched ih =>
    simp only [successesU64, allocU64, Nat.lt_irrefl, if_false, List.filterMap_cons]
    exact ih

private theorem successes_lt {α : Type} (sched : List α) (c : Nat) (hc : c ≤ u64Max) :
    ∀ x ∈ successesU64 sched c, x < u64Max := by
  induction sched generalizing c with
  | nil => simp [successesU64, allocU64]
  | cons a sched ih =>
    intro x hx
    simp only [successesU64, allocU64] at hx
    split at hx
    · simp only [List.filterMap_cons, List.mem_cons] at hx
      rcases hx with rfl | hx
      · assumption
      · exact ih (c + 1) (by omega) x hx
    · simp only [List.filterMap_cons] at hx
      exact ih c hc x hx

/-- A `none` event (the exhaustion panic) occurs only after every identity from the start value
`c` to `u64::MAX - 1` was handed out, in increasing order. -/
theorem capacity_none_only_at_exhaustion {α : Type} (sched : List α) (c : Nat)
    (hc : c ≤ u64Max) (hnone : ∃ a, (a, none) ∈ (allocU64 sched c).1) :
    successesU64 sched c = (List.range (u64Max - c)).map (c + ·) := by
  induction sched generalizing c with
  | nil => simp [allocU64] at hnone
  | cons a sched ih =>
    by_cases hlt : c < u64Max
    · have hrest : ∃ b, (b, none) ∈ (allocU64 sched (c + 1)).1 := by
        obtain ⟨b, hb⟩ := hnone
        simp only [allocU64, hlt, if_true, List.mem_cons, Prod.mk.injEq, reduceCtorEq,
          and_false, false_or] at hb
        exact ⟨b, hb⟩
      have := ih (c + 1) (by omega) hrest
      simp only [successesU64, allocU64, hlt, if_true, List.filterMap_cons] at this ⊢
      rw [this]
      have hlen : u64Max - c = (u64Max - (c + 1)) + 1 := by omega
      rw [hlen, List.range_succ_eq_map, List.map_cons, List.map_map]
      simp only [Nat.add_zero, List.cons.injEq, true_and]
      apply List.map_congr_left
      intro i _
      simp only [Function.comp]
      omega
    · have hmax : c = u64Max := by omega
      subst hmax
      simp [allocU64_from_max]

/-- **O15.** The identities one thread obtains are strictly increasing (`Ids.capacity_distinct`
restricted to the thread) and below `u64::MAX`, so they are exact `u64` values. -/
theorem capacity_projection_increasing (t : Nat) (sched : List Nat) :
    ((capacityOf t sched).filterMap id).Pairwise (· < ·) ∧
      ∀ x ∈ (capacityOf t sched).filterMap id, x < u64Max ∧ (toU64 x).val = x := by
  have hsub : ((capacityOf t sched).filterMap id).Sublist (successesU64 sched 1) := by
    simp only [capacityOf, successesU64, List.filterMap_map]
    exact List.filter_sublist.filterMap _
  refine ⟨(capacity_distinct sched 1).sublist hsub, fun x hx => ?_⟩
  have hlt := successes_lt sched 1 (by decide) x (hsub.subset hx)
  refine ⟨hlt, ?_⟩
  simp only [toU64, UScalar.val]
  exact Nat.mod_eq_of_lt (by unfold u64Max at hlt; omega)

/-- **O15.** One step of the Lean model of `fresh_id` on the projected oracle: the next
identity of the thread, or the exhaustion panic exactly where the native counter is
exhausted. -/
theorem fresh_id_projection (t : Nat) (sched : List Nat) (n : Nat) :
    (∀ x, (capacityOf t sched)[n]? = some (some x) →
      chelis_ir.capacity_key.CapacityScopeSupply.fresh_id
        { draw := capacityOracle t sched, taken := n } =
        .ok (toU64 x, { draw := capacityOracle t sched, taken := n + 1 })) ∧
    ((capacityOf t sched)[n]? = some none →
      chelis_ir.capacity_key.CapacityScopeSupply.fresh_id
        { draw := capacityOracle t sched, taken := n } = .fail .panic) := by
  refine ⟨fun x hx => ?_, fun hx => ?_⟩
  · simp [chelis_ir.capacity_key.CapacityScopeSupply.fresh_id, capacityOracle,
      List.getD_eq_getElem?_getD, hx]
  · simp [chelis_ir.capacity_key.CapacityScopeSupply.fresh_id, capacityOracle,
      List.getD_eq_getElem?_getD, hx]

/-- A recorded exhaustion event proves both the model panic and global supply exhaustion. -/
theorem fresh_id_recorded_exhaustion (t : Nat) (sched : List Nat) (n : Nat)
    (hn : (capacityOf t sched)[n]? = some none) :

    chelis_ir.capacity_key.CapacityScopeSupply.fresh_id
      { draw := capacityOracle t sched, taken := n } = .fail .panic ∧

    successesU64 sched 1 = (List.range (u64Max - 1)).map (1 + ·) := by
  refine ⟨(fresh_id_projection t sched n).2 hn, ?_⟩

  have hm : none ∈ capacityOf t sched := List.mem_of_getElem? hn
  simp only [capacityOf, List.mem_map] at hm

  rcases hm with ⟨⟨issuer, value⟩, he, hv⟩
  dsimp only at hv
  subst value

  exact capacity_none_only_at_exhaustion sched 1 (by decide)
    ⟨issuer, (List.mem_filter.mp he).1⟩

/-! ## Preflight frames -/

/-- `k` successive calls of the Lean model of `fresh_preflight_frame`. -/
def frameDraws : chelis_ir.lowering_runtime.LoweringRuntime → Nat →
    Result (List U64 × chelis_ir.lowering_runtime.LoweringRuntime)
  | s, 0 => .ok ([], s)
  | s, k + 1 => do
      let r ← chelis_ir.lowering_runtime.LoweringRuntime.fresh_preflight_frame s
      let rs ← frameDraws r.2 k
      .ok (r.1 :: rs.1, rs.2)

private theorem u64_size : UScalar.size .U64 = 2 ^ 64 := by
  simp [UScalar.size_def]

/-- `k` frame ids of one thread are its counter plus `0, …, k - 1`, modulo `2^64`. -/
theorem frameDraws_vals (s : chelis_ir.lowering_runtime.LoweringRuntime) (k : Nat) :
    ∃ ids s', frameDraws s k = .ok (ids, s') ∧
      ids.map (·.val) = (List.range k).map (fun i => (s.nextPreflightFrame.val + i) % 2 ^ 64) := by
  induction k generalizing s with
  | zero => exact ⟨[], s, rfl, rfl⟩
  | succ k ih =>
    obtain ⟨ids, s', h, hv⟩ :=
      ih { s with nextPreflightFrame := UScalar.wrapping_add s.nextPreflightFrame 1#u64 }
    refine ⟨s.nextPreflightFrame :: ids, s', ?_, ?_⟩
    · simp only [frameDraws, chelis_ir.lowering_runtime.LoweringRuntime.fresh_preflight_frame,
        bind_tc_ok, h]
    · have hlt := s.nextPreflightFrame.hBounds
      simp only [List.map_cons, List.range_succ_eq_map, hv, List.map_map, Nat.add_zero,
        UScalar.wrapping_add_val_eq, u64_size, List.cons.injEq]
      refine ⟨(Nat.mod_eq_of_lt (by simpa using hlt)).symm, List.map_congr_left fun i _ => ?_⟩
      simp only [Function.comp]
      have h1 : (1#u64 : U64).val = 1 := rfl
      rw [h1, Nat.mod_add_mod, Nat.add_assoc, Nat.add_comm 1 i]

/-- **O15.** On one thread, fewer than `2^64` (at most `2^64`) successive preflight frame ids are
pairwise distinct, from any counter value. -/
theorem preflight_frames_distinct (s : chelis_ir.lowering_runtime.LoweringRuntime) (k : Nat)
    (hk : k ≤ 2 ^ 64) :
    ∃ ids s', frameDraws s k = .ok (ids, s') ∧ ids.Nodup := by
  obtain ⟨ids, s', h, hv⟩ := frameDraws_vals s k
  refine ⟨ids, s', h, ?_⟩
  have hvals : ((List.range k).map (fun i => (s.nextPreflightFrame.val + i) % 2 ^ 64)).Nodup := by
    unfold List.Nodup
    rw [List.pairwise_map]
    refine (List.pairwise_lt_range (n := k)).imp_of_mem ?_
    intro i j hi hj hij heq
    simp only [List.mem_range] at hi hj
    have := Nat.ModEq.add_left_cancel' s.nextPreflightFrame.val heq
    have hi' : i % 2 ^ 64 = i := Nat.mod_eq_of_lt (by omega)
    have hj' : j % 2 ^ 64 = j := Nat.mod_eq_of_lt (by omega)
    unfold Nat.ModEq at this
    omega
  rw [← hv] at hvals
  exact List.Nodup.of_map _ hvals

end ChelisBridge.Runtime.IdsLink
