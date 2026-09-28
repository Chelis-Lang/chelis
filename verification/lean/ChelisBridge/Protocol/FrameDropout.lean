import ChelisBridge.Protocol.RandomRun
import ChelisBridge.Protocol.NonF64

/-!
# The extracted `ExecutionFrame::dropout`, composed

The production `chelis_ir::evaluation::ExecutionFrame::dropout` casts the plan's f64 rate to
the storage dtype (`cast_raw`), validates it (`PreparedDropout::new`), enters the node's random
site (`ExecutionFrame::enter`), applies the keyed mask (`PreparedDropout::apply`) and wraps the
output with a clone of the input shape (`TensorValue::from_storage`). An accepted (`Ok`) call of
the extracted function is decomposed into exactly these accepted steps (`dropout_ok_inv`), so:

* its frame transition is the one of `enter` at the same node and seed: every run of accepted
  dropouts is an `AcceptedRun` of `FrameOp.site` steps (`dropout_enter`), and
  `accepted_run_simulates` relates it to the reference Random protocol;
* the rate Rust compares with is the widened cast rate `W = rate.as_f64_lossy()`, and
  acceptance alone implies that `W` is finite with exact value `value W ∈ [0, 1)`
  (`new_ok_inv`);
* per storage dtype, output element `i` is the converted zero exactly when the reference unit
  of the drawn key (`DrawKey.unitValue`, f64) or its f32 rounding (f32, f16, bf16) is below
  `value W`, and the dtype's quotient `x / (1 - w)` otherwise; the output shape is the input
  shape (`dropout_f64`, `dropout_f32`, `dropout_f16`, `dropout_bf16`);
* at a forward site of a well-formed frame, the call draws exactly the reference key of the
  active scope (`dropout_forward_simulates`), composed with the output per dtype in
  `dropout_forward_f64`, `dropout_forward_f32`, `dropout_forward_f16`, `dropout_forward_bf16`;
* a whole run of source controls and dropouts (`DropoutRun`) is an `AcceptedRun`
  (`dropoutRun_accepted`), simulates the reference `runProtocol` and passes `checkProtocol`
  from and back to a root frame; each dropout is masked by its used key (`dropoutRun_simulates`);
* the tensor-level replay theorem (`dropout_replay_tensor_f64`) follows the recorded key with
  no frame change, and `dropoutRun_tensor_f64` composes actual f64 output tensors at each key
  under explicit per-dropout shape, decoding, realization, and representability hypotheses.

Acceptance alone requires no caller-supplied rate or input-value hypothesis: non-finite and
out-of-range rates are rejected by the extracted code. Rational tensor correspondence additionally
requires the exact per-call decoding and representability premises stated in its theorems.
-/

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Protocol ChelisBridge.Random
open_chelis_model

namespace ChelisBridge.Random

abbrev TV := chelis_ir.eval.TensorValue

theorem bind_ok_elim {α β : Type} {x : Result α} {f : α → Result β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp_all

theorem map_ok_elim {α β : Type} {x : Result α} {f : α → β} {b : β}
    (h : (f <$> x) = .ok b) : ∃ a, x = .ok a ∧ f a = b := by
  cases x with
  | ok a => exact ⟨a, rfl, Result.ok.inj h⟩
  | fail e => cases h
  | div => cases h

/-- A `?`-propagated residual is always an `Err`. -/
theorem from_residual_err {T E F : Type} {inst : core.convert.From F E}
    {res : core.result.Result core.convert.Infallible E} {r : core.result.Result T F}
    (h : core.result.FromResidualResult.from_residual T inst res = .ok r) : ∃ e, r = .Err e := by
  cases res with
  | Ok i => nomatch i
  | Err err =>
    unfold core.result.FromResidualResult.from_residual at h
    obtain ⟨a, -, rfl⟩ := map_ok_elim h
    exact ⟨a, rfl⟩

/-- `Result::map_err` of an `Err` is an `Err`. -/
theorem map_err_err {T E F O : Type} {inst : core.ops.function.FnOnce O E F} {e : E} {op : O}
    {r : core.result.Result T F}
    (h : core.result.Result.map_err inst (.Err e) op = .ok r) : ∃ f, r = .Err f := by
  unfold core.result.Result.map_err at h
  obtain ⟨⟨m, _⟩, -, h⟩ := bind_ok_elim h
  exact ⟨m, (Result.ok.inj h).symm⟩

theorem map_err_ok {T E F O : Type} {inst : core.ops.function.FnOnce O E F} {t : T} {op : O} :
    core.result.Result.map_err inst (.Ok t : core.result.Result T E) op = .ok (.Ok t) := by
  simp [core.result.Result.map_err]

theorem prim_disc_inj {p q : Prim}
    (h : chelis_types.types.Prim.read_discriminant p = chelis_types.types.Prim.read_discriminant q) :
    p = q := by
  cases p <;> cases q <;> first | rfl | (simp [chelis_types.types.Prim.read_discriminant] at h)

theorem realize_f64_zero_bits : realize (F64.ofBits 0#u64) = some 0 := by decide +kernel

theorem realize_f64_one_bits : realize (F64.ofBits 4607182418800017408#u64) = some 1 := by
  decide +kernel

/-- **Rate validation.** `PreparedDropout::new` accepts exactly with the parameters `{ rate }`
over the given storage, and only a rate of the storage dtype whose widened value `W` is finite
with `0 ≤ value W < 1`. -/
theorem new_ok_inv {s : Storage} {rsv : SV} {pd : chelis_types.dtype_semantics.PreparedDropout}
    (h : chelis_types.dtype_semantics.PreparedDropout.new s rsv = .ok (.Ok pd)) :
    pd = { input := s, «parameters» := ⟨rsv⟩ } ∧
    (∃ p, chelis_types.dtype_semantics.TensorStorage.prim s = .ok p ∧
      chelis_types.dtype_semantics.ScalarValue.prim rsv = .ok p) ∧
    ∃ W, chelis_types.dtype_semantics.ScalarValue.as_f64_lossy rsv = .ok W ∧
      W.isFinite = true ∧ 0 ≤ value W ∧ value W < 1 := by
  unfold chelis_types.dtype_semantics.PreparedDropout.new at h
  obtain ⟨p, hp, h⟩ := bind_ok_elim h
  obtain ⟨r, hd, h⟩ := bind_ok_elim h
  obtain ⟨cf, hcf, h⟩ := bind_ok_elim h
  rcases r with d | e
  swap
  · simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf
    subst hcf
    obtain ⟨e', he'⟩ := from_residual_err h
    cases he'
  simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf
  subst hcf
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at h
  subst h
  unfold chelis_types.dtype_semantics.DropoutParameters.new at hd
  obtain ⟨b, hf, hd⟩ := bind_ok_elim hd
  cases b
  · simp at hd
  simp only [if_true] at hd
  obtain ⟨q, hq, hd⟩ := bind_ok_elim hd
  obtain ⟨b1, hb1, hd⟩ := bind_ok_elim hd
  simp only [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq, bind_tc_ok] at hb1
  cases b1
  swap
  · simp at hd
  have hqp : q = p := prim_disc_inj (by simpa using hb1)
  subst hqp
  simp only [Bool.false_eq_true, if_false] at hd
  obtain ⟨W, hw, hd⟩ := bind_ok_elim hd
  obtain ⟨b2, hb2, hd⟩ := bind_ok_elim hd
  simp only [F64.is_finite, Result.ok.injEq] at hb2
  subst hb2
  obtain ⟨j, hj, hd⟩ := bind_ok_elim hd
  by_cases hfin : W.isFinite = true
  swap
  · simp only [hfin, Bool.false_eq_true, if_false, Result.ok.injEq] at hj
    subst hj
    obtain ⟨_, -, hd⟩ := bind_ok_elim hd
    simp at hd
  simp only [hfin, if_true, core.ops.range.Range.contains,
    core.cmp.impls.PartialOrdF64.le, core.cmp.impls.PartialOrdF64.lt, bind_tc_ok] at hj
  by_cases hle : F64.le (F64.ofBits 0#u64) W = true
  swap
  · simp only [hle, Bool.false_eq_true, if_false, bind_tc_ok, Result.ok.injEq] at hj
    subst hj
    obtain ⟨_, -, hd⟩ := bind_ok_elim hd
    simp at hd
  by_cases hlt : F64.lt W (F64.ofBits 4607182418800017408#u64) = true
  swap
  · simp only [hle, hlt, Bool.false_eq_true, if_true, if_false, bind_tc_ok,
      Result.ok.injEq] at hj
    subst hj
    obtain ⟨_, -, hd⟩ := bind_ok_elim hd
    simp at hd
  simp only [hle, hlt, if_true, bind_tc_ok, Result.ok.injEq] at hj
  subst hj
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at hd
  subst hd
  obtain ⟨hzf, hz⟩ := realize_eq_some.mp realize_f64_zero_bits
  obtain ⟨hof, ho⟩ := realize_eq_some.mp realize_f64_one_bits
  refine ⟨rfl, ⟨q, hp, hq⟩, W, hw, hfin, ?_, ?_⟩
  · have := value_le_of_le hzf hfin hle
    rwa [hz] at this
  · unfold F64.lt at hlt
    rw [lt_eq_decide hfin hof, ho] at hlt
    exact of_decide_eq_true hlt

/-- `TensorValue::from_storage` returns the given storage with the given shape. -/
theorem from_storage_ok {shape : alloc.vec.Vec Usize} {s : Storage} {tv : TV}
    (h : chelis_ir.eval.TensorValue.from_storage shape s = .ok tv) :
    tv.storage = s ∧ tv.shape = shape := by
  unfold chelis_ir.eval.TensorValue.from_storage at h
  obtain ⟨_, -, h⟩ := bind_ok_elim h
  obtain ⟨_, -, h⟩ := bind_ok_elim h
  obtain ⟨_, hm, h⟩ := bind_ok_elim h
  cases Result.ok.inj h
  exact ⟨rfl, rfl⟩

/-- Cloning a `Vec<usize>` returns the same vector. -/
theorem clone_usize_vec (v : alloc.vec.Vec Usize) :
    alloc.vec.CloneVec.clone core.clone.CloneUsize v = .ok v := by
  unfold alloc.vec.CloneVec.clone
  obtain ⟨s', hs', he⟩ := WP.spec_imp_exists
    (Slice.clone_spec (clone := core.clone.CloneUsize.clone) (s := v) (fun _ _ => rfl))
  rw [hs', ← he]
  rfl

/-- **Decomposition of an accepted `ExecutionFrame::dropout`.** An `Ok` result is produced by
the accepted steps of the Rust body in order: the cast rate `rsv`, the prepared kernel over the
input storage with parameters `{ rate: rsv }`, the entered site (whose frame is returned), the
applied mask with the entered key, and the input shape. -/
theorem dropout_ok_inv {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (p : Prim) (rsv : SV) (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame),
      chelis_types.dtype_semantics.TensorStorage.prim input.storage = .ok p ∧
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) p = .ok (.Ok rsv) ∧
      chelis_types.dtype_semantics.PreparedDropout.new input.storage rsv =
        .ok (.Ok { input := input.storage, «parameters» := ⟨rsv⟩ }) ∧
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') ∧
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := input.storage, «parameters» := ⟨rsv⟩ } k.seed k.ordinal =
        .ok (.Ok out.storage) ∧
      out.shape = input.shape := by
  unfold chelis_ir.evaluation.ExecutionFrame.dropout at h
  simp only [chelis_ir.eval.TensorValue.prim, chelis_ir.eval.TensorValue.impl.storage,
    bind_tc_ok] at h
  obtain ⟨p, hp, h⟩ := bind_ok_elim h
  obtain ⟨b, hf, h⟩ := bind_ok_elim h
  cases b
  · simp only [Bool.false_eq_true, if_false] at h
    obtain ⟨_, -, h⟩ := bind_ok_elim h
    simp at h
  simp only [if_true] at h
  obtain ⟨r, hc, h⟩ := bind_ok_elim h
  obtain ⟨r1, hr1, h⟩ := bind_ok_elim h
  obtain ⟨cf, hcf, h⟩ := bind_ok_elim h
  rcases r with rsv | e
  swap
  · obtain ⟨f, rfl⟩ := map_err_err hr1
    simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf
    subst hcf
    obtain ⟨r4, hr4, h⟩ := bind_ok_elim h
    obtain ⟨_, rfl⟩ := from_residual_err hr4
    simp at h
  rw [map_err_ok] at hr1
  cases Result.ok.inj hr1
  simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf
  subst hcf
  obtain ⟨r2, hn, h⟩ := bind_ok_elim h
  obtain ⟨r3, hr3, h⟩ := bind_ok_elim h
  obtain ⟨cf1, hcf1, h⟩ := bind_ok_elim h
  rcases r2 with pd | e
  swap
  · obtain ⟨f, rfl⟩ := map_err_err hr3
    simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf1
    subst hcf1
    obtain ⟨r4, hr4, h⟩ := bind_ok_elim h
    obtain ⟨_, rfl⟩ := from_residual_err hr4
    simp at h
  rw [map_err_ok] at hr3
  cases Result.ok.inj hr3
  simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf1
  subst hcf1
  obtain ⟨rfl, -, -⟩ := new_ok_inv hn
  obtain ⟨⟨r4, F1, eb⟩, he, h⟩ := bind_ok_elim h
  obtain ⟨cf2, hcf2, h⟩ := bind_ok_elim h
  rcases r4 with k | e
  swap
  · simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf2
    subst hcf2
    obtain ⟨r5, hr5, h⟩ := bind_ok_elim h
    obtain ⟨_, rfl⟩ := from_residual_err hr5
    simp at h
  simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf2
  subst hcf2
  obtain ⟨r5, ha, h⟩ := bind_ok_elim h
  obtain ⟨r6, hr6, h⟩ := bind_ok_elim h
  obtain ⟨cf3, hcf3, h⟩ := bind_ok_elim h
  rcases r5 with s | e
  swap
  · obtain ⟨f, rfl⟩ := map_err_err hr6
    simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf3
    subst hcf3
    obtain ⟨r7, hr7, h⟩ := bind_ok_elim h
    obtain ⟨_, rfl⟩ := from_residual_err hr7
    simp at h
  rw [map_err_ok] at hr6
  cases Result.ok.inj hr6
  simp only [core.result.TryResult.branch, Result.ok.injEq] at hcf3
  subst hcf3
  obtain ⟨v, hv, h⟩ := bind_ok_elim h
  rw [clone_usize_vec] at hv
  cases Result.ok.inj hv
  obtain ⟨tv, ht, h⟩ := bind_ok_elim h
  simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at h
  obtain ⟨rfl, rfl, -⟩ := h
  obtain ⟨hs, hsh⟩ := from_storage_ok ht
  exact ⟨p, rsv, k, eb, hp, hc, hn, he, by rw [hs]; exact ha, hsh⟩

/-! ## The frame transition is the one of `enter` -/

/-- An accepted dropout leaves exactly the frame an accepted `enter` at the same node and seed
leaves: a dropout is a `FrameOp.site` step of an `AcceptedRun`. -/
theorem dropout_enter {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame),
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') := by
  obtain ⟨-, -, k, back', -, -, -, he, -, -⟩ := dropout_ok_inv h
  exact ⟨k, back', he⟩

/-- **Protocol step of a forward dropout.** At a forward site of a well-formed frame (no
counter overflow), an accepted dropout draws the key `(seed, c)` of the active scope's counter
`c`, and the reference protocol takes the same `draw` step between the abstracted frames. -/
theorem dropout_forward_simulates (target : Target) {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    (hsite : siteOf F node = .ok (some (.Forward draw s))) (hwf : StackWF F)
    (hno : ∀ c, counterAt F s = some c → c.val + 1 < 2 ^ 64)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ c, counterAt F s = some c ∧
      (∃ back', chelis_ir.evaluation.ExecutionFrame.enter F node seed =
        .ok (.Ok ⟨seed, c⟩, F', back')) ∧
      runProtocol target [.draw (keyOfDraw seed c)] (absState F) = some (absState F') ∧
      StackWF F' := by
  obtain ⟨k, back', he⟩ := dropout_enter h
  obtain ⟨c, -, -, hc, -, rfl, -⟩ := enter_forward_inv F node seed draw s k F' back' hsite he
  have hrun : AcceptedRun F [.site node seed] [.draw (keyOfDraw seed c)] F' :=
    .forward F F' F' node seed draw s ⟨seed, c⟩ back' [] [] hsite he (hno c hc) (.nil F')
  obtain ⟨h1, h2⟩ := accepted_run_simulates target hrun hwf
  exact ⟨c, hc, ⟨back', he⟩, h1, h2⟩

/-! ## Output of an accepted dropout, per storage dtype -/

theorem cast_raw_float_f64 (rate : F64) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .F64 =
      .ok (.Ok ⟨.F64 rate⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, chelis_types.dtype_semantics.finalize_scalar,
    lift, chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
    chelis_types.types.Prim.read_discriminant]

theorem cast_raw_float_f32 (rate : F64) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .F32 =
      .ok (.Ok ⟨.F32 (F64.toF32 rate)⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, chelis_types.dtype_semantics.finalize_scalar,
    lift, chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
    chelis_types.types.Prim.read_discriminant]

theorem sv_prim_f16 {rsv : SV} (h : chelis_types.dtype_semantics.ScalarValue.prim rsv = .ok .F16) :
    ∃ w, rsv = ⟨.F16 w⟩ := by
  rcases rsv with ⟨b⟩
  cases b <;> simp [chelis_types.dtype_semantics.ScalarValue.prim] at h ⊢

theorem sv_prim_bf16 {rsv : SV}
    (h : chelis_types.dtype_semantics.ScalarValue.prim rsv = .ok .Bf16) :
    ∃ w, rsv = ⟨.Bf16 w⟩ := by
  rcases rsv with ⟨b⟩
  cases b <;> simp [chelis_types.dtype_semantics.ScalarValue.prim] at h ⊢

/-- The f32-rounded unit comparison of the non-f64 paths, for a finite wide rate `W`. -/
theorem nonf64_step (seed ordinal : U64) (W : F64) (hW : W.isFinite = true) (i : ℕ) :
    F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) W =
      decide (value (F64.toF32 (unitAt seed ordinal i)) < value W) := by
  have hu := realize_unit (keyOfDraw seed ordinal) i
  have hrange := DrawKey.unitValue_range (keyOfDraw seed ordinal) i
  exact nonf64_lt_eq (u := unitAt seed ordinal i) hu
    (by rw [abs_of_nonneg hrange.1]; exact hrange.2) (realize_eq_some.mpr ⟨hW, rfl⟩)

/-- **Accepted dropout, f64 storage.** The rate is used as is; acceptance implies it is finite
with `0 ≤ value rate < 1`. Output element `i` is the converted zero exactly when the reference
unit `DrawKey.unitValue` of the entered key is below `value rate`, and the binary64 quotient
`x / (1 - rate)` otherwise; the output shape is the input shape. -/
theorem dropout_f64 {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame} {xs : alloc.vec.Vec F64}
    (hs : input.storage = ⟨.F64 xs⟩)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame) (ys : alloc.vec.Vec F64),
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') ∧
      rate.isFinite = true ∧ 0 ≤ value rate ∧ value rate < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.F64 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if (keyOfDraw k.seed k.ordinal).unitValue i < value rate then zero64
        else F64.div x (F64.sub one64 rate)) := by
  obtain ⟨p, rsv, k, back', hp, hc, hn, he, ha, hsh⟩ := dropout_ok_inv h
  rw [hs] at hp hn ha
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hp
  subst hp
  rw [cast_raw_float_f64] at hc
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at hc
  subst hc
  obtain ⟨-, -, W, hW, hfin, h0, h1⟩ := new_ok_inv hn
  simp only [chelis_types.dtype_semantics.ScalarValue.as_f64_lossy, Result.ok.injEq] at hW
  subst hW
  obtain ⟨ys, happ, hys⟩ :=
    mask_agrees_f64 xs rate (value rate) (realize_eq_some.mpr ⟨hfin, rfl⟩) k.seed k.ordinal
  rw [happ] at ha
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at ha
  exact ⟨k, back', ys, he, hfin, h0, h1, hsh, ha.symm, hys⟩

/-- **Accepted dropout, f32 storage.** The rate is `rate as f32` (`cast_raw`), widened back to
`W = f64::from(rate as f32)`; acceptance implies `W` finite with `0 ≤ value W < 1`. Output
element `i` is the converted zero exactly when the f32 rounding of the reference unit is below
`value W` (the corrected relation, `f32_naive_counterexample`), and the binary32 quotient
`x / (1 - w)` otherwise; the output shape is the input shape. -/
theorem dropout_f32 {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame} {xs : alloc.vec.Vec F32}
    (hs : input.storage = ⟨.F32 xs⟩)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame) (ys : alloc.vec.Vec F32),
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') ∧
      (F32.toF64 (F64.toF32 rate)).isFinite = true ∧ 0 ≤ value (F32.toF64 (F64.toF32 rate)) ∧
      value (F32.toF64 (F64.toF32 rate)) < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.F32 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt k.seed k.ordinal i)) < value (F32.toF64 (F64.toF32 rate))
        then zero32 else F32.div x (F32.sub one32 (F64.toF32 rate))) := by
  obtain ⟨p, rsv, k, back', hp, hc, hn, he, ha, hsh⟩ := dropout_ok_inv h
  rw [hs] at hp hn ha
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hp
  subst hp
  rw [cast_raw_float_f32] at hc
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at hc
  subst hc
  obtain ⟨-, -, W, hW, hfin, h0, h1⟩ := new_ok_inv hn
  simp only [chelis_types.dtype_semantics.ScalarValue.as_f64_lossy, Result.ok.injEq] at hW
  subst hW
  obtain ⟨ys, happ, hys⟩ := apply_f32 xs (F64.toF32 rate) k.seed k.ordinal
  rw [happ] at ha
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at ha
  refine ⟨k, back', ys, he, hfin, h0, h1, hsh, ha.symm, ?_⟩
  rw [hys]
  congr 1
  funext i x
  rw [nonf64_step k.seed k.ordinal _ hfin i]
  simp only [decide_eq_true_eq]

/-- **Accepted dropout, f16 storage.** The rate is the f16 `w` returned by the extracted
`cast_raw`, widened to `W = f64::from(w)`; acceptance implies `W` finite with
`0 ≤ value W < 1`. Output element `i` is `0` exactly when the f32 rounding of the reference unit
is below `value W`, and the f16 quotient `x / (1 - w)` (through f32) otherwise. -/
theorem dropout_f16 {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame} {xs : alloc.vec.Vec F16}
    (hs : input.storage = ⟨.F16 xs⟩)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame) (w : F16)
      (ys : alloc.vec.Vec F16),
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') ∧
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .F16 =
        .ok (.Ok ⟨.F16 w⟩) ∧
      (f16To64 w).isFinite = true ∧ 0 ≤ value (f16To64 w) ∧ value (f16To64 w) < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.F16 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt k.seed k.ordinal i)) < value (f16To64 w) then F16.ZERO
        else f16Div x (f16Sub F16.ONE w)) := by
  obtain ⟨p, rsv, k, back', hp, hc, hn, he, ha, hsh⟩ := dropout_ok_inv h
  rw [hs] at hp hn ha
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hp
  subst hp
  obtain ⟨-, ⟨q, hq, hrp⟩, W, hW, hfin, h0, h1⟩ := new_ok_inv hn
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hq
  subst hq
  obtain ⟨w, rfl⟩ := sv_prim_f16 hrp
  rw [as_f64_lossy_f16, Result.ok.injEq] at hW
  subst hW
  obtain ⟨ys, happ, hys⟩ := apply_f16 xs w k.seed k.ordinal
  rw [happ] at ha
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at ha
  refine ⟨k, back', w, ys, he, hc, hfin, h0, h1, hsh, ha.symm, ?_⟩
  rw [hys]
  congr 1
  funext i x
  rw [nonf64_step k.seed k.ordinal _ hfin i]
  simp only [decide_eq_true_eq]

/-- **Accepted dropout, bf16 storage.** As `dropout_f16`, with the bf16 rate and quotient. -/
theorem dropout_bf16 {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame} {xs : alloc.vec.Vec BF16}
    (hs : input.storage = ⟨.Bf16 xs⟩)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame) (w : BF16)
      (ys : alloc.vec.Vec BF16),
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') ∧
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .Bf16 =
        .ok (.Ok ⟨.Bf16 w⟩) ∧
      (bf16To64 w).isFinite = true ∧ 0 ≤ value (bf16To64 w) ∧ value (bf16To64 w) < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.Bf16 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt k.seed k.ordinal i)) < value (bf16To64 w) then BF16.ZERO
        else bf16Div x (bf16Sub BF16.ONE w)) := by
  obtain ⟨p, rsv, k, back', hp, hc, hn, he, ha, hsh⟩ := dropout_ok_inv h
  rw [hs] at hp hn ha
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hp
  subst hp
  obtain ⟨-, ⟨q, hq, hrp⟩, W, hW, hfin, h0, h1⟩ := new_ok_inv hn
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hq
  subst hq
  obtain ⟨w, rfl⟩ := sv_prim_bf16 hrp
  rw [as_f64_lossy_bf16, Result.ok.injEq] at hW
  subst hW
  obtain ⟨ys, happ, hys⟩ := apply_bf16 xs w k.seed k.ordinal
  rw [happ] at ha
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at ha
  refine ⟨k, back', w, ys, he, hc, hfin, h0, h1, hsh, ha.symm, ?_⟩
  rw [hys]
  congr 1
  funext i x
  rw [nonf64_step k.seed k.ordinal _ hfin i]
  simp only [decide_eq_true_eq]

/-- **Composed f64 dropout at a forward site.** For f64 storage at a forward site of a
well-formed frame without counter overflow, an accepted `ExecutionFrame::dropout` draws the
reference key `(seed, c)` of the active scope (the reference protocol takes the same `draw`
step between the abstracted frames), and its output is the input masked by that key: element
`i` is `0` exactly when `DrawKey.unitValue (seed, c) i < value rate`, and `x / (1 - rate)`
otherwise, with the input shape. -/
theorem dropout_forward_f64 (target : Target) {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    {xs : alloc.vec.Vec F64} (hs : input.storage = ⟨.F64 xs⟩)
    (hsite : siteOf F node = .ok (some (.Forward draw s))) (hwf : StackWF F)
    (hno : ∀ c, counterAt F s = some c → c.val + 1 < 2 ^ 64)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (c : U64) (ys : alloc.vec.Vec F64), counterAt F s = some c ∧
      runProtocol target [.draw (keyOfDraw seed c)] (absState F) = some (absState F') ∧
      StackWF F' ∧ rate.isFinite = true ∧ 0 ≤ value rate ∧ value rate < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.F64 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if (keyOfDraw seed c).unitValue i < value rate then zero64
        else F64.div x (F64.sub one64 rate)) := by
  obtain ⟨c, hc, ⟨b1, he1⟩, hrun, hwf'⟩ := dropout_forward_simulates target hsite hwf hno h
  obtain ⟨k, b2, ys, he2, hfin, h0, h1, hsh, hout, hys⟩ := dropout_f64 hs h
  rw [he1] at he2
  simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he2
  obtain ⟨rfl, -⟩ := he2
  exact ⟨c, ys, hc, hrun, hwf', hfin, h0, h1, hsh, hout, hys⟩

/-- **Composed tensor f64 dropout at a forward site.** The successful extracted production call
returns the actual `PreparedDropout::apply` output. The reference shape is *exactly* the
production input's dimension vector, not merely a shape with the same element count. With
an exactly decoded f64 input and representability of the denominator and every input-element
quotient, the actual output decodes to the rational tensor masked by the forward site's
reference draw key. The same call takes that protocol step and preserves frame well-formedness. -/
theorem dropout_forward_tensor_f64 (target : Target) {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    {shape : Shape} {xs : alloc.vec.Vec F64} {x : Tensor ℚ shape}
    {dropoutRate : DropoutRate} (hs : input.storage = ⟨.F64 xs⟩)
    (hinputShape : input.shape.val.map (fun d => d.val) = shape)
    (hx : decodeTensor shape xs.val = some x)
    (hrate : realize rate = some dropoutRate.val)
    (hden : Representable .binary64 (1 - dropoutRate.val))
    (hq : ∀ (j : ℕ) (hj : j < xs.val.length) (a : ℚ), realize xs.val[j] = some a →
      Representable .binary64 (a / (1 - dropoutRate.val)))
    (hsite : siteOf F node = .ok (some (.Forward draw s))) (hwf : StackWF F)
    (hno : ∀ c, counterAt F s = some c → c.val + 1 < 2 ^ 64)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (c : U64) (ys : alloc.vec.Vec F64) (y : Tensor ℚ shape),
      counterAt F s = some c ∧
      runProtocol target [.draw (keyOfDraw seed c)] (absState F) = some (absState F') ∧
      StackWF F' ∧ out.shape = input.shape ∧
      out.shape.val.map (fun d => d.val) = shape ∧ out.storage = ⟨.F64 ys⟩ ∧
      decodeTensor shape ys.val = some y ∧
      ∀ i, y i = x i * dropoutMask dropoutRate (keyOfDraw seed c) i := by
  obtain ⟨c, ys₀, hc, hrun, hwf', _, _, _, hshape, hstorage, _⟩ :=
    dropout_forward_f64 target hs hsite hwf hno h
  obtain ⟨p, rsv, k, back', hp, hcast, hnew, he, happly, hshape'⟩ := dropout_ok_inv h
  rw [hs] at hp hnew happly
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hp
  subst p
  rw [cast_raw_float_f64] at hcast
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at hcast
  subst rsv
  obtain ⟨c', -, -, hc', -, hk', -⟩ :=
    enter_forward_inv F node seed draw s k F' back' hsite he
  have hcEq : c' = c := (Option.some.inj (hc.symm.trans hc')).symm
  have hk : k = ⟨seed, c⟩ := by
    rw [hcEq] at hk'
    exact hk'
  have hkey : keyOfDraw k.seed k.ordinal = keyOfDraw seed c := by
    rw [hk]
  obtain ⟨ys, y, hmask, hdecode, hy⟩ :=
    mask_tensor_f64 xs rate dropoutRate hrate k.seed k.ordinal x hx hden hq
  have houtStorage : out.storage = ⟨.F64 ys⟩ := by
    rw [hmask] at happly
    simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at happly
    exact happly.symm
  refine ⟨c, ys, y, hc, hrun, hwf', hshape, ?_, houtStorage, hdecode, ?_⟩
  · rw [hshape]
    exact hinputShape
  rw [hkey] at hy
  exact hy

/-- The counter of the active scope at a forward site: an accepted dropout there draws the key
`(seed, c)` of that counter (no well-formedness needed for the key itself). -/
theorem dropout_forward_key {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    (hsite : siteOf F node = .ok (some (.Forward draw s)))
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ c, counterAt F s = some c ∧
      ∃ back', chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok ⟨seed, c⟩, F', back') := by
  obtain ⟨k, back', he⟩ := dropout_enter h
  obtain ⟨c, -, -, hc, -, rfl, -⟩ := enter_forward_inv F node seed draw s k F' back' hsite he
  exact ⟨c, hc, back', he⟩

/-- **Composed f32 dropout at a forward site.** As `dropout_forward_f64`, with the f32 rate
`w = rate as f32`, its widening `W = f64::from(w)`, the f32 rounding of the reference unit of the
drawn key `(seed, c)`, and the binary32 quotient. -/
theorem dropout_forward_f32 (target : Target) {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    {xs : alloc.vec.Vec F32} (hs : input.storage = ⟨.F32 xs⟩)
    (hsite : siteOf F node = .ok (some (.Forward draw s))) (hwf : StackWF F)
    (hno : ∀ c, counterAt F s = some c → c.val + 1 < 2 ^ 64)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (c : U64) (ys : alloc.vec.Vec F32), counterAt F s = some c ∧
      runProtocol target [.draw (keyOfDraw seed c)] (absState F) = some (absState F') ∧
      StackWF F' ∧
      (F32.toF64 (F64.toF32 rate)).isFinite = true ∧ 0 ≤ value (F32.toF64 (F64.toF32 rate)) ∧
      value (F32.toF64 (F64.toF32 rate)) < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.F32 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt seed c i)) < value (F32.toF64 (F64.toF32 rate))
        then zero32 else F32.div x (F32.sub one32 (F64.toF32 rate))) := by
  obtain ⟨c, hc, ⟨b1, he1⟩, hrun, hwf'⟩ := dropout_forward_simulates target hsite hwf hno h
  obtain ⟨k, b2, ys, he2, hfin, h0, h1, hsh, hout, hys⟩ := dropout_f32 hs h
  rw [he1] at he2
  simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he2
  obtain ⟨rfl, -⟩ := he2
  exact ⟨c, ys, hc, hrun, hwf', hfin, h0, h1, hsh, hout, hys⟩

/-- **Composed f16 dropout at a forward site.** As `dropout_forward_f64`, with the f16 rate `w`
returned by the extracted `cast_raw`, its widening, the f32 rounding of the reference unit of
the drawn key `(seed, c)`, and the f16 quotient. -/
theorem dropout_forward_f16 (target : Target) {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    {xs : alloc.vec.Vec F16} (hs : input.storage = ⟨.F16 xs⟩)
    (hsite : siteOf F node = .ok (some (.Forward draw s))) (hwf : StackWF F)
    (hno : ∀ c, counterAt F s = some c → c.val + 1 < 2 ^ 64)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (c : U64) (w : F16) (ys : alloc.vec.Vec F16), counterAt F s = some c ∧
      runProtocol target [.draw (keyOfDraw seed c)] (absState F) = some (absState F') ∧
      StackWF F' ∧
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .F16 =
        .ok (.Ok ⟨.F16 w⟩) ∧
      (f16To64 w).isFinite = true ∧ 0 ≤ value (f16To64 w) ∧ value (f16To64 w) < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.F16 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt seed c i)) < value (f16To64 w) then F16.ZERO
        else f16Div x (f16Sub F16.ONE w)) := by
  obtain ⟨c, hc, ⟨b1, he1⟩, hrun, hwf'⟩ := dropout_forward_simulates target hsite hwf hno h
  obtain ⟨k, b2, w, ys, he2, hcast, hfin, h0, h1, hsh, hout, hys⟩ := dropout_f16 hs h
  rw [he1] at he2
  simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he2
  obtain ⟨rfl, -⟩ := he2
  exact ⟨c, w, ys, hc, hrun, hwf', hcast, hfin, h0, h1, hsh, hout, hys⟩

/-- **Composed bf16 dropout at a forward site.** As `dropout_forward_f16`, with the bf16 rate and
quotient. -/
theorem dropout_forward_bf16 (target : Target) {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId} {s : chelis_ir.evaluation.ScopeId}
    {xs : alloc.vec.Vec BF16} (hs : input.storage = ⟨.Bf16 xs⟩)
    (hsite : siteOf F node = .ok (some (.Forward draw s))) (hwf : StackWF F)
    (hno : ∀ c, counterAt F s = some c → c.val + 1 < 2 ^ 64)
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (c : U64) (w : BF16) (ys : alloc.vec.Vec BF16), counterAt F s = some c ∧
      runProtocol target [.draw (keyOfDraw seed c)] (absState F) = some (absState F') ∧
      StackWF F' ∧
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .Bf16 =
        .ok (.Ok ⟨.Bf16 w⟩) ∧
      (bf16To64 w).isFinite = true ∧ 0 ≤ value (bf16To64 w) ∧ value (bf16To64 w) < 1 ∧
      out.shape = input.shape ∧ out.storage = ⟨.Bf16 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt seed c i)) < value (bf16To64 w) then BF16.ZERO
        else bf16Div x (bf16Sub BF16.ONE w)) := by
  obtain ⟨c, hc, ⟨b1, he1⟩, hrun, hwf'⟩ := dropout_forward_simulates target hsite hwf hno h
  obtain ⟨k, b2, w, ys, he2, hcast, hfin, h0, h1, hsh, hout, hys⟩ := dropout_bf16 hs h
  rw [he1] at he2
  simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he2
  obtain ⟨rfl, -⟩ := he2
  exact ⟨c, w, ys, hc, hrun, hwf', hcast, hfin, h0, h1, hsh, hout, hys⟩

/-! ## Runs of dropouts -/

/-- The output relation of one accepted dropout against the key `k` it used, per float
storage dtype (the characterizations of `dropout_f64` … `dropout_bf16` with `k`). Integer
storages are outside the reference Random domain and are not related. -/
def Masked (k : chelis_ir.evaluation.DrawKey) (input : TV) (rate : F64) (out : TV) : Prop :=
  out.shape = input.shape ∧
  (∀ xs, input.storage = ⟨.F64 xs⟩ →
    rate.isFinite = true ∧ 0 ≤ value rate ∧ value rate < 1 ∧
    ∃ ys : alloc.vec.Vec F64, out.storage = ⟨.F64 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if (keyOfDraw k.seed k.ordinal).unitValue i < value rate then zero64
        else F64.div x (F64.sub one64 rate))) ∧
  (∀ xs, input.storage = ⟨.F32 xs⟩ →
    (F32.toF64 (F64.toF32 rate)).isFinite = true ∧ 0 ≤ value (F32.toF64 (F64.toF32 rate)) ∧
    value (F32.toF64 (F64.toF32 rate)) < 1 ∧
    ∃ ys : alloc.vec.Vec F32, out.storage = ⟨.F32 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt k.seed k.ordinal i)) < value (F32.toF64 (F64.toF32 rate))
        then zero32 else F32.div x (F32.sub one32 (F64.toF32 rate)))) ∧
  (∀ xs, input.storage = ⟨.F16 xs⟩ →
    ∃ (w : F16) (ys : alloc.vec.Vec F16),
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .F16 =
        .ok (.Ok ⟨.F16 w⟩) ∧
      (f16To64 w).isFinite = true ∧ 0 ≤ value (f16To64 w) ∧ value (f16To64 w) < 1 ∧
      out.storage = ⟨.F16 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt k.seed k.ordinal i)) < value (f16To64 w) then F16.ZERO
        else f16Div x (f16Sub F16.ONE w))) ∧
  (∀ xs, input.storage = ⟨.Bf16 xs⟩ →
    ∃ (w : BF16) (ys : alloc.vec.Vec BF16),
      chelis_types.dtype_semantics.cast_raw (toStr "dropout") (.Float rate) .Bf16 =
        .ok (.Ok ⟨.Bf16 w⟩) ∧
      (bf16To64 w).isFinite = true ∧ 0 ≤ value (bf16To64 w) ∧ value (bf16To64 w) < 1 ∧
      out.storage = ⟨.Bf16 ys⟩ ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt k.seed k.ordinal i)) < value (bf16To64 w) then BF16.ZERO
        else bf16Div x (bf16Sub BF16.ONE w)))

/-- Per-row tensor correspondence for the actual f64 output of an accepted dropout. The
shape, decoding, rate realization, denominator, and quotient exactness obligations stay local
to each row; no global representability assumption is hidden in `Masked`. -/
def TensorMaskedF64 (d : TV × F64 × TV) (k : chelis_ir.evaluation.DrawKey) : Prop :=
  ∀ (shape : Shape) (xs : alloc.vec.Vec F64) (x : Tensor ℚ shape)
      (dropoutRate : DropoutRate),
    d.1.shape.val.map (fun dim => dim.val) = shape →
    d.1.storage = ⟨.F64 xs⟩ →
    decodeTensor shape xs.val = some x →
    realize d.2.1 = some dropoutRate.val →
    Representable .binary64 (1 - dropoutRate.val) →
    (∀ (j : ℕ) (hj : j < xs.val.length) (a : ℚ), realize xs.val[j] = some a →
      Representable .binary64 (a / (1 - dropoutRate.val))) →
    ∃ (ys : alloc.vec.Vec F64) (y : Tensor ℚ shape),
      d.2.2.shape = d.1.shape ∧
      d.2.2.shape.val.map (fun dim => dim.val) = shape ∧
      d.2.2.storage = ⟨.F64 ys⟩ ∧
      decodeTensor shape ys.val = some y ∧
      ∀ i, y i = x i * dropoutMask dropoutRate (keyOfDraw k.seed k.ordinal) i


/-- An accepted dropout's output is masked by the key its `enter` returned. -/
theorem dropout_masked {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (back' : Frame → Frame),
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') ∧
      Masked k input rate out := by
  obtain ⟨k, back', he⟩ := dropout_enter h
  have same : ∀ {k' : chelis_ir.evaluation.DrawKey} {b' : Frame → Frame},
      chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k', F', b') → k' = k := by
    intro k' b' he'
    rw [he] at he'
    simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he'
    exact he'.1.symm
  obtain ⟨-, -, -, -, -, -, -, -, -, hsh⟩ := dropout_ok_inv h
  refine ⟨k, back', he, hsh, ?_, ?_, ?_, ?_⟩
  · intro xs hs
    obtain ⟨k', b', ys, he', hfin, h0, h1, -, hout, hys⟩ := dropout_f64 hs h
    cases same he'
    exact ⟨hfin, h0, h1, ys, hout, hys⟩
  · intro xs hs
    obtain ⟨k', b', ys, he', hfin, h0, h1, -, hout, hys⟩ := dropout_f32 hs h
    cases same he'
    exact ⟨hfin, h0, h1, ys, hout, hys⟩
  · intro xs hs
    obtain ⟨k', b', w, ys, he', hc, hfin, h0, h1, -, hout, hys⟩ := dropout_f16 hs h
    cases same he'
    exact ⟨w, ys, hc, hfin, h0, h1, hout, hys⟩
  · intro xs hs
    obtain ⟨k', b', w, ys, he', hc, hfin, h0, h1, -, hout, hys⟩ := dropout_bf16 hs h
    cases same he'
    exact ⟨w, ys, hc, hfin, h0, h1, hout, hys⟩

/-- One operation of a run with dropouts: a source control, or an `ExecutionFrame::dropout` of
`input` at `node` with the plan's `rate` and raw `seed`, returning `out`. -/
inductive DropOp where
  | control (c : chelis_ir.execution_spine.Control)
  | dropout (node : chelis_ir.dag.NodeId) (input : TV) (rate : F64) (seed : U64) (out : TV)

/-- The frame operation of a run operation (a dropout enters its random site). -/
def DropOp.frameOp : DropOp → FrameOp
  | .control c => .control c
  | .dropout node _ _ seed _ => .site node seed

/-- The `(input, rate, output)` of the dropouts of a run, in order. -/
def DropOp.dropouts : List DropOp → List (TV × F64 × TV)
  | [] => []
  | .control _ :: ops => DropOp.dropouts ops
  | .dropout _ input rate _ out :: ops => (input, rate, out) :: DropOp.dropouts ops

/-- `DropoutRun F ops evs ks F'`: the extracted `control` and `ExecutionFrame::dropout` accept
every operation of `ops` in order (each returns `Ok`, a dropout returning its `out`) and lead
from `F` to `F'`; `evs` are the reference protocol events and `ks` the key each dropout used,
in order. A forward dropout draws `(seed, c)` from the counter `c` of its active scope; a replay
uses the key recorded for its draw. The premises beyond acceptance are the domain restrictions
of `AcceptedRun` (fresh seeded scopes, no forward counter overflow); `c` and the recorded key
are not restrictions: an accepted dropout has them (`dropout_forward_key`,
`dropout_replay_key`). -/
inductive DropoutRun : Frame → List DropOp → List ProtocolEvent →
    List chelis_ir.evaluation.DrawKey → Frame → Prop
  | nil (F : Frame) : DropoutRun F [] [] [] F
  | enter (F F' F'' : Frame) (scope : chelis_ir.evaluation.ScopeId) (payload sd : U64)
      (back : Frame → Frame) (ops : List DropOp) (evs : List ProtocolEvent)
      (ks : List chelis_ir.evaluation.DrawKey)
      (h : chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
        .ok (.Ok (), F', back))
      (hseed : F.metadata.scopes.val[scope.val]? = some ⟨some sd⟩)
      (hfresh : scope ∉ F.scopes.val)
      (rest : DropoutRun F' ops evs ks F'') :
      DropoutRun F (.control (.Enter scope payload) :: ops)
        (.enter (UInt64.ofBitVec sd.bv) :: evs) ks F''
  | leave (F F' F'' : Frame) (scope : chelis_ir.evaluation.ScopeId) (back : Frame → Frame)
      (ops : List DropOp) (evs : List ProtocolEvent) (ks : List chelis_ir.evaluation.DrawKey)
      (h : chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) = .ok (.Ok (), F', back))
      (rest : DropoutRun F' ops evs ks F'') :
      DropoutRun F (.control (.Leave scope) :: ops) (.leave :: evs) ks F''
  | forward (F F' F'' : Frame) (node : chelis_ir.dag.NodeId) (input : TV) (rate : F64)
      (seed : U64) (out : TV) (draw : chelis_ir.evaluation.DrawId)
      (s : chelis_ir.evaluation.ScopeId) (c : U64) (back : Frame → Frame) (ops : List DropOp)
      (evs : List ProtocolEvent) (ks : List chelis_ir.evaluation.DrawKey)
      (hsite : siteOf F node = .ok (some (.Forward draw s)))
      (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
        .ok (.Ok out, F', back))
      (hc : counterAt F s = some c) (hno : c.val + 1 < 2 ^ 64)
      (rest : DropoutRun F' ops evs ks F'') :
      DropoutRun F (.dropout node input rate seed out :: ops)
        (.draw (keyOfDraw seed c) :: evs) (⟨seed, c⟩ :: ks) F''
  | replay (F F' F'' : Frame) (node : chelis_ir.dag.NodeId) (input : TV) (rate : F64)
      (seed : U64) (out : TV) (draw : chelis_ir.evaluation.DrawId)
      (k : chelis_ir.evaluation.DrawKey) (back : Frame → Frame) (ops : List DropOp)
      (evs : List ProtocolEvent) (ks : List chelis_ir.evaluation.DrawKey)
      (hsite : siteOf F node = .ok (some (.Replay draw)))
      (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
        .ok (.Ok out, F', back))
      (hk : F.keys.val[draw.val]? = some (some k))
      (rest : DropoutRun F' ops evs ks F'') :
      DropoutRun F (.dropout node input rate seed out :: ops) evs (k :: ks) F''

/-- An accepted replay dropout used the key recorded for its draw. -/
theorem dropout_replay_key {F : Frame} {node : chelis_ir.dag.NodeId} {input : TV} {rate : F64}
    {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId}
    (hsite : siteOf F node = .ok (some (.Replay draw)))
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ k, F.keys.val[draw.val]? = some (some k) ∧ F' = F ∧
      ∃ back', chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back') := by
  obtain ⟨k, back', he⟩ := dropout_enter h
  obtain ⟨hF, hk⟩ := enter_replay_inv F node seed draw k F' back' hsite he
  exact ⟨k, hk, hF, back', he⟩

/-- **Composed tensor f64 dropout at a replay site.** The accepted production call uses exactly
the key currently recorded at `F.keys[draw]`, and replay leaves the frame unchanged. Under the
same exact-shape, decoded-input, realized-rate, denominator, and per-element quotient hypotheses
as the forward theorem, the actual returned f64 storage decodes to the rational mask for that
recorded key. -/
theorem dropout_replay_tensor_f64 {F : Frame} {node : chelis_ir.dag.NodeId}
    {input : TV} {rate : F64} {seed : U64} {out : TV} {F' : Frame} {back : Frame → Frame}
    {draw : chelis_ir.evaluation.DrawId}
    {shape : Shape} {xs : alloc.vec.Vec F64} {x : Tensor ℚ shape}
    {dropoutRate : DropoutRate} (hs : input.storage = ⟨.F64 xs⟩)
    (hinputShape : input.shape.val.map (fun d => d.val) = shape)
    (hx : decodeTensor shape xs.val = some x)
    (hrate : realize rate = some dropoutRate.val)
    (hden : Representable .binary64 (1 - dropoutRate.val))
    (hq : ∀ (j : ℕ) (hj : j < xs.val.length) (a : ℚ), realize xs.val[j] = some a →
      Representable .binary64 (a / (1 - dropoutRate.val)))
    (hsite : siteOf F node = .ok (some (.Replay draw)))
    (h : chelis_ir.evaluation.ExecutionFrame.dropout F node input rate seed =
      .ok (.Ok out, F', back)) :
    ∃ (k : chelis_ir.evaluation.DrawKey) (ys : alloc.vec.Vec F64) (y : Tensor ℚ shape),
      F.keys.val[draw.val]? = some (some k) ∧ F' = F ∧
      out.shape = input.shape ∧
      out.shape.val.map (fun d => d.val) = shape ∧
      out.storage = ⟨.F64 ys⟩ ∧
      decodeTensor shape ys.val = some y ∧
      ∀ i, y i = x i * dropoutMask dropoutRate (keyOfDraw k.seed k.ordinal) i := by
  obtain ⟨replayKey, hrecorded, hframe, enterBack, he⟩ := dropout_replay_key hsite h
  obtain ⟨p, rsv, actualKey, actualBack, hp, hcast, hnew, he', happly, hshape⟩ :=
    dropout_ok_inv h
  have hkey : actualKey = replayKey := by
    rw [he] at he'
    simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he'
    exact he'.1.symm
  rw [hs] at hp hnew happly
  simp only [chelis_types.dtype_semantics.TensorStorage.prim, Result.ok.injEq] at hp
  subst p
  rw [cast_raw_float_f64] at hcast
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at hcast
  subst rsv
  obtain ⟨ys, y, hmask, hdecode, hy⟩ :=
    mask_tensor_f64 xs rate dropoutRate hrate actualKey.seed actualKey.ordinal x hx hden hq
  rw [hmask] at happly
  simp only [Result.ok.injEq, core.result.Result.Ok.injEq] at happly
  have hstorage : out.storage = ⟨.F64 ys⟩ := happly.symm
  rw [hkey] at hy
  exact ⟨replayKey, ys, y, hrecorded, hframe, hshape,
    by rw [hshape]; exact hinputShape, hstorage, hdecode, hy⟩


/-- **A dropout run is an accepted frame run.** From a well-formed stack, the frame operations
of a dropout run form an `AcceptedRun` with the same events, and the stack stays well formed. -/
theorem dropoutRun_accepted {F : Frame} {ops : List DropOp} {evs : List ProtocolEvent}
    {ks : List chelis_ir.evaluation.DrawKey} {F' : Frame} (h : DropoutRun F ops evs ks F')
    (hwf : StackWF F) :
    AcceptedRun F (ops.map DropOp.frameOp) evs F' ∧ StackWF F' := by
  induction h with
  | nil F => exact ⟨.nil F, hwf⟩
  | enter F F' F'' scope payload sd back ops evs ks h hseed hfresh rest ih =>
    have step := AcceptedRun.enter F F' F' scope payload sd back [] [] h hseed hfresh (.nil F')
    obtain ⟨hr, hwf1⟩ := ih (accepted_run_simulates .c step hwf).2
    exact ⟨.enter F F' F'' scope payload sd back _ evs h hseed hfresh hr, hwf1⟩
  | leave F F' F'' scope back ops evs ks h rest ih =>
    have step := AcceptedRun.leave F F' F' scope back [] [] h (.nil F')
    obtain ⟨hr, hwf1⟩ := ih (accepted_run_simulates .c step hwf).2
    exact ⟨.leave F F' F'' scope back _ evs h hr, hwf1⟩
  | forward F F' F'' node input rate seed out draw s c back ops evs ks hsite h hc hno rest ih =>
    have hno' : ∀ c', counterAt F s = some c' → c'.val + 1 < 2 ^ 64 := by
      intro c' hc'
      rw [hc] at hc'
      cases hc'
      exact hno
    obtain ⟨c', hc', ⟨b', he⟩, -, hwf1⟩ := dropout_forward_simulates .c hsite hwf hno' h
    rw [hc] at hc'
    cases hc'
    obtain ⟨hr, hwf2⟩ := ih hwf1
    exact ⟨.forward F F' F'' node seed draw s ⟨seed, c⟩ b' _ evs hsite he hno hr, hwf2⟩
  | replay F F' F'' node input rate seed out draw k back ops evs ks hsite h hk rest ih =>
    obtain ⟨k', hk', rfl, b', he⟩ := dropout_replay_key hsite h
    rw [hk] at hk'
    cases hk'
    obtain ⟨hr, hwf1⟩ := ih hwf
    exact ⟨.replay _ _ F'' node seed draw k b' _ evs hsite he hr, hwf1⟩

/-- **Run-level dropout composition.** From a well-formed stack, every dropout run of the
extracted frame is a run of the reference `runProtocol` between the abstracted frames (for
every target), keeps the stack well formed, and each of its dropouts, in order, returns its
input masked (`Masked`) by the key it used: the drawn reference key `(seed, c)` of the active
scope for a forward dropout (the event emitted at that step), the recorded key for a replay. -/
theorem dropoutRun_simulates (target : Target) {F : Frame} {ops : List DropOp}
    {evs : List ProtocolEvent} {ks : List chelis_ir.evaluation.DrawKey} {F' : Frame}
    (h : DropoutRun F ops evs ks F') (hwf : StackWF F) :
    runProtocol target evs (absState F) = some (absState F') ∧ StackWF F' ∧
      List.Forall₂ (fun (d : TV × F64 × TV) k => Masked k d.1 d.2.1 d.2.2)
        (DropOp.dropouts ops) ks := by
  refine ⟨(accepted_run_simulates target (dropoutRun_accepted h hwf).1 hwf).1,
    (dropoutRun_accepted h hwf).2, ?_⟩
  clear hwf
  induction h with
  | nil F => exact .nil
  | enter F F' F'' scope payload sd back ops evs ks h hseed hfresh rest ih => exact ih
  | leave F F' F'' scope back ops evs ks h rest ih => exact ih
  | forward F F' F'' node input rate seed out draw s c back ops evs ks hsite h hc hno rest ih =>
    obtain ⟨c', hc', b', he⟩ := dropout_forward_key hsite h
    rw [hc] at hc'
    cases hc'
    obtain ⟨k, b'', he', hm⟩ := dropout_masked h
    rw [he] at he'
    simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he'
    obtain ⟨rfl, -⟩ := he'
    exact .cons hm ih
  | replay F F' F'' node input rate seed out draw k back ops evs ks hsite h hk rest ih =>
    obtain ⟨k', hk', -, b', he⟩ := dropout_replay_key hsite h
    rw [hk] at hk'
    cases hk'
    obtain ⟨k'', b'', he', hm⟩ := dropout_masked h
    rw [he] at he'
    simp only [Result.ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at he'
    obtain ⟨rfl, -⟩ := he'
    exact .cons hm ih

/-- **Run-level tensor dropout composition.** The protocol transition and final stack
well-formedness come from `dropoutRun_simulates`; each row is then related by induction over
the actual `DropoutRun` constructors. Forward rows use their fresh-scope key, replay rows use
the currently recorded key and emit no event. Every row retains explicit shape, decode, rate,
denominator, and input-quotient hypotheses. -/
theorem dropoutRun_tensor_f64 (target : Target) {F : Frame} {ops : List DropOp}
    {evs : List ProtocolEvent} {ks : List chelis_ir.evaluation.DrawKey} {F' : Frame}
    (h : DropoutRun F ops evs ks F') (hwf : StackWF F) :
    runProtocol target evs (absState F) = some (absState F') ∧ StackWF F' ∧
      List.Forall₂ (fun d k => TensorMaskedF64 d k) (DropOp.dropouts ops) ks := by
  have hrows : List.Forall₂ (fun d k => TensorMaskedF64 d k) (DropOp.dropouts ops) ks := by
    revert hwf
    induction h with
    | nil F =>
        intro hwf
        exact .nil
    | enter F F' F'' scope payload sd back ops evs ks h hseed hfresh rest ih =>
        intro hwf
        have step := AcceptedRun.enter F F' F' scope payload sd back [] [] h hseed hfresh
          (.nil F')
        exact ih (accepted_run_simulates .c step hwf).2
    | leave F F' F'' scope back ops evs ks h rest ih =>
        intro hwf
        have step := AcceptedRun.leave F F' F' scope back [] [] h (.nil F')
        exact ih (accepted_run_simulates .c step hwf).2
    | forward F F' F'' node input rate seed out draw s c back ops evs ks
        hsite h hc hno rest ih =>
        intro hwf
        have hno' : ∀ c', counterAt F s = some c' → c'.val + 1 < 2 ^ 64 := by
          intro c' hc'
          rw [hc] at hc'
          cases hc'
          exact hno
        obtain ⟨_, _, _, _, hWFout⟩ := dropout_forward_simulates .c hsite hwf hno' h
        refine .cons ?_ (ih hWFout)
        · intro shape xs x dropoutRate hshape hs hx hrate hden hq
          obtain ⟨c', ys, y, hc', -, -, houtShape, houtDims,
            houtStorage, hdecode, hy⟩ :=
            dropout_forward_tensor_f64 .c hs hshape hx hrate hden hq hsite hwf hno' h
          have hcEq : c' = c := (Option.some.inj (hc.symm.trans hc')).symm
          subst c'
          exact ⟨ys, y, houtShape, houtDims, houtStorage, hdecode, hy⟩
    | replay F F' F'' node input rate seed out draw k back ops evs ks
        hsite h hk rest ih =>
        intro hwf
        obtain ⟨recorded, hrecorded, hframe, -, -⟩ := dropout_replay_key hsite h
        have hrow : TensorMaskedF64 (input, rate, out) k := by
          intro shape xs x dropoutRate hshape hs hx hrate hden hq
          obtain ⟨k', ys, y, hk', -, houtShape, houtDims, houtStorage, hdecode, hy⟩ :=
            dropout_replay_tensor_f64 hs hshape hx hrate hden hq hsite h
          have hkEq : k' = k :=
            Option.some.inj (Option.some.inj (hk'.symm.trans hk))
          subst k'
          exact ⟨ys, y, houtShape, houtDims, houtStorage, hdecode, hy⟩
        have hwf' : StackWF F' := by
          rw [hframe]
          exact hwf
        exact .cons hrow (ih hwf')
  obtain ⟨hrun, hwf', -⟩ := dropoutRun_simulates target h hwf
  exact ⟨hrun, hwf', hrows⟩

/-- **Protocol acceptance of a dropout run.** A dropout run from a root frame back to a root
frame passes the reference `checkProtocol`. -/
theorem dropoutRun_checks (target : Target) {F : Frame} {ops : List DropOp}
    {evs : List ProtocolEvent} {ks : List chelis_ir.evaluation.DrawKey} {F' : Frame}
    (h : DropoutRun F ops evs ks F') (h0 : F.scopes.val = [0#usize])
    (h1 : F'.scopes.val = [0#usize]) :
    checkProtocol target evs (absState F) = true :=
  accepted_run_checks target (dropoutRun_accepted h (stackWF_root F h0)).1 h0 h1

end ChelisBridge.Random
