import ChelisBridge.Numeric.Loop
import ChelisBridge.Numeric.Primal

/-!
# End-to-end gradient observation (bridge item 4) for the flat node loop

The post-AD graph returned by `grad_dag_checked` is decoded by the bridge over the source
entry context (`decodeArtifact` totalized, `decodeArtifactRoots` over its own Rust roots).
`artifactRootsF` is the same decoding with IEEE constants: the decoder's own node pass
lifted with the Rust payload bits, followed by the totalization's zero constants as `+0`.

`grad_observation`: if the reference correspondence check accepts the artifact against
`p.compileGrad`, then on inputs bound by name that realize to `ρq`, under an exact run of
the artifact's IEEE program, the flat node loop on the returned graph succeeds and

* each returned gradient root realizes, row-major, to the entry's component of
  `p.compileGrad.eval ρq` (the reference all-entry gradient);
* each entry without a returned gradient has reference gradient `0` (the totalization's
  zero fill is exact);
* if `acceptPrimal` accepts, the root holding `output_node` realizes to `p.eval ρq`.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore ChelisBridge

variable {f : BinaryFormat}

/-- The IEEE zero constant appended for a disconnected entry. -/
def liftZero : GraphSyntax.Operation ℚ → Option (GraphSyntax.Operation (BinaryFloat f))
  | .const s d =>
      if d = List.replicate s.prod 0 then some (.const s (List.replicate s.prod (BinaryFloat.zero f false)))
      else none
  | _ => none

theorem value_zero : value (BinaryFloat.zero f false) = 0 :=
  (realize_eq_some.mp (realize_zero f false)).2

theorem liftZero_sound {o : GraphSyntax.Operation ℚ} {o' : GraphSyntax.Operation (BinaryFloat f)}
    (h : liftZero o = some o') : mapData value o' = o := by
  cases o <;> simp [liftZero] at h
  obtain ⟨rfl, rfl⟩ := h
  simp [mapData, List.map_replicate, value_zero]

theorem forall₂_liftZero :
    ∀ {zs : List (GraphSyntax.Operation ℚ)} {zsF : List (GraphSyntax.Operation (BinaryFloat f))},
      List.Forall₂ (fun o o' => liftZero o = some o') zs zsF → zsF.map (mapData value) = zs
  | _, _, .nil => rfl
  | _, _, .cons ha h => by simp [liftZero_sound ha, forall₂_liftZero h]

theorem mapM_liftZero_sound {zs : List (GraphSyntax.Operation ℚ)}
    {zsF : List (GraphSyntax.Operation (BinaryFloat f))} (h : zs.mapM liftZero = some zsF) :
    zsF.map (mapData value) = zs :=
  forall₂_liftZero ((mapM_eq_some_iff _ _ _).mp h)

theorem totalizeAux_zeros (st : State) :
    ∀ (items : List (Shape × RootSource)) (zs0 zs : List (GraphSyntax.Operation ℚ))
      (owners : List Nat), totalizeAux st items zs0 = some (zs, owners) →
      (∀ o ∈ zs0, ∃ s, o = zeroConst s) → ∀ o ∈ zs, ∃ s, o = zeroConst s
  | [], zs0, zs, owners, h, h0 => by
      simp only [totalizeAux, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, _⟩ := h; exact h0
  | (_, .present g) :: rest, zs0, zs, owners, h, h0 => by
      simp only [totalizeAux, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨sl, _, h⟩ := h
      split at h
      · simp only [Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
        obtain ⟨⟨zs', owners'⟩, hr, h⟩ := h
        simp only [Prod.mk.injEq] at h
        obtain ⟨rfl, _⟩ := h
        exact totalizeAux_zeros st rest zs0 zs' owners' hr h0
      · cases h
  | (s, .zero) :: rest, zs0, zs, owners, h, h0 => by
      simp only [totalizeAux, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨⟨zs', owners'⟩, hr, h⟩ := h
      simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, _⟩ := h
      refine totalizeAux_zeros st rest _ zs' owners' hr ?_
      intro o ho
      rcases List.mem_append.mp ho with ho | ho
      · exact h0 o ho
      · exact ⟨s, by simpa using ho⟩

theorem mapM_liftZero_complete :
    ∀ {zs : List (GraphSyntax.Operation ℚ)}, (∀ o ∈ zs, ∃ s, o = zeroConst s) →
      ∃ zsF : List (GraphSyntax.Operation (BinaryFloat f)), zs.mapM liftZero = some zsF
  | [], _ => ⟨[], rfl⟩
  | o :: rest, h => by
      obtain ⟨s, rfl⟩ := h o (by simp)
      obtain ⟨zsF, hF⟩ := mapM_liftZero_complete (fun o ho => h o (List.mem_cons_of_mem _ ho))
      exact ⟨.const s (List.replicate s.prod (BinaryFloat.zero f false)) :: zsF,
        by simp [List.mapM_cons, liftZero, zeroConst, hF]⟩

/-- The artifact's `Load` node ids for the source entries (by input name). -/
def artifactIds (source : Model.Dag) (entries : List Nat) (r : Model.GradResult) :
    Option (List (Option Nat)) :=
  (entryNames source entries).bind (entryIdsIn r.dag)

/-- **The IEEE program of the post-AD artifact over its own roots.** -/
def artifactRootsF (dt : DType) (source : Model.Dag) (entries : List Nat) {Γ : Context}
    (p : Program ℚ Γ []) (r : Model.GradResult) (Ω : Context) :
    Option (Circuit (BinaryFloat dt.format) Γ Ω) :=
  (artifactParts dt source entries p r).bind fun x =>
    (artifactIds source entries r).bind fun ids =>
      (runF dt ids Γ 0 (State.init Γ) [] r.dag.nodes.val).bind fun y =>
        (x.2.1.mapM liftZero).bind fun zsF =>
          (rootOffsets x.1 x.2.1.length (artifactRootIds r)).bind fun offs =>
            (⟨y.2 ++ zsF, offs⟩ : GraphSyntax.Target (BinaryFloat dt.format)).decode Γ Ω

theorem artifactParts_unfold {dt : DType} {source : Model.Dag} {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {r : Model.GradResult} {st : State}
    {zs : List (GraphSyntax.Operation ℚ)} {owners : List Nat}
    (h : artifactParts dt source entries p r = some (st, zs, owners)) :
    ∃ ids, artifactIds source entries r = some ids ∧
      run dt ids Γ 0 (State.init Γ) r.dag.nodes.val = some st ∧
      ∀ o ∈ zs, ∃ s, o = zeroConst s := by
  unfold artifactParts at h
  split at h
  · simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨ids, hids, st', hg, ⟨zs', owners'⟩, ht, hx⟩ := h
    simp only [Prod.mk.injEq] at hx
    obtain ⟨rfl, rfl, rfl⟩ := hx
    refine ⟨ids, by simp only [artifactIds, Option.bind_eq_some_iff]; exact hids, ?_, ?_⟩
    · simp only [decodeGraphOver, Option.bind_eq_bind, Option.bind_eq_some_iff] at hg
      obtain ⟨_, _, hg⟩ := hg
      exact hg
    · simp only [totalizeParts, Option.bind_eq_some_iff] at ht
      obtain ⟨srcs, _, ht⟩ := ht
      exact totalizeAux_zeros _ _ [] _ _ ht (by simp)
  · cases h

/-- Completeness and soundness of the artifact's IEEE program. -/
theorem artifactRootsF_complete {dt : DType} {source : Model.Dag} {entries : List Nat}
    {Γ : Context} {p : Program ℚ Γ []} {r : Model.GradResult} {Ω : Context} {rc : Circuit ℚ Γ Ω}
    (hR : decodeArtifactRoots dt source entries p r = some ⟨Ω, rc⟩) :
    ∃ cf, artifactRootsF dt source entries p r Ω = some cf ∧ ratCircuit cf = rc := by
  obtain ⟨st, zs, owners, offs, hp, ho, hd⟩ := decodeArtifactRoots_spec hR
  obtain ⟨ids, hids, hrun, hzero⟩ := artifactParts_unfold hp
  obtain ⟨ops, hrunF⟩ := runF_complete [] hrun
  obtain ⟨hrun', hops⟩ := runF_sound (by rfl) hrunF
  obtain ⟨zsF, hzF⟩ := mapM_liftZero_complete (f := dt.format) hzero
  have hzs := mapM_liftZero_sound hzF
  have hdec := targetDecode_mapData (Γ := Γ) (Ω := Ω) value
    (⟨ops ++ zsF, offs⟩ : GraphSyntax.Target (BinaryFloat dt.format))
  simp only [List.map_append, hops, hzs] at hdec
  rw [hd] at hdec
  obtain ⟨cf, hcf, hmap⟩ := Option.map_eq_some_iff.mp hdec.symm
  refine ⟨cf, ?_, ?_⟩
  · simp only [artifactRootsF, hp, Option.bind_some, hids, hrunF, hzF, ho, hcf]
  · rw [← hmap]; simp only [ratCircuit, ratPrefix_eq_mapPrefix]

/-- The flat node loop on the returned graph computes the artifact's IEEE program at
every Rust root. -/
theorem artifact_simulates {dt : DType} {source : Model.Dag} {entries : List Nat}
    {Γ : Context} {p : Program ℚ Γ []} {r : Model.GradResult} {Ω : Context}
    {cf : Circuit (BinaryFloat dt.format) Γ Ω} {ids : List (Option Nat)}
    (hcf : artifactRootsF dt source entries p r Ω = some cf)
    (hids : artifactIds source entries r = some ids)
    {live : Nat → Bool} {inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format))}
    {ρ : Values (BinaryFloat dt.format) Γ} (hinputs : InputsOK dt r.dag ids inputs ρ)
    (hclosed : LiveClosed r.dag live) (hroots : ∀ g ∈ artifactRootIds r, live g = true) :
    ∃ vals, specRun dt inputs live r.dag.nodes.val (fun _ => none) = some vals ∧
      ∀ {s : Shape} (w : Var Ω s) (g : Nat), (artifactRootIds r)[w.offset]? = some g →
        ∃ d, vals g = some ⟨s, d⟩ ∧ Tensor.ofList s d = some (w.get (ieeeEval cf ρ)) := by
  simp only [artifactRootsF, Option.bind_eq_some_iff] at hcf
  obtain ⟨⟨st, zs, owners⟩, hp, ids', hids', ⟨st', ops⟩, hrunF, zsF, hzF, offs, ho, hc⟩ := hcf
  dsimp only at hzF ho hc
  rw [hids] at hids'
  cases Option.some.inj hids'
  obtain ⟨ids0, hids0, hrun, _⟩ := artifactParts_unfold hp
  rw [hids] at hids0
  cases Option.some.inj hids0
  obtain ⟨hrun', _⟩ := runF_sound (by rfl) hrunF
  rw [hrun] at hrun'
  cases Option.some.inj hrun'
  obtain ⟨Δ, code, σ, hdp, hσ, rfl⟩ := Target.decode_spec hc
  have hlenF := runF_len (by rfl) (by simp [State.init]) hrunF
  have hΔ := (decodePrefix_spec _ ⟨Δ, code⟩ hdp).1
  have hzlen : zsF.length = zs.length := by rw [← mapM_liftZero_sound hzF, List.length_map]
  obtain ⟨vals, hrunS, hinv⟩ := runF_simulates (inputs := inputs) (live := live) (ρ := ρ)
    hdp hinputs hclosed (fun n hn => hn) hrunF (by rfl) (by simp [State.init]) Inv.init
  refine ⟨vals, hrunS, fun {s} w g hr => ?_⟩
  have hf := (mapM_eq_some_iff _ _ _).mp ho
  obtain ⟨i, hi, hgi⟩ := forall₂_getElem? hf hr
  simp only [Option.bind_eq_some_iff] at hgi
  obtain ⟨sl, hsl, hgi⟩ := hgi
  split at hgi
  · rename_i hlt
    cases Option.some.inj hgi
    have hvar := decodeRoots_get hσ w hi
    obtain ⟨d, hvd, hok⟩ := hinv g sl hsl (hroots g (List.mem_of_getElem? hr))
    obtain ⟨w0, hw0⟩ := hok.1
    have hoff := Var.offset_of_decode hvar
    dsimp only at hlenF hΔ
    simp only [List.length_append] at hΔ
    have hs : s = sl.shape := by
      have h1 := Var.getElem_offset (σ w)
      have h2 := Var.getElem_offset w0
      rw [hoff] at h1
      have : w0.offset = st.len + zs.length - 1 - sl.owner := by omega
      rw [this, h1] at h2
      exact Option.some.inj h2
    subst hs
    refine ⟨d, hvd, ?_⟩
    simp only [ieeeEval, Renaming.get_read]
    exact hok.2 (σ w) (by omega)
  · cases hgi

/-- **Gradient observation (bridge item 4) for the flat node loop on the post-AD graph.** -/
theorem grad_observation {dt : DType} {source : Model.Dag} {entries : List Nat} {loss : Nat}
    {Γ : Context} {p : Program ℚ Γ []} {r : Model.GradResult} {actual : Circuit ℚ Γ Γ}
    {Ω : Context} {rc : Circuit ℚ Γ Ω} {adW : List (Option Nat)}
    (hsrc : decodeSource dt source entries loss = some ⟨Γ, p⟩)
    (hA : decodeArtifact dt source entries p r = some actual)
    (hR : decodeArtifactRoots dt source entries p r = some ⟨Ω, rc⟩)
    (hcheck : GraphCorrespondence.check p.compileGrad actual adW = true) :
    ∃ cf ids, artifactRootsF dt source entries p r Ω = some cf ∧ ratCircuit cf = rc ∧
      artifactIds source entries r = some ids ∧
      ∀ {live : Nat → Bool} {inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format))}
        {ρf : Values (BinaryFloat dt.format) Γ} {ρq : Values ℚ Γ},
        InputsOK dt r.dag ids inputs ρf → Realizes ρf ρq → ExactRun cf ρf →
        LiveClosed r.dag live → (∀ g ∈ artifactRootIds r, live g = true) →
        ∃ vals, specRun dt inputs live r.dag.nodes.val (fun _ => none) = some vals ∧
          (∀ {s : Shape} (u : Var Γ s) (w : Var Ω s) (e g : Nat),
            entries[u.offset]? = some e → (GradArtifact.grad r e).join = some g →
            (artifactRootIds r)[w.offset]? = some g →
            ∃ d, vals g = some ⟨s, d⟩ ∧ decodeTensor s d = some (u.get (p.compileGrad.eval ρq))) ∧
          (∀ {s : Shape} (u : Var Γ s) (e : Nat), entries[u.offset]? = some e →
            (GradArtifact.grad r e).join = none → u.get (p.compileGrad.eval ρq) = 0) ∧
          (∀ (w : Var Ω []), (artifactRootIds r)[w.offset]? = some r.output_node.val →
            (∀ ρ : Values ℚ Γ, w.get (rc.eval ρ) = p.eval ρ) →
            ∃ d, vals r.output_node.val = some ⟨[], d⟩ ∧ decodeTensor [] d = some (p.eval ρq)) := by
  have _ := hsrc
  obtain ⟨cf, hcf, hrat⟩ := artifactRootsF_complete hR
  obtain ⟨st, zs, owners, _, hp, _, _⟩ := decodeArtifactRoots_spec hR
  obtain ⟨ids, hids, _, _⟩ := artifactParts_unfold hp
  refine ⟨cf, ids, hcf, hrat, hids, fun {live inputs ρf ρq} hin hρ hex hcl hro => ?_⟩
  obtain ⟨vals, hrun, hroot⟩ := artifact_simulates hcf hids hin hcl hro
  have hreal := ieeeEval_realize cf hρ hex
  rw [hrat] at hreal
  have hq : ∀ {s : Shape} (w : Var Ω s) (g : Nat), (artifactRootIds r)[w.offset]? = some g →
      ∃ d, vals g = some ⟨s, d⟩ ∧ decodeTensor s d = some (w.get (rc.eval ρq)) := by
    intro s w g hr
    obtain ⟨d, hvd, hd⟩ := hroot w g hr
    exact ⟨d, hvd, decodeTensor_of_ofList hd (hreal.get w)⟩
  refine ⟨vals, hrun, ?_, ?_, ?_⟩
  · intro s u w e g he hg hr
    obtain ⟨d, hvd, hd⟩ := hq w g hr
    refine ⟨d, hvd, ?_⟩
    rw [hd, artifact_root_link hA hR u w he hg hr ρq,
      ← GraphCorrespondence.check_sound _ _ _ hcheck ρq]
  · intro s u e he hmiss
    rw [Program.compileGrad_correct]
    exact grad_get_of_disconnected p ρq u (decodeArtifact_missing_disconnected hA he hmiss)
  · intro w hr hprim
    obtain ⟨d, hvd, hd⟩ := hq w _ hr
    exact ⟨d, hvd, by rw [hd, hprim]⟩

/-- The realized Rust gradient, read in `ℝ`, is the real derivative of the source loss at the
realized inputs: `fderiv ℝ p.eval (toReal ρq) dρ () = ⟨dρ, toReal (p.compileGrad.eval ρq)⟩`. -/
theorem compileGrad_real {Γ : Context} (p : Program ℚ Γ []) (ρq : Values ℚ Γ) (dρ : Values ℝ Γ) :
    fderiv ℝ p.eval (toReal ρq) dρ () = Values.dot dρ (toReal (p.compileGrad.eval ρq)) := by
  rw [← circuit_eval_toReal]
  exact p.compiledGrad_correct (toReal ρq) dρ

end ChelisBridge.Numeric
