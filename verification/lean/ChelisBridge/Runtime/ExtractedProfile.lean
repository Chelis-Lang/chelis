import ChelisBridge.Runtime.ProfileModels

/-!
These primitive relations support different profile oracles and profile receipts.
The runtime observation includes hook installation.
Detailed clock allocation needs a clock count below `Usize.max`.

The model retains `maximumSizeExceeded` outside that domain.
Declaration reports can change the profile-only size receipt, not the runtime.
Whole-core caller proofs must use these relations at the sites in `runtime_sites.json`.
-/

namespace ChelisBridge.Runtime.ExtractedProfile

open Aeneas.Std
open ChelisBridge.Runtime.ProfileModels

abbrev Services := chelis_types.native.NativeServices
abbrev Runtime := chelis_ir.lowering_runtime.LoweringRuntime

def RuntimeRelated (a b : Runtime) : Prop :=
  runtimeCore a = runtimeCore b

def ServicesRelated (a b : Services) : Prop :=
  servicesCore a = servicesCore b

theorem profile_start_related (a b : Services) (h : ServicesRelated a b) :
    ∃ ma a' mb b',
      chelis_types.native.NativeServices.profile_start a = .ok (ma, a') ∧
      chelis_types.native.NativeServices.profile_start b = .ok (mb, b') ∧ ServicesRelated a' b' := by
  rcases profile_start_core a with ⟨ma, a', ha, hca⟩
  rcases profile_start_core b with ⟨mb, b', hb, hcb⟩
  exact ⟨ma, a', mb, b', ha, hb, hca.trans (h.trans hcb.symm)⟩

theorem profile_phase_related (a b : Services) (h : ServicesRelated a b)
    (ma mb : Usize) (sa la sb lb : Str) :
    chelis_types.native.NativeServices.profile_phase a ma sa la = .ok a ∧
    chelis_types.native.NativeServices.profile_phase b mb sb lb = .ok b ∧ ServicesRelated a b :=
  ⟨profile_phase_core a ma sa la, profile_phase_core b mb sb lb, h⟩

theorem detail_profile_start_related (a b : Runtime) (h : RuntimeRelated a b)
    (ma mb : Usize) (a' b' : Runtime)
    (ha : chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start a = .ok (ma, a'))
    (hb : chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start b = .ok (mb, b')) :
    RuntimeRelated a' b' := by
  exact ((detail_profile_start_core a).1 _ ha).trans
    (h.trans ((detail_profile_start_core b).1 _ hb).symm)

theorem detail_profile_phase_related (a b : Runtime) (h : RuntimeRelated a b)
    (ma mb : Usize) (la lb : Str) :
    chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_phase a ma la = .ok a ∧
    chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_phase b mb lb = .ok b ∧
      RuntimeRelated a b :=
  ⟨detail_profile_phase_core a ma la, detail_profile_phase_core b mb lb, h⟩

theorem detail_profile_declaration_start_related
    (a b : Runtime) (h : RuntimeRelated a b) (sa sb ma mb : Usize) (a' b' : Runtime)
    (ha : chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start a sa = .ok (ma, a'))
    (hb : chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start b sb = .ok (mb, b')) :
    RuntimeRelated a' b' := by
  exact ((detail_profile_declaration_start_core a sa).1 _ ha).trans
    (h.trans ((detail_profile_declaration_start_core b sb).1 _ hb).symm)

theorem detail_profile_declaration_related
    (a b : Runtime) (h : RuntimeRelated a b)
    (sa sb lasta lastb total : Usize) (la lb : Str) :
    ∃ ra rb,
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration a sa lasta total la =
        .ok (a, ra) ∧
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration b sb lastb total lb =
        .ok (b, rb) ∧
      RuntimeRelated a b :=
  ⟨_, _, detail_profile_declaration_core a sa lasta total la,
    detail_profile_declaration_core b sb lastb total lb, h⟩

theorem report_internal_related (a b : Runtime) (h : RuntimeRelated a b) (la lb : Str) :
    chelis_ir.lowering_runtime.LoweringRuntime.report_internal a la = .ok a ∧
    chelis_ir.lowering_runtime.LoweringRuntime.report_internal b lb = .ok b ∧ RuntimeRelated a b :=
  ⟨report_internal_core a la, report_internal_core b lb, h⟩

theorem detail_profile_start_defined (s : Runtime)
    (h : s.profileMarks.val < Usize.max) :
    ∃ mark s', chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start s =
      .ok (mark, s') := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start
  dsimp only
  split
  · unfold chelis_ir.lowering_runtime.LoweringRuntime.profile_mark
    simp only [h, if_true]
    exact ⟨_, _, rfl⟩
  · exact ⟨_, _, rfl⟩

theorem detail_profile_declaration_start_defined (s : Runtime) (start : Usize)
    (h : s.profileMarks.val < Usize.max) :
    ∃ mark s', chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start s start =
      .ok (mark, s') := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start
  split
  · exact ⟨_, _, rfl⟩
  · unfold chelis_ir.lowering_runtime.LoweringRuntime.profile_mark
    simp only [h, if_true]
    exact ⟨_, _, rfl⟩

theorem detail_profile_start_noninterference (a b : Runtime) (h : RuntimeRelated a b)
    (ha : a.profileMarks.val < Usize.max) (hb : b.profileMarks.val < Usize.max) :
    ∃ ma a' mb b',
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start a = .ok (ma, a') ∧
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start b = .ok (mb, b') ∧
      RuntimeRelated a' b' := by
  rcases detail_profile_start_defined a ha with ⟨ma, a', hra⟩
  rcases detail_profile_start_defined b hb with ⟨mb, b', hrb⟩
  exact ⟨ma, a', mb, b', hra, hrb, detail_profile_start_related a b h ma mb a' b' hra hrb⟩

theorem detail_profile_declaration_start_noninterference
    (a b : Runtime) (h : RuntimeRelated a b) (sa sb : Usize)
    (ha : a.profileMarks.val < Usize.max) (hb : b.profileMarks.val < Usize.max) :
    ∃ ma a' mb b',
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start a sa = .ok (ma, a') ∧
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start b sb = .ok (mb, b') ∧
      RuntimeRelated a' b' := by
  rcases detail_profile_declaration_start_defined a sa ha with ⟨ma, a', hra⟩
  rcases detail_profile_declaration_start_defined b sb hb with ⟨mb, b', hrb⟩
  exact ⟨ma, a', mb, b', hra, hrb,
    detail_profile_declaration_start_related a b h sa sb ma mb a' b' hra hrb⟩

theorem detail_profile_start_exhaustion (s : Runtime)
    (hmax : s.profileMarks.val = Usize.max) :
    chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start { s with profileEnv := fun _ => true } =
        .fail .maximumSizeExceeded ∧
      chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start { s with profileEnv := fun _ => false } =
        .ok (0#usize, { s with
          profileEnv := fun _ => false
          profileEnvReads := s.profileEnvReads + 1 }) := by
  simp [chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start,
    chelis_ir.lowering_runtime.LoweringRuntime.profile_mark, hmax]

end ChelisBridge.Runtime.ExtractedProfile
