import ChelisBridge.Float
import LaCaDiLECore.Circuit
import Mathlib.Algebra.BigOperators.Fin

/-!
# IEEE reference evaluation of the seven-operation fragment and exact observation

The decoded reference syntax (`Op`, `Prefix`, `Circuit`, `Program` of `LaCaDiLECore`) is
parametric in its coefficient type. Instantiated at `BinaryFloat f`, constants are the
actual IEEE encodings of the Rust payload. `ratOp`/`ratPrefix`/`ratCircuit`/`ratProgram`
replace every constant by its exact binary rational (`value`) and are the `ℚ` programs
the decoder produces.

`opEval`/`prefixEval`/`ieeeEval`/`programEval` evaluate the IEEE syntax the way the Rust
kernels in `crates/chelis-types/src/dtype_semantics.rs` and `crates/chelis-ir/src/eval.rs`
do for a homogeneous floating dtype whose `Sum` accumulator is that dtype:

* `add`/`mul`: one IEEE operation per element of equal-shape operands, `lhs ∘ rhs` in port
  order (`float_vec_binop_f64` / `float_vec_binop_f32`). Rank-zero broadcasting is not part
  of the fragment (the decoder rejects it, `classify_add`/`classify_mul`).
* `sum a`: the group of an output coordinate lists the reduced axis in increasing order
  (`axis_reduction_groups`), and `checked_adjacent_pair_fold` combines adjacent pairs one
  level at a time, carrying an odd last element unchanged; an empty group is `+0`
  (`reduction_seed … 0.0`). A one-element group returns its element without an addition.
* `insert`/`expand`: element-preserving gathers (`expand`: rank-increasing reads the
  coordinate with the axis removed, same-rank reads coordinate `0` of the singleton axis).
* `const`/`copy`: the payload/operand unchanged.

`ExactRun` is the per-execution exactness obligation of bridge item 3: along the IEEE
evaluation on the actual operand values, every constant element is finite and every
executed `add`, `mul` and pairwise `sum` combination has finite operands whose exact
rational result is representable in the format (`ExactAdd`/`ExactMul`), i.e. the step
neither rounds, overflows nor produces NaN. It is stated over the operand encodings and
never mentions the reference result.

`exact_run_realize` / `ieeeEval_realize` / `programEval_realize`: under `ExactRun` and
finite inputs, realizing the IEEE outputs is the reference evaluation of the rational
program on the realized inputs; `ieeeEval_realize_real` transports this to `ℝ`.

Counterexamples (kernel-evaluated, binary64): `finite_run_not_exact` (every value of the
run finite, output `0`, reference `1`), `reassociation_changes_result` (two programs with
equal reference semantics, IEEE results `1` and `0`), `rust_sum_order_matters` (the Rust
pairwise sum is not invariant under permuting its leaves) and `mul_overflow_not_finite`.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore ChelisBridge

variable {f : BinaryFormat}

/-! ## Rust's canonical pairwise sum -/

/-- One level of `checked_adjacent_pair_fold`: combine adjacent pairs, carry an odd last
element unchanged. -/
def pairLevel : List (BinaryFloat f) → List (BinaryFloat f)
  | a :: b :: rest => BinaryFloat.add a b :: pairLevel rest
  | [a] => [a]
  | [] => []

/-- `checked_adjacent_pair_fold` with fuel (structural, hence kernel-reducible); `pairFold`
supplies enough fuel. An empty group is the positive zero seed. -/
def pairFoldN : Nat → List (BinaryFloat f) → BinaryFloat f
  | _, [] => BinaryFloat.zero f false
  | _, [x] => x
  | 0, x :: _ :: _ => x
  | n + 1, a :: b :: rest => pairFoldN n (pairLevel (a :: b :: rest))

/-- Rust's typed `Sum` reduction of one ordered group. -/
def pairFold (l : List (BinaryFloat f)) : BinaryFloat f := pairFoldN l.length l

/-- Every combination of one pairing level is exact. -/
def PairLevelExact : List (BinaryFloat f) → Prop
  | a :: b :: rest => ExactAdd a b ∧ PairLevelExact rest
  | [_] => True
  | [] => True

/-- Every combination executed by `pairFoldN` is exact. -/
def ExactFoldN : Nat → List (BinaryFloat f) → Prop
  | _, [] => True
  | _, [_] => True
  | 0, _ :: _ :: _ => True
  | n + 1, a :: b :: rest => PairLevelExact (a :: b :: rest) ∧ ExactFoldN n (pairLevel (a :: b :: rest))

def ExactFold (l : List (BinaryFloat f)) : Prop := ExactFoldN l.length l

/-- Exact rational image of a pairing level. -/
def pairLevelQ : List ℚ → List ℚ
  | a :: b :: rest => (a + b) :: pairLevelQ rest
  | [a] => [a]
  | [] => []

theorem pairLevelQ_sum : ∀ l : List ℚ, (pairLevelQ l).sum = l.sum
  | a :: b :: rest => by simp [pairLevelQ, pairLevelQ_sum rest, add_assoc]
  | [a] => rfl
  | [] => rfl

theorem pairLevel_length_le : ∀ l : List (BinaryFloat f), (pairLevel l).length ≤ l.length
  | a :: b :: rest => by
      have := pairLevel_length_le rest
      simp only [pairLevel, List.length_cons]; omega
  | [a] => le_refl _
  | [] => le_refl _

/-- Elementwise realization relation. -/
def Realizes {ι : Type} (x : ι → BinaryFloat f) (y : ι → ℚ) : Prop := ∀ i, realize (x i) = some (y i)

theorem pairLevel_realize :
    ∀ (l : List (BinaryFloat f)) (qs : List ℚ),
      List.Forall₂ (fun x q => realize x = some q) l qs → PairLevelExact l →
        List.Forall₂ (fun x q => realize x = some q) (pairLevel l) (pairLevelQ qs)
  | [], qs, h, _ => by cases h; exact List.Forall₂.nil
  | [a], qs, h, _ => by
      rcases h with _ | ⟨ha, hrest⟩
      cases hrest
      exact List.Forall₂.cons ha List.Forall₂.nil
  | a :: b :: rest, qs, h, hex => by
      match qs, h with
      | qa :: qb :: qrest, .cons ha (.cons hb hrest) =>
        obtain ⟨hab, hex⟩ := hex
        obtain ⟨a', b', ha', hb', hrep⟩ := hab
        rw [ha] at ha'; rw [hb] at hb'
        cases Option.some.inj ha'; cases Option.some.inj hb'
        exact List.Forall₂.cons (realize_add ha hb hrep) (pairLevel_realize rest qrest hrest hex)

/-- **Exact pairwise sum.** If every combination of the Rust pairwise fold is exact, the
fold realizes to the exact sum of the realized leaves (for any adequate fuel). -/
theorem pairFoldN_realize :
    ∀ (n : Nat) (l : List (BinaryFloat f)) (qs : List ℚ), l.length ≤ n + 1 →
      List.Forall₂ (fun x q => realize x = some q) l qs → ExactFoldN n l →
        realize (pairFoldN n l) = some qs.sum
  | n, [], qs, _, h, _ => by
      cases h; simp only [pairFoldN, List.sum_nil]; exact realize_zero f false
  | n, [a], qs, _, h, _ => by
      rcases h with _ | ⟨ha, hrest⟩
      cases hrest
      simpa [pairFoldN] using ha
  | 0, a :: b :: rest, qs, hlen, _, _ => by simp at hlen
  | n + 1, a :: b :: rest, qs, hlen, h, hex => by
      obtain ⟨hlev, hex⟩ := hex
      have hl := pairLevel_realize (a :: b :: rest) qs h hlev
      have hle : (pairLevel (a :: b :: rest)).length ≤ n + 1 := by
        have := pairLevel_length_le rest
        simp only [pairLevel, List.length_cons] at hlen ⊢; omega
      rw [pairFoldN, pairFoldN_realize n _ _ hle hl hex, pairLevelQ_sum]

theorem pairFold_realize {l : List (BinaryFloat f)} {qs : List ℚ}
    (h : List.Forall₂ (fun x q => realize x = some q) l qs) (hex : ExactFold l) :
    realize (pairFold l) = some qs.sum :=
  pairFoldN_realize l.length l qs (Nat.le_succ _) h hex

/-! ## IEEE evaluation of the reference syntax -/

variable {Γ Δ Ω : Context} {s : Shape}

/-- The Rust leaf order of the group of output coordinate `i`: the reduced axis in
increasing order. -/
def sumLeaves {n : Nat} {s t : Shape} (a : Axis s n t) (x : Tensor (BinaryFloat f) s) (i : Index t) :
    List (BinaryFloat f) :=
  List.ofFn fun j : Fin n => x (a.join (j, i))

/-- IEEE evaluation of one operation, mirroring the Rust kernels. -/
def opEval : Op (BinaryFloat f) Γ s → Values (BinaryFloat f) Γ → Tensor (BinaryFloat f) s
  | .const x, _ => x
  | .copy v, ρ => v.get ρ
  | .add v w, ρ => fun i => BinaryFloat.add (v.get ρ i) (w.get ρ i)
  | .mul v w, ρ => fun i => BinaryFloat.mul (v.get ρ i) (w.get ρ i)
  | .sum a v, ρ => fun i => pairFold (sumLeaves a (v.get ρ) i)
  | .insert a v, ρ => fun i => v.get ρ (a.split i).2
  | .expand pre post n v, ρ => fun i =>
      v.get ρ ((Axis.atPosition pre 1 post).join (0, ((Axis.atPosition pre n post).split i).2))

def prefixEval {Γ Δ : Context} : Prefix (BinaryFloat f) Γ Δ → Values (BinaryFloat f) Γ →
    Values (BinaryFloat f) Δ
  | .nil, ρ => ρ
  | .bind op rest, ρ => prefixEval rest (Values.cons (opEval op ρ) ρ)

/-- **IEEE reference evaluation** of a decoded circuit (prefix and ordered roots). -/
def ieeeEval (c : Circuit (BinaryFloat f) Γ Ω) (ρ : Values (BinaryFloat f) Γ) :
    Values (BinaryFloat f) Ω :=
  Renaming.read c.result (prefixEval c.code ρ)

def programEval {Γ : Context} {s : Shape} : Program (BinaryFloat f) Γ s →
    Values (BinaryFloat f) Γ → Tensor (BinaryFloat f) s
  | .ret v, ρ => v.get ρ
  | .letBind op body, ρ => programEval body (Values.cons (opEval op ρ) ρ)

/-! ## The per-execution exactness predicate -/

/-- Exactness of one executed operation on its actual operand encodings. -/
def OpExact : Op (BinaryFloat f) Γ s → Values (BinaryFloat f) Γ → Prop
  | .const x, _ => ∀ i, (x i).isFinite = true
  | .copy _, _ => True
  | .add v w, ρ => ∀ i, ExactAdd (v.get ρ i) (w.get ρ i)
  | .mul v w, ρ => ∀ i, ExactMul (v.get ρ i) (w.get ρ i)
  | .sum a v, ρ => ∀ i, ExactFold (sumLeaves a (v.get ρ) i)
  | .insert _ _, _ => True
  | .expand _ _ _ _, _ => True

/-- **`ExactRun`**: every operation executed by the IEEE evaluation of the prefix is exact
on the values it actually receives. -/
def PrefixExact {Γ Δ : Context} : Prefix (BinaryFloat f) Γ Δ → Values (BinaryFloat f) Γ → Prop
  | .nil, _ => True
  | .bind op rest, ρ => OpExact op ρ ∧ PrefixExact rest (Values.cons (opEval op ρ) ρ)

def ExactRun (c : Circuit (BinaryFloat f) Γ Ω) (ρ : Values (BinaryFloat f) Γ) : Prop :=
  PrefixExact c.code ρ

def ProgramExact {Γ : Context} {s : Shape} : Program (BinaryFloat f) Γ s →
    Values (BinaryFloat f) Γ → Prop
  | .ret _, _ => True
  | .letBind op body, ρ => OpExact op ρ ∧ ProgramExact body (Values.cons (opEval op ρ) ρ)

/-! ## Exact rational programs -/

def ratOp : Op (BinaryFloat f) Γ s → Op ℚ Γ s
  | .const x => .const fun i => value (x i)
  | .copy v => .copy v
  | .add v w => .add v w
  | .mul v w => .mul v w
  | .sum a v => .sum a v
  | .insert a v => .insert a v
  | .expand pre post n v => .expand pre post n v

def ratPrefix {Γ Δ : Context} : Prefix (BinaryFloat f) Γ Δ → Prefix ℚ Γ Δ
  | .nil => .nil
  | .bind op rest => .bind (ratOp op) (ratPrefix rest)

def ratCircuit (c : Circuit (BinaryFloat f) Γ Ω) : Circuit ℚ Γ Ω :=
  ⟨c.context, ratPrefix c.code, c.result⟩

def ratProgram {Γ : Context} {s : Shape} : Program (BinaryFloat f) Γ s → Program ℚ Γ s
  | .ret v => .ret v
  | .letBind op body => .letBind (ratOp op) (ratProgram body)

/-! ## Exact observation -/

theorem Realizes.get {ρf : Values (BinaryFloat f) Γ} {ρq : Values ℚ Γ} (h : Realizes ρf ρq)
    (v : Var Γ s) : Realizes (v.get ρf) (v.get ρq) := fun i => h (v.coord i)

theorem Realizes.cons {ρf : Values (BinaryFloat f) Γ} {ρq : Values ℚ Γ}
    {x : Tensor (BinaryFloat f) s} {y : Tensor ℚ s} (hx : Realizes x y) (h : Realizes ρf ρq) :
    Realizes (Values.cons x ρf) (Values.cons y ρq) := by
  intro c
  cases c with
  | inl i => exact hx i
  | inr c => exact h c

theorem Realizes.read :
    ∀ {Ω : Context} (σ : Renaming Ω Δ) {ρf : Values (BinaryFloat f) Δ} {ρq : Values ℚ Δ},
      Realizes ρf ρq → Realizes (Renaming.read σ ρf) (Renaming.read σ ρq)
  | [], _, _, _, _ => fun c => nomatch c
  | _ :: _, σ, _, _, h => by
      simp only [Renaming.read]
      exact Realizes.cons (h.get _) (Realizes.read (Renaming.tail σ) h)

theorem sumLeaves_realize {n : Nat} {s t : Shape} (a : Axis s n t) {x : Tensor (BinaryFloat f) s}
    {y : Tensor ℚ s} (h : Realizes x y) (i : Index t) :
    List.Forall₂ (fun x q => realize x = some q) (sumLeaves a x i)
      (List.ofFn fun j : Fin n => y (a.join (j, i))) := by
  unfold sumLeaves
  rw [List.forall₂_iff_get]
  refine ⟨by simp, fun k h1 h2 => ?_⟩
  simp only [List.get_eq_getElem, List.getElem_ofFn]
  exact h _

/-- One exact operation realizes to the reference operation on the realized operands. -/
theorem opEval_realize (op : Op (BinaryFloat f) Γ s) {ρf : Values (BinaryFloat f) Γ}
    {ρq : Values ℚ Γ} (hρ : Realizes ρf ρq) (hex : OpExact op ρf) :
    Realizes (opEval op ρf) ((ratOp op).eval ρq) := by
  cases op with
  | const x =>
      intro i
      simp only [opEval, ratOp, Op.eval, Tensor.interpret, Function.comp_apply,
        Algebra.algebraMap_self, RingHom.id_apply]
      exact realize_eq_some.mpr ⟨hex i, rfl⟩
  | copy v => exact hρ.get v
  | add v w =>
      intro i
      obtain ⟨a, b, ha, hb, hrep⟩ := hex i
      have ha' := hρ.get v i; have hb' := hρ.get w i
      rw [ha] at ha'; rw [hb] at hb'
      cases Option.some.inj ha'; cases Option.some.inj hb'
      exact realize_add ha hb hrep
  | mul v w =>
      intro i
      obtain ⟨a, b, ha, hb, hrep⟩ := hex i
      have ha' := hρ.get v i; have hb' := hρ.get w i
      rw [ha] at ha'; rw [hb] at hb'
      cases Option.some.inj ha'; cases Option.some.inj hb'
      exact realize_mul ha hb hrep
  | sum a v =>
      intro i
      simp only [opEval, ratOp, Op.eval, LaCaDiLECore.sum]
      rw [pairFold_realize (sumLeaves_realize a (hρ.get v) i) (hex i), List.sum_ofFn]
  | insert a v => intro i; exact hρ.get v _
  | expand pre post n v =>
      intro i
      simp only [opEval, ratOp, Op.eval, LaCaDiLECore.expand, LaCaDiLECore.insert,
        LaCaDiLECore.sum, Fin.sum_univ_one]
      exact hρ.get v _

/-- **Exact observation of a prefix.** -/
theorem exact_run_realize (p : Prefix (BinaryFloat f) Γ Δ) {ρf : Values (BinaryFloat f) Γ}
    {ρq : Values ℚ Γ} (hρ : Realizes ρf ρq) (hex : PrefixExact p ρf) :
    Realizes (prefixEval p ρf) ((ratPrefix p).eval ρq) := by
  induction p with
  | nil => exact hρ
  | bind op rest ih =>
      simp only [PrefixExact] at hex
      obtain ⟨hop, hrest⟩ := hex
      exact ih (Realizes.cons (opEval_realize op hρ hop) hρ) hrest

/-- **Numerical observation (bridge item 3) for the IEEE reference evaluation.** Under
finite inputs and an exact run, realizing every root of the IEEE evaluation of a decoded
circuit is the reference evaluation of the exact rational circuit on the realized inputs:
`realize (ieeeEval c xs) = (ratCircuit c).eval (realize xs)`. -/
theorem ieeeEval_realize (c : Circuit (BinaryFloat f) Γ Ω) {ρf : Values (BinaryFloat f) Γ}
    {ρq : Values ℚ Γ} (hρ : Realizes ρf ρq) (hex : ExactRun c ρf) :
    Realizes (ieeeEval c ρf) ((ratCircuit c).eval ρq) :=
  Realizes.read c.result (exact_run_realize c.code hρ hex)

theorem programEval_realize (p : Program (BinaryFloat f) Γ s) {ρf : Values (BinaryFloat f) Γ}
    {ρq : Values ℚ Γ} (hρ : Realizes ρf ρq) (hex : ProgramExact p ρf) :
    Realizes (programEval p ρf) ((ratProgram p).eval ρq) := by
  induction p with
  | ret v => exact hρ.get v
  | letBind op body ih =>
      simp only [ProgramExact] at hex
      obtain ⟨hop, hbody⟩ := hex
      exact ih (Realizes.cons (opEval_realize op hρ hop) hρ) hbody

/-! ## Real semantics -/

/-- Real values of rational inputs. -/
noncomputable def toReal {ι : Type} (x : ι → ℚ) : ι → ℝ := fun i => algebraMap ℚ ℝ (x i)

theorem toReal_get (v : Var Γ s) (ρ : Values ℚ Γ) : v.get (toReal ρ) = toReal (v.get ρ) := rfl

theorem toReal_cons (x : Tensor ℚ s) (ρ : Values ℚ Γ) :
    toReal (Values.cons x ρ) = Values.cons (toReal x) (toReal ρ) := by
  funext c; cases c <;> rfl

theorem op_eval_toReal (op : Op ℚ Γ s) (ρ : Values ℚ Γ) :
    op.eval (toReal ρ) = toReal (op.eval ρ) := by
  funext i
  cases op <;>
    simp [Op.eval, toReal, Var.get, Tensor.interpret, LaCaDiLECore.sum, LaCaDiLECore.insert,
      LaCaDiLECore.expand, map_sum]

theorem prefix_eval_toReal (p : Prefix ℚ Γ Δ) (ρ : Values ℚ Γ) :
    p.eval (toReal ρ) = toReal (p.eval ρ) := by
  induction p with
  | nil => rfl
  | bind op rest ih =>
      simp only [Prefix.eval]
      rw [op_eval_toReal, ← toReal_cons, ih]

theorem read_toReal : ∀ {Ω : Context} (σ : Renaming Ω Δ) (ρ : Values ℚ Δ),
    Renaming.read σ (toReal ρ) = toReal (Renaming.read σ ρ)
  | [], _, _ => funext fun c => nomatch c
  | _ :: _, σ, ρ => by
      simp only [Renaming.read]
      rw [read_toReal (Renaming.tail σ) ρ, toReal_get, toReal_cons]

theorem circuit_eval_toReal (c : Circuit ℚ Γ Ω) (ρ : Values ℚ Γ) :
    c.eval (toReal ρ) = toReal (c.eval ρ) := by
  simp only [Circuit.eval]
  rw [prefix_eval_toReal, read_toReal]

theorem program_eval_toReal (p : Program ℚ Γ s) (ρ : Values ℚ Γ) :
    p.eval (toReal ρ) = toReal (p.eval ρ) := by
  induction p with
  | ret v => rfl
  | letBind op body ih =>
      simp only [Program.eval]
      rw [op_eval_toReal, ← toReal_cons, ih]

/-- The real-number form: each root element realizes to a rational whose real value is the
reference real evaluation at the real values of the realized inputs. -/
theorem ieeeEval_realize_real (c : Circuit (BinaryFloat f) Γ Ω) {ρf : Values (BinaryFloat f) Γ}
    {ρq : Values ℚ Γ} (hρ : Realizes ρf ρq) (hex : ExactRun c ρf) (i : Coordinates Ω) :
    ∃ q, realize (ieeeEval c ρf i) = some q ∧
      algebraMap ℚ ℝ q = (ratCircuit c).eval (R := ℝ) (toReal ρq) i :=
  ⟨_, ieeeEval_realize c hρ hex i, by rw [circuit_eval_toReal]; rfl⟩

/-! ## Counterexamples (binary64, kernel-evaluated) -/

/-- Three scalar entries `x, y, z`. -/
abbrev Γ3 : Context := [[], [], []]

/-- `(x + y) + z`. -/
def leftAssoc : Program F64 Γ3 [] :=
  .letBind (.add .here (.there .here)) (.letBind (.add .here (.there (.there (.there .here)))) (.ret .here))

/-- `x + (y + z)`. -/
def rightAssoc : Program F64 Γ3 [] :=
  .letBind (.add (.there .here) (.there (.there .here))) (.letBind (.add (.there .here) .here) (.ret .here))

def scalars (x y z : F64) : Values F64 Γ3 :=
  Values.cons (fun _ => x) (Values.cons (fun _ => y) (Values.cons (fun _ => z) (fun c => nomatch c)))

def scalarsQ (x y z : ℚ) : Values ℚ Γ3 :=
  Values.cons (fun _ => x) (Values.cons (fun _ => y) (Values.cons (fun _ => z) (fun c => nomatch c)))

theorem scalars_realizes : Realizes (scalars c16 cneg16 cone) (scalarsQ (10 ^ 16) (-(10 ^ 16)) 1) := by
  intro c
  rcases c with _ | _ | _ | c
  · exact realize_c16
  · exact realize_cneg16
  · exact realize_cone
  · exact nomatch c

/-- Both association orders have the same reference semantics. -/
theorem assoc_same_reference (ρ : Values ℚ Γ3) :
    (ratProgram leftAssoc).eval ρ = (ratProgram rightAssoc).eval ρ := by
  funext i
  simp [ratProgram, ratOp, leftAssoc, rightAssoc, Program.eval, Op.eval, add_assoc]

theorem reference_value (i : Index []) :
    (ratProgram rightAssoc).eval (scalarsQ (10 ^ 16) (-(10 ^ 16)) 1) i = 1 := by
  simp [ratProgram, ratOp, rightAssoc, Program.eval, Op.eval, scalarsQ, Values.cons, Var.get,
    Var.coord]

/-- The documented native smoke `(1e16 + -1e16) + 1 = 1` versus `1e16 + (-1e16 + 1) = 0`:
reassociating a decoded program changes its IEEE result although its reference
semantics is unchanged (`assoc_same_reference`). -/
theorem reassociation_changes_result :
    realize (programEval leftAssoc (scalars c16 cneg16 cone) ()) = some 1 ∧
      realize (programEval rightAssoc (scalars c16 cneg16 cone) ()) = some 0 := by
  decide +kernel

/-- Every value of this run is finite: the inputs, the intermediate `-10^16 + 1` and the
output. -/
theorem rightAssoc_all_finite :
    c16.isFinite = true ∧ cneg16.isFinite = true ∧ cone.isFinite = true ∧
      (BinaryFloat.add cneg16 cone).isFinite = true ∧
      (programEval rightAssoc (scalars c16 cneg16 cone) ()).isFinite = true := by
  decide +kernel

/-- **Finiteness is not exactness.** A run whose inputs, intermediates and output are all
finite realizes to `0` while the reference evaluation of the decoded program is `1`; the
exactness hypothesis of `programEval_realize` fails on this run. -/
theorem finite_run_not_exact :
    (programEval rightAssoc (scalars c16 cneg16 cone) ()).isFinite = true ∧
      realize (programEval rightAssoc (scalars c16 cneg16 cone) ()) = some 0 ∧
      (ratProgram rightAssoc).eval (scalarsQ (10 ^ 16) (-(10 ^ 16)) 1) () = 1 ∧
      ¬ ProgramExact rightAssoc (scalars c16 cneg16 cone) := by
  refine ⟨rightAssoc_all_finite.2.2.2.2, reassociation_changes_result.2, reference_value (), ?_⟩
  intro hex
  have h := programEval_realize rightAssoc scalars_realizes hex ()
  rw [reassociation_changes_result.2, reference_value] at h
  exact absurd (Option.some.inj h) (by norm_num)

/-- The Rust pairwise `Sum` is not invariant under permuting its leaves, so the IEEE
reference must follow the Rust group order (reduced axis increasing) and fold shape. -/
theorem rust_sum_order_matters :
    realize (pairFold [c16, cneg16, cone]) = some 1 ∧
      realize (pairFold [cneg16, cone, c16]) = some 0 := by
  decide +kernel

/-- The largest finite binary64 value. -/
def maxF64 : F64 := BinaryFloat.pack .binary64 false (BinaryFormat.binary64.maxExponent - 1)
  (2 ^ 52 - 1)

def two64 : F64 := BinaryFloat.ofInt .binary64 2

/-- Overflow: finite operands whose exact product is not representable yield infinity,
which has no value; `ExactMul` excludes it. -/
theorem mul_overflow_not_finite :
    maxF64.isFinite = true ∧ two64.isFinite = true ∧
      (BinaryFloat.mul maxF64 two64).isFinite = false ∧
      realize (BinaryFloat.mul maxF64 two64) = none := by
  decide +kernel

end ChelisBridge.Numeric
