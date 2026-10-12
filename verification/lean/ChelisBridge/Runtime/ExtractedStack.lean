import ChelisBridge.Runtime.Stack
import ChelisBridge.Model

open Aeneas.Std
open_chelis_model

set_option maxHeartbeats 1000000

namespace ChelisBridge.Runtime.ExtractedStack

abbrev Budget := chelis_types.infer.stack.StackBudget
abbrev Record := chelis_types.infer.stack.StackBailRecord
abbrev Sink := chelis_types.session.DiagnosticSink
abbrev Services := Aeneas.Std.chelis_types.native.NativeServices

/-- The projection uses explicit label maps. -/
def project (label : String → Nat) (r : Record) (n : Services) : Stack.Explicit :=
  Stack.ofServices n r.scope_depth.val
    (r.first.map fun b => (label b.1, b.2.map label))

/-- The extracted frame exit is the model frame exit. -/
theorem exit_frame_eq (s : Budget) :
    chelis_types.infer.stack.StackBudget.exit_frame s =
      .ok ({ s with native := { s.native with
        depth := UScalar.saturating_sub s.native.depth 1#usize } }, id) := rfl

/-- Each successful exit moves the explicit model state. -/
theorem exit_frame_project (label : String → Nat) (s : Budget) :
    ∃ s' back, chelis_types.infer.stack.StackBudget.exit_frame s = .ok (s', back) ∧
      project label s'.record s'.native = (project label s.record s.native).exitFrame := by
  obtain ⟨n, hn, hp, _, _⟩ := Stack.cn_exit_frame s.native s.record.scope_depth.val
    (s.record.first.map fun b => (label b.1, b.2.map label))
  refine ⟨{ s with native := n }, id, ?_, hp⟩
  simp only [chelis_types.infer.stack.StackBudget.exit_frame, hn]
  rfl

/-- The passing guard consumes one probe and keeps its frame open. -/
theorem frame_exhausted_pass {F : Type}
    (inst : core.ops.function.FnOnce F Unit (Option Str)) (s : Budget)
    (site : Str) (span : F) (n : Services)
    (he : Aeneas.Std.chelis_types.native.NativeServices.enter_frame s.native = .ok n)
    (hp : n.stack n.stackTaken = false) :
    chelis_types.infer.stack.StackBudget.frame_exhausted inst s site span =
      .ok (false, { s with native := { n with stackTaken := n.stackTaken + 1 } }, id) := by
  simp only [chelis_types.infer.stack.StackBudget.frame_exhausted, he,
    Aeneas.Std.chelis_types.native.NativeServices.stack_exhausted, hp]
  rfl

/-- The exhausted guard records the bail and exits its frame. -/
theorem frame_exhausted_bail {F : Type}
    (inst : core.ops.function.FnOnce F Unit (Option Str)) (s : Budget)
    (site : Str) (span span' : F) (location : Option Str) (n : Services) (r : Record)
    (he : Aeneas.Std.chelis_types.native.NativeServices.enter_frame s.native = .ok n)
    (hp : n.stack n.stackTaken = true)
    (hs : inst.call_once span () = .ok (location, span'))
    (hr : chelis_types.infer.stack.StackBailRecord.record s.record site location = .ok r) :
    chelis_types.infer.stack.StackBudget.frame_exhausted inst s site span =
      .ok (true, { native := { n with stackTaken := n.stackTaken + 1,
        depth := UScalar.saturating_sub n.depth 1#usize }, record := r }, id) := by
  simp only [chelis_types.infer.stack.StackBudget.frame_exhausted, he,
    Aeneas.Std.chelis_types.native.NativeServices.stack_exhausted, hp, hs, hr,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame]
  rfl

/-- A previous bail stays first, even at another exhausted site. -/
theorem record_first (r : Record) (b : String × Option String) (site : Str)
    (span : Option Str) (h : r.first = some b) :
    chelis_types.infer.stack.StackBailRecord.record r site span = .ok r := by
  simp [chelis_types.infer.stack.StackBailRecord.record, h]

/-- The extracted outermost entry resets both fields before the increment. -/
theorem enter_scope_eq (r : Record) (n : Services) :
    chelis_types.infer.stack.StackBailRecord.enter_scope r n = (do
      let d ← r.scope_depth + 1#usize
      .ok ({ first := if r.scope_depth = 0#usize then none else r.first,
        scope_depth := d },
        if r.scope_depth = 0#usize then { n with depth := 0#usize } else n)) := by
  by_cases h : r.scope_depth = 0#usize <;>
    simp [chelis_types.infer.stack.StackBailRecord.enter_scope,
      Aeneas.Std.chelis_types.native.NativeServices.reset_frames, h]

/-- Draining an empty record gives no diagnostic and preserves the scope. -/
theorem take_error_none (r : Record) (h : r.first = none) :
    chelis_types.infer.stack.StackBailRecord.take_error r =
      .ok (none, { r with first := none }) := by
  simp [chelis_types.infer.stack.StackBailRecord.take_error, h,
    core.option.Option.map]

theorem exit_scope_eq (r : Record) :
    chelis_types.infer.stack.StackBailRecord.exit_scope r =
      .ok { r with scope_depth := UScalar.saturating_sub r.scope_depth 1#usize } := rfl

theorem take_error_eq (r : Record) :
    chelis_types.infer.stack.StackBailRecord.take_error r = (do
      let error ← core.option.Option.map
        chelis_types.infer.stack.StackBailRecord.take_error.closure.Insts.CoreOpsFunctionFnOnceTuplePairStringOptionStringCheckError
        r.first ()
      .ok (error, { r with first := none })) := rfl

theorem take_error_some (r : Record) (b : String × Option String)
    (error : chelis_types.errors.CheckError)
    (hf : r.first = some b)
    (he :
      chelis_types.infer.stack.StackBailRecord.take_error.closure.Insts.CoreOpsFunctionFnOnceTuplePairStringOptionStringCheckError.call_once
        () b = .ok error) :
    chelis_types.infer.stack.StackBailRecord.take_error r =
      .ok (some error, { r with first := none }) := by
  simp [take_error_eq, hf, core.option.Option.map, he,
    chelis_types.infer.stack.StackBailRecord.take_error.closure.Insts.CoreOpsFunctionFnOnceTuplePairStringOptionStringCheckError]

theorem take_error_idempotent (r : Record) :
    (do
      let (_, r') ← chelis_types.infer.stack.StackBailRecord.take_error r
      chelis_types.infer.stack.StackBailRecord.take_error r') = (do
      let (_, r') ← chelis_types.infer.stack.StackBailRecord.take_error r
      .ok (none, r')) := by
  simp only [take_error_eq]
  generalize core.option.Option.map
    chelis_types.infer.stack.StackBailRecord.take_error.closure.Insts.CoreOpsFunctionFnOnceTuplePairStringOptionStringCheckError
    r.first () = x
  cases x
  all_goals simp [core.option.Option.map]

theorem drain_stack_bail_none (s : Sink)
    (h : s.stack_bails.first = none) :
    chelis_types.session.DiagnosticSink.drain_stack_bail s =
      .ok ({ s with stack_bails := { s.stack_bails with first := none } },
        fun s' => { s with errors := s'.errors, native := s'.native }) := by
  simp only [chelis_types.session.DiagnosticSink.drain_stack_bail,
    take_error_none s.stack_bails h, bind_tc_ok]

theorem drain_stack_bail_some (s : Sink)
    (error : chelis_types.errors.CheckError) (r : Record)
    (errors : alloc.vec.Vec chelis_types.errors.CheckError)
    (ht : chelis_types.infer.stack.StackBailRecord.take_error s.stack_bails =
      .ok (some error, r))
    (hp : alloc.vec.Vec.push s.errors error = .ok errors) :
    chelis_types.session.DiagnosticSink.drain_stack_bail s =
      .ok ({ s with stack_bails := r, errors := errors },
        fun s' => { s with errors := s'.errors, native := s'.native }) := by
  simp only [chelis_types.session.DiagnosticSink.drain_stack_bail, ht, hp, bind_tc_ok]

theorem enter_scope_project (label : String → Nat) (r : Record) (n : Services)
    (d : Usize) (hadd : r.scope_depth + 1#usize = .ok d) :
    ∃ r' n', chelis_types.infer.stack.StackBailRecord.enter_scope r n =
      .ok (r', n') ∧ project label r' n' = (project label r n).enterScope := by
  have hd := UScalar.add_equiv r.scope_depth 1#usize
  rw [hadd] at hd
  have hv : d.val = r.scope_depth.val + 1 := by simpa using hd.2.1
  by_cases h : r.scope_depth = 0#usize
  · refine ⟨{ first := none, scope_depth := d }, { n with depth := 0#usize }, ?_, ?_⟩
    · simp [enter_scope_eq, h, hadd]
    · have hz : r.scope_depth.val = 0 := congrArg UScalar.val h
      simp [project, Stack.ofServices, Stack.Explicit.enterScope, hz, hv]
  · refine ⟨{ first := r.first, scope_depth := d }, n, ?_, ?_⟩
    · simp [enter_scope_eq, h, hadd]
    · have hz : r.scope_depth.val ≠ 0 := by
        intro hz
        exact h (UScalar.eq_of_val_eq hz)
      simp [project, Stack.ofServices, Stack.Explicit.enterScope, hz, hv]

theorem take_error_clears (r r' : Record) (error : Option chelis_types.errors.CheckError)
    (h : chelis_types.infer.stack.StackBailRecord.take_error r = .ok (error, r')) :
    r'.first = none ∧ r'.scope_depth = r.scope_depth := by
  rw [take_error_eq] at h
  generalize core.option.Option.map
    chelis_types.infer.stack.StackBailRecord.take_error.closure.Insts.CoreOpsFunctionFnOnceTuplePairStringOptionStringCheckError
    r.first () = x at h
  cases x <;> simp_all

/-- All return values retain the explicit frame exit. -/
theorem annotate_annotate_expr_with_scope_returns_exit
  (expr : chelis_deep.ast.Expr)
  (product : chelis_types.infer.checked.InferenceProduct)
  (annotation_context :
  chelis_types.infer.annotate.AnnotationResolutionContext)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.annotate.annotate_expr_with_scope expr product annotation_context errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.annotate.annotate_expr_with_scope.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "annotate_expr_with_scope") expr
  if b
  then
    let result ← chelis_deep.ast.Expr.Insts.CoreCloneClone.clone expr
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (result, errors1, back)
  else
    let (result, errors2, annotate_expr_with_scope_frame_back) ←
      chelis_types.infer.annotate.annotate_expr_with_scope_frame expr
        product annotation_context errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           annotate_expr_with_scope_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (result, errors3, back)) := by
  simp only [chelis_types.infer.annotate.annotate_expr_with_scope, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_helpers_descend_to_tail_var_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_helpers.descend_to_tail_var expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_helpers.descend_to_tail_var.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "descend_to_tail_var") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, descend_to_tail_var_frame_back) ←
      chelis_types.infer.app_helpers.descend_to_tail_var_frame expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := descend_to_tail_var_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.app_helpers.descend_to_tail_var, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_shape_extract_int_for_dim_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_shape.extract_int_for_dim expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_shape.extract_int_for_dim.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "extract_int_for_dim") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, extract_int_for_dim_frame_back) ←
      chelis_types.infer.app_shape.extract_int_for_dim_frame expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := extract_int_for_dim_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.app_shape.extract_int_for_dim, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_shape_helpers_classify_size_in_returns_exit
  (expr : chelis_deep.ast.Expr)
  (ctx : chelis_types.infer.app_shape_helpers.SizeCtx)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_shape_helpers.classify_size_in expr ctx stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_shape_helpers.classify_size_in.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "classify_expand_size") expr
  if b
  then
    let back :=
      (fun ctx1 => { ctx with subst := ctx1.subst } :
        chelis_types.infer.app_shape_helpers.SizeCtx →
        chelis_types.infer.app_shape_helpers.SizeCtx)
    let back1 :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok
      (chelis_types.infer.app_shape_helpers.SizeClass.Unknown, ctx,
      back, stack1, back1)
  else
    let (result, ctx1, classify_size_in_frame_back, stack2,
      classify_size_in_frame_back1) ←
      chelis_types.infer.app_shape_helpers.classify_size_in_frame expr ctx
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun ctx2 =>
         let sc :=
           classify_size_in_frame_back { ctx1 with subst := ctx2.subst }
         { ctx with subst := sc.subst } :
        chelis_types.infer.app_shape_helpers.SizeCtx →
        chelis_types.infer.app_shape_helpers.SizeCtx)
    let back1 :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := classify_size_in_frame_back1 sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, ctx1, back, stack3, back1)) := by
  simp only [chelis_types.infer.app_shape_helpers.classify_size_in, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_shape_helpers_extract_numeric_leaf_for_shape_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_shape_helpers.extract_numeric_leaf_for_shape expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_shape_helpers.extract_numeric_leaf_for_shape.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "extract_numeric_leaf_for_shape") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, extract_numeric_leaf_for_shape_frame_back) ←
      chelis_types.infer.app_shape_helpers.extract_numeric_leaf_for_shape_frame
        expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := extract_numeric_leaf_for_shape_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.app_shape_helpers.extract_numeric_leaf_for_shape, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_shape_helpers_shape_operand_type_returns_exit
  (operand : chelis_deep.ast.Expr)
  (ctx : chelis_types.infer.app_shape_helpers.SizeCtx)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_shape_helpers.shape_operand_type operand ctx stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_shape_helpers.shape_operand_type.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "shape_operand_type") operand
  if b
  then
    let back :=
      (fun ctx1 => { ctx with subst := ctx1.subst } :
        chelis_types.infer.app_shape_helpers.SizeCtx →
        chelis_types.infer.app_shape_helpers.SizeCtx)
    let back1 :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, ctx, back, stack1, back1)
  else
    let (result, ctx1, shape_operand_type_frame_back, stack2,
      shape_operand_type_frame_back1) ←
      chelis_types.infer.app_shape_helpers.shape_operand_type_frame operand ctx
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun ctx2 =>
         let sc :=
           shape_operand_type_frame_back { ctx1 with subst := ctx2.subst }
         { ctx with subst := sc.subst } :
        chelis_types.infer.app_shape_helpers.SizeCtx →
        chelis_types.infer.app_shape_helpers.SizeCtx)
    let back1 :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := shape_operand_type_frame_back1 sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, ctx1, back, stack3, back1)) := by
  simp only [chelis_types.infer.app_shape_helpers.shape_operand_type, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_shape_helpers_shape_read_operand_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_shape_helpers.shape_read_operand expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_shape_helpers.shape_read_operand.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "shape_read_operand") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, shape_read_operand_frame_back) ←
      chelis_types.infer.app_shape_helpers.shape_read_operand_frame expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := shape_read_operand_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.app_shape_helpers.shape_read_operand, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem app_shape_helpers_walk_static_cons_chain_shape_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.app_shape_helpers.walk_static_cons_chain_shape expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.app_shape_helpers.walk_static_cons_chain_shape.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "walk_static_cons_chain_shape") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, walk_static_cons_chain_shape_frame_back) ←
      chelis_types.infer.app_shape_helpers.walk_static_cons_chain_shape_frame
        expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := walk_static_cons_chain_shape_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.app_shape_helpers.walk_static_cons_chain_shape, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem checked_collect_defsig_param_types_returns_exit
  (expr : chelis_deep.ast.Expr)
  (map : chelis_unord.UnordMap String
  chelis_types.infer.checked.DeclaredSigMetadata)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.checked.collect_defsig_param_types expr map stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.checked.collect_defsig_param_types.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "collect_defsig_param_types") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (map, stack1, back)
  else
    let (map1, stack2, collect_defsig_param_types_frame_back) ←
      chelis_types.infer.checked.collect_defsig_param_types_frame expr map
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := collect_defsig_param_types_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (map1, stack3, back)) := by
  simp only [chelis_types.infer.checked.collect_defsig_param_types, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem checked_effects_only_expr_matches_returns_exit
  (before : chelis_deep.ast.Expr) (after : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.checked.effects_only_expr_matches before after stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.checked.effects_only_expr_matches.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "effects_only_expr_matches") before
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (false, stack1, back)
  else
    let (result, stack2, effects_only_expr_matches_frame_back) ←
      chelis_types.infer.checked.effects_only_expr_matches_frame before after
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := effects_only_expr_matches_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.checked.effects_only_expr_matches, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem checked_expression_without_derived_effects_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.checked.expression_without_derived_effects expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.checked.expression_without_derived_effects.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "expression_without_derived_effects") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (Aeneas.Std.core.result.Result.Err (),
      stack1, back)
  else
    let (result, stack2, expression_without_derived_effects_frame_back) ←
      chelis_types.infer.checked.expression_without_derived_effects_frame expr
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := expression_without_derived_effects_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.checked.expression_without_derived_effects, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem checked_register_annotation_owners_returns_exit
  (expr : chelis_deep.ast.Expr)
  (epoch : chelis_types.infer.checked.TypeStampEpoch)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.checked.register_annotation_owners expr epoch stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.checked.register_annotation_owners.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "register_annotation_owners") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (epoch, stack1, back)
  else
    let (epoch1, stack2, register_annotation_owners_frame_back) ←
      chelis_types.infer.checked.register_annotation_owners_frame expr epoch
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := register_annotation_owners_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (epoch1, stack3, back)) := by
  simp only [chelis_types.infer.checked.register_annotation_owners, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem common_check_rank_body_discipline_returns_exit
  (def_name : Aeneas.Std.Str) (expr : chelis_deep.ast.Expr)
  (user_def_names : chelis_unord.UnordSet String)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.common.check_rank_body_discipline def_name expr user_def_names errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.common.check_rank_body_discipline.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "check_rank_body_discipline") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors1, back)
  else
    let (errors2, check_rank_body_discipline_frame_back) ←
      chelis_types.infer.common.check_rank_body_discipline_frame def_name expr
        user_def_names errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           check_rank_body_discipline_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors3, back)) := by
  simp only [chelis_types.infer.common.check_rank_body_discipline, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem common_collect_nominal_kind_evidence_returns_exit
  (expr : chelis_deep.ast.Expr)
  (context : Option chelis_types.types.NominalParamKind)
  (own_params : chelis_unord.UnordMap Aeneas.Std.Str
  Aeneas.Std.Usize)
  (evidence : chelis_unord.UnordMap String
  (Aeneas.Std.alloc.vec.Vec (Bool × Bool)))
  (owner : Aeneas.Std.Str)
  (existing_headers : chelis_types.deep_type.TypeResolutionEnv)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.common.collect_nominal_kind_evidence expr context own_params evidence owner existing_headers stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.common.collect_nominal_kind_evidence.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "collect_nominal_kind_evidence") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (evidence, stack1, back)
  else
    let (evidence1, stack2, collect_nominal_kind_evidence_frame_back) ←
      chelis_types.infer.common.collect_nominal_kind_evidence_frame expr
        context own_params evidence owner existing_headers stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := collect_nominal_kind_evidence_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (evidence1, stack3, back)) := by
  simp only [chelis_types.infer.common.collect_nominal_kind_evidence, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem common_is_static_numeric_bound_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.common.is_static_numeric_bound expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.common.is_static_numeric_bound.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "is_static_numeric_bound") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (false, stack1, back)
  else
    let (result, stack2, is_static_numeric_bound_frame_back) ←
      chelis_types.infer.common.is_static_numeric_bound_frame expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := is_static_numeric_bound_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.common.is_static_numeric_bound, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declarations_collect_pattern_names_for_signature_returns_exit
  (expr : chelis_deep.ast.Expr)
  (names : chelis_unord.UnordSet String)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.declarations.collect_pattern_names_for_signature expr names stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.declarations.collect_pattern_names_for_signature.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "collect_pattern_names_for_signature")
      expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (names, stack1, back)
  else
    let (names1, stack2, collect_pattern_names_for_signature_frame_back) ←
      chelis_types.infer.declarations.collect_pattern_names_for_signature_frame
        expr names stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := collect_pattern_names_for_signature_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (names1, stack3, back)) := by
  simp only [chelis_types.infer.declarations.collect_pattern_names_for_signature, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declarations_collect_top_level_references_returns_exit
  (expr : chelis_deep.ast.Expr)
  (vertex_by_name : chelis_unord.UnordMap String
  Aeneas.Std.Usize)
  (bound : Aeneas.Std.alloc.vec.Vec (chelis_unord.UnordSet
  String))
  (references : Aeneas.Std.alloc.collections.btree.set.BTreeSet
  chelis_types.infer.declarations.TopLevelReference)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.declarations.collect_top_level_references expr vertex_by_name bound references stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.declarations.collect_top_level_references.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "collect_top_level_references") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (bound, references, stack1, back)
  else
    let (bound1, references1, stack2, collect_top_level_references_frame_back)
      ←
      chelis_types.infer.declarations.collect_top_level_references_frame expr
        vertex_by_name bound references stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := collect_top_level_references_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (bound1, references1, stack3, back)) := by
  simp only [chelis_types.infer.declarations.collect_top_level_references, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declarations_expr_mentions_unshadowed_name_returns_exit
  (expr : chelis_deep.ast.Expr) (name1 : Aeneas.Std.Str)
  (bound : Aeneas.Std.alloc.vec.Vec (chelis_unord.UnordSet
  String)) (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.declarations.expr_mentions_unshadowed_name expr name1 bound stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.declarations.expr_mentions_unshadowed_name.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "expr_mentions_unshadowed_name") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (false, bound, stack1, back)
  else
    let (result, bound1, stack2, expr_mentions_unshadowed_name_frame_back) ←
      chelis_types.infer.declarations.expr_mentions_unshadowed_name_frame expr
        name1 bound stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := expr_mentions_unshadowed_name_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, bound1, stack3, back)) := by
  simp only [chelis_types.infer.declarations.expr_mentions_unshadowed_name, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declarations_param_has_consuming_use_inner_returns_exit
  (expr : chelis_deep.ast.Expr) (param : Aeneas.Std.Str)
  (bound : Aeneas.Std.alloc.vec.Vec (chelis_unord.UnordSet
  String))
  (available_signatures : chelis_unord.UnordMap String
  chelis_types.types.Type)
  (type_env : Aeneas.Std.alloc.collections.btree.map.BTreeMap
  String chelis_deep.ast.Expr)
  (type_headers : chelis_types.deep_type.TypeResolutionEnv)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.declarations.param_has_consuming_use_inner expr param bound available_signatures type_env type_headers errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.declarations.param_has_consuming_use_inner.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "param_has_consuming_use_inner") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (false, bound, errors1, back)
  else
    let (result, bound1, errors2, param_has_consuming_use_inner_frame_back) ←
      chelis_types.infer.declarations.param_has_consuming_use_inner_frame expr
        param bound available_signatures type_env type_headers errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           param_has_consuming_use_inner_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (result, bound1, errors3, back)) := by
  simp only [chelis_types.infer.declarations.param_has_consuming_use_inner, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declarations_param_name_for_refs_returns_exit
  (param : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.declarations.param_name_for_refs param stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.declarations.param_name_for_refs.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "param_name_for_refs") param
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, param_name_for_refs_frame_back) ←
      chelis_types.infer.declarations.param_name_for_refs_frame param stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := param_name_for_refs_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.declarations.param_name_for_refs, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declared_surface_resolve_declared_surface_in_session_scoped_items_returns_exit
  (expr : chelis_deep.ast.Expr) (module : Option String)
  (out : Aeneas.Std.alloc.vec.Vec ((Option String) ×
  chelis_deep.ast.Expr))
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.items expr module out stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.items.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "declared_surface_items") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (out, stack1, back)
  else
    let (out1, stack2, items_frame_back) ←
      chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.items_frame
        expr module out stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := items_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (out1, stack3, back)) := by
  simp only [chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.items, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem declared_surface_resolve_declared_surface_in_session_scoped_validate_carriers_returns_exit
  (expr : chelis_deep.ast.Expr)
  (sink : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.validate_carriers expr sink = (do
  let (b, sink1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.validate_carriers.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      sink (Aeneas.Std.toStr "validate_carriers") expr
  if b
  then
    let back :=
      (fun sink2 =>
         let ds :=
           frame_exhausted_back
             { sink1 with errors := sink2.errors, native := sink2.native }
         { sink with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (sink1, back)
  else
    let (sink2, validate_carriers_frame_back) ←
      chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.validate_carriers_frame
        expr sink1
    let (sink3, exit_frame_back) := ({ sink2 with native := { sink2.native with depth := UScalar.saturating_sub sink2.native.depth 1#usize } }, (fun self1 => { sink2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun sink4 =>
         let ds :=
           exit_frame_back
             { sink3 with errors := sink4.errors, native := sink4.native }
         let ds1 :=
           validate_carriers_frame_back
             { sink2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { sink1 with errors := ds1.errors, native := ds1.native }
         { sink with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (sink3, back)) := by
  simp only [chelis_types.infer.declared_surface.resolve_declared_surface_in_session_scoped.validate_carriers, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem expr_pattern_pattern_bindings_returns_exit
  (pat : chelis_deep.ast.Expr)
  (scrutinee_ty : chelis_types.types.Type)
  (env : chelis_types.env.Env) (vg : chelis_types.types.VarGen)
  (subst : chelis_types.unify.Subst)
  (adt_reg : chelis_types.adt.AdtRegistry)
  (errors : chelis_types.session.DiagnosticSink)
  (product : chelis_types.infer.checked.InferenceProduct)
  (covered_variants : Aeneas.Std.alloc.vec.Vec String)
  (has_wildcard : Bool) :
    chelis_types.infer.expr_pattern.pattern_bindings pat scrutinee_ty env vg subst adt_reg errors product covered_variants has_wildcard = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.expr_pattern.pattern_bindings.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "pattern_bindings") pat
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (env, vg, subst, errors1, back, product,
      covered_variants, has_wildcard)
  else
    let (env1, vg1, subst1, errors2, pattern_bindings_frame_back, product1,
      covered_variants1, has_wildcard1) ←
      chelis_types.infer.expr_pattern.pattern_bindings_frame pat scrutinee_ty
        env vg subst adt_reg errors1 product covered_variants has_wildcard
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           pattern_bindings_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (env1, vg1, subst1, errors3, back, product1,
      covered_variants1, has_wildcard1)) := by
  simp only [chelis_types.infer.expr_pattern.pattern_bindings, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem program_deep_type_contains_hole_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.program.deep_type_contains_hole expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.program.deep_type_contains_hole.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "deep_type_contains_hole") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (true, stack1, back)
  else
    let (result, stack2, deep_type_contains_hole_frame_back) ←
      chelis_types.infer.program.deep_type_contains_hole_frame expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := deep_type_contains_hole_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.program.deep_type_contains_hole, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem shape_honesty_expr_shape_type_fact_returns_exit
  (expr : chelis_deep.ast.Expr)
  (type_env : chelis_unord.UnordMap String
  chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.shape_honesty.expr_shape_type_fact expr type_env stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.shape_honesty.expr_shape_type_fact.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "expr_shape_type_fact") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, expr_shape_type_fact_frame_back) ←
      chelis_types.infer.shape_honesty.expr_shape_type_fact_frame expr type_env
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := expr_shape_type_fact_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.shape_honesty.expr_shape_type_fact, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem shape_honesty_peel_borrow_returns_exit
  (expr : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.shape_honesty.peel_borrow expr stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.shape_honesty.peel_borrow.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "peel_borrow") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (expr, stack1, back)
  else
    let (result, stack2, peel_borrow_frame_back) ←
      chelis_types.infer.shape_honesty.peel_borrow_frame expr stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := peel_borrow_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.shape_honesty.peel_borrow, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_bind_core_transform_pattern_returns_exit
  (pattern : chelis_deep.ast.Expr)
  (value : chelis_types.infer.validate.CoreTransformValue)
  (scope : chelis_types.infer.validate.CoreTransformScope)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.validate.bind_core_transform_pattern pattern value scope stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.validate.bind_core_transform_pattern.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "bind_core_transform_pattern") pattern
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (scope, stack1, back)
  else
    let (scope1, stack2, bind_core_transform_pattern_frame_back) ←
      chelis_types.infer.validate.bind_core_transform_pattern_frame pattern
        value scope stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := bind_core_transform_pattern_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (scope1, stack3, back)) := by
  simp only [chelis_types.infer.validate.bind_core_transform_pattern, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_classify_core_transform_value_returns_exit
  (value : chelis_deep.ast.Expr)
  (top_level_functions : chelis_unord.UnordSet String)
  (module_values : chelis_unord.UnordMap String
  chelis_types.infer.validate.CoreTransformValue)
  (lexical_scope : chelis_types.infer.validate.CoreTransformScope)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.validate.classify_core_transform_value value top_level_functions module_values lexical_scope stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.validate.classify_core_transform_value.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "classify_core_transform_value") value
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok
      (chelis_types.infer.validate.CoreTransformValue.Ordinary, stack1,
      back)
  else
    let (result, stack2, classify_core_transform_value_frame_back) ←
      chelis_types.infer.validate.classify_core_transform_value_frame value
        top_level_functions module_values lexical_scope stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := classify_core_transform_value_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.validate.classify_core_transform_value, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_core_transform_value_matches_pattern_returns_exit
  (value : chelis_types.infer.validate.CoreTransformValue)
  (pattern : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.validate.core_transform_value_matches_pattern value pattern stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.validate.core_transform_value_matches_pattern.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "core_transform_value_matches_pattern")
      pattern
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (false, stack1, back)
  else
    let (result, stack2, core_transform_value_matches_pattern_frame_back) ←
      chelis_types.infer.validate.core_transform_value_matches_pattern_frame
        value pattern stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := core_transform_value_matches_pattern_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.validate.core_transform_value_matches_pattern, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_expr_type_expr_returns_exit
  (expr : chelis_deep.ast.Expr)
  (type_env : Aeneas.Std.alloc.collections.btree.map.BTreeMap
  String chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.validate.expr_type_expr expr type_env stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.validate.expr_type_expr.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "expr_type_expr") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, expr_type_expr_frame_back) ←
      chelis_types.infer.validate.expr_type_expr_frame expr type_env stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := expr_type_expr_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.validate.expr_type_expr, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_reject_unknown_metadata_forms_returns_exit
  (expr : chelis_deep.ast.Expr)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.validate.reject_unknown_metadata_forms expr errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.validate.reject_unknown_metadata_forms.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "reject_unknown_metadata_forms") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors1, back)
  else
    let (errors2, reject_unknown_metadata_forms_frame_back) ←
      chelis_types.infer.validate.reject_unknown_metadata_forms_frame expr
        errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           reject_unknown_metadata_forms_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors3, back)) := by
  simp only [chelis_types.infer.validate.reject_unknown_metadata_forms, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_validate_ir_expr_returns_exit
  (expr : chelis_deep.ast.Expr)
  (type_env : chelis_unord.UnordMap String
  chelis_deep.ast.Expr)
  (static_env : chelis_unord.UnordMap String
  chelis_types.infer.static_value.StaticValue)
  (declared_signatures : chelis_unord.UnordMap String
  chelis_types.infer.checked.DeclaredSigMetadata)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.validate.validate_ir_expr expr type_env static_env declared_signatures errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.validate.validate_ir_expr.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "validate_ir_expr") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok
      (chelis_types.infer.static_value.StaticValue.Unknown, static_env,
      errors1, back)
  else
    let (result, static_env1, errors2, validate_ir_expr_frame_back) ←
      chelis_types.infer.validate.validate_ir_expr_frame expr type_env
        static_env declared_signatures errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           validate_ir_expr_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (result, static_env1, errors3, back)) := by
  simp only [chelis_types.infer.validate.validate_ir_expr, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_vmap_parameter_type_has_unmapped_hole_returns_exit
  (ty : chelis_deep.ast.Expr)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.validate.vmap_parameter_type_has_unmapped_hole ty stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.validate.vmap_parameter_type_has_unmapped_hole.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "vmap_parameter_type_has_unmapped_hole")
      ty
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (true, stack1, back)
  else
    let (result, stack2, vmap_parameter_type_has_unmapped_hole_frame_back) ←
      chelis_types.infer.validate.vmap_parameter_type_has_unmapped_hole_frame
        ty stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := vmap_parameter_type_has_unmapped_hole_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.validate.vmap_parameter_type_has_unmapped_hole, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_walk_core_transform_targets_returns_exit
  (expr : chelis_deep.ast.Expr)
  (top_level_functions : chelis_unord.UnordSet String)
  (module_values : chelis_unord.UnordMap String
  chelis_types.infer.validate.CoreTransformValue)
  (lexical_scope : chelis_types.infer.validate.CoreTransformScope)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.validate.walk_core_transform_targets expr top_level_functions module_values lexical_scope errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.validate.walk_core_transform_targets.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "walk_core_transform_targets") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors1, back)
  else
    let (errors2, walk_core_transform_targets_frame_back) ←
      chelis_types.infer.validate.walk_core_transform_targets_frame expr
        top_level_functions module_values lexical_scope errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           walk_core_transform_targets_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors3, back)) := by
  simp only [chelis_types.infer.validate.walk_core_transform_targets, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem validate_walk_for_tensor_precision_returns_exit
  (expr : chelis_deep.ast.Expr)
  (errors : chelis_types.session.DiagnosticSink)
  (seen : chelis_unord.UnordSet
  (chelis_types.infer.validate.TensorPrecisionOwner × String))
  (owner : chelis_types.infer.validate.TensorPrecisionOwner) :
    chelis_types.infer.validate.walk_for_tensor_precision expr errors seen owner = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.validate.walk_for_tensor_precision.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr
      "validate_tensor_precisions (walk_for_tensor_precision)") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors1, back, seen)
  else
    let (errors2, walk_for_tensor_precision_frame_back, seen1) ←
      chelis_types.infer.validate.walk_for_tensor_precision_frame expr errors1
        seen owner
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           walk_for_tensor_precision_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors3, back, seen1)) := by
  simp only [chelis_types.infer.validate.walk_for_tensor_precision, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem vmap_extent_analyze_expr_returns_exit
  (expr : chelis_deep.ast.Expr)
  (locals : chelis_unord.UnordMap String
  (Aeneas.Std.alloc.collections.btree.set.BTreeSet
  Aeneas.Std.Usize))
  (summaries : Aeneas.Std.alloc.collections.btree.map.BTreeMap
  String chelis_types.infer.vmap_extent.FunctionSummary)
  (movement_deps : Aeneas.Std.alloc.collections.btree.map.BTreeMap
  Aeneas.Std.Usize
  chelis_types.infer.vmap_extent.MovementWitness)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.vmap_extent.analyze_expr expr locals summaries movement_deps stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.vmap_extent.analyze_expr.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "analyze_vmap_extent_dependencies") expr
  if b
  then
    let result ←
      Aeneas.Std.alloc.collections.btree.set.BTreeSet.new
        Aeneas.Std.Usize
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, locals, movement_deps, stack1, back)
  else
    let (result, locals1, movement_deps1, stack2, analyze_expr_frame_back) ←
      chelis_types.infer.vmap_extent.analyze_expr_frame expr locals summaries
        movement_deps stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := analyze_expr_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, locals1, movement_deps1, stack3, back)) := by
  simp only [chelis_types.infer.vmap_extent.analyze_expr, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem vmap_extent_transformed_callee_returns_exit
  (expr : chelis_deep.ast.Expr)
  (defs : Aeneas.Std.alloc.collections.btree.map.BTreeMap String
  chelis_types.infer.vmap_extent.FunctionDef)
  (summaries : Aeneas.Std.alloc.collections.btree.map.BTreeMap
  String chelis_types.infer.vmap_extent.FunctionSummary)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.vmap_extent.transformed_callee expr defs summaries stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.vmap_extent.transformed_callee.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "transformed_vmap_extent_callee") expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, transformed_callee_frame_back) ←
      chelis_types.infer.vmap_extent.transformed_callee_frame expr defs
        summaries stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := transformed_callee_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.vmap_extent.transformed_callee, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem vmap_extent_transformed_function_def_returns_exit
  (expr : chelis_deep.ast.Expr)
  (defs : Aeneas.Std.alloc.collections.btree.map.BTreeMap String
  chelis_types.infer.vmap_extent.FunctionDef)
  (stack : chelis_types.infer.stack.StackBudget) :
    chelis_types.infer.vmap_extent.transformed_function_def expr defs stack = (do
  let (b, stack1, frame_exhausted_back) ←
    chelis_types.infer.stack.StackBudget.frame_exhausted
      chelis_types.infer.vmap_extent.transformed_function_def.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      stack (Aeneas.Std.toStr "transformed_vmap_extent_function_def")
      expr
  if b
  then
    let back :=
      (fun stack2 => frame_exhausted_back stack2 :
        chelis_types.infer.stack.StackBudget →
        chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (none, stack1, back)
  else
    let (result, stack2, transformed_function_def_frame_back) ←
      chelis_types.infer.vmap_extent.transformed_function_def_frame expr defs
        stack1
    let (stack3, exit_frame_back) := ({ stack2 with native := { stack2.native with depth := UScalar.saturating_sub stack2.native.depth 1#usize } }, id)
    let back :=
      (fun stack4 =>
         let sb := exit_frame_back stack4
         let sb1 := transformed_function_def_frame_back sb
         frame_exhausted_back sb1 : chelis_types.infer.stack.StackBudget
        → chelis_types.infer.stack.StackBudget)
    Aeneas.Std.Result.ok (result, stack3, back)) := by
  simp only [chelis_types.infer.vmap_extent.transformed_function_def, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

/-- All return values retain the explicit frame exit. -/
theorem vmap_extent_walk_vmap_sites_returns_exit
  (expr : chelis_deep.ast.Expr)
  (defs : Aeneas.Std.alloc.collections.btree.map.BTreeMap String
  chelis_types.infer.vmap_extent.FunctionDef)
  (summaries : Aeneas.Std.alloc.collections.btree.map.BTreeMap
  String chelis_types.infer.vmap_extent.FunctionSummary)
  (errors : chelis_types.session.DiagnosticSink) :
    chelis_types.infer.vmap_extent.walk_vmap_sites expr defs summaries errors = (do
  let (b, errors1, frame_exhausted_back) ←
    chelis_types.session.DiagnosticSink.frame_exhausted
      chelis_types.infer.vmap_extent.walk_vmap_sites.closure.Insts.CoreOpsFunctionFnOnceTupleOptionSharedStr
      errors (Aeneas.Std.toStr "walk_vmap_extent_sites") expr
  if b
  then
    let back :=
      (fun errors2 =>
         let ds :=
           frame_exhausted_back
             { errors1 with errors := errors2.errors, native := errors2.native
             }
         { errors with errors := ds.errors, native := ds.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors1, back)
  else
    let (errors2, walk_vmap_sites_frame_back) ←
      chelis_types.infer.vmap_extent.walk_vmap_sites_frame expr defs summaries
        errors1
    let (errors3, exit_frame_back) := ({ errors2 with native := { errors2.native with depth := UScalar.saturating_sub errors2.native.depth 1#usize } }, (fun self1 => { errors2 with errors := self1.errors, native := self1.native }))
    let back :=
      (fun errors4 =>
         let ds :=
           exit_frame_back
             { errors3 with errors := errors4.errors, native := errors4.native
             }
         let ds1 :=
           walk_vmap_sites_frame_back
             { errors2 with errors := ds.errors, native := ds.native }
         let ds2 :=
           frame_exhausted_back
             { errors1 with errors := ds1.errors, native := ds1.native }
         { errors with errors := ds2.errors, native := ds2.native } :
        chelis_types.session.DiagnosticSink →
        chelis_types.session.DiagnosticSink)
    Aeneas.Std.Result.ok (errors3, back)) := by
  simp only [chelis_types.infer.vmap_extent.walk_vmap_sites, chelis_types.infer.stack.StackBudget.exit_frame,
    chelis_types.session.DiagnosticSink.exit_frame,
    Aeneas.Std.chelis_types.native.NativeServices.exit_frame, Result.bind]

end ChelisBridge.Runtime.ExtractedStack
