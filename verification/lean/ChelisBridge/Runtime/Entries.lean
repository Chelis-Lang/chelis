import ChelisBridge.Runtime.ExtractedUnwind
import ChelisBridge.Runtime.ExtractedHostScopes

/-!
This relation preserves successful inner results and their explicit states.
It distinguishes ordinary errors from typed boundary unwinds.
The extracted continuation resumes the exact payload of the typed failure.

An arbitrary Lean failure or divergence supplies no native outcome.
Unexpected unwinds and aborts need separate native execution evidence.
The public-entry inventory names the caller correspondence requirements.
-/

open Aeneas.Std
open_chelis_model

namespace ChelisBridge.Runtime.Entries

abbrev Failure := chelis_ir.lowering_runtime.LoweringFailure

abbrev Diagnostic := chelis_ir.lower.LowerDiagnostic

abbrev Runtime := Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime

inductive NativeOutcome (Value Diagnostic : Type) where
  | success (value : Value)
  | ordinaryError (diagnostic : Diagnostic)
  | typedUnwind (failure : Failure)
  | unexpectedUnwind (payload : ChelisBridge.Runtime.Unwind.Payload)
  | abort

def continueAtBoundary {Value Diagnostic : Type}
    (inner : core.result.Result (core.result.Result Value Diagnostic) Failure) :
    Result (core.result.Result Value Diagnostic) :=
  match inner with
  | .Ok result => .ok result
  | .Err failure => do
      let never ← chelis_ir.lowering_runtime.LoweringFailure.resume_at_boundary failure
      nomatch never

inductive BoundaryRelation {Value Diagnostic State : Type} :
    Result (core.result.Result (core.result.Result Value Diagnostic) Failure × State) →
    NativeOutcome Value Diagnostic → State → Prop where
  | success (value : Value) (state : State) :
      BoundaryRelation (.ok (.Ok (.Ok value), state)) (.success value) state
  | ordinaryError (diagnostic : Diagnostic) (state : State) :
      BoundaryRelation (.ok (.Ok (.Err diagnostic), state)) (.ordinaryError diagnostic) state
  | typedUnwind (failure : Failure) (state : State) :
      BoundaryRelation (.ok (.Err failure, state)) (.typedUnwind failure) state

theorem continue_success {Value Diagnostic : Type} (value : Value) :
    continueAtBoundary (Diagnostic := Diagnostic) (.Ok (.Ok value)) = .ok (.Ok value) :=
  rfl

theorem continue_ordinary_error {Value Diagnostic : Type} (diagnostic : Diagnostic) :
    continueAtBoundary (Value := Value) (.Ok (.Err diagnostic)) = .ok (.Err diagnostic) :=
  rfl

theorem continue_typed_unwind {Value Diagnostic : Type} (failure : Failure) :
    continueAtBoundary (Value := Value) (Diagnostic := Diagnostic) (.Err failure) =
      .fail .panic := by
  unfold continueAtBoundary
  rw [ChelisBridge.Runtime.ExtractedScopes.resumeAtBoundary_never_ok]
  rfl

theorem boundary_no_claim_for_lean_failure {Value Diagnostic State : Type}
    (error : Error) (native : NativeOutcome Value Diagnostic) (state : State) :
    ¬ BoundaryRelation (.fail error) native state := by
  intro h
  cases h

theorem continue_typed_payload {Value Diagnostic : Type} (failure : Failure) :
    continueAtBoundary (Value := Value) (Diagnostic := Diagnostic) (.Err failure) =
      (do
        let never ← Aeneas.Std.chelis_ir.lowering_runtime.resume_native_unwind
          (ChelisBridge.Runtime.ExtractedScopes.failurePayload failure).2.1
          (ChelisBridge.Runtime.ExtractedScopes.failurePayload failure).2.2
        nomatch never) := by
  cases failure <;> rfl

theorem boundary_no_claim_for_divergence {Value Diagnostic State : Type}
    (native : NativeOutcome Value Diagnostic) (state : State) :
    ¬ BoundaryRelation .div native state := by
  intro h
  cases h

theorem boundary_no_unexpected_claim {Value Diagnostic State : Type}
    (inner : Result (core.result.Result (core.result.Result Value Diagnostic) Failure × State))
    (payload : ChelisBridge.Runtime.Unwind.Payload) (state : State) :
    ¬ BoundaryRelation inner (.unexpectedUnwind payload) state := by
  intro h
  cases h

theorem boundary_no_abort_claim {Value Diagnostic State : Type}
    (inner : Result (core.result.Result (core.result.Result Value Diagnostic) Failure × State))
    (state : State) :
    ¬ BoundaryRelation inner (.abort : NativeOutcome Value Diagnostic) state := by
  intro h
  cases h

theorem boundary_unique {Value Diagnostic State : Type}
    (inner : Result (core.result.Result (core.result.Result Value Diagnostic) Failure × State))
    (left right : NativeOutcome Value Diagnostic) (leftState rightState : State)
    (hl : BoundaryRelation inner left leftState)
    (hr : BoundaryRelation inner right rightState) :
    left = right ∧ leftState = rightState := by
  cases hl
  · cases hr
    exact ⟨rfl, rfl⟩
  · cases hr
    exact ⟨rfl, rfl⟩
  · cases hr
    exact ⟨rfl, rfl⟩

def liftResumeResult {Value Diagnostic State : Type}
    (inner : Result (core.result.Result Value Failure × State)) :
    Result (core.result.Result (core.result.Result Value Diagnostic) Failure × State) :=
  match inner with
  | .ok (.Ok value, state) => .ok (.Ok (.Ok value), state)
  | .ok (.Err failure, state) => .ok (.Err failure, state)
  | .fail error => .fail error
  | .div => .div

theorem liftResumeResult_success {Value Diagnostic State : Type}
    (value : Value) (state : State) :
    liftResumeResult (Diagnostic := Diagnostic) (.ok (.Ok value, state)) =
      .ok (.Ok (.Ok value), state) :=
  rfl

theorem liftResumeResult_typed_unwind {Value Diagnostic State : Type}
    (failure : Failure) (state : State) :
    liftResumeResult (Value := Value) (Diagnostic := Diagnostic) (.ok (.Err failure, state)) =
      .ok (.Err failure, state) :=
  rfl

theorem liftResumeResult_failure {Value Diagnostic State : Type} (error : Error) :
    liftResumeResult (Value := Value) (Diagnostic := Diagnostic) (State := State)
      (.fail error) = .fail error :=
  rfl

theorem liftResumeResult_divergence {Value Diagnostic State : Type} :
    liftResumeResult (Value := Value) (Diagnostic := Diagnostic) (State := State) .div = .div :=
  rfl

theorem liftResumeResult_success_iff {Value Diagnostic State : Type}
    (inner : Result (core.result.Result Value Failure × State))
    (value : Value) (state : State) :
    BoundaryRelation (liftResumeResult inner) (.success value : NativeOutcome Value Diagnostic)
      state ↔ inner = .ok (.Ok value, state) := by
  constructor
  · intro h
    cases inner with
    | ok pair =>
        obtain ⟨result, returned⟩ := pair
        cases result <;> cases h
        rfl
    | fail error => cases h
    | div => cases h
  · intro h
    rw [h]
    exact .success value state

theorem liftResumeResult_typed_unwind_iff {Value Diagnostic State : Type}
    (inner : Result (core.result.Result Value Failure × State))
    (failure : Failure) (state : State) :
    BoundaryRelation (liftResumeResult inner)
      (.typedUnwind failure : NativeOutcome Value Diagnostic) state ↔
      inner = .ok (.Err failure, state) := by
  constructor
  · intro h
    cases inner with
    | ok pair =>
        obtain ⟨result, returned⟩ := pair
        cases result <;> cases h
        rfl
    | fail error => cases h
    | div => cases h
  · intro h
    rw [h]
    exact .typedUnwind failure state

theorem liftResumeResult_state_preservation {Value Diagnostic State : Type}
    (inner : Result (core.result.Result Value Failure × State))
    (native : NativeOutcome Value Diagnostic) (state : State)
    (h : BoundaryRelation (liftResumeResult inner) native state) :
    ∃ result, inner = .ok (result, state) := by
  cases inner with
  | ok pair =>
      obtain ⟨result, returned⟩ := pair
      cases result <;> cases h
      · exact ⟨_, rfl⟩
      · exact ⟨_, rfl⟩
  | fail error => cases h
  | div => cases h

theorem liftResumeResult_no_ordinary_error {Value Diagnostic State : Type}
    (inner : Result (core.result.Result Value Failure × State))
    (diagnostic : Diagnostic) (state : State) :
    ¬ BoundaryRelation (liftResumeResult inner) (.ordinaryError diagnostic) state := by
  intro h
  cases inner with
  | ok pair =>
      obtain ⟨result, returned⟩ := pair
      cases result <;> cases h
  | fail error => cases h
  | div => cases h

/-!
These equations specify lower.rs:89-100 and span.rs:18-20.
They do not identify an extracted Display body or establish caller correspondence.
Default format arguments select semantic num.DisplayUsize.fmt.
The source evaluates Span.end only after the message write succeeds.
-/

def spanEndSource (offset length : Usize) : Result Usize :=

  UScalar.add offset length

def DisplayArithmeticDomain (spanId : Option String) (span : Option (Usize × Usize)) : Prop :=

  match spanId, span with

  | none, some (offset, length) =>

      length.val = 0 ∨ offset.val + length.val ≤ Usize.max

  | _, _ => True

def displayArgumentSource (callback : core.fmt.Callback) (fmt : core.fmt.Formatter) :

    core.fmt.Outcome := do

  let (status, after) ← callback { fmt with options := {} }

  .ok (status, { after with options := fmt.options })

def diagnosticSuffixSource (spanId : Option String) (span : Option (Usize × Usize))

    (fmt : core.fmt.Formatter) : core.fmt.Outcome :=

  match spanId with

  | some id =>

      core.fmt.thenFormat (fmt.append " at source span `") (fun fmt =>

        core.fmt.thenFormat (displayArgumentSource (core.fmt.DisplayString.fmt id) fmt)

          (fun fmt => fmt.append "`"))

  | none =>

      match span with

      | none => .ok (.Ok (), fmt)

      | some (offset, length) =>

          if length.val = 0 then .ok (.Ok (), fmt)

          else do

            let stop ← spanEndSource offset length

            core.fmt.thenFormat (fmt.append " at byte ") (fun fmt =>

              core.fmt.thenFormat

                (displayArgumentSource (core.fmt.num.DisplayUsize.fmt offset) fmt) (fun fmt =>

                core.fmt.thenFormat (fmt.append "..") (fun fmt =>

                  displayArgumentSource (core.fmt.num.DisplayUsize.fmt stop) fmt)))

def diagnosticDisplaySource (message : String) (spanId : Option String)

    (span : Option (Usize × Usize)) (fmt : core.fmt.Formatter) : core.fmt.Outcome :=

  core.fmt.thenFormat (displayArgumentSource (core.fmt.DisplayString.fmt message) fmt)

    (diagnosticSuffixSource spanId span)

def diagnosticTextSource (message : String) (spanId : Option String)

    (span : Option (Usize × Usize)) : Result String :=

  match spanId with

  | some id => .ok (message ++ " at source span `" ++ id ++ "`")

  | none =>

      match span with

      | none => .ok message

      | some (offset, length) =>

          if length.val = 0 then .ok message

          else do

            let stop ← spanEndSource offset length

            .ok (message ++ " at byte " ++ core.fmt.digits offset.val ++ ".." ++

              core.fmt.digits stop.val)

theorem spanEndSource_value (offset length stop : Usize)

    (h : spanEndSource offset length = .ok stop) :

    stop.val = offset.val + length.val := by

  have ha := UScalar.tryMk_eq .Usize (offset.val + length.val)

  rw [show UScalar.tryMk .Usize (offset.val + length.val) = .ok stop from h] at ha

  exact ha.1

theorem spanEndSource_exists (offset length : Usize)

    (h : offset.val + length.val ≤ Usize.max) :

    ∃ stop, spanEndSource offset length = .ok stop ∧

      stop.val = offset.val + length.val := by

  have hb : UScalar.inBounds .Usize (offset.val + length.val) := by

    simp only [UScalar.inBounds]

    simp only [Usize.max, Usize.numBits] at h

    have hp : 0 < 2 ^ UScalarTy.Usize.numBits := by simp

    omega

  have ha := UScalar.tryMk_eq .Usize (offset.val + length.val)

  cases he : UScalar.tryMk .Usize (offset.val + length.val) with

  | ok stop =>

      rw [he] at ha

      exact ⟨stop, he, ha.1⟩

  | fail error =>

      rw [he] at ha

      exact False.elim (ha hb)

  | div =>

      rw [he] at ha

      exact False.elim ha

theorem spanEndSource_overflow (offset length : Usize)

    (h : Usize.max < offset.val + length.val) :

    spanEndSource offset length = .fail .integerOverflow := by

  have hb : ¬ offset.val + length.val < 2 ^ System.Platform.numBits := by

    simp only [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq] at h

    omega

  simp [spanEndSource, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, hb,

    Result.ofOption]

/-- Successful Span.end values exist exactly on the nonoverflow domain. -/

theorem spanEndSource_success_iff (offset length : Usize) :

    (∃ stop, spanEndSource offset length = .ok stop) ↔

      offset.val + length.val ≤ Usize.max := by

  constructor

  · rintro ⟨stop, hstop⟩

    have hv := spanEndSource_value offset length stop hstop

    have hb := stop.hBounds

    rw [hv] at hb

    simp only [Usize.max, Usize.numBits]

    omega

  · intro h

    obtain ⟨stop, hstop, _⟩ := spanEndSource_exists offset length h

    exact ⟨stop, hstop⟩

theorem displayArgumentSource_error (callback : core.fmt.Callback)

    (before after : core.fmt.Formatter) (error : core.fmt.Error)

    (h : callback { before with options := {} } = .ok (.Err error, after)) :

    displayArgumentSource callback before =

      .ok (.Err error, { after with options := before.options }) := by

  simp [displayArgumentSource, h]

/-- Default string formatting uses the writer once and restores outer options. -/

theorem displayArgumentSource_string (message : String) (fmt : core.fmt.Formatter) :

    displayArgumentSource (core.fmt.DisplayString.fmt message) fmt = fmt.append message := by

  simp [displayArgumentSource, core.fmt.DisplayString.fmt, core.fmt.displayString,

    core.fmt.Formatter.pad, core.fmt.Formatter.padWith, core.fmt.Options.bit,

    core.fmt.writeFill, core.fmt.writeRepeated, core.fmt.thenFormat, core.fmt.Formatter.append]

  intro status output _

  cases status with

  | Ok value =>

    cases value

    rfl

  | Err error => rfl

/-- Default usize formatting selects decimal digits without sign or padding. -/

theorem displayArgumentSource_usize (value : Usize) (fmt : core.fmt.Formatter) :

    displayArgumentSource (core.fmt.num.DisplayUsize.fmt value) fmt =

      fmt.append (core.fmt.digits value.val) := by

  simp [displayArgumentSource, core.fmt.num.DisplayUsize.fmt,

    core.fmt.Formatter.padIntegral, core.fmt.Options.bit,

    core.fmt.thenFormat, core.fmt.Formatter.append]

theorem diagnosticDisplaySource_message_error (message : String) (spanId : Option String)

    (span : Option (Usize × Usize)) (before after : core.fmt.Formatter)

    (error : core.fmt.Error)

    (h : displayArgumentSource (core.fmt.DisplayString.fmt message) before =

      .ok (.Err error, after)) :

    diagnosticDisplaySource message spanId span before = .ok (.Err error, after) := by

  simp [diagnosticDisplaySource, core.fmt.thenFormat, h]

theorem diagnosticDisplaySource_message_failure (message : String) (spanId : Option String)

    (span : Option (Usize × Usize)) (before : core.fmt.Formatter) (error : Error)

    (h : displayArgumentSource (core.fmt.DisplayString.fmt message) before = .fail error) :

    diagnosticDisplaySource message spanId span before = .fail error := by

  simp [diagnosticDisplaySource, core.fmt.thenFormat, h]

theorem diagnosticDisplaySource_message_divergence (message : String)

    (spanId : Option String) (span : Option (Usize × Usize)) (before : core.fmt.Formatter)

    (h : displayArgumentSource (core.fmt.DisplayString.fmt message) before = .div) :

    diagnosticDisplaySource message spanId span before = .div := by

  simp [diagnosticDisplaySource, core.fmt.thenFormat, h]

theorem diagnosticSuffixSource_span_id (id : String) (span : Option (Usize × Usize))

    (fmt : core.fmt.Formatter) :

    diagnosticSuffixSource (some id) span fmt =

      core.fmt.thenFormat (fmt.append " at source span `") (fun fmt =>

        core.fmt.thenFormat (displayArgumentSource (core.fmt.DisplayString.fmt id) fmt)

          (fun fmt => fmt.append "`")) :=

  rfl

theorem diagnosticSuffixSource_overflow (offset length : Usize)

    (fmt : core.fmt.Formatter) (hlen : length.val ≠ 0)

    (hover : Usize.max < offset.val + length.val) :

    diagnosticSuffixSource none (some (offset, length)) fmt = .fail .integerOverflow := by

  simp [diagnosticSuffixSource, hlen, spanEndSource_overflow offset length hover]

/-- A source-span prefix error prevents the ID and closing quote writes. -/

theorem diagnosticSuffixSource_id_prefix_error (id : String)

    (span : Option (Usize × Usize)) (before after : core.fmt.Formatter)

    (error : core.fmt.Error)

    (h : before.append " at source span `" = .ok (.Err error, after)) :

    diagnosticSuffixSource (some id) span before = .ok (.Err error, after) := by

  simp [diagnosticSuffixSource, core.fmt.thenFormat, h]

/-- A byte-prefix error occurs after Span.end and prevents all decimal writes. -/

theorem diagnosticSuffixSource_byte_prefix_error (offset length stop : Usize)

    (before after : core.fmt.Formatter) (error : core.fmt.Error)

    (hlen : length.val ≠ 0) (hend : spanEndSource offset length = .ok stop)

    (h : before.append " at byte " = .ok (.Err error, after)) :

    diagnosticSuffixSource none (some (offset, length)) before =

      .ok (.Err error, after) := by

  simp [diagnosticSuffixSource, hlen, hend, core.fmt.thenFormat, h]

/-- An offset formatter error prevents the separator and end formatter calls. -/

theorem diagnosticSuffixSource_offset_error (offset length stop : Usize)

    (before middle after : core.fmt.Formatter) (error : core.fmt.Error)

    (hlen : length.val ≠ 0) (hend : spanEndSource offset length = .ok stop)

    (hprefix : before.append " at byte " = .ok (.Ok (), middle))

    (hoffset : displayArgumentSource (core.fmt.num.DisplayUsize.fmt offset) middle =

      .ok (.Err error, after)) :

    diagnosticSuffixSource none (some (offset, length)) before =

      .ok (.Err error, after) := by

  simp [diagnosticSuffixSource, hlen, hend, core.fmt.thenFormat, hprefix, hoffset]

theorem diagnosticTextSource_span_id (message id : String)

    (span : Option (Usize × Usize)) :

    diagnosticTextSource message (some id) span =

      .ok (message ++ " at source span `" ++ id ++ "`") :=

  rfl

theorem diagnosticTextSource_no_span (message : String) :

    diagnosticTextSource message none none = .ok message :=

  rfl

theorem diagnosticTextSource_empty_span (message : String) (offset length : Usize)

    (h : length.val = 0) :

    diagnosticTextSource message none (some (offset, length)) = .ok message := by

  simp [diagnosticTextSource, h]

theorem diagnosticTextSource_byte_span (message : String) (offset length stop : Usize)

    (hlen : length.val ≠ 0) (hend : spanEndSource offset length = .ok stop) :

    diagnosticTextSource message none (some (offset, length)) =

      .ok (message ++ " at byte " ++ core.fmt.digits offset.val ++ ".." ++

        core.fmt.digits (offset.val + length.val)) := by

  simp [diagnosticTextSource, hlen, hend, spanEndSource_value offset length stop hend]

theorem diagnosticTextSource_overflow (message : String) (offset length : Usize)

    (hlen : length.val ≠ 0) (hover : Usize.max < offset.val + length.val) :

    diagnosticTextSource message none (some (offset, length)) = .fail .integerOverflow := by

  simp [diagnosticTextSource, hlen, spanEndSource_overflow offset length hover]

/-- The plain String writer preserves formatter errors and extraction failures. -/

def diagnosticOutputSource (message : String) (spanId : Option String)

    (span : Option (Usize × Usize)) :

    Result (core.result.Result String core.fmt.Error) := do

  let (status, fmt) ← diagnosticDisplaySource message spanId span {}

  .ok (match status with

    | .Ok () => .Ok fmt.output

    | .Err error => .Err error)

/-- The plain writer appends one text without any indent transformation. -/

theorem formatterAppend_plain (output text : String)

    (options : core.fmt.Options) (lineStart : Bool) :

    ({ output, options, lineStart } : core.fmt.Formatter).append text =

      .ok (.Ok (),

        ({ output := output ++ text,

           options := options,

           lineStart := if text.isEmpty then lineStart else text.endsWith "\n" } :

            core.fmt.Formatter)) :=

  rfl

/-- The concrete String writer yields the exact source text or arithmetic failure. -/

theorem diagnosticOutputSource_eq_text (message : String) (spanId : Option String)

    (span : Option (Usize × Usize)) :

    diagnosticOutputSource message spanId span =

      (do

        let text ← diagnosticTextSource message spanId span

        .ok (.Ok text)) := by

  cases spanId with

  | some id =>

    simp only [diagnosticOutputSource, diagnosticDisplaySource, diagnosticSuffixSource,

      diagnosticTextSource, displayArgumentSource_string, formatterAppend_plain,

      core.fmt.thenFormat, bind_tc_ok, String.empty_append]

    rfl

  | none =>

    cases span with

    | none =>

      simp only [diagnosticOutputSource, diagnosticDisplaySource, diagnosticSuffixSource,

        diagnosticTextSource, displayArgumentSource_string, formatterAppend_plain,

        core.fmt.thenFormat, bind_tc_ok, String.empty_append]

      rfl

    | some span =>

      obtain ⟨offset, length⟩ := span

      by_cases hlen : length.val = 0

      · simp only [diagnosticOutputSource, diagnosticDisplaySource, diagnosticSuffixSource,

          diagnosticTextSource, hlen, if_pos, displayArgumentSource_string,

          formatterAppend_plain, core.fmt.thenFormat, bind_tc_ok, String.empty_append]

        rfl

      · cases hend : spanEndSource offset length with

        | ok stop =>

          simp only [diagnosticOutputSource, diagnosticDisplaySource, diagnosticSuffixSource,

            diagnosticTextSource, hlen, if_false, hend, displayArgumentSource_string,

            displayArgumentSource_usize, formatterAppend_plain,

            core.fmt.thenFormat, bind_tc_ok, String.empty_append]

          rfl

        | fail error =>

          simp only [diagnosticOutputSource, diagnosticDisplaySource, diagnosticSuffixSource,

            diagnosticTextSource, hlen, if_false, hend, displayArgumentSource_string,

            formatterAppend_plain, core.fmt.thenFormat, bind_tc_ok, bind_tc_fail,

            String.empty_append]

        | div =>

          simp only [diagnosticOutputSource, diagnosticDisplaySource, diagnosticSuffixSource,

            diagnosticTextSource, hlen, if_false, hend, displayArgumentSource_string,

            formatterAppend_plain, core.fmt.thenFormat, bind_tc_ok, bind_tc_div,

            String.empty_append]

theorem diagnosticOutputSource_text_iff (message : String) (spanId : Option String)
    (span : Option (Usize × Usize)) (text : String) :
    diagnosticOutputSource message spanId span = .ok (.Ok text) ↔
      diagnosticTextSource message spanId span = .ok text := by
  rw [diagnosticOutputSource_eq_text]
  cases h : diagnosticTextSource message spanId span <;> simp

theorem diagnosticOutputSource_valid_text (message : String) (spanId : Option String)
    (span : Option (Usize × Usize)) (valid : DisplayArithmeticDomain spanId span) :
    ∃ text, diagnosticTextSource message spanId span = .ok text ∧
      diagnosticOutputSource message spanId span = .ok (.Ok text) := by
  suffices h : ∃ text, diagnosticTextSource message spanId span = .ok text by
    obtain ⟨text, htext⟩ := h
    exact ⟨text, htext, (diagnosticOutputSource_text_iff message spanId span text).mpr htext⟩
  cases spanId with
  | some id => exact ⟨_, rfl⟩
  | none =>
      cases span with
      | none => exact ⟨_, rfl⟩
      | some pair =>
          obtain ⟨offset, length⟩ := pair
          by_cases hlen : length.val = 0
          · exact ⟨message, diagnosticTextSource_empty_span message offset length hlen⟩
          · have bound : offset.val + length.val ≤ Usize.max := by
              simpa only [DisplayArithmeticDomain, hlen, false_or] using valid
            obtain ⟨stop, hstop, _⟩ := spanEndSource_exists offset length bound
            exact ⟨_, diagnosticTextSource_byte_span message offset length stop hlen hstop⟩

/-!
Catch bodies return one lowering result, not an ordinary diagnostic result.
The catch converts a typed failure and clears quiet output.
The explicit state remains the state that the body returns.
-/

inductive CatchBoundaryRelation {Value State : Type} :

    Result (core.result.Result Value Failure × Runtime × State) →

    NativeOutcome Value Diagnostic → Runtime → State → Prop where

  | success (value : Value) (runtime : Runtime) (state : State) :

      CatchBoundaryRelation (.ok (.Ok value, runtime, state)) (.success value)

        { runtime with o := false } state

  | rejection (failure : Failure) (diagnostic : Diagnostic)

      (runtime : Runtime) (state : State)

      (converted : chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic failure =

        .ok diagnostic) :

      CatchBoundaryRelation (.ok (.Err failure, runtime, state)) (.ordinaryError diagnostic)

        { runtime with o := false } state

/-!
The panic wrapper projects a successful composed result to its DAG.
On rejection, it clears quiet output and raises a new String panic.

That panic runs a new hook. It does not resume the earlier payload.
The renderer needs actual Rust Display correspondence in each caller proof.

This relation does not supply that correspondence or an actual caller theorem.
-/

inductive PanicBoundaryRelation {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String) :

    Result (core.result.Result Raw Failure × Runtime × State) →

    NativeOutcome Value Diagnostic → Runtime → State → Prop where

  | success (value : Raw) (runtime : Runtime) (state : State) :

      PanicBoundaryRelation project render (.ok (.Ok value, runtime, state))

        (.success (project value)) { runtime with o := false } state

  | rejection (failure : Failure) (diagnostic : Diagnostic)

      (runtime : Runtime) (state : State)

      (converted : chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic failure =

        .ok diagnostic) :

      PanicBoundaryRelation project render (.ok (.Err failure, runtime, state))

        (.unexpectedUnwind (.str (render diagnostic)))

        { runtime with o := false,

          hook := runtime.hook ++ [(runtime.u, false, runtime.installed)] } state

/-- Actual catch execution establishes the boundary relation for both Rust result cases. -/

theorem catchLoweringWith_boundary {State Value Body : Type}

    (inst : core.ops.function.FnOnceMutArgMutArg Body Runtime State

      (core.result.Result Value Failure))

    (before after : Runtime) (state returned : State) (body : Body)

    (result : core.result.Result Value Diagnostic)

    (h : chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering_with

      inst before state body = .ok (result, after, returned)) :

    CatchBoundaryRelation

      (inst.call_once body ({ before with o := true, installed := true }, state))

      (match result with

        | .Ok value => .success value

        | .Err diagnostic => .ordinaryError diagnostic)

      after returned := by

  rcases ChelisBridge.Runtime.ExtractedScopes.catchLoweringWith_spec

    inst before after state returned body result h with

      ⟨inner, runtime, hb, ha, _, hok, herr⟩

  cases result with

  | Ok value =>

    have hi := (hok value).mp rfl

    rw [hi] at hb

    rw [hb, ha]

    exact .success value runtime returned

  | Err diagnostic =>

    rcases (herr diagnostic).mp rfl with ⟨failure, hi, converted⟩

    rw [hi] at hb

    rw [hb, ha]

    exact .rejection failure diagnostic runtime returned converted

theorem catch_boundary_clears_output {Value State : Type}

    {inner : Result (core.result.Result Value Failure × Runtime × State)}

    {native : NativeOutcome Value Diagnostic} {runtime : Runtime} {state : State}

    (h : CatchBoundaryRelation inner native runtime state) :

    runtime.o = false := by

  cases h

  all_goals rfl

theorem catch_boundary_preserves_state {Value State : Type}

    {result : core.result.Result Value Failure} {before after : Runtime}

    {state state' : State} {native : NativeOutcome Value Diagnostic}

    (h : CatchBoundaryRelation (.ok (result, before, state)) native after state') :

    state' = state ∧ after = { before with o := false } := by

  cases h

  all_goals exact ⟨rfl, rfl⟩

theorem catch_boundary_diagnostic (diagnostic : Diagnostic) (runtime : Runtime)

    {Value State : Type} (state : State) :

    CatchBoundaryRelation (Value := Value)

      (.ok (.Err (.Diagnostic diagnostic), runtime, state)) (.ordinaryError diagnostic)

      { runtime with o := false } state := by

  exact .rejection _ _ _ _ (ChelisBridge.Runtime.ExtractedScopes.intoDiagnostic_diagnostic diagnostic)

theorem catch_boundary_no_lean_failure {Value State : Type}

    (error : Error) (native : NativeOutcome Value Diagnostic)

    (runtime : Runtime) (state : State) :

    ¬ CatchBoundaryRelation (.fail error) native runtime state := by

  intro h

  cases h

theorem catch_boundary_no_divergence {Value State : Type}

    (native : NativeOutcome Value Diagnostic) (runtime : Runtime) (state : State) :

    ¬ CatchBoundaryRelation .div native runtime state := by

  intro h

  cases h

theorem catch_boundary_no_typed_unwind {Value State : Type}

    (inner : Result (core.result.Result Value Failure × Runtime × State))

    (failure : Failure) (runtime : Runtime) (state : State) :

    ¬ CatchBoundaryRelation inner (.typedUnwind failure) runtime state := by

  intro h

  cases h

theorem catch_boundary_no_unexpected_claim {Value State : Type}

    (inner : Result (core.result.Result Value Failure × Runtime × State))

    (payload : ChelisBridge.Runtime.Unwind.Payload) (runtime : Runtime) (state : State) :

    ¬ CatchBoundaryRelation inner (.unexpectedUnwind payload) runtime state := by

  intro h

  cases h

theorem catch_boundary_no_abort_claim {Value State : Type}

    (inner : Result (core.result.Result Value Failure × Runtime × State))

    (runtime : Runtime) (state : State) :

    ¬ CatchBoundaryRelation inner (.abort : NativeOutcome Value Diagnostic) runtime state := by

  intro h

  cases h

theorem panic_boundary_success_projection {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String)

    (value : Raw) (runtime : Runtime) (state : State) :

    PanicBoundaryRelation project render (.ok (.Ok value, runtime, state))

      (.success (project value)) { runtime with o := false } state := by

  exact .success _ _ _

theorem panic_boundary_rejection_new_hook {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String)

    (failure : Failure) (diagnostic : Diagnostic) (runtime : Runtime) (state : State)

    (converted : chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic failure =

      .ok diagnostic) :

    PanicBoundaryRelation project render (.ok (.Err failure, runtime, state))

      (.unexpectedUnwind (.str (render diagnostic)))

      { runtime with o := false,

        hook := runtime.hook ++ [(runtime.u, false, runtime.installed)] } state := by

  exact .rejection _ _ _ _ converted

theorem panic_boundary_clears_output {Raw Value State : Type}

    {project : Raw → Value} {render : Diagnostic → String}

    {inner : Result (core.result.Result Raw Failure × Runtime × State)}

    {native : NativeOutcome Value Diagnostic} {runtime : Runtime} {state : State}

    (h : PanicBoundaryRelation project render inner native runtime state) :

    runtime.o = false := by

  cases h

  all_goals rfl

theorem panic_boundary_rejection_effect {Raw Value State : Type}

    {project : Raw → Value} {render : Diagnostic → String}

    {failure : Failure} {before after : Runtime} {state state' : State}

    {native : NativeOutcome Value Diagnostic}

    (h : PanicBoundaryRelation project render

      (.ok (.Err failure, before, state)) native after state') :

    ∃ diagnostic,

      chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic failure = .ok diagnostic ∧

      native = .unexpectedUnwind (.str (render diagnostic)) ∧

      state' = state ∧

      after = { before with o := false,

        hook := before.hook ++ [(before.u, false, before.installed)] } := by

  cases h with

  | rejection failure diagnostic runtime state converted =>

    exact ⟨diagnostic, converted, rfl, rfl, rfl⟩

theorem panic_boundary_no_lean_failure {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String) (error : Error)

    (native : NativeOutcome Value Diagnostic) (runtime : Runtime) (state : State) :

    ¬ PanicBoundaryRelation project render (.fail error) native runtime state := by

  intro h

  cases h

theorem panic_boundary_no_divergence {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String)

    (native : NativeOutcome Value Diagnostic) (runtime : Runtime) (state : State) :

    ¬ PanicBoundaryRelation project render .div native runtime state := by

  intro h

  cases h

theorem panic_boundary_no_typed_resume {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String)

    (inner : Result (core.result.Result Raw Failure × Runtime × State))

    (failure : Failure) (runtime : Runtime) (state : State) :

    ¬ PanicBoundaryRelation project render inner (.typedUnwind failure) runtime state := by

  intro h

  cases h

theorem panic_boundary_no_returned_diagnostic {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String)

    (inner : Result (core.result.Result Raw Failure × Runtime × State))

    (diagnostic : Diagnostic) (runtime : Runtime) (state : State) :

    ¬ PanicBoundaryRelation project render inner (.ordinaryError diagnostic) runtime state := by

  intro h

  cases h

theorem panic_boundary_no_abort_claim {Raw Value State : Type}

    (project : Raw → Value) (render : Diagnostic → String)

    (inner : Result (core.result.Result Raw Failure × Runtime × State))

    (runtime : Runtime) (state : State) :

    ¬ PanicBoundaryRelation project render inner (.abort : NativeOutcome Value Diagnostic)

      runtime state := by

  intro h

  cases h

end ChelisBridge.Runtime.Entries
