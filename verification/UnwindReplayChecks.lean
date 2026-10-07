import ChelisBridge.Runtime.Unwind

/-!
# Lean side of the unwind replay differential (NATIVE_SHELL_SCOPE.md O8)

The file prints one line for each segment of `Unwind.nativeSegments`, in the format of
`verification/unwind_replay_oracle.rs`. `check_models.py` compares the two outputs. The lines come
from the new semantics. The old semantics must give the same events and values. Else the file
prints a `MISMATCH` line, and the comparison fails. The theorem
`Unwind.replay_native_observations` states the same table.
-/

open ChelisBridge.Runtime.Unwind

namespace UnwindReplayChecks

def diagnosticLabel (d : Diag) : String :=
  d.msg ++ (if d.fatal then "!" else "")

def payloadLabel : Payload → String
  | .diag d => s!"diag({diagnosticLabel d})"
  | .unrep => "unrep"
  | .str s => s!"str({s})"
  | .other id _ => s!"other({id})"

def observedLabel : Observed → String
  | .hook p => s!"hook:{payloadLabel p}"
  | .drop fid => s!"drop:{fid}"
  | .payloadDrop id => s!"payload-drop:{id}"

def returnedLabel : Returned → String
  | .ok v => s!"ok({v})"
  | .err d => s!"err({diagnosticLabel d})"
  | .unwind p => s!"unwind:{payloadLabel p}"
  | .abort => "abort"

def line (segment : String × List Observed × List Returned) : String :=
  let (name, events, returned) := segment
  s!"segment {name}: [{", ".intercalate (events.map observedLabel)}] " ++
    s!"[{", ".intercalate (returned.map returnedLabel)}]"

#eval show IO Unit from do
  let new := replay newSem nativeSegments s0
  let old := replay old nativeSegments s0
  unless new.length == old.length do
    IO.println s!"MISMATCH: {new.length} new and {old.length} old segments"
  for (n, o) in new.zip old do
    unless n == o do
      IO.println s!"MISMATCH old: {line o}"
    IO.println (line n)

end UnwindReplayChecks
