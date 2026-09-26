import ChelisIr

/-!
# The only contact point with the generated `chelis_effects` items

`chelis_ir` depends on `chelis_effects`, so the `chelis_effects` roots are extracted into the
same model as the IR roots (one Lean environment: separately extracted models cannot be
imported together, since each defines the same root-level instances for shared types).
There they are named `<root>.chelis_effects.<path>`, and the crates they depend on
`<root>.<crate>.<path>` (`chelis_deep`, `chelis_vocab`). With the root namespace opened,
`chelis_effects.<path>` and `<crate>.<path>` resolve, exactly as in `ChelisBridge.Model`.
Swapping the model changes the `import` and the namespace opened by
`open_chelis_effects_model`.
-/

-- `open_chelis_effects_model` opens the configured extraction root namespace (the swap point).
set_option hygiene false in
macro "open_chelis_effects_model" : command => `(open chelis_ir)

open_chelis_effects_model

namespace ChelisBridge.EffectsModel

/-- The extracted production host-C device check `chelis_effects::is_c_host_device_designator`. -/
abbrev isCHostDeviceDesignator : Aeneas.Std.Str → Aeneas.Std.Result Bool :=
  chelis_effects.is_c_host_device_designator

end ChelisBridge.EffectsModel
