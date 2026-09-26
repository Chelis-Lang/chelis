import ChelisBridge.Model

/-!
# The only contact point with the generated `chelis_effects` items

`chelis_ir` depends on `chelis_effects`, so the `chelis_effects` roots are extracted into the
same Lean environment as the IR roots (the combined model, or the `chelis_effects` chunks of a
whole-core run). Their names follow `ChelisBridge.Model`: with `open_chelis_model`,
`chelis_effects.<path>` and the crates they depend on, `<crate>.<path>` (`chelis_deep`,
`chelis_vocab`), resolve in both layouts.
-/

open_chelis_model

namespace ChelisBridge.EffectsModel

/-- The extracted production host-C device check `chelis_effects::is_c_host_device_designator`. -/
abbrev isCHostDeviceDesignator : Aeneas.Std.Str → Aeneas.Std.Result Bool :=
  chelis_effects.is_c_host_device_designator

end ChelisBridge.EffectsModel
