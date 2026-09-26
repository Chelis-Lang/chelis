import Lake
open Lake DSL

/-!
Proof bridge between the Aeneas-extracted production Rust model and the pinned
LaCaDiLE reference core (`LaCaDiLECore`).

Configuration (all optional, `lake -K key=value ...`):

* `aeneas`: the built patched Aeneas `backends/lean` library, required by path and never
  written here. Lake materializes a locked path dependency from `lake-manifest.json` /
  `.lake/package-overrides.json`, not from this declaration, so select a library with
  `prepare_bridge.py --aeneas-lean-dir <dir>` (writes the override) or
  `lake -K aeneas=<dir> update aeneas`.
* `model`: directory holding the generated model sources (default: the `model` symlink).
* `modelRoots`: comma-separated root module(s) of the generated model (default `ChelisIr`:
  one extraction whose roots include the IR and the `chelis_effects` entry points).
  `ChelisBridge/Model.lean` is the only Lean file naming generated declarations.

LaCaDiLE is pinned by URL and revision in `lake-manifest.json`. The repository is not
publicly clonable, so `prepare_bridge.py` seeds `.lake/packages/LaCaDiLE` from a local
clone at exactly the pinned revision. Mathlib is never built from source: its prebuilt
oleans come from `lake exe cache get`.
-/

def aeneasDir : System.FilePath :=
  (get_config? aeneas).getD "../../.work/aeneas-verification/aeneas/backends/lean"

def modelDir : System.FilePath := (get_config? model).getD "model"

def modelRoots : Array Lean.Name :=
  (((get_config? modelRoots).getD "ChelisIr").splitOn ",").toArray.map String.toName

package chelis_bridge

require aeneas from aeneasDir

require LaCaDiLE from git
  "https://github.com/Chelis-Lang/LaCaDiLE" @ "11531c8b0e3e6fb6650ddbc404937e5c527b0785" / "lean"

/-- The generated production model, compiled from source by this package. -/
lean_lib ChelisModel where
  srcDir := modelDir
  roots := modelRoots

@[default_target]
lean_lib ChelisBridge where
  roots := #[`ChelisBridge]
  -- Every module under `ChelisBridge/` is built, including ones not imported by the root.
  globs := #[.andSubmodules `ChelisBridge]
  leanOptions := #[⟨`autoImplicit, false⟩]
