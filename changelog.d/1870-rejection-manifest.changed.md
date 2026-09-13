Derive the unimplemented-rejection issue manifest from exact production source
citations, rejecting missing, stale, dynamic, or malformed issue authority
before generated Rust membership is accepted. Production source discovery now
uses every repository-local Cargo workspace package, one cfg(test=false)
parser view shared with the direct-construction boundary, and an independent
rustc dep-info closure. Required production-target features are derived from
Cargo metadata, and every exact target identity must emit non-test dep-info.
Every Rust or Cargo manifest edit retriggers live issue validation. The
dedicated liveness job owns fresh source and compiler evidence; ordinary
script-unit tests use injected evidence instead of repeating the full Cargo
derivation four times.
