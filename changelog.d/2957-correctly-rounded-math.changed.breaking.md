Transcendentals are correctly rounded in every CPU lane. `exp`, `log`, `sin`,
`cos`, `tan`, `atan`, and `tanh` return the exact value rounded once, at f32
and f64 alike, in `chelis eval`, `chelis prove`, and built C programs. The kernels are the vendored CORE-MATH sources, which generated C
defines as `static` functions; built programs no longer call the platform math
library, Accelerate vForce, or Sleef, and the `sleef` feature is gone. See
[#2957](https://github.com/Chelis-Lang/chelis/issues/2957).

What changes for existing programs:

- Results of transcendentals, and of the activations and compounds built from
  them (`sigmoid`, `silu`, `gelu`, `softmax`, `normal_cdf`), change in their
  last bits wherever the old lanes were not correctly rounded. `chelis eval` and
  the built executable now agree bit for bit, and [05-OBS-3] grants no
  cross-lane tolerance for any operation.
- `tanh` is a primitive operation with its own correctly rounded kernel and
  adjoint rather than a composition of `sigmoid`.
- `gelu` is respelled as `x * sigmoid(2u)`, which no longer overflows to
  infinity near the largest finite input.
- Contract verdicts at f32 are computed at f32. A property that held only
  because f32 was evaluated at f64 can now fail; `std.exp.positivity` is stated
  as `exp(x) >= 0`. `chelis prove` discharges the standard float contracts at
  each width ([#2965](https://github.com/Chelis-Lang/chelis/issues/2965)):
  `std.normal_cdf.reflection` reports a counterexample at f16, bf16, and f32,
  and `std.normal_cdf.monotonicity` reports one at f16.
- Built C programs compile with one strict profile, `-O2 -ffp-contract=off
  -fno-fast-math`, and `-march=native` is gone. Native tools run with a cleared
  environment that keeps only `PATH` and `TMPDIR`, so `CFLAGS`, `CPATH`,
  `CCC_OVERRIDE_OPTIONS`, `NIX_CFLAGS_COMPILE`, `SDKROOT`, and the like no
  longer change a build. On macOS the build sets `SDKROOT` itself, to the SDK of
  the `xcode-select` default, so naming another Xcode's `clang` in `CHELIS_CC`
  selects that compiler, and the commands `--emit-c` prints run their tools
  through `env SDKROOT=…`. A compiler named in `CHELIS_CC`, a wrapper script
  included, that does not honor the profile (for example one adding `-O0` or
  `-ffast-math`) fails the build with a diagnostic
  ([#2962](https://github.com/Chelis-Lang/chelis/issues/2962)).
- The serialized graph schema moves from 23 to 24.
- HIP and Metal reject the transcendentals and `sqrt` at build time until the
  device lanes have correctly rounded kernels for them.
- Every arithmetic result and every float-to-float conversion that produces a
  NaN stores the canonical quiet NaN in both `chelis eval` and built C
  programs, while selection and data movement keep the operand's NaN bits
  ([#2964](https://github.com/Chelis-Lang/chelis/issues/2964)).
- f64 accumulators narrow to f16 and bf16 storage with one round to nearest
  even; the runtime previously misrounded some values, for example
  `1 + 2^-8 + 2^-30` to bf16 `0x3f80` instead of `0x3f81`
  ([#3041](https://github.com/Chelis-Lang/chelis/issues/3041)).
- `einsum` in `chelis eval` accumulates at the default accumulator (f32 for
  f16, bf16, and f32 operands) in the same order as built programs, instead of
  in f64, so the two lanes agree bit for bit; for example `einsum` over
  `[2^24, 1, 1, 1]` at f32 now gives `16777218` in both.
- f16 and bf16 scatter-add, including the gradient of a gather with repeated
  indices, adds at f32 and narrows once in built C programs; it previously
  added the storage bit patterns as integers
  ([#3047](https://github.com/Chelis-Lang/chelis/issues/3047)).
- A top-level scalar f32 gradient in a built program is computed and printed
  at f32 rather than in double from the f64 literal
  ([#2993](https://github.com/Chelis-Lang/chelis/issues/2993)), and built
  scalar gradients follow the reverse-mode differentiation order `chelis eval`
  uses, so the two lanes agree bit for bit on multi-operation bodies
  ([#3017](https://github.com/Chelis-Lang/chelis/issues/3017)).
