Transcendentals are correctly rounded in every CPU lane. `exp`, `log`, `sin`,
`cos`, `tan`, `atan`, and `tanh` return the exact value rounded once, at f32
and f64 alike, in `chelis eval`, constant folding, `chelis prove`, and built C
programs. The kernels are the vendored CORE-MATH sources, which generated C
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
  as `exp(x) >= 0`.
- The serialized graph schema moves from 23 to 24.
- HIP and Metal reject the transcendentals at build time until the device
  lanes have correctly rounded kernels.
