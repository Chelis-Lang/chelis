# Effects and Handlers

Chelis tracks effects explicitly. Pure tensor code stays pure, and the places that touch
randomness, host input and output, or a specific device are surfaced in the type and
discharged by a handler. Effect inference runs after type checking: a function's effect set
is the union of the effects of the operations in its body.

## The effects

- `Random` comes from `dropout` and `uniform_like`, and from the `Std.Init` initializers
  that call them. It is discharged by `with seed(...)`.
- `IO` is inferred from host operations such as `print` and the file builtins. It is
  permitted at the top level rather than requiring a handler.
- `Resource("device")` marks a region that runs on a named device. It is introduced by
  `with device(...)` and validated against the build target.

## Annotating effects

A signature or a `def` carries its effect set as a `! { ... }` suffix. The annotation is
optional; the checker infers the set and verifies any annotation you supply.

```chelis-surf-fragment
sig predict[n]: tensor[n, f32] -> tensor[n, f32] ! { Random }
```

In Deep the effect set is `eff` metadata on the function type:

```chelis-deep-fragment
(t-fn {eff: (effects {} random)}
  (t-tensor {} (d-var {} n) (t-prim {} f32))
  (t-tensor {} (d-var {} n) (t-prim {} f32)))
```

## Handlers

A handler is a `with` block. `with seed(...)` takes an i64-suffixed integer literal
(`42i64`; the seed is semantically i64, so an unsuffixed literal is a type error) and makes
the randomness inside it deterministic; an unhandled `Random` effect at the top level is a
check error with repair guidance. `with device(...)` takes a string literal naming the device.

The core host-C build accepts only `with device("cpu")` or a named host selector such as
`with device("cpu:worker_0")`. Any accelerator, unknown, or malformed selector—including
`cuda:0`, `metal`, an empty string, or `cpu:`—produces a `BuildTargetMismatch` before
`chelis build --target c` writes artifacts. Accelerator placement and transfer semantics are
experimental; a device request is never treated as permission to run the region on host C.

```chelis-surf-fragment
with seed(42i64) {
  dropout(x, 0.5)
}
```

Handlers nest. A region can sit on a device and seed its randomness at once:

```chelis-surf-fragment
with device("gpu:0") {
  with seed(42i64) {
    dropout(x, 0.5)
  }
}
```

The handled names are `seed` and `device`. For the effect-checking details see
`spec/04-type-system.md` and the effect-checking crates and tests.
