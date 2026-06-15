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
sig predict: tensor[n, f32] -> tensor[n, f32] ! { Random }
```

In Deep the effect set is `eff` metadata on the function type:

```chelis-deep-fragment
(t-fn {eff: (effects {} random)}
  (t-tensor {} (d-name {} n) (t-prim {} f32))
  (t-tensor {} (d-name {} n) (t-prim {} f32)))
```

## Handlers

A handler is a `with` block. `with seed(...)` takes an integer literal and makes the
randomness inside it deterministic; an unhandled `Random` effect at the top level is a check
error with repair guidance. `with device(...)` takes a string literal naming the device.

```chelis-surf-fragment
with seed(42) {
  dropout(x, 0.5)
}
```

Handlers nest. A region can sit on a device and seed its randomness at once:

```chelis-surf-fragment
with device("gpu:0") {
  with seed(42) {
    dropout(x, 0.5)
  }
}
```

The handled names are `seed` and `device`. For the effect-checking details see
`spec/04-type-system.md` and the effect-checking crates and tests.
