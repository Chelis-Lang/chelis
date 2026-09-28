# Effects and Handlers

Chelis tracks effects explicitly. Pure tensor code stays pure, and the places that touch
host input and output or a specific device are surfaced in the type. Effect inference runs
after type checking: a function's effect set is the union of the effects of the operations
in its body.

## The effects

- `IO` is inferred from host operations such as `print` and the file builtins. It is
  permitted at the top level rather than requiring a handler.
- `Resource("device")` marks a region that runs on a named device. It is introduced by
  `with device(...)` and validated against the build target.
- `Test` comes from the assertion operations; only `chelis test` handles it.

## Randomness is not an effect

A random draw is a pure function of the key it is given. `dropout`, `uniform_like` and the
`Std.Init` initializers take a `key` as their first argument and contribute no effect, so a
function that draws takes a `key` parameter and needs no annotation. `key_from_seed(42i64)`
makes a root key; `split_key`, `split_keys` and `fold_in` derive fresh keys from one. A key
is used at most once on every path, so two draws need two keys:

```chelis-surf-fragment
(k1, k2) = split_key(key_from_seed(42i64))
first = dropout(k1, x, 0.5)
second = dropout(k2, x, 0.5)
```

`with seed(...)` and a `Random` effect annotation are retired spellings; the parser rejects
both. See [Types](type-reference.md) for the `key` type.

## Annotating effects

A signature or a `def` carries its effect set as a `! { ... }` suffix. The annotation is
optional; the checker infers the set and verifies any annotation you supply.

```chelis-surf-fragment
sig report[n]: tensor[n, f32] -> unit ! { IO }
```

In Deep the effect set is `eff` metadata on the function type:

```chelis-deep-fragment
(t-fn {eff: (effects {} io)}
  (t-tensor {} (d-var {} n) (t-prim {} f32))
  (t-unit {}))
```

## Handlers

A handler is a `with` block, and `with device(...)` is the one user handler. It takes a
string literal naming the device.

The core host-C build accepts only exact `with device("cpu")`. Any labeled CPU,
accelerator, unknown, or malformed selector—including `cpu:worker_0`, `cuda:0`,
`metal`, an empty string, or `cpu:`—produces a `BuildTargetMismatch` before
`chelis build --target c` writes artifacts. Device-label vocabulary, placement,
and transfer semantics remain experimental; a device request is never treated
as permission to run the region on host C.

A draw inside a device region takes its key like any other:

```chelis-surf-fragment
with device("gpu:0") {
  dropout(k, x, 0.5)
}
```

The handled name is `device`. For the effect-checking details see
`spec/04-type-system.md` and the effect-checking crates and tests.
