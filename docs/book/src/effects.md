# Effects and Handlers

Chelis tracks observable host work and resource requirements in function types.
The checker infers effects after checking types. An effect annotation can set an
upper bound on what a function may do.

## Host I/O and tests

- `IO` covers host operations such as `print` and file access. It is allowed
  at the program boundary.
- `Test` comes from assertions and is handled by `chelis test`.

A function may declare an effect bound with `! { ... }`:

```chelis-surf
def report(value: f32) -> unit ! { IO } = print(value)
```

`! { IO }` permits host I/O. `! {}` declares a pure upper bound; leaving off
the suffix lets the checker infer the effects.

## Random keys

Random draws are pure functions of explicit keys. `dropout` and `uniform_like`
take a key as their first argument and add no effect. Create a key from a seed,
then split it when you need more than one draw. Each key can be used at most
once along any execution path:

```chelis-surf-fragment
(first_key, second_key) = split_key(key_from_seed(42i64))
first = dropout(first_key, x, 0.5f32)
second = dropout(second_key, x, 0.5f32)
```

See [Type System Reference](type-reference.md) for the `key` type.

## Device regions

`with device(...)` marks a requested placement region. The device name must
be a string literal, and a build checks whether its target accepts that
request. For the C target, the accepted name is exactly `"cpu"`:

```chelis-surf
def relu_on_cpu[n](x: tensor[n, f32]) -> tensor[n, f32] = with device("cpu") { relu(x) }
```

A C build rejects another device name with `BuildTargetMismatch` before
writing artifacts. For other targets and device support, see [Backends](backends.md).
The precise effect and target rules are in `spec/04-type-system.md` §7.
