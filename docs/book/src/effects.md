# Effects

Chelis tracks observable host work and resource requirements in function types.
The checker infers effects after checking types. An effect annotation can set an
upper bound on what a function may do.

## Host I/O and tests

- `IO` covers every host interaction: `print` and `debug`, file reads and
  writes, directory listing, running a subprocess, and reading the wall or
  monotonic clock. A function's inferred effects are the union of the effects
  of the operations in its body, so any caller of an `IO` function is `IO`
  too. `IO` is allowed at the top level of a program.
- `Test` comes from assertions such as `test_assert`. `chelis test` runs
  them and reports a failed assertion as a failed test; see
  [Testing](testing.md).

A function may declare an effect bound with `! { ... }`:

```chelis-surf
def report(value: f32) -> unit ! { IO } = print(value)
r = report(1.5f32)
```

`chelis eval --file` prints `1.5` from the call and then the value of `r`,
`r = ()`. `! { IO }` permits host I/O. `! {}` declares the function pure;
leaving off the suffix lets the checker infer the effects. A body that does
more than its bound allows fails `chelis check`. With `! {}` on `report`:

```text
{"kind":"UnhandledEffect","message":"Function `report` is declared with effects `{}` but its body performs effects `{IO}` that were not declared","severity":0.8,"suggestions":["Either add the missing effect(s) to the signature of `report` (e.g. `! { IO }`) or refactor the body so it does not perform them."]}
```

## Random keys

Random draws are pure functions of explicit keys and add no effect.
`dropout(k, x, rate)` zeroes each element of the float tensor `x` with
probability `rate` (finite, `0 <= rate < 1`) and divides the kept elements by
`1 - rate`; `uniform_like(k, x, low, high)` returns a tensor of `x`'s shape
filled from `[low, high]` (finite bounds, `low <= high`), ignoring `x`'s
values. A rate or bound outside its domain stops evaluation before any value is
drawn. See [explicit randomness](stdlib.md#explicit-randomness) for
the key functions. Create a key from a seed,
then split it when you need more than one draw. Each key can be used at most
once along any execution path. Tuple destructuring such as the first binding below
is written inside a function body:

```chelis-surf
def two_masks[n](x: tensor[n, f32]) -> tensor[n, f32] = {
  (first_key, second_key) = 42i64 |> key_from_seed |> split_key
  first = dropout(first_key, x, 0.5f32)
  second = dropout(second_key, x, 0.5f32)
  add(first, second)
}
```

See [Type System Reference](type-reference.md) for the `key` type.

## Device regions

`with device(...)` marks a requested placement region. The device name must
be a string literal, and a build checks whether its target accepts that
request. For the C target, the accepted name is exactly `"cpu"`:

```chelis-surf
def relu_on_cpu[n](x: tensor[n, f32]) -> tensor[n, f32] = with device("cpu") { relu(x) }
```

Any other name is accepted by `chelis check` and `chelis eval`, but
`chelis build` rejects it with an error that names the region and says that
only `cpu` is accepted, before it writes any file. Nested regions are checked
one by one, so a `cpu` region inside another region does not make the outer
name acceptable.
