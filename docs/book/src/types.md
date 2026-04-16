# Type System Basics

Chelis keeps tensor dimensions and precision explicit.

## Primitive Ideas

- No implicit precision promotion
- No implicit broadcasting
- Named tensor dimensions must match by name
- Integer literals default to `int32`, float literals to `f32`

## Tensor Type Fragments

```chelis-deep-fragment
(t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))
```

```chelis-surf-fragment
tensor[batch, seq, f32]
```

## A Small Typed Program

```chelis-surf
def add_vec(x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = add(x, y)
```

## Reading the Deep Shape

```chelis-deep-fragment
(defsig {}
  add_vec
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))
```
