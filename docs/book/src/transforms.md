# Transforms: grad and vmap

Transforms are compiler features, not library macros.

## `grad`

Use `grad` for reverse-mode differentiation over supported programs.

```chelis-surf-fragment
grad(loss_fn)
```

```chelis-deep-fragment
(grad {} (var {} loss_fn))
```

## `vmap`

Use `vmap` to vectorize a per-example function over a batch dimension.

```chelis-surf-fragment
vmap(per_example_fn)
```

```chelis-deep-fragment
(vmap {} (var {} per_example_fn) (d-name {} batch))
```

The exact supported surface is still defined by the compiler and current tests; prefer
small working programs and compiler feedback over extrapolating from planned future
surface.
