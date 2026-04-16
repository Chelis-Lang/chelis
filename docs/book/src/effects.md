# Effects And Handlers

Chelis tracks effects explicitly. Pure tensor code stays pure; host interaction and
randomness are surfaced in types and syntax.

## Common Effects

- `Random`
- `IO`
- device-selection handlers for compiled execution paths

## Handler Fragments

```chelis-surf-fragment
with seed(42) {
  sample_fn(x)
}
```

```chelis-deep-fragment
(with-handler {effect Random seed 42}
  (app {} (var {} sample_fn) (var {} x)))
```

Use fragment fences here because the book is teaching handler shape, not promising that
this exact snippet is a full standalone program.

For the active effect semantics, use `spec/07-concurrency.md` and the effect-checking
crates/tests.
