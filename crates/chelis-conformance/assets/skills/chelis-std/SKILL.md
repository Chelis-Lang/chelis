---
name: chelis-std
description: Use when writing downstream Chelis Surf or Deep programs, validating Chelis examples, or authoring Reef shell packages that depend on the bundled chelis-std runtime. Not for modifying the Chelis compiler repository itself.
---

# Chelis Downstream Skill

Use this skill to write Chelis that works with the current compiler release and to help
shell authors prepare Reef packages. Keep context small: open the relevant public docs
when exact grammar, type, CLI, or package semantics matter.

This skill is for writing downstream programs and packages. It does not cover work on
the Chelis compiler itself.

## Setup and Discovery

- Follow the [installation guide](https://chelis.ch/docs/chelis/install/) and verify
  that `chelis --version` and `chelis --help` work. A compiler source checkout is not
  needed to write Chelis programs or use Reef packages.
- Use Surf (`.ch`) for human-facing code. Use Deep (`.dp`) for canonical machine output.
- Inspect current CLI commands with `chelis --help` and subcommand help, especially
  `chelis fmt --help`, `chelis check --help`, and `chelis reef --help`.
- For language syntax, types, commands, examples, and package workflows, use the
  [Chelis documentation](https://chelis.ch/docs/chelis/).

## Successful Language Subset

These common forms provide a useful starting point:

- Function definitions: `def f[n](x: tensor[n, f32]) -> tensor[n, f32] = ...`
- Blocks with local bindings: `{ y = relu(x); softmax(y, 0) }`
- Tensor types with explicit dimensions and precision: `tensor[f32]`,
  `tensor[n, f32]`, `tensor[batch, hidden, f32]`
- Elementwise ops: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `relu`,
  `sigmoid`
- Comparisons/logical ops: `cmplt`, `eq`, `neq`, `gt`, `lte`, `gte`, `and`, `or`, `not`
- Reductions with explicit integer axes: `sum(x, 0)`, `mean(x, 0)`,
  `max_reduce(x, 0)`, `softmax(x, 0)`
- Structural/tensor helpers are available but more shape-sensitive: `matmul`, `reshape`,
  `permute`, `expand`, `pad`, `cast`

Rules to preserve:

- No implicit broadcasting. Shapes must match unless an explicit helper changes them.
- No implicit precision promotion. Use `cast` when changing precision.
- `sum`, `cumsum`, `trace`, and `einsum` over `i8` or `i16` return `i32`;
  declare the result as `i32`, pass `accumulator=i64` to `sum` or `einsum`, or narrow
  it with an explicit `cast`.
- Named dimensions are nominal: `batch` and `seq` do not unify by size.
- Integer literals default to `i32`; float literals default to `f32`. A suffix states
  another dtype, and so does the construct that directly contains the literal: a
  declaration (`x: f64 = 1.1`), a `cast` (`cast(1.1, f64)` is `1.1f64`), or the dtype
  argument of `to_tensor` (`to_tensor([1.1, 2.2], f64)`). An unsuffixed `to_tensor`
  element needs a suffix or the dtype argument.
- Reduction-style calls need an explicit axis argument.

## Style Rules for Generated Surf

- Prefer `def ... -> T = ...`.
- Put types on function parameters and returns; avoid unnecessary intermediate
  ascriptions.
- Use snake_case for functions and values, PascalCase for types and module segments.
- Use symbolic names for runtime-varying dimensions (`batch`, `seq`, `hidden`) and
  concrete dimensions for fixed architecture sizes.
- Run the formatter instead of hand-tuning whitespace.

## Minimal Examples

```chelis-surf
def square(x: tensor[f32]) -> tensor[f32] = mul(x, x)
```

```chelis-surf
def relu_then_softmax[n](x: tensor[n, f32]) -> tensor[n, f32] = x |> relu |> softmax(0)
```

```chelis-surf
def add_vec[n](x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = add(x, y)
```

```chelis-surf
def twice_then_relu[n](x: tensor[n, f32]) -> tensor[n, f32] = {
  y = add(x, x)
  relu(y)
}
```

```chelis-surf
def classify[n](x: tensor[n, f32], labels: tensor[n, f32]) -> tensor[f32] = {
  logits = x |> relu |> add(labels)
  loss =
    softmax(logits, 0)
    |> log
    |> mul(labels)
    |> sum(0)
  loss
}
```

```chelis-surf
def logistic_step[n](x: tensor[n, f32]) -> tensor[n, f32] = sigmoid(x)
```

```chelis-surf
def clamp_low[n](x: tensor[n, f32], low: tensor[n, f32]) -> tensor[n, f32] = max_elem(x, low)
```

```chelis-surf
def identity[a](x: tensor[a, f32]) -> tensor[a, f32] = x
```

```chelis-surf
type Weights = tensor[n, f32]
def keep(w: Weights) -> Weights = w
```

```chelis-surf
type Activation =
  | Relu
  | Sigmoid
def activate[n](act: Activation, x: tensor[n, f32]) -> tensor[n, f32] =
  match act with {
    | Relu => relu(x)
    | Sigmoid => sigmoid(x)
  }
```

```chelis-surf
type Optimizer =
  | Sgd { lr: tensor[f32] }
  | Adam { lr: tensor[f32], beta1: tensor[f32], beta2: tensor[f32], eps: tensor[f32] }
def learning_rate(opt: Optimizer) -> tensor[f32] =
  match opt with {
    | Sgd { lr } => lr
    | Adam { lr, beta1, beta2, eps } => lr
  }
```

## Canonical Deep Examples

Generate Deep only when a tool needs the canonical AST. Every node includes its metadata
map, calls use `app`, references use `var`, and literals carry a type.

```chelis-deep
(defsig {}
  square
  (t-fn {} (t-tensor {} (t-prim {} f32)) (t-tensor {} (t-prim {} f32))))

(def {}
  square
  (fn {}
    (params {} (x {type: (t-tensor {} (t-prim {} f32))}))
    (app {} (var {} mul) (var {} x) (var {} x))))
```

```chelis-deep
(defsig {}
  add_vec
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  add_vec
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
      (y {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (app {} (var {} add) (var {} x) (var {} y))))
```

```chelis-deep
(defsig {}
  twice_then_relu
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  twice_then_relu
  (fn {}
    (params {} (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (let {}
      (bind {}
        y
        (app {} (var {} add) (var {} x) (var {} x)))
      (app {} (var {} relu) (var {} y)))))
```

```chelis-deep
(defsig {}
  relu_then_softmax
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  relu_then_softmax
  (fn {}
    (params {} (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (app {}
      (var {} softmax)
      (app {} (var {} relu) (var {} x))
      (lit {type: (t-prim {} i32)} 0))))
```

```chelis-deep
(defsig {}
  classify
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (t-prim {} f32))))

(def {}
  classify
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
      (labels {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (let {}
      (bind {}
        logits
        (app {} (var {} add) (app {} (var {} relu) (var {} x)) (var {} labels))
        loss
        (app {}
          (var {} sum)
          (app {}
            (var {} mul)
            (app {}
              (var {} log)
              (app {}
                (var {} softmax)
                (var {} logits)
                (lit {type: (t-prim {} i32)} 0)))
            (var {} labels))
          (lit {type: (t-prim {} i32)} 0)))
      (var {} loss))))
```

```chelis-deep
(defsig {}
  logistic_step
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  logistic_step
  (fn {}
    (params {} (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (app {} (var {} sigmoid) (var {} x))))
```

```chelis-deep
(defsig {}
  clamp_low
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  clamp_low
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
      (low {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (app {} (var {} max_elem) (var {} x) (var {} low))))
```

```chelis-deep
(defsig {}
  identity
  (a)
  (t-fn {}
    (t-tensor {} (d-var {} a) (t-prim {} f32))
    (t-tensor {} (d-var {} a) (t-prim {} f32))))

(def {}
  identity
  (fn {}
    (params {} (x {type: (t-tensor {} (d-var {} a) (t-prim {} f32))}))
    (var {} x)))
```

```chelis-deep
(typealias {} Weights () (t-tensor {} (d-name {} n) (t-prim {} f32)))

(defsig {} keep (t-fn {} (t-adt {} Weights) (t-adt {} Weights)))

(def {} keep (fn {} (params {} (w {type: (t-adt {} Weights)})) (var {} w)))
```

```chelis-deep
(deftype {} Activation () (variant {} Relu) (variant {} Sigmoid))

(defsig {}
  activate
  (n)
  (t-fn {}
    (t-adt {} Activation)
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {}
  activate
  (fn {}
    (params {}
      (act {type: (t-adt {} Activation)})
      (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
    (match {}
      (var {} act)
      (arm {} (pat-ctor {} Relu) () (app {} (var {} relu) (var {} x)))
      (arm {} (pat-ctor {} Sigmoid) () (app {} (var {} sigmoid) (var {} x))))))
```

## Validation Commands

For a single file:

```sh
chelis fmt --inplace app.ch
chelis lint --check app.ch
chelis check app.ch
chelis deep app.ch > app.dp
chelis surf app.dp
```

For build artifacts:

```sh
chelis build app.ch --target c --output out/
```

`chelis build`, `chelis check`, `chelis validate`, and `chelis eval --file` run the
style gate before semantic work. Do not use `--allow-style-violations` in package or CI
flows.

## Reef Context

- `chelis-std` is the bundled runtime. It ships inside the compiler and is not installed
  with `chelis reef install`.
- Shell packages such as `nautilus`, `coral`, and `shoals` are Reef
  dependencies and can be installed or bootstrapped.
- `reef.toml` pins the compiler exactly and declares a module prefix. `chelis reef init`
  writes the pin for the toolchain that runs it; `X.Y.Z` below stands for that version:

```toml
schema = "3"

[package]
name = "demo"
version = "0.1.0"
compiler = "=X.Y.Z"
module_prefix = "Demo"
resolver = "2"

[dependencies]
nautilus = "^0.7"
```

- Package source lives under `src/`; module names should match the prefix and path.
- `CHELIS_REEF_HOME` defaults to `~/.chelis/reef`.

Common Reef loop:

```sh
chelis reef init demo --module-prefix Demo --output demo
cd demo
chelis reef build
```

## Where to Load More Detail

- Start with the [Chelis documentation](https://chelis.ch/docs/chelis/).
- Browse [examples](https://chelis.ch/docs/chelis/examples/), the
  [CLI reference](https://chelis.ch/docs/chelis/cli/), and the
  [Reef package guide](https://chelis.ch/docs/chelis/reef/).
