# Chelis SKILL

Use this file to prompt a frontier model to write Chelis that passes the compiler in
this repository today.
This is not the language spec.
It is a compact teaching document optimized for compiler-in-the-loop generation.

## 1. What Chelis Is

Chelis is a functional language for AI and tensor programs.
It has two syntaxes:
Surf (`.ch`) is the human-facing syntax, and Deep (`.dp`) is the canonical
s-expression syntax.
The compiler works on Deep internally.
If you want the highest success rate, generate Deep first.
Chelis programs are mostly tensor pipelines with explicit dimensions and explicit
precision.
The compiler returns a fitness score in `[0, 1]` plus structured errors.
For this repository snapshot, target the subset that passes the current parser and
type checker, not the broader planned surface from later phases.

## 2. Deep Syntax

Deep is the primary generation target.
Every real Deep node has the form `(tag {} children...)`.
The metadata map is always present, even when empty.

### 2a. Node Structure

Canonical shape:

```scheme
(tag {} child1 child2 ...)
```

Increasing-complexity examples:

```scheme
(lit {type: (t-prim {} f32)} 3.14)
(var {} x)
(app {} (var {} add) (var {} x) (var {} y))
(def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))
(module {}
  demo.core
  (export {} f)
  (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
  (def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))))
```

Rules that matter most:

- Every tagged node includes `{}` as element two.
- Function calls use `app`.
- Names use `var`.
- Literals use `lit` with a `type` metadata entry.
- Function parameters live inside `(params {} ...)`.
- `let` bindings live inside `(bind {} name expr ...)`.

### 2b. Complete Tag Vocabulary

Arity is the number of children after the metadata map.
Examples are illustrative; only complete `chelis-deep` blocks later in this file are
validated automatically.

#### Module

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `module` | 2+ | module name, declarations | `(module {} demo.core (def {} x ...))` |
| `import` | 2 | path, imported names list | `(import {} foo.bar (x y))` |
| `import-all` | 1 | path | `(import-all {} foo.bar)` |
| `export` | 1+ | exported names | `(export {} f g)` |

#### Declarations

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `def` | 2 | name, expression | `(def {} x (lit {type: (t-prim {} int32)} 1))` |
| `defsig` | 2 | name, type | `(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))` |
| `deftype` | 3+ | name, type params list, variants | `(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))` |
| `typealias` | 3 | name, type params list, aliased type | `(typealias {} Weights () (t-tensor {} (d-name {} n) (t-prim {} f32)))` |
| `variant` | 1+ | ctor name, fields or positional members | `(variant {} Adam (field {} lr (t-tensor {} (t-prim {} f32))))` |
| `field` | 2 | field name, field type | `(field {} lr (t-tensor {} (t-prim {} f32)))` |
| `defdim` | 1 | dimension name | `(defdim {} batch)` |

#### Expressions

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `fn` | 2 | params node, body | `(fn {} (params {} x) (var {} x))` |
| `app` | 1+ | function, arguments | `(app {} (var {} add) (var {} x) (var {} y))` |
| `let` | 2 | bind node, body | `(let {} (bind {} y (var {} x)) (var {} y))` |
| `match` | 2+ | scrutinee, arms | `(match {} (var {} opt) (arm {} (pat-wild {}) () (var {} x)))` |
| `arm` | 3 | pattern, guard or `()`, body | `(arm {} (pat-var {} x) () (var {} x))` |
| `if` | 3 | cond, then, else | `(if {} c t e)` |
| `var` | 1 | name | `(var {} x)` |
| `lit` | 1 | literal value | `(lit {type: (t-prim {} int32)} 0)` |
| `record` | 2+ | type/ctor name, kv pairs | `(record {} Adam (kv {} lr (var {} lr)))` |
| `access` | 2 | record expr, field | `(access {} (var {} opt) lr)` |
| `pipe` | 2+ | seed expr, stages | `(pipe {} (var {} x) (var {} relu))` |
| `block` | 1+ | sequenced expressions | `(block {} (var {} x) (var {} y))` |
| `tuple` | 2+ | tuple elements | `(tuple {} (var {} a) (var {} b))` |
| `tuple-get` | 2 | tuple expr, index | `(tuple-get {} (var {} pair) (lit {type: (t-prim {} int32)} 0))` |
| `record-update` | 2+ | base expr, kv pairs | `(record-update {} r (kv {} x v))` |
| `par` | 1+ | parallel expressions | `(par {} a b)` |

#### Patterns

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `pat-var` | 1 | bound name | `(pat-var {} x)` |
| `pat-lit` | 1 | literal value | `(pat-lit {} 0)` |
| `pat-ctor` | 1+ | ctor name, nested patterns | `(pat-ctor {} Some (pat-var {} x))` |
| `pat-tuple` | 2+ | tuple patterns | `(pat-tuple {} (pat-var {} a) (pat-var {} b))` |
| `pat-record` | 2+ | type/ctor name, kv pattern pairs | `(pat-record {} Adam (kv {} lr (pat-var {} lr)))` |
| `pat-wild` | 0 | none | `(pat-wild {})` |
| `pat-as` | 2 | bound name, nested pattern | `(pat-as {} whole (pat-ctor {} Some (pat-wild {})))` |

#### Types

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `t-prim` | 1 | primitive name | `(t-prim {} f32)` |
| `t-fn` | 2+ | argument types, return type | `(t-fn {} (t-prim {} f32) (t-prim {} f32))` |
| `t-tensor` | 1+ | dimensions, precision | `(t-tensor {} (d-name {} n) (t-prim {} f32))` |
| `t-adt` | 1+ | ADT name, type args | `(t-adt {} Option (t-prim {} f32))` |
| `t-var` | 1 | type variable | `(t-var {} a)` |
| `t-unit` | 0 | none | `(t-unit {})` |
| `t-tuple` | 2+ | tuple member types | `(t-tuple {} (t-prim {} f32) (t-prim {} f32))` |

#### Dimensions

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `d-name` | 1 | named dimension | `(d-name {} batch)` |
| `d-var` | 1 | polymorphic dimension variable | `(d-var {} a)` |
| `d-lit` | 1 | literal dimension size | `(d-lit {} 512)` |

#### Transforms

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `grad` | 1 | expression | `(grad {} (var {} f))` |
| `vmap` | 2 | expression, dimension | `(vmap {} (var {} f) (d-name {} batch))` |
| `jit` | 1 | expression | `(jit {} (var {} f))` |
| `realize` | 1 | expression | `(realize {} (var {} x))` |
| `cast` | 2 | expression, target type | `(cast {} (var {} x) (t-prim {} bf16))` |
| `copy` | 1 | expression | `(copy {} (var {} x))` |

#### Metaprogramming

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `quote` | 1 | quoted expr | `(quote {} (var {} x))` |
| `unquote` | 1 | expr to splice as value | `(unquote {} (var {} x))` |
| `splice` | 1 | expr producing list of nodes | `(splice {} xs)` |

#### Helpers

| Tag | Arity | Children | Example |
|---|---:|---|---|
| `params` | 0+ | bare names or typed helper pairs | `(params {} x (y {type: (t-prim {} f32)}))` |
| `bind` | even | `name expr` pairs | `(bind {} x (var {} y) z (var {} x))` |
| `kv` | 2 | key, value | `(kv {} lr (var {} lr))` |

### 2c. Built-in Scope

The safest rule is: only use names listed here.
If a name is not listed here, assume it does not exist.

Notation:

- `tensor[p]` means a rank-0 tensor scalar such as `tensor[f32]`
- `tensor[D, p]` means a tensor with one or more named dimensions followed by precision
- `bool-tensor[D]` means `tensor[D, bool]`
- reductions currently require an explicit integer axis argument

#### Stable in the current checker

| Name | Shape | Use |
|---|---|---|
| `add`, `mul`, `sub`, `div`, `max_elem`, `min_elem` | `(tensor[D, p], tensor[D, p]) -> tensor[D, p]` | elementwise binary ops |
| `neg`, `exp`, `log`, `sin`, `sqrt`, `relu`, `sigmoid` | `tensor[D, p] -> tensor[D, p]` | elementwise unary ops |
| `cmplt`, `eq`, `neq`, `gt`, `lte`, `gte` | `(tensor[D, p], tensor[D, p]) -> bool-tensor[D]` | comparisons |
| `and`, `or` | `(bool-tensor[D], bool-tensor[D]) -> bool-tensor[D]` | logical binary ops |
| `not` | `bool-tensor[D] -> bool-tensor[D]` | logical unary op |
| `sum`, `mean`, `max_reduce`, `softmax` | `(tensor[D, p], int32) -> tensor[D, p]` | reductions; always pass an axis |

#### Present but shape rules are still moving

These names exist in the environment, but do not build your first attempts around them
unless you are prepared to iterate with compiler feedback:

- `matmul`
- `layer_norm`
- `conv2d`
- `normalize`
- `reshape`
- `permute`
- `expand`
- `pad`
- `shrink`
- `stride`

#### Do not invent these yet

These appear in the broader roadmap or spec language, but this repository snapshot does
not make them reliable teaching targets for the skill:

- `dot`
- `linear`
- `cross_entropy`
- `embedding`
- `mha`
- `tensor_const`

### 2d. Canonical Form Rules

For Deep, always emit canonical form:

- every node is `(tag {} ...)`
- metadata map is always present
- use `app`, `var`, and `lit` explicitly
- record `kv` pairs are alphabetized by key
- `params` and `bind` use helper form, not `var`
- use the canonical printer style: two-space indentation in nested forms, no alternate
  parenthesization
- if you have Surf and need Deep, prefer Surf parse + desugar + canonical print rather
  than writing Deep freehand

## 3. Surf Syntax

Surf is the readable syntax.
It is sugar over Deep.
If you are unsure, write Deep instead.

### Keywords

Current Surf keywords:

```text
def sig let in type dim match with fn module import export if then else
grad vmap jit cast realize copy par true false
```

### Operator Precedence

From lowest to highest:

1. `|>`
2. `||`
3. `&&`
4. `==` `!=`
5. `<` `>` `<=` `>=`
6. `+` `-`
7. `*` `/` `%`
8. unary `-` `!`
9. function application
10. field access `.`

### Key Constructs

| Surf | Deep |
|---|---|
| `def f(x: T): U = body` | `defsig` + `def` with typed `params` |
| `let x = e in body` | `(let {} (bind {} x e') body')` |
| `{ let x = e; body }` | same `let` shape after block desugaring |
| `match x with { | P => b }` | `(match {} x' (arm {} P' () b'))` |
| `fn (x) -> body` | `(fn {} (params {} x) body')` |
| `type Option[a] = | Some(a) | None` | `(deftype {} Option (a) ...)` |
| `type Weights = tensor[n, f32]` | `(typealias {} Weights () ...)` |
| `dim batch` | `(defdim {} batch)` |
| `x |> f |> g` | `(pipe {} x' f' g')` |
| `x : T` | metadata annotation on the desugared Deep node |

### Current-Snapshot Surf Advice

- Use named dimensions such as `n`, `batch`, `hidden`.
- Do not rely on integer literal dimensions in Surf type positions yet.
- All reduction-style calls currently need an explicit axis: `sum(x, 0)`, `softmax(x, 0)`.
- Tensor math expects tensors, not bare `f32`.
  For scalar arithmetic that passes today, use rank-0 tensors such as `tensor[f32]`.

## 4. Type System Rules

What the compiler will check in practice:

- Tensor types are `tensor[dim1, dim2, ..., precision]`.
  Precision is always last.
- `tensor[f32]` is a rank-0 tensor scalar and is the safest scalar-like numeric type in
  the current checker.
- Named dimensions are nominal.
  `tensor[batch, f32]` and `tensor[seq, f32]` do not unify just because they might have
  the same runtime size.
- No implicit broadcasting.
  If two tensor shapes differ, the compiler treats that as an error.
- No implicit precision promotion.
  `tensor[n, f32]` and `tensor[n, bf16]` do not mix silently.
  Use `cast`.
- Function types are written `A -> B -> C` in Surf and `(t-fn {} A' B' C')` in Deep.
- ADTs require exhaustive pattern matches.
- Type aliases are transparent.
- Reductions currently require an explicit integer axis argument.

## 5. Common Patterns

The validated examples in this section are the core few-shot material.
Every `chelis-surf` and `chelis-deep` block below is checked automatically in the repo
test suite.

### 5.1 Rank-0 Tensor Square

Surf:

```chelis-surf
def square(x: tensor[f32]): tensor[f32] = mul(x, x)
```

Deep:

```chelis-deep
(defsig {} square (t-fn {} (t-tensor {} (t-prim {} f32)) (t-tensor {} (t-prim {} f32))))

(def {}
  square
  (fn {}
    (params {} (x {type: (t-tensor {} (t-prim {} f32))}))
    (app {} (var {} mul) (var {} x) (var {} x))))
```

### 5.2 Vector Addition

Surf:

```chelis-surf
def add_vec(x: tensor[n, f32], y: tensor[n, f32]): tensor[n, f32] = add(x, y)
```

Deep:

```chelis-deep
(defsig {}
  add_vec
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  add_vec
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})
      (y {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (app {} (var {} add) (var {} x) (var {} y))))
```

### 5.3 Let-Binding Pipeline

Surf:

```chelis-surf
def twice_then_relu(x: tensor[n, f32]): tensor[n, f32] =
  let y = add(x, x)
  in relu(y)
```

Deep:

```chelis-deep
(defsig {}
  twice_then_relu
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  twice_then_relu
  (fn {}
    (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (let {}
      (bind {} y (app {} (var {} add) (var {} x) (var {} x)))
      (app {} (var {} relu) (var {} y)))))
```

### 5.4 ReLU Then Softmax With Explicit Axis

Surf:

```chelis-surf
def relu_then_softmax(x: tensor[n, f32]): tensor[n, f32] =
  softmax(relu(x), 0)
```

Deep:

```chelis-deep
(defsig {}
  relu_then_softmax
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  relu_then_softmax
  (fn {}
    (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (app {}
      (var {} softmax)
      (app {} (var {} relu) (var {} x))
      (lit {type: (t-prim {} int32)} 0))))
```

### 5.5 Pipe Composition

Surf:

```chelis-surf
def classify(x: tensor[n, f32]): tensor[n, f32] =
  x |> relu |> fn (v) -> softmax(v, 0)
```

Deep:

```chelis-deep
(defsig {}
  classify
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  classify
  (fn {}
    (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (pipe {}
      (var {} x)
      (var {} relu)
      (fn {} (params {} v) (app {} (var {} softmax) (var {} v) (lit {type: (t-prim {} int32)} 0))))))
```

### 5.6 Sigmoid Step

Surf:

```chelis-surf
def logistic_step(x: tensor[n, f32]): tensor[n, f32] =
  sigmoid(x)
```

Deep:

```chelis-deep
(defsig {}
  logistic_step
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  logistic_step
  (fn {}
    (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (app {} (var {} sigmoid) (var {} x))))
```

### 5.7 Elementwise Clamp

Surf:

```chelis-surf
def clamp_low(x: tensor[n, f32], low: tensor[n, f32]): tensor[n, f32] = max_elem(x, low)
```

Deep:

```chelis-deep
(defsig {}
  clamp_low
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  clamp_low
  (fn {}
    (params {}
      (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})
      (low {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (app {} (var {} max_elem) (var {} x) (var {} low))))
```

### 5.8 Dimension-Polymorphic Identity

Surf:

```chelis-surf
def identity[a](x: tensor[a, f32]): tensor[a, f32] = x
```

Deep:

```chelis-deep
(defsig {}
  identity
  (t-fn {}
    (t-tensor {} (d-var {} a) (t-prim {} f32))
    (t-tensor {} (d-var {} a) (t-prim {} f32))))

(def {}
  identity
  (fn {}
    (params {} (x {type: (t-tensor {} (d-var {} a) (t-prim {} f32))}))
    (var {} x)))
```

### 5.9 Transparent Type Alias

Surf:

```chelis-surf
type Weights = tensor[n, f32]

def keep(w: Weights): Weights = w
```

Deep:

```chelis-deep
(typealias {} Weights () (t-tensor {} (d-name {} n) (t-prim {} f32)))

(defsig {} keep (t-fn {} (t-adt {} Weights) (t-adt {} Weights)))

(def {}
  keep
  (fn {}
    (params {} (w {type: (t-adt {} Weights)}))
    (var {} w)))
```

### 5.10 ADT + Pattern Match

Surf:

```chelis-surf
type Optimizer =
  | Sgd { lr: tensor[f32] }
  | Adam { lr: tensor[f32], beta1: tensor[f32], beta2: tensor[f32], eps: tensor[f32] }

def learning_rate(opt: Optimizer): tensor[f32] =
  match opt with {
    | Sgd { lr } => lr
    | Adam { lr, beta1, beta2, eps } => lr
  }
```

Deep:

```chelis-deep
(deftype {}
  Optimizer
  ()
  (variant {} Sgd (field {} lr (t-tensor {} (t-prim {} f32))))
  (variant {}
    Adam
    (field {} beta1 (t-tensor {} (t-prim {} f32)))
    (field {} beta2 (t-tensor {} (t-prim {} f32)))
    (field {} eps (t-tensor {} (t-prim {} f32)))
    (field {} lr (t-tensor {} (t-prim {} f32)))))

(defsig {} learning_rate (t-fn {} (t-adt {} Optimizer) (t-tensor {} (t-prim {} f32))))

(def {}
  learning_rate
  (fn {}
    (params {} (opt {type: (t-adt {} Optimizer)}))
    (match {}
      (var {} opt)
      (arm {} (pat-record {} Sgd (kv {} lr (pat-var {} lr))) () (var {} lr))
      (arm {}
        (pat-record {}
          Adam
          (kv {} beta1 (pat-var {} beta1))
          (kv {} beta2 (pat-var {} beta2))
          (kv {} eps (pat-var {} eps))
          (kv {} lr (pat-var {} lr)))
        ()
        (var {} lr)))))
```

### 5.11 ADT-Driven Activation Choice

Surf:

```chelis-surf
type Activation =
  | Relu
  | Sigmoid

def activate(act: Activation, x: tensor[n, f32]): tensor[n, f32] =
  match act with {
    | Relu => relu(x)
    | Sigmoid => sigmoid(x)
  }
```

Deep:

```chelis-deep
(deftype {} Activation () (variant {} Relu) (variant {} Sigmoid))

(defsig {}
  activate
  (t-fn {}
    (t-adt {} Activation)
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  activate
  (fn {}
    (params {}
      (act {type: (t-adt {} Activation)})
      (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (match {}
      (var {} act)
      (arm {} (pat-ctor {} Relu) () (app {} (var {} relu) (var {} x)))
      (arm {} (pat-ctor {} Sigmoid) () (app {} (var {} sigmoid) (var {} x))))))
```

### 5.12 Current Limit: Do Not Use MLP or MNIST as Your First Prompt

The spec and roadmap include `matmul`, multi-layer networks, and MNIST-scale examples.
The current teaching file intentionally does not validate those yet because the
repository snapshot still has moving parts around shape rules and broader frontend
ergonomics.
Start with the validated patterns above, then grow from compiler feedback.

## 6. Common Mistakes and Fixes

- Writing Deep as `(mul x y)` instead of `(app {} (var {} mul) (var {} x) (var {} y))`.
  Fix: every call uses `app`; every referenced name uses `var`.
- Omitting `{}`.
  Fix: every tagged node needs a metadata map, even when empty.
- Using bare `f32` with tensor built-ins.
  Fix: use `tensor[f32]` for scalar-like numeric values that must pass today.
- Forgetting the axis on reductions.
  Fix: write `sum(x, 0)`, `mean(x, 0)`, `softmax(x, 0)`.
- Mixing dimension names.
  Fix: `add(x, y)` needs matching nominal dims, not just matching intent.
- Mixing precisions.
  Fix: insert `cast`.
- Inventing functions such as `dot`, `tensor_const`, or `cross_entropy`.
  Fix: stay inside the built-in scope listed in this file.
- Reaching for `normalize` because it appears in older examples.
  Fix: treat it as unstable implementation surface until the spec and lowering agree on
  its semantics.
- Writing malformed Deep params.
  Fix: use `(params {} x)` or `(params {} (x {type: ...}))`, never `(params {} (var {} x))`.
- Writing Surf types with literal numeric dimensions.
  Fix: use named dims in Surf for now; reserve `d-lit` for Deep only.

## 7. Compiler Feedback Interpretation

- `fitness = 1.0` and no errors means the program passed the current checker.
- A high fitness score with remaining errors is still a failing program.
  Treat `errors.is_empty()` as the real pass condition.
- `UnboundVariable` means you used a name outside the built-in scope or local bindings.
- `DimensionMismatch` means the dimension names do not unify.
- `PrecisionMismatch` means you mixed precisions without `cast`.
- `ArityMismatch` usually means the current built-in expects a different number of
  arguments.
  For current reduction-style built-ins, the missing argument is often the axis.
- Parse errors in Deep almost always mean one of:
  missing `{}`, missing `app`, missing `var`, or malformed helper nodes such as
  `params` / `bind`.

### Manual Verification Protocol

Use this file in a fresh LLM conversation with no extra Chelis context.
Ask for programs in Surf first and Deep second.
For this repository snapshot, prefer prompts based on the validated examples above:

1. write a rank-0 tensor square
2. write vector addition over `tensor[n, f32]`
3. write `relu` followed by `softmax(..., 0)`
4. write a sigmoid transform over `tensor[n, f32]`
5. write a dimension-polymorphic identity function
6. write a type alias plus a passthrough function
7. write an optimizer ADT and extract the learning rate
8. write an activation ADT and branch between `relu` and `sigmoid`
9. rewrite one of those programs in canonical Deep
10. repair a broken Deep snippet after compiler feedback

Success criterion for the current skill:

- more than half of these prompts should reach `fitness >= 0.9`
- and end with zero checker errors
- within five repair iterations
