# Chelis Language Specification: Type System

**Version:** 0.1.0-draft
**Status:** Authoritative specification draft

---

## 1. Type Universe

Every expression in a Chelis program has a type drawn from the following universe. Types are classified into categories; each category is described with its syntax, its internal representation, and its role in the system.

### 1.1 Base Types (Scalars)

The scalar types are the atoms of the type system. They are not parameterized.

| Type | Category | Size | Description |
|------|----------|------|-------------|
| `i8` | Integer | 8 bits | Signed 8-bit integer |
| `i16` | Integer | 16 bits | Signed 16-bit integer |
| `i32` | Integer | 32 bits | Signed 32-bit integer |
| `i64` | Integer | 64 bits | Signed 64-bit integer |
| `u8` | Integer | 8 bits | Unsigned 8-bit integer |
| `u16` | Integer | 16 bits | Unsigned 16-bit integer |
| `u32` | Integer | 32 bits | Unsigned 32-bit integer |
| `u64` | Integer | 64 bits | Unsigned 64-bit integer |
| `f16` | Float | 16 bits | IEEE 754 half-precision float |
| `bf16` | Float | 16 bits | Brain floating point (8-bit exponent, 7-bit mantissa) |
| `f32` | Float | 32 bits | IEEE 754 single-precision float |
| `f64` | Float | 64 bits | IEEE 754 double-precision float |
| `bool` | Logical | 1 bit | Boolean (`true` / `false`) |
| `unit` | Unit | 0 bits | The unit type; single value `()` |

Scalar types are distinct from tensor types. A bare `f32` is a scalar; `tensor[f32]` is a 0-dimensional tensor. These are different types. Scalars cannot be used where tensors are expected, and vice versa, without explicit construction or extraction.

### 1.2 Tensor Types

Tensor types are the core innovation of Chelis's type system. A tensor type is written:

```
tensor[d1, d2, ..., dn, P]
```

where `d1` through `dn` are **dimension specifiers** and `P` is a **precision type**. The precision is always the last element within the brackets. Everything before it is a dimension.

**Dimension specifiers** come in three forms:

| Form | Example | Meaning |
|------|---------|---------|
| Named (symbolic) | `batch`, `hidden`, `seq_len` | A dimension identified by name; size determined at runtime |
| Concrete (literal) | `784`, `10`, `3` | A dimension with a fixed compile-time size |
| Variable | `d` (as a type variable) | A dimension that is polymorphic; bound by let-polymorphism |

Examples:

```
tensor[784, f32]                     -- 1D, 784 elements, f32
tensor[batch, seq_len, f32]          -- 2D, named dimensions, f32
tensor[batch, seq_len, 512, bf16]    -- 3D, mixed named/concrete, bf16
tensor[3, 3, f64]                    -- 2D, 3x3 matrix, f64
tensor[f32]                          -- 0D scalar tensor, f32
```

**Internal representation.** A tensor type is a pair:

```
TensorType = (DimList, Precision)
DimList    = [Dim]
Dim        = Named(String) | Concrete(Int) | Var(TypeVar)
Precision  = f16 | bf16 | f32 | f64 | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | bool
```

The dimension list is **unordered for compatibility checking** but **ordered for layout**. See Section 2 for the full semantics of named dimensions.

### 1.3 Function Types

Function types are written `A -> B` in Surf. The arrow is right-associative:

```
f32 -> f32                           -- function from f32 to f32
f32 -> f32 -> f32                    -- curried: f32 -> (f32 -> f32)
tensor[n, f32] -> tensor[n, f32]     -- tensor to tensor
(f32, f32) -> f32                    -- takes a tuple, returns f32
```

Functions are curried by default. A multi-parameter `def`:

```
def f(x: f32, y: f32): f32 = x + y
```

has type `f32 -> f32 -> f32`. Application is left-associative: `f a b` is `(f a) b`.

**Internal representation:**

```
FunType = Arrow(Type, Type)
```

Multi-parameter functions are nested arrows: `Arrow(f32, Arrow(f32, f32))`.

### 1.4 Algebraic Data Types (ADTs)

Users define ADTs with the `type` keyword:

```
type Option a = Some a | None
type List a = Cons a (List a) | Nil
type Result a e = Ok a | Err e
type Tree a = Leaf a | Branch (Tree a) (Tree a)
```

An ADT definition introduces:
- A **type constructor** (e.g., `Option`) that takes type arguments and produces a type.
- One or more **data constructors** (e.g., `Some`, `None`) that are values (or functions producing values).

Data constructors are typed as functions:
- `Some : forall a. a -> Option a`
- `None : forall a. Option a`
- `Cons : forall a. a -> List a -> List a`
- `Nil  : forall a. List a`
- `Leaf : forall a. a -> Tree a`
- `Branch : forall a. Tree a -> Tree a -> Tree a`

ADT types are applied by juxtaposition in Surf: `Option f32`, `List (tensor[n, f32])`, `Result f32 String`.

ADTs are **nominal**: two structurally identical ADT definitions with different names produce incompatible types. The name is part of the type's identity.

**Recursive types.** ADTs may be recursive (e.g., `List`, `Tree`). The type checker handles recursive types through the usual nominal recursion: the type name is in scope within its own definition.

### 1.5 Tuple Types

Tuple types are product types of fixed arity:

```
(f32, f32)                          -- pair of f32
(tensor[n, f32], f32, bool)         -- triple
```

Tuples are anonymous and structural (unlike ADTs, which are nominal). Two tuple types are equal if they have the same arity and corresponding element types are equal.

The unit type `unit` is equivalent to the empty tuple `()`.

### 1.6 Type Variables

Type variables are lowercase identifiers: `a`, `b`, `elem`, `dim`. They range over all types (or, in dimension position, over dimensions).

Type variables are **universally quantified at let-bindings** (let-polymorphism). The quantification is implicit:

```
def id(x: a): a = x
-- Inferred type: forall a. a -> a
```

Each use site of a polymorphic binding gets a fresh instantiation of its type variables:

```
let f = id
in (f 42, f true)     -- f is used at type i64 -> i64 and bool -> bool
```

**Internal representation:**

```
TypeScheme = Forall([TypeVar], Type)
TypeVar    = TVar(String, Int)    -- name + unique id
```

### 1.7 The Bottom Type

The bottom type (written `!` or called "never") is the type of computations that do not return. It is a subtype of every type. Currently, Chelis has no surface syntax for diverging computations, but the type checker uses the bottom type internally for exhaustiveness checking and unreachable branches.

---

## 2. Named Dimensions

Named dimensions are the most distinctive feature of Chelis's type system. They provide compile-time safety for tensor operations by identifying axes by name rather than by position.

### 2.1 Dimension Identity

A dimension is identified by its **name**, not its position in the tensor type. The types `tensor[batch, hidden, f32]` and `tensor[hidden, batch, f32]` describe tensors with the **same set of dimensions** but **different memory layouts**.

For the purpose of type compatibility (can these two tensors be added?), dimension identity is checked by name, ignoring order. For the purpose of memory layout and code generation, dimension order determines the physical arrangement of data.

### 2.2 Elementwise Compatibility

Two tensors are compatible for elementwise operations (Add, Mul, Max, CmpLt) if and only if:

1. They have the **same set of dimension names** (order-independent).
2. For any concrete dimensions, the sizes match exactly.
3. They have the **same precision type**.

Examples:

```
tensor[batch, hidden, f32] + tensor[batch, hidden, f32]    -- OK
tensor[batch, hidden, f32] + tensor[hidden, batch, f32]    -- OK (same dim set)
tensor[batch, hidden, f32] + tensor[batch, seq_len, f32]   -- ERROR: hidden != seq_len
tensor[batch, hidden, f32] + tensor[batch, hidden, f64]    -- ERROR: f32 != f64
tensor[batch, hidden, f32] + tensor[batch, f32]            -- ERROR: different number of dims
tensor[3, 3, f32] + tensor[3, 4, f32]                      -- ERROR: concrete sizes differ
```

When two tensors with the same dimension names but different order are combined in an elementwise operation, the result has the dimension order of the **left** operand. The compiler inserts an implicit permutation on the right operand.

### 2.3 No Implicit Broadcasting

Chelis never implicitly broadcasts. If you have a tensor `x: tensor[batch, hidden, f32]` and a bias `b: tensor[hidden, f32]`, you cannot write `x + b` directly. The dimensions do not match: `x` has `{batch, hidden}` but `b` has `{hidden}`.

To broadcast, use the explicit `expand` operation:

```
let b_expanded = expand(b, batch)    -- tensor[batch, hidden, f32]
in x + b_expanded
```

This makes every broadcast visible in the source code. When a dimension mismatch occurs, the error message can point to the exact line and suggest an `expand`, because there is no ambiguity about whether broadcasting was intended.

### 2.4 Dimension Variables and Rest-Dimensions

Generic functions can be polymorphic over dimension names. There are two mechanisms:

**Dimension variables.** A dimension name that appears in a type signature but is not concretely defined acts as a dimension variable. It is universally quantified at the enclosing `def`:

```
def scale(x: tensor[d, f32], s: f32): tensor[d, f32] = x * s
-- Works for any single-dimension tensor
```

**Rest-dimensions (`..d`).** A rest-dimension variable captures zero or more dimensions at once. It is written `..d` (two dots followed by a variable name):

```
def add_bias(x: tensor[..d, features, f32], b: tensor[features, f32]): tensor[..d, features, f32] =
  let b_expanded = expand(b, ..d)
  in x + b_expanded
```

Here `..d` matches any prefix of dimensions. Calling `add_bias` with a `tensor[batch, seq_len, features, f32]` binds `..d` to `[batch, seq_len]`.

Rest-dimension variables must appear at most once per tensor type parameter. They can appear at the beginning or end of a dimension list, but not in the middle between concrete named dimensions.

**Rest-dimension unification.** When `..d` appears in one type and a concrete dimension list appears in the other, the rest-dimension absorbs the extra dimensions:

```
unify_dims([..d, features], [batch, seq_len, features])
    -> {..d := [batch, seq_len]}

unify_dims([..d, k, n], [batch, k, n])
    -> {..d := [batch]}

unify_dims([..d, k, n], [k, n])
    -> {..d := []}
```

### 2.5 Dimension Constraints

Functions can require that two arguments share a dimension name. This is expressed implicitly: if the same dimension name appears in two different parameter types, they must agree.

```
def dot(a: tensor[n, f32], b: tensor[n, f32]): tensor[f32] =
  reduce_sum(a * b)
```

Here `n` appears in both `a` and `b`, so any call to `dot` must provide two tensors with the same dimension name and size.

### 2.6 The Matmul Dimension Rule

Matrix multiplication has a special typing rule. The `matmul` operation requires:

```
matmul(A: tensor[..d, m, k, P], B: tensor[..d, k, n, P]) -> tensor[..d, m, n, P]
```

- The **last two** dimensions of each operand are the matrix dimensions.
- The inner dimensions must share the name `k` (the contraction dimension).
- The outer dimensions `m` and `n` become the result dimensions.
- Any batch dimensions `..d` must match between the two operands.
- The precision `P` must match.

Example:

```
let weights: tensor[hidden, features, f32] = ...
let input: tensor[features, f32] = ...
-- To multiply: we need matmul(tensor[hidden, features, f32], tensor[features, 1, f32])
-- Or use a dedicated matvec operation.
```

---

## 3. Precision Types

### 3.1 No Implicit Promotion

Chelis never implicitly converts between numeric types. This is a hard rule with no exceptions.

```
let a: tensor[n, f32] = ...
let b: tensor[n, f64] = ...
let c = a + b                -- TYPE ERROR: precision_mismatch, f32 != f64
```

To combine tensors of different precisions, use `cast`:

```
let c = cast(a, f64) + b     -- OK: both f64
let d = a + cast(b, f32)     -- OK: both f32
```

This applies to all operations, including:
- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `<`, `>`, `==`
- Reduction: `reduce_sum`, `reduce_max`
- Any function that takes multiple tensor arguments

Further examples of rejected expressions:

```
let x: f32 = 1.0
let y: f64 = 2.0
let z = x + y                -- TYPE ERROR: precision_mismatch, f32 != f64

let a: tensor[n, f32] = ...
let b: tensor[n, bf16] = ...
let c = a + b                -- TYPE ERROR: precision_mismatch, f32 != bf16
```

### 3.2 Precision as Part of the Type

Precision is a compile-time property, not a runtime tag. The type `tensor[batch, hidden, f32]` is a different type from `tensor[batch, hidden, f64]`. There is no "generic numeric tensor" type that abstracts over precision (except through type variables).

```
def scale(x: tensor[d, p], s: p): tensor[d, p] = ...
-- Here p is a type variable; it will be instantiated to a specific precision at each call site
```

### 3.3 The Precision Lattice (Reference Only)

For documentation and tooling purposes, the precision types form an ordering by information content:

```
f64 > f32 > f16
                > bf16
i64 > i32 > i16 > i8
u64 > u32 > u16 > u8
```

The compiler **never** traverses this lattice automatically. It exists solely to:
- Guide repair suggestions ("you might want to cast to f32, which is higher precision")
- Inform the user about potential information loss in explicit casts
- Order overload resolution in future type class extensions

A `cast` from higher to lower precision is permitted but the compiler may emit a warning about potential information loss. A `cast` from lower to higher precision is always silent.

### 3.4 Cast Semantics

The `cast` operation has the following type:

```
cast : tensor[D, P1] -> P2 -> tensor[D, P2]
```

It preserves all dimensions and changes only the precision. The runtime behavior depends on the source and target precisions:

| From | To | Behavior |
|------|----|----------|
| `f32` | `f64` | Exact widening |
| `f64` | `f32` | Round to nearest, ties to even |
| `f32` | `bf16` | Round to nearest, truncate mantissa |
| `f32` | `f16` | Round to nearest, may overflow to inf |
| `i32` | `f32` | Exact if value fits in 24-bit mantissa; rounded otherwise |
| `f32` | `i32` | Truncate toward zero; undefined if out of range |
| `bool` | `i32` | `false` -> 0, `true` -> 1 |
| `i32` | `bool` | 0 -> `false`, nonzero -> `true` |

Casting between integer and float types is always explicit. Casting between signed and unsigned integers of the same width reinterprets the bits.

---

## 4. Hindley-Milner Type Inference

Chelis uses Hindley-Milner (HM) type inference, extended to handle tensor types, named dimensions, and precision types. The algorithm is a variant of Algorithm W.

### 4.1 Overview

The type inference algorithm takes an untyped Deep AST and produces a typed Deep AST where every node is annotated with its inferred type. If type errors are found, inference continues past them (see Section 5, Fitness Scoring) and produces partial type annotations.

### 4.2 Type Environments

A type environment (context) `G` maps variable names to type schemes:

```
G ::= {} | G, x : sigma
sigma ::= forall a1 ... an. tau    -- type scheme (polymorphic)
tau   ::= P                         -- base type (precision)
        | tensor[D, P]              -- tensor type
        | tau -> tau                 -- function type
        | (tau, ..., tau)            -- tuple type
        | T tau1 ... taun            -- ADT application
        | a                          -- type variable
```

The initial environment contains:
- All RISC primitive operations with their types (see spec/05-risc-primitives.md)
- All data constructors from in-scope ADT definitions
- All imported definitions from other modules

### 4.3 Algorithm W

The core algorithm proceeds by structural recursion on expressions, producing a **substitution** (mapping from type variables to types) and a **type** for each expression.

```
W(G, e) -> (S, tau)
```

where `S` is a substitution and `tau` is the inferred type.

**Step 1: Fresh variables.** At the start of inference for each expression, generate fresh type variables as needed.

**Step 2: Recursive inference.** Infer types for subexpressions.

**Step 3: Unification.** Unify types at each application, let-binding, and other combining form. Unification produces a substitution that is composed with the accumulated substitution.

**Step 4: Generalization.** At `let`-bindings, generalize the inferred type by quantifying over type variables that are not free in the environment.

### 4.4 Unification

Unification takes two types and produces a most general substitution that makes them equal, or fails.

**Standard rules:**

```
unify(a, tau)           = {a := tau}  if a not in ftv(tau)    -- variable binding
unify(tau, a)           = {a := tau}  if a not in ftv(tau)    -- symmetric
unify(P, P)             = {}                                   -- same base type
unify(tau1 -> tau2, tau3 -> tau4) = S2 . S1
    where S1 = unify(tau1, tau3), S2 = unify(S1(tau2), S1(tau4))
unify((t1,...,tn), (s1,...,sn)) = Sn . ... . S1
    where Si = unify(S_{i-1}(...(t_i)), S_{i-1}(...(s_i)))
unify(T t1...tn, T s1...sn)    = compose unifications of corresponding args
unify(tau1, tau2)       = FAIL      -- otherwise
```

**Tensor-specific unification:**

```
unify(tensor[D1, P1], tensor[D2, P2]) =
    S1 = unify_precision(P1, P2)
    S2 = unify_dims(S1(D1), S1(D2))
    return S2 . S1
```

Dimension list unification (`unify_dims`) proceeds as follows:

1. Both lists must have the same length (after expanding rest-dimension variables), or unification fails with `dimension_mismatch`.
2. Dimensions are matched by **name** when both are named, not by position. The algorithm collects all named dimensions from both lists into sets, then verifies that the sets are equal.
3. For each pair of matching dimensions:
   - `Named(x)` unifies with `Named(x)` (same name) -> success
   - `Named(x)` unifies with `Named(y)` (different name) -> FAIL (`dimension_mismatch`)
   - `Concrete(n)` unifies with `Concrete(n)` (same size) -> success
   - `Concrete(n)` unifies with `Concrete(m)` (different size) -> FAIL (`dimension_mismatch`)
   - `Var(a)` unifies with any dimension `d` -> `{a := d}`
   - `Named(x)` unifies with `Concrete(n)` -> binds the name to the concrete size (recorded for code generation)

**Rest-dimension unification:**

When a rest-dimension `..d` appears in one type and a concrete dimension list appears in the other, `..d` is bound to the "extra" dimensions:

```
unify_dims([..d, features], [batch, seq_len, features])
    -> {..d := [batch, seq_len]}

unify_dims([..d, m, n], [batch, m, n])
    -> {..d := [batch]}

unify_dims([..d, m, n], [m, n])
    -> {..d := []}
```

The algorithm for rest-dimension matching:
1. Identify the non-rest dimensions in the pattern (those after `..d`).
2. Match those against the tail of the concrete list (by name).
3. Bind `..d` to whatever prefix remains.
4. If the tail does not match, unification fails.

**Occurs check.** Before binding `a := tau`, check that `a` does not appear in `ftv(tau)`. If it does, unification fails with `occurs_check`. This prevents infinite types.

### 4.5 Let-Polymorphism

At a `let` binding:

```
let x = e1 in e2
```

1. Infer the type of `e1`: `(S1, tau1) = W(G, e1)`
2. Generalize `tau1`: `sigma = generalize(S1(G), tau1)` -- quantify over type variables in `tau1` that are not free in `S1(G)`
3. Infer `e2` in the extended environment: `(S2, tau2) = W(S1(G) + {x : sigma}, e2)`
4. Return `(S2 . S1, tau2)`

Generalization at `let` is what gives Chelis its polymorphism. The binding `let id = fn (x) -> x` gets type `forall a. a -> a`, and each use of `id` gets a fresh instantiation.

**Value restriction.** To maintain soundness in the presence of future mutable references or effects, only syntactic values (lambdas, constructors, literals) are generalized. Non-value expressions (function applications, etc.) are not generalized -- their types are monomorphic. This follows the ML value restriction.

### 4.6 Annotated vs. Unannotated Definitions

**Annotated definitions:**

```
def f(x: tensor[n, f32]): tensor[n, f32] = ...
```

The type checker:
1. Parses the annotation into a type `tau_ann`.
2. Infers the body type `tau_body`.
3. Unifies `tau_ann` with `tau_body`. If unification fails, reports a `type_mismatch` error.

**Unannotated definitions:**

```
def f(x) = x * x
```

The type checker infers the type entirely from the body. The inferred type is recorded and reported.

**Partially annotated definitions:**

```
def f(x: f32) = x * x
```

Partial annotations are permitted. The checker uses annotations as constraints and infers the rest.

---

## 5. Fitness Scoring Algorithm

The fitness score is the mechanism by which the Chelis compiler acts as a collaborator rather than a gatekeeper. Every program, no matter how broken, receives a score between 0.0 and 1.0.

### 5.1 Computation

The type checker annotates every AST node with one of three states:

- **Typed**: the node has a fully resolved type.
- **Partially typed**: the node has a type with unresolved variables or error markers (`?`).
- **Untyped**: the node could not be typed at all (e.g., depends on an unbound variable).

The fitness score is:

```
score = typed_nodes / total_nodes
```

where `typed_nodes` counts nodes in the Typed or Partially-typed states, and `total_nodes` counts all AST nodes (expressions, patterns, type annotations).

A fully correct program has `score = 1.0`. A program where nothing can be inferred has `score = 0.0`.

### 5.2 Partial Inference

The key property of the fitness scoring system is that **inference continues past errors**. When the type checker encounters an error:

1. It records the error with full diagnostic information.
2. It assigns a **fresh type variable** (marked as an error variable) to the problematic node.
3. It continues inference for the rest of the program.

This means that if `x` has an error but `y` depends only on correctly-typed things, `y` still gets a type. The fitness score reflects this: fixing the error on `x` will increase the score, because `x` (and any nodes that transitively depend only on `x`) will move from untyped to typed.

Example:

```
let a = unknown_function(1)    -- error: unbound variable
let b = 2 + 3                  -- typed: i64
let c = a + b                  -- partially typed: ? + i64 = ?
let d = b * b                  -- typed: i64
```

Here: `b` and `d` are typed (2 nodes), `c` is partially typed (1 node), `a` is untyped (1 node). Score: 3/4 = 0.75.

### 5.3 Output Format

The `chelis check --json` command produces the following structure:

```json
{
  "score": 0.75,
  "typed_nodes": 42,
  "total_nodes": 50,
  "errors": [
    {
      "span": {
        "file": "model.ch",
        "line": 10,
        "col": 5,
        "end_line": 10,
        "end_col": 30
      },
      "kind": "dimension_mismatch",
      "expected": "tensor[batch, hidden, f32]",
      "actual": "tensor[batch, seq_len, f32]",
      "message": "dimension mismatch: expected 'hidden', found 'seq_len'",
      "context": {
        "operation": "Add",
        "left_type": "tensor[batch, hidden, f32]",
        "right_type": "tensor[batch, seq_len, f32]"
      },
      "repairs": [
        {
          "description": "Did you mean to use 'hidden' instead of 'seq_len'?",
          "action": "replace",
          "span": { "line": 10, "col": 20, "end_line": 10, "end_col": 27 },
          "replacement": "hidden",
          "confidence": 0.8
        }
      ]
    }
  ],
  "partial_types": {
    "x": "tensor[?, hidden, f32]",
    "y": "? -> f32"
  },
  "inferred_types": {
    "add_bias": "tensor[batch, hidden, f32] -> tensor[hidden, f32] -> tensor[batch, hidden, f32]",
    "scale": "tensor[d, f32] -> f32 -> tensor[d, f32]"
  }
}
```

The `partial_types` map uses `?` to indicate type positions that could not be resolved due to errors. The `inferred_types` map contains the final types of all successfully typed top-level definitions.

### 5.4 Score Properties

The fitness score has the following design properties:

1. **Monotonic improvement.** Fixing a type error never decreases the score (assuming the fix doesn't introduce new errors).
2. **Local sensitivity.** A small code change produces a small score change. This makes the score useful as a gradient signal for AI agents performing hill-climbing.
3. **Decomposable.** The score for a module is the weighted average of scores for its definitions, weighted by AST node count. An agent can focus on the lowest-scoring definition.
4. **Deterministic.** The same source code always produces the same score.

### 5.5 Fitness Score for Parse Errors

When the parser fails, the fitness score is:

```
parse_score = (successfully_parsed_bytes / total_bytes) * 0.3
```

The 0.3 multiplier reflects the fact that a file that doesn't parse is far from correct. Within the parsed portion, the type checker runs normally and contributes to the score as described above.

---

## 6. Type Checking Rules

This section gives formal-ish inference rules for each language construct. The notation is:

```
G |- e : tau ~> S
```

meaning "under environment G, expression e has type tau, producing substitution S."

### 6.1 Variables

```
      x : forall a1...an. tau  in  G
      tau' = tau[a1 := fresh_1, ..., an := fresh_n]
      -----------------------------------------------
      G |- x : tau' ~> {}
```

Look up `x` in the environment. If it has a polymorphic type scheme, instantiate all quantified variables with fresh type variables.

### 6.2 Literals

```
      --------------------------------
      G |- integer_literal : a ~> {}        (a is fresh; defaults to i64)

      --------------------------------
      G |- float_literal : a ~> {}          (a is fresh; defaults to f64)

      --------------------------------
      G |- true : bool ~> {}

      --------------------------------
      G |- false : bool ~> {}

      --------------------------------
      G |- () : unit ~> {}
```

Numeric literals are initially assigned fresh type variables. If the variable is never constrained by context, it defaults to `i64` (integers) or `f64` (floats) at the end of inference.

### 6.3 Lambda

```
      a1, ..., an are fresh type variables
      G, x1:a1, ..., xn:an |- body : tau_body ~> S
      ----------------------------------------------
      G |- fn (x1, ..., xn) -> body : S(a1) -> ... -> S(an) -> tau_body ~> S
```

Create fresh type variables for each parameter, add them to the environment, and infer the body. The function type is constructed from the (possibly substituted) parameter types and the body type.

### 6.4 Application

```
      G |- f : tau_f ~> S1
      G' = S1(G)
      G' |- arg : tau_arg ~> S2
      a is fresh
      S3 = unify(S2(tau_f), tau_arg -> a)
      -------------------------------------------------
      G |- f(arg) : S3(a) ~> S3 . S2 . S1
```

Infer the function type, infer the argument type, then unify the function type with `arg_type -> result_type` where `result_type` is a fresh variable.

For multi-argument application `f(a1, a2, ..., an)`, this is applied repeatedly (curried).

### 6.5 Let Binding

```
      G |- e1 : tau1 ~> S1
      sigma = generalize(S1(G), tau1)
      S1(G), x:sigma |- e2 : tau2 ~> S2
      -----------------------------------
      G |- let x = e1 in e2 : tau2 ~> S2 . S1
```

Infer the binding's type, generalize it (quantify free variables not in the environment), then infer the body with the generalized type in scope.

### 6.6 If/Then/Else

```
      G |- cond : tau_c ~> S1
      S2 = unify(S1(tau_c), bool)
      S2(S1(G)) |- then_branch : tau_t ~> S3
      S3(S2(S1(G))) |- else_branch : tau_e ~> S4
      S5 = unify(S4(tau_t), tau_e)
      -------------------------------------------
      G |- if cond then then_branch else else_branch : S5(tau_e) ~> S5.S4.S3.S2.S1
```

The condition must be `bool`. The two branches must have the same type (unified).

### 6.7 Match

```
      G |- scrutinee : tau_s ~> S0

      For each arm  | pattern_i -> body_i:
          (G_i, S_i) = check_pattern(pattern_i, S_{i-1}(tau_s), S_{i-1}(G))
          G_i |- body_i : tau_i ~> S_i'

      S_final = unify(tau_1, tau_2, ..., tau_n)   -- all branches must agree
      Exhaustiveness check: patterns must cover all constructors of tau_s
      -------------------------------------------------------------------
      G |- match scrutinee with arms : S_final(tau_1) ~> S_final . ...
```

Pattern checking introduces bindings: `Some x` against `Option tau` binds `x : tau`. The exhaustiveness check ensures all constructors of the scrutinee's type are covered. Non-exhaustive matches produce an `incomplete_match` error.

### 6.8 Type Annotation

```
      G |- e : tau_inferred ~> S1
      S2 = unify(S1(tau_inferred), tau_annotated)
      -------------------------------------------
      G |- (e : tau_annotated) : S2(tau_annotated) ~> S2 . S1
```

Infer the expression's type, then unify it with the annotation. This either constrains inference or produces a `type_mismatch` error.

### 6.9 Tensor Operations

Tensor operations follow the typing rules of the RISC primitives (see spec/05-risc-primitives.md). In summary:

**Elementwise (binary):**
```
      G |- a : tensor[D, P] ~> S1
      G |- b : tensor[D', P'] ~> S2
      S3 = unify_dims(D, D')
      S4 = unify(P, P')
      ------------------------------------------
      G |- a + b : tensor[S4(S3(D)), S4(S3(P))] ~> S4.S3.S2.S1
```

Both operands must have the same dimensions (by name) and the same precision.

**Elementwise (unary):**
```
      G |- x : tensor[D, P] ~> S1
      ------------------------------------------
      G |- neg(x) : tensor[D, P] ~> S1
```

The result has the same dimensions and precision as the input.

**Reduction:**
```
      G |- x : tensor[D, P] ~> S1
      axis is a dimension name in D
      D' = D \ {axis}            -- D with the named axis removed
      -------------------------------------------
      G |- reduce_sum(x, axis) : tensor[D', P] ~> S1
```

The reduction removes the specified dimension from the type.

**Reshape:**
```
      G |- x : tensor[D1, P] ~> S1
      product(D1) = product(D2)    -- total elements must match
      -------------------------------------------
      G |- reshape(x, D2) : tensor[D2, P] ~> S1
```

**Expand (broadcast):**
```
      G |- x : tensor[D, P] ~> S1
      new_dim is a dimension name not in D
      -------------------------------------------
      G |- expand(x, new_dim) : tensor[D + {new_dim}, P] ~> S1
```

**Permute:**
```
      G |- x : tensor[D, P] ~> S1
      D' is a permutation of D   -- same set of dimension names, different order
      -------------------------------------------
      G |- permute(x, D') : tensor[D', P] ~> S1
```

**Matmul:**
```
      G |- a : tensor[..d, m, k, P] ~> S1
      G |- b : tensor[..d, k, n, P] ~> S2
      S3 = unify_dims(..d_a, ..d_b)
      S4 = unify(k_a, k_b)               -- contraction dimension must match by name
      S5 = unify(P_a, P_b)
      -------------------------------------------
      G |- matmul(a, b) : tensor[..d, m, n, P] ~> S5.S4.S3.S2.S1
```

### 6.10 grad

```
      G |- f : tau_f ~> S1
      tau_f unifies with  tensor[D_in, P] -> tensor[D_out, P]
      (D_out must be empty or scalar -- f must return a scalar or scalar tensor)
      -----------------------------------------------------------------------
      G |- grad(f) : tau_f_with_gradient_return ~> S1
```

The precise return type of `grad(f)` depends on the arity:

- If `f : A -> scalar`, then `grad(f) : A -> (scalar, A)` -- returns (value, gradient).
- If `f : (A, B) -> scalar`, then `grad(f) : (A, B) -> (scalar, (A, B))` -- returns (value, (grad_A, grad_B)).
- If `f : A -> tensor[D, P]` (non-scalar output), then `grad(f) : A -> (tensor[D, P], A)` -- returns (value, gradient), where the gradient is the Jacobian-vector product with an implicit identity cotangent. In practice, `grad` is most commonly used with scalar-output functions.

The input types `A`, `B` must be tensor types or tuples of tensor types. Non-tensor types (bool, ADTs) cannot be differentiated; applying `grad` to a function with non-differentiable inputs produces a `non_differentiable` error.

**`grad` with `wrt` (with-respect-to).** An optional second argument specifies which parameters to differentiate with respect to:

```
grad(f, wrt=[param1, param2])
```

If `wrt` is omitted, the default is all tensor-typed parameters. Non-tensor parameters are skipped and receive unit `()` in the gradient tuple.

### 6.11 cast

```
      G |- x : tensor[D, P1] ~> S1
      P2 is a valid precision type
      -------------------------------------------
      G |- cast(x, P2) : tensor[D, P2] ~> S1
```

Cast preserves dimensions and changes precision. Casting a non-tensor scalar is also valid:

```
      G |- x : P1 ~> S1
      -------------------------------------------
      G |- cast(x, P2) : P2 ~> S1
```

### 6.12 Tuple Construction and Access

```
      G |- e1 : tau1 ~> S1
      ...
      G |- en : taun ~> Sn
      -------------------------------------------
      G |- (e1, ..., en) : (tau1, ..., taun) ~> Sn . ... . S1
```

Tuple element access is done via pattern matching, not via positional indexing.

### 6.13 Constructor Application

```
      Con : forall a1...an. tau1 -> ... -> tauk -> T a1...an   in  G
      G |- e1 : tau1' ~> S1, ..., G |- ek : tauk' ~> Sk
      Si = unify(taui', taui[ai := fresh_i])
      -------------------------------------------
      G |- Con(e1, ..., ek) : T tau1'...taun' ~> Sk . ... . S1
```

Data constructors are treated as regular functions and unified accordingly.

---

## 7. Type Error Categories

Every type error produced by the Chelis compiler falls into one of the following categories. Each category has a unique `kind` string used in the JSON output.

### 7.1 `type_mismatch`

**Trigger:** Expected one type, got another, and they cannot be unified.

**Examples:**
- Using an `i32` where `f32` is expected
- Returning a `tensor[n, f32]` from a function annotated as returning `bool`
- Passing an `Option f32` to a function expecting `List f32`

**Repair strategies:**
- If the types differ only in precision: suggest `cast`
- If the types are structurally similar (e.g., both ADTs): suggest the correct constructor
- If a function returns the wrong type: suggest wrapping in a constructor

### 7.2 `dimension_mismatch`

**Trigger:** Two tensor types should have matching dimensions, but they don't.

**Examples:**
- `tensor[batch, hidden, f32] + tensor[batch, seq_len, f32]` -- `hidden` vs `seq_len`
- `matmul(tensor[m, k, f32], tensor[j, n, f32])` -- `k` vs `j` (contraction dims don't match)
- `tensor[3, f32] + tensor[4, f32]` -- concrete sizes differ

**Repair strategies:**
- If one dimension name is close to another (edit distance 1-2): suggest correction
- If dimensions differ by one: suggest `expand` or `reduce_sum`
- If all dimensions match but in different number: suggest which dimension to add/remove

### 7.3 `precision_mismatch`

**Trigger:** Two tensors in an operation have different precision types.

**Examples:**
- `tensor[n, f32] + tensor[n, f64]`
- `matmul(tensor[m, k, f32], tensor[k, n, bf16])`

**Repair strategies:**
- Suggest `cast(expr, target_precision)` on one operand
- Prefer casting the lower-precision operand up (reference the precision lattice)
- If in a context where lower precision is clearly intended (e.g., loss computation with bf16 weights), suggest casting the higher-precision operand down

### 7.4 `arity_mismatch`

**Trigger:** A function is applied to the wrong number of arguments.

**Examples:**
- `f(1, 2, 3)` where `f` takes 2 arguments
- `g()` where `g` takes 1 argument

**Repair strategies:**
- State the expected and actual arity
- If extra arguments: suggest removing them
- If too few arguments: suggest which argument types are missing

### 7.5 `unbound_variable`

**Trigger:** A name is used that is not in scope.

**Examples:**
- Referencing `foo` when no `foo` is defined
- Typo: `reudce_sum` instead of `reduce_sum`

**Repair strategies:**
- Find names in scope with small edit distance (Levenshtein distance <= 2)
- Suggest imports if the name exists in a known module

### 7.6 `occurs_check`

**Trigger:** Unification would produce an infinite type.

**Examples:**
- `let f = fn (x) -> f(x)` without a type annotation (but note: this specific case is a legitimate recursive function and would need a different treatment; occurs check fires on `a = a -> b`)
- Constructing a type like `a = List a` outside of an ADT definition

**Repair strategies:**
- Suggest adding a type annotation
- If the infinite type resembles a known recursive ADT: suggest using that ADT

### 7.7 `not_a_function`

**Trigger:** Attempting to apply something that is not a function type.

**Examples:**
- `let x = 5 in x(3)` -- applying an integer
- `let t: tensor[n, f32] = ... in t(1)` -- applying a tensor

**Repair strategies:**
- If the expression is a tensor and the "argument" is an index: suggest using a slicing operation
- If the expression has a similar name to a function in scope: suggest the correction

### 7.8 `incomplete_match`

**Trigger:** A `match` expression does not cover all constructors of the scrutinee's type.

**Examples:**
```
type Color = Red | Green | Blue
match c with
  | Red -> 1
  | Green -> 2
-- Missing: Blue
```

**Repair strategies:**
- List the missing constructors
- Suggest adding a wildcard `_` arm (with a note that this may hide bugs)
- Generate skeleton arms for each missing constructor

### 7.9 `non_differentiable`

**Trigger:** Applying `grad` to a function that operates on non-differentiable types.

**Examples:**
- `grad(fn (x: bool) -> ...)` -- bool is not differentiable
- `grad(fn (x: Option f32) -> ...)` -- ADTs are not differentiable

**Repair strategies:**
- Identify which parameters are non-differentiable
- Suggest restructuring to separate differentiable and non-differentiable parts

### 7.10 `duplicate_dimension`

**Trigger:** A tensor type has the same dimension name appearing twice.

**Examples:**
- `tensor[batch, batch, f32]` -- `batch` appears twice

**Repair strategies:**
- Suggest renaming one of the duplicate dimensions

### 7.11 `unknown_dimension_in_reduction`

**Trigger:** A reduction or other axis-specific operation references a dimension name that does not exist in the tensor.

**Examples:**
- `reduce_sum(x, axis=hidden)` where `x: tensor[batch, features, f32]` -- `hidden` is not a dimension of `x`

**Repair strategies:**
- List the available dimension names
- Suggest the closest matching dimension name

---

## 8. Interaction with Other Spec Components

### 8.1 Surf Syntax (spec/02)

The type system operates on Deep AST, not Surf. However, type annotations written in Surf (e.g., `x: tensor[batch, f32]`) are desugared into Deep type expressions (e.g., `(tensor (batch) f32)`) before type checking. The desugaring is purely syntactic and does not affect type semantics.

### 8.2 Deep Syntax (spec/03)

The type checker receives Deep AST and produces typed Deep AST. Every node in the output carries `:type` metadata with the inferred type. The `sig` form in `def` is used as the type annotation; if the sig is `_`, the type is inferred.

### 8.3 RISC Primitives (spec/05)

Each RISC primitive has a defined type signature. The type checker uses these signatures when checking lowered code, and the higher-level operations (which desugar to RISC primitives) inherit their typing rules from the primitives' signatures.

### 8.4 Transformations (spec/06)

The transformations `grad`, `vmap`, and `jit` have type rules defined in this document (Section 6.10 and elsewhere). These rules are enforced during the Check stage; the actual DAG rewriting happens in the Transform stage.

---

## 9. Implementation Notes

### 9.1 Representation of Substitutions

Substitutions should be represented as persistent maps from type variable IDs to types. Composition `S2 . S1` means "apply S1 first, then S2". Substitution application walks the type structure, replacing variables according to the map.

For performance, use union-find for type variables during unification (path compression and union by rank). This keeps unification near-linear in practice.

### 9.2 Error Recovery Strategy

When unification fails:

1. Record the error with both types involved, their locations, and the context (which language construct triggered the unification).
2. Assign a fresh "error" type variable to the problematic node. Error type variables are distinct from normal type variables: they do not unify with anything (they absorb further errors silently) and they propagate the "partially typed" status to dependent nodes.
3. Continue inference.

This strategy ensures that a single error does not cascade into dozens of spurious errors.

### 9.3 Dimension Name Resolution

During type checking, the compiler maintains a dimension name table that tracks:
- Which dimension names are in scope
- Whether a dimension name has a known concrete size
- Which dimension names are linked (must have the same size at runtime)

This table is populated from type annotations and propagated through unification. At code generation time, dimensions without concrete sizes become runtime parameters.

### 9.4 Performance Considerations

For large programs, the type checker should:
- Process definitions in dependency order (topological sort on the call graph)
- Cache instantiations of heavily-used polymorphic functions
- Use level-based generalization (Remy's optimization) to avoid scanning the entire environment during generalization
- Limit the depth of error recovery (if a module has more than N errors, stop inference and report the score based on what was completed)

The recommended limit is N = 100 errors per module. Beyond this, the fitness score is likely close to 0.0 and further inference is unlikely to be productive.
