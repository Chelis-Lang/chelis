# Hull - Executable Language Specification Shell

**Shell name:** Hull (`chelis-lang/hull`)
**Marine rationale:** The hull defines the shape of the vessel. The spec defines the shape of the language.
**Depends on:** `chelis-std` (required). No other shells.
**Status:** Stub (Hull itself is not yet built). Phase 4/5 item. Prerequisite state as of v0.7.19: the Deep parser is shipped (Phase 0b), the `chelis prove` / property-runner infrastructure Hull's generator reuses is shipped (v0.7.1), and the scalar/string foundation the Deep parser needs is shipped. The LaCaDiLE typing rules are stabilizing (the in-repo `proof/lean/LaCaDiLE` mechanization is partial; the full effort is OOPSLA-targeted). The remaining hard gate is freezing the final typing-rule set so Hull's reference checker has a stable target.

---

## 1. What Hull Is

Hull is a self-hosted executable language specification for Chelis. It implements the LaCaDiLE typing rules and operational semantics as Chelis functions operating on ADT representations of the Deep AST. It is a Chelis shell that defines what Chelis programs mean, written in Chelis, type-checked by Chelis.

Hull is Chelis's answer to PLT Redex (Racket's language specification toolkit), but without requiring an external language or runtime. Because Deep is homoiconic (programs are s-expression data), Hull operates directly on the same representation the compiler uses. No bridge, no translation, no serialization between the spec and the implementation.

Hull provides three capabilities:

1. **Reference type checker.** Given a Deep program, derive its type and effects according to the formal rules. Compare against `chelis check`. Disagreements surface spec-implementation bugs.

2. **Reference evaluator.** Given a well-typed Deep program, reduce it step by step according to the operational semantics. Compare against `chelis eval`. Disagreements surface semantic divergence.

3. **Spec-driven test generation.** Generate random well-typed Deep programs by using the typing rules as constraints. Each generated program comes with its expected type from the reference checker. This is conformance testing at the language level.

---

## 2. Core Data Model - The Deep AST as ADTs

The grammar of Chelis's Deep syntax, expressed as Chelis ADTs. Each constructor corresponds to one Deep s-expression form.

```chelis
-- Expressions
type Expr =
  | EVar(String)
  | ELit(Literal)
  | ELam(String, Type, Expr)        -- Deep `(fn {} (params ...) body)`; named ELam for the calculus
  | EApp(Expr, Expr)
  | ELet(String, Expr, Expr)
  | EIf(Expr, Expr, Expr)
  | EAdd(Expr, Expr)
  | EMul(Expr, Expr)
  | ESub(Expr, Expr)
  | EDiv(Expr, Expr)
  | ENeg(Expr)
  | EExp(Expr)
  | ELog(Expr)
  | ESqrt(Expr)
  | ESin(Expr)
  | ECast(Expr, Type)               -- Deep `(cast {} expr target-type)`; target is a full type
  | ESum(Expr, int64)             -- positional axis; int64 is model storage, Deep uses int32
  | EGather(Expr, Expr, int64)
  | EScatter(Expr, Expr, Expr, ScatterMode)
  | EMatmul(Expr, Expr)
  | EWhere(Expr, Expr, Expr)
  | EConcat(List[Expr], int64)
  | EReshape(Expr, List[Dim])      -- Deep `(app {} (var {} reshape) tensor shape-list)`
  | EPermute(Expr, List[int64])    -- Deep `(app {} (var {} permute) tensor axis0 axis1 ...)`; full permutation. NOTE: spec/05 [05-DIM-1] (2026-08-03) classifies permutation entries as axis-domain int32; this int64 encoding disagrees and needs reconciling when hull is built.
  | EExpand(Expr, int64, Dim)      -- Deep `(app {} (var {} expand) tensor axis size)`; the shipped `expand` is a (tensor, axis, size) triop, not a shape-list op. `axis` is a position index (int64 here; spec/05 [05-DIM-1] classifies rank indices as int32 — same reconciliation as EPermute); `size` is the new dimension (`Dim`: literal size is `DLit`, symbolic-dim-name size is `DName`)
  | ECumsum(Expr, int64)
  | ESort(Expr, int64)
  | EGrad(Expr)
  | EVmap(Expr, int64)
  -- Effect-handling forms. The shipped Deep grammar has NO `with-seed` /
  -- `with-handler` tags; the real form is `(handle-effect {effect: name} arg body)`
  -- (spec/03-deep-syntax.md §2.3, "Phase 2a effect handler block"). EWithSeed is
  -- retained as the calculus name for the Random-discharging special case.
  -- BOTH are OUTSIDE the v0.1.0 supported fragment: the parser may build them, but
  -- `type_check` returns `None` for them in v0.1.0 (effect handling lands in a later
  -- phase). See §3 "v0.1.0 supported fragment".
  | EWithSeed(int64, Expr)              -- discharges Random; v0.1.0: parsed, not checked
  | EHandleEffect(Effect, Expr, Expr)   -- `(handle-effect {effect: name} arg body)`; v0.1.0: parsed, not checked
  | EMatch(Expr, List[MatchArm])
  | ETuple(List[Expr])
  | ETupleGet(Expr, int64)
  | EConstruct(String, List[Expr])

-- Literals
type Literal =
  | LInt(int64)
  | LFloat(f32)
  | LBool(bool)
  | LString(String)

-- Types
type Type =
  | TF32
  | TInt64
  | TBool
  | TString
  | TTensor(List[Dim], ElemType)
  | TArrow(Type, Type, EffectRow)
  | TForall(String, Type)
  | TTuple(List[Type])
  | TADT(String, List[Type])
  | TList(Type)
  | TDict(Type, Type)
  | TOption(Type)

-- Element types for tensors
type ElemType = EF32 | EInt64 | EBool

-- Dimensions
type Dim =
  | DName(String)           -- named dimension: batch, hidden, etc.
  | DLit(int64)             -- literal dimension: 3, 784, etc.
  | DVar(String)            -- dimension variable (for polymorphism)

-- Effects -- mirror the shipped `Effect` enum at
-- crates/chelis-types/src/types.rs (Random, Accum, Io, Test, Resource(String)).
-- The stub must track the real taxonomy, not an invented one.
type Effect =
  | Random
  | Accum
  | Io
  | Test
  | Resource(String)
-- `Network` and `Filesystem` are planned additions
-- (see effect_taxonomy_expansion.md), not yet shipped.

-- Effect rows
type EffectRow = List[Effect]

-- Scatter modes
type ScatterMode = ScatterAdd | ScatterReplace

-- Pattern match arms
type MatchArm = Arm(Pattern, Expr)

type Pattern =
  | PVar(String)
  | PLit(Literal)
  | PConstruct(String, List[Pattern])
  | PWildcard

-- Typing context
type Ctx = List[(String, Type)]
```

This ADT set is the grammar. Every valid Deep program parses into a value of type `Expr`. The ADT is the spec's definition of "what programs exist."

**Calculus names vs. Deep tags.** The LaCaDiLE proof calculus uses names like `tlt`
(typed less-than), `btrue`/`bfalse` (boolean literals), and `ite` (if-then-else). Those
are *proof-calculus* names; they are **not** Deep tags and they do **not** add new `Expr`
constructors here. In shipped Deep:

- comparisons are builtin *applications* — `tlt` is `EApp(EVar("cmplt"), ...)` (the
  `cmplt` / `CmpLt` builtin, see `crates/chelis-ir/src/grad.rs`), not a dedicated tag;
- boolean literals are `ELit(LBool(true))` / `ELit(LBool(false))`, not `btrue`/`bfalse`;
- conditionals are `EIf` (Deep `(if {} cond then else)`), not `ite`.

So no comparison or boolean `Expr` constructor is introduced: the calculus-level names map
onto existing `EApp` / `ELit` / `EIf` forms.

---

## 3. Reference Type Checker - `Hull.Typing`

Each function implements one or more typing rules from the LaCaDiLE paper. The function names correspond to rule names. Comments cite the paper section and figure.

```chelis
-- Main entry point: type-check an expression in a context
-- Returns Some((type, effects)) if well-typed, None if ill-typed
def type_check(ctx: Ctx, e: Expr) -> Option[(Type, EffectRow)] =
  match e {
    -- T-Var (LaCaDiLE Section 3, Figure 4)
    EVar(x) -> lookup(ctx, x) |> map(fn(t) -> (t, []))

    -- T-Lit
    ELit(lit) -> Some((type_of_lit(lit), []))

    -- T-Lam (LaCaDiLE Section 3, Figure 4)
    ELam(x, t_arg, body) -> {
      (t_ret, effs) = type_check(extend(ctx, x, t_arg), body)?
      Some((TArrow(t_arg, t_ret, effs), []))
    }

    -- T-App (LaCaDiLE Section 3, Figure 4)
    EApp(e1, e2) -> {
      (t1, effs1) = type_check(ctx, e1)?
      match t1 {
        TArrow(t_arg, t_ret, effs_f) -> {
          (t2, effs2) = type_check(ctx, e2)?
          if types_unify(t_arg, t2)
            then Some((t_ret, merge_effects([effs_f, effs1, effs2])))
            else None
        }
        _ -> None
      }
    }

    -- T-Let
    ELet(x, e_bound, e_body) -> {
      (t_bound, effs1) = type_check(ctx, e_bound)?
      (t_body, effs2) = type_check(extend(ctx, x, t_bound), e_body)?
      Some((t_body, merge_effects([effs1, effs2])))
    }

    -- T-If
    EIf(cond, then_br, else_br) -> {
      (t_cond, effs_c) = type_check(ctx, cond)?
      if not(types_equal(t_cond, TBool)) then None else {
        (t_then, effs_t) = type_check(ctx, then_br)?
        (t_else, effs_e) = type_check(ctx, else_br)?
        if types_unify(t_then, t_else)
          then Some((t_then, merge_effects([effs_c, effs_t, effs_e])))
          else None
      }
    }

    -- T-Add (LaCaDiLE Section 3, Figure 4)
    -- Requires dimension equality on both operands
    EAdd(e1, e2) -> check_binary_tensor_op(ctx, e1, e2)

    -- T-Mul (LaCaDiLE Section 3, Figure 4)
    EMul(e1, e2) -> check_binary_tensor_op(ctx, e1, e2)

    -- T-Sub, T-Div - same rule structure as T-Add
    ESub(e1, e2) -> check_binary_tensor_op(ctx, e1, e2)
    EDiv(e1, e2) -> check_binary_tensor_op(ctx, e1, e2)

    -- T-Neg, T-Exp, T-Log, T-Sqrt, T-Sin - unary elementwise
    ENeg(e1) -> check_unary_tensor_op(ctx, e1)
    EExp(e1) -> check_unary_tensor_op(ctx, e1)
    ELog(e1) -> check_unary_tensor_op(ctx, e1)
    ESqrt(e1) -> check_unary_tensor_op(ctx, e1)
    ESin(e1) -> check_unary_tensor_op(ctx, e1)

    -- T-Sum: remove exactly one position, not every equal dimension.
    ESum(e1, axis) -> {
      (t1, effs) = type_check(ctx, e1)?
      match t1 {
        TTensor(dims, elem) ->
          k = if axis < 0 then length(dims) + axis else axis
          if is_int32(axis) and 0 <= k and k < length(dims) and numeric(elem)
            then Some((TTensor(remove_at(dims, k), elem), effs))
            else None
        _ -> None
      }
    }

    -- T-Gather (LaCaDiLE Section 3)
    -- gather(input, indices, axis) -> output
    -- input: tensor[..., n, ..., T]
    -- indices: tensor[..., k, ..., int64]
    -- output: tensor[..., k, ..., T]  (n replaced by k at axis position)
    EGather(input, indices, axis) -> {
      (t_in, effs1) = type_check(ctx, input)?
      (t_idx, effs2) = type_check(ctx, indices)?
      match (t_in, t_idx) {
        (TTensor(dims_in, elem), TTensor(dims_idx, EInt64)) -> {
          -- Replace dimension at axis position with the index dimension
          new_dims = replace_at(dims_in, axis, get_at(dims_idx, axis))
          Some((TTensor(new_dims, elem), merge_effects([effs1, effs2])))
        }
        _ -> None
      }
    }

    -- T-Matmul
    -- matmul(a, b) where a: tensor[..., m, k, T] and b: tensor[..., k, n, T]
    -- result: tensor[..., m, n, T]
    EMatmul(e1, e2) -> {
      (t1, effs1) = type_check(ctx, e1)?
      (t2, effs2) = type_check(ctx, e2)?
      match (t1, t2) {
        (TTensor(dims1, elem1), TTensor(dims2, elem2)) -> {
          if not(elem_equal(elem1, elem2)) then None
          else {
            -- Last dim of first must equal second-to-last dim of second
            k1 = last(dims1)
            k2 = second_last(dims2)
            if not(dims_unify(k1, k2)) then None
            else {
              result_dims = concat([drop_last(dims1), [last(dims2)]])
              Some((TTensor(result_dims, elem1), merge_effects([effs1, effs2])))
            }
          }
        }
        _ -> None
      }
    }

    -- T-Reshape
    -- Mirrors the shipped checker's `infer_reshape_app`
    -- (`crates/chelis-types/src/infer.rs`, dispatched from the `reshape` builtin).
    -- reshape(e, new_dims): e must be a tensor; new_dims is a value-level shape list.
    -- The shape list elements are int64 (the shipped path unifies the list against
    -- List[Int64]; an int32 shape element is a PrecisionMismatch there). The output
    -- element type is INVARIANT (precision is copied unchanged from the input). The
    -- output dims are rebuilt element-by-element from new_dims (lit/cast -> DLit, a
    -- shape(input, k) reference -> the input's dim at axis k, otherwise DVar/wildcard).
    -- NO element-count or product-of-dims guard at the type level: a rank/size change
    -- that does not preserve element count still type-checks here (it is a runtime/IR
    -- concern, not a type-level one). Effects pass through unchanged.
    EReshape(e, new_dims) -> {
      (t, effs) = type_check(ctx, e)?
      match t {
        TTensor(_, elem) ->
          Some((TTensor(map(new_dims, dim_of_shape_elem), elem), effs))
        _ -> None
      }
    }

    -- T-Permute
    -- Mirrors the shipped checker's `infer_permute_app`
    -- (`crates/chelis-types/src/infer.rs`, dispatched from the `permute` builtin).
    -- permute(e, perm): e must be a tensor; perm must be a FULL permutation of the
    -- tensor's axes -- length(perm) == rank, every axis in 0..rank (negative rejected),
    -- and every axis unique. A wrong length is an ArityMismatch, out-of-bounds or
    -- duplicate axes are DimensionMismatch. The output dims are gathered in perm order
    -- (out[i] = dims[perm[i]]); element type is INVARIANT. Effects pass through.
    EPermute(e, perm) -> {
      (t, effs) = type_check(ctx, e)?
      match t {
        TTensor(dims, elem) ->
          if length(perm) == length(dims)
             and all(perm, fn(a) -> a >= 0 and a < length(dims))
             and all_unique(perm)
            then Some((TTensor(map(perm, fn(a) -> get_at(dims, a)), elem), effs))
            else None
        _ -> None
      }
    }

    -- T-Expand: spec/04 section 4.7.2, spec/05 section 2.4.1.
    -- Same-rank replacement of one unit axis; insertion is a distinct operation.
    -- A symbolic operand extent needs the concrete unit-extent guard at evaluation.
    EExpand(e, axis, size) -> {
      (t, effs) = type_check(ctx, e)?
      match t {
        TTensor(dims, elem) ->
          if not is_int32(axis) or axis < 0 or axis >= length(dims) then None
          else if is_literal(dims[axis]) and dim_lit(dims[axis]) != 1 then None
          else if not (is_named(size) or (is_literal(size) and dim_lit(size) >= 0)) then None
          else Some((TTensor(replace_at(dims, axis, size), elem), effs))
        _ -> None
      }
    }

    -- T-Grad (LaCaDiLE Section 3, Figure 4)
    -- grad(f) where f : tensor[dims, T] -> tensor[[], T] ! {}
    -- result : tensor[dims, T] -> tensor[dims, T] ! {}
    -- Requirements: f must be pure (no effects), return scalar
    EGrad(e1) -> {
      (t1, effs) = type_check(ctx, e1)?
      match t1 {
        TArrow(t_in, t_out, effs_f) -> {
          -- Function must be pure
          if not(is_empty(effs_f)) then None
          -- Output must be scalar tensor
          else match t_out {
            TTensor([], _) ->
              Some((TArrow(t_in, t_in, []), effs))
            _ -> None
          }
        }
        _ -> None
      }
    }

    -- T-WithSeed (LaCaDiLE Section 3) -- discharges Random from e's effects.
    -- OUT of the v0.1.0 supported fragment (§3.1): in v0.1.0 this arm is `EWithSeed(_, _)
    -- -> None`. The rule below is the calculus-level semantics that lands once effect
    -- handling is frozen (a later version). Same for the EHandleEffect arm.
    EWithSeed(seed, body) -> {
      (t_body, effs) = type_check(ctx, body)?
      Some((t_body, remove_effect(effs, Random)))
    }

    -- T-Tuple
    ETuple(exprs) -> {
      results = map_option(exprs, fn(ei) -> type_check(ctx, ei))
      results? |> fn(typed) -> {
        types = map(typed, fn(pair) -> pair.0)
        all_effs = flat_map(typed, fn(pair) -> pair.1)
        Some((TTuple(types), all_effs))
      }
    }

    -- T-Match
    EMatch(scrut, arms) -> {
      (t_scrut, effs_s) = type_check(ctx, scrut)?
      -- Check each arm: pattern must match scrutinee type,
      -- body type must be consistent across arms
      check_match_arms(ctx, t_scrut, arms, effs_s)
    }

    _ -> None  -- Unhandled forms
  }


-- Helper: check a binary elementwise tensor operation
-- Both operands must have the same tensor type (dims + elem type)
def check_binary_tensor_op(ctx: Ctx, e1: Expr, e2: Expr) -> Option[(Type, EffectRow)] = {
  (t1, effs1) = type_check(ctx, e1)?
  (t2, effs2) = type_check(ctx, e2)?
  match (t1, t2) {
    (TTensor(dims1, elem1), TTensor(dims2, elem2)) ->
      if dims_equal(dims1, dims2) and elem_equal(elem1, elem2)
        then Some((TTensor(dims1, elem1), merge_effects([effs1, effs2])))
        else None
    -- Scalar overloads
    (TF32, TF32) -> Some((TF32, merge_effects([effs1, effs2])))
    (TInt64, TInt64) -> Some((TInt64, merge_effects([effs1, effs2])))
    _ -> None
  }
}


-- Helper: check a unary elementwise tensor operation
def check_unary_tensor_op(ctx: Ctx, e1: Expr) -> Option[(Type, EffectRow)] = {
  (t1, effs) = type_check(ctx, e1)?
  match t1 {
    TTensor(dims, elem) -> Some((TTensor(dims, elem), effs))
    TF32 -> Some((TF32, effs))
    _ -> None
  }
}
```

Each pattern-match arm in `type_check` implements exactly one typing rule. A reviewer can read the function and check it against the paper's Figure 4 line by line. The code IS the specification.

### 3.1 v0.1.0 supported fragment

Hull v0.1.0 does not implement the entire LaCaDiLE calculus. It pins a concrete
**supported fragment** so differential testing has a stable, honest target:

> The v0.1.0 supported fragment = the `AdjointSupported` boundary (the set of RISC
> primitives that have an adjoint rule in `crates/chelis-ir/src/grad.rs`) **plus** the
> non-AD constructs needed to build, bind, and reduce programs that exercise it: `EVar`,
> `ELit`, `ELam`/`EApp`, `ELet`, `EIf`, the elementwise/reduction tensor ops with
> adjoints (`EAdd`, `EMul`, `ESub`, `EDiv`, `ENeg`, `EExp`, `ELog`, `ESqrt`, `ESin`,
> `ESum`, `EMatmul`, `EGather`), the **shape/movement ops with adjoints — exactly
> `EReshape`, `EPermute`, and `EExpand`** (these three are the movement primitives that
> carry adjoint rules and so fall inside the `AdjointSupported` boundary), `ECast` (scalar
> precision only, see below), `ETuple`/`ETupleGet`, `EMatch`, and `EGrad`.
> (`EConstruct` is **not** in the v0.1.0 fragment — see the scope-out below.)

The shape/movement set is closed at those three; the "etc." in earlier drafts is narrowed
here. The other movement-shaped `Expr` constructors are **explicitly scoped out of
v0.1.0**, each for a concrete reason that follows the same `AdjointSupported` boundary the
fragment is defined by:

- **`EConstruct` / `PConstruct` (the ADT constructor forms)** are deferred to **v0.2.0**.
  The v0.1.0 grammar is `Expr`-only: there is no declaration layer in Hull from which to
  source the user constructor signatures (`type Foo = Ctor(...)`) that `EConstruct` /
  `PConstruct` checking would need. Until a declaration layer lands there is no signature
  to check a constructor application against, so these forms are out of the v0.1.0
  fragment. (They appear in the §2 ADT and the evaluator's `is_value`, but `type_check`
  returns `None` for them in v0.1.0.)
- **`EConcat`, `EWhere`, `ECumsum`, `ESort`, `EScatter`, and `EVmap`** are out of the
  fragment because they sit **outside the `AdjointSupported` boundary** — none of them has
  an adjoint rule, and the AD path fails closed on every one of them:
  - `EConcat` / `EWhere` / `ECumsum` / `ESort` lower as host / non-DAG builtins (no
    `RiscOp::Concat` / `Where` / `Cumsum` / `Sort` exists in `crates/chelis-ir/src/dag.rs`),
    so they never become differentiable IR nodes and `grad_dag` returns `None` for them.
  - `EScatter` lowers to `RiscOp::Scatter`, which *does* exist, but is **explicitly
    fail-closed for AD**: `crates/chelis-ir/src/grad.rs` rejects `scatter_replace` with
    `NotSupported { reason: NonDeterministicAtDuplicateIndices }` because the forward
    result depends on iteration order at duplicate target indices (no well-defined
    adjoint; `spec/05-risc-primitives.md` §3.5).
  - `EVmap` is a vectorization **transform** over a function, not a tensor primitive with
    an adjoint; it has no `RiscOp` and no `grad.rs` arm (it is `lower_unsupported` /
    `lower_unrepresentable("vmap")` in `crates/chelis-ir/src/lower.rs`).

  Because `§3.1` defines the fragment *by* the `AdjointSupported` boundary, these
  zero-adjoint / fail-closed ops are out of the v0.1.0 fragment by construction:
  `type_check` returns `None` for each.

`AdjointSupported` is **not** a named symbol in the repo; it is the operational boundary
established by `crates/chelis-ir/src/grad.rs`, which carries one adjoint arm per
differentiable `RiscOp` (`Add`, `Mul`, `Div`, `Neg`, `Recip`, `Exp`, `Log`, `Sin`, `Cos`,
`Sqrt`, `Tan`, `Atan`, `Abs`, `MaxElem`, `CmpLt`, …) and explicitly *rejects*
non-differentiable ops with structural reasons — `Argmax`/`Argmin` (integer-index
output), `Floor`/`Ceil` (piecewise constant), and `Scatter` (non-deterministic at
duplicate indices). Hull v0.1.0 tracks exactly this differentiable set as the AD-reachable
core.

Out of the v0.1.0 fragment (parsed by the Deep parser, but `type_check` returns `None`):
`EWithSeed` and `EHandleEffect` (effect handling is a later phase); the zero-adjoint /
fail-closed ops `EConcat`, `EWhere`, `ECumsum`, `ESort`, `EScatter`, and `EVmap`
(enumerated with their structural reasons above); `EConstruct` / `PConstruct` (no
declaration layer, deferred to v0.2.0, above); and any construct whose checking depends on
the LaCaDiLE linearity / Δ-capability judgment (deferred to v0.2.0, see §3.3).

### 3.2 `ECast` is scalar-precision-only in v0.1.0

Although `ECast(Expr, Type)` carries a full target `Type` (matching shipped Deep
`(cast {} expr target-type)`), the v0.1.0 reference checker admits **only scalar precision
casts** — float-to-float / int-to-int precision changes on a scalar or rank-0 operand,
mirroring the shipped checker's `cast` validation
(`crates/chelis-types/src/infer.rs`, which rejects unsupported target precisions and
restricts the cast surface). Tensor-shape or structural casts are out of the v0.1.0
fragment.

### 3.3 EGrad is a surface check only in v0.1.0 (dominant CompilerTooConservative source)

The v0.1.0 `EGrad` rule is deliberately a **surface check**. It accepts `grad(f)` when:

1. `f` is a function (`TArrow(...)`) whose effect row is empty (pure), **and**
2. `f`'s return type is a scalar floating value — `TF32`, or a rank-0 float tensor
   `TTensor([], EF32)`.

This mirrors the shipped checker's `infer_grad` / `grad_output_supported`
(`crates/chelis-types/src/infer.rs`): `grad_output_supported` accepts exactly
`Prim` float or `Tensor(dims, prim)` with `dims.is_empty() && prim.is_float()`, and
otherwise emits `grad requires a scalar floating output`.

What v0.1.0 EGrad does **not** do: it does not run the LaCaDiLE **linearity /
Δ-capability** judgment over the body to confirm every primitive on the
differentiation path is adjoint-supported. That deeper check is **deferred to v0.2.0**.
The consequence is explicit and expected: **EGrad is the dominant known
`CompilerTooConservative` source.** The shipped compiler performs the full AD-reachability
analysis at lowering (`grad.rs` rejects `grad` over a body that touches `Argmax`,
`Floor`, `Scatter`, etc.), so for programs whose grad body contains a non-differentiable
op the *compiler* rejects while Hull's surface-only rule *accepts* — i.e. Hull
under-rejects relative to the compiler. These cases are catalogued via the
`corpus/known_conservative.json` whitelist (§11) rather than being treated as soundness
findings.

Note on effect representation: the shipped `Type::Fn(Vec<Type>, Box<Type>)`
(`crates/chelis-types/src/types.rs`) does **not** carry an effect row in the type itself;
effects are tracked as `effects` metadata on `fn` nodes. Hull's `TArrow(Type, Type,
EffectRow)` is a richer model. The v0.1.0 purity component of the EGrad rule is therefore
part of Hull's reference model and is itself a potential `CompilerTooConservative` source
where the compiler's effect tracking and Hull's diverge; such divergences are whitelisted,
not flagged unsound.

---

## 4. Reference Evaluator - `Hull.Eval`

Small-step operational semantics. The `step` function takes an expression and returns either the next expression (one reduction step) or None (the expression is a value or stuck).

```chelis
-- Values: expressions that cannot be reduced further
def is_value(e: Expr) -> bool =
  match e {
    ELam(_, _, _) -> true
    ELit(_) -> true
    ETuple(es) -> all(es, is_value)
    EConstruct(_, es) -> all(es, is_value)
    _ -> false
  }


-- One step of reduction
def step(e: Expr) -> Option[Expr] =
  match e {
    -- Beta reduction: an ELam (Deep `fn`) applied to a value
    EApp(ELam(x, _, body), v) ->
      if is_value(v) then Some(substitute(body, x, v))
      else None

    -- Reduce the function position first
    EApp(e1, e2) ->
      if not(is_value(e1))
        then step(e1) |> map(fn(e1p) -> EApp(e1p, e2))
      else if not(is_value(e2))
        then step(e2) |> map(fn(e2p) -> EApp(e1, e2p))
      else None

    -- Let: evaluate the binding, then substitute
    ELet(x, e_bound, body) ->
      if is_value(e_bound)
        then Some(substitute(body, x, e_bound))
        else step(e_bound) |> map(fn(ebp) -> ELet(x, ebp, body))

    -- If: evaluate condition, then pick branch
    EIf(ELit(LBool(true)), then_br, _) -> Some(then_br)
    EIf(ELit(LBool(false)), _, else_br) -> Some(else_br)
    EIf(cond, t, f) ->
      step(cond) |> map(fn(cp) -> EIf(cp, t, f))

    -- Arithmetic on literal values
    EAdd(ELit(LFloat(a)), ELit(LFloat(b))) -> Some(ELit(LFloat(a + b)))
    EAdd(ELit(LInt(a)), ELit(LInt(b))) -> Some(ELit(LInt(a + b)))
    EAdd(e1, e2) -> step_binary(e1, e2, fn(a, b) -> EAdd(a, b))

    EMul(ELit(LFloat(a)), ELit(LFloat(b))) -> Some(ELit(LFloat(a * b)))
    EMul(e1, e2) -> step_binary(e1, e2, fn(a, b) -> EMul(a, b))

    -- Tensor operations on tensor values would go here.
    -- For the reference evaluator, tensors are represented as
    -- nested lists of scalars, and operations compute element-by-element.
    -- This is intentionally slow - correctness, not performance.

    -- with-seed: when the body is a value, strip the handler
    EWithSeed(_, v) -> if is_value(v) then Some(v) else {
      step(v) |> map(fn(vp) -> EWithSeed(_, vp))
    }

    -- Tuple: step the first non-value element
    ETuple(es) -> step_in_list(es) |> map(fn(esp) -> ETuple(esp))

    -- Match: evaluate scrutinee, then match
    EMatch(scrut, arms) ->
      if is_value(scrut)
        then match_arms(scrut, arms)
        else step(scrut) |> map(fn(sp) -> EMatch(sp, arms))

    _ -> None  -- value or stuck
  }


-- Helper: step inside a binary expression (left-to-right evaluation)
def step_binary(e1: Expr, e2: Expr, rebuild: Expr -> Expr -> Expr) -> Option[Expr] =
  if not(is_value(e1))
    then step(e1) |> map(fn(e1p) -> rebuild(e1p, e2))
  else if not(is_value(e2))
    then step(e2) |> map(fn(e2p) -> rebuild(e1, e2))
  else None


-- Multi-step evaluation to a value (or stuck)
def eval_to_value(e: Expr, max_steps: int64) -> (Expr, int64) = {
  if max_steps <= 0 then (e, 0)
  else match step(e) {
    Some(ep) -> eval_to_value(ep, max_steps - 1)
    None -> (e, max_steps)
  }
}
```

The evaluator is intentionally simple and slow. Tensors are nested lists of scalars. Operations are element-by-element loops. This is the reference semantics - what programs MEAN - not a practical execution engine. The real compiler's evaluator and code generators must agree with this reference on every well-typed program.

### 4.1 Pinned evaluator decisions for v0.1.0

**Independent directional reference (2026-09-09; implementation pending).**
Add a separate Hull numerical kernel over the existing locally nameless `Term`
and an explicit environment of primal/direction pairs. Its first profile admits
f32 scalar/tensor values, bound references, strict lets, add, multiply, positional
sum and same-rank unit-axis expand. A let alias models numerical sharing; it
does not model an owner allocation. This kernel does not evaluate `TGrad`, call
Chelis's AD implementation or LaCaDiLE's generated reverse evaluator, or change
the ordinary reference evaluator/checker/generator fragment.

Each successful result contains the complete primal and directional f32 buffers
and their ordered shape. Validate every supplied pair's equal shape, nonnegative
extents, exact bounded cardinality and finite elements before use, including
unused environment entries. Constants have zero direction. Add acts on both
components; multiplication uses `(x*y, dx*y + x*dy)` with both consumer ports.
Sum and expand apply their existing validated coordinate operation to each
component. Lets extend both environments together, respecting de Bruijn scope.
Invalid references, shapes or buffers, unsupported terms, nonfinite numerical
results and exhausted traversal depth must be distinct non-success outcomes.
The caller still bounds total work/memory; this is not a hardened service.

This is a finite-f32 execution of the formal directional rules, not the
derivative of a rounded machine function or an IEEE/real agreement theorem.
Acceptance tests use bounded exactly representable examples: identity, constants,
`x*x+x`, two independent inputs, unused inputs, sharing and nested/shadowed lets,
coordinate-distinct rectangular and repeated axes, empty/singleton sum and
inner/middle expand. Check primals against ordinary Hull value evaluation and
directions against hand calculations. Mutation tests must detect a missing
multiplication port, zero direction and wrong axis/coordinate mapping. Run the
package build and complete suite; existing campaign obligations remain intact.
Compiler-gradient comparison, dtype/bit transport, fixed randomness, Resource,
ownership checking, arbitrary calls/control flow and exact integers are separate
integration obligations, not completed or silently skipped by this first kernel.

**Independent fixed-path reference (2026-09-10; implementation obligation).**
Extend the separate directional reference, not the ordinary checker or evaluator,
with a bounded interpreter of Hull's named `Expr` source. Reuse the existing
validated f32 buffer operations. Admit the pure directional fragment above,
strict named lets, literal seed handlers, fixed literal f32 dropout rates and
literal Resource handlers. The initial environment contains named complete
primal/direction pairs; reject duplicate names and validate even unused pairs.
Ordinary lexical shadowing of tensor variables is supported. Builtin shadowing,
arbitrary calls, runtime rates/seeds, rate differentiation, general control flow,
other handlers and `EGrad` execution are explicitly outside this profile, not
compiler rejections. The calling harness must retain and validate source dtype
and syntax information before any lossy ordinary `Expr` parsing; an `LFloat`
alone cannot establish that the original source declared f32.

The reference derives Random state and masks independently from source order.
Implement [05-RNG-1]'s exact modulo-2^64 word operations without invoking a compiler
random primitive, consulting a candidate trace or transporting words through
floating values. A signed int64 may carry the exact word bits; checked arithmetic
must not accidentally replace modular arithmetic. Convert the high 53 bits to
their exact unit value before the single f32 arithmetic-width rounding. Apply
[05-OP-37]'s strict comparison and finalized subtraction then division, including
positive zero on dropped primal and direction coordinates. This is pathwise
directional propagation through the selected mask, not differentiation of the
rounded machine function or of the sampling distribution.

Evaluate operands and strict lets in source order, including unused results.
Check each dropout's shape and finite rate in [0,1) before Random entry. Every
successful forward call, including empty tensors and zero rates, consumes exactly
one ordinal. Advance the word counter modulo 2^64 and record the actual seed/key,
rate and ordered shape. Nested seed entry saves the parent and resets the child;
successful exit restores the exact parent. Record entry/exit even around no draws.
Resource checks use the language's device/target compatibility rule, not a model
of physical allocation. Record successful checks in order, before their bodies.
The reference differentiates the source forward and has no replay opcode; a native
backward pass must preserve the same final ambient state and saved forward mask.

Return the ordered executed event prefix and current/saved random state with
both success and failure. Distinguish malformed input, unsupported syntax,
nonfinite numerical results, traversal exhaustion and semantic guard failure.
A failed guard does not manufacture a successful event or run the continuation.
A failure inside a handler exposes the state at that instruction, before pending
normal exits; this is a prefix observation, not a claim about recovery, exception
unwinding or C's abort cleanup. The caller bounds total work and memory separately.

The owning acceptance suite combines independent full-word RNG vectors and
f32 threshold controls with nested seeds, a subsequent real draw, dead/empty/
zero-rate draws, nonunit directions and multiple input ports. Require positive
and negative Resource targets, invalid rate before entry and failure after a
prior draw, malformed unused inputs, shadowing, nonfinite results and exhaustion.
Equal numeric outputs with different effect traces must remain distinguishable.
Check pure programs against the unchanged directional reference. Source pairing,
full candidate observations, exact-eligibility classification and the generated
AD campaign remain separate integration obligations; this kernel alone accepts
no compiler certificate and discharges no logical ownership theorem. Preserve
the existing ordinary campaigns, released host pin and full package gates.

**Broadcast-coordinate repair (Hull #20, 2026-09-09; acceptance pending).**
The `EExpand`/`TExpand` model implements the same-rank unit-axis rule above.
For concrete `TData`, validate nonnegative extents, exact buffer cardinality,
an in-range int32-compatible axis, operand extent 1 and nonnegative new size.
Preserve all other axes. For each row-major output coordinate, read the input
at the same coordinates except that the selected coordinate is zero. Repeating
the entire flat buffer is correct only for an outer axis, not in general.
Size zero yields an empty buffer with the requested shape; singleton expansion
is identity. Axis equal to rank is not an insertion fallback. A malformed
operand or failed unit-extent claim has no successful reference reduction.

`type_check` rejects known nonunit extents and negative literal sizes, while
retaining symbolic names and operand effects. Concrete evaluation enforces
the unit-extent obligation for symbolic input shapes. The generator's operand
has literal extent 1 at the chosen axis, not a freshly invented dimension;
the output keeps the requested dimension at that position. Other bystanders
and element-type checking are unchanged. This is a Hull model repair, not a
change to Chelis's decided semantics, compiler behavior or any shell pin.

Acceptance requires coordinate-distinct inner/outer 2-D and middle-axis 3-D
examples, singleton/empty axes and bystanders, invalid buffers/axes/known
nonunit extents, and generated unit-axis operands with checked round trips.
Run the complete Hull suite and two-pass 10,000-program/28-rule generator
oracle; report a same-seed bounded compiler-check pilot before/after this repair
separately from the standing full campaign. A released compiler's historical
type rule is not grounds to weaken the current normative unit-axis contract.
This does not add an `insert` AST constructor, runtime dimension evaluation,
exact integer data evaluation, a trap outcome, or a derivative evaluator.
The f32-buffer and symbolic-evaluation limitations remain explicit; the
repair must not be described as full language or AD conformance.

**Ordered reduction repair (2026-09-09; Hull implementation acceptance pending).**
The modeled single-axis `sum` uses the `ESum(operand, axis)` representation above.
The parser/emitter use an integer literal expression at the axis port, not a
`Dim` node. Negative indices normalize against operand rank; invalid indices,
rank-zero operands and boolean tensors fail checking. Equal literal extents at
different positions remain distinct axes. Type checking removes only the chosen
position and preserves the other names, extents, element type and operand effects.
This changes Hull's constructor API, not Chelis syntax, behavior or shell pins.

For the existing f32-buffer evaluator, each output coordinate gathers the input
slice along that position in increasing coordinate order. Its sum uses the
adjacent-pair balanced tree of `[05-OP-30]`, carrying an odd tail unchanged; an
empty slice returns positive zero. Remove only that axis from the result shape.
Validate the concrete input buffer's element count, nonnegative extents, and
axis before indexing. Never replace a per-axis result with the whole-buffer
total. Term equality must distinguish reduction axes. The generator chooses an
insertion position and records that position as the reduction axis; repeated
extents do not require inventing a unique dimension.

This repair does not add named-axis expression resolution, multi-axis sums,
explicit accumulators, exact integer evaluation, or new dtype transport. Those
remain Hull's dated model-alignment work; the int64 element type may be checked
but is not thereby given an exact numerical evaluator. Acceptance requires
coordinate-distinct 2-by-3 and equal-extent 2-by-2 cases, rank-three middle axes,
negative indices, singleton/empty axes, invalid axes/buffers, canonical Deep text
and constructed round trips, plus the standing 10,000-check/1,000-eval campaign.
Until those gates pass, this paragraph states the repair contract, not a completed
alignment or an AD theorem.

- **Tensor representation: nested-lists-of-scalars.** A `TTensor(dims, elem)` value is a
  nested `List` of scalars whose nesting depth equals the rank and whose shape equals
  `dims`. Every tensor op (`EAdd`, `EMatmul`, `ESum`, …) is implemented as scalar loops
  over this representation. Performance is an explicit **non-goal**; this is the
  reference semantics, not an execution engine.

- **Capture-avoiding substitution is required.** `substitute(body, x, v)` in `step` must
  be capture-avoiding: substituting `v` for `x` must never let a free variable of `v` be
  captured by a binder inside `body`. Naive textual substitution is a correctness bug
  here. This reuses the substitution work mechanized in LaCaDiLE
  (`proof/lean/LaCaDiLE`); Hull's `substitute` is the executable counterpart of that
  proof-level definition and must agree with it.

- **Recommended internal representation: locally-nameless / de Bruijn.** To get
  capture-avoidance for free, the recommended internal `Expr` representation for the
  evaluator is **locally nameless** (free variables by name, bound variables by de Bruijn
  index) or fully de-Bruijn-indexed. The parser still produces the named surface `Expr`;
  the evaluator converts to the internal representation before reduction.

- **AD evaluation is OUT of v0.1.0.** Hull v0.1.0 **type-checks** `grad(f)` (§3.3) but
  does **not** evaluate it. `step` leaves `EGrad(_)` stuck (returns `None`); the
  differential eval harness (§6) does not submit grad-containing programs to
  `eval`-agreement. Evaluating AD is scoped to a later version once the linearity /
  Δ-capability judgment lands.

---

## 5. Deep Parser - `Hull.Parse`

Parse a Deep s-expression string into the `Expr` ADT, and (for round-trip / generation)
`unparse` an `Expr` back to canonical Deep. The parser must match the *shipped* Deep
grammar (`spec/03-deep-syntax.md`), which differs from the original sketch in three ways
that the sketch got wrong:

1. **Lambdas are `fn`, not `lam`.** The shipped form is `(fn {} (params ...) body)` — a
   `fn` head, a metadata slot, a `(params ...)` child, then the body. There is no `lam`
   tag.
2. **Arithmetic/tensor ops are builtin applications, not dedicated tags.** There are no
   `add` / `mul` / `sub` / `div` Deep tags. `x + y` is `(app {} (var {} add) x y)`;
   `sexpr_to_expr` **desugars** the recognized builtin heads (`add`, `mul`, `sub`, `div`,
   `neg`, `exp`, `log`, `sqrt`, `sin`, `sum`, `matmul`, `gather`, …) into the
   corresponding `EAdd` / `EMul` / `ESum` / `EMatmul` / … constructors. An `app` whose
   head is not a recognized builtin stays `EApp`.
3. **Every Deep node carries a metadata slot `{}` as element 1.** A node is
   `(tag {…} children…)`; element 1 is always the metadata map (possibly `{key: val …}`,
   e.g. `(lit {type: (t-prim {} f32)} 1.0)`, `(handle-effect {effect: name} …)`, `(grad {wrt: …}
   …)`). The tokenizer/parser must lex and skip/parse `{…}`; the original sketch ignored
   it entirely.

Deep source looks like:

```text
(fn {} (params {} (x {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))}))
  (app {} (var {} add) (var {} x) (lit {type: (t-prim {} f32)} 1.0)))
```

The parser needs:
- Tokenize: split on whitespace and parens, **lex `{` … `}` metadata maps**, handle
  string literals
- Parse atoms: integers, floats, booleans, strings, identifiers
- Parse lists: `(` `tag` `{meta}` atoms-and-lists `)` recursively
- Map s-expression structure to `Expr` constructors by head symbol, **desugaring builtin
  applications** into the dedicated tensor-op constructors

```chelis
-- Tokenize a Deep source string. Note the `{` / `}` metadata tokens.
def tokenize(src: String) -> List[Token]

type Token = LParen | RParen | LBrace | RBrace | Atom(String)

-- Parse a token stream into an s-expression tree. SMeta carries the `{}` slot.
type SExpr = SAtom(String) | SList(List[SExpr]) | SMeta(List[(String, SExpr)])

def parse_sexpr(tokens: List[Token]) -> Option[(SExpr, List[Token])]

-- Convert an s-expression to a typed Expr. Element 1 is always the metadata map,
-- so children start at index 2.
def sexpr_to_expr(s: SExpr) -> Option[Expr] =
  match s {
    -- (var {} name)
    SList([SAtom("var"), _meta, SAtom(name)]) -> Some(EVar(name))

    -- (lit {type: ...} value)
    SList([SAtom("lit"), meta, value]) -> parse_lit(meta, value)

    -- (fn {} (params ...) body) -- NOT `lam`
    SList([SAtom("fn"), _meta, params_sexpr, body]) ->
      parse_params(params_sexpr) |> flat_map(fn(ps) ->
        sexpr_to_expr(body) |> map(fn(b) -> build_lambda(ps, b)))

    -- (app {} func arg...) -- desugar recognized builtin heads
    SList([SAtom("app"), _meta, ...rest]) ->
      desugar_app(rest)   -- (var {} add) e1 e2 -> EAdd(e1, e2); else EApp chain

    -- (grad {} expr) | (grad {wrt: ...} expr idx)
    SList([SAtom("grad"), _meta, ...rest]) ->
      sexpr_to_expr(head(rest)) |> map(fn(e) -> EGrad(e))

    -- (cast {} expr target-type)
    SList([SAtom("cast"), _meta, e_sexpr, ty_sexpr]) ->
      sexpr_to_expr(e_sexpr) |> flat_map(fn(e) ->
        sexpr_to_type(ty_sexpr) |> map(fn(t) -> ECast(e, t)))

    -- ... one arm per Deep form (§2.3-§2.7); builtin-app heads desugar in desugar_app
    _ -> None
  }

-- Desugar an application's children: when the head is `(var {} <builtin>)` for a
-- recognized arithmetic/tensor builtin, build the dedicated Expr constructor; otherwise
-- fold into a left-nested EApp chain.
def desugar_app(children: List[SExpr]) -> Option[Expr]

-- Canonical Deep printer: inverse of sexpr_to_expr, for round-trip and the generator.
-- unparse(e) must re-parse to an Expr equal to e (modulo alpha-renaming), and the
-- emitted text must include the `{}` metadata slot on every node and re-sugar the
-- dedicated tensor-op constructors back into `(app {} (var {} <builtin>) ...)`.
def unparse(e: Expr) -> String
```

This is **larger than the original 100-150 line sketch**: the `{…}` metadata tokenizer,
the builtin-app desugaring/re-sugaring (`unparse` must re-emit `EAdd` as
`(app {} (var {} add) …)`), and the canonical printer for round-trip all add surface.
Deep's syntax is still regular (head symbol + fixed-position children, no operator
precedence, no ambiguity, no context-sensitivity), so the parser remains a direct
structural mapping — just one with three more moving parts than the sketch admitted.

**Round-trip property.** `parse ∘ unparse = id` (modulo alpha-renaming) is a checked
invariant **on the well-typed in-fragment domain** — i.e. on the *parser image* of the
v0.1.0 supported fragment (§3.1), the `Expr` values that `parse` actually produces from
in-fragment Deep and that `type_check` accepts. It is not claimed over arbitrary `Expr`
values: out-of-fragment forms (`EWithSeed`, `EHandleEffect`, `EConcat`, `EWhere`,
`ECumsum`, `ESort`, `EScatter`, `EVmap`, `EConstruct`) are not in the round-trip domain
because the generator does not emit them and `type_check` rejects them. `unparse` is what
the generator (§7) uses to emit `.dp` corpus files for differential testing.

**Reserved builtin op names make desugaring unambiguous.** The round-trip property
depends on the desugaring being a *function* of the s-expression — for it to hold, an
`(app {} (var {} add) e1 e2)` must mean `EAdd(e1, e2)` and nothing else. That holds
because the recognized builtin op names are **reserved**: they are not user-bindable
identifiers, so a `(var {} add)` head can only ever be the `add` builtin, never a
user-bound local named `add` shadowing it. There are **19 reserved op names** in the
v0.1.0 fragment, each the head of a dedicated-constructor desugaring:

- arithmetic binops (4): `add`, `mul`, `sub`, `div` → `EAdd` / `EMul` / `ESub` / `EDiv`;
- unary elementwise (5): `neg`, `exp`, `log`, `sqrt`, `sin` → `ENeg` / `EExp` / `ELog` /
  `ESqrt` / `ESin`;
- comparison (1): `cmplt` → `EApp(EVar("cmplt"), …)` (kept as an `EApp` head, not a
  dedicated tag; reserved so the head is unambiguous — see the §2 calculus-names note);
- reduction / contraction / index (3): `sum`, `matmul`, `gather` → `ESum` / `EMatmul` /
  `EGather`;
- shape / movement (3): `reshape`, `permute`, `expand` → `EReshape` / `EPermute` /
  `EExpand`;
- grad / cast transform heads (3): `grad`, `cast`, `vmap` — `grad` and `cast` are Deep
  *transform tags* (`(grad …)`, `(cast …)`) rather than `app` heads, but their names are
  reserved on the same footing so they cannot be rebound; `vmap` is reserved even though
  it is out of the v0.1.0 fragment (§3.1), so its name is never available to shadow.

Because none of these 19 names can be rebound, `desugar_app` (and the inverse re-sugar in
`unparse`) is deterministic, and the `parse ∘ unparse = id` invariant is well-defined on
the in-fragment parser image.

---

## 6. Differential Testing Harness - `Hull.Check`

### Host and candidate identity (2026-09-08)

Hull's released host and the compiler under test are distinct roles. The
campaign orchestrator accepts `--host-chelis-bin` and `--target-chelis-bin`.
The existing `--chelis-bin` spelling remains an alias for the host selector;
the two host spellings are mutually exclusive. If no target is supplied, the
target is the host, preserving the ordinary release-pinned campaign.

Both selectors are absolute executable paths. The host must report the exact
`reef.toml` compiler version. An explicitly selected candidate need not match
that version: it is an externally supplied artifact, never a compiler built
or vendored by Hull. Do not change the shell pin or the machine-global default
to select it. This host/target facility is Hull-specific test infrastructure,
not a new requirement on other shells.

The host launches the reference driver. The absolute target path is written
into every shard configuration and used for both nested compiler-check and
compiler-evaluation calls; a missing target field fails closed rather than
selecting a compiler from PATH. Record each selected executable's path, version
output, and SHA-256 in the campaign result. Recheck both file hashes before
accepting the aggregate to reject differences visible at those observations.
These endpoint checks assume files remain immutable during the campaign; they
cannot detect a temporary replacement restored before the final check or prove
which bytes every invocation executed. They identify selected files, not their
build provenance or an interpreter behind a wrapper. The campaign retains its existing identity,
locking, accounting, conservative-whitelist, and acceptance rules.

Test host/target routing independently of verdict agreement: matching outputs
cannot show which executable ran. Cover same-binary compatibility, different
host and target versions, wrong host pin, invalid target selection, explicit
target propagation, and changed executable detection. Compiler validation,
formal-model certification, and reference agreement remain separate claims.

The core use case. Given a Deep source file, type-check it with both Hull's reference checker and the real compiler, and compare results.

```chelis
import Hull.Parse (parse_deep_file)
import Hull.Typing (type_check)
import Std.Io (read_file, process_run)

-- Parse a Deep file and type-check with the reference checker. The Io effect uses the
-- shipped `Effect::Io` (lowercase casing in the enum; §2).
def reference_check(path: String) -> Option[(Type, EffectRow)] ! { IO } = {
  src = read_file(path)
  expr = parse_deep_file(src)?
  type_check([], expr)
}

-- Run the real compiler's structured check and parse the JSON result. `process_run` is
-- the new subprocess builtin (under Io) that the next-phase monorepo work adds (§8.1).
def compiler_check(path: String) -> Option[(Type, EffectRow)] ! { IO } = {
  result = process_run("chelis", ["check", path, "--json"])
  parse_check_result(result)
}

-- Compare both results
def differential_check(path: String) -> CheckResult ! { IO } = {
  ref_result = reference_check(path)
  comp_result = compiler_check(path)
  match (ref_result, comp_result) {
    (Some((t1, e1)), Some((t2, e2))) ->
      if types_equal(t1, t2) and effects_equal(e1, e2)
        then Agree(t1, e1)
        else Disagree(ref_result, comp_result)

    (None, None) -> AgreeReject

    (Some(_), None) -> CompilerTooConservative(ref_result)

    (None, Some(_)) -> CompilerUnsound(comp_result)
  }
}

type CheckResult =
  | Agree(Type, EffectRow)
  | AgreeReject
  | Disagree(Option[(Type, EffectRow)], Option[(Type, EffectRow)])
  | CompilerTooConservative(Option[(Type, EffectRow)])
  | CompilerUnsound(Option[(Type, EffectRow)])
```

`CompilerUnsound` is the critical finding: the compiler accepted a program that the reference checker rejects. This means the compiler has a soundness bug - it allows a program that violates the typing rules.

`CompilerTooConservative` is less critical but still worth investigating: the compiler rejects a program the spec says is valid. This means the compiler is stricter than the spec - either the compiler has an unnecessary restriction, or the spec is too permissive.

---

### Versioned scalar comparison observations

The released-host differential campaign dispatches explicitly on execution
envelope version, requiring exit zero and exactly one root. Schema 2 retains its
historical scalar interpretation: existing scalar conversions and rank-zero f32
tensor data `{dtype: "f32", values: [number]}` with exactly one numeric element.
Other tensor ranks, malformed data and missing versions fail this leg. The old
array-shaped data form and absent/extra/nonnumeric elements remain invalid.
This is a historical observation relation, not an alternative compiler codec.

Schema 3 uses spec/10 §3.2 to decode numeric scalars, booleans and tensors into
typed observations retaining exact integer values, floating bit strings, dtype
and ordered shape. Validate payload members, dtype widths/ranges, nonnegative
exact-int64 extents, dynamic-int32 rank and complete cardinality before
projection. Signed zero, infinity signs and NaN payloads remain distinct in the
observation. Aggregates outside this profile are not silently projected.

The ordinary campaign still compares a separately named scalar projection under
its historical f32 tolerance and nonfinite-collapse policy; this is neither
full-tensor nor IEEE-class agreement. Nonzero exits, malformed/out-of-profile
observations and unsupported explicit schema versions receive distinct failing
outcomes. All remain in the requested denominator. No compiler wire format,
behavior or pin changes follow from this Hull observation boundary; the
compiler's normative decoder still rejects versions other than 3.

Acceptance requires real captured schema-3 output and released schema-2
end-to-end cases, retaining the rank-zero f32 conditional cast returning 8 and
a different scalar tensor value. Include positive/negative dtype, payload, rank,
cardinality and version tests, exact-bit retention, and campaign accounting
controls for every new failure category. The full standing 10,000-check/
1,000-eval campaign remains a separate obligation; neither decoding nor scalar
tolerance agreement certifies the current normative exact codec.

### Opt-in compiler trace for the canonical LaCaDiLE revision

The `chelis-ir/lowering-trace` Cargo feature enables an additive, in-memory
`try_lower_program_to_library_with_trace` entry point. It calls the same lowering
implementation as the ordinary entry point. The ordinary entry point does not
collect snapshots, even in a feature-enabled build; feature-disabled builds have
no collector field or instrumentation. No CLI, shell interface, default output,
runtime operation, or serialized format changes.

This is an observation tool, **not a certificate or a validation result**. Its
initial acceptance oracle is `cargo nextest run -p chelis-ir --features
lowering-trace --lib --test lowering_trace`. It must cover:

1. Exact parity of the returned library and diagnostics with ordinary lowering,
   including empty programs, multiple invocations, and rejected AD.
2. Actual ordinary-grad pre/post snapshots and ordered `wrt` references; repeated
   operand ports, full constants, dimensions, shape dependencies, and roots stay
   in their existing `Dag` representation. Snapshots are taken at the production
   pass invocation, not obtained by invoking AD again.
3. Distinct context identities for nested lowering, with parent links. Unlowered
   library definitions, unresolved callable gradients, and vectorization are
   explicit trace boundaries, not evidence of an accepted ordinary-grad transformation.
4. Actual library normalization snapshots before DCE, after DCE, after consuming
   fanout copies, and after drops, together with the two production remap tables.
   The last snapshot must match the returned library DAG exactly.
5. Each successful ordinary-gradient observation has its actual call-site
   application: formal/actual dimension-specialization types, the specialized
   backward DAG, name-to-caller argument bindings, ordered actual differentiated
   inputs, the production splice map, and caller snapshots immediately before
   splicing, immediately after splicing, and after result packing/reuse hints.
   The returned value retains tuple/ADT structure, field order and names, including
   empty discrete cotangent slots. Missing raw gradients remain missing in the AD
   observation; their subsequent shaped zeros appear only in the packing snapshot.
   Nested applications use their gradient context's parent as the caller context.
   This does not extend capture to host-classified structured/List applications;
   those still have `UnloweredDefinitions` boundaries. Structural value copying
   has a separate unit test, not a claim of host-entry coverage. Tensor result
   packing alone is not evidence that host-visible List checks were preserved.

The trace deliberately has no `Serialize`/`Deserialize` implementation and is not
a new numeric wire transport: it retains the existing compiler `Dag` carrier
without converting constants or introducing scalar numeric payloads.
A later external evidence envelope must use exact tagged numeric carriers and
extend the numeric-surface enumerators in that same change. Graph-local IDs are
not cross-pass identities; consumers must check, not trust, the recorded maps.

The application snapshots and maps are observations, not proofs of call-site
splicing/result-packing correspondence. Remaining obligations include checking them,
pre-erasure Random protocol and Resource metadata, the selected emission's
`VerifiedDagProgram.emission()` ownership actions, external decoding and checking,
and independent Hull numerical tests. This trace does not cover contextual/host
subexpression entry points or certify floating-point AD. In particular, observing
a Dropout node does not discharge the required Dropout conformance lane. Existing
compiler/spec discrepancies require separately approved compatibility work; this
tool must not repair or conceal them.

For example, a host-classified function using a runtime shape can leave this
library DAG empty. `UnloweredDefinitions` records that boundary; the returned
library's `lowered_names` table identifies the definitions. An empty trace is
never evidence that the source program's obligations were discharged.

### Source-owned execution sequence and AD capture

Fixed-control evaluation plans retain one ordered sequence of actual graph
nodes and explicit seed entry/exit controls. A handler that returns an existing
value, with no draw or newly computed body node, still has both controls.
Lowering independently records an occurrence census; source occurrence IDs,
scope IDs, forward draw IDs, graph IDs and raw seed values are distinct.
Backward replay is not another source occurrence and does not repeat controls.

Declaration selection follows the independently recorded declaration
dependencies, not only the final graph's value liveness. An alias declaration
uses its existing named observation carrier so a reference retains that
declaration's controls; unrelated sibling declarations remain excluded. The
explicit plan-selection API also admits a draw-free region without changing
the profile used by ordinary evaluator dispatch. Other exclusions remain errors.
The owning copy/drop, context-composition and AD/splice boundaries map the existing
sequence rather than recovering source controls by sorting a resulting graph.
Host partitioning uses occurrence cuts recorded at the actual source action;
even a control-only segment executes before that action. Segment frames retain
the same invocation's saved keys and scope state across those cuts.
Validation requires the complete selected graph, source census in order,
balanced unique seed scopes and forward/replay dominance. Joint deletion of
runtime controls or draw metadata cannot delete the source census.

The same `lowering-trace` feature additionally exposes
`try_lower_program_to_evaluation_library_with_trace`. Its ordinary counterpart
does not collect full snapshots. The additive `EvaluationLoweringTrace` links
execution-bearing pre/post AD regions to the existing gradient/application
maps, at the actual production call. It retains exact existing DAG types and
constants; its occurrence census checks execution identity, not an independent
arithmetic theorem. `LoweringTrace`, public legacy graph/kernel products and
serialized formats remain unchanged. Inspection views cannot construct plans.

The owning oracle adds value-free/equal-seed controls, independently selected
dependencies, joint omission/substitution negatives, and execution of the
captured backward graph with unchanged outer stream state. This is source/IR
infrastructure, not a compiled-C capability, host-emission snapshot, new wire
profile or mixed certificate. Joining actual selected emission and independently
checking both graph correspondences remains required.

The feature also provides explicit helper-observing counterparts for manifested
host execution lowering and fixed-control named-entry lowering. Each successful
host tensor helper retains one private lowering product containing its existing
execution metadata and, only for the observing ingress, an owned helper trace.
The trace reuses the production collector at the helper's actual AD, splice,
result-packing and normalization boundaries. The pre-normalization helper result
retains its tuple/ADT structure and exact ordered, duplicate-split root IDs. It
adds execution-splice mappings
for occurrence, draw and scope identities, because the existing
`Application.remap` intentionally carries node identities only. Execution
normalization observations are present only for helpers already lowered through
the fixed-control execution path; observing an ordinary helper neither creates
execution metadata nor changes its AD or normalization route. Historical pass
snapshots remain immutable. When host lowering later rebinds helper dimensions,
the trace records that actual boundary output separately instead of rewriting
the earlier snapshots retroactively.

Helper products are appended only after the helper succeeds. A rejected or
speculative lowering therefore contributes no observation. Function projection
moves the exact function, helper graphs, execution metadata and optional traces
together; discarded siblings do not become evidence for the selected function.
The host execution plan exposes only borrowed, function/helper-indexed or
global-helper-indexed trace access. Its consuming ownership boundary still
extracts the original graph/metadata association, after an observer has had a
chance to copy the selected trace. Ordinary lowering and ordinary execution
lowering do not collect helper snapshots, including in a feature-enabled build.
An explicit consuming trace-discard operation removes only these optional
observations; it cannot remove execution metadata or substitute helper graphs.
The retained helper trace is owned and `Send + Sync`; the collector's local
`Rc<RefCell<_>>` never enters the returned carrier.

The focused IR oracle for this companion is `cargo nextest run -p chelis-ir
--features lowering-trace --test helper_lowering_trace --test lowering_trace`.
It compares traced/untraced helper graphs, raw versus shaped-zero gradients,
tuple root order, actual fixed-control pre/post-AD execution and function
projection. This IR evidence still does not bind a trace to final emitted bytes:
the compiler API must perform that later successful-artifact join exactly once.

### Opt-in observation of selected compiler emission

The `chelis-compiler-api/emission-observer` feature supplies
`compile_for_execution_with_observer`, an observational counterpart of the
strict `compile_for_execution` API. It shares source checking, entry selection,
optimization, ownership verification, and code generation with that API.
The callback receives immutable native views immediately before code generation:
the checked/manifested source and either the exact verified standalone DAG or
the exact verified host payload (including its verified nested DAG cursors).
Standalone DAG observations also retain the selected pre-specialization,
pre-fusion DAG. A host observation makes no claim to have such a single graph.

`EmissionObservation.lowered_host` additionally borrows a snapshot of the actual
initial host lowering, when one exists, before entry projection and backend
preparation. It is captured only when an observer is installed, without another
lowering or ownership-verification pass. In particular, a tuple-gradient entry
may emit only its derivative while this snapshot still contains its scalar loss.
The snapshot is not ownership-verified, is not necessarily pre-AD, and does not
make an unselected function part of the artifact. Consumers must establish their
own source/snapshot/selected-payload correspondence and separately identify any
primal compilation used for numerical comparison. Ordinary compilation, including
feature-enabled calls without an observer, does not make this snapshot copy.

This feature changes no default output, public wire schema, CLI, shell pin, or
language behavior. Ordinary compilation does not invoke an observer, including
in feature-enabled builds. Views borrow existing tagged compiler carriers; they
are not a new serialization format. The callback may copy observations for later
inspection but cannot mutate the verified payload. Observations may precede a
later compilation failure: only the enclosing API's successful result establishes
that code generation and artifact construction completed. An observation is not
an acceptance verdict, and callback failures are the opt-in caller's failures.

The acceptance oracle is `cargo nextest run -p chelis-compiler-api --features
emission-observer --test emission_observer --test execution_artifact_metadata`.
It must compare complete artifacts and diagnostics with ordinary compilation,
check actual selected entry ownership actions rather than the library DAG,
exercise standalone and host emission, and preserve rejection without treating
an empty observation as successful certification. Tuple-gradient cases must
distinguish the full initial host lowering from the selected emitted functions,
including partial and complete primal disconnection. Feature-disabled compilation
is checked separately. This does not yet join the library AD trace to selected
emission, check fusion or effect erasure, or implement an external certificate.

## 7. Spec-Driven Test Generation - `Hull.Generate`

Generate random well-typed Deep programs. Naive approach (generate random AST, check if it types) has near-zero hit rate for non-trivial programs. The useful approach is top-down, type-directed generation.

```chelis
-- Generate a random expression of a given type at a given depth
def gen_expr(ctx: Ctx, target: Type, depth: int64, rng: RngState)
    -> (Expr, RngState) =
  if depth <= 0 then gen_leaf(ctx, target, rng)
  else {
    -- Choose a generation strategy based on the target type
    (choice, rng) = random_int(rng, 0, num_strategies(target))
    match (target, choice) {

      -- Strategy: variable lookup
      (_, 0) -> {
        candidates = filter(ctx, fn((_, t)) -> types_equal(t, target))
        if is_empty(candidates) then gen_leaf(ctx, target, rng)
        else {
          (idx, rng) = random_int(rng, 0, length(candidates))
          (EVar(fst(get(candidates, idx))), rng)
        }
      }

      -- Strategy: application (target = T, find f : S -> T, generate e : S)
      (_, 1) -> {
        -- Pick a random argument type
        (arg_type, rng) = gen_type(rng, depth - 1)
        -- Generate the function
        (fn_expr, rng) = gen_expr(ctx, TArrow(arg_type, target, []),
                                  depth - 1, rng)
        -- Generate the argument
        (arg_expr, rng) = gen_expr(ctx, arg_type, depth - 1, rng)
        (EApp(fn_expr, arg_expr), rng)
      }

      -- Strategy: add (target must be tensor)
      (TTensor(dims, elem), 2) -> {
        (e1, rng) = gen_expr(ctx, target, depth - 1, rng)
        (e2, rng) = gen_expr(ctx, target, depth - 1, rng)
        (EAdd(e1, e2), rng)
      }

      -- Strategy: let binding
      (_, 3) -> {
        (bound_type, rng) = gen_type(rng, depth - 1)
        (name, rng) = gen_fresh_name(rng)
        (bound_expr, rng) = gen_expr(ctx, bound_type, depth - 1, rng)
        (body_expr, rng) = gen_expr(
          extend(ctx, name, bound_type), target, depth - 1, rng)
        (ELet(name, bound_expr, body_expr), rng)
      }

      -- Strategy: lambda (target must be arrow type)
      (TArrow(t_arg, t_ret, _), _) -> {
        (name, rng) = gen_fresh_name(rng)
        (body, rng) = gen_expr(
          extend(ctx, name, t_arg), t_ret, depth - 1, rng)
        (ELam(name, t_arg, body), rng)
      }

      -- Fallback: literal or variable
      _ -> gen_leaf(ctx, target, rng)
    }
  }


-- Generate a leaf expression (literal or variable) of the target type
def gen_leaf(ctx: Ctx, target: Type, rng: RngState) -> (Expr, RngState) =
  match target {
    TF32 -> {
      (v, rng) = random_float(rng, -10.0, 10.0)
      (ELit(LFloat(v)), rng)
    }
    TInt64 -> {
      (v, rng) = random_int(rng, -100, 100)
      (ELit(LInt(v)), rng)
    }
    TBool -> {
      (v, rng) = random_bool(rng)
      (ELit(LBool(v)), rng)
    }
    TTensor(dims, EF32) -> {
      -- Generate a const tensor expression
      (v, rng) = random_float(rng, -1.0, 1.0)
      (EApp(EVar("const"), ETuple([ELit(LFloat(v)),
        dims_to_expr(dims)])), rng)
    }
    _ -> {
      -- Try to find a variable of the right type in context
      candidates = filter(ctx, fn((_, t)) -> types_equal(t, target))
      if is_empty(candidates)
        then (ELit(LInt(0)), rng)  -- stuck: wrong type, will fail check
        else {
          (idx, rng) = random_int(rng, 0, length(candidates))
          (EVar(fst(get(candidates, idx))), rng)
        }
    }
  }


-- Generate random types for argument positions
def gen_type(rng: RngState, depth: int64) -> (Type, RngState) = {
  (choice, rng) = random_int(rng, 0, 5)
  match choice {
    0 -> (TF32, rng)
    1 -> (TInt64, rng)
    2 -> (TBool, rng)
    3 -> {
      (ndims, rng) = random_int(rng, 1, 3)
      (dims, rng) = gen_dims(rng, ndims)
      (TTensor(dims, EF32), rng)
    }
    _ -> if depth > 0 then {
      (t_arg, rng) = gen_type(rng, depth - 1)
      (t_ret, rng) = gen_type(rng, depth - 1)
      (TArrow(t_arg, t_ret, []), rng)
    } else (TF32, rng)
  }
}
```

The generator produces programs that are well-typed by construction (each generation step picks a strategy that produces the target type). The reference type checker then verifies the generated program - if the generator has a bug, the checker catches it. The verified program is then run through `chelis check` for differential testing.

### 7.1 Pinned coverage target

> **Coverage target:** every typing rule the reference checker implements — i.e. every
> match arm of `type_check` (and the helpers `check_binary_tensor_op` /
> `check_unary_tensor_op`) — is exercised by **at least one** generation strategy, and
> this is **verified by a coverage report**, not asserted.

`scripts/coverage_report.py` (pure tabulation, §8) maps generated programs to the
`type_check` arms they exercise and fails if any arm in the v0.1.0 supported fragment
(§3.1) has zero generated coverage. Adding a new `type_check` arm without a generation
strategy that reaches it is a coverage-report failure, which keeps the generator and the
checker in lockstep.

### 7.2 Pinned generation budget

Generation is bounded so the suite is deterministic and CI-affordable:

- **Depth bound:** `gen_expr` is called with an initial `depth` bound (e.g. `depth = 6`);
  every recursive strategy decrements `depth`, and `depth <= 0` falls back to
  `gen_leaf`. This bounds AST size and guarantees termination.
- **Per-program step budget:** each generated program carries a reduction `max_steps`
  budget for the §4 evaluator (`eval_to_value(e, max_steps)`); programs that do not reach
  a value within budget are recorded as `eval`-timeouts, not failures, and excluded from
  the eval-agreement count.
- **Suite-level seed:** generation is seeded so the conformance suite is reproducible
  byte-for-byte from a recorded seed (see §11 generation-time budget).

---

## 8. Module Structure

```text
chelis-lang/hull/
├── reef.toml                         -- depends on chelis-std only
├── SKILL.md                          -- agent docs
├── README.md
├── src/
│   ├── core.ch                       -- version, re-exports
│   ├── ast.ch                        -- Expr, Type, Dim, Effect ADTs (Section 2)
│   ├── typing.ch                     -- Reference type checker (Section 3)
│   ├── eval.ch                       -- Reference evaluator (Section 4)
│   ├── parse.ch                      -- Deep s-expression parser (Section 5)
│   ├── check.ch                      -- Differential testing harness (Section 6)
│   ├── generate.ch                   -- Type-directed program generator (Section 7)
│   └── helpers.ch                    -- Ctx lookup, substitution, unification
├── tests/
│   ├── goldens/
│   │   ├── typing/*.json             -- expected types for test programs
│   │   ├── eval/*.json               -- expected values for test programs
│   │   └── differential/*.json       -- programs with known compiler results
│   └── *.ch                          -- native Chelis golden assertions (Test effect)
├── drivers/
│   ├── gen_conformance_suite.ch      -- Chelis driver: generate, ref-check, export
│   └── run_differential_suite.ch     -- Chelis driver: batch differential testing
├── scripts/
│   └── coverage_report.py            -- pure tabulation: which typing rules are exercised
└── corpus/
    ├── generated/                    -- generated well-typed programs (output of gen)
    └── known_conservative.json       -- whitelist of documented CompilerTooConservative families (§11)
```

### 8.1 No-Python framing reconciled

The Hull stub describes Hull as "depends on `chelis-std` only / no Python," but the
original sketch listed `gen_conformance_suite.py` and `run_differential.py`. These are
reconciled as follows, and this is the pinned decision:

- **`gen_conformance_suite` and `run_differential_suite` are Chelis drivers, not Python.**
  They are `.ch` programs with `def main() -> unit ! { IO }` that read Deep files, run the
  reference checker/evaluator, shell out to the compiler, and write the corpus. They are
  pure Chelis because Hull gains a new `process_run` exec builtin (under `Io`) to invoke
  the compiler.
- **Golden assertions are native `tests/*.ch`** carrying the `Test` effect (named
  `test_*` / `example_*` per §10.1), not Python golden runners.
- **Only `coverage_report.py` stays Python** — it is pure tabulation over the generated
  corpus and the `type_check` arm list (§7.1), with no language semantics in it.

**Enabling dependencies (monorepo, next phase).** Making the drivers pure Chelis requires
three additions that do **not** exist yet (verified: no `process_run`, `eval --json`, or
`check --json` in `crates/chelis-cli/src/` as of v0.7.19):

1. a `process_run` subprocess builtin under the `Io` effect, so a Chelis driver can invoke
   `chelis check` / `chelis eval`;
2. `chelis eval --json` (machine-readable values) for eval-agreement;
3. structured `chelis check --json` (machine-readable type + effect row) for
   check-agreement.

These are added in the next phase in the monorepo; the `chelis-std`-only dependency story
holds once they ship.

---

## 9. Scope Boundaries

### What Hull implements:
- The LaCaDiLE core calculus (effects, linearity, dimensions, AD rules)
- The full Chelis typing rules for all shipped constructs (ADTs, pattern matching, tuples, collections, scalars, strings)
- Small-step operational semantics for all constructs
- Deep parsing from s-expression strings
- Differential testing against `chelis check` and `chelis eval`
- Type-directed random program generation

### What Hull does NOT implement:
- Compilation to C / HIP / Metal (Hull is an interpreter, not a compiler)
- Fusion, optimization, or performance (Hull's evaluator is intentionally naive)
- Surf parsing (Hull operates on Deep only - Surf desugars to Deep before Hull sees it)
- Module system / reef package resolution (Hull type-checks single-file programs)
- The full standard library (Hull knows the types of builtins but doesn't implement their tensor operations - it uses scalar representations for reference evaluation)
- Formal proofs (that's Lean/LaCaDiLE - Hull is an executable spec, not a proof assistant)

### Relationship to other tools:
- **LaCaDiLE (Lean):** Proves the typing rules are sound. Hull implements the rules as executable code. Lean proves "these rules are correct"; Hull checks "the compiler follows these rules."
- **`chelis prove`:** Language-level random generation. `chelis prove` generates random inputs to test individual functions. Hull generates random programs to test the compiler. Different levels of abstraction, complementary.
- **Conformance test suite:** Hull generates the suite. The suite is checked into the repo. The suite runs in CI against the Rust compiler. Hull is the test generator; the suite is the test artifact.
- **Trust stack — proof of pattern:** Hull validates the executable-properties-as-spec pattern on the highest-stakes code in the system: the compiler itself. The reference type checker is a property ("the compiler's type-checking behavior matches the formal rules"). Differential testing is the verification mechanism. The same architecture — reference implementation + production implementation + agreement on random inputs — is what user-facing `@property matches_reference forall(...)` does for application code. Hull proves the pattern works; `chelis prove` makes it available to users. See `chelis_trust_stack.md`.
- **Hull as the internal version of the canonical-properties pattern:** Domain shells ship `@property` functions in a co-located `properties/` directory that verify their implementations against domain invariants (Shoals: put-call parity, no-arbitrage; Octant: round-trip, provenance). Hull does the same thing for the compiler: the reference type checker is a "property" that verifies the compiler's behavior against the formal typing rules. The same architecture (reference implementation + production implementation + agreement checking) applies at both levels. Hull is the proof that the pattern works on the highest-stakes code; domain shell properties are the user-facing application.

---

## 10. Prerequisites and Timing

| Prerequisite | Status | Why Hull needs it |
|---|---|---|
| LaCaDiLE typing rules finalized | Stabilizing (in-repo `proof/lean/LaCaDiLE` partial, OOPSLA-targeted). **Honest scope:** the rules for the v0.1.0 *supported fragment* (§3.1) — the `AdjointSupported` boundary plus the non-AD core — are stable enough to target; the linearity / Δ-capability and effect-handling rules are **not yet frozen**, which is why those constructs are out of the v0.1.0 fragment. | Hull implements these rules - they must be stable |
| Deep syntax stable | Shipped/stable (Deep parser shipped Phase 0b; tag vocabulary in `spec/03-deep-syntax.md`) | Hull parses Deep - the grammar must not change |
| ADTs + pattern matching in Chelis | Shipped | Hull's entire data model is ADTs |
| Option type + `?` operator | Shipped | Hull returns `Option` from every check |
| String operations in Chelis | Foundation shipped (`String` primitive + `string_*` builtins, `to_int`/`to_float`; see `examples/scalar_string_foundation.ch`) | Hull parses source strings |
| `chelis prove` infrastructure | Shipped (v0.7.1: `@property`, type-directed sampling) | Hull's generator is the language-level version |
| `process_run` subprocess builtin (under `Io`) | **In-flight** (not present in `crates/chelis-cli/src/` as of v0.7.19) | Chelis drivers invoke `chelis check` / `chelis eval` (§8.1) |
| `chelis eval --json` | **In-flight** (no `--json` on `eval` as of v0.7.19) | Machine-readable values for eval-agreement (§6) |
| Structured `chelis check --json` | **In-flight** (no `--json` on `check` as of v0.7.19) | Machine-readable type + effect row for check-agreement (§6) |

**Timing:** Phase 4 or Phase 5. Not before the language-spec paper (OOPSLA-targeted) finalizes the linearity / Δ-capability rules and the ICLR pipeline establishes the AI training loop. The mechanical prerequisites for the v0.1.0 *supported fragment* are met (Deep parser, ADTs + pattern matching, `Option`/`?`, string foundation, `chelis prove`); the remaining work is the three in-flight monorepo enablement deps above (`process_run`, `eval --json`, structured `check --json`) and freezing the linearity / effect-handling rules that the out-of-fragment constructs need. Hull's value increases as the language stabilizes - building the full checker while those rules are still changing means constant maintenance, so v0.1.0 deliberately scopes to the already-stable fragment and prevents regression there.

---

## 11. What Success Looks Like

Hull v0.1.0 is complete when the following concrete numbers and mechanisms hold:

1. Every typing rule in the v0.1.0 supported fragment (§3.1) has a corresponding match
   arm in `Hull.Typing.type_check`, annotated with the rule name and paper reference, and
   `scripts/coverage_report.py` shows every such arm is exercised by ≥ 1 generation
   strategy (§7.1).

2. **Check agreement:** the reference type checker and the Rust compiler agree on
   **≥ 10,000** generated well-typed programs, with **zero `CompilerUnsound`** findings
   (the compiler never accepts what the reference rejects) and **near-zero, fully
   whitelisted `CompilerTooConservative`** findings. Every non-whitelisted
   `CompilerTooConservative` is a release blocker.

3. **Eval agreement:** the reference evaluator and `chelis eval` agree on **≥ 1,000**
   well-typed programs, with float values matching within an explicit **f32 tolerance**.
   AD-containing programs are excluded (§4.1: grad is type-checked, not evaluated, in
   v0.1.0).

4. **Conformance suite in CI:** the generated conformance suite is checked into the main
   Chelis monorepo and runs in CI against every compiler change.

5. **Generation-time budget:** the suite regenerates from a recorded seed within a pinned
   wall-clock budget (CI-affordable) under the depth/step bounds of §7.2, so CI does not
   regenerate unbounded work per run.

6. **`corpus/known_conservative.json` whitelist mechanism:** documented families where the
   compiler is intentionally or known-to-be stricter than the v0.1.0 reference checker —
   chiefly the EGrad surface-vs-deep-AD divergence (§3.3) and effect-representation
   divergences — are recorded in `corpus/known_conservative.json` with a per-family
   justification. A `CompilerTooConservative` finding matching a whitelisted family does
   not fail the suite; one that does not match any family does. The whitelist shrinks as
   v0.2.0 adds the linearity / Δ-capability check.

7. A new typing rule added to the compiler requires a corresponding addition to Hull - the
   differential testing catches any rule that exists in one but not the other.

This is the self-referential loop in operation: the compiler checks Hull, Hull checks the compiler, and disagreements are found automatically.
