# Hull - Executable Language Specification Shell

**Shell name:** Hull (`chelis-lang/hull`)
**Marine rationale:** The hull defines the shape of the vessel. The spec defines the shape of the language.
**Depends on:** `chelis-std` (required). No other shells.
**Status:** Stub. Phase 4/5 item. Prerequisites: LaCaDiLE typing rules finalized, Deep parser in Chelis, `chelis fuzz` infrastructure.

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
  | ELam(String, Type, Expr)
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
  | ECast(Expr, ElemType)
  | ESum(Expr, Dim)
  | EGather(Expr, Expr, int64)
  | EScatter(Expr, Expr, Expr, ScatterMode)
  | EMatmul(Expr, Expr)
  | EWhere(Expr, Expr, Expr)
  | EConcat(List[Expr], int64)
  | EReshape(Expr, List[Dim])
  | EPermute(Expr, List[int64])
  | EExpand(Expr, List[Dim])
  | ECumsum(Expr, int64)
  | ESort(Expr, int64)
  | EGrad(Expr)
  | EVmap(Expr, int64)
  | EWithSeed(int64, Expr)
  | EWithHandler(Effect, Expr)
  | EMatch(Expr, List[MatchArm])
  | ETuple(List[Expr])
  | ETupleGet(Expr, int64)
  | EConstruct(String, List[Expr])
  | EImport(String, List[String])

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

-- Effects
type Effect = Random | IO | Resource | Fail | Accum

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

    -- T-Sum (LaCaDiLE Section 3, Figure 4)
    -- Reduces one dimension from the tensor type
    ESum(e1, dim) -> {
      (t1, effs) = type_check(ctx, e1)?
      match t1 {
        TTensor(dims, elem) ->
          if member(dim, dims)
            then Some((TTensor(remove(dims, dim), elem), effs))
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

    -- T-WithSeed (LaCaDiLE Section 3)
    -- with-seed n e : removes Random from e's effects
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
    -- Beta reduction: (lam x body) applied to a value
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

---

## 5. Deep Parser - `Hull.Parse`

Parse a Deep s-expression string into the `Expr` ADT. Deep's syntax is minimal enough that the parser is small.

Deep source looks like:

```text
(def f (lam (x : (tensor [batch hidden] f32))
  (add x (const 1.0 [batch hidden]))))
```

The parser needs:
- Tokenize: split on whitespace and parens, handling string literals
- Parse atoms: integers, floats, booleans, strings, identifiers
- Parse lists: `(` atoms-and-lists `)` recursively
- Map s-expression structure to `Expr` constructors by the head symbol

```chelis
-- Tokenize a Deep source string
def tokenize(src: String) -> List[Token]

type Token = LParen | RParen | Atom(String)

-- Parse a token stream into an s-expression tree
type SExpr = SAtom(String) | SList(List[SExpr])

def parse_sexpr(tokens: List[Token]) -> Option[(SExpr, List[Token])]

-- Convert an s-expression to a typed Expr
def sexpr_to_expr(s: SExpr) -> Option[Expr] =
  match s {
    SAtom(name) -> Some(EVar(name))  -- or parse as literal

    SList([SAtom("lam"), SList([SAtom(x), SAtom(":"), type_sexpr]), body]) ->
      sexpr_to_type(type_sexpr) |> flat_map(fn(t) ->
        sexpr_to_expr(body) |> map(fn(b) -> ELam(x, t, b)))

    SList([SAtom("app"), e1_sexpr, e2_sexpr]) ->
      sexpr_to_expr(e1_sexpr) |> flat_map(fn(e1) ->
        sexpr_to_expr(e2_sexpr) |> map(fn(e2) -> EApp(e1, e2)))

    SList([SAtom("add"), e1_sexpr, e2_sexpr]) ->
      sexpr_to_expr(e1_sexpr) |> flat_map(fn(e1) ->
        sexpr_to_expr(e2_sexpr) |> map(fn(e2) -> EAdd(e1, e2)))

    SList([SAtom("grad"), e_sexpr]) ->
      sexpr_to_expr(e_sexpr) |> map(fn(e) -> EGrad(e))

    -- ... one arm per Deep form
    _ -> None
  }
```

This is 100-150 lines of pure Chelis. Deep's syntax is regular (head symbol determines the form, fixed positional arguments). No operator precedence, no ambiguity, no context-sensitivity. The parser is a direct mapping from s-expression structure to ADT constructors.

---

## 6. Differential Testing Harness - `Hull.Check`

The core use case. Given a Deep source file, type-check it with both Hull's reference checker and the real compiler, and compare results.

```chelis
import Hull.Parse (parse_deep_file)
import Hull.Typing (type_check)
import Std.Io (read_file, exec_command)

-- Parse a Deep file and type-check with the reference checker
def reference_check(path: String) -> Option[(Type, EffectRow)] ! { IO } = {
  src = read_file(path)
  expr = parse_deep_file(src)?
  type_check([], expr)
}

-- Run the real compiler's check and parse the fitness JSON
def compiler_check(path: String) -> Option[(Type, EffectRow)] ! { IO } = {
  result = exec_command("chelis check " ++ path ++ " --json")
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
│   ├── run_hull_tests.py             -- golden-file assertions
│   └── run_differential.py           -- batch differential testing
├── scripts/
│   ├── gen_conformance_suite.py      -- generate random programs, check, export
│   └── coverage_report.py            -- which typing rules are exercised
└── corpus/
    └── generated/                    -- generated well-typed programs (output of gen)
```

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
- **`chelis fuzz`:** Language-level random generation. `chelis fuzz` generates random inputs to test individual functions. Hull generates random programs to test the compiler. Different levels of abstraction, complementary.
- **Conformance test suite:** Hull generates the suite. The suite is checked into the repo. The suite runs in CI against the Rust compiler. Hull is the test generator; the suite is the test artifact.
- **Trust stack — proof of pattern:** Hull validates the executable-properties-as-spec pattern on the highest-stakes code in the system: the compiler itself. The reference type checker is a property ("the compiler's type-checking behavior matches the formal rules"). Differential testing is the verification mechanism. The same architecture — reference implementation + production implementation + agreement on random inputs — is what user-facing `@property fn matches_reference(...)` does for application code. Hull proves the pattern works; `chelis fuzz` makes it available to users. See `chelis_trust_stack.md`.
- **Hull as the internal version of the canonical-properties pattern:** Domain shells ship `@property` functions in a co-located `properties/` directory that verify their implementations against domain invariants (Shoals: put-call parity, no-arbitrage; Octant: round-trip, provenance). Hull does the same thing for the compiler: the reference type checker is a "property" that verifies the compiler's behavior against the formal typing rules. The same architecture (reference implementation + production implementation + agreement checking) applies at both levels. Hull is the proof that the pattern works on the highest-stakes code; domain shell properties are the user-facing application.

---

## 10. Prerequisites and Timing

| Prerequisite | Status | Why Hull needs it |
|---|---|---|
| LaCaDiLE typing rules finalized | In progress (POPL Jul 9) | Hull implements these rules - they must be stable |
| Deep syntax stable | Stable since v0.1.0 | Hull parses Deep - the grammar must not change |
| ADTs + pattern matching in Chelis | Working (v0.1.7) | Hull's entire data model is ADTs |
| Option type + `?` operator | Working | Hull returns `Option` from every check |
| String operations in Chelis | Working (3c) | Hull parses source strings |
| `chelis fuzz` design | Planned | Hull's generator is the language-level version |

**Timing:** Phase 4 or Phase 5. Not before the POPL paper finalizes the typing rules and the ICLR pipeline establishes the AI training loop. Hull's value increases as the language stabilizes - building it while the typing rules are still changing means constant maintenance. Build it when the rules are final and use it to prevent regression.

---

## 11. What Success Looks Like

Hull is complete when:

1. Every typing rule from the LaCaDiLE paper has a corresponding match arm in `Hull.Typing.type_check`, annotated with the rule name and paper reference.

2. The reference type checker and the Rust compiler agree on 10,000+ randomly generated programs (zero `CompilerUnsound` findings, near-zero `CompilerTooConservative` findings).

3. The reference evaluator and `chelis eval` agree on 1,000+ well-typed programs (values match within f32 tolerance).

4. The generated conformance test suite is checked into the main Chelis repo and runs in CI against every compiler change.

5. A new typing rule added to the compiler requires a corresponding addition to Hull - the differential testing catches any rule that exists in one but not the other.

This is the self-referential loop in operation: the compiler checks Hull, Hull checks the compiler, and disagreements are found automatically.
