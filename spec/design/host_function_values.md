# Host Function Values: Shared Representation and C-Host Boundary

**Status:** Ratified implementation plan; implementation pending.
**Owners:** [#866] (shared representation and boundary placement), [#867]
(capability attribution), [#868] (unsupported-diagnostic provenance), and
[#879] (the future general C-host function-value ABI).
**Owning language contracts:** `spec/03-deep-syntax.md` §4.2 (explicit
closure construction), `spec/04-type-system.md` §8.3 (capture ownership),
`spec/05-risc-primitives.md` §7 (loud unsupported), and
`spec/06-transformations.md` §§2-4 (typed function-producing transforms).
**Owning compiler contracts:** `loud_unsupported.md` §§C2/C6.3,
`capability_table.md`, and `chelis_span_survival.md` §2.

This document records the post-PR-[#799] correction. It is deliberately a
specification lock, not a claim that the implementation below has landed.
Until the two implementation phases in §8 are complete, the current compiler
still has the split and dead-span behavior described in [#866] and [#868].

## 1. Decision

Function and closure values are valid Chelis logical values. The shared host
IR must therefore represent them through logical type resolution, even when a
selected backend cannot emit them. A target-free lowering rejection is not a
third capability category.

The C backend may continue to support the contextual cases it can already
emit, but every other function value is rejected in the private
`ConcreteHostType -> HostAbiType` projection. That rejection is:

- `UnsupportedKind::HostAbi`;
- `Stage::Codegen("c")`;
- attributed to the originating source span when one exists; and
- classified as `Unimplemented { issue: #879 }`, not as a permanent language
  rejection and not merely as part of the plan that made it loud.

This correction does **not** implement closure conversion or a general C
closure ABI. [#879] owns that later capability.

## 2. What C supports today

The C backend has contextual higher-order mechanisms, not a partial general
closure ABI:

| Source use | Current representation | Why no general closure value is needed |
|---|---|---|
| ordinary named call | direct generated C symbol call | the callee identity is static |
| declared function-typed parameter | exact C function-pointer declarator, for example `int8_t (*callback)(int8_t)` | the surrounding signature supplies the only legal ABI context |
| direct named callback argument | the named C symbol in that contextual signature | no environment or independently stored value is produced |
| forwarded callback parameter | the existing typed function pointer | the value never leaves its declared callback position |
| inline callback to `map`/`filter`/`fold`/`scan`/`partition`/`flat_map` | callback body emitted inside the generated host loop | there is no callback object |
| immediately applied lambda or statically known transform | beta reduction or DAG specialization before ordinary emission | the function-producing expression is eliminated |

The private `HostAbiType::Callback` is consequently a contextual declarator.
It has no standalone C type spelling and cannot be selected for a return,
binding, aggregate field, collection element, or dynamically selected value.

A general closure ABI would additionally need a representation such as a
typed code pointer plus an environment, environment layout, capture lifetime,
copy/drop ownership, aggregate and collection layout, return rules, and
indirect application. None of those rules exists today. The existing paths
are intentional fast paths and remain valid after both remediation phases.

## 3. Capability-table placement

The capability schema gains a subject sum so executable constructs can use
the same Table-A/Table-B distinction as operations:

```rust
pub enum CapabilitySubject {
    Operation {
        builtin: BuiltinId,
        surface: Surface,
        dtype: Prim,
    },
    Construct {
        construct: ConstructId,
        context: ConstructContext,
    },
}

pub enum ConstructId {
    FunctionValue,
}

pub enum ConstructContext {
    ContextualCallback,
    FirstClassValue,
}
```

`sig` in a Table-A `Supported` cell is the checked type/placement contract for
either subject kind. Backend completeness is calculated over an exhaustive
`applicable_backends(subject)` function rather than an author-controlled
optional list. Operation subjects retain the existing full backend set.
Function-value subjects select `eval` and `c-host`; `c-dag` never consumes
host function values. A HIP or Metal build that takes the current host
fallback selects the `c-host` cell and therefore reports `codegen:c`; it does
not pretend that a native HIP or Metal host ABI exists.

The initial construct rows are:

| Table A subject | Table A | `eval` | `c-host` |
|---|---|---|---|
| `FunctionValue / ContextualCallback` | `Supported` — a function value may occur in a checked contextual callback position | `Implemented` by the runtime closure/transform application path | `Implemented` by exact callback declarators, direct symbols, forwarding, body inlining, and specialization |
| `FunctionValue / FirstClassValue` | `Supported` — function values may be returned, bound, selected, captured, transformed, and stored | `Implemented` by `RuntimeValue::{Closure, Transform}` | `Unimplemented { issue: #879, diagnostic: ... }` |

Returned, stored, dynamically selected, capturing, and transformed forms are
fixtures for the single `FirstClassValue` row, not independent syntax-policy
rows. `chelis check` consumes Table A and must continue to accept the
well-typed forms. `chelis build` consumes Table B after target selection.

## 4. Shared typed representation

### 4.1 IR shape

The shared host expression vocabulary gains a typed callable payload:

```rust
pub struct HostExpr<T = HostTypeTerm, F = HostFunctionValue<T>> {
    pub kind: HostExprKind<T, F>,
    pub span_id: Option<String>,
    pub merged_spans: Vec<String>,
}

pub enum HostExprKind<T = HostTypeTerm, F = HostFunctionValue<T>> {
    // existing variants, recursively using HostExpr<T, F>
    FunctionValue(F),
}

pub struct HostFunctionValue<T = HostTypeTerm> {
    pub params: Vec<HostParam<T>>,
    pub ret_ty: T,
    pub origin: HostFunctionValueOrigin<T>,
}

pub enum HostFunctionValueOrigin<T = HostTypeTerm> {
    Reference {
        name: String,
    },
    Closure {
        body: Box<HostExpr<T>>,
        captures: Vec<HostCapture<T>>,
    },
    Transformed {
        transform: HostFunctionTransform,
        input: Box<HostExpr<T>>,
    },
}

pub struct HostCapture<T = HostTypeTerm> {
    pub name: String,
    pub ty: T,
}

pub enum HostFunctionTransform {
    Grad {
        wrt: Option<Vec<usize>>,
    },
    Vmap {
        axis: usize,
    },
    VmapGrad {
        wrt: Option<Vec<usize>>,
        axis: usize,
    },
    Jit,
}
```

`HostBinding`, `HostFunction`, `HostCallback`, `HostMatchArm`, and
`HostProgram` thread the same `F` parameter through every nested expression.
Their default remains `HostFunctionValue<T>`, keeping shared term/concrete
call sites concise.

Shared programs therefore use:

```rust
HostProgram<HostTypeTerm, HostFunctionValue<HostTypeTerm>>
HostProgram<ConcreteHostType, HostFunctionValue<ConcreteHostType>>
```

The private C program uses:

```rust
HostProgram<HostAbiType, std::convert::Infallible>
```

Projection cannot construct an ABI-program function-value node. The emitter's
exhaustive `FunctionValue(Infallible)` arm eliminates the impossible payload
with `match value {}`; it is not a late unsupported fallback.

### 4.2 Meaning of each origin

- `Reference` identifies a named definition, a function parameter, or a local
  callable binding. Its exact signature lives on `HostFunctionValue`.
  Contextual C adaptation still checks the backend projection's closed
  allowed-symbol set; a name alone never grants callback eligibility.
- `Closure` stores the typed parameter list, typed result, lowered logical
  body, and explicit typed captures. Captures are by value. The checker has
  already applied `spec/04-type-system.md` §8.3: capturing a tensor consumes
  the outer binding, borrow escapes are invalid, and inserted copy/drop nodes
  remain visible in the body where required.
- `Transformed` stores normalized checked transform configuration and a typed
  function-valued input. It never stores raw `grad`/`vmap`/`jit` syntax.
  Direct applications may still be specialized away before this node is
  needed; the node represents the value-producing form that survives.

The enclosing `HostExpr` owns the source span. The function-value payload does
not duplicate it. Branching remains structural: an `If` or `Match` whose
branches are function values remains an `If` or `Match` over
`FunctionValue` children. Aggregate and collection storage likewise uses the
existing container node with function-valued children.

### 4.3 Capture discovery

Capture discovery must not duplicate the checker with a second scoping model.
Extract the current binder-aware free-variable walk from
`chelis-types::linearity` into a workspace-visible ordered helper and use it
from both linearity and host lowering.

The helper:

- understands function parameters, sequential `let` binding, match-pattern
  binders, nested functions, and metadata wrappers;
- returns the first lexical occurrence of each free name, with no
  `HashSet`-iteration ordering;
- intersects the result with the checked lexical scope, excluding builtins and
  globally resolved definitions;
- looks up every capture's checked `HostTypeTerm`; and
- raises a shared resolution/lowering diagnostic if a capture has no checked
  type instead of inventing an inference variable or deferring the question
  to a backend.

Tests lock shadowing, nested closures, sequential-let scope, match binders,
duplicate uses, deterministic ordering, tensor capture ownership, and the
missing-type failure.

### 4.4 Logical resolution

`HostTypeTerm -> ConcreteHostType` resolves every function signature,
capture, closure body, transform input, and nested expression recursively.
An unresolved signature or capture produces `HostTypeResolutionError`.
Function values never become decode/resolution errors, `Unit`, an unresolved
call marker, or an ABI type at this boundary.

## 5. Private C projection

Projection first handles the contextual cases:

- A function-typed declared parameter uses
  `HostAbiType::try_callback_signature`.
- A function-typed call argument is accepted only when its shared node is a
  `Reference` naming a declared function or an in-scope callback parameter.
- `HostCallbackKind::Inline` remains the existing contextual body-inlining
  path for higher-order collection operations.
- Immediate lambdas and direct transform applications continue through their
  current beta-reduction/specialization paths.

Any remaining `HostExprKind::FunctionValue` returns one error from the private
C projection:

```text
kind:  HostAbi
stage: codegen:c
owner: chelis#879
span:  the enclosing HostExpr location, when present
```

This includes named and anonymous returns, lexical captures, dynamic
selection, local binding, tuple/ADT/option/list/dict storage, and stored or
returned transform-produced values. A named function stored in an ADT and an
inline closure stored in the same field therefore have identical capability
classification and diagnostic stage.

`HostAbiType::try_from_concrete_at` takes the containing value's provenance
for type-driven decisions. Binding types use the binding value location,
return types use the function body location, expression result/container
types use the current expression location, and function parameters use a
new optional `span_id` on `HostParam`. `HostCallback` also carries its
originating optional `span_id` so contextual-signature failures can name the
callback expression. Logical types remain span-free.

## 6. Unsupported diagnostic provenance

[#868] is repaired as a general `Unsupported` channel correction before the
function-value move.

### 6.1 Construction and rendering

`Unsupported::new` becomes
`new(what, context, stage, span: Option<SpanRef>, hint)`. Every convenience
constructor requires the same explicit optional location. The unused
post-construction `with_span` method is removed. This makes `None` a reviewed
producer decision instead of an invisible default.

The legacy spanless rendering remains byte-identical:

```text
unsupported: <what> on <context> (<stage>); <hint>
```

When a location exists, the location clause appears before the semicolon:

```text
unsupported: <what> on <context> (<stage>) at source span `<id>`; <hint>
unsupported: <what> on <context> (<stage>) at bytes <start>..<end>; <hint>
unsupported: <what> on <context> (<stage>) at byte <offset>; <hint>
```

A non-empty `span_id` wins the textual rendering; structured output may carry
both ID and byte range. An empty ID falls back to the numeric form or to the
spanless legacy form. Backslashes, backticks, line terminators, and control
characters in opaque IDs are escaped so one diagnostic remains one line.
A length without an offset is retained structurally but has no textual
location. Numeric end calculation uses checked addition; overflow falls back
to the offset-only form rather than panicking.

### 6.2 Compiler API

`schema::Diagnostic` gains:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub span_id: Option<String>,
```

The existing `span: Option<schema::Span>` remains and is populated when an
offset and length are both present. `unsupported_stage_error` destructures
the `Unsupported` once, renders it with its location, and moves both
structured location forms into the compiler error envelope. Generic
`stage_error` remains the spanless convenience path; its diagnostic
construction initializes `span_id: None`.

CLI reporting consumes the same compiler diagnostic. No CLI-only scan or
message parser becomes the provenance authority.

### 6.3 Producer sweep

The span implementation audits every production `Unsupported` constructor in
C host/DAG emission, HIP, Metal, shared lowering, and compiler gates:

- a rejection with a raw Deep `Expr` supplies byte offset, byte length, and
  `expr.span_id()`;
- a DAG or `HostExpr` rejection supplies its existing `span_id`;
- type-driven C decisions receive the containing expression/parameter
  provenance as specified in §5; and
- runtime/infrastructure cases that genuinely have no source location pass
  `None` and appear in a reviewed negative allowlist.

No producer parses a `surf:<start>..<end>` opaque ID back into a byte range.

## 7. Tests and invariants

Test stubs land before implementation, with positive/negative parity.

### 7.1 Span phase

- Exact byte compatibility for a spanless `Unsupported`.
- Span-ID, numeric-range, offset-only, both-fields, empty-ID, Unicode, and
  control-character rendering.
- Compiler API and JSON shape for `span`, `span_id`, both, and neither.
- Public `compile` and `compile_for_execution` rejection localization.
- CLI stderr localization without prose parsing.
- Representative C host-type, C host-expression, C DAG, HIP, and Metal
  rejection paths.
- A source/constructor census that fails on an unreviewed implicit spanless
  producer.
- A negative test proving a source-free runtime/infrastructure rejection does
  not fabricate a location.

The authoritative oracle is:

```sh
.venv/bin/python scripts/unsupported_span_oracle.py
```

It exits zero only after the focused Rust/API/CLI suites and producer census
pass, and ends with `UNSUPPORTED SPAN ORACLE: PASS`.

### 7.2 Function-value boundary phase

Shared-lowering positives inspect the IR for:

- returned named and anonymous functions;
- nested capturing closures;
- named and anonymous functions stored in the same ADT field;
- tuple, option, list, and dictionary storage;
- local binding and dynamic `if`/`match` selection;
- stored/returned `grad`, `vmap`, `vmap-grad`, and `jit` values; and
- exact `int8`/`int16`, symbolic-dimension, capture, origin, and span data.

Resolution negatives cover every unresolved signature/capture/body position.
Capture tests cover nesting, shadowing, lexical order, and ownership.

C-projection negatives assert the same `HostAbi`/`codegen:c`/#879/span
classification for every first-class form through both public compiler APIs
and CLI `c`, `hip`, and `metal` host-fallback builds. No failed projection may
produce a source, header, or runtime artifact.

Positive regression tests compile and run generated C for:

- exact named `int8` and `int16` callbacks;
- forwarded callback parameters;
- inline higher-order collection callbacks;
- immediately applied lambdas; and
- direct `grad`/`vmap` specialization.

Endpoint tests reject any `void *` function erasure, placeholder zero,
unresolved callable marker, raw unresolved `call(...)`, or ABI-program
function-value payload.

The authoritative oracle is:

```sh
.venv/bin/python scripts/host_function_value_boundary_oracle.py
```

It exits zero only after the focused Rust/API/CLI suites, generated-C
compile/run checks, and endpoint scans pass, and ends with
`HOST FUNCTION-VALUE BOUNDARY ORACLE: PASS`.

## 8. Implementation sequence

### FV0 — this specification lock

Land this document and synchronize the active capability, unsupported,
span-survival, transformations, roadmap, canonical-reference, surface, and
oracle documents. File [#879]. Do not claim that [#866]-[#868] are fixed.

### FV1 — unsupported span channel ([#868])

Write the §7.1 tests and oracle first. Then change `Unsupported`
construction/rendering, compiler diagnostic schema and envelope plumbing,
producer sites, and positional host projection APIs. Update this document and
the synchronized status banners from `pending` to `implemented` only when the
oracle and repository gate are green.

### FV2 — shared function values and C boundary ([#866], [#867])

Write the §7.2 tests and oracle first. Then add the typed shared
representation, ordered capture discovery, term-to-concrete resolution, and
private C rejection. Repoint callable hints from #730 to [#879]. Update the
capability/current-state documents only after the oracle, repository gate,
and a fresh-context execution red team are green.

### FV3 — general C-host ABI ([#879])

Choose and specify closure conversion or defunctionalization before code.
This later phase implements environment representation, ownership, storage,
return, and indirect application. It flips the `c-host /
FirstClassValue` cell from `Unimplemented` to `Implemented`; FV1/FV2 do not.

## 9. Non-goals

- No language-level closure prohibition or checker rejection.
- No raw unlowered Deep syntax in shared host IR.
- No general C closure ABI, environment allocation, or indirect application
  in FV1/FV2.
- No removal of exact contextual callback and specialization fast paths.
- No span fields on `HostTypeTerm`, `ConcreteHostType`, or `HostAbiType`.
- No backend guessing of missing capture types.
- No claim that HIP or Metal has a native host function-value ABI while its
  host route delegates to C.

[#799]: https://github.com/Chelis-Lang/chelis/pull/799
[#866]: https://github.com/Chelis-Lang/chelis/issues/866
[#867]: https://github.com/Chelis-Lang/chelis/issues/867
[#868]: https://github.com/Chelis-Lang/chelis/issues/868
[#879]: https://github.com/Chelis-Lang/chelis/issues/879
