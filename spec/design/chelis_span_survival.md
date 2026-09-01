# Chelis Span Survival: End-to-End Audit Chain

**Status:** S0-S6 are the shipped external-audit contract. S7 is the active
source-identity correction owned by chelis#1172; until its oracle is green,
Surf ranges survive as metadata but are not a source-qualified semantic
identity.
**Owners:** chelis-core (this repo). Octant ships span-attributed Deep upstream.
**Companion specs:** `spec/03-deep-syntax.md` §1.1.1 (the `span` key + `span_*` namespace + synthesized markers); `spec/design/chelis_trust_stack.md` (audit story).

---

## 1. Purpose

Octant emits span-attributed Deep with a sidecar `.spans.json` mapping each span ID to
a LaTeX byte range. Octant's tests confirm spans are correct in the *emitted* Deep.
Chelis preserves that audit chain (LaTeX byte range → Deep node → IR node / HostExpr
→ backend output line) so a runtime trace or generated C source line can be traced back
to the original external-source byte range. This document specifies the shipped
contract and preserves the phased implementation record that established it.

External-source agnosticism: the span ID is a string. Octant happens to use
LaTeX-derived dot-paths (`n_001`, `expr.5.lhs`); other producers may use
other conventions. Chelis preserves the strings and stays
producer-interpretation-agnostic.

Surf parsing participates in the same contract during desugaring: when a
Surf AST node carries a real parser byte range, the desugarer writes a
Deep `span` ID of the form `surf:<start>..<end>` onto the corresponding
Deep node. Hand-constructed Surf nodes with the zero-length sentinel are
treated as spanless, so synthesized-marker fallbacks remain limited to
genuinely absent source ranges.

That metadata path is an audit label, not a complete source-location model.
The current desugarer constructs most Deep nodes with `Span::new(0, 0)`, and
the checker/lowerer reuse map is keyed by the bare numeric offset. Two source
units can therefore alias, while synthesized nodes can masquerade as source
offset zero. Section 2.6 replaces those representations; no consumer may
parse `surf:<start>..<end>` to reconstruct semantic identity after S7.

## 2. Contract

### 2.1 Deep-side (already shipped)

Deep AST nodes carry an arbitrary metadata map at element 1 (per
`spec/03-deep-syntax.md` §1). The `span` key is a documented active key
holding an opaque string identifier (see §1.1.1 of that spec). The chelis
Deep parser preserves arbitrary metadata; the printer emits it canonically;
round-trip through `chelis fmt <file>.dp` is loss-free for `span` values.

### 2.2 IR-side (S2 + S3)

Every IR `DagNode` carries two new fields:

```rust
pub struct DagNode {
    // …existing fields…
    pub span_id: Option<String>,         // canonical span (origin of this op)
    pub merged_spans: Vec<String>,       // additional spans from N→1 merges
}
```

`span_id` is the canonical origin span; `merged_spans` carries additional
spans accumulated when transformation passes merge multiple source nodes
into a single result (Fusion, CSE, constant fold). The audit invariant is:

> Every span ID present on any input Deep node appears either as `span_id`
> or as a `merged_spans` entry on at least one node in the final IR (and
> thus, after S4, as a `// span:` comment in the generated C/HIP/Metal
> source).

**This is locked architecturally.** Collapsing back to a single
`Option<String>` would force a "winner span" on every N→1 merge and silently
drop the others, contradicting the audit invariant. Reject simplification
proposals on those grounds.

### 2.3 Per-pass propagation rules

| Pass | Rule | Synthesized-node rule |
|------|------|------------------------|
| Lowering (Deep→IR) | `span_id` inherited from `meta["span"]` of the Deep `Expr` being lowered. | (a) **1→N (one Deep expr → multiple IR nodes):** all introduced nodes inherit that expr's span (region-corresponding). (b) **N→1 lowering collapse (multiple Deep exprs → one existing IR node):** when a parent Deep expr (e.g., a `(def {span: a} name body)`) lowers to a body that already corresponds to an existing IR node, append the parent's `span_id` to that node's `merged_spans` (lex-sorted, deduped). This uses `merged_spans` for its design-stated purpose — preserving the audit chain through every contributing source span — and is the lowering analogue of the same rule used by Fusion / CSE / fold during S3. |
| Constant fold | Replacement (folded) node inherits the **operation node's** `span_id`. If operands carried distinct span IDs that differ from the operation's, those operand spans append to `merged_spans` (lex-sorted). | Const-result node = synthesized at the operation's span. No `__synthesized_*__` marker — the operation node had a real source span. |
| DCE / remap | Pure copy; clone `span_id` and `merged_spans`. | n/a |
| CSE | Keep first node's `span_id`; append duplicate's `span_id` to `merged_spans`. | n/a |
| Tier 2 decomposition | Synthesized sub-nodes inherit the decomposed parent's `span_id`. | If parent had no span, mark sub-nodes with `__synthesized_tier2__`. |
| Automatic differentiation | Forward nodes: `span_id` cloned. Backward (adjoint) nodes: canonical `span_id = "__synthesized_grad__"`, plus the forward node's `span_id` appended to `merged_spans`. | All backward nodes are synthesized. |
| Fusion | `FusedElem` `span_id` = first contributor's `span_id`; `merged_spans` = remaining contributors' spans (lex-sorted). | n/a |
| Vmap | Clone `span_id` and `merged_spans` unchanged. | n/a |
| Verify / Eval | Read-only; no propagation. | n/a |
| Codegen | Read-only consumer; emits one `// span:` line per `span_id ∪ merged_spans` (canonical first, `merged_spans` lex-sorted). | n/a |

Synthesized markers always carry forward-node spans alongside on
`merged_spans`. A node with `span_id = "__synthesized_grad__"` and empty
`merged_spans` is invalid — the grad pass MUST attach the forward span.

#### Host-side passes (S6)

The host-side AST (`HostExpr`) carries the same `span_id: Option<String>`
+ `merged_spans: Vec<String>` schema as `DagNode`. The host-side rule
table mirrors the DAG-side table above for the lowering and rewriting
passes that produce `HostExpr` trees. Backend host-side emission (the
`// span:` comment block emitted by `chelis-backend-c`'s `host_emit.rs`)
is the read-only consumer; it follows the same shape as the DAG-side
codegen rule.

| Host-side pass | Rule | Synthesized-node rule |
|----------------|------|------------------------|
| Lowering (Deep→`HostExpr`) | `span_id` inherited from `meta["span"]` of the Deep `Expr` being lowered. | (a) **1→1 / 1→N:** every freshly-produced `HostExpr` inherits the enclosing Deep expr's span (region-corresponding). Implemented via `lower_host_expr`'s top-level wrapper that stamps `expr.span_id()` onto the result before returning. (b) **N→1 lowering collapse:** when a parent Deep expr lowers to a body that already corresponds to an existing `HostExpr` (e.g. `(realize {} x)`, `(handle-effect {} seed body)`, `(lit {span: a} child)` — all of which return the inner lowered child verbatim — and the top-level `(def {span: a} name body)` → `HostBinding`/`HostFunction.body` collapse), the parent's `span_id` appends to the existing node's `merged_spans` via `HostExpr::append_merged_span` (lex-sorted, deduped, None-safe, canonical-equal-no-op). Same primitive shape as the DAG-side rule (b). |
| `Let`-binding lowering | Each binding's value is a fresh `HostExpr` carrying that binding's source span (region-corresponding). When the surrounding `(bind {span: b} name value)` carries its own span, that span appends to the value HostExpr's `merged_spans` per rule (b) (the bind wrapper collapses onto the value). The enclosing `Let` node's `span_id` is the parent `(let ...)` Deep expr's span. | n/a — no synthesis. |
| `If` / `Match` lowering | The result `HostExprKind::If` / `MatchOption` / `MatchAdt` is a fresh node carrying the enclosing `(if ...)` / `(match ...)` Deep expr's span. Arm bodies recurse through `lower_host_expr` and so each arm's `HostExpr` carries the arm-body's own span. | n/a — no synthesis. |
| `Cast` / `Builtin` rewriting | Rewriting `(cast {} x)` → `HostExprKind::Builtin { name: "cast", args: [x] }` and similar `(copy {} x)` rewrites: the result `Builtin` node's `span_id` is the enclosing operation's span (region-corresponding). The argument's own span lives on the argument `HostExpr`'s `span_id` and travels with it. No append to the operation's `merged_spans` — the argument is a child node, not a merged contributor. | n/a — `cast` / `copy` are real source operations. |
| `(grad ...)` / `(vmap ...)` / `(vmap-grad ...)` host-position fallback | Produces the unspellable unresolved-callee marker (`HOST_UNRESOLVED_TRANSFORM_MARKER`, `#chelis-unresolved-transform`, for a transform in callee position; `HOST_UNRESOLVED_CALLABLE_MARKER` otherwise) as the fallback `Builtin`/`Call` name so `host_program_unresolved_transform_sites` / `host_program_unresolved_call_sites` can surface a clean pre-codegen error (chelis#841). The result's `span_id` is the source `(grad ...)` Deep expr's own span (region-corresponding rule), set by `lower_host_expr`'s wrapper. The marker lives in the name field — it is NOT a synthesized `span_id`. | n/a — the result has a real source span. The marker name is a payload string, not a synthesized span marker. |
| Tensor-helper extraction (`try_lower_tensor_helper_call` / `finish_tensor_helper_call`) | The produced `HostExprKind::TensorCall { helper, args, .. }` wraps an extracted DAG. The result's `span_id` is the originating Deep expr's source span. Two callsite shapes produce a `TensorCall`: (1) `lower_host_expr_kind` invokes `try_lower_tensor_helper_call` and the surrounding `lower_host_expr` wrapper stamps the source span onto the result; (2) `lower_host_function` invokes `try_lower_tensor_helper_call` directly on the fn-body inner expr (`body_expr = kids[1]`), bypassing the wrapper — this path appends `body_expr.span_id()` (the inner expr's span) AND `body.span_id()` (the surrounding `(fn …)` form's own meta-span) to the result's `merged_spans` after the if/else, alongside the def's outer `expr.span_id()` appended by `lower_host_program`. All three appends use `HostExpr::append_merged_span` (lex-sorted, deduped, None-safe, canonical-equal-no-op). The helper's INTERNAL DAG nodes carry their own per-node spans via DAG-side lowering rules (S2). | n/a — TensorCall has a real source span. |
| Top-level fn inlining (`inline_top_level_host_call` and the `lower_app_host_expr` HOF specialization path) | When a callee is inlined for HOF specialization (e.g. `grad(local_fn)(theta)`), the inlined-substituted body is re-lowered through `lower_host_expr`. Each lowered node carries its own region-corresponding span via the rule above; the call-site's outer span (the `(app ...)` expr) appends to the result's `merged_spans` per rule (b) (the call-site collapses onto the inlined body). | n/a — the inlined body is real source. |
| Argument hoisting (`hoist_host_lane_tensor_bindings`) | Hoisted `__host_tensor_arg_<index>` bindings live inside a synthetic `HostExprKind::Let` whose `span_id` is the original `(app ...)` expr's span (region-corresponding via `lower_host_expr`'s wrapper). Each hoisted binding's value carries its own argument-expr span. | n/a — the wrapper Let inherits the originating `(app ...)` span (a real source region). |
| Type-refinement passes (`refine_host_expr_types`, `refine_host_function_signatures`, `refine_host_globals`, `propagate_named_callback_signatures`) | Type-only updates; `span_id` and `merged_spans` are pure copies (mutation in place, never reset). | n/a — no synthesis. |
| Host emit (S6 step 5/8 — read-only consumer) | Backend host-side emitter prepends one `// span:` line per `span_id ∪ merged_spans` immediately before the C statement(s) implementing the node. Order: canonical first, then `merged_spans` lex-sorted (deterministic, reproducible across runs). Out of scope for S6 steps 1-4; landed in S6 step 5. | n/a |

**Synthesized-marker invariant (forward-looking).** Today's host-side
passes do not mint `__synthesized_<host_pass>__` span IDs — every
shipped rule above resolves to a real source span via the
region-corresponding rule. The invariant is locked architecturally for
when host-side passes (e.g. a future host-side fusion or HOF specializer
that synthesizes wrapper nodes with no source region) DO need synthesized
markers: any `HostExpr` with `span_id` matching `__synthesized_<host_pass>__`
MUST have non-empty `merged_spans` carrying the originating source span,
mirroring the same invariant the S3 grad/Tier 2 passes enforce on
`DagNode`. Locked via a test that constructs such a HostExpr
programmatically and asserts the invariant.

### 2.4 Backend emission (S4 + S6)

Each backend emits one `// span: <id>` line per `span_id ∪ merged_spans`
immediately preceding the line(s) that implement the operation. Order:
canonical first, then `merged_spans` lex-sorted (deterministic,
reproducible across runs).

There are two emission paths:

| Path | Reads | Owning module | Phase |
|------|-------|---------------|-------|
| DAG path | `DagNode.span_id` + `DagNode.merged_spans` | `chelis-backend-c::emit::CEmitter::emit_span_comments` | S4.1 (C), S4.2 (HIP), S4.3 (Metal) |
| Host path | `HostExpr.span_id` + `HostExpr.merged_spans` | `chelis-backend-c::host_emit::HostEmitter::emit_span_comments` | S6 step 5 |

The two helpers share the same shape, ordering rule, and
defense-in-depth sanitizer
(`chelis_ir::span_sanitize::sanitize_for_comment`). Programs lower into
both worlds: the DAG path runs for tensor-shaped operations
(elementwise tensor algebra, reductions, matmul, etc.) and the host
path runs for scalar / list / ADT / control-flow code (the ambient
glue around tensor cores). Per spec §2.4.2, both paths emit on the
host-side `.c` / `.cpp` / `.mm` source — the audit-trace resolution
target — so the audit chain is recoverable regardless of which path a
particular operation lowers through.

HIP and Metal currently emit DAG-path spans only (per S4.2 and S4.3);
host-path emission for HIP / Metal is not in S6 scope. If a future
pure-scalar program routes through HIP or Metal host-side scaffolding,
extending host-path emission to those backends is a forward-looking
hardening item.

#### 2.4.1 Embedded device-kernel source strings (HIP / Metal)

Per-node kernels — one IR node, one kernel definition (e.g., HIP's
`kernel_fused_<id>` and Metal's per-node entry points) — carry their
spans inside the embedded kernel source string, immediately preceding
the per-op kernel-side emission, with the same shape and ordering as
the host-side comment block.

Shared kernels — one kernel definition, many IR-node launchers (e.g.,
HIP's `kernel_neg`, `kernel_add`, `kernel_sum_ax0`) — carry NO spans
inside the kernel source string. Each IR-node launcher's host-side
launch site carries the spans for that specific launch.

#### 2.4.2 Principle — spans live wherever audit traces resolve

The rule above isn't an asymmetry for its own sake. It encodes a
principle that future backends and future kernel-emission patterns
should also follow:

> **Spans live wherever audit traces resolve.**

Audit traces (a runtime event, a stack frame, a profiler hit, a
generated source line in a debugger) resolve to host-side source
lines — the `.c` / `.cpp` / `.mm` file that customers, profilers, and
debuggers read. Embedded kernel source is a runtime artifact compiled
by the GPU driver; it is not itself an audit-trace target.

This is also why §2.4 has both a DAG path and a host path: a scalar /
control-flow / glue operation lowered to a `HostExpr` resolves to the
exact same host-side `.c` line a tensor operation lowered to a
`DagNode` does. Both paths emit `// span:` comments immediately
preceding the implementing C statement so the audit chain is
recoverable from either.

- Per-node kernels CAN carry meaningful spans because there is a 1:1
  correspondence between the kernel definition and the IR node, so
  spans inside the kernel body unambiguously attribute to the same
  source region as the launching node.
- Shared kernels CANNOT carry meaningful spans because the same kernel
  body serves many launchers; aggregating launcher spans inside the
  kernel body produces noise (every launcher's spans appear regardless
  of which launch is currently executing) rather than per-launch
  attribution. The host-side launch site is the unambiguous,
  per-launch-correct place for the spans.

If a future backend introduces a new artifact kind (e.g., a SPIR-V
intermediate, a precompiled `.metallib`, a kernel cache distributed
separately from the host source), this principle answers what should
happen with spans there: if customers will trace through the artifact
to find original source, spans go in it; if the artifact is a runtime
build product not on the audit-trace path, spans don't.

### 2.5 CLI surface (S5)

`chelis build` adds a `--deep` flag and auto-detects `.dp` extension:

| Invocation | Behavior |
|------------|----------|
| `chelis build foo.dp` | Deep path (auto-detect by extension). |
| `chelis build foo.dp --deep` | Deep path (flag agrees with extension; no-op). |
| `chelis build foo.ch --deep` | Deep path with explicit override; warn if the file is clearly Surf source. |
| `chelis build foo.ch` | Surf path (today's behavior). |

`--no-deep` is intentionally NOT added. If a user needs to force a `.dp`
file through the Surf path, rename the extension or pipe through
`chelis surf`. Conflicting cases (e.g., `--deep` with garbage in the file)
produce the parse error from `chelis-deep`, not a silent fallback to Surf.

### 2.6 Source-qualified identity (S7; chelis#1172)

No numbered specification currently defines source-unit identity, qualified
node identity, or its generated-source representation. This section records
the implementation invariants that S7 must submit to the normative tier; it
does not allocate wire spellings, digest byte layouts, or internal adapter and
cache names. S7.0 separates two kinds of inventory instead of turning current
implementation structure into timeless semantics:

- `spec/03-deep-syntax.md` owns identity lifetime, derivation, equality,
  provenance, and generated-source representation. Any pass/builder token
  serialized into that public representation belongs in an identity-keyed
  normative registry there.
- Sealed implementation descriptors own private parser adapters, transformation
  entry points, and source-bearing cache codecs. They are closed by Rust types
  and exhaustive matches, not copied into a numbered-spec table or certified
  by a hand-written list.

The exact shared diagnostic/check-result contract remains owned by #886, as
recorded by tracker #883. Its controlling amendment belongs in
`spec/04-type-system.md` §6.4; `spec/09-tide.md` owns only Tide-specific
HTTP/MCP/batch placement. #1172 supplies opaque typed identity values and
referential-integrity rules but does not define their JSON placement.

S7.0 is therefore a mandatory prerequisite. A separate numbered-spec change
must establish the identity lifetime and derivation rules, any public token
registry, the generated-source encoding, and the boundary with #886 before
implementation or public output changes land. It does not promise executable
bijection against APIs that have not yet acquired typed identity entry points.
Because
`spec/03-deep-syntax.md` is frozen by the numeric-remediation guard, that
prerequisite must also follow the full freeze protocol named in S7.0.

The source model has three independent concepts. They must not share one
string or integer field:

```rust
pub struct SourceUnitId(private::Digest32);
pub struct LocalNodeId(private::NonZeroLocal);
pub struct NodeKey(private::ArenaIssuedKey);
pub struct SourceSite(private::ValidatedSite);
pub struct NodeOrigin(private::SealedOrigin);
pub struct NodeRef(private::ArenaIssuedNodeRef);
pub struct DiagnosticSourceRef(private::ValidatedDiagnosticRef);

pub struct SourcedSurfProgram {
    tree: surf::Program,
    ledger: private::SurfSourceLedger,
    evidence: private::IssuanceEvidence,
}

pub struct SourcedDeepProgram {
    tree: deep::Program,
    ledger: private::DeepSourceLedger,
    evidence: private::IssuanceEvidence,
}
```

The implementation preserves the dependency direction. A new dependency-leaf
crate, `chelis-source`, owns the opaque identity/site types, typed paths,
`SourceInput<R>`, sealed evidence traits, and cache-validation contexts without
depending on either AST crate. `chelis-surf` owns `SourcedSurfProgram` and the
generated Surf traversal. `chelis-deep` owns `SourcedDeepProgram` and the
generated Deep traversal. The already downstream Surf desugarer may therefore
consume the Surf carrier and construct the Deep carrier without making
`chelis-deep` depend on `chelis-surf` or introducing a crate cycle. Checker,
pipeline, and cache crates consume the Deep carrier downstream of both.

These are opaque values with read-only accessors. They have no public field,
constructor, unchecked `Deserialize`, or `From` path from their component
parts. Only a `SourceArena` can issue a key and seal its origin into a
`NodeRef`; only a parser-owned source context can issue a `SourceSite` before a
node exists. A raw decoded tree is `UntrustedSurf` or `UntrustedDeep`, never a
sourced program. Promotion does not trust a decoded ledger or a digest computed
from it. It first reconstructs the expected issuance from independently
anchored evidence, then validates all of the following in one boundary
operation:

- every structural node occurrence has exactly one ledger row and every row
  resolves to exactly one occurrence;
- every identity-bearing Deep row has one unique key whose embedded unit
  belongs to the issuing arena; Surf rows bind canonical Surf paths to exact
  parser-issued sites and do not fabricate semantic `NodeKey`s before
  desugaring;
- the reconstructed parsed ledger comes from the exact retained source bytes,
  typed logical-input identity, parser schema, and canonical Surf or Deep
  traversal, rather than from serialized key/origin rows;
- the reconstructed transformation ledger comes from the already validated
  input ledger, sealed pass identity and complete configuration, semantic
  output, and that pass's exhaustive output-path attribution function;
- the reconstructed source-less ledger comes from the sealed builder identity,
  complete configuration, semantic output, and that builder's exhaustive
  output traversal;
- every decoded key, origin, external audit ID, and contributor set equals the
  independently reconstructed row. Recomputing stored digests after swapping
  otherwise valid origins or contributor sets cannot make promotion succeed.

The reconstructible evidence is a private, versioned recipe graph rooted only
in parsed source snapshots or registered source-less builders. A parsed root
retains the exact bytes and typed logical identity needed to rerun the parser
and traversal. A transformation root retains references to validated input
roots plus the sealed pass/configuration needed to recompute its output
attribution. Cache formats may deduplicate content-addressed source blobs, but
an unresolvable blob or recipe is a hard decode failure. A serialized digest or
caller-authored origin row is never an evidence root.

Nor may the serialized artifact choose which otherwise legitimate recipe is
expected. Every cache decode takes an `ExpectedSourceGraph<C>` constructed by
the current typed compiler request, package lock/source digests, and cache
kind's sealed pipeline schema. It fixes the logical inputs, exact source blobs,
pass order, pass configurations, builder roles, and output artifact kind before
bytes are decoded. Serialized recipe nodes are compared with that graph; they
do not select it. A standalone artifact for which the consumer cannot construct
that independent expectation remains untrusted and cannot expose node-keyed
facts. Thus changing source bytes, substituting another registered pass, or
changing a valid configuration and recomputing the full artifact still misses
the consumer-supplied expectation.

`SourcedSurfProgram` and `SourcedDeepProgram` expose no mutable raw tree that
could bypass those checks. Syntax consumers such as formatting and LSP indexing
receive a read-only `SourcedSurfProgram`; semantic consumers receive a
`SourcedDeepProgram`. Surf desugaring consumes the former and, through the
registered `SurfDesugar` transform, returns the latter while reconstructing the
Deep ledger from the Surf path-to-site ledger and the Deep output. The Surf
ledger therefore preserves source structure without inventing a checker/
lowering identity early; the Deep ledger is the first carrier of `NodeRef` and
`NodeKey`. Cloning a whole immutable artifact preserves the same tree-plus-ledger
identity; cloning or splicing a
subtree into a new occurrence is available only through a registered
transformation builder, which remints output keys and records contributors.
The migration removes `Clone`/`Deserialize` from identity-bearing node carriers
even if raw AST values retain them inside untrusted or pre-seal modules.

- `SourceUnitId` identifies one immutable input snapshot, not a logical module
  across edits. Its normative derivation binds a versioned domain, the exact
  source bytes, and a typed logical-input identity supplied by the registered
  ingress edge. Re-parsing unchanged bytes at the same logical identity
  reproduces the unit; changing either input mints another unit; equal bytes in
  two paths, request fields, or package modules remain distinct. Callers never
  inject raw unit or digest bytes. A byte offset or display filename is not a
  unit ID.
- `NodeKey` is the source-qualified semantic identity used by checker and
  lowering maps. Its representation is structurally
  `(SourceUnitId, NonZeroU64)`, so two arenas that both allocate local key `1`
  cannot collide. The parser/desugarer allocates every Deep node a nonzero
  local identity in canonical construction order. It is never derived from a
  byte offset, an external span string, or structural equality. Construction
  is private to `SourceArena`; a producer cannot restart a counter for an
  existing unit.
- "Structural occurrence" is defined by a versioned, generated
  `IdentityStructure` traversal over every Surf and Deep declaration,
  expression, pattern, type, effect, tagged/meta key and value atom, and
  compiler-authored wrapper/helper that can carry or be referenced by
  provenance. Its path is a typed sequence, never a debug/display string:
  `TopLevel(index, variant)`, `Field(owner_variant, field_tag)`,
  `Sequence(index)`, `MapKey(canonical_bytes)`, `MapValue(canonical_bytes)`,
  and `Variant(variant_tag)`. Top-level and sequence order are source/AST
  order; unordered maps are traversed by canonical key bytes and reject
  duplicate canonical keys. Scalar payload bytes participate in the owning
  node's semantic encoding; key/value atoms that can carry metadata or
  provenance receive their own path step and ledger row. Exhaustive generated
  matches make a new AST variant or structural field a compile failure until
  its traversal is declared. The grammar version is bound into source-unit,
  transformation, builder, and assignment derivations.
- Transitional Deep carriers are not silently normalized during promotion.
  A legacy `meta["span"]`, structural source site, and external audit ID may
  coexist only under an explicit versioned migration that proves their roles
  do not conflict and then emits one canonical occurrence graph. An unknown
  carrier, a conflicting legacy/structural range, or a path that the current
  grammar cannot represent is rejected before identity or cached facts are
  exposed.
- A later pass creates one deterministic synthesized arena for each distinct
  semantic output. The identity input includes a versioned domain, a pass ID
  from the closed pass registry, its schema version, a complete canonical
  encoding of every invocation option that can affect output, the semantic
  input/contributors, and the semantic output. Omitting pass configuration or
  identifying an invocation only by its input is forbidden: two different
  outputs must not alias. Private pass adapters compute the identity inputs;
  callers cannot inject a pass string, digest, contributor list, or local
  counter. Multi-unit helpers belong to a fresh synthesized unit and retain
  their actual contributor sites in `NodeOrigin::Synthesized`.
- Programmatic source-less construction follows the same rule. Its qualified
  unit binds a builder ID from the closed builder registry, builder schema and
  complete configuration, and the semantic output. Private builders attach
  keys in canonical traversal order after constructing that output. There is
  no process-global counter, zero unit, free-form builder namespace, or public
  raw-digest escape hatch. Nodes carry `NodeOrigin::Unavailable`.
- `SourceSite` is a human location. `Span` remains a byte range in exactly one
  identified source unit; it is not meaningful without that unit.
- `external_span_id` is the existing opaque producer audit label from Deep
  metadata. Chelis preserves it byte-for-byte but never parses it into a
  source range or uses it as a map key. A Deep textual parser range and an
  Octant/other-producer ID can coexist and remain distinguishable.
- `NodeOrigin::Unavailable` is an honest absence state for hand-constructed
  API inputs. It never renders as offset zero. Once a parsed source unit has
  entered the compiler, later passes may produce only `Parsed` or
  `Synthesized`; dropping to `Unavailable` is an invariant failure.
- `NodeRef` is the provenance carrier of an AST/IR node. A diagnostic uses
  `DiagnosticSourceRef` instead: lexer and parser errors may carry an exact
  `SourceSite` before any node exists, while a node-backed diagnostic may
  additionally carry its `NodeRef`. If both exist, `site` is the exact blame
  range and need not equal the node's wider parsed origin. No diagnostic
  fabricates a `NodeKey` merely to report a source range.

`NodeKey` and provenance are compiler identity, not language semantics.
`Expr::PartialEq`, canonical Deep printing, parse/reprint comparison, and the
semantic `expanded_deep_digest` exclude them. That exclusion never weakens a
cache carrying source sites or node-keyed facts. The two sourced carriers have
different, explicitly tagged assignment schemas:

- `SurfSiteAssignmentDigest` covers every Surf structural occurrence. Each
  reconstructed row contains its canonical Surf path, exact parser-issued
  `SourceSite`, and any opaque external audit ID. It contains no `NodeKey`,
  `NodeOrigin`, or semantic-fact key.
- `DeepNodeAssignmentDigest` begins only at direct Deep parsing or Surf
  desugaring and covers every identity-bearing Deep occurrence. Each
  reconstructed row contains its canonical Deep path, issued `NodeKey`, sealed
  origin, external audit ID, and contributor references.

Both digests bind the identity-schema and path-grammar versions and order rows
by structural path, not allocation or hash-map iteration order. The
Surf-to-Deep transform consumes the validated Surf site digest as evidence but
never reinterprets a Surf path/site row as a pre-existing semantic key.

Any source-aware artifact uses a provenance-sensitive `SourceIdentityDigest`
over the cache-format version, ordered source-unit table, semantic digest,
consumer-supplied expected-graph digest, issuance-recipe digest, and the
carrier-tagged `SurfSiteAssignmentDigest` or `DeepNodeAssignmentDigest`. A
syntax-only Surf cache carries the Surf form and cannot expose node-keyed
facts; any cache that carries such facts must carry the Deep form. Decode first
resolves the exact source blobs and
recipe graph, reparses parsed roots, replays every registered traversal or
attribution function, and derives the expected ledger without consulting the
decoded ledger rows or their stored digests. It compares the decoded tree,
ledger, and all stored digests with that reconstruction before exposing
linearity or diagnostic facts. Swapping two local IDs, swapping two valid
parsed origins, changing synthesized contributors, or recomputing every stored
digest after any such mutation therefore fails promotion. S7 inventories and
bumps every then-live cache format capable of retaining source declarations or
node-keyed facts; a sealed source-aware cache codec and exhaustive cache-kind
dispatch close that implementation inventory. Version numbers copied into
this design are not evidence of completeness.

#### Attribution rules

Surf AST nodes retain the parser's exact `SourceSite` through desugaring.
Every source-spelled Deep operation receives that site. Compiler-authored
tag/name/meta atoms and structural wrappers receive a distinct `NodeKey` and
`Synthesized { pass: SurfDesugar, contributors: [...] }`; they do not inherit
a fake textual range. A 1-to-N expansion cites the originating Surf site on
every product. An N-to-1 collapse carries the lexicographically sorted,
deduplicated union of contributor sites. A declaration-created helper cites
the whole declaration plus the exact body/parameter sites that determine it.

Later lowering and transformation passes apply the same rule already used by
`span_id`/`merged_spans`: a region-corresponding node retains `Parsed`, while
a new helper is `Synthesized` with every source contributor. Missing
provenance is not repaired with an enclosing range, offset zero, or a parsed
external ID.

#### Consumer rules

`CheckError`, lowering diagnostics, and compiler-api diagnostics carry the
typed `DiagnosticSourceRef`; AST, IR, and cached linearity records carry
`NodeRef`/`NodeKey`. These types preserve source unit, blame range, node key,
origin, contributor sites, and external audit ID as distinct values. Human
rendering may abbreviate them, but cannot invent a node for a parse error or a
range for `Unavailable`/`Synthesized`.

#886 owns the exact machine-readable envelope: top-level versus per-result
source tables, batch behavior, diagnostic field placement, literal unions,
`WireDeepExpr`/`WireDagNode` placement, null-versus-omission, and unknown-field
policy. S7.0 records the boundary and the identity value encodings needed by
that contract. #1172 implementation tests prove referential integrity and
source separation through typed values; byte-for-byte JSON fixtures land with
#886's controlling wire change, not from this design paragraph.

`LinearityInfo` is keyed by `NodeKey`, replacing
`HashMap<usize, usize>`. `mark_reusable_input` records the checked app node's
qualified key and lowering queries that same qualified key. Merging separately
checked programs is a validated identity join, not an unconditional disjoint
union. Distinct source units remain disjoint even when their byte offsets or
local counters match. The same qualified key from two independently loaded
copies of one immutable artifact is coalesced only when its reconstructed
source evidence, structural path, node semantic fingerprint, origin, and
node-keyed fact are identical; any disagreement is corruption and rejects the
composition. A diamond dependency therefore carries one validated copy of its
shared unit and facts. If a composition intentionally materializes the same
source subtree at two distinct output occurrences, it must use the registered
composition transform, remint both output keys, and retain the original sites
as contributors. It may not duplicate a key or choose a library-half-wins
policy.

Generated source keeps an opaque external audit ID separate from structural
source identity. S7.0 defines a versioned, comment-safe encoding whose tokens
come only from the normative public-token registry or sealed typed values;
display paths and caller-provided strings never enter it unchecked.

#### Sealed implementation closure and public tokens

S7 does not discover source-bearing code by names, request-field reflection, or
AST heuristics. It replaces each open construction path with one of four sealed
typed edges:

- `SourceInput<R>` binds exact bytes to a typed logical-input role `R` and is
  the only argument accepted by public Surf/Deep parsing. Surf success returns
  `SourcedSurfProgram`; direct Deep success returns `SourcedDeepProgram`;
  parser failure returns a validated `SourceSite` without a fabricated node.
  `SourceInput` has no public/default constructor and cannot derive logical
  identity from source bytes alone. A sealed adapter-owned
  `IngressContext<R>` constructs it from the exact bytes plus the typed file,
  request-field, batch-element, buffer-version, or package-member identity
  supplied by that adapter. Parser modules hold no ingress authority. The raw
  lexer/parser core is a private child module and additionally requires an
  unnameable `ParserLease` issued only by consuming a valid `SourceInput`, so
  another adapter—or a raw-input export in the parser module itself—cannot call
  it with bare `&str`/`&[u8]` or mint a default identity.
- `TransformContext<P>` exists only for a sealed registered pass `P`. Its
  private builder binds the pass schema, complete output-affecting
  configuration, semantic input/contributors, semantic output, and canonical
  node-assignment ledger before it can return a sourced output. In particular,
  `TransformContext<SurfDesugar>` is the only Surf-to-Deep edge: it consumes a
  `SourcedSurfProgram`, exhaustively maps Surf paths/sites to Deep output paths,
  and returns `SourcedDeepProgram`.
- `SourceLessContext<B>` does the same for a sealed registered builder `B`.
- `SourceAwareCache<C>` is the only codec admitted for an artifact containing
  a sourced program, `NodeKey`, `NodeRef`, or a node-keyed fact. The sealed
  cache kind `C` owns its format version and invokes evidence reconstruction,
  promotion, and digest comparison before returning a value.

Closed enums/exhaustive dispatch own the implementation sets for roles,
passes, builders, and caches. Adding a variant without its schema,
configuration encoder, identity builder, and both test polarities is a compile
failure. A hand-edited manifest cannot bless a row, and S7.0 does not copy
those private rows into the normative spec. Only a stable token that appears
in generated source or another #1172-owned public representation receives a
normative registry row in `spec/03-deep-syntax.md`; #886 separately owns tokens
and placements in its machine-wire contract.

The atomic adoption covers compiler-api request fields, direct public pipeline
helpers including `run_source`/`prepare_source`, prove helpers and MCP
`chelis_prove`, HTTP/MCP/batch request fields, CLI files/stdin/imports, LSP
buffers, Reef package composition, parser entry points, tree decode/composition,
and source-bearing caches. Every source field receives a typed logical identity
distinct from its display label. Multi-buffer requests keep every field
distinct, batch identities bind the element and field, and no public adapter
can recover a raw parser or construct a sourced tree from `Vec<Expr>`.

Closure against future bypass exports is mechanically derived, not asserted by
a fixed fixture. A Python S7 API-surface guard derives every workspace Rust
library target from `cargo metadata --workspace`, generates rustdoc JSON for
all of them, resolves aliases and generic/result wrappers, and classifies
signatures by type shape rather than item name. Raw input is classified
independently of raw output: any public callable whose result contains a raw or
sourced Surf/Deep program or a qualified Surf/Deep parse failure must take
`SourceInput<R>`, never `&str`, `String`, `Cow<str>`, `&[u8]`, `Vec<u8>`,
`Box<[u8]>`, or an alias of those closed raw carrier families. Thus both
`fn parse(&str) -> Vec<surf::Decl>` and
`fn parse(&str) -> SourcedSurfProgram` fail.

The same guard rejects any semantic consumer that accepts raw AST instead of a
sourced carrier, any public constructor/decoder for identity-bearing parts,
and any cache decoder that returns facts without a sourced carrier. These
closed rules classify the complete cargo-derived public universe in the same
run; there is no crate list, expected-row baseline, or editable allowlist. A
new public parser, adapter, AST consumer, or identity constructor therefore
fails the guard even when no role/pass/cache enum was edited.

A second, body-aware leg derives the reverse Rust call graph from the same
cargo-metadata universe. Its typed roots are the private `ParserLease` consumer
and sealed `IngressContext::issue` methods, not function-name patterns. Every
public ancestor that can reach either root must accept `SourceInput<R>` or a
registered request/file/buffer carrier whose type contains the required
logical-instance identity; a direct raw text/byte parameter is forbidden even
when the callable returns only a score, diagnostic, or other non-AST result.
This is the mechanically derived check for `run_source`, `prepare_source`, and
prove/MCP-style adapters that the result-shape leg cannot identify. Adding a
new caller changes the reverse graph in the same build and cannot be hidden by
leaving a manifest row unchanged.

Compile-fail controls reject public construction/recombination of identity
parts and an unregistered pass/builder/cache; the derived API-surface guard,
rather than a hand-maintained compile-fail example, proves that no raw public
parser or AST-consuming semantic edge exists. Runtime mutation controls reject
duplicate keys, a key moved to another structural occurrence, forged or
swapped origins/contributors even after every stored digest is recomputed, an
unresolved evidence root, and a cache whose source, semantic, recipe,
assignment, or format digest changed.

No fallback reparses `surf:<start>..<end>`, uses `Expr::span().offset` as an
identity, or treats `Span::new(0, 0)` as both a real location and a missing
sentinel. Those representations are deleted from semantic consumers when S7
lands.

## 3. Phasing

Each phase has one named acceptance oracle. A phase is not done until the
oracle is green.

### S0 — Spec record + canary fixture + HIP toolchain side-quest

1. This document committed as the canonical span-survival design doc.
2. `spec/03-deep-syntax.md` §1.1.1 documents the `span` key, the `span_*`
   namespace, and the `__synthesized_<pass>__` marker shape.
3. `crates/chelis-cli/tests/fixtures/octant/black_scholes/` carries a
   committed Octant-produced `call_price.dp` + `call_price.spans.json`
   fixture for the audit canary.
4. HIP toolchain reconciliation (workstation side-quest, parallel to S1–S3)
   produces a real fix — header alignment or wheel-only routing — not a
   `__AMDGCN_WAVEFRONT_SIZE` band-aid.

**Oracle:** workspace gate green except the pre-existing reef failure;
fixture parses cleanly via `chelis fmt <fixture>.dp`.

### S1 — Deep AST verification + helper

Add `Expr::span_id(&self) -> Option<&str>` to chelis-deep that pulls
`meta["span"]` if present. Add round-trip test using the Black-Scholes
fixture (parse → reprint → re-parse, every span ID present after both
reprints). Negative tests: empty span ID, Unicode chars, >1KB span ID,
1000-span synthetic input.

**Oracle:** new `roundtrip_span_metadata` test in
`chelis-deep/tests/roundtrip.rs` green; existing 21 tests still green.

### S2 — IR span schema + lowering

Add `span_id: Option<String>` and `merged_spans: Vec<String>` to `DagNode`.
Update every `add_node()` callsite in `crates/chelis-ir/src/lower.rs` to
thread `span_id` from the Deep `Expr` being lowered (using S1's accessor).
Update every non-lowering `add_node()` callsite to default `None` /
`vec![]`. Greppable invariant: zero `add_node(` calls without an explicit
`span_id` argument.

**Oracle:** new `span_threading_through_lowering` test in `chelis-ir/tests/`;
full workspace gate green.

🔴 **Red-team gate after S2.** Fresh local subagent via `redteam-exec` skill.
Adversarial probes: synthesized-node correctness, span identity through
lowering, edge cases (empty programs, deeply nested, no-span input,
Unicode), greppable invariant (no silent `None` span on a node lowered from
a span-bearing Deep expr), handcrafted Deep input exercising every IR node
variant.

### S3 — Span propagation through transformation passes

Per the table in §2.3. One commit per pass (or per logical group). Order:
low-risk forwarders first (vmap, DCE-remap), then high-risk (constant fold
→ CSE → Tier 2 → AD → fusion).

**Mandatory tests:**

- **AD three-hop test:** lower a span-attributed forward program, run AD,
  run codegen, assert the emitted C for the backward node contains BOTH
  `// span: __synthesized_grad__` AND `// span: <forward-node-span-id>`.
- **Constant-fold three-source test:** fold an expression where the
  operation and its operands carry distinct span IDs. Assert the folded
  result carries `span_id = "<op-span>"` and
  `merged_spans = [<operand-spans>...]` (lex-sorted).

**Oracle:** end-to-end test runs lowering → all enabled passes → final IR.
Asserts (a) every input span appears as `span_id` or `merged_spans` on at
least one final node, and (b) every `__synthesized_*__` marker has a
non-empty `merged_spans` (synthesized markers are never the *only*
provenance).

S2 → S3 checkpoint: re-confirm the per-pass rule table after the S2 red
team's findings before S3 begins.

### S4 — Backend span emission

One commit per backend (C, HIP, Metal). Per-op emitters prepend
`// span: <id>` lines via `self.line()` for `span_id ∪ merged_spans`. HIP
and Metal embed the same comments inside device-kernel source strings.

**Oracle:** `chelis build --target c --deep <fixture>.dp -o /tmp/out.c`
produces a `.c` file where `grep -c '// span:' /tmp/out.c` ≥
`span_count(<fixture>.spans.json)`, and the output compiles via `gcc`.
Equivalent oracle for HIP (`hipcc`) and Metal (existing backend's
compile-test). HIP's compile half assumes the toolchain side-quest has
landed; if not, escalate before downgrading.

### S5 — `chelis build --deep` flag + auto-detect

Add `--deep` flag and `.dp` auto-detect to the Build variant in
`crates/chelis-cli/src/main.rs`. Wire a Deep ingestion path that runs
typecheck/effects/lower/optimize/codegen with metadata preserved. Flag /
extension conflict resolution per §2.5.

**Oracle:** span-free Deep produces output equivalent to the Surf path;
span-attributed Deep produces span-annotated C.

🔴 **Red-team gate after S5 (final, pre-release).** Fresh local subagent.
End-to-end Black-Scholes canary: verify `// span:` count in `.c` matches
`.spans.json` count, pick a span ID and traverse back to original LaTeX
substring via `jq`. Conversion of "audit story conditional" → "audit story
shipped."

### S6 — Host-side `HostExpr` span schema + emission

Per the host-side table in §2.3, the host AST mirrors the DAG schema:
`HostExpr` carries `span_id: Option<String>` + `merged_spans:
Vec<String>`, threaded through host-side lowering/rewriting passes
(steps 1–4, landed at `46ed812`/`e590fec`). Backend host-side emission
(step 5) prepends `// span:` lines per `span_id ∪ merged_spans` to the
implementing C statement(s), reusing the same shape and sanitizer as
the DAG-path helper.

**Oracle:** end-to-end Black-Scholes audit chain on the natural
scalar form of `call_price_wrapped.dp` (S6 step 6 rewrote that
fixture from rank-0 tensors to scalar `(t-prim {} f32)` so the
program routes through host_emit). Lock command:

```sh
cargo test -p chelis-cli --test build_deep_audit_chain_canary
```

The oracle test asserts (a) emitted `// span:` count >= sidecar
entries, (b) every sidecar `deep_node_id` appears as a `// span:`
line in the emitted C (the at-the-emission-level audit invariant),
(c) a representative span resolves to the canonical LaTeX byte
range, and (d) the emitted C compiles via gcc.

🔴 **Red-team gate after S6.** Fresh local subagent. Same shipping
bar as S5: span-survival is a customer-visible audit promise; any
gap between sidecar and emitted C breaks the trust stack.

### S7 — Source-qualified node identity (chelis#1172)

S7 is delivered test-first in four dependency-ordered landing slices. Slice 1
is one atomic change set; its parser and Surf-carrier halves may be developed
as commits but cannot merge independently.

0. Submit §2.6's invariants as a separately reviewed numbered-spec amendment.
   `spec/03-deep-syntax.md` owns identity lifetime, equality, derivation,
   external-ID separation, generated-source encoding, and versioning. The
   change adds an identity-keyed normative registry only for stable
   pass/builder tokens that #1172 serializes into generated source. Private
   ingress, pass, builder, and cache inventories remain implementation-owned
   sealed types, not timeless spec rows. #886 must separately amend
   `spec/04-type-system.md` §6.4 with the shared check-result/diagnostic wire
   and `spec/09-tide.md` with Tide-specific HTTP/MCP/batch placement before
   source-bearing public output lands; this design does not pre-allocate them.
   The normative change runs
   `.venv/bin/python scripts/generate_rejection_registries.py --write` and
   commits the generated registry. Because `spec/03-deep-syntax.md` is a
   frozen Phase 4B input, the same change amends
   `spec/design/dtype_semantics.md` and tracker #729, deliberately updates
   the full-file digest, and runs
   `.venv/bin/python scripts/dtype_phase4b_oracle.py` to the exact final line
   `DTYPE PHASE 4B ORACLE: PASS`. Until this slice lands, the remaining
   bullets are proposed implementation work and no source-identity wire or
   generated-comment promise is in force.
1. Add failing source-identity, construction-privacy, ledger-bijection, and
   raw-ingress compile-fail tests. Add the dependency-leaf `chelis-source`
   crate for opaque `SourceUnitId`, `LocalNodeId`, `NodeKey`, `SourceSite`,
   `NodeOrigin`, `NodeRef`, `DiagnosticSourceRef`, and sealed
   role/pass/builder/cache descriptors. Define `SourcedSurfProgram` in
   `chelis-surf` and `SourcedDeepProgram` in `chelis-deep`; neither AST crate
   depends on the other in the wrong direction. Define the generated
   `IdentityStructure` traversal/path grammar, independently reconstructible
   issuance recipes, and rustdoc-derived public API-surface guard before
   migration begins. In the same atomic landing change, make every Surf and
   Deep parser plus every public adapter require `SourceInput<R>`, return the
   appropriate sourced success or qualified parser error, and thread the
   issued sites through every Surf declaration, expression, pattern, type, and
   desugaring helper. Formatting and LSP indexing consume
   `SourcedSurfProgram`; `TransformContext<SurfDesugar>` is the sole edge to
   `SourcedDeepProgram`; checking and lowering accept only the Deep carrier.
   No intermediate public parser may accept raw text or return an
   identity-free parsed tree, and a private `ParserLease` prevents internal
   adapter bypass.
   Source-less construction requires `SourceLessContext<B>`; subtree
   clone/splice requires `TransformContext<P>` and remints keys. Lock exact
   nested ranges, the attribution table, source-evidence reconstruction,
   duplicate/recombined-key rejection, same-unit composition, and
   unchanged-rebuild determinism.
2. Replace diagnostic scalar offsets/IDs and `LinearityInfo`'s bare-offset
   key with typed `DiagnosticSourceRef`/qualified `NodeKey`. Serialize the
   exact source-evidence roots, issuance-recipe graph, canonical
   node-assignment ledger, and provenance-sensitive digest with checked and
   reusable contexts. Convert every capable cache to `SourceAwareCache<C>`,
   require its decoder's independent `ExpectedSourceGraph<C>`,
   bump its then-live format, and lock immediate-predecessor,
   duplicate/recombined-key, forged-origin-after-redigest, unresolved-source,
   source-table, semantic-digest, recipe-digest, Surf-site-assignment, and
   Deep-node-assignment rejection before facts are exposed. Delete every
   semantic call to `parse_span_offset` and every source-unit merge policy
   based on numeric offset precedence.
3. Thread the same carrier through lowering, IR, host IR, every synthesizing
   transformation, and generated-source emission. Every new artifact is
   returned only by `TransformContext<P>` or `SourceLessContext<B>` after the
   output ledger is reconstructed and sealed. Checked-program and dependency
   composition deduplicates identical same-unit facts, rejects disagreement,
   and uses a registered reminting transform whenever the same input is
   materialized twice. Keep opaque external audit IDs separate from textual
   source ranges and synthesized provenance. #1172 exposes typed identity
   values to #886; the exact compiler-api/CLI/Tide/MCP/batch envelope and
   byte-for-byte fixtures land only under #886's controlling §04/§09 wire
   changes.

Each slice includes positive/negative parity. Required counterexamples are:

- two nested expressions on one line retain their exact distinct ranges;
- two source units with the same local offsets retain distinct node keys,
  diagnostics, and reusable-input facts after composition;
- reparsing the same logical file after a one-byte edit mints a new unit and
  rejects a serialized reusable-input record from the prior snapshot, while
  an unchanged reparse keeps deterministic keys;
- rebuilding a synthesized or unavailable tree with the same pass/builder
  schema, complete invocation configuration, and semantic output reproduces
  its unit and keys, while changing an output-affecting option or semantic
  output produces a distinct unit and no reusable-fact alias;
- public code cannot construct a `NodeKey`, `NodeOrigin`, `NodeRef`, or sourced
  tree from fields, raw digest bytes, deserialization, or a cloned/spliced raw
  subtree; registered reminting is the only admitted new-occurrence path;
- promotion and composition reject a duplicate key, a legitimate key paired
  with another origin, a parsed node relabelled synthesized/unavailable,
  forged synthesized contributors, and a ledger row with no exact structural
  occurrence even when the mutation recomputes every serialized assignment,
  recipe, and source-identity digest;
- the canonical path traversal covers top-level reordering, nested sequence
  indices, every Surf/Deep variant, structural atoms/helpers, and metadata keys
  and values; adding an AST field without a generated traversal arm fails to
  compile, while an unknown or conflicting transitional Deep carrier fails
  promotion rather than normalizing silently;
- Surf parsing returns `SourcedSurfProgram`; formatting and LSP indexing retain
  that carrier; only `TransformContext<SurfDesugar>` produces
  `SourcedDeepProgram`; checking/lowering reject raw Surf/Deep AST values; the
  Surf ledger-bijection oracle proves every path has a site and no pre-desugar
  `NodeKey`, while the Deep oracle proves every identity-bearing path has
  exactly one key/origin row;
- a Surf or Deep parse error before AST construction carries a qualified
  source range and no node key;
- every sealed `SourceInput<R>` role has a successful parse and parse-before-
  node failure; multi-buffer and batch fields remain distinct; private-module
  tests cannot obtain `ParserLease`; and a rustdoc-JSON mutation that adds a
  raw-input public parser returning raw Surf/Deep, `SourcedSurfProgram`,
  `SourcedDeepProgram`, or a qualified parse failure is rejected for both text
  and byte carriers; mutations adding raw public `run_source`, `prepare_source`,
  prove/MCP edges, raw AST semantic consumers, or identity constructors are
  likewise discovered without editing a crate list or expected manifest;
- parser-only, parsed-node, synthesized-node, and unavailable-node diagnostics
  preserve the typed identity distinctions through the compiler API and
  resolve every referenced unit; #886 separately locks their exact JSON
  placement and literal encoding;
- an offset-zero parsed token remains a real `Parsed` site, while a helper
  node is explicitly `Synthesized` and missing provenance is
  `Unavailable`—the three states never compare or render alike;
- an opaque external span ID that looks numeric is never interpreted as a
  text range;
- compiled-context, library, stdlib, and prepared-Reef-graph caches reject
  their immediate predecessor version and any stale source table or provenance
  digest before exposing a declaration, diagnostic, or reusable-input fact;
- swapping two same-unit local IDs while preserving exact source bytes and
  semantic Deep changes the canonical `DeepNodeAssignmentDigest` and rejects
  the cache before a node-keyed fact is exposed;
- swapping two valid parsed origins or changing a synthesized contributor set
  while keeping keys and semantics fixed, then recomputing every stored digest,
  still disagrees with reconstruction from the source/transform evidence and
  rejects before facts are exposed;
- replacing the serialized source bytes, registered pass, or otherwise valid
  pass configuration and recomputing the tree, ledger, and every digest still
  disagrees with the `ExpectedSourceGraph<C>` supplied by the current build or
  package request; decoding without such an expectation cannot expose facts;
- loading the same unchanged artifact through two sides of a diamond
  dependency coalesces identical reconstructed units and node-keyed facts;
  disagreeing facts reject, while intentionally materializing the same input
  twice requires a composition transform that remints output keys;
- every sealed pass/builder changes identity when an output-affecting
  invocation option changes; adding a pass, builder, input role, or cache
  without its required trait members or exhaustive dispatch is a compile
  failure, and deleting a public serialized-token registry row makes the
  generated-source guard fail;
- deleting a source qualifier, restoring `HashMap<usize, usize>`, or replacing
  synthesized origin with `Span::new(0, 0)` makes the suite red.

**Authoritative oracle:**

```sh
cargo nextest run -p chelis-cli --test issue_1172_source_identity --no-fail-fast
```

The named suite exercises Surf parse → Deep desugar → check/linearity →
lowering → diagnostic and generated-source output for every counterexample
above and invokes the construction-privacy, ledger-bijection, sealed-edge, and
cache-freshness mutation oracles, including the generated path traversal and
rustdoc-derived public API-surface guards. #886's §04/§09 wire suite is an
explicit integration prerequisite for source-bearing machine output but does
not replace #1172's identity oracle. Supporting crate tests do not replace
this end-to-end oracle.

🔴 **Red-team gate after S7.** A fresh local subagent executes the oracle,
  plants the qualified-key, changed-snapshot, fabricated-parser-node,
  duplicate/recombined-key, swapped-assignment, redigested-origin,
  same-unit-diamond, raw-API-export, and Surf-carrier-bypass regressions named
  above, and tries to bypass every sealed input/pass/builder/cache edge with an
  unqualified source path or unbound output-affecting option.
S7 is not complete while any entry point can construct a parsed program
without a source unit, any public value can recombine identity components, any
cache can expose facts before independent evidence reconstruction, any Surf
consumer must discard its sourced carrier, or any semantic map remains
offset-keyed.

## 4. Canary verification

Post-S5, the audit chain is exercised end-to-end:

```
# Octant produces span-attributed Deep
octant translate <octant>/references/black_scholes_call.tex \
  --output /tmp/bs.dp --spans /tmp/bs.spans.json

# Chelis preserves spans through the compile pipeline
chelis build /tmp/bs.dp --deep --target c --output /tmp/bs.c

# Audit chain is recoverable from /tmp/bs.c alone
grep -c '// span:' /tmp/bs.c        # ≥ jq '.spans | length' /tmp/bs.spans.json
SPAN_ID=$(grep -o '// span: [a-zA-Z0-9._:]*' /tmp/bs.c | head -1 | cut -d' ' -f3)
jq ".spans[] | select(.deep_node_id == \"$SPAN_ID\") | .latex_text" \
   /tmp/bs.spans.json
# Outputs the original LaTeX substring.
```

This test is committed to chelis CI as the canary regression once S5 lands.

## 5. Out of scope

Surf user-authored metadata syntax (`chelis build --deep` remains the
metadata-preserving path for producer-supplied Deep metadata beyond parser
byte ranges); runtime trace tooling; Octant-side
changes; span survival through external compilation/linking (DWARF,
post-roadmap); span performance optimization until profiling shows cost.
S7 does not define an IDE synchronization protocol or path-canonicalization
policy outside the compiler ingestion boundary. It does require each adapter
to supply the stable logical identity and exact bytes from which the private
`SourceUnitId` constructor derives an immutable snapshot identity.

## 6. Backward compatibility

Existing source programs compile and run unchanged. S7 intentionally changes
compiler-internal and compiler-api location carriers: a parsed program has a
source-qualified identity even when it has no external `span` metadata, while
a hand-constructed tree must state that its origin is unavailable or
synthesized. No compatibility adapter may recreate identity from a bare
offset. The workspace gate plus the S7 oracle are the regression backstop;
any unrelated test that was not red before this work and is red after is a
regression.
