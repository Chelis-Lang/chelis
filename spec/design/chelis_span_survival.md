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
node identity, or the source-qualified diagnostic wire. This section records
the implementation candidate that S7 must submit to the normative tier; it is
not an in-force language, wire, or generated-artifact contract. S7.0 is a
mandatory prerequisite: a separate numbered-spec change must establish the
rules in the chapters that own Deep identity/generated-source attribution and
Tide's public wire before any implementation or public output change lands.
Because `spec/03-deep-syntax.md` is frozen by the numeric-remediation guard,
that prerequisite must also follow the full freeze protocol named in S7.0.

The source model has three independent concepts. They must not share one
string or integer field:

```rust
pub struct SourceUnitId([u8; 32]);
pub struct LocalNodeId(NonZeroU64);
pub struct NodeKey {
    pub unit: SourceUnitId,
    pub local: LocalNodeId,
}
pub struct SourceSite {
    pub unit: SourceUnitId,
    pub range: Span,
}

pub enum NodeOrigin {
    Parsed(SourceSite),
    Synthesized {
        pass: SynthesizingPass,
        contributors: Vec<SourceSite>,
    },
    Unavailable,
}

pub struct NodeRef {
    pub key: NodeKey,
    pub origin: NodeOrigin,
    pub external_span_id: Option<String>,
}

pub struct DiagnosticSourceRef {
    pub site: Option<SourceSite>,
    pub node: Option<NodeRef>,
}
```

- `SourceUnitId` identifies one immutable input snapshot, not a logical module
  across edits. Let `lp(x)` be the unsigned 64-bit little-endian byte length of
  `x` followed by the bytes of `x`. A parsed unit is exactly
  `SHA256(b"chelis-source-unit-v1\0" || lp(namespace_utf8) ||
  lp(logical_identity_utf8) || lp(exact_source_bytes))`. Re-parsing unchanged
  bytes at the same logical identity produces the same unit; changing even one
  byte produces a different unit; identical bytes in two paths, request fields,
  or package modules remain distinct. File-based CLI adapters supply the
  canonical module/input identity. Compiler-api, Tide, Reef, and cache adapters
  supply an explicit logical identity for every input buffer; callers never
  inject unchecked raw digest bytes. A byte offset or display filename is not
  a unit ID.
- `NodeKey` is the source-qualified semantic identity used by checker and
  lowering maps. Its representation is structurally
  `(SourceUnitId, NonZeroU64)`, so two arenas that both allocate local key `1`
  cannot collide. The parser/desugarer allocates every Deep node a nonzero
  local identity in canonical construction order. It is never derived from a
  byte offset, an external span string, or structural equality. Construction
  is private to `SourceArena`; a producer cannot restart a counter for an
  existing unit.
- A later pass creates one deterministic synthesized arena per output artifact
  and pass. Its unit is exactly
  `SHA256(b"chelis-synth-unit-v1\0" || lp(stable_closed_pass_id_utf8) ||
  lp(pass_schema_version_le_u64) || lp(input_semantic_artifact_digest) ||
  contributor_count_le_u64 || lp(contributor_1_raw_unit_bytes) || ... ||
  lp(contributor_n_raw_unit_bytes))`. The input semantic-artifact digest is
  computed by a private builder from the pass's immutable input artifact and
  excludes source identity; a caller cannot inject it. Contributor units are
  deduplicated and sorted lexicographically by their raw 32 bytes before the
  count and rows are encoded; call order is never an identity input. The arena
  allocates local IDs in canonical output traversal order. A helper with
  contributors from multiple units therefore belongs to a
  fresh synthesized unit rather than borrowing one contributor's unit or
  restarting any parsed-unit counter. `NodeOrigin::Synthesized` still records
  the actual contributor sites. An unchanged deterministic pass over unchanged
  semantic input reproduces its unit and node keys; a pass-schema, semantic-
  input, or contributor change cannot alias the prior arena.
- Programmatic source-less construction still uses a qualified key. Its unit is
  exactly `SHA256(b"chelis-unavailable-unit-v1\0" ||
  lp(closed_construction_namespace_utf8) || lp(builder_schema_version_le_u64)
  || lp(semantic_artifact_digest))`. The private builder first constructs the
  source-less semantic tree, computes its canonical digest excluding source
  identity, then attaches keys in canonical traversal order. Neither a raw
  digest nor a counter is accepted from public callers. Its nodes carry
  `NodeOrigin::Unavailable`; there is no process-global counter and no zero
  unit. An unchanged rebuild reproduces keys, while any semantic-tree or
  builder-schema change mints a different unit. The compiler API rejects a
  builder that supplies neither parsed, synthesized, nor unavailable arena
  context.
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
semantic `expanded_deep_digest` exclude them. Any artifact that stores node-
keyed facts instead uses a provenance-sensitive `SourceIdentityDigest` over
the cache-format version, the ordered source-unit table, and the semantic
digest. Decode validates that digest and the source table before exposing any
linearity or diagnostic facts. The S7 implementation bumps all four current
formats that can retain source declarations or node-keyed facts: compiled
context V13, library cache V6, stdlib cache V9, and
`PreparedReefGraph` V3. Each moves to its next live version and gains
preceding-version rejection tests. If one advances before implementation, S7
allocates the next then-live version rather than reusing it.

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
`NodeRef`/`NodeKey`. Machine output exposes source unit, blame range, node key,
origin kind, contributor sites, and external audit ID as distinct fields.
Human rendering may abbreviate them, but cannot invent a node for a parse
error or a range for `Unavailable`/`Synthesized`.

The candidate public JSON is exact rather than an abstract Rust-enum sketch.
Every source-bearing response has `"source_identity_version": 1` and a
`"source_units"` array sorted by `unit`, with duplicate units forbidden.
Every unit referenced by a node key, site, origin, or contributor must resolve
to exactly one row in that array. Rows use the required `kind` discriminator
and exactly one of these shapes:

```json
{"kind":"parsed","unit":"su1:<64-lowercase-hex>","role":"<closed-endpoint.field>","display_name":null,"sha256":"<64-lowercase-hex>"}
{"kind":"synthesized","unit":"su1:<64-lowercase-hex>","pass":"<closed-pass-id>","pass_schema_version":1,"input_artifact_digest":"sha256:<64-lowercase-hex>","contributors":["su1:<64-lowercase-hex>"]}
{"kind":"unavailable","unit":"su1:<64-lowercase-hex>","construction_namespace":"<closed-builder-id>","builder_schema_version":1,"semantic_artifact_digest":"sha256:<64-lowercase-hex>"}
```

`display_name` is required on a parsed row and is a string or explicit
`null`; it is presentation-only and is excluded from identity. Parsed
`role` is the manifest identity below, not free text. Synthesized and
unavailable rows have no request role or source-byte hash. Variant-inapplicable
fields are omitted, never emitted as `null`. Synthesized contributor units are
sorted and deduplicated by raw unit bytes, exactly as in the identity input.
Schema versions are unsigned JSON integers in the exact `u64` range.

A diagnostic's `source` value is either `null` or an object with both
`site` and `node` keys present. Each key is independently an object or
explicit `null`:

```json
{"source":{"site":{"unit":"su1:<64-lowercase-hex>","start":0,"end":1},"node":null}}
{"source":{"site":null,"node":{"key":"nk1:<64-lowercase-hex>:<nonzero-decimal>","origin":{"kind":"unavailable"},"external_span_id":null}}}
```

`external_span_id` is always present and is a string or explicit `null`.
`origin` is exactly one of:

```json
{"kind":"parsed","site":{"unit":"su1:<64-lowercase-hex>","start":0,"end":1}}
{"kind":"synthesized","pass":"<closed-pass-id>","contributors":[{"unit":"su1:<64-lowercase-hex>","start":0,"end":1}]}
{"kind":"unavailable"}
```

No extra variant fields are accepted. Ranges are half-open byte ranges with
`start <= end`. The 32-byte unit embedded in `key` must equal the unit of
the row describing that node arena. Parser-only, parsed-node, synthesized-node,
and unavailable-node fixtures lock the literal JSON and generated schema,
including null-versus-omission and referential-integrity failures.

`LinearityInfo` is keyed by `NodeKey`, replacing
`HashMap<usize, usize>`. `mark_reusable_input` records the checked app node's
qualified key and lowering queries that same qualified key. Merging separately
checked programs is then an ordinary disjoint-key union; an equal byte offset
or equal local node counter in two files is not a collision and no
library-half-wins rule is needed.

The candidate generated-source format keeps `// span: <external-id>` for the
external audit chain and emits source identity separately. Unit text is
`su1:<64 lowercase hexadecimal digits>`; node-key text is
`nk1:<the same 64 hexadecimal digits>:<nonzero decimal local id>`. The exact
comment rows are:

```text
// chelis-source: <node-key> parsed <source-unit> <start>..<end>
// chelis-source: <node-key> synthesized <closed-pass-id>
// chelis-source-contributor: <source-unit> <start>..<end>
// chelis-source: <node-key> unavailable
```

A synthesized row is followed by its contributor rows sorted and deduplicated
by `(raw unit bytes, start, end)`. Closed pass and builder IDs match
`[a-z0-9][a-z0-9._-]*`; no whitespace, newline, comment delimiter, display
path, or unvalidated caller string can enter a record. Consumers can therefore
distinguish a Deep parser range from an opaque external ID and trace two Surf
files that use the same local byte range without aliasing them.

#### Public-ingress inventory

S7 closes every row below. The manifest is generated from public request
schemas and adapter registrations; a hand-edited expected list cannot certify
itself. Every listed source field gets a distinct parsed unit and both a
successful-node fixture and a parse-before-node diagnostic fixture.

| compiler API request/function | source fields | HTTP route | MCP tool | batch role |
|---|---|---|---|---|
| `parse(ParseRequest)` | `source` | `/parse` | — | `requests[i].parse.source` |
| `desugar(DesugarRequest)` | `source` | `/desugar` | `chelis_desugar` | `requests[i].desugar.source` |
| `check(CheckRequest)` | `source` | `/check` | `chelis_check` | `requests[i].check.source` |
| `lower(LowerRequest)` | `source` | `/lower` | — | `requests[i].lower.source` |
| `compile(CompileRequest)` | `source` | `/compile` | `chelis_compile` | `requests[i].compile.source` |
| `eval(EvalRequest)` | `source` | `/eval` | `chelis_eval` | `requests[i].eval.source` |
| `grad(GradRequest)` | `source` | `/grad` | `chelis_grad` | `requests[i].grad.source` |
| `validate(ValidateRequest)` | `source` | `/validate` | `chelis_validate` | `requests[i].validate.source` |
| `decompile(DecompileRequest)` | `source` | `/decompile` | `chelis_decompile` | `requests[i].decompile.source` |
| `replace_function_body(ReplaceFunctionBodyRequest)` | `module`, `new_body` | `/replace_function_body` | `chelis_replace_function_body` | — |
| `add_function(AddFunctionRequest)` | `module`, `new_decls` | `/add_function` | `chelis_add_function` | — |
| `deep_outline(DeepOutlineRequest)` | `module` | `/deep_outline` | `chelis_deep_outline` | — |
| `deep_references(DeepReferencesRequest)` | `module` | `/deep_references` | `chelis_deep_references` | — |
| `deep_call_graph(DeepCallGraphRequest)` | `module` | `/deep_call_graph` | `chelis_deep_call_graph` | — |
| `replace_function(ReplaceFunctionRequest)` | `module`, `new_decls` | `/replace_function` | `chelis_replace_function` | — |
| `rename(RenameRequest)` | `module` | `/rename` | `chelis_rename` | — |
| `change_signature(ChangeSignatureRequest)` | `module`, `new_defsig`, `new_params` | `/change_signature` | `chelis_change_signature` | — |
| `add_property(AddPropertyRequest)` | `module`, `new_decls` | `/add_property` | `chelis_add_property` | — |

HTTP and MCP deserialize these shared request structs and therefore must consume
the same generated endpoint/field manifest rather than maintain independent
copies. Batch roles include the request index so two equal variants in one
envelope cannot alias. Compiler-API direct calls use
`compiler_api.<function>.<field>`; HTTP and MCP use
`http.<route>.<field>` and `mcp.<tool>.<field>`, respectively. Those closed
role strings, plus the caller-supplied logical request identity, are inputs to
the parsed-unit namespace; transport display labels are not.

The remaining adapters have these exact source boundaries:

| adapter | source fields | identity and freshness |
|---|---|---|
| Surf and Deep parser APIs | source byte slice | require a logical identity; parse failures carry `SourceSite` and no fabricated node |
| CLI file/stdin/import | each file's exact bytes; stdin bytes; each imported module | canonical package/module or input identity; stdin adds an invocation nonce; response keeps identity separate from display path |
| LSP `didOpen` | `textDocument.uri`, `textDocument.version`, `textDocument.text` | URI and version are carried into analysis; equal URI with changed version/bytes cannot reuse facts |
| LSP `didChange` (FULL sync) | `textDocument.uri`, `textDocument.version`, `contentChanges[0].text` | exactly one full-text change is admitted; stale/nonmonotonic versions fail before analysis |
| Reef package composition | every discovered module's package identity, canonical inventory path, and exact bytes | composition unions qualified keys and retains the package/module display table |
| cache decode | compiled context, library, stdlib, and `PreparedReefGraph` source tables and provenance digest | format, source table, semantic digest, or caller-source mismatch rejects before facts are exposed |

Generated tests enumerate HTTP router registrations, MCP tool registrations,
`BatchRequest` variants, public compiler functions/request schemas, LSP sync
handlers, Reef module ingestion, and all four cache decoders. The enumerated
set must be bijective with this typed manifest. Adding a source-bearing field,
endpoint, adapter, or cache without a role and both test polarities makes the
S7 oracle red.

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

S7 is delivered test-first in five dependency-ordered slices:

0. Submit the candidate contract in §2.6 as a separately reviewed numbered-
   spec amendment to `spec/03-deep-syntax.md` and `spec/09-tide.md`. That
   normative change fixes the identity byte encodings, lifetime/equality
   boundaries, exact public JSON, generated-comment encoding, external-ID
   separation, compatibility/versioning, and the closed ingress roles before
   implementation begins. It runs
   `.venv/bin/python scripts/generate_rejection_registries.py --write` and
   commits the generated registry. Because `spec/03-deep-syntax.md` is a
   frozen Phase 4B input, the same change amends
   `spec/design/dtype_semantics.md` and tracker #729, deliberately updates
   the full-file digest, and runs
   `.venv/bin/python scripts/dtype_phase4b_oracle.py` to the exact final line
   `DTYPE PHASE 4B ORACLE: PASS`. Until this slice lands, the remaining
   bullets are proposed implementation work and no source-identity wire or
   generated-comment promise is in force.
1. Add failing source-identity tests, then introduce `SourceUnitId`,
   `LocalNodeId`, qualified `NodeKey`, `SourceSite`, `NodeOrigin`, `NodeRef`,
   and `DiagnosticSourceRef` in `chelis-deep`. Private parsed,
   synthesized, and unavailable builders implement the exact encodings from
   S7.0 and prohibit caller-injected unit bytes, artifact digests, and local
   counters. Source-aware Surf and Deep parser entry points allocate nonzero
   node keys; source-less construction requires an explicit unavailable or
   synthesized builder rather than the zero-span sentinel.
2. Thread the source unit and origin through every Surf declaration,
   expression, pattern, type, and desugaring helper. Lock exact nested
   expression ranges and the attribution table in §2.6, including helpers
   created for declarations, destructuring, pipes, and annotations. Lock
   unchanged-rebuild determinism and changed semantic-artifact non-aliasing
   for both synthesized and unavailable arenas.
3. Replace diagnostic scalar offsets/IDs and `LinearityInfo`'s bare-offset
   key with typed `DiagnosticSourceRef`/qualified `NodeKey`. Serialize the
   source table and provenance-sensitive digest with checked and reusable
   contexts; bump compiled-context, library-cache, stdlib-cache, and
   `PreparedReefGraph` formats and lock rejection of every immediate
   predecessor plus source-table/digest mismatches. Delete every semantic call
   to `parse_span_offset` and every source-unit merge policy based on numeric
   offset precedence.
4. Thread the same carrier through lowering, IR, host IR, generated-source
   emission, and the exact compiler-api/Tide/MCP/batch/LSP/Reef ingress
   manifest and JSON wire in §2.6. Keep opaque external audit IDs separate
   from textual source ranges and synthesized provenance. Serialization and
   schema tests lock parser-only, parsed-node, synthesized-node, and
   unavailable-node responses byte-for-byte.

Each slice includes positive/negative parity. Required counterexamples are:

- two nested expressions on one line retain their exact distinct ranges;
- two source units with the same local offsets retain distinct node keys,
  diagnostics, and reusable-input facts after composition;
- reparsing the same logical file after a one-byte edit mints a new unit and
  rejects a serialized reusable-input record from the prior snapshot, while
  an unchanged reparse keeps deterministic keys;
- rebuilding a synthesized or unavailable tree with the same pass/builder
  schema and semantic input reproduces its unit and keys, while one semantic
  input change produces a distinct unit and no reusable-fact alias;
- a Surf or Deep parse error before AST construction carries a qualified
  source range and no node key;
- every field in the compiler-api/HTTP/MCP/batch manifest has a successful
  parse and parse-before-node failure; in particular,
  `replace_function_body` keeps `module` and `new_body` distinct,
  `replace_function`/`add_function`/`add_property` keep `module` and
  `new_decls` distinct, and `change_signature` keeps `module`,
  `new_defsig`, and `new_params` distinct;
- parser-only, parsed-node, synthesized-node, and unavailable-node diagnostics
  serialize to the exact JSON shapes in §2.6, reject wrong
  null-versus-omission/discriminator shapes, and resolve every referenced unit;
- an offset-zero parsed token remains a real `Parsed` site, while a helper
  node is explicitly `Synthesized` and missing provenance is
  `Unavailable`—the three states never compare or render alike;
- an opaque external span ID that looks numeric is never interpreted as a
  text range;
- compiled-context, library, stdlib, and prepared-Reef-graph caches reject
  their immediate predecessor version and any stale source table or provenance
  digest before exposing a declaration, diagnostic, or reusable-input fact;
- deleting a source qualifier, restoring `HashMap<usize, usize>`, or replacing
  synthesized origin with `Span::new(0, 0)` makes the suite red.

**Authoritative oracle:**

```sh
cargo nextest run -p chelis-cli --test issue_1172_source_identity --no-fail-fast
```

The named suite exercises Surf parse → Deep desugar → check/linearity →
lowering → diagnostic and generated-source output for every counterexample
above. Supporting crate tests do not replace this end-to-end oracle.

🔴 **Red-team gate after S7.** A fresh local subagent executes the oracle,
  plants the qualified-key, changed-snapshot, and fabricated-parser-node
  regressions named above, and checks the complete public-ingress inventory
  and cache entry points for an unqualified source path.
S7 is not complete while any entry point can construct a parsed program
without a source unit or while any semantic map remains offset-keyed.

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
