# Chelis Span Survival: End-to-End Audit Chain

**Status:** current shipped audit contract. S0-S6 are preserved below as the historical
implementation plan and named-oracle record; the active contract is in §2.
**Owners:** chelis-core (this repo). Octant ships span-attributed Deep upstream.
**Companion specs:** `spec/03-deep-syntax.md` §1.1.1 (the `span` key + `span_*` namespace + synthesized markers); `spec/design/chelis_trust_stack.md` (audit story).
Unsupported-diagnostic provenance is specified jointly with
`spec/design/host_function_values.md` §6 and tracked by chelis#868.

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

### 2.6 Unsupported diagnostics (S7; ratified, implementation pending)

Backend emission is not the only terminal audit sink. If compilation stops
before producing source, the structured `Unsupported` diagnostic must retain
the location of the rejected node.

`Unsupported.span: Option<SpanRef>` may carry:

- a numeric source byte offset and length when the producer still owns the
  source `Expr`;
- the opaque `span_id` already carried by a `DagNode` or `HostExpr`; or
- both when both are available.

The compiler API exposes the numeric range through its existing optional
`Diagnostic.span` and the opaque identifier through a new optional
`Diagnostic.span_id`. CLI rendering uses the same diagnostic. A producer must
not parse an opaque `surf:<start>..<end>` ID to reconstruct a numeric range.

The per-path rules are:

| Rejection path | Provenance source |
|---|---|
| checker/lowering while a Deep `Expr` is in scope | byte offset + length from the expression, plus `Expr::span_id()` |
| DAG backend | current `DagNode.span_id` |
| host-expression backend | current `HostExpr.span_id` |
| C ABI selection for a binding or expression result | containing binding value or expression `span_id` |
| C ABI selection for a function result | function body `span_id` |
| C ABI selection for a parameter or contextual callback signature | optional `span_id` preserved on `HostParam` / `HostCallback` |
| runtime or infrastructure failure with no source node | explicit `None`; no fabricated location |

Logical type values remain span-free. Type-driven target decisions receive
the containing value/parameter provenance as a separate argument, so source
location does not become type identity.

The located rendering is the additive §C2 form in
`spec/design/loud_unsupported.md`; the old spanless bytes remain frozen.
Opaque IDs are escaped so one source ID cannot inject a second diagnostic
line.

**Current divergence:** `Unsupported::with_span` has no production callers,
constructors default to `None`, its `Display` ignores the field, and
`unsupported_stage_error` discards it. This section records the S7 target,
not shipped behavior, until the S7 oracle below passes.

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

### S7 — Unsupported diagnostic provenance (chelis#868)

This remediation is sequenced before the shared function-value boundary so
the later C-host rejection can use the live location channel.

1. Write the diagnostic/API/CLI and producer-census tests first.
2. Require every `Unsupported` constructor to take an explicit optional
   `SpanRef`; remove the unused post-construction builder.
3. Implement the located §C2 rendering while preserving spanless bytes.
4. Add optional `Diagnostic.span_id` and move both structured location forms
   through `unsupported_stage_error`.
5. Thread existing Deep, DAG, and host-expression provenance through every
   production producer; add parameter/callback provenance required by
   type-driven C projection.
6. Keep a reviewed negative allowlist for genuinely source-free producers.

**Oracle:**

```sh
.venv/bin/python scripts/unsupported_span_oracle.py
```

Success is exit 0 with final line `UNSUPPORTED SPAN ORACLE: PASS`. The runner
covers spanless byte compatibility, ID/range/both/empty-ID cases, compiler
API JSON, both public compile entry points, CLI localization, representative
C-host/C-DAG/HIP/Metal rejections, and the constructor census.

🔴 **Red-team gate after S7.** A fresh local subagent drives at least one
unsupported source case through each public compiler surface and verifies
that the diagnostic identifies the rejected node without inventing a span.

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

## 6. Backward compatibility

Existing programs without span metadata compile and run unchanged. All span
fields are optional; missing metadata is the normal case for hand-written
Chelis. Existing structured consumers see only the additive optional
`span_id` field, and spanless unsupported messages remain byte-identical.
Workspace test suite is the regression backstop; any test that wasn't red
before this work and is red after is a regression.
