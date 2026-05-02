# Chelis Span Survival: End-to-End Audit Chain

**Status:** active design — implementation phased S0–S5 (see "Phasing").
**Owners:** chelis-core (this repo). Octant ships span-attributed Deep upstream.
**Companion specs:** `spec/03-deep-syntax.md` §1.1.1 (the `span` key + `span_*` namespace + synthesized markers); `spec/design/chelis_trust_stack.md` (audit story).

---

## 1. Purpose

Octant emits span-attributed Deep with a sidecar `.spans.json` mapping each
span ID to a LaTeX byte range. Octant's tests confirm spans are correct in
the *emitted* Deep. The audit chain (LaTeX byte range → Deep node → IR node
→ backend output line) currently breaks somewhere between Deep ingestion and
backend emission inside chelis. This document specifies the contract that
closes that chain so a runtime trace or generated C source line can be
traced back to the original external-source byte range.

External-source agnosticism: the span ID is a string. Octant happens to use
LaTeX-derived dot-paths (`n_001`, `expr.5.lhs`); other producers may use
other conventions. Chelis preserves the strings and stays
producer-interpretation-agnostic.

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

### 2.4 Backend emission (S4)

Each backend (C / HIP / Metal) emits one `// span: <id>` line per
`span_id ∪ merged_spans` immediately preceding the line(s) that implement
the operation. Order: canonical first, then `merged_spans` lex-sorted
(deterministic, reproducible across runs). HIP and Metal emit the same
comment shape inside the embedded device-kernel source strings.

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
SPAN_ID=$(grep -o '// span: [a-z0-9._]*' /tmp/bs.c | head -1 | cut -d' ' -f3)
jq ".spans[] | select(.deep_node_id == \"$SPAN_ID\") | .latex_text" \
   /tmp/bs.spans.json
# Outputs the original LaTeX substring.
```

This test is committed to chelis CI as the canary regression once S5 lands.

## 5. Out of scope

Surf metadata syntax (Surf is metadata-lossy by design; `chelis build --deep`
is the metadata-preserving path); runtime trace tooling; Octant-side
changes; span survival through external compilation/linking (DWARF,
post-roadmap); span performance optimization until profiling shows cost.

## 6. Backward compatibility

Existing programs without span metadata compile and run unchanged. All span
fields are optional; missing metadata is the normal case for hand-written
Chelis. Workspace test suite is the regression backstop; any test that
wasn't red before this work and is red after is a regression.
