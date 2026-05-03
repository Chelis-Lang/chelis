# producer-string-sanitization: extend defense-in-depth to all producer-supplied strings flowing into generated source

**Status:** open; partial fix in S4 close-out (constructor-side validation for `Load`/`Store` names); deferred to a future hardening workstream
**Filed:** 2026-05-01
**Owning phase:** chelis-core (chelis-ir + all backends)
**Discovered by:** S4 re-red-team gate (post span-injection fix), F.2 sibling-class probe

## Summary

The S4 work introduced defense-in-depth for the `span` metadata key:
parser-side rejection of forbidden characters (`spec/03-deep-syntax.md`
§1.1.1) plus emit-side escape via `chelis_ir::span_sanitize`. The
post-fix re-red-team's sibling-class probe found that `RiscOp::Load
{ name }` and `RiscOp::Store { name }` are interpolated into generated
code unsanitized across all three backends:

- **Comment context** (Metal `// node 0 = Load x`): newlines terminate
  the `//` comment and emit subsequent text as raw source — same
  injection class as the original span bug.
- **Identifier context** (C/HIP variable names, function names, struct
  fields): not a comment-injection but the same producer-supplied
  string flows into a context where naïve escape (`\n` → `\\n`) is not
  even valid C identifier syntax.
- **Format-string context** (C/HIP `printf("... %s ... \n");`): same
  producer-supplied string flows into a `printf` format with `%`
  specifiers and additional escape concerns.

The S4 close-out commit lands a **bounded, atomic fix**: a `LoadStoreName`
newtype whose constructor enforces the same identifier grammar the Deep
parser enforces. This kills the programmatic-IR-construction attack
vector at the construction boundary, in the same architectural pattern
as parser-side rejection for spans (validate at the trust boundary).

This file tracks the **deferred deeper work**: extend the defense-in-
depth model to every producer-supplied string flowing into generated
source, with per-emission-context sanitization rules.

## Mitigations in place after S4 close-out

- Surf parser restricts identifiers to `[A-Za-z0-9_]`.
- Deep parser restricts identifiers to `[A-Za-z0-9_-]`.
- `LoadStoreName` newtype constructor enforces the Deep grammar — programmatic IR construction with control characters rejects at construction time.
- Today's only producer (Octant) emits identifiers from validated LaTeX variable names; never carries control chars in practice.

## Why this isn't fully closed by the S4 work

The constructor-side validation kills the **programmatic-IR-construction
attack vector** for `Load`/`Store` names. It does NOT establish a
**uniform architectural pattern** that every producer-supplied string
reaching generated source is sanitized at its emission context. The
broader pattern requires:

1. **Comment-context escape for any IR-string emission into comments.**
   Today this is just spans (handled) and `Load`/`Store` names (now
   transitively safe via grammar restriction at construction). But the
   **architectural pattern** ("every IR string flowing into a comment
   context goes through a sanitizer") isn't established — a future
   contributor adding a new IR field that gets emitted into a comment
   has no shared helper to use.

2. **Format-string-context sanitization.** The C/HIP backends emit
   `printf` format strings interpolating `Load { name }`. Even with
   valid identifier-grammar names, format strings have additional risks
   (`%` specifiers reaching the format positionally; `\n` not relevant
   given identifier grammar but other future fields might not be
   identifier-grammar-restricted). This is its own scope with its own
   sanitization invariant.

3. **Comprehensive grep for sibling-class instances.** Are there other
   producer-supplied strings flowing into generated source today?
   Module names? Type names? Effect names? Custom error message
   strings? Each is a potential instance of the bug class and deserves
   the same constructor-side OR parser-side validation. The S4
   re-red-team found Load/Store names; a focused audit might surface
   more.

## Reproduction recipe

(Recorded as discovered by the S4 re-red-team gate, before the
constructor-side fix lands. After the fix, this recipe should fail at
construction time, not at codegen.)

```rust
// In a chelis-ir test harness in /tmp:
use chelis_ir::dag::{Dag, RiscOp, TensorType};
use chelis_ir::eval::Prim;

let mut dag = Dag::new();
let bad = dag.add_node(
    RiscOp::Load { name: "x\nINJECT_C_CODE".to_string() },
    vec![],
    TensorType::scalar(Prim::F32),
    None,  // span_id — irrelevant; this bypasses the span sanitizer entirely
);
// Codegen: chelis_backend_c::codegen(&dag, ...) etc.
// Result before fix: emitted source contains the injection.
// Result after fix: construction itself fails before reaching codegen.
```

The injection lands in:
- C: `printf` format string newline embedded
- HIP: same
- Metal: `// node 0 = Load x` followed by `INJECT` on next line

## Scope of the deferred work

Three discrete pieces, each implementable as its own atomic commit:

### 1. Comment-context shared sanitizer

Establish an architectural pattern: any IR string flowing into a
generated `//` comment goes through a shared sanitizer (likely the
existing `chelis_ir::span_sanitize::sanitize_for_comment`, generalized
or re-exported). Update every `format!` callsite that interpolates a
producer-supplied string into a comment to call the sanitizer first.

**Scope:** roughly per-backend grep + helper-call insertion. Bounded by
the count of comment-emission sites with string interpolation; estimate
under 100 LOC.

### 2. Format-string-context sanitizer

Define the sanitization rule for `printf`-style format strings. The
invariant is broader than `//` comments — control chars AND `%`
format specifiers AND backslashes need handling. Likely a new function
`sanitize_for_format_string` in chelis-ir.

**Scope:** new sanitizer + per-backend update. Architectural decision
needed: should producer strings reach `printf` formats at all, or
should they always go through `%s` substitution (where the sanitizer
applies to runtime data, not format-time)? This deserves a small
design discussion before implementing.

### 3. Comprehensive audit

Grep across `crates/chelis-backend-{c,hip,metal}/src/` for every
`format!` callsite that interpolates a `String` from `RiscOp` or any
other producer-supplied source. For each, classify the emission
context (comment / identifier / format string / raw source line) and
confirm appropriate sanitization is in place. Likely surfaces 0-N new
findings.

**Scope:** read-only audit + per-finding fix. Could be a single
combined PR or split per finding, depending on what surfaces.

## Acceptance for closing this entry

- All three discrete pieces above land.
- A red-team gate (fresh-context subagent) probes producer-supplied
  string injection across every emission context and finds zero
  bypasses.
- Workspace gate green at every commit boundary.
- Spec records the architectural pattern: "every producer-supplied
  string flowing into generated source is sanitized per its emission
  context" — with the sanitizers and contexts enumerated.

## Out of scope (already closed)

- The `span` metadata key. Closed by S4 (spec §1.1.1 + parser rejection
  + emit-side `span_sanitize`).
- `RiscOp::Load { name }` and `RiscOp::Store { name }` programmatic-IR-
  construction injection. Closed by S4 close-out (`LoadStoreName`
  newtype with validating constructor).

## Why deferred

The S4 named scope was span emission. The sibling class is
architecturally adjacent but has different fix shapes per emission
context (comment escape, identifier validation, format-string
double-escape). These deserve focused architectural thinking, not
last-minute scoping inside an S4 close-out. The atomic constructor-side
fix establishes the trust-boundary pattern (validate at construction or
parse boundary, no exceptions); the deeper emission-context work is
the second layer of defense in depth and lands as its own focused
workstream.

The S5 work (`chelis build --deep` flag + final canary) is on the
critical path for the trust-stack audit-story claim and shouldn't be
gated on this hardening work.
