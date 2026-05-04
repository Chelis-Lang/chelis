# producer-string-sanitization: extend defense-in-depth to all producer-supplied strings flowing into generated source

**Status:** RESOLVED 2026-05-01. Three deferred pieces landed in a single
atomic commit; see "Resolution summary" below for sanitizer locations,
callsite-update count, and tests added. The original "deferred deeper
work" entry follows for historical reference.

**Original status:** open; partial fix in S4 close-out (constructor-side validation for `Load`/`Store` names); deferred to a future hardening workstream
**Filed:** 2026-05-01
**Owning phase:** chelis-core (chelis-ir + all backends)
**Discovered by:** S4 re-red-team gate (post span-injection fix), F.2 sibling-class probe

## Resolution summary

### Sanitizer surface (`crates/chelis-ir/src/span_sanitize.rs`)

- `sanitize_for_comment(&str) -> Cow<'_, str>` — generalised from the
  S4 span-only helper. Module docs broadened; parameter renamed `span`
  → `s` to reflect the broadened scope. The function body is unchanged
  (the comment-context invariant — escape control bytes, preserve
  everything else verbatim — is identical for any producer-supplied
  string).
- `sanitize_for_format_string(&str) -> Cow<'_, str>` — new sanitizer
  for the C/HIP `printf`/`fprintf` compile-time format-string emission
  context. Escapes control bytes plus `%` (positional specifier doubling),
  `\\` (string-literal escape leader doubling), and `"` (string-literal
  terminator backslash-escape). Returns `Cow::Borrowed` for clean inputs
  (verbatim-preservation contract). 17 unit tests in the same module.

### Callsites updated

- **Comment-context (`sanitize_for_comment`)** — 4 net additions
  beyond the existing `// span:` callsites:
  - `crates/chelis-backend-metal/src/emit.rs::emit_load` (Load name in
    `// node N = Load <name>`).
  - `crates/chelis-backend-metal/src/emit.rs::emit_store` (Store name in
    `` // node N = Store `<name>` ``).
  - `crates/chelis-backend-metal/src/emit.rs::emit_root_writeback`
    (root label in `` // root output N = `<label>` ``).
  - `crates/chelis-backend-hip/src/emit.rs::emit_store` (Store name in
    `/* store: <name> */` block comment, with a documented residual
    note that block-comment `*/` injection is a separate concern not
    addressed by the comment-control-byte sanitizer).

- **Format-string-context (`sanitize_for_format_string`)** — 22 net
  additions across both backends:
  - `crates/chelis-backend-c/src/emit.rs` host entrypoint preamble:
    `func_name` and Load `label` in 7 `fprintf` format-string sites
    plus `binding.name` and `occurrence.input_label` in the symbolic-
    dim mismatch site.
  - `crates/chelis-backend-hip/src/emit.rs` host entrypoint preamble:
    same 7 sites + symbolic-dim mismatch.
  - `crates/chelis-backend-hip/src/emit.rs` device entrypoint preamble
    (`emit_input_shape_preamble_device`): mirror set of 7 sites +
    symbolic-dim mismatch.
  - `crates/chelis-backend-c/src/host_emit.rs::emit_labeled_root`:
    producer-supplied `name` lands in a `printf("%s = ", "<name>")` C
    string literal as a runtime arg via `%s`. Rust's `{:?}` debug
    format previously emitted `\u{XX}` for control bytes, which is
    NOT valid C. Replaced with `sanitize_for_format_string` + manual
    `"…"` wrapping so the C string-literal lexer sees only valid
    escapes.

### Tests added

- `crates/chelis-ir/src/span_sanitize.rs` — 17 unit tests for
  `sanitize_for_format_string` covering: clean Borrowed fast path,
  Unicode preservation, `%` doubling, `\\` doubling, `"` escape, every
  comment-context control byte still escapes (newline, CR, NUL, tab,
  other C0, DEL, space passthrough, empty-string Borrowed), combined
  attack vectors, and the single-pass invariant.
- `crates/chelis-backend-metal/tests/producer_string_sanitization.rs`
  — 4 tests covering Load name / Store name / root label comment-
  context sanitization (via the `serde_json` deserialize bypass to
  smuggle a forbidden byte past `LoadStoreName::new`) plus a clean-
  input-emitted-verbatim audit lock.
- `crates/chelis-backend-c/tests/producer_string_sanitization.rs`
  — 8 tests covering `func_name` `%`/newline/`"`-escape, Load name
  `%`-escape, symbolic-dim binding.name verbatim verification, the
  `codegen_with_options` API parity, the audit-invariant clean
  passthrough, and a `gcc -fsyntax-only` compile-success harness on
  the dirty-Load-name + clean-func-name input.
- `crates/chelis-backend-hip/tests/producer_string_sanitization.rs`
  — 4 tests covering the host AND device entrypoints' format-string
  sanitization for `func_name` and Load `label`.

### Audit findings (Piece 3)

A grep across `crates/chelis-backend-{c,hip,metal}/src/` for every
`format!` callsite that interpolates a producer-supplied string into
generated source was performed. Each finding is classified below by
emission context with disposition:

| Site (file:line) | Producer-supplied string | Context | Disposition |
|---|---|---|---|
| `metal/emit.rs:494` | `Load { name }` (LoadStoreName) | `// node N = Load <name>` line comment | sanitize_for_comment (new) |
| `metal/emit.rs:897` | `Store { name }` (LoadStoreName) | `` // node N = Store `<name>` `` line comment | sanitize_for_comment (new) |
| `metal/emit.rs:316` | OutputSpec.label (LoadStoreName or `rootN` synth) | `` // root output N = `<label>` `` line comment | sanitize_for_comment (new) |
| `metal/emit.rs:198,210` | DagNode.span_id / merged_spans (parser-validated) | `// span: <id>` line comment | sanitize_for_comment (existing, S4) |
| `c/emit.rs:370,382` | DagNode.span_id / merged_spans | `// span: <id>` line comment | sanitize_for_comment (existing, S4) |
| `hip/emit.rs:1735,1747` | DagNode.span_id / merged_spans | `// span: <id>` line comment | sanitize_for_comment (existing, S4) |
| `hip/emit.rs:1551` | `Store { name }` (LoadStoreName) | `/* store: <name> */` block comment | sanitize_for_comment (new) — control-byte coverage; `*/` injection documented as residual concern (LoadStoreName grammar excludes `*` and `/`, deserialize-bypass is hypothetical) |
| `c/emit.rs:99,108,118,127` | CLI-derived `func_name` | `fprintf` format-string | sanitize_for_format_string (new) |
| `c/emit.rs:526,537,550` | CLI-derived `func_name` + Load `label` (LoadStoreName) | `fprintf` format-string | sanitize_for_format_string (new) |
| `c/emit.rs:573` | CLI-derived `func_name` + binding.name (SymbolicDimBinding, plain String) + occurrence.input_label (Load, LoadStoreName) | `fprintf` format-string | sanitize_for_format_string (new) |
| `hip/emit.rs:96,105` | CLI-derived `func_name` | host-entrypoint `fprintf` format-string | sanitize_for_format_string (new) |
| `hip/emit.rs:231,240` | CLI-derived `func_name` | device-entrypoint `fprintf` format-string | sanitize_for_format_string (new) |
| `hip/emit.rs:414,425,438,461` | CLI-derived `func_name` + Load `label` + binding.name + occurrence.input_label | host-preamble `fprintf` format-string | sanitize_for_format_string (new) |
| `hip/emit.rs:496,507,520,543` | (same set) | device-preamble `fprintf` format-string | sanitize_for_format_string (new) |
| `c/host_emit.rs:2276` | HostBinding.display_name (plain String, producer-supplied) | C-string-literal-as-printf-runtime-arg | sanitize_for_format_string (new) — replaces `{:?}` Rust-debug |
| `metal/emit.rs:585,656,751` | RiscOp variant Debug (`unary {:?}` etc) | `// node N = unary <op>` line comment | NOT producer-supplied — RiscOp Debug only emits enum variant identifiers and validated LoadStoreName fields. Out of scope. |
| `metal/emit.rs:494,889,892,897` | `Err(format!("Load \`{name}\` not registered..."))` | `Result::Err` propagation, NEVER reaches generated source | Out of scope — the bug class targets generated-source emission, not in-process diagnostic strings. |
| `hip/emit.rs:126,262,760` | kernel name (`k_unary_<id>` synthesized by emitter) | identifier in declarator | NOT producer-supplied — compiler-internal. |
| `metal/kernels.rs:64,70` | tensor parameter name (`t<id>` synthesized) | identifier in MSL declarator | NOT producer-supplied — compiler-internal. |

**Surface-area note on `SymbolicDimBinding.name`.** This field is plain
`String` with no constructor-side validation today (>100 callsites
across `chelis-ir`, backends, and tests construct `DimInfo::Named(name,
size)` directly). Per the orchestrator's "escalate structural blockers"
constraint and the audit's >100-site cascade rule, adding a
`SymbolicDimName` newtype is intentionally out of scope for this
hardening workstream — the emit-side format-string sanitizer is the
defense-in-depth seat belt today, and a future workstream tracking
`SymbolicDimName` newtype + parser-side validation can layer on top
without invalidating the architectural pattern this work establishes.

### Architectural pattern locked

After this work, the rule for any future producer-supplied-string
emission site is uniform:

1. Identify the emission context (line comment / block comment /
   compile-time format string / runtime printf arg via `%s` / C
   identifier / etc.).
2. Route through the matching sanitizer once at the emission boundary
   (`sanitize_for_comment` for `// ...`, `sanitize_for_format_string`
   for any string flowing into a `"..."` C string literal — including
   compile-time format-string interpolation AND `printf %s` runtime
   arguments).
3. Identifier contexts continue to rely on parser-side or
   construction-side validation (`LoadStoreName`, future
   `SymbolicDimName`); the validation may legitimately reject and
   propagate as a real error, while the emit-side sanitizer is a
   silent safety net for the C string literal.

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
