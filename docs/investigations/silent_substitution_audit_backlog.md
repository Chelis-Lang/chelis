# Silent-substitution audit: unverified backlog

Status as of 2026-07-15. Companion to the two tracking issues:

- **[chelis#695]** - integers have no representation in the numeric layer
  (integer arithmetic laundered through `f64`).
- **[chelis#703]** - unsupported cases silently substitute a value instead of
  failing, so wrong programs compile and run.
- **[chelis#709]** - the checker's analogue: a construct with no `infer.rs`
  case types as silent `Type::Error` and disables type checking for its body.

This file records what the audit **did not** finish, so it is not lost. Every
item below is either unverified or unswept. Verified findings live in the
issues, not here.

## Ground rules learned the hard way

Five claims in this audit came from reading code (and code comments) and were
**refuted by executing it**. Do not promote anything below to an issue without
running it.

1. "The C backend corrupts int64 literals via `RiscOp::Const{value: f64}`" -
   false; that is the DAG *tensor* lane, which rejects int64. Scalars use the
   host lane and are exact.
2. "int32 tensors lose precision above 2^24 because the C runtime stores them
   as f32" (per the verbatim comment at `crates/chelis-runtime/src/lib.rs:144-152`) -
   not reproducible; exact in both lanes.
3. "`optimize::constant_fold` bakes wrong integer constants" - true of the
   code, but the pass has zero non-test call sites. Dormant.
4. "A one-line neural-network layer returns zero" (an early draft of #704) -
   false. Tensor `relu` is correct; only the **scalar** overload stubs.
5. "The unknown-Deep-tag catch-all silently returns the last trailing child" -
   false; the parser enforces the 62-tag vocabulary first (item 1 below).

All four are locked as refutation tests in
`crates/chelis-cli/tests/precision_matrix.rs` and
`crates/chelis-cli/tests/issue_703_silent_placeholders.rs`.

## Unverified, ranked by likelihood of being real

### 1. The unknown-Deep-tag catch-all - **REFUTED** (chelis#710)

`crates/chelis-ir/src/lower.rs:4926-4938`. A sweep ranked this its single worst
finding: a misspelled Deep form would be silently accepted and its last trailing
child returned.

**Refuted by execution.** The Deep parser enforces the 62-tag closed vocabulary
before lowering ever runs:

```
$ chelis check unk.dp
"score": 0
"errors": [{"kind":"Other","message":"expected valid Deep tag, found unknown tag
  'totally-bogus-tag'. Not in the 62-tag vocabulary at byte 42","severity":0.5}]
```

The catch-all is dead code behind a real guard. This was the fifth claim in the
audit sourced from reading code and refuted by running it.

### 2. Malformed `.dp` type-checks clean - **CONFIRMED, far milder than claimed** (chelis#710)

The mechanism is real: `infer.rs`'s arity guards return `Type::Error` **without**
pushing a `CheckError`, so `chelis check` reports `"score": 1, "errors": []` on
malformed Deep.

But the sweep claimed fourteen guards were live. Executed, **4 of 6 tested are
correctly rejected**:

| form | `chelis check` | `chelis eval` |
|---|---|---|
| `(def {} orphan)` | **score 1, errors []** | `error: unknown runtime name 'orphan'` |
| `(cast {} (lit ...))` | **score 1, errors []** | `error: cast missing target type` |
| `(let {} (bind {} x))` | score 0 - rejected | - |
| `(fn {} (params {}))` | score 0 - rejected | - |
| `(app {})` | score 0 - rejected | - |
| `(if {})` | score 0 - rejected | - |

And critically: **the `Const 0.0` placeholders are never reached**. Both silent
forms fail loudly at eval. This is a false green from `chelis check`, not a
silent-wrong-answer generator. The predicted "#695 and #703 meeting" does not
happen.

Items 4, 5, 6 and 8 below are all `.dp`-reachability claims from the same sweep
that ranked items 1 and 2 highest. Given that item 1 was refuted and item 2
over-stated by 7x, **discount that sweep's `.dp` reachability estimates
accordingly** and execute before believing any of them.

### 3. HIP's other seven `elem_kind` call sites

Only `Neg` is proven (#689: `kernel_neg_f32` with `const float *a` / `float *out`
over a `CHELIS_I64` allocation). Unproven, one-line probes each:

- `emit_const` (`crates/chelis-backend-hip/src/emit.rs:1806-1853`)
- `MaxElem` (`:1199`), `Abs` (`:1214`)
- `MinReduce` (`:1281`), `ProdReduce` (`:1285`), `Argmax` (`:1289`), `Argmin` (`:1293`)

Verifiable **without a GPU**: `chelis build --target hip` emits the HIP source and
prints the compile command without invoking `hipcc`. Grep the emitted kernel name
for an `_f32` suffix on an int64 program.

### 4. `Atom::Keyword` -> `Const 0.0`

`crates/chelis-ir/src/lower.rs:4861-4866`. The `Atom::Str` sibling is protected
by `def_body_requires_host_runtime` (`lower.rs:2338`), which treats any string
literal in a def body as host-forcing. That filter does **not** special-case
`Atom::Keyword` (`:2339`). `.dp`-only: Surf never emits a bare keyword atom in
expression position.

### 5. `reduce_window` `unwrap_or_default()`

`crates/chelis-ir/src/lower.rs:7522-7537`. `window_shape` / `strides` come from
`collect_cons_chain(...).and_then(...).unwrap_or_default()`, so a malformed or
non-literal window shape silently becomes `vec![]`. The comment claims the type
checker has already validated the shape; unverified for every input shape.

### 6. `lower_cast`'s `Prim::parse_name(...).unwrap_or(Prim::F32)`

`crates/chelis-ir/src/lower.rs:10235`. A bare-symbol cast target that fails
`Prim::parse_name` silently becomes `f32`. `infer_cast` (`infer.rs:18720-18725`)
rejects the unsigned-int-family typo case explicitly, but it is unconfirmed that
every malformed-symbol case is caught.

### 7. "Should never happen" defaults

- `crates/chelis-compiler-api/src/runtime/named_axis.rs:430` -
  `dag.get(*root).map(...).unwrap_or(Prim::F32)` silently defaults precision on a
  `values`/`dag` desync.
- `crates/chelis-backend-c/src/host_emit.rs:4060-4070` - `assign_partition`'s
  `let HostType::Tuple(parts) = ty else { ... return }` leaves the C target
  variable **unassigned** (undefined-behavior read) if reached. Requires a
  type-inference hole.

Low confidence of reachability; same anti-pattern (silent default vs. loud
`expect`).

### 8. `fail(non_literal_string)` outside an `if`

`crates/chelis-ir/src/lower.rs:8130-8161`. The `fail` zero-placeholder is
designed to be reachable only as the pruned side of an `if`, where `lower_if`'s
mask arithmetic zeroes it. A *literal* string argument forces the whole def to
the host lane, which normally keeps a bare `fail(...)` out of DAG lowering. A
**non-literal** message (`fail(some_string_variable)`) could dodge that filter
and reach the arm unmasked, producing a silent zero instead of aborting. No
check specifically forbids `fail` outside `if` was found.

## Pathways not swept at all

- **HIP / Metal runtime behavior.** Everything GPU is emission-inspection only;
  no GPU and no `hipcc` on the audit machine (arm64 macOS). #689 is proven by
  emitted source, not by running it. `docs/local_hip_environment.md` describes
  the gfx1151 box that could.
- **`chelis prove` Tier C** (#688). Code-confirmed (`ExecutionValue::Int64 { value } => *value as f64`
  at `obligation_engine.rs:1857` and `opaque.rs:1536`), never executed.
- **The `.dp` ingestion path generally.** A whole front end that bypasses Surf's
  structural guarantees. #709 proves the checker behind it is weaker than
  assumed. Items 1, 2 and 4 all live here.
- **`grad` / `vmap` interaction** with any of the above. Untouched.
  `transforms.rs:619-630` marshals int64 scalars through `as f64` into the DAG.
- **Metal's `Const 0`.** #699 explains the C lane's `lower_transcendental` zero;
  Metal's identical symptom was never re-confirmed to share that root cause
  after the diagnosis landed.
- **`chelis-cli`'s parallel pipeline.** #697, #698 and #705 each found a gate in
  `chelis-compiler-api` whose CLI twin is more permissive and guards the path
  that actually runs. Nobody has enumerated how many such pairs exist.

## Recommended next step

Items 1 and 2 are **settled** (2026-07-15): item 1 refuted, item 2 confirmed but
low-severity and 7x narrower than claimed. Both are recorded in chelis#710.

The next-highest-value item is **3** (HIP's seven unproven `elem_kind` call
sites), because it needs no GPU - `chelis build --target hip` emits inspectable
source without invoking `hipcc`, and #689's `Neg` probe is a one-line template
for each.

After that, **the `chelis-cli` parallel-pipeline enumeration** under "Pathways
not swept": #697, #698 and #705 each found a `chelis-compiler-api` gate whose CLI
twin is more permissive and guards the path that actually runs. Three instances
found by accident, none by search. That is the highest-yield unswept area.

[chelis#695]: https://github.com/Chelis-Lang/chelis/issues/695
[chelis#703]: https://github.com/Chelis-Lang/chelis/issues/703
[chelis#709]: https://github.com/Chelis-Lang/chelis/issues/709
