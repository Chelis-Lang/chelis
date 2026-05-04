# Black-Scholes Octant Fixture

Span-attributed Deep + spans manifest produced by Octant. Used by the
chelis span-survival pipeline tests as the load-bearing audit canary:
every span ID in the `.dp` must round-trip through the chelis compile
pipeline and surface in the generated C output.

The directory ships TWO fixtures with different roles.

## `call_price.dp` — equation-only audit canary (S0–S4)

Octant's verbatim translation of `references/black_scholes_call.tex`
(the canonical hero-example LaTeX in the Octant repo). The body is
just the Black-Scholes equation `c = s * Phi(d_1) - k * exp(-r*t) *
Phi(d_2)` with free variables `s, d_1, k, e, r, t, d_2` — it does NOT
typecheck on its own (the free vars are unbound) and is **not** routed
through `chelis check` / `chelis build`. It is used by:

- `chelis-deep/tests/span_metadata.rs` — round-trip every span through
  parse → reprint → re-parse, lock the 20 expected `n_001..n_020` IDs
  in place. This is the parse-and-print audit canary that proved the
  S0 oracle and continues to be the regression backstop for any
  metadata-handling change in `chelis-deep`.

Keep this fixture exactly as Octant emits it; do not normalize, fold,
or annotate. It is the unaltered upstream artifact that downstream
chelis tools must accept.

## `call_price_wrapped.dp` — typecheckable function-shape fixture (S5+, S6)

The wrapper around Octant's translation of
`references/black_scholes_call_function.tex` (the multi-equation
companion file in the Octant repo). As of Octant commit 468bdc6 the
function-shape wrapper is produced natively by
`octant translate ... --wrap-as-function call_price`; the fixture
tracks Octant's natural output for the wrapper-marker shape (canonical
`__synthesized_wrap__` per `spec/03-deep-syntax.md` §1.1.1) and inlines
Octant's emitted `d_1`, `d_2`, and `c` bodies as `let` bindings inside
a single function def:

```
(def {span: "__synthesized_wrap__"} call_price
  (fn {} (params {} (s {type: (t-prim {} f32)}) ...)
    (let {} (bind {} d_1 <octant-d_1-body> d_2 <octant-d_2-body>)
      <octant-c-body>)))
```

Every Octant-emitted span ID (`eq:d1_*`, `eq:d2_*`, `eq:c_*`) is
inlined verbatim. Both outer defs (`normal_cdf` stub and
`call_price`) carry the canonical `__synthesized_wrap__` marker, the
reserved synthesized-marker shape from the Deep syntax spec.

Two divergences from Octant's natural `--wrap-as-function` output are
maintained in this fixture:

1. **Local `normal_cdf` stub.** Octant's natural output references
   `Nautilus.Special.normal_cdf` via an `(access ... normal_cdf)`
   chain. Chelis test infrastructure does not currently expose a
   Nautilus reef package on the Deep ingestion path, so the fixture
   inlines a typecheck-only identity stub for `normal_cdf`. Full
   regeneration (replacing the stub with Octant's natural access
   chain) is gated on resolving how Nautilus is provisioned for these
   tests; track as a future follow-up.
2. **Explicit `f32` literals.** Octant emits int literals (`2`, `0`)
   for the `\sigma^2 / 2` and `-r` LaTeX subexpressions. Chelis Deep
   has no implicit precision promotion (`spec/04-type-system.md`),
   so the fixture annotates these literals with explicit
   `(t-prim {} f32)` types and `2.0` / `0.0` values. This is a
   precision-defaults question that would also benefit from a future
   Octant-side fix or a chelis-side coercion step.

Type shape (S6 step 6): natural scalar `(t-prim {} f32)` throughout —
parameters, locals, and literals. Black-Scholes is a scalar formula;
the canonical customer-shape program. With S6 step 5 landing host-path
span emission in `chelis-backend-c::host_emit`, the audit chain is
exercised end-to-end on `host_emit` for this fixture. Pre-S6 the
fixture used rank-0 tensors `(t-tensor {} (t-prim {} f32))` to force
routing through the DAG codegen path (which was the only path that
emitted span comments at the time); that workaround is obsolete and
the fixture was rewritten to its natural scalar shape.

Linearity wrinkle: scalars are non-linear (the consume-by-default
rules in `spec/04-type-system.md` §8.1 only apply to tensors), so the
post-S6 form drops the `(copy {} (var {} x))` helper bindings the
rank-0-tensor form needed for multi-use of `s`, `k`, `r`, `t`,
`sigma`, and `d_1`.

Used by:

- `chelis-cli/tests/wrapped_black_scholes_fixture.rs` — fmt round-trip,
  Phase 0e typecheck, span-vs-sidecar parity (S5.0 oracle).
- `chelis-cli/tests/build_deep_ingestion.rs` — `chelis build --deep`
  end-to-end tests including the S5 audit chain canary and gcc
  compile-success on the emitted host-side C.
- `chelis-cli/tests/build_deep_audit_chain_canary.rs` — S6 named
  oracle: §9 canary exercised through `host_emit` end-to-end.

## Files

- `call_price.dp` — Octant's raw 20-node equation-only Deep
  (IDs `n_001`..`n_020`).
- `call_price.spans.json` — Octant's sidecar mapping each
  `deep_node_id` to its LaTeX byte range.
- `call_price_wrapped.dp` — wrapper produced by Octant
  `--wrap-as-function`, with two local divergences (see above);
  natural scalar `(t-prim {} f32)` throughout. Body spans (44) come
  from Octant; both outer defs share the single canonical
  `__synthesized_wrap__` marker.
- `call_price_wrapped.spans.json` — wrapper sidecar; reuses Octant's
  byte-range entries for inlined span IDs and adds a single
  synthesized entry for `__synthesized_wrap__` (with `latex_text` set
  to `__synthesized_wrap__` and zero byte ranges). Per Octant's
  `SourceId(u32)`, the synthesized entry uses `source_id: 0`. Total
  45 entries (44 body + 1 wrapper marker).
- `README.md` — this file.

## Regeneration

`call_price.dp` and `call_price.spans.json` come straight from
Octant's `translate` subcommand:

```
octant translate <octant>/references/black_scholes_call.tex \
  --output call_price.dp \
  --spans call_price.spans.json
```

After running, sanitize `source` in `call_price.spans.json` from the
absolute path Octant emits to `references/black_scholes_call.tex`
(relative to the Octant repo). The `source_hash` field stays as
Octant emitted it.

`call_price_wrapped.spans.json` is generated by
`scripts/build_wrapped_spans_sidecar.py`, which:

1. Runs `octant translate` on
   `<octant>/references/black_scholes_call_function.tex` in a temp
   dir to capture the canonical sidecar.
2. Filters that sidecar to just the span IDs that appear in
   `call_price_wrapped.dp` (in appearance order, deduplicated).
3. Adds a single synthesized entry for `__synthesized_wrap__`.

`call_price_wrapped.dp` itself is produced by running
`octant translate ... --wrap-as-function call_price` against
`references/black_scholes_call_function.tex` and then applying the
two divergences described above (local `normal_cdf` stub instead of
the `Nautilus.Special.normal_cdf` access chain; explicit `f32`
typing on the int literals `2` and `0`). If the wrapper shape
changes upstream, regenerate from Octant and re-apply those local
edits, then re-run the spans-sidecar script.

## Spec

See `spec/design/chelis_span_survival.md` for the contract this
fixture exercises and `spec/03-deep-syntax.md` §1.1.1 for the
documented `span` metadata key and the reserved
`__synthesized_<pass>__` marker form.
