# host-emit-hashmap-iteration-nondeterminism: input-validation block ordering varies across builds

**Status:** open; not blocking the audit chain semantically
**Filed:** 2026-05-01
**Owning phase:** chelis-core (`chelis-backend-c::host_emit`)
**Discovered by:** S5.1+S5.2 implementation (build_deep_ingestion test wiring)

## Summary

`chelis_backend_c::host_emit` produces non-byte-identical C output across
runs from the same `.dp` input. Specifically the input-validation block
emission iterates a HashMap, which has non-deterministic iteration
order; the emitted validation lines for input parameters appear in
different orders across runs.

The S5.1+S5.2 flag-noop test (`chelis build foo.dp` vs.
`chelis build foo.dp --deep`) had to assert **set equality of `// span:`
comments** rather than **byte equality of the full file** as a
workaround. Set equality is correct for span semantics — the spans ARE
preserved — but the broader byte-equal-builds-from-equal-inputs
property is broken on the host_emit path.

## Why this is filed but not blocking

- The audit chain semantically holds: every span ID present in the
  input Deep appears as a `// span:` comment in the generated C
  (post-S5.4 host_emit span emission). Bit-reproducibility of the
  surrounding non-comment code is independent of the audit invariant.
- Bit-reproducibility is a real property worth having (audit-friendly
  diffs across builds, deterministic-build verification, content-
  addressed artifact correctness) but doesn't gate the trust-stack
  audit-chain claim.
- The fix is mechanical (replace HashMap iteration with a sorted
  iteration over collected keys) but touches host_emit's existing
  code structure; not in scope for S5.

## Reproduction recipe

```
# Build the wrapped Black-Scholes fixture twice; diff the outputs.
chelis build /path/to/call_price_wrapped.dp --deep --target c \
  --output /tmp/run1.c
chelis build /path/to/call_price_wrapped.dp --deep --target c \
  --output /tmp/run2.c
diff /tmp/run1.c /tmp/run2.c
# Expect: same set of lines, different order in the input-validation block.
```

The `// span:` comments themselves are deterministic (they go through
`sanitize_for_comment` and emit per-IR-node in DAG order). The
non-determinism is in the *non-span-emission* host_emit code that
emits validation `if (...) { abort(); }` blocks per input parameter.

## Sketch of fix

In `chelis-backend-c/src/host_emit.rs` (or wherever the input-
validation block is constructed): replace any `for (k, v) in
hash_map.iter()` over input-parameter HashMaps with `for k in {
  let mut keys: Vec<_> = hash_map.keys().collect();
  keys.sort();
  keys
} { let v = &hash_map[k]; … }`. Or use `BTreeMap` from the start if the
ordering matters semantically. Audit other host_emit HashMap iterations
for the same pattern — likely a small bug class.

## Impact on tests

- S5.1+S5.2 flag-noop test asserts `// span:` set equality. Documented
  inline. Once this bug is fixed, the test can be tightened to byte
  equality.
- Future bit-reproducible-build-verification work will need this fix
  before claims can be made.

## Out of scope

The trust-stack pitch's audit-chain claim is unaffected; this is a
quality issue on a different invariant (build determinism) that
deserves its own focused workstream.
