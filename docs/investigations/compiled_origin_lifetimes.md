# Compiled-host provenance lifetimes

Generated C carries private producer origins beside values and projects pending
claim frames while constructing aggregates. Invocation-end cleanup alone retains
discarded construction metadata through every loop iteration, including programs
without extent claims. This investigation isolates those allocations from tensor
payloads, input ranges, output histories and the process allocator.

The owning oracle is `issue_3457_origin_lifetimes`. It instruments generated C's
`malloc`, `calloc`, `realloc` and `free`, recording actual allocation sizes,
current bytes before invocation teardown, peak live bytes and zero current bytes
after teardown. Its bookkeeping allocations and the separately linked payload
runtime are excluded. Sanitized controls execute generated binaries with Address
Sanitizer, Undefined Behavior Sanitizer and stack-use-after-return detection;
their success is a lifetime check, not a whole-program leak claim.

## Fixed live state and necessary growth

The baseline runtime is commit `5003150142d6a0c922438aef83254904f82b34c1`;
the owning oracle contains the source fixtures and allocator ledger.
For a fixed Box containing one tensor, origin peak and pre-teardown current bytes
are 6,688 / 25,888 / 102,688 at 100 / 400 / 1,600 iterations. Claimed Box
reconstruction additionally retains 8,000 / 32,000 / 128,000 frame bytes.
Fresh tensors and projections reproduce the same retention, and the origin
growth occurs with and without claims. A fixed three-element List retains
8,304 / 32,304 / 128,304 origin bytes and, when claimed,
24,000 / 96,000 / 384,000 projected-frame bytes.

With scoped ownership, the Box's peak is 416 origin bytes and 80 claimed-frame
bytes at each of those iteration counts; unclaimed frames remain zero. The
three-element List peaks at 552 origin bytes and 240 claimed-frame bytes.
Current frame bytes before teardown are zero. Initial List sizes 10 and 100
produce origin peaks 504 and 1,224 bytes respectively, each unchanged between
100 and 400 iterations. This separates seed storage from completed iterations.

The escaped growing-Chain control retains increasing necessary metadata:
10,008 / 38,808 peak origin bytes at 100 / 400 links. Reclamation must preserve
this growth. Tensor, tuple and nested nominal reconstruction, returned prefix
and suffix views, and distinct logical owners sharing an origin supply further
controls. These figures describe private metadata families, not whole RSS,
payload allocation, execution speed or all programs' memory use.

## Ownership and semantic controls

The representation is specified in `spec/design/runtime_extents.md`: immutable
origin edges and explicit roots own references, views retain their flattened
backing allocation, and iterative retirement processes each retired edge once.
Construction, branch, iteration and private-call scopes retire temporary roots
and projected frames. A private return retains its caller's origin slot before
callee cleanup. The bound includes reachable origins and backing storage, active
construction/call roots, and the finite invocation-owned producer-leaf cache.
It does not perform a reachability sweep over a growing accumulator.

Eval/native controls preserve independent claims and exact effect counts for
empty activation, initial failures, next-entry failures and invalid final steps
in fold and scan. Returned views retain their original producer diagnostics.
Existing nested-claim and accumulator suites cover caller/callee witnesses,
local and result claims, recursive and mutual calls, and bounded-stack deep
walks. No private origin enters the published value ABI.

Run the focused oracle serially:

```text
CARGO_BUILD_JOBS=1 NEXTEST_TEST_THREADS=1 cargo nextest run -p chelis-cli --test issue_3457_origin_lifetimes
```

Generated sources and family receipts are retained under `target/issue3457/`.
Timing and RSS observations on a shared machine do not establish this bound;
the allocation ledger and executable semantic/sanitizer controls do.
