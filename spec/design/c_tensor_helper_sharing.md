# Private C tensor implementation sharing

`spec/08-backends.md` §2 owns C execution; `spec/04-type-system.md` §4.7
owns extent checks and result provenance. The span-survival design §2.4 owns
the source annotations on generated statements. Sharing implementation text
does not change those contracts.

The C host emitter retains an adapter for every tensor extraction site and
entry-coverage variant. An adapter passes its original diagnostic name and
the invocation's result-claim frames to a translation-unit-private kernel.
The kernel takes the diagnostic name as data, never as a format string.
Host-side result provenance remains attached to the original verified helper.

The implementation pool compares complete emitted code, with two explicit
holes: the kernel declaration's private symbol and the source-span comment
blocks. All other bytes participate, including input labels, constants,
dtypes, extent checks, entry-coverage decisions, and result-claim checks.
Each site is independently verified and emitted before comparison. A digest
indexes candidates; exact code equality decides sharing. This is an equality
test on one emitter's output, not a C semantic-equivalence analysis.

Corresponding statement positions retain the union of contributing source
spans. The first site's canonical annotation stays first; additional spans
are sorted and deduplicated. Adapters retain their original site symbols.
Shared kernels use the private `noinline` definition permitted by spec/08,
so native optimization does not inline their bodies into every host site.
Kernel numbering follows first occurrence in deterministic emission order.

Identity helpers retain their direct transport adapter. Helpers containing
runtime recurrence callbacks retain separate definitions: their generated
child symbols are outside this pool's declaration hole. External helpers
retain their peer translation unit's ABI.

The bounded regression command is:

```sh
cargo nextest run -p chelis-backend-c --test shared_tensor_helpers
cargo nextest run -p chelis-cli --test shared_tensor_helpers
```

It checks increasing extraction-site counts, distinct implementations,
attribution, deterministic output, and compiled execution.
The CLI cases exercise real extraction, with different input labels as a
non-sharing control, and compare a built pipeline's values and effect order
with evaluation. Build-cost results
for retained large models belong to the investigation and issue #4103. Small
fixtures do not establish full-model native build or numerical parity.
