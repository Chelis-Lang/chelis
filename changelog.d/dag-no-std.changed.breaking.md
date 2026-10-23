Move graph models and graph-only passes to the allocation-backed `no_std` crate `chelis-dag-core`.
Consumers now import `dag`, checked names, structural verification, axis sources, Tier-2 decomposition, AD, fusion, pure optimization, spans, and copy analysis directly.
The old `chelis_ir` graph modules and root exports no longer exist.

Graph-only ownership normalization and drop removal also use the portable owner.
Declaration owners, shape dependencies, roots, remaps, reuse hints, spans, valid wire data, and numeric delegation retain their contracts.
`chelis-numeric-core::bitwise::BitwiseKind` and `chelis-numeric-core::key_admission::{KeyPrimitive, KeyAdmission}` provide the sole shared pure identities.
The existing eight f64-image folds now use closed numeric-core APIs with sealed inputs and an explicit destination dtype.
Their exact-integer guard, refusal behavior, and specified final stored bits retain their contracts.
Ordinary typed numeric kernels retain their canonical NaN behavior.

The fold oracle checks signaling-NaN quiet bits from the numeric contract, independent of native rounding behavior.

Source lowering, evaluation, source linearity conversion, host diagnostics, ABI layout admission, target fences, and production orchestration stay in host crates.
The embedding supplies an allocator and stack policy.
The DAG oracle covers isolated target features, fixed-base behavior, rejection controls, native output, generated sources, graph scale, and transform allocation costs.
AD retains sparse adjoint storage.
Fusion reserves lookup capacity before graph construction.
Host comparisons use each Cargo target's declared feature requirements.

Move the closed-enum and exhaustive semantic-match guards to the canonical graph owner.
Remove incidental source-layout and class-dispatch assertions.
Register both standalone fixtures with exact source compilation ownership.
Retain the payload census and source-policy tripwire at the canonical graph owner.
Preserve the exact moved host test source and split frozen rows without an allowance increase.

The builtin registry uses compiled declarations to distinguish public callables from internal primitive identities.
The semantic obligations cover every compiled internal primitive, including casts and guarded failure.
The native execution oracle reads the runtime owner's staging receipt and verifies the archive and public header digests.
Compile-fail ownership examples use valid fully qualified graph types.
The portability guard keeps explicit sibling subjects.
It rejects generic false pipeline claims, also when a comparison names a sibling.

Known empty shapes remain valid after large prefix extents.
Known nonzero counts must fit the graph index domain.
Cardinality uses bounded u128 intermediates without allocation.
Runtime metadata retains extent and suffix-stride checks.
The C inventory associates typedef owners through compiler declaration identities.
Refresh the independent pipeline artifact fixture lock for the portable graph dependencies.

This ownership change transfers no language, transform, span, serialization, or backend authority.
