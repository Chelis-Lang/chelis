Move graph models and graph-only passes to the allocation-backed `no_std` crate `chelis-dag-core`.
Consumers now import `dag`, checked names, structural verification, axis sources, Tier-2 decomposition, AD, fusion, pure optimization, spans, and copy analysis directly.
The old `chelis_ir` graph modules and root exports no longer exist.

Graph-only ownership normalization and drop removal also use the portable owner.
Declaration owners, shape dependencies, roots, remaps, reuse hints, spans, valid wire data, and numeric delegation retain their contracts.
`chelis-numeric-core::bitwise::BitwiseKind` and `chelis-numeric-core::key_admission::{KeyPrimitive, KeyAdmission}` provide the sole shared pure identities.
The existing eight f64-image folds now use closed numeric-core APIs with sealed inputs and an explicit destination dtype.
Their exact-integer guard, refusal behavior, and final stored bits remain unchanged.
Ordinary typed numeric kernels retain their canonical NaN behavior.

Source lowering, evaluation, source linearity conversion, host diagnostics, ABI layout admission, target fences, and production orchestration stay in host crates.
The embedding supplies an allocator and stack policy.
The DAG oracle covers isolated target features, fixed-base behavior, rejection controls, native output, generated sources, graph scale, and transform allocation costs.
AD retains sparse adjoint storage.
Fusion reserves lookup capacity before graph construction.
Host comparisons use each Cargo target's declared feature requirements.

Remove incidental source-layout tests from the former graph owner.
Retain the payload census and source-policy tripwire at the canonical graph owner.
Preserve the exact moved host test source and split frozen rows without an allowance increase.
The policy record maps each changed address to the accepted numeric base.
The builtin closure oracle selects the portable graph library.
This ownership change transfers no language, transform, span, serialization, or backend authority.
