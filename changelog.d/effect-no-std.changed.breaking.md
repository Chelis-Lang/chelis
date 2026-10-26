Move pure effect analysis and shared effect data into allocation-based `chelis-effects-core` with unconditional `no_std`.

Consumers import `EffectError`, `EffectErrorKind`, and `DefEffectRows` directly from the portable core.
The old host definitions and compatibility re-exports do not remain.
The core borrows Deep expressions and an optional library input for inference, annotation, handler checks, and declared-row checks.
Deep owns the shared lexical and string readers, and the effect core owns the single builtin effect table.

Host adapters retain checked occurrence identity, checked reconstruction, type-totality rejection, diagnostics, target admission, realizability, root policy, and cache trust.
The complete acceptance command preserves immutable-source parity, full-host cost bounds, exact execution identities, timing values, and independent target graphs.
The vocabulary control checks numeric admission before dependent consumers and restores both source owners after each probe.
Policy commands retain repository-root imports and raw compiler rejections.
Remove obsolete setup-copy assertions between numeric and support jobs.
Retain numeric budgets, canonical command ownership, and the shared cache-writer authority guard.
The host compiler pipeline still needs `std`.
See [#4042](https://github.com/Chelis-Lang/chelis/issues/4042).
