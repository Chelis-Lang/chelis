The C ABI contract now consistently defines every authored non-`main`
function through an injective `chelis_fn_<utf8-hex>` symbol. Generated-code
tests and embedding probes consume generated declaration metadata rather than
reconstructing that identity. Source-level `main` retains its
module-qualified `<module>__main` ABI through Reef/package qualification,
while checks for C keywords, platform symbols, prefix-shaped names, stale
headers, exact exported definition-set agreement, and ownership accounting
remain active. Partial or metadata-swapped generated headers now fail before
native execution. A versioned artifact envelope now binds the exact program
identity, source digest, source identities, canonical symbols, exact declaration
bytes, external linkage, and one AST-identified C definition per export. Each
definition commitment covers its exact bytes and source-local preprocessing
context, so whole-source resealing cannot authorize exchanged bodies, indirect
macro retargeting, declaration/symbol or linkage spoofing, or unregistered
external definitions. Multiline declarations and comment-separated `static
inline` helpers remain valid, and translation-unit-private helpers stay outside
the published set. See
[#2107](https://github.com/Chelis-Lang/chelis/issues/2107).
