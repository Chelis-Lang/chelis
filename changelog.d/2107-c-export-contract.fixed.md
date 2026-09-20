The C ABI contract now consistently defines every authored non-`main`
function through an injective `chelis_fn_<utf8-hex>` symbol. Generated-code
tests and embedding probes consume generated declaration metadata rather than
reconstructing that identity. Source-level `main` retains its
module-qualified `<module>__main` ABI through Reef/package qualification,
while checks for C keywords, platform symbols, prefix-shaped names, stale
headers, and ownership accounting remain active. See
[#2107](https://github.com/Chelis-Lang/chelis/issues/2107).
