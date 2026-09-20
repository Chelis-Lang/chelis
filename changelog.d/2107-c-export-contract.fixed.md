The C ABI contract now consistently defines every authored non-`main`
function through an injective `chelis_fn_<utf8-hex>` symbol. Generated-code
tests and embedding probes consume that compiler-owned identity rather than
assuming a bare Surf spelling, while checks for C keywords, platform symbols,
prefix-shaped names, and ownership accounting remain active. See
[#2107](https://github.com/Chelis-Lang/chelis/issues/2107).
