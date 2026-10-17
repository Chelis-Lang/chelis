Compiled dictionaries find keys through an index instead of scanning every
entry, and compare stored keys without revalidating them, so `dict_of`,
`dict_get`, `dict_contains`, and `dict_insert` or `dict_remove` accumulators
cost linear rather than quadratic time: `dict_of` over 16,000 string keys
drops from 20.6 to 0.1 billion instructions. Compiled `string_slice` on
non-ASCII text starts from a nearby character checkpoint instead of the start
of the string, so a character-at-a-time scanner is linear, at about the ASCII
cost. Results, entry order and failure messages are unchanged. See
[#2301](https://github.com/Chelis-Lang/chelis/issues/2301),
[#3733](https://github.com/Chelis-Lang/chelis/issues/3733) and
[#3725](https://github.com/Chelis-Lang/chelis/issues/3725).
