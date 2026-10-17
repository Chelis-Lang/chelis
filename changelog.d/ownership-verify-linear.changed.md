`chelis build` spends less time in ownership verification. Each operation's
live-set bounds are kept as running totals instead of a rescan of the whole
live set, call-graph components are found in one linear pass instead of by
pairwise reachability, and the scheduled program is verified once instead of
twice. A nested 1280-operation program retires a third fewer instructions and
an 800-function recursive call chain about 2.8 times fewer. Programs,
diagnostics and emitted C are unchanged. See
[#2330](https://github.com/Chelis-Lang/chelis/issues/2330),
[#3723](https://github.com/Chelis-Lang/chelis/issues/3723) and
[#3730](https://github.com/Chelis-Lang/chelis/issues/3730).
