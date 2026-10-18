A recursive macro set whose argument grows by a bounded amount at each
expansion is now rejected with the expansion-limit diagnostic instead of
exhausting memory. The `source` provenance recorded on every expanded node
copied the previous expansions' records, about four times over per
expansion, so `chelis check` passed 2 GB within ten expansions and never
reached the limit. When the program's macros might recur, expansion now
finds the limit without building provenance first. A program that finishes
expanding is expanded and recorded exactly as before. A macro set that
cannot recur expands exactly as before even when it exceeds the limit, and
a template that duplicates its argument, such as `rec(x) = rec(add(x, x))`,
still builds a program that doubles at each expansion; both belong to
[#3857](https://github.com/Chelis-Lang/chelis/issues/3857)
([#3853](https://github.com/Chelis-Lang/chelis/issues/3853)).
