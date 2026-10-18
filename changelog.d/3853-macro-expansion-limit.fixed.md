A macro that never finishes expanding is now rejected with the expansion-limit
diagnostic instead of exhausting memory. When a recursive macro's argument
grew at each expansion, the `source` provenance recorded on every expanded
node copied the previous expansions' records, about four times over per
expansion, so `chelis check` passed 2 GB within ten expansions and never
reached the limit. Expansion now finds the limit without building provenance.
A program that finishes expanding is expanded and recorded exactly as before
([#3853](https://github.com/Chelis-Lang/chelis/issues/3853)).
