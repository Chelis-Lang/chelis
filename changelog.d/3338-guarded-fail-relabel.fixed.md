`grad` through a function that takes an ADT parameter and guards its body with
`fail` differentiates again. A caller-side dimension label is no longer stamped
onto the guarded abort alone, which had left it disagreeing with its fallback
and failed backward-DAG verification with "guarded_fail at node N must have
exactly its fallback's type". A guard that fires still aborts with its message.
See [#3338](https://github.com/Chelis-Lang/chelis/issues/3338).
