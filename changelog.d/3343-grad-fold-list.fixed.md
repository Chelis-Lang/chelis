`grad` now differentiates `fold` and `scan` over a List-typed argument, such as
a list of layer weights, by reversing the executed recurrence. Previously the
lowering replaced the `fold` result with a rank-0 placeholder, so `grad` failed
with an out-of-range `sum` axis, a backward-DAG verification error, or a missing
`fold` input, and `scan` had no numeric lowering. A List whose length is runtime
data inside the differentiated body is now a named rejection. See
[#3343](https://github.com/Chelis-Lang/chelis/issues/3343).
