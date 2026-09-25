DAG evaluation rejects a tensor supplied to a declared scalar input. Evaluator
and direct DAG C entries validate inputs in ABI slot order and report one typed
numeric trap line for rank, dtype, and literal-extent failures. Checked value
declarations also carry their tensor rank into direct external loads before
this validation. See
[#2530](https://github.com/Chelis-Lang/chelis/issues/2530) and
[#2531](https://github.com/Chelis-Lang/chelis/issues/2531).
