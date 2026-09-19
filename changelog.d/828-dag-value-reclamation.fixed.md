Root-scoped tensor DAG evaluation now frees each non-root intermediate once the
last step that reads it has run, so peak host memory tracks the live working
set instead of the executed node count. A deep chain that previously held one
tensor per executed node now holds two. Entry points that name no roots
(`eval_tensor`, `eval_tensor_with`, `eval_tensor_with_strict`, and the plan
entry points) still return every executed node's value. See
[#828](https://github.com/Chelis-Lang/chelis/issues/828).
