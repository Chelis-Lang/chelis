The runtime extent and non-negativity guards listed below now report their
failure in the form spec/04 section 4.7 fixes, with identical text in
`chelis eval` and compiled programs. The report is a context line naming the operation,
the axis or entry, and the value observed, then
`numeric trap: domain in <op> at i64`. This covers:
- negative `expand` and `insert` sizes, and negative `pad` and `shrink`
  bounds. The evaluator used to print a message about its own graph nodes
  ([#1802](https://github.com/Chelis-Lang/chelis/issues/1802)), and compiled
  programs named no axis or value;
- `concat` parts that disagree on an axis that is not concatenated. Compiled
  programs used to print the trap line before its context, and an empty List
  of parts printed a non-trap message;
- a `to_tensor` List whose children differ in shape. Compiled programs used to
  name an internal runtime function;
- negative `split` sizes, and sizes that do not sum to the axis extent;
- a negative `tensor_scan` length;
- `reduce_window_*` windows that are wider than their input axis, and other
  invalid windows and strides.

A `shrink` range that is inverted or overshoots, and a non-positive `stride`
step, still print a context line that names no axis or value
([#3193](https://github.com/Chelis-Lang/chelis/issues/3193)).
