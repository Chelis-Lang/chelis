`clamp` with a NaN bound or a lower bound above the upper bound now reports
`numeric trap: domain in clamp at <dtype>`, the [04-NUM-9] form, in both
`chelis eval` and compiled programs, followed by the offending row-major
position on its own line. It used to print `Domain: clamp ... at row-major
position N`. See [#3010](https://github.com/Chelis-Lang/chelis/issues/3010).
