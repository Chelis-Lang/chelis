Generated C reshape now validates target metadata through the checked runtime
before allocation or copying. Valid empty shapes such as
`[INT64_MAX, INT64_MAX, 0]` no longer overflow an intermediate product, and invalid
counts, strides, or byte sizes fail before access. Generated tensor snapshots
consume checked strides and logical byte counts. The C runtime also exposes
metadata observation, reshape validation, and an owned reshape operation that
preserves exact stored bits for every supported dtype. Part of
[#889](https://github.com/Chelis-Lang/chelis/issues/889) and
[#893](https://github.com/Chelis-Lang/chelis/issues/893).
