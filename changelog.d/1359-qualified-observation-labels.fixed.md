`chelis eval` preserves binding-qualified record and tuple descendant labels
such as `gadt.text`, matching compiled C observation output byte-for-byte.
Legal repeated underscores in roots and fields, such as `root__tuple` and
`record_root.field__name`, are also preserved across standalone and Reef
package output. Previously display dequalification could mistake authored
`__` for a linker separator and drop part of the source label. Batched test
diagnostics now also remove exact synthetic `__Eval` and
`__ChelisTestBatchN` module provenance without exposing those linker-private
prefixes or truncating names such as `bad__forge`. See
[#1359](https://github.com/Chelis-Lang/chelis/issues/1359) and
[#2193](https://github.com/Chelis-Lang/chelis/pull/2193).
