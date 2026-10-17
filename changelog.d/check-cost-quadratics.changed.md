`chelis check` no longer copies module-sized or scope-sized state once per
item. Signature-metadata inference runs one sweep over the function
declarations instead of one per declaration. Each linearity declaration reads
a shared frozen program scope instead of a private copy of it. Validation and
effect annotation keep one environment with nested scopes instead of copying
it at every `let`, function, and match arm. The style gate and the advisory
lint pass no longer spawn `cargo locate-project` unless a lint exception could
apply. Results and diagnostics are unchanged. On a file of 1,600 one-line
definitions, `check` drops from about 145 G to 17 G instructions. See
[#3717](https://github.com/Chelis-Lang/chelis/issues/3717),
[#3719](https://github.com/Chelis-Lang/chelis/issues/3719),
[#3740](https://github.com/Chelis-Lang/chelis/issues/3740) and
[#1329](https://github.com/Chelis-Lang/chelis/issues/1329).
