Chelis now provides `char_code(string) -> int64` and
`char_from_code(int64) -> string` for Unicode scalar conversion in both the
evaluator and compiled C lanes. Standard-library JSON serialization emits
RFC-valid escapes for every control character and parsing decodes complete
`\uXXXX` sequences, including valid surrogate pairs while rejecting malformed
or unpaired sequences. Compiled strings now preserve embedded NUL bytes and
other admitted Unicode escapes without ambiguous C literals or silent
truncation. See [#953](https://github.com/Chelis-Lang/chelis/issues/953) and
[#1855](https://github.com/Chelis-Lang/chelis/issues/1855).
