`chelis build` refuses a program whose compiled tensor function would read a
runtime extent before declaring it, with the same "rendered but never
declared" class of rejection under
[#1277](https://github.com/Chelis-Lang/chelis/issues/1277). This applies to
`--target c` and to `--target hip` builds whose host-side tensor helpers the C
emitter writes; a Metal build of such a program, already refused, now reports
this refusal first. Previously a `--target c` build succeeded and wrote C that
failed to compile with an undeclared identifier, as for a `.0` projection of a
data value built with `where`. Such programs are refused until they can be
compiled; destructuring the result instead of projecting it compiles. See
[#2883](https://github.com/Chelis-Lang/chelis/issues/2883).
