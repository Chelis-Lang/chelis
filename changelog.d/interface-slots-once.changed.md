C emission no longer rebuilds the interface-input list for each extent check of a
function with runtime-extent tensor inputs. With 256 such inputs,
`chelis build --emit-c` took 2.6 CPU s instead of 14.2 CPU s, and the emitted C
did not change. See [#3726](https://github.com/Chelis-Lang/chelis/issues/3726).
