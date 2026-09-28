A generated C public entry now compares each supplied tensor's dtype with the
declared parameter dtype before it reads the tensor. A mismatch prints a line
naming the input and both dtypes, then traps with
`numeric trap: domain in load at <declared dtype>`. Previously the entry
checked only rank and extents, so a wrong-dtype `chelis_tensor` was accepted
and its storage was read at the declared dtype. The check covers the
four-argument tensor entry and every authored host entry, whatever it returns,
including a tensor nested inside a parameter value. See
[#2490](https://github.com/Chelis-Lang/chelis/issues/2490).
