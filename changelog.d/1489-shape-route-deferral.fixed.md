`gather`, `scatter`, `scatter_replace` and `trace` no longer reject an operand
whose type is merely unresolved at the moment the call is checked. Each now
suspends its whole decision on the operand's type variable and settles it when
that variable is bound. The axis is normalized against the operand's settled
rank, so a negative axis on a deferred operand means the same axis it means on
a resolved one, and `scatter`'s updates-shape equation and mode check are
replayed at the binding rather than dropped. An operand that is never resolved
is still rejected. `concat` and `diagonal` still reject an unresolved operand:
they compute a new extent from the operand's dims, and deferring them admitted
programs whose declared result shape was false. See
[#1489](https://github.com/Chelis-Lang/chelis/issues/1489).
