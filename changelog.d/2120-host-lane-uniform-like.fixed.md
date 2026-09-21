`chelis build` now compiles `uniform_like` when the template tensor's contents
are computed at run time, instead of refusing with `[04-TOT-2]`. Only the C
host lane lacked an emission arm; template foldability decided which lane the
draw routed to, so a parameterised helper met the refusal immediately while an
equivalent literal template compiled. The compiled draw matches `chelis eval`
bit for bit across every active float dtype, and the template's element values
are still never observed, per [05-OP-8]. See
[#2120](https://github.com/Chelis-Lang/chelis/issues/2120).
