Compiled C now releases the values the ownership schedule frees at the start
of a `match` arm or of a `map` or `fold` body, even when that arm or body
branches. A payload nothing reads, such as the one a `None` arm's test
extracts, and a scrutinee whose last use is its field extraction were never
released in a branching arm, so each call leaked them; a branching `map` or
`fold` body whose item nothing reads produced C that did not compile.

A `match` nested in another match's arm no longer reuses the enclosing
match's generated payload names. The nested name shadowed the enclosing
payload in C, so releasing that payload inside the nested arm either failed to
compile or released the nested value instead. See
[#2485](https://github.com/Chelis-Lang/chelis/issues/2485) and
[#2458](https://github.com/Chelis-Lang/chelis/issues/2458).
