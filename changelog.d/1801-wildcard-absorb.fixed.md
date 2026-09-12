A call whose result extent is decided only at run time no longer leaves the
callee's dimension variable unresolved. A dimension variable an application's
instantiation mints, that unifies with a runtime extent `*` and that no
argument of that application binds to a literal or named dimension, now
denotes that runtime extent. A nullary root reached through such a call used to
generalize the variable instead, which left the root with no ABI and dropped it
from both lanes in silence: `chelis eval --file` warned that the input held
only declarations, and `chelis build --target c` emitted no entry point. Both
lanes now execute the root, and a claim the call refutes traps as
`spec/04-type-system.md` §4.7 requires. See
[chelis#1801](https://github.com/Chelis-Lang/chelis/issues/1801).
