Host lowering now derives each definition's kernel decision from a `HostLoweringSession` bound to the program
it describes, replacing thread-local caches keyed on the program's address and gated on a flag that an entry
point had to arm. Forgetting to establish one is now a compile error rather than a silent full recomputation.
The callee summary probe also moved to last among the host-lane predicates, so a definition whose declared
result cannot be a kernel pays none. On a thirteen-definition fan-out chain, per-definition summary builds go
from 78 to 12 where the chain returns tensors and from 398,574 to 0 where it returns scalars. End-to-end
interpreter speed is unchanged: the arming call that recovered it shipped separately, and this change replaces
that call with the structure rather than adding a further speedup.
