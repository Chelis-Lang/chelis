Function signatures now quantify type, dimension, and rank variables only
from explicit `[...]` binder lists. Add every formerly implicit name to the
owning `sig` or `def`; a standalone `sig` owns the declaration's sole list.
Canonical polymorphic Deep inserts a structural `(binders...)` child between
the `defsig` name and type, while monomorphic `defsig` remains two-child.
Binder lists reject active, reserved, retired, and deferred dtype spellings
even when unused or used as dimension/rank variables. Unknown dtype-like
names are rejected with the nearest active spelling and one diagnostic per
declaration/name.
