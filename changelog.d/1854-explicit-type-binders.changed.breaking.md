Function signatures now quantify type, dimension, and rank variables only
from explicit `[...]` binder lists. Add every formerly implicit name to the
owning `sig` or `def`; a standalone `sig` owns the declaration's sole list.
Canonical polymorphic Deep inserts a structural `(binders...)` child between
the `defsig` name and type, while monomorphic `defsig` remains two-child.
Unknown dtype-like names are rejected with the nearest active spelling.
