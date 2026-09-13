Named-axis host evaluation now prepares selected tensor inputs and required shape
witnesses on the lowered graph before execution, instead of pre-initializing
syntactically referenced library values. Statically dead captures and same-named
formals no longer trigger unrelated initializers; optional shape witnesses only
use available tensors. Entered initializer errors and existing input precedence
are preserved. This does not change general declaration caching or caller-frame
ownership.
