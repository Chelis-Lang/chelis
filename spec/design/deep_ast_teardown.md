# Deep AST ownership teardown

`chelis_deep::Expr` owns child expressions through stamped nodes, bare lists,
unknown forms, legacy metadata wrappers, and expression-valued annotations.
The depth of an admitted tree is not a safe bound on the native call stack used
to release it. Dropping an owned tree therefore drains these edges with an
explicit work queue before Rust releases each emptied carrier.

Annotation storage may be shared. A metadata handle that is not the final
owner releases only its reference; the final owner extracts the expressions
and nested annotation maps from storage before those payloads are dropped.
The storage destructor owns the final-release path, including a final release
that races another thread's reference release.
The extraction matches every typed annotation variant, so adding a new
expression-bearing variant requires an explicit teardown disposition.
Historical raw source arguments held by an annotation are drained through
their own explicit queue before annotation storage is released.

`Expr` implements `Drop`, which prevents callers from moving a variant's
fields out by pattern matching. The consuming `into_*` methods detach fields
through safe replacements and leave an empty carrier for ordinary release.
These methods preserve ownership on a mismatched variant by returning the
original expression.

The acceptance oracle is
`cargo nextest run -p chelis-deep --test issue_3394_owned_teardown`.
It releases long stamped list spines, mixed non-node carriers, partial
diagnostic trees, historical raw source arguments, and shared
expression-valued annotations on a 2 MiB worker.
The same test checks heap balance and concurrent release of shared annotations.
