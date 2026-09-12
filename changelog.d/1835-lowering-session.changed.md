Host lowering now derives each definition's kernel decision once per program instead of re-deriving it per
caller. The memo behind that decision was a set of thread-local caches keyed on the program's address and gated
on a flag an entry point had to arm, and the entry the interpreter uses never armed it, so evaluating a program
whose definitions call one another re-expanded the call graph per application. It is now a field of a
`HostLoweringSession` bound to the program it describes, which the type system requires every caller to
establish. On a thirteen-definition fan-out chain the work drops from 398,574 summary builds to 12.
