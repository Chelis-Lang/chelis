`vmap` lowers registry-admitted builtin values through existing numeric row-DAG
owners, including `neg`, `abs`, `add`, `key_from_seed`, and Float classifiers.
Static aliases and lexical captures preserve operation identity and authored
literal-extent callback contracts; Bool results and nonzero mapped axes retain
their types.
Mapped row entry guards compare physical input dimensions with authored formals,
including supplied builtin values and typed lambdas, before row arithmetic.
Explicit local function ascriptions retain their concrete callable representation
and lexical alias route for direct calls and collection callbacks.
Local callable literal-extent input and result obligations travel independently
of operation identity through aliases, captures and named type aliases. Mapped
input guards run before row arithmetic and later effects. Retaining an enclosing
dimension's witness in a callable contract remains separate pending work under
chelis#3517. Independent literal result obligations use the shared producer-claim
owner, retaining physical output axes and checking the result before later
effects even when the callable input is wildcard. Fused gradients retain their
own live-output completeness check for unresolved calls.
Static builtin values retain complete normalized invocation contracts through
global alias chains, Copy, captures and forwarded higher-order formals. A
result-only formal keeps its producer obligation even when it has no input
entry guard; lexical helpers consume the whole resolved value rather than
reconstructing a contract-free operation descriptor.
Complete callable frames actualize executable precision and rank metadata per
call site while preserving authored borrow, effect and source annotations.
Static transform callers use their supplied target's lane evidence rather than
inheriting an unresolved template's standalone exclusion.

Compiled named calls retain the actual definition's complete entry and result
contract when a tensor helper cannot lower the whole call. Pass-through bodies
preserve shared rank-row checks on unused arguments before later effects.

C builds keep closed List actuals' formed constructor geometry through the
complete named call and reject contradictory literal operand extents before
emitting artifacts or executing the program. Typed numeric leaves, discarded
traps and runtime List controls retain their existing execution owners; opaque
and effectful producers keep the complete host call. Eval's pre-execution
literal-refutation gap remains separate under chelis#3115.

Host-only builtin invocation retains nested entry and result contracts through
the shared callable owner. Host numeric result extents precede element traps
after operand agreement and operation controls; builtin selection preserves
the selected value's original producer.
Tuple-bound aliases retain whole callable values through nested construction,
copying and projection; typed builtin formals remain executable collection callbacks.
Independent result frames check in nested invocation return order on Eval and C.
Sequential host bindings avoid a lowering call frame per binding while retaining
their source scope and result boundaries.
