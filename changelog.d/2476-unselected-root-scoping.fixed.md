Evaluating selected roots runs only the declarations the selection enters, so
a seeded node of another declaration cannot pull that declaration's
parameters into the run as a `missing required input`, the shape of
[#991](https://github.com/Chelis-Lang/chelis/issues/991). Every graph node
records its declaration, and the evaluator and dead-code elimination seed an
abort, a potentially trapping node or a draw that can trap only when its
declaration is a selected root's. A function runs inlined where it is
applied, and a value declaration whose initializer can trap runs inlined
where it is referenced, so an uncalled function's nodes do not run and its
parameters are not inputs, and a function named as a value and not applied
(`g = f`) runs nothing and initializes no value declaration its body
names. A selected
entry's required inputs are the evaluator's own live set. Fixes
[#2476](https://github.com/Chelis-Lang/chelis/issues/2476).
