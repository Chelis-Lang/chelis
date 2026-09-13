The host evaluator now keeps successful declaration initialization separate from
caller locals, reusing it within an evaluation context without caching failures
or function-call results. Ordinary named and named-axis calls retain declaration
scope; anonymous host closures retain lexical shadows. Caller argument order and
handled Random streams are preserved. Ordinary host requests start fresh caches;
invariant predicates keep separate contexts. This repairs the host declaration
slice of [#1956](https://github.com/Chelis-Lang/chelis/issues/1956), not general
named-gradient capture behavior.
