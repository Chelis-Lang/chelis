Concrete tensor-returning dropout helpers retain source-static rate arguments
through evaluator calls, including encoded Reef library contexts. Local helpers
also use the existing fixed-control C plan. Runtime-rate public C entries,
generic evaluator helpers, and compiled-in-context C plan transport remain
unsupported. Part of [#1764](https://github.com/Chelis-Lang/chelis/issues/1764).
