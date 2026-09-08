Producer AST extensions are opaque data: compiler passes preserve their contents
without expanding macros, renaming references, inferring effects, or interpreting
nested annotation-like keys. Compiler-owned annotations retain their dedicated
types and validation. Deep-to-Surf conversion rejects extensions it cannot preserve.
Property discovery requires canonical property annotations; `c_earchin_role` is
producer data. The Rust extension API and serialized extension format change;
regenerate older AST caches. Part of [#908](https://github.com/Chelis-Lang/chelis/issues/908).
