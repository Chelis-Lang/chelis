`f8e4m3` and `f8e5m2` now reject in scalar and nested type positions.
The checker reports invalid parameters, results, and body expressions separately.
Valid parts of a rejected signature still constrain its body.

Regenerate canonical Deep for inline-typed ordinary functions.

Parameter types now reside only in the generated signature.
Body annotations use inference holes instead of duplicate type syntax.
Standalone signatures and independently authored parameter annotations remain independent.

Rust callers constructing `TypeExpr::Tensor` now pass `TensorPrecision` rather than
`String`. Direct human-readable Serde for the Rust `chelis_surf::TypeExpr` AST now
uses `{name, span}` and accepts the prior string input with an unknown `0..0` token
span. Current binary Serde round-trips the structured form; no legacy binary format
is promised. The compiler API's separate structured `surf_ast` wire remains a
string-valued precision field.

Addresses [#1606](https://github.com/Chelis-Lang/chelis/issues/1606) and
[#1527](https://github.com/Chelis-Lang/chelis/issues/1527).
