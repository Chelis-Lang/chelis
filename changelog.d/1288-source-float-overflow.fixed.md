Reject Deep floating literals whose exponents overflow binary64 at lexical
ingress, before checking, evaluation, validation, or decompilation can consume
them. Finite source values, including signed zero, retain their exact lexical
bits. Runtime IEEE infinity and NaN transport remains supported.
