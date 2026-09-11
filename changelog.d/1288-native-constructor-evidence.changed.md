Native binding verification records raw constructor function values, including
casts and const/static initializers. The owner verifier requires this evidence
and rejects guarded constructors passed as callbacks or constructed inside
closures; a private tagged enum alone does not establish validated ownership.
