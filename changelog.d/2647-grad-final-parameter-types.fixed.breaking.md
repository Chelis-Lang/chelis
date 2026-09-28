`grad` determines which parameters contribute gradients from their final
inferred types, including nested components and disconnected parameters.
A previously unannotated float parameter contributes its cotangent instead
of being silently omitted from the checked result type. Unresolved selected parameter
types at the declaration boundary are rejected. A result annotation alone does
not determine an unresolved parameter type, even when the differentiated
callable's type is determined later; neither does a typed sibling branch or
aggregate. Helpers retain this distinction through generalization, recursion,
record updates, collection operations, and cached checking environments, while
retaining the ordinary type equalities required by their checked signatures.
Result-origin transport also preserves concrete key values and the declared
owner of generic key restrictions.
Published schemes keep dtype bounds on the correct variables; concrete
first-order value annotations remain available to subsequent shape checks.
Independent declared callable inputs and results remain available to ordinary
generic instantiation, branch unification, and transforms without using a
gradient result constraint to infer its selected parameters.
Tensor concatenation retains computed extents when a transported callable's
declared result has runtime dimensions.
