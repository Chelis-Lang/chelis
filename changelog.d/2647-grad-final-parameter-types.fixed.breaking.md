`grad` determines which parameters contribute gradients from their final
inferred types, including nested components and disconnected parameters.
A previously unannotated float parameter contributes its cotangent instead
of being silently omitted from the checked result type. Unresolved selected parameter
types at the declaration boundary are rejected. A result annotation alone does
not determine an unresolved parameter type, even when the differentiated
callable's type is determined later; neither does a typed sibling branch or
aggregate. Helpers retain this distinction through generalization, recursion,
record updates, and cached checking environments.
