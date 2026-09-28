`grad` determines which parameters contribute gradients from their final
inferred types, including nested components and disconnected parameters.
A previously unannotated float parameter contributes its cotangent instead
of being silently omitted from the checked result type. Unresolved selected parameter
types at the declaration boundary are rejected.
