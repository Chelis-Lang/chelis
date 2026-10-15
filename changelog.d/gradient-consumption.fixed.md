Fix compiled field reads of lexical aggregate values whose names also identify
top-level values. Helpers consuming computed ADT gradients read the cotangent
fields instead of replaying the same-spelled primal initializer.
