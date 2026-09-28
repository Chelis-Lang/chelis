Preserve exact string selectors through differentiated helper calls, closures,
and structured values. String equality chooses the corresponding numeric
branch without storing strings in tensor IR; host components of structured
gradient arguments retain unit cotangents.
Top-level tuple and ADT selectors retain their declaring values. Unsupported
string computations in Grad report the operation's structural rejection.
Preserve host observations and their order around tensor result guards when
numeric helper probes encounter exact string or aggregate arguments.
