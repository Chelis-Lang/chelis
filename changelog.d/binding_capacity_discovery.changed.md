The Python capacity census now checks return types and reachable aliases and
containers for final registrations. Nine source, path, name, dtype-vocabulary,
and opaque-handle methods retire their legacy exceptions; eight numeric JSON
and tensor methods retain their existing migration gates. Source-only JSON
results retain their concrete Rust types until Python string conversion, with
unchanged Python result formats.
Callable discovery binds the actual PyO3 registration, Rust implementation,
source span, and live descriptor kind. Renamed functions, methods and properties
cannot inherit a same-named helper's authority; unsupported registration forms
fail closed. Source JSON authority cannot admit unrelated sibling text results.
