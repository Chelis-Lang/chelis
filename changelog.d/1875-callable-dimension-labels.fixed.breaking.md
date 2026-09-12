The checker retains authored dimension-binder identity through compatible named
axes and generic helper calls, including aliases and trusted serialized checker
contexts. Bodies that give such a polymorphic result a false concrete dimension
now reject instead of losing the binder constraint through the axis name. This
addresses the bounded checker hole in
[#1875](https://github.com/Chelis-Lang/chelis/issues/1875); it does not introduce
global equality between separately named values or change School's public APIs.

Rust embeddings must regenerate stored checker contexts: direct `TypeEnv` serde
now uses format 1 with a mandatory private dimension-label summary. Dependency,
stdlib and compiled-context payload versions are 15, 19 and 21 respectively.
Source-free snapshots remain a faithful transport from a trusted checker
producer; structural decoding does not prove an adversarial summary complete.

Compiled-context named-query checking also enters the lowerer. Its separate
bounded helper-result repair is tracked by
[#1889](https://github.com/Chelis-Lang/chelis/issues/1889); checker snapshot
transport alone does not establish native-C or worker/disk query compatibility.
This change does not claim all of #1875 or downstream School closure.
