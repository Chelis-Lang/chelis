The compiler API evaluator now routes its seven filesystem builtins and
`process_run` through one injected system boundary with independently checked
Filesystem and Process permissions. Normal evaluation permits both, while
invariant predicate revalidation refuses both before invoking the host
adapter. The default adapter preserves existing filesystem, process, and
error behavior, including byte-ordered directory
names and strict UTF-8 failure under [05-HOST-4]. This evaluator-only change
does not alter compiled binaries or the runtime ABI; compiled host
`process_run` support remains required by [05-HOST-2] and tracked under
chelis#1297. See
[#1323](https://github.com/Chelis-Lang/chelis/pull/1323).
