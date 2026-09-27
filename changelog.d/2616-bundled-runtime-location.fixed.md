The bundled `chelis-std` runtime is now read from the compiler in memory and
carries no filesystem location. Previously every invocation whose package graph
contained the runtime extracted it into a new `chelis-std-bundle-*` directory
under the system temporary directory and never removed it, and that directory's
random path was recorded in the prepared-graph cache, the compiled-context
cache and the `chelis test` worker hand-off, so those bytes differed between
two runs over the same package. The first run after upgrading rebuilds those
caches once. See [#2616](https://github.com/Chelis-Lang/chelis/issues/2616).
