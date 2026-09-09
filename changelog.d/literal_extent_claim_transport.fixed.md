Declared tensor result extents survive function checking and inlining. The
literal claim in a shape-derived `insert` keeps its argument witness and
required Domain failure, including nested calls and discarded results.
WireDag v8 carries the witness, its invocation dependencies and provenance;
older wire and compiler-cache versions are rejected or missed. Part of
[#1277](https://github.com/Chelis-Lang/chelis/issues/1277); resolves the public
[#1377](https://github.com/Chelis-Lang/chelis/issues/1377) reproduction.
