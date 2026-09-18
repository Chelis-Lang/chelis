IR lowering classifiers and scope walkers now read Deep expressions through
the shared total carrier view while preserving source-specific legacy,
`UnknownForm`, metadata, and malformed-list behavior. Malformed raw parameter
forms are rejected instead of disappearing during binder collection. This is
the classifier/scoping reader slice of
[#1125](https://github.com/Chelis-Lang/chelis/issues/1125); core lowering
dispatch and static-control readers remain follow-up work.
